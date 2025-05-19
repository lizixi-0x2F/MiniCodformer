#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
内存优化版知识蒸馏脚本
- 解决词汇表不匹配问题
- 优化内存使用
- 避免CUDA OOM错误
- 支持单GPU训练
"""

import os
import json
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
    AutoModel,
    PretrainedConfig
)

# 导入自定义模型
from model import (
    DistillationModel, 
    JointKDLoss, 
    TeacherOutputAdapter,
    MiniTransformerEncoder,
    LTCNCPDecoder
)

# 设置日志 - 减少不必要的输出
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
# 减少transformers库的日志输出
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """文本数据集类"""
    
    def __init__(self, tokenizer, file_path, max_length=512, is_json=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        logger.info(f"加载数据文件: {file_path}")
        
        if is_json:
            # JSON行格式
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f.readlines(), desc="加载JSON数据"):
                    try:
                        item = json.loads(line)
                        if "text" in item:
                            self.examples.append(item["text"])
                    except json.JSONDecodeError:
                        logger.warning(f"无法解析JSON行: {line[:50]}...")
        else:
            # 纯文本格式
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                texts = text.split("\n\n")
                self.examples = [t for t in texts if len(t.strip()) > 0]
        
        logger.info(f"数据集大小: {len(self.examples)} 个样本")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings.input_ids[0],
            "attention_mask": encodings.attention_mask[0],
            "labels": encodings.input_ids[0].clone()
        }

class MemoryEfficientTeacherAdapter:
    """内存高效的教师模型输出适配器 - 强制确保尺寸匹配
    - 不使用巨大的投影矩阵
    - 使用分块处理避免OOM
    - 仅在CPU上处理过大的数据
    - 严格检查确保词汇表尺寸匹配
    """
    
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self._has_logged = False  # 控制日志只打印一次
    
    def __call__(self, teacher_logits, target_batch_size, target_seq_length):
        if teacher_logits is None:
            # 如果没有教师输出，返回零张量
            return torch.zeros(
                (target_batch_size, target_seq_length, self.vocab_size),
                dtype=torch.float16,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        
        batch_size, seq_length, vocab_size = teacher_logits.shape
        device = teacher_logits.device
        dtype = teacher_logits.dtype
        
        # 总是处理词汇表大小调整，确保维度匹配
        if vocab_size != self.vocab_size:
            if not self._has_logged:
                logger.info(f"词汇表大小调整: {vocab_size} → {self.vocab_size} (仅显示一次)")
                self._has_logged = True
                
            # 创建新的张量来适配目标词汇表大小
            resized_logits = torch.zeros(
                (batch_size, seq_length, self.vocab_size),
                dtype=dtype,
                device=device
            )
            
            # 复制共同部分
            common_size = min(vocab_size, self.vocab_size)
            resized_logits[:, :, :common_size] = teacher_logits[:, :, :common_size]
            
            # 如果教师词汇表更大，将多余部分的信息进行汇总到最后几个token
            if vocab_size > self.vocab_size:
                # 分块处理以节省内存
                chunk_size = 1000  # 每次处理1000个词汇
                num_chunks = (vocab_size - common_size + chunk_size - 1) // chunk_size
                
                # 如果块数量太多，移到CPU处理
                if num_chunks > 10:
                    # 仅复制需要处理的部分到CPU，节省显存
                    teacher_logits_extra = teacher_logits[:, :, common_size:].cpu()
                    
                    # 计算平均值并加到最后一个token
                    extra_mean = teacher_logits_extra.mean(dim=2, keepdim=True)
                    resized_logits[:, :, -1:] += extra_mean.to(device)
                else:
                    # GPU上分块处理
                    for i in range(common_size, vocab_size, chunk_size):
                        end = min(i + chunk_size, vocab_size)
                        chunk = teacher_logits[:, :, i:end]
                        chunk_mean = chunk.mean(dim=2, keepdim=True)
                        resized_logits[:, :, -1:] += chunk_mean / num_chunks
            
            teacher_logits = resized_logits
        
        # 处理批次大小和序列长度不匹配
        if batch_size != target_batch_size or seq_length != target_seq_length:
            # 创建适应目标尺寸的新张量
            adjusted_logits = torch.zeros(
                (target_batch_size, target_seq_length, vocab_size),
                dtype=dtype,
                device=device
            )
            
            # 复制共同部分
            copy_batch = min(batch_size, target_batch_size)
            copy_length = min(seq_length, target_seq_length)
            adjusted_logits[:copy_batch, :copy_length, :] = teacher_logits[:copy_batch, :copy_length, :]
            
            # 如果需要扩展批次
            if target_batch_size > batch_size:
                # 复制最后一个批次样本
                last_batch = teacher_logits[-1:] 
                for i in range(batch_size, target_batch_size):
                    adjusted_logits[i:i+1, :copy_length, :] = last_batch[:, :copy_length, :]
            
            # 如果需要扩展序列长度
            if target_seq_length > seq_length:
                # 复制最后一个位置
                last_tokens = adjusted_logits[:, -1:, :]
                for i in range(seq_length, target_seq_length):
                    adjusted_logits[:, i:i+1, :] = last_tokens
            
            return adjusted_logits
        
        return teacher_logits

class CustomDistillationTrainer(Trainer):
    """自定义知识蒸馏训练器，基于model.py中的JointKDLoss"""
    
    def __init__(self, teacher_model=None, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model.eval()
            # 将教师模型移动到与学生模型相同的设备
            if torch.cuda.is_available():
                self.teacher_model.cuda()
        
        self.alpha = alpha  # 蒸馏损失权重
        self.temperature = temperature  # 温度参数
        # 使用模型中的联合KD损失
        self.joint_kd_loss = JointKDLoss(
            temperature=temperature,
            alpha=alpha,
            beta=1.0 - alpha,
            gamma=0.0  # 不使用特征蒸馏
        )
        
        # 创建内存高效的教师输出适配器
        if hasattr(self.model, "config") and hasattr(self.model.config, "vocab_size"):
            vocab_size = self.model.config.vocab_size
        else:
            vocab_size = 151669  # 使用默认值
            
        # 使用内存高效的适配器
        self.output_adapter = MemoryEfficientTeacherAdapter(vocab_size)
        
        logger.info(f"初始化自定义蒸馏训练器: alpha={alpha}, temperature={temperature}")
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 标准语言模型损失
        outputs = model(**inputs)
        
        # 检查outputs是字典还是带有loss属性的对象
        if isinstance(outputs, dict):
            student_loss = outputs.get("loss", None)
            if student_loss is None and "ltc_ncp_logits" in outputs:
                # 如果字典中没有loss但有logits，手动计算损失
                student_logits = outputs["ltc_ncp_logits"]
                labels = inputs.get("labels")
                if labels is not None:
                    student_loss = F.cross_entropy(
                        student_logits.view(-1, student_logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100
                    )
                else:
                    # 如果没有标签，默认设置一个零损失
                    student_loss = torch.tensor(0.0, device=model.device)
        else:
            # 标准Hugging Face模型输出
            student_loss = outputs.loss
        
        # 如果没有教师模型，则返回学生损失
        if self.teacher_model is None:
            return (student_loss, outputs) if return_outputs else student_loss
        
        # 计算蒸馏损失
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            # 确保教师输出是标准形式
            if isinstance(teacher_outputs, dict):
                teacher_logits = teacher_outputs.get("logits", None)
                if teacher_logits is None and "ltc_ncp_logits" in teacher_outputs:
                    teacher_logits = teacher_outputs["ltc_ncp_logits"]
            else:
                teacher_logits = teacher_outputs.logits
        
        # 获取学生logits
        if isinstance(outputs, dict) and "ltc_ncp_logits" in outputs:
            student_logits = outputs["ltc_ncp_logits"]
        else:
            student_logits = outputs.logits
        
        # 确保尺寸匹配 - 使用内存高效的适配器
        batch_size, seq_length = inputs["input_ids"].shape
        
        # 使用内存高效的适配器
        teacher_logits = self.output_adapter(
            teacher_logits,
            target_batch_size=batch_size,
            target_seq_length=seq_length
        )
        
        # 确保词汇表大小完全匹配 - 直接截断或填充
        if teacher_logits.size(-1) != student_logits.size(-1):
            # 获取目标尺寸
            target_vocab_size = student_logits.size(-1)
            
            # 创建新的尺寸匹配的张量
            matched_teacher_logits = torch.zeros(
                teacher_logits.shape[0],
                teacher_logits.shape[1],
                target_vocab_size,
                dtype=teacher_logits.dtype,
                device=teacher_logits.device
            )
            
            # 复制共同部分
            common_size = min(teacher_logits.size(-1), target_vocab_size)
            matched_teacher_logits[:, :, :common_size] = teacher_logits[:, :, :common_size]
            
            # 使用尺寸匹配的版本
            teacher_logits = matched_teacher_logits
        
        # 使用JointKDLoss - 确保尺寸已完全匹配
        kd_total_loss, kd_losses = self.joint_kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=inputs["labels"],
            ignore_index=-100
        )
        
        # 打印损失信息 - 减少输出频率
        if self.state.global_step % 500 == 0:
            logger.info(f"步骤 {self.state.global_step}: "
                      f"总损失={kd_total_loss.item():.4f}, "
                      f"学生损失={student_loss.item():.4f}, "
                      f"KD损失={kd_losses.get('soft_loss', 0):.4f}")
        
        return (kd_total_loss, outputs) if return_outputs else kd_total_loss

# 自定义配置类
class CustomModelConfig(PretrainedConfig):
    """继承自PretrainedConfig的自定义模型配置类"""
    model_type = "distillation_model"
    
    def __init__(
        self,
        vocab_size=151669,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=2048,
        ltc_ncp_hidden_size=2048,
        ltc_ncp_num_layers=12,
        ltc_kernel_size=3,
        distill_supervision=True,
        share_embeddings=True,
        distill_logits=True,
        distill_hiddens=True,
        temperature=2.0,
        distill_top_k=20,
        ignore_padding=True,
        pad_token_id=0,
        eos_token_id=2,
        bos_token_id=1,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.ltc_ncp_hidden_size = ltc_ncp_hidden_size
        self.ltc_ncp_num_layers = ltc_ncp_num_layers
        self.ltc_kernel_size = ltc_kernel_size
        self.distill_supervision = distill_supervision
        self.share_embeddings = share_embeddings
        self.distill_logits = distill_logits
        self.distill_hiddens = distill_hiddens
        self.temperature = temperature
        self.distill_top_k = distill_top_k
        self.ignore_padding = ignore_padding
        
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            **kwargs
        )

def create_custom_model_config(vocab_size, hidden_size=768, num_hidden_layers=12, 
                              num_attention_heads=12, intermediate_size=3072):
    """创建自定义模型配置"""
    return CustomModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size, 
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=2048,
        ltc_ncp_hidden_size=hidden_size,
        ltc_ncp_num_layers=num_hidden_layers//2,  # 解码器层数为编码器的一半
        ltc_kernel_size=3,
        model_type="distillation_model",
        distill_supervision=True,
        share_embeddings=True,
        distill_logits=True,
        distill_hiddens=True,
        temperature=2.0,
        distill_top_k=20,
        ignore_padding=True,
        pad_token_id=0,
        eos_token_id=2
    )

def main():
    parser = argparse.ArgumentParser(description="使用自定义模型架构进行知识蒸馏")
    
    # 模型和训练参数
    parser.add_argument("--teacher_model", type=str, required=True, help="教师模型路径或名称")
    parser.add_argument("--student_model", type=str, help="学生模型路径，不填则创建自定义小模型")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--hidden_size", type=int, default=256, help="隐藏层大小")
    parser.add_argument("--num_layers", type=int, default=3, help="层数")
    parser.add_argument("--num_heads", type=int, default=4, help="注意力头数")
    
    # 数据参数
    parser.add_argument("--train_file", type=str, required=True, help="训练数据文件")
    parser.add_argument("--eval_file", type=str, help="评估数据文件，不填则从训练集划分")
    parser.add_argument("--is_json", action="store_true", help="数据是否为JSON格式")
    parser.add_argument("--max_seq_length", type=int, default=512, help="最大序列长度")
    
    # 训练超参数
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="每个设备的训练批次大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="每个设备的评估批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")
    parser.add_argument("--alpha", type=float, default=0.5, help="蒸馏损失权重")
    parser.add_argument("--temperature", type=float, default=2.0, help="蒸馏温度")
    
    # 其他参数
    parser.add_argument("--fp16", action="store_true", help="是否使用fp16精度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--save_steps", type=int, default=1000, help="保存检查点的步数")
    parser.add_argument("--eval_steps", type=int, default=500, help="评估的步数")
    parser.add_argument("--logging_steps", type=int, default=100, help="日志记录的步数")
    parser.add_argument("--save_total_limit", type=int, default=3, help="保存的检查点总数限制")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置PyTorch内存分配
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载Tokenizer
    logger.info(f"加载教师模型tokenizer: {args.teacher_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    
    # 加载数据集
    logger.info("准备数据集")
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=args.train_file,
        max_length=args.max_seq_length,
        is_json=args.is_json
    )
    
    # 准备评估数据集
    eval_dataset = None
    if args.eval_file:
        eval_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=args.eval_file,
            max_length=args.max_seq_length,
            is_json=args.is_json
        )
    
    # 加载教师模型
    logger.info(f"加载教师模型: {args.teacher_model}")
    try:
        # 为了节省内存，设置半精度加载
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        # 设置为评估模式
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
    except Exception as e:
        logger.warning(f"使用AutoModelForCausalLM加载教师模型失败: {e}")
        # 尝试使用基础预训练模型加载
        try:
            logger.info("尝试使用AutoModel加载教师模型...")
            teacher_model = AutoModel.from_pretrained(
                args.teacher_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
        except Exception as e2:
            logger.warning(f"使用AutoModel加载教师模型也失败: {e2}")
            # 如果模型有自定义结构，尝试使用CustomModel加载
            logger.info("尝试使用自定义模型加载教师模型...")
            custom_config = create_custom_model_config(len(tokenizer))
            teacher_model = DistillationModel(custom_config)
            try:
                # 尝试直接加载状态字典
                missing_keys, unexpected_keys = teacher_model.load_state_dict(
                    torch.load(os.path.join(args.teacher_model, "pytorch_model.bin")), 
                    strict=False
                )
                logger.info(f"使用自定义模型加载教师模型成功，未加载的键: {len(missing_keys)}, 未使用的键: {len(unexpected_keys)}")
            except Exception as e3:
                logger.warning(f"所有加载方法都失败: {e3}")
                raise RuntimeError("无法加载教师模型，请检查模型路径和格式")
            
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
    
    # 加载或创建学生模型
    if args.student_model:
        try:
            logger.info(f"加载学生模型: {args.student_model}")
            # 首先尝试直接使用自定义模型加载
            custom_config = create_custom_model_config(len(tokenizer))
            student_model = DistillationModel(custom_config)
            
            # 尝试使用模型的保存路径和load_state_dict方式加载
            try:
                model_path = os.path.join(args.student_model, "pytorch_model.bin")
                missing_keys, unexpected_keys = student_model.load_state_dict(
                    torch.load(model_path, map_location="cpu"), 
                    strict=False
                )
                logger.info(f"使用state_dict方式加载学生模型: {model_path}")
                logger.info(f"未加载的键: {len(missing_keys)}, 未使用的键: {len(unexpected_keys)}")
            except Exception as e:
                logger.warning(f"使用state_dict加载失败: {e}")
                # 尝试使用from_pretrained方法
                try:
                    if hasattr(DistillationModel, "from_pretrained"):
                        student_model = DistillationModel.from_pretrained(args.student_model)
                        logger.info("使用from_pretrained方法加载学生模型成功")
                    else:
                        raise AttributeError("DistillationModel没有from_pretrained方法")
                except Exception as e2:
                    logger.warning(f"使用from_pretrained加载失败: {e2}")
                    # 最后尝试Hugging Face方式
                    logger.info("使用AutoModelForCausalLM最后尝试")
                    student_model = AutoModelForCausalLM.from_pretrained(args.student_model)
        except Exception as e:
            logger.warning(f"所有加载学生模型的方法都失败: {e}")
            # 创建新的自定义小型学生模型
            logger.info("创建新的自定义学生模型...")
            custom_config = create_custom_model_config(
                vocab_size=len(tokenizer),
                hidden_size=args.hidden_size,
                num_hidden_layers=args.num_layers,
                num_attention_heads=args.num_heads
            )
            student_model = DistillationModel(custom_config)
    else:
        # 创建自定义小型学生模型
        logger.info("创建自定义学生模型...")
        custom_config = create_custom_model_config(
            vocab_size=len(tokenizer),
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads
        )
        student_model = DistillationModel(custom_config)
    
    # 计算并打印模型参数量
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    logger.info(f"教师模型参数量: {teacher_params:,}")
    logger.info(f"学生模型参数量: {student_params:,}")
    
    # 设置训练参数 - 修复错误：使用兼容方式设置评估策略
    training_args_dict = {
        "output_dir": args.output_dir,
        "overwrite_output_dir": True,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio,
        "fp16": args.fp16,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": "tensorboard",
        "seed": args.seed,
        "dataloader_num_workers": 4,  # 使用多进程加载数据
        "disable_tqdm": False
    }
    
    # 根据是否有验证集来设置评估参数
    if eval_dataset:
        training_args_dict["eval_steps"] = args.eval_steps
        # 尝试使用字符串 'steps' 而不是 enum
        try:
            # 优先尝试使用字符串方式
            training_args_dict["evaluation_strategy"] = "steps"
            training_args_dict["load_best_model_at_end"] = True
            training_args = TrainingArguments(**training_args_dict)
        except TypeError:
            # 回退到枚举类型方式
            logger.info("使用IntervalStrategy.STEPS方式设置评估策略...")
            from transformers.trainer_utils import IntervalStrategy
            training_args_dict.pop("evaluation_strategy", None)  # 移除字符串方式
            training_args = TrainingArguments(**training_args_dict)
            training_args.evaluation_strategy = IntervalStrategy.STEPS
            training_args.load_best_model_at_end = True
    else:
        # 不设置评估策略
        training_args = TrainingArguments(**training_args_dict)
    
    # 创建数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 创建自定义蒸馏训练器
    trainer = CustomDistillationTrainer(
        teacher_model=teacher_model,
        alpha=args.alpha,
        temperature=args.temperature,
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    logger.info("开始蒸馏训练...")
    trainer.train()
    
    # 保存最终模型
    logger.info(f"保存最终学生模型到 {args.output_dir}")
    if hasattr(student_model, "save_pretrained"):
        student_model.save_pretrained(args.output_dir)
    else:
        # 手动保存权重
        torch.save(student_model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
        # 保存配置
        if hasattr(student_model, "config"):
            student_model.config.save_pretrained(args.output_dir)
    
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("蒸馏训练完成！")

if __name__ == "__main__":
    main()