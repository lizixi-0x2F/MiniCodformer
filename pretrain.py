#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepSeek 500M模型预训练脚本
- 使用维基百科数据进行预训练
- 支持单GPU训练
- 针对大型模型架构优化
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
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)

# 导入自定义模型
from model import DistillationModel
from optimized_distill import CustomModelConfig, create_custom_model_config

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
# 减少transformers库的日志输出
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
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

class CustomPreTrainer(Trainer):
    """自定义预训练训练器 - 支持困惑度计算"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.perplexity_history = []
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 将标准语言模型损失与自定义模型输出格式相适应，并计算困惑度
        outputs = model(**inputs, calculate_metrics=True)
        
        # 检查outputs是字典还是带有loss属性的对象
        if isinstance(outputs, dict):
            loss = outputs.get("loss", None)
            if loss is None and "ltc_ncp_logits" in outputs:
                # 如果字典中没有loss但有logits，手动计算损失
                logits = outputs["ltc_ncp_logits"]
                labels = inputs.get("labels")
                if labels is not None:
                    # 确保标签在有效范围内
                    vocab_size = logits.size(-1)
                    
                    # 将所有标签设为有效值或忽略索引
                    # 任何小于0的值都设为忽略索引
                    # 任何大于等于vocab_size的值也设为忽略索引
                    valid_labels = torch.where(
                        (labels >= 0) & (labels < vocab_size),
                        labels,
                        torch.tensor(-100, device=labels.device, dtype=labels.dtype)
                    )
                    
                    loss = F.cross_entropy(
                        logits.view(-1, vocab_size),
                        valid_labels.view(-1),
                        ignore_index=-100
                    )
                else:
                    # 如果没有标签，默认设置一个零损失
                    loss = torch.tensor(0.0, device=model.device)
        else:
            # 标准Hugging Face模型输出
            loss = outputs.loss
            
        # 获取困惑度指标
        perplexity = None
        if isinstance(outputs, dict) and "metrics" in outputs:
            metrics = outputs["metrics"]
            if "perplexity" in metrics:
                perplexity = metrics["perplexity"]
                self.perplexity_history.append(perplexity)
        
        # 记录损失和困惑度
        if self.state.global_step % 100 == 0:
            if perplexity is not None:
                logger.info(f"步骤 {self.state.global_step}: 损失={loss.item():.4f}, 困惑度={perplexity:.2f}")
            else:
                logger.info(f"步骤 {self.state.global_step}: 损失={loss.item():.4f}")
            
        return (loss, outputs) if return_outputs else loss
        
    def log(self, logs):
        """添加困惑度到日志"""
        # 计算平均困惑度
        if hasattr(self, 'perplexity_history') and self.perplexity_history:
            # 取最近10个值计算平均困惑度
            recent_perplexity = self.perplexity_history[-min(10, len(self.perplexity_history)):]
            valid_values = [p for p in recent_perplexity if p != float('inf')]
            if valid_values:
                avg_perplexity = sum(valid_values) / len(valid_values)
                logs["perplexity"] = avg_perplexity
            
        super().log(logs)

def main():
    parser = argparse.ArgumentParser(description="DeepSeek 500M模型预训练")
    
    # 模型和训练参数
    parser.add_argument("--tokenizer_path", type=str, required=True, help="分词器路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--hidden_size", type=int, default=2048, help="隐藏层大小")
    parser.add_argument("--num_layers", type=int, default=24, help="Transformer层数")
    parser.add_argument("--num_heads", type=int, default=16, help="注意力头数")
    parser.add_argument("--intermediate_size", type=int, default=8192, help="前馈网络大小")
    
    # 数据参数
    parser.add_argument("--train_file", type=str, required=True, help="训练数据文件")
    parser.add_argument("--eval_file", type=str, help="评估数据文件，不填则从训练集划分")
    parser.add_argument("--is_json", action="store_true", help="数据是否为JSON格式")
    parser.add_argument("--max_seq_length", type=int, default=512, help="最大序列长度")
    
    # 训练超参数
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="每个设备的训练批次大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="每个设备的评估批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")
    
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
    logger.info(f"加载tokenizer: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    # 加载数据集
    logger.info("准备训练数据集")
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=args.train_file,
        max_length=args.max_seq_length,
        is_json=args.is_json
    )
    
    # 准备评估数据集
    eval_dataset = None
    if args.eval_file:
        logger.info("准备评估数据集")
        eval_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=args.eval_file,
            max_length=args.max_seq_length,
            is_json=args.is_json
        )
    
    # 创建模型配置
    logger.info("创建500M模型架构")
    config = create_custom_model_config(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers, 
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size
    )
    
    # 创建模型
    logger.info("初始化模型")
    model = DistillationModel(config)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {total_params:,}")
    
    # 设置训练参数
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
        training_args_dict["evaluation_strategy"] = "steps"
        training_args_dict["load_best_model_at_end"] = True
    
    training_args = TrainingArguments(**training_args_dict)
    
    # 创建数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 创建自定义训练器
    trainer = CustomPreTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    logger.info("开始预训练...")
    trainer.train()
    
    # 保存最终模型
    logger.info(f"保存最终模型到 {args.output_dir}")
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(args.output_dir)
    else:
        # 手动保存权重
        torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
        # 保存配置
        if hasattr(model, "config") and hasattr(model.config, "save_pretrained"):
            model.config.save_pretrained(args.output_dir)
        elif hasattr(config, "save_pretrained"):
            config.save_pretrained(args.output_dir)
        else:
            # 保存为JSON
            with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)
    
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("预训练完成！")

if __name__ == "__main__":
    main() 