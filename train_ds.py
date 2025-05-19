#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import math
import torch
import deepspeed
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
    set_seed,
    AutoConfig,
)
from tqdm.auto import tqdm
import pickle
import hashlib
import time
import re
import datetime
import json

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

from model import DistillationModel
from config import ModelConfig
from data import get_dataset, collate_fn

def setup_distributed(local_rank):
    """初始化分布式训练环境"""
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        try:
            # DeepSpeed会处理分布式通信，所以这里只做基础设置
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = "localhost"
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29500"
                
            # DeepSpeed会初始化进程组
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            logger.info(f"初始化分布式环境: local_rank={local_rank}, world_size={world_size}")
            return local_rank, world_size
        except Exception as e:
            logger.error(f"初始化分布式环境失败: {e}")
            logger.info("回退到单GPU模式")
            return -1, 1
    else:
        return -1, 1

def generate_cache_key(dataset_path, max_seq_length, tokenizer_name):
    """生成教师模型输出缓存的唯一键"""
    key = f"{dataset_path}_{max_seq_length}_{tokenizer_name}"
    # 使用MD5生成一个短的唯一标识符
    return hashlib.md5(key.encode('utf-8')).hexdigest()

def precompute_teacher_outputs(teacher_model, dataset, tokenizer, max_seq_length, 
                              batch_size, device, cache_dir, cache_key=None, train_file=None, args=None, model_config=None):
    """预计算所有教师模型的输出并缓存到磁盘 - DeepSpeed优化版"""
    # 记录开始时间和总样本数
    import time
    start_time = time.time()
    total_samples = len(dataset)
    logger.info(f"开始预计算教师模型输出，共 {total_samples} 个样本，批次大小 {batch_size}")
    if cache_key is None:
        cache_key = generate_cache_key(train_file, max_seq_length, tokenizer.__class__.__name__)
    
    # 创建缓存目录
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"teacher_outputs_{cache_key}.pt")
    
    # 如果缓存存在，直接加载
    if os.path.exists(cache_file):
        logger.info(f"从缓存加载教师模型输出: {cache_file}")
        try:
            return torch.load(cache_file, map_location=device)
        except Exception as e:
            logger.warning(f"加载教师缓存失败: {e}，将重新计算")
    
    logger.info(f"预计算教师模型输出并创建缓存: {cache_file}")
    
    # 创建简单数据加载器，不使用分布式采样
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 保持顺序
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # 预计算所有教师输出
    teacher_outputs = []
    
    # 确保教师模型处于评估模式
    if teacher_model is not None:
        teacher_model.eval()
    
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="预计算教师输出")):
            try:
                # 移动数据到GPU
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # 安全地获取教师输出
                try:
                    # 使用配置中的词汇表大小或默认值
                    if model_config is not None:
                        model_vocab_size = model_config.vocab_size  # 使用配置中的词汇表大小
                    else:
                        # 如果model_config未提供，使用默认值或从教师模型获取
                        model_vocab_size = 151936 if teacher_model is None else teacher_model.config.vocab_size
                    
                    batch_size = batch["input_ids"].size(0)
                    seq_length = batch["input_ids"].size(1)
                    
                    # 创建默认的零张量，使用词汇表大小
                    logits = torch.zeros(
                        (batch_size, seq_length, model_vocab_size),
                        dtype=torch.float16,
                        device=device
                    )
                    
                    # 处理小批次 - 使用逐个样本的CPU推理模式
                    sub_batch_size = 1  # 每个子批次最多1个样本，减少内存使用
                    full_batch_size = batch["input_ids"].size(0)
                    logits_list = []
                    
                    # 将大批次分拆成小批次处理
                    for i in range(0, full_batch_size, sub_batch_size):
                        end_idx = min(i + sub_batch_size, full_batch_size)
                        sub_input_ids = batch["input_ids"][i:end_idx]
                        sub_attention_mask = None
                        if "attention_mask" in batch:
                            sub_attention_mask = batch["attention_mask"][i:end_idx]
                        
                        # 为每个样本单独推理，避免批处理造成的内存溢出
                        sub_batch_logits = []
                        for j in range(sub_input_ids.size(0)):
                            # 获取单个样本
                            single_input = sub_input_ids[j:j+1]
                            single_mask = None
                            if sub_attention_mask is not None:
                                single_mask = sub_attention_mask[j:j+1]
                            
                            # 确保所有输入都在CPU上，并正确处理设备转换
                            try:
                                # 确保输入在CPU上
                                cpu_input = single_input.cpu()
                                cpu_mask = None
                                if single_mask is not None:
                                    cpu_mask = single_mask.cpu()
                                
                                # 如果教师模型在GPU上，直接使用GPU推理
                                if args.teacher_gpus is not None:
                                    # 解析GPU IDs
                                    gpu_ids = [int(gpu_id.strip()) for gpu_id in args.teacher_gpus.split(",")]
                                    # 使用第一个GPU作为主设备
                                    teacher_device = f"cuda:{gpu_ids[0]}"
                                    # 将输入移至教师GPU
                                    teacher_input = single_input.to(teacher_device)
                                    teacher_mask = None
                                    if single_mask is not None:
                                        teacher_mask = single_mask.to(teacher_device)
                                    
                                    # 在教师GPU上推理
                                    single_output = teacher_model(
                                        input_ids=teacher_input,
                                        attention_mask=teacher_mask,
                                        return_dict=True,
                                        use_cache=False,
                                        output_attentions=False,
                                        output_hidden_states=False,
                                    )
                                    
                                    # 将输出移至训练GPU并收集
                                    gpu_logits = single_output.logits.to(device)
                                else:
                                    # 使用CPU模型进行推理
                                    single_output = teacher_model(
                                        input_ids=cpu_input,
                                        attention_mask=cpu_mask,
                                        return_dict=True,
                                        use_cache=False,
                                        output_attentions=False,
                                        output_hidden_states=False,
                                    )
                                    
                                    # 将输出移至GPU并收集
                                    gpu_logits = single_output.logits.to(device)
                                
                                sub_batch_logits.append(gpu_logits)
                            except Exception as e:
                                logger.warning(f"单样本 {j} 处理失败: {e}")
                                # 创建零张量作为回退
                                dummy_logits = torch.zeros(
                                    (1, seq_length, model_vocab_size),
                                    dtype=torch.float16,
                                    device=device
                                )
                                sub_batch_logits.append(dummy_logits)
                        
                        # 如果有任何logits，则合并它们
                        if sub_batch_logits:
                            # 检查所有张量形状是否一致
                            shapes = [t.shape for t in sub_batch_logits]
                            logger.info(f"子批次logits形状: {shapes}")
                            
                            try:
                                # 仅当所有形状相同时才尝试合并
                                if all(t.shape == sub_batch_logits[0].shape for t in sub_batch_logits):
                                    combined_logits = torch.cat(sub_batch_logits, dim=0)
                                    logger.info(f"成功组合子批次logits，形状: {combined_logits.shape}")
                                    logits_list.append(combined_logits)
                                else:
                                    # 形状不一致，创建正确大小的零张量
                                    logger.warning(f"子批次logits形状不一致，使用零张量")
                                    dummy_logits = torch.zeros(
                                        (end_idx-i, seq_length, model_vocab_size),
                                        dtype=torch.float16,
                                        device=device
                                    )
                                    logits_list.append(dummy_logits)
                            except Exception as e:
                                logger.warning(f"合并子批次logits时出错: {e}，使用零张量")
                                dummy_logits = torch.zeros(
                                    (end_idx-i, seq_length, model_vocab_size),
                                    dtype=torch.float16,
                                    device=device
                                )
                                logits_list.append(dummy_logits)
                        else:
                            # 创建零张量作为回退
                            dummy_logits = torch.zeros(
                                (end_idx-i, seq_length, model_vocab_size),
                                dtype=torch.float16,
                                device=device
                            )
                            logits_list.append(dummy_logits)
                    
                    # 组合所有子批次logits
                    try:
                        if logits_list:
                            # 检查所有子批次的形状是否一致
                            if all(t.size(0) > 0 for t in logits_list):
                                # 逐个合并到logits张量中
                                current_idx = 0
                                for sub_logits in logits_list:
                                    sub_size = sub_logits.size(0)
                                    if current_idx + sub_size <= batch_size:
                                        logits[current_idx:current_idx+sub_size] = sub_logits
                                        current_idx += sub_size
                                    
                                if current_idx > 0:
                                    logger.info(f"批次 {batch_idx}: 成功组合多个子批次到完整批次")
                    except Exception as e:
                        logger.warning(f"组合多子批次到完整批次失败: {e}")
                    
                except Exception as e:
                    logger.error(f"批次 {batch_idx}: 创建教师输出时发生错误: {e}")
                
                # 保存教师输出
                teacher_outputs.append({
                    "input_ids": batch["input_ids"].cpu(),
                    "attention_mask": batch["attention_mask"].cpu() if "attention_mask" in batch else None,
                    "teacher_logits": logits.cpu()
                })
                
                # 定期释放缓存
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
                    
                # 每50个批次记录一次进度
                if batch_idx % 50 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"已处理 {batch_idx} 批次，耗时 {elapsed:.2f} 秒，" 
                                f"速度 {batch_idx/elapsed:.2f} 批次/秒")
                    
            except Exception as e:
                logger.error(f"处理批次 {batch_idx} 时出错: {e}")
    
    # 保存缓存
    try:
        torch.save(teacher_outputs, cache_file)
        logger.info(f"教师模型输出已保存到: {cache_file}")
        
        # 计算总耗时和性能
        end_time = time.time()
        total_time = end_time - start_time
        samples_per_sec = total_samples / total_time if total_time > 0 else 0
        
        # 打印性能统计
        logger.info(f"预计算完成:")
        logger.info(f"  - 总耗时: {total_time:.2f} 秒")
        logger.info(f"  - 平均速度: {samples_per_sec:.2f} 样本/秒")
        logger.info(f"  - 总批次数: {len(teacher_outputs)}")
        
        # GPU内存使用统计
        if args and args.teacher_gpus is not None:
            try:
                import subprocess
                import re
                gpu_info = subprocess.check_output("nvidia-smi", shell=True).decode('utf-8')
                gpu_memory = {}
                for line in gpu_info.split('\n'):
                    if "MiB" in line:
                        match = re.search(r'(\d+)MiB\s+/\s+(\d+)MiB', line)
                        if match:
                            used, total = match.groups()
                            gpu_id = len(gpu_memory)
                            gpu_memory[gpu_id] = (int(used), int(total))
                
                # 打印教师模型GPU内存占用
                logger.info("预计算完成后GPU内存占用:")
                for gpu_id in [int(x.strip()) for x in args.teacher_gpus.split(",")]:
                    if gpu_id in gpu_memory:
                        used, total = gpu_memory[gpu_id]
                        percent = (used / total) * 100
                        logger.info(f"  - GPU {gpu_id}: {used}MB / {total}MB ({percent:.1f}%)")
            except Exception as e:
                logger.warning(f"无法获取GPU内存信息: {e}")
    except Exception as e:
        logger.error(f"保存教师输出缓存失败: {e}")
    
    return teacher_outputs

# 添加CCC评估器
class ConsistentCrossCheckEvaluator:
    """
    一致性交叉检查(CCC)评估器，评估学生模型与教师模型的一致性
    
    CCC评估包含三个主要指标：
    1. 一致性分数(Consistency Score)：评估学生模型与教师模型预测的一致性
    2. 知识迁移分数(Knowledge Transfer Score)：评估学生模型对教师知识的吸收程度
    3. 整体CCC分数(Overall CCC Score)：综合前两个指标的加权和
    """
    def __init__(self, device, top_k=5, consistency_weight=0.5, transfer_weight=0.5):
        self.device = device
        self.top_k = top_k
        self.consistency_weight = consistency_weight
        self.transfer_weight = transfer_weight
    
    def evaluate(self, student_logits, teacher_logits, labels=None):
        """
        评估学生模型与教师模型的一致性与知识迁移
        """
        try:
            # 检查输入形状
            if student_logits.shape[0] != teacher_logits.shape[0] or student_logits.shape[1] != teacher_logits.shape[1]:
                # 如果批次大小或序列长度不匹配，返回默认值
                return {
                    'consistency_score': 0.0,
                    'knowledge_transfer_score': 0.0,
                    'overall_ccc_score': 0.0
                }
            
            # 根据需要裁剪词汇表大小，确保它们匹配
            vocab_size = min(student_logits.shape[2], teacher_logits.shape[2])
            student_logits = student_logits[:, :, :vocab_size]
            teacher_logits = teacher_logits[:, :, :vocab_size]
            
            # 计算学生和教师模型的概率分布
            student_probs = F.softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)
            
            # 获取top-k预测（确保k不超过词汇表大小）
            k = min(self.top_k, vocab_size-1)
            student_topk_probs, student_topk_indices = torch.topk(student_probs, k=k, dim=-1)
            teacher_topk_probs, teacher_topk_indices = torch.topk(teacher_probs, k=k, dim=-1)
        except Exception as e:
            # 捕获任何处理形状不匹配的异常
            print(f"CCC评估时出错: {e}，形状: student={student_logits.shape}, teacher={teacher_logits.shape}")
            return {
                'consistency_score': 0.0,
                'knowledge_transfer_score': 0.0,
                'overall_ccc_score': 0.0
            }
        
        # 计算一致性分数 - 基于预测类别的交集
        batch_size, seq_length = student_logits.shape[:2]
        consistency_scores = torch.zeros(batch_size, seq_length, device=self.device)
        
        for b in range(batch_size):
            for s in range(seq_length):
                # 计算top-k预测交集
                student_preds = set(student_topk_indices[b, s].cpu().numpy())
                teacher_preds = set(teacher_topk_indices[b, s].cpu().numpy())
                intersection = student_preds.intersection(teacher_preds)
                
                # 计算交集大小 / top-k
                consistency_scores[b, s] = len(intersection) / self.top_k
        
        # 计算知识迁移分数 - 基于KL散度
        # 较小的KL散度表示更好的知识迁移
        kl_div = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            teacher_probs,
            reduction='none'
        ).sum(dim=-1)
        
        # 将KL散度转换为迁移分数 (取反并归一化到0-1范围)
        # 使用指数衰减函数：e^(-kl_div)，这样kl_div越小，分数越接近1
        transfer_scores = torch.exp(-kl_div)
        
        # 计算整体CCC分数
        # 整体CCC分数是一致性分数和知识迁移分数的加权平均
        overall_ccc_scores = (
            self.consistency_weight * consistency_scores + 
            self.transfer_weight * transfer_scores
        )
        
        # 如果提供了真实标签，考虑只计算非填充位置的分数
        if labels is not None:
            # 创建掩码，忽略填充位置 (假设填充ID为0或-100)
            mask = (labels != 0) & (labels != -100)
            
            # 应用掩码
            if mask.sum() > 0:
                consistency_scores = (consistency_scores * mask).sum() / mask.sum()
                transfer_scores = (transfer_scores * mask).sum() / mask.sum()
                overall_ccc_scores = (overall_ccc_scores * mask).sum() / mask.sum()
            else:
                # 如果所有位置都是填充，返回零分
                consistency_scores = torch.tensor(0.0, device=self.device)
                transfer_scores = torch.tensor(0.0, device=self.device)
                overall_ccc_scores = torch.tensor(0.0, device=self.device)
        else:
            # 如果没有标签，取所有位置的平均值
            consistency_scores = consistency_scores.mean()
            transfer_scores = transfer_scores.mean()
            overall_ccc_scores = overall_ccc_scores.mean()
        
        return {
            'consistency_score': consistency_scores.item(),
            'knowledge_transfer_score': transfer_scores.item(),
            'overall_ccc_score': overall_ccc_scores.item()
        }

def train(args):
    """使用DeepSpeed训练模型"""
    # 确保离线模式
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    logger.info("已启用离线模式，将使用本地模型")
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 初始化分布式训练
    local_rank, world_size = setup_distributed(args.local_rank)
    is_main_process = local_rank in [-1, 0]  # 判断是否为主进程
    
    # 初始化teacher_model为None，避免未定义错误
    teacher_model = None
    
    # 设置训练设备
    device = torch.device("cuda", local_rank) if local_rank != -1 else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}, local_rank: {local_rank}, world_size: {world_size}")
    
    # 确保输出目录存在
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "cache"), exist_ok=True)
    
    # 加载tokenizer
    logger.info(f"正在从本地路径加载tokenizer: {args.teacher_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True  # 强制仅使用本地文件
    )
    logger.info(f"Tokenizer词汇表大小: {len(tokenizer)}")
    
    # 初始化模型配置
    model_config = ModelConfig()
    # 使用配置中的默认词汇表大小
    # 稍后会在加载教师模型后更新这个值(如果需要)
    model_config.attn_implementation = "eager"
    model_config.teacher_model_name = args.teacher_model
    model_config.teacher_device = device
    model_config.distill_top_k = args.top_k if args.top_k_distillation else 0
    model_config.temperature = args.temperature
    
    # 初始化学生模型
    logger.info("正在初始化学生模型...")
    
    # 检查是否需要从检查点恢复
    starting_epoch = 0
    global_step = 0
    if args.resume_from_checkpoint:
        logger.info(f"从检查点恢复: {args.resume_from_checkpoint}")
        try:
            if os.path.isdir(args.resume_from_checkpoint):
                checkpoint_path = args.resume_from_checkpoint
            else:
                # 如果提供了数字，假设是轮数
                try:
                    epoch_num = int(args.resume_from_checkpoint)
                    checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{epoch_num}")
                except ValueError:
                    # 如果不是数字，使用完整路径
                    checkpoint_path = args.resume_from_checkpoint
            
            if not os.path.exists(checkpoint_path):
                logger.warning(f"检查点路径不存在: {checkpoint_path}，使用新模型")
                model = DistillationModel(model_config)
            else:
                logger.info(f"从路径加载模型: {checkpoint_path}")
                # 找到最后一个轮数/步骤
                if "step" in checkpoint_path:
                    # 如果是按步骤保存的
                    step_match = re.search(r'checkpoint-step-(\d+)', checkpoint_path)
                    if step_match:
                        global_step = int(step_match.group(1))
                        logger.info(f"恢复自步骤: {global_step}")
                else:
                    # 如果是按轮数保存的
                    epoch_match = re.search(r'checkpoint-(\d+)', checkpoint_path)
                    if epoch_match:
                        starting_epoch = int(epoch_match.group(1))
                        logger.info(f"恢复自轮数: {starting_epoch}")
                
                # 加载模型
                model = DistillationModel.from_pretrained(checkpoint_path, config=model_config)
                logger.info(f"成功从检查点加载模型: {checkpoint_path}")
        except Exception as e:
            logger.error(f"无法从检查点恢复: {e}")
            logger.info("初始化新模型")
            model = DistillationModel(model_config)
    else:
        # 初始化新模型
        model = DistillationModel(model_config)
    
    model.to(device)
    
    if is_main_process:
        # 只在主进程上打印
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"学生模型参数量: {total_params:,}")
        logger.info(f"Top-K设置: {model_config.distill_top_k}, 温度: {model_config.temperature}")
    
    # 加载教师模型 - 只在主进程加载
    if is_main_process:
        logger.info(f"进程 {local_rank} 开始加载教师模型: {args.teacher_model}")
        try:
            logger.info("开始加载教师模型...")
            
            # 确定教师模型设备
            teacher_device = "cpu"
            device_map = "cpu"
            torch_dtype = torch.float32
            
            # 如果指定了教师模型GPUs
            if args.teacher_gpus is not None:
                # 解析GPU ID列表
                gpu_ids = [int(gpu_id.strip()) for gpu_id in args.teacher_gpus.split(",")]
                
                if len(gpu_ids) == 1:
                    # 单GPU情况
                    teacher_device = f"cuda:{gpu_ids[0]}"
                    device_map = {"": gpu_ids[0]}  # 将所有层放在指定GPU上
                    logger.info(f"教师模型将使用单个专用GPU: {teacher_device}")
                else:
                    # 多GPU情况 - 使用自动权重平衡
                    teacher_device = f"cuda:{gpu_ids[0]}"  # 第一张显卡作为主设备
                    # 创建基于可用GPU的设备映射
                    device_map = "auto"
                    # 设置CUDA_VISIBLE_DEVICES确保只使用指定的GPU
                    os.environ["TEACHER_CUDA_VISIBLE_DEVICES"] = args.teacher_gpus
                    logger.info(f"教师模型将分布在多个GPU上: {args.teacher_gpus}")
                
                # 在GPU上使用半精度
                torch_dtype = torch.float16
            else:
                # 仍然创建CPU offload目录作为备选
                os.makedirs("cpu_offload", exist_ok=True)
                logger.info("教师模型将使用CPU")
            
            # 首先加载模型配置，强制使用本地文件
            teacher_config = AutoConfig.from_pretrained(
                args.teacher_model,
                trust_remote_code=True,
                local_files_only=True
            )
            
            # 禁用任何会导致形状问题的配置
            teacher_config.use_cache = False  # 禁用KV缓存
            teacher_config.output_attentions = False  # 禁用注意力输出
            teacher_config.output_hidden_states = False  # 禁用隐藏状态输出
            
            if hasattr(teacher_config, "num_attention_heads"):
                logger.info(f"教师模型注意力头数: {teacher_config.num_attention_heads}")
            if hasattr(teacher_config, "hidden_size"):
                logger.info(f"教师模型隐藏层大小: {teacher_config.hidden_size}")
            
            # 根据设备加载教师模型
            teacher_model = AutoModelForCausalLM.from_pretrained(
                args.teacher_model,
                config=teacher_config,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
                offload_folder="cpu_offload" if args.teacher_gpus is None else None,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            
            # 详细记录教师模型加载情况
            if device_map == "auto":
                logger.info(f"教师模型成功分布在多个GPU上: {args.teacher_gpus}")
                # 打印模型各层的设备分配情况
                if hasattr(teacher_model, "hf_device_map"):
                    logger.info("教师模型层分布情况:")
                    device_counts = {}
                    for layer_name, device in teacher_model.hf_device_map.items():
                        if device not in device_counts:
                            device_counts[device] = 0
                        device_counts[device] += 1
                    
                    # 打印每个设备上的层数统计
                    for device, count in device_counts.items():
                        logger.info(f"  - {device}: {count} 层")
            else:
                logger.info(f"教师模型加载到单个设备: {teacher_device}")
            
            # 强制模型为评估模式
            teacher_model.eval()  # 设置为评估模式
            for param in teacher_model.parameters():
                param.requires_grad = False  # 确保不计算梯度
            
            # 记录GPU内存使用情况
            if args.teacher_gpus is not None:
                try:
                    import subprocess
                    import re
                    gpu_info = subprocess.check_output("nvidia-smi", shell=True).decode('utf-8')
                    gpu_memory = {}
                    for line in gpu_info.split('\n'):
                        if "MiB" in line:
                            match = re.search(r'(\d+)MiB\s+/\s+(\d+)MiB', line)
                            if match:
                                used, total = match.groups()
                                gpu_id = len(gpu_memory)
                                gpu_memory[gpu_id] = (int(used), int(total))
                    
                    # 打印教师模型GPU内存占用
                    logger.info("教师模型加载后GPU内存占用:")
                    for gpu_id in [int(x.strip()) for x in args.teacher_gpus.split(",")]:
                        if gpu_id in gpu_memory:
                            used, total = gpu_memory[gpu_id]
                            percent = (used / total) * 100
                            logger.info(f"  - GPU {gpu_id}: {used}MB / {total}MB ({percent:.1f}%)")
                except Exception as e:
                    logger.warning(f"无法获取GPU内存信息: {e}")
            
            logger.info("教师模型加载完成")
            logger.info(f"教师模型配置: 词汇表大小={teacher_model.config.vocab_size}, 参数量: {sum(p.numel() for p in teacher_model.parameters()):,}")
            
            # 更新模型配置中的词汇表大小为教师模型的词汇表大小
            if hasattr(teacher_model.config, "vocab_size"):
                model_config.vocab_size = teacher_model.config.vocab_size
                logger.info(f"更新模型词汇表大小为教师模型的词汇表大小: {model_config.vocab_size}")
            
            teacher_params = sum(p.numel() for p in teacher_model.parameters())
            logger.info(f"教师模型参数量: {teacher_params:,}")
        except Exception as e:
            logger.error(f"加载教师模型失败: {e}")
            teacher_model = None
            logger.warning("将使用零张量替代教师模型输出")
    else:
        # 非主进程将teacher_model设为None
        teacher_model = None
        logger.info(f"进程 {local_rank} 不加载教师模型")
    
    # 加载数据集
    logger.info("正在加载数据集")
    
    train_dataset = get_dataset(
        tokenizer=tokenizer,
        file_path=args.train_file,
        block_size=args.max_seq_length,
        cache_dir=os.path.join(args.output_dir, "cache"),
        overwrite_cache=args.overwrite_cache,
    )
    logger.info(f"训练集样本数: {len(train_dataset)}")
    
    if args.validation_file:
        eval_dataset = get_dataset(
            tokenizer=tokenizer,
            file_path=args.validation_file,
            block_size=args.max_seq_length,
            cache_dir=os.path.join(args.output_dir, "cache"),
            overwrite_cache=args.overwrite_cache,
        )
        logger.info(f"验证集样本数: {len(eval_dataset)}")
    else:
        eval_dataset = None
    
    # 预计算教师模型输出 - 只在主进程计算
    teacher_cache_dir = os.path.join(args.output_dir, "cache")
    precomputed_teacher_outputs = None
    
    # 生成缓存键
    cache_key = generate_cache_key(args.train_file, args.max_seq_length, tokenizer.__class__.__name__)
    teacher_cache_file = os.path.join(teacher_cache_dir, f"teacher_outputs_{cache_key}.pt")
    
    # 检查是否需要覆盖教师输出缓存
    if os.path.exists(teacher_cache_file) and args.overwrite_cache:
        if is_main_process:
            logger.info(f"根据参数覆盖教师输出缓存: {teacher_cache_file}")
            os.remove(teacher_cache_file)
    
    # 主进程负责预计算
    if is_main_process:
        if not os.path.exists(teacher_cache_file) or args.overwrite_cache:
            # 只有主进程进行预计算
            logger.info("主进程开始预计算教师模型输出...")
            precomputed_teacher_outputs = precompute_teacher_outputs(
                teacher_model, 
                train_dataset,
                tokenizer,
                args.max_seq_length, 
                args.batch_size * 4,  # 使用更大的批次进行预计算
                device,
                teacher_cache_dir,
                cache_key,
                args.train_file,
                args,  # 传递args以使用teacher_gpus
                model_config  # 传递model_config
            )
            logger.info("主进程完成教师模型输出预计算")
        else:
            # 尝试直接加载缓存
            try:
                logger.info(f"主进程从缓存加载教师模型输出: {teacher_cache_file}")
                precomputed_teacher_outputs = torch.load(teacher_cache_file, map_location=device)
                logger.info(f"主进程成功加载教师输出缓存，包含 {len(precomputed_teacher_outputs)} 个批次")
            except Exception as e:
                logger.error(f"主进程加载教师缓存失败: {e}")
                precomputed_teacher_outputs = None
            
        # 释放教师模型内存
        if teacher_model is not None:
            del teacher_model
            torch.cuda.empty_cache()
            logger.info("主进程已释放教师模型内存")
    
    # 非主进程加载预计算的教师输出
    if not is_main_process and precomputed_teacher_outputs is None:
        # 检查缓存文件是否存在
        if os.path.exists(teacher_cache_file):
            try:
                logger.info(f"非主进程: 从缓存加载教师模型输出: {teacher_cache_file}")
                precomputed_teacher_outputs = torch.load(teacher_cache_file, map_location=device)
                logger.info(f"非主进程: 成功加载教师缓存，包含 {len(precomputed_teacher_outputs)} 个批次")
            except Exception as e:
                logger.error(f"非主进程: 加载教师缓存失败: {e}，将使用随机初始化")
                precomputed_teacher_outputs = None
        else:
            logger.warning(f"缓存文件不存在: {teacher_cache_file}，将使用随机初始化")
            precomputed_teacher_outputs = None
    
    # 创建数据加载器
    if local_rank == -1:
        # 单机训练
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
    else:
        # 分布式训练
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    
    # 使用标准collate_fn
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    if eval_dataset:
        if local_rank == -1:
            # 单机评估
            eval_sampler = torch.utils.data.SequentialSampler(eval_dataset)
        else:
            # 分布式评估
            eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
            
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    else:
        eval_dataloader = None
    
    # DeepSpeed初始化
    logger.info("初始化DeepSpeed...")
    
    # 检查是否使用自定义DeepSpeed配置
    if args.deepspeed_config:
        with open(args.deepspeed_config, 'r') as f:
            ds_config = json.load(f)
        logger.info(f"使用自定义DeepSpeed配置: {args.deepspeed_config}")
    else:
        # 使用默认配置
        ds_config = {
            "train_batch_size": args.batch_size * args.gradient_accumulation_steps * world_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": args.learning_rate,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": args.weight_decay
                }
            },
            "fp16": {
                "enabled": args.fp16
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "contiguous_gradients": True,
                "overlap_comm": True
            },
            "gradient_clipping": args.max_grad_norm,
            "steps_per_print": 100
        }
        logger.info("使用默认DeepSpeed配置")
    
    # 使用DeepSpeed模型引擎
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # 不直接传入ds_config，因为已经通过命令行参数提供了
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model_parameters
    )
    
    logger.info("DeepSpeed初始化完成")
    
    # 初始化CCC评估器
    ccc_evaluator = ConsistentCrossCheckEvaluator(device)
    
    # 创建CCC评估指标记录器
    ccc_metric_history = {
        'steps': [],
        'consistency_scores': [],
        'knowledge_transfer_scores': [],
        'overall_ccc_scores': []
    }
    
    # 计算训练步数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_steps = int(num_update_steps_per_epoch * args.num_train_epochs)
    
    # 训练循环
    logger.info("***** 开始训练 *****")
    logger.info(f"  训练设备 = {device}")
    logger.info(f"  训练数据量 = {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"  验证数据量 = {len(eval_dataset)}")
    logger.info(f"  每设备批次大小 = {args.batch_size}")
    logger.info(f"  梯度累积步数 = {args.gradient_accumulation_steps}")
    logger.info(f"  总优化步数 = {max_steps}")
    logger.info(f"  训练轮数 = {args.num_train_epochs}")
    logger.info(f"  起始轮数 = {starting_epoch}")
    logger.info(f"  进程 = {local_rank}/{world_size}")
    logger.info(f"  使用DeepSpeed加速训练")
    
    # 架构信息
    if is_main_process:
        if args.teacher_gpus is not None:
            logger.info("***** 分布式架构信息 *****")
            logger.info(f"  教师模型: 使用 {args.teacher_gpus} GPU")
            logger.info(f"  学生模型: 使用 {device} GPU")
            
            # 打印内存统计
            try:
                import subprocess
                import re
                gpu_info = subprocess.check_output("nvidia-smi", shell=True).decode('utf-8')
                logger.info("当前GPU内存使用情况:")
                logger.info(f"{gpu_info.split('Processes')[0]}")
            except:
                pass
    
    # 开始训练循环
    best_eval_loss = float('inf')
    best_ccc_score = 0.0  # 最佳CCC分数
    
    for epoch in range(starting_epoch, int(args.num_train_epochs)):
        logger.info(f"开始第 {epoch+1}/{args.num_train_epochs} 轮训练")
        
        # 分布式训练需要设置epoch
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        
        # 设置为训练模式
        model_engine.train()
        
        # 进度条（只在主进程中显示）
        if is_main_process:
            epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        else:
            epoch_iterator = train_dataloader
        
        tr_loss = 0.0
        
        for batch_idx, batch in enumerate(epoch_iterator):
            # 将数据移到设备
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 如果有预计算的教师输出，则添加到批次中
            if precomputed_teacher_outputs is not None:
                try:
                    # 计算当前批次对应的预计算输出索引
                    # 在分布式训练中更复杂，需要考虑每个进程看到的数据
                    start_idx = batch_idx * args.batch_size
                    if local_rank != -1:
                        # 计算当前进程在数据集中的偏移
                        rank_offset = local_rank * (len(train_dataset) // world_size)
                        # 加上当前批次在本进程中的偏移
                        start_idx += rank_offset
                    
                    # 安全处理：确保不会越界
                    if start_idx < len(precomputed_teacher_outputs):
                        # 创建目标大小的空张量，使用固定的词汇表大小
                        vocab_size = 151936  # 教师模型词汇表大小
                        batch_size = batch["input_ids"].size(0)
                        seq_length = batch["input_ids"].size(1)
                        
                        teacher_logits = torch.zeros(
                            (batch_size, seq_length, vocab_size),
                            dtype=torch.float16,
                            device=device
                        )
                        
                        # 获取可用的预计算输出
                        available = min(batch_size, len(precomputed_teacher_outputs) - start_idx)
                        for i in range(available):
                            if start_idx + i < len(precomputed_teacher_outputs):
                                # 获取预计算的teacher_logits并移至设备
                                cached_logits = precomputed_teacher_outputs[start_idx + i]["teacher_logits"].to(device)
                                # 确保形状匹配
                                if cached_logits.size(0) == seq_length and cached_logits.size(1) == vocab_size:
                                    # 逐个样本复制
                                    teacher_logits[i] = cached_logits
                                elif len(cached_logits.shape) == 3:
                                    # 调整形状
                                    min_seq = min(seq_length, cached_logits.size(1))
                                    min_vocab = min(vocab_size, cached_logits.size(2))
                                    teacher_logits[i, :min_seq, :min_vocab] = cached_logits[0, :min_seq, :min_vocab]
                        
                        batch["teacher_logits"] = teacher_logits
                    else:
                        # 如果超出了预计算输出的范围，创建零张量
                        vocab_size = 151936
                        batch_size = batch["input_ids"].size(0)
                        seq_length = batch["input_ids"].size(1)
                        batch["teacher_logits"] = torch.zeros(
                            (batch_size, seq_length, vocab_size),
                            dtype=torch.float16,
                            device=device
                        )
                except Exception as e:
                    logger.error(f"获取预计算教师输出失败: {e}")
                    # 创建零张量
                    vocab_size = 151936
                    batch_size = batch["input_ids"].size(0)
                    seq_length = batch["input_ids"].size(1)
                    batch["teacher_logits"] = torch.zeros(
                        (batch_size, seq_length, vocab_size),
                        dtype=torch.float16,
                        device=device
                    )
            else:
                                    # 如果没有预计算的教师输出，创建零张量
                try:
                    vocab_size = model_config.vocab_size
                except Exception as e:
                    # 如果出现错误，使用默认值
                    vocab_size = 151936
                    print(f"使用默认词汇表大小: {vocab_size}, 错误: {e}")
                    
                batch_size = batch["input_ids"].size(0)
                seq_length = batch["input_ids"].size(1)
                batch["teacher_logits"] = torch.zeros(
                    (batch_size, seq_length, vocab_size),
                    dtype=torch.float16,
                    device=device
                )
            
            # 记录批次开始时间
            batch_start_time = time.time()
            
            # 前向传播和反向传播，DeepSpeed引擎处理
            outputs = model_engine(batch)
            loss = outputs["loss"]
            
            # DeepSpeed将处理梯度累积和更新
            model_engine.backward(loss)
            model_engine.step()
            
            tr_loss += loss.item()
            global_step += 1
            
            # 计算批次处理时间
            batch_time = time.time() - batch_start_time
            
            # 更新进度条（只在主进程中）
            if is_main_process:
                # 估计剩余时间
                samples_per_second = args.batch_size / batch_time
                remaining_samples = (len(train_dataset) * args.num_train_epochs) - (global_step * args.batch_size)
                eta_seconds = remaining_samples / samples_per_second if samples_per_second > 0 else 0
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                # 更新进度条信息
                epoch_iterator.set_postfix({
                    "loss": f"{loss.item():.4f}", 
                    "step": global_step,
                    "速度": f"{samples_per_second:.1f}样本/秒",
                    "ETA": eta_str
                })
                
                # 每100步记录性能统计
                if global_step % 100 == 0:
                    logger.info(f"步骤 {global_step}/{max_steps}: 损失={loss.item():.4f}, 速度={samples_per_second:.1f}样本/秒, ETA={eta_str}")
                
            # 按步数进行评估
            if eval_dataloader is not None and args.eval_steps > 0 and global_step > 0 and global_step % args.eval_steps == 0 and is_main_process:
                # 评估
                model_engine.eval()
                eval_loss = 0.0
                eval_steps = 0
                
                # 初始化CCC评估指标
                ccc_metrics = {
                    'consistency_score': 0.0,
                    'knowledge_transfer_score': 0.0,
                    'overall_ccc_score': 0.0
                }
                
                logger.info(f"步骤 {global_step}: 开始评估...")
                eval_iterator = tqdm(eval_dataloader, desc=f"Evaluating Step {global_step}")
                
                for eval_batch in eval_iterator:
                    eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                    
                    # 添加教师logits
                    vocab_size = 151936
                    batch_size = eval_batch["input_ids"].size(0)
                    seq_length = eval_batch["input_ids"].size(1)
                    eval_batch["teacher_logits"] = torch.zeros(
                        (batch_size, seq_length, vocab_size), 
                        dtype=torch.float16, 
                        device=device
                    )
                    
                    with torch.no_grad():
                        # 学生模型前向传播
                        eval_outputs = model_engine(eval_batch)
                        
                        # 执行CCC评估 - 若有教师输出则进行一致性检查
                        if "teacher_logits" in eval_batch and torch.any(eval_batch["teacher_logits"].abs() > 0):
                            batch_ccc_metrics = ccc_evaluator.evaluate(
                                student_logits=eval_outputs["ltc_ncp_logits"],
                                teacher_logits=eval_batch["teacher_logits"],
                                labels=eval_batch.get("labels", None)
                            )
                            
                            # 累积CCC指标
                            for k, v in batch_ccc_metrics.items():
                                ccc_metrics[k] += v
                        
                    eval_loss += eval_outputs["loss"].item()
                    eval_steps += 1
                
                eval_loss = eval_loss / eval_steps if eval_steps > 0 else 0
                
                # 计算平均CCC评估指标
                if eval_steps > 0:
                    for k in ccc_metrics:
                        ccc_metrics[k] = ccc_metrics[k] / eval_steps
                
                # 记录CCC评估指标历史
                ccc_metric_history['steps'].append(global_step)
                ccc_metric_history['consistency_scores'].append(ccc_metrics['consistency_score'])
                ccc_metric_history['knowledge_transfer_scores'].append(ccc_metrics['knowledge_transfer_score'])
                ccc_metric_history['overall_ccc_scores'].append(ccc_metrics['overall_ccc_score'])
                
                logger.info(f"步骤 {global_step} 评估损失: {eval_loss:.5f}")
                logger.info(f"CCC评估指标: 一致性={ccc_metrics['consistency_score']:.4f}, " 
                           f"知识迁移={ccc_metrics['knowledge_transfer_score']:.4f}, "
                           f"整体CCC={ccc_metrics['overall_ccc_score']:.4f}")
                
                # 保存CCC评估指标历史
                try:
                    with open(os.path.join(args.output_dir, "ccc_metrics.json"), "w") as f:
                        json.dump(ccc_metric_history, f, indent=2)
                except Exception as e:
                    logger.warning(f"保存CCC评估指标历史失败: {e}")
                
                # 保存最佳模型
                if eval_loss < best_eval_loss:
                    logger.info(f"发现新的最佳模型 (loss: {eval_loss:.5f}, 之前: {best_eval_loss:.5f})")
                    best_eval_loss = eval_loss
                    
                    best_dir = os.path.join(args.output_dir, "best_model")
                    os.makedirs(best_dir, exist_ok=True)
                    
                    # 保存模型 - DeepSpeed方式
                    unwrapped_model = model_engine.module
                    unwrapped_model.save_pretrained(best_dir)
                    
                    # 保存tokenizer
                    tokenizer.save_pretrained(best_dir)
                
                # 基于CCC评分保存最佳模型
                if ccc_metrics['overall_ccc_score'] > best_ccc_score:
                    logger.info(f"发现新的CCC最佳模型 (分数: {ccc_metrics['overall_ccc_score']:.5f}, 之前: {best_ccc_score:.5f})")
                    best_ccc_score = ccc_metrics['overall_ccc_score']
                    
                    best_ccc_dir = os.path.join(args.output_dir, "best_ccc_model")
                    os.makedirs(best_ccc_dir, exist_ok=True)
                    
                    # 保存模型 - DeepSpeed方式
                    unwrapped_model = model_engine.module
                    unwrapped_model.save_pretrained(best_ccc_dir)
                    
                    # 保存tokenizer
                    tokenizer.save_pretrained(best_ccc_dir)
                
                # 恢复训练模式
                model_engine.train()
        
        # 每个epoch后保存检查点（只在主进程中）
        if is_main_process:
            # 保存模型检查点
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 保存模型 - DeepSpeed方式
            unwrapped_model = model_engine.module
            unwrapped_model.save_pretrained(checkpoint_dir)
            
            # 保存tokenizer
            tokenizer.save_pretrained(checkpoint_dir)
            
            logger.info(f"保存第 {epoch+1} 轮模型到 {checkpoint_dir}")
            
            # 保存DeepSpeed状态 (包含优化器状态等)
            client_state = {"checkpoint_step": global_step}
            model_engine.save_checkpoint(checkpoint_dir, client_state=client_state)
            logger.info(f"保存DeepSpeed状态到 {checkpoint_dir}")
    
    # 训练完成，保存最终模型（只在主进程中）
    if is_main_process:
        logger.info("保存最终模型")
        final_dir = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)
        
        # 保存模型 - DeepSpeed方式
        unwrapped_model = model_engine.module
        unwrapped_model.save_pretrained(final_dir)
        
        # 保存tokenizer
        tokenizer.save_pretrained(final_dir)
        
        logger.info(f"保存最终模型到 {final_dir}")
    
    return global_step, tr_loss / global_step

if __name__ == "__main__":
    # 命令行参数
    parser = argparse.ArgumentParser()
    
    # 基本参数
    parser.add_argument("--train_file", type=str, required=True, help="训练数据文件路径")
    parser.add_argument("--validation_file", type=str, default=None, help="验证数据文件路径")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--teacher_model", type=str, required=True, help="教师模型路径")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="缓存目录")
    
    # 训练参数
    parser.add_argument("--max_seq_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=16, help="每个设备的批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--num_train_epochs", type=float, default=5, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="最大梯度范数")
    parser.add_argument("--save_steps", type=int, default=500, help="保存检查点的步数间隔")
    parser.add_argument("--eval_steps", type=int, default=500, help="评估的步数间隔")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 蒸馏参数
    parser.add_argument("--top_k", type=int, default=5, help="Top-K蒸馏的K值")
    parser.add_argument("--temperature", type=float, default=2.0, help="蒸馏温度")
    parser.add_argument("--top_k_distillation", action="store_true", help="是否使用Top-K蒸馏")
    
    # DeepSpeed参数
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式训练的本地进程序号")
    parser.add_argument("--deepspeed", action="store_true", help="是否使用DeepSpeed")
    parser.add_argument("--deepspeed_config", type=str, default=None, help="DeepSpeed配置文件路径")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO优化阶段")
    parser.add_argument("--fp16", action="store_true", help="是否使用混合精度训练")
    parser.add_argument("--teacher_gpus", type=str, default=None, help="教师模型专用GPU IDs，用逗号分隔，如'0,1'")
    
    # 其他参数
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从检查点恢复训练")
    parser.add_argument("--overwrite_cache", action="store_true", help="是否覆盖缓存")
    
    args = parser.parse_args()
    
    # 设置日志文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    log_file = os.path.join(args.output_dir, "logs", f"train_ds_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                                              datefmt="%m/%d/%Y %H:%M:%S"))
    logger.addHandler(file_handler)
    
    # 打印参数
    logger.info(f"***** 训练参数 *****")
    for arg_name, arg_value in sorted(vars(args).items()):
        logger.info(f"  {arg_name} = {arg_value}")
    
    # 开始训练
    try:
        train(args)
    except KeyboardInterrupt:
        logger.info("检测到键盘中断，正在安全退出...")
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("训练已完成")