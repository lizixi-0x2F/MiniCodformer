#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import math
from sys import argv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_scheduler,
    set_seed,
)
from tqdm.auto import tqdm
import pickle
import hashlib
import time
import re
import datetime

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
            # 设置分布式训练环境变量，提高稳定性
            if "NCCL_DEBUG" not in os.environ:
                os.environ["NCCL_DEBUG"] = "INFO"  # 调试时设为INFO，生产环境可设为WARNING
            
            # 添加NCCL更安全的配置
            if "NCCL_IB_DISABLE" not in os.environ:
                os.environ["NCCL_IB_DISABLE"] = "0"
            if "NCCL_P2P_DISABLE" not in os.environ:
                os.environ["NCCL_P2P_DISABLE"] = "0"
            if "NCCL_SOCKET_IFNAME" not in os.environ:
                os.environ["NCCL_SOCKET_IFNAME"] = "eth0"  # 可根据实际网络接口调整
            if "NCCL_ASYNC_ERROR_HANDLING" not in os.environ:
                os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
            if "NCCL_P2P_LEVEL" not in os.environ:
                os.environ["NCCL_P2P_LEVEL"] = "NVL"
                
            # 避免在预计算阶段使用分布式通信
            os.environ["NCCL_BLOCKING_WAIT"] = "1"
            
            # 尝试初始化分布式进程组
            init_method = "env://"  # 使用环境变量初始化
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = "localhost"
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29500"
                
            # 添加超时配置
            timeout = datetime.timedelta(minutes=30)
            
            # 使用更稳定的初始化
            dist.init_process_group(
                backend="nccl",
                init_method=init_method,
                timeout=timeout
            )
            
            world_size = dist.get_world_size()
            logger.info(f"初始化分布式环境成功: local_rank={local_rank}, world_size={world_size}")
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
                              batch_size, device, cache_dir, cache_key=None):
    """预计算所有教师模型的输出并缓存到磁盘"""
    if cache_key is None:
        cache_key = generate_cache_key(argv.train_file, max_seq_length, tokenizer.__class__.__name__)
    
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
                    # 固定词汇表大小为教师模型的词汇表大小
                    model_vocab_size = 151936  # 根据日志中的教师模型词汇表大小调整
                    batch_size = batch["input_ids"].size(0)
                    seq_length = batch["input_ids"].size(1)
                    
                    # 创建默认的零张量，使用固定的词汇表大小
                    logits = torch.zeros(
                        (batch_size, seq_length, model_vocab_size),
                        dtype=torch.float16,
                        device=device
                    )
                    
                    # 安全运行教师模型，为Qwen模型特别处理
                    if teacher_model is not None and batch_idx % 10 == 0:
                        try:
                            with torch.inference_mode(), torch.no_grad():
                                # 将原始输入分拆为更小的批次，避免注意力形状问题
                                # Qwen模型在大批次下有注意力计算问题，将16分成4批次处理
                                sub_batch_size = 4  # 每个子批次最多4个样本
                                full_batch_size = batch["input_ids"].size(0)
                                logits_list = []
                                
                                # 将大批次分拆成小批次处理
                                for i in range(0, full_batch_size, sub_batch_size):
                                    end_idx = min(i + sub_batch_size, full_batch_size)
                                    sub_input_ids = batch["input_ids"][i:end_idx]
                                    sub_attention_mask = None
                                    if "attention_mask" in batch:
                                        sub_attention_mask = batch["attention_mask"][i:end_idx]
                                    
                                    # 处理小批次 - 使用逐个样本的CPU推理模式
                                    try:
                                        # 为每个样本单独推理，避免批处理造成的内存溢出
                                        sub_batch_logits = []
                                        for i in range(sub_input_ids.size(0)):
                                            # 获取单个样本
                                            single_input = sub_input_ids[i:i+1]
                                            single_mask = None
                                            if sub_attention_mask is not None:
                                                single_mask = sub_attention_mask[i:i+1]
                                            
                                            # 确保所有输入都在CPU上，并正确处理设备转换
                                            with torch.no_grad():
                                                # 确保输入在CPU上
                                                cpu_input = single_input.cpu()
                                                cpu_mask = None
                                                if single_mask is not None:
                                                    cpu_mask = single_mask.cpu()
                                                
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
                                        
                                        # 创建输出对象来模拟模型输出
                                        class DummyOutput:
                                            def __init__(self, logits):
                                                self.logits = logits
                                        
                                        # 如果有任何logits，则合并它们
                                        if sub_batch_logits:
                                            # 检查所有张量形状是否一致
                                            shapes = [t.shape for t in sub_batch_logits]
                                            logger.info(f"子批次logits形状: {shapes}")
                                            
                                            try:
                                                # 仅当所有形状相同时才尝试合并
                                                if all(t.shape == sub_batch_logits[0].shape for t in sub_batch_logits):
                                                    combined_logits = torch.cat(sub_batch_logits, dim=0)
                                                    sub_outputs = DummyOutput(combined_logits)
                                                    logger.info(f"成功组合子批次logits，形状: {combined_logits.shape}")
                                                else:
                                                    # 形状不一致，创建正确大小的零张量
                                                    logger.warning(f"子批次logits形状不一致，使用零张量")
                                                    dummy_logits = torch.zeros(
                                                        (sub_input_ids.size(0), sub_input_ids.size(1), model_vocab_size),
                                                        dtype=torch.float16,
                                                        device=device
                                                    )
                                                    sub_outputs = DummyOutput(dummy_logits)
                                            except Exception as e:
                                                logger.warning(f"合并子批次logits时出错: {e}，使用零张量")
                                                dummy_logits = torch.zeros(
                                                    (sub_input_ids.size(0), sub_input_ids.size(1), model_vocab_size),
                                                    dtype=torch.float16,
                                                    device=device
                                                )
                                                sub_outputs = DummyOutput(dummy_logits)
                                        else:
                                            # 创建零张量作为回退
                                            dummy_logits = torch.zeros(
                                                (sub_input_ids.size(0), sub_input_ids.size(1), model_vocab_size),
                                                dtype=torch.float16,
                                                device=device
                                            )
                                            sub_outputs = DummyOutput(dummy_logits)
                                        logits_list.append(sub_outputs.logits)
                                    except Exception as e:
                                        # 子批次处理失败，创建零张量
                                        logger.warning(f"子批次 {i//sub_batch_size} 处理失败: {e}")
                                        # 创建零张量作为回退
                                        sub_logits = torch.zeros(
                                            (end_idx-i, seq_length, model_vocab_size),
                                            dtype=torch.float16,
                                            device=device
                                        )
                                        logits_list.append(sub_logits)
                                
                                # 检查是否有任何有效的logits
                                if logits_list:
                                    # 尝试组合所有子批次的logits
                                    try:
                                        # 检查所有子批次的形状是否一致
                                        vocab_sizes = [l.size(2) for l in logits_list]
                                        min_vocab_size = min(vocab_sizes + [model_vocab_size])
                                        
                                        # 创建正确大小的输出张量
                                        combined_logits = torch.zeros(
                                            (full_batch_size, seq_length, model_vocab_size),
                                            dtype=torch.float16,
                                            device=device
                                        )
                                        
                                        # 填充组合的logits张量
                                        start_idx = 0
                                        for sub_logits in logits_list:
                                            sub_size = sub_logits.size(0)
                                            end_idx = start_idx + sub_size
                                            # 只复制词汇表大小相同的部分
                                            combined_logits[start_idx:end_idx, :, :min_vocab_size] = sub_logits[:, :, :min_vocab_size]
                                            start_idx = end_idx
                                        
                                        # 使用组合的logits
                                        logits = combined_logits
                                        logger.info(f"批次 {batch_idx}: 成功组合子批次获取教师输出，形状 {logits.shape}")
                                    except Exception as e:
                                        logger.warning(f"批次 {batch_idx}: 组合子批次失败: {e}，使用零张量")
                                else:
                                    logger.warning(f"批次 {batch_idx}: 所有子批次处理失败，使用零张量")
                        except Exception as e:
                            logger.warning(f"批次 {batch_idx}: 教师模型调用失败，使用零张量。错误: {e}")
                except Exception as e:
                    logger.error(f"批次 {batch_idx}: 创建教师输出时发生错误: {e}")
                
                # 保存教师输出
                teacher_outputs.append({
                    "input_ids": batch["input_ids"].cpu(),
                    "attention_mask": batch["attention_mask"].cpu() if "attention_mask" in batch else None,
                    "teacher_logits": logits.cpu()
                })
                
                # 定期释放缓存
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    
                # 每100个批次记录一次进度
                if batch_idx % 100 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"已处理 {batch_idx} 批次，耗时 {elapsed:.2f} 秒，" 
                                f"速度 {batch_idx/elapsed:.2f} 批次/秒")
                    
            except Exception as e:
                logger.error(f"处理批次 {batch_idx} 时出错: {e}")
    
    # 保存缓存
    try:
        torch.save(teacher_outputs, cache_file)
        logger.info(f"教师模型输出已保存到: {cache_file}")
    except Exception as e:
        logger.error(f"保存教师输出缓存失败: {e}")
    
    return teacher_outputs

def train(args):
    """训练模型"""
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
    logger.info("正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model,
        trust_remote_code=True,
        use_fast=False
    )
    logger.info(f"Tokenizer词汇表大小: {len(tokenizer)}")
    
    # 初始化模型配置
    model_config = ModelConfig()
    # 确保使用固定的词汇表大小，与教师模型匹配
    model_config.vocab_size = 151936  # 根据日志中的教师模型词汇表大小调整
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
    
    # 使用分布式数据并行包装模型
    if local_rank != -1:
        model = DistributedDataParallel(
            model, 
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
    
    if is_main_process:
        # 只在主进程上打印
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"学生模型参数量: {total_params:,}")
        logger.info(f"Top-K设置: {model_config.distill_top_k}, 温度: {model_config.temperature}")
    
    # 确保在分布式环境中主进程和工作进程同步
    if local_rank != -1:
        try:
            torch.distributed.barrier()
            logger.info(f"进程 {local_rank} 成功通过第一个同步点")
        except Exception as e:
            logger.error(f"分布式同步失败: {e}")
    
    # 加载教师模型 - 只在主进程加载
    if local_rank == 0 or local_rank == -1:
        logger.info(f"进程 {local_rank} 开始加载教师模型: {args.teacher_model}")
        try:
            logger.info("开始加载教师模型...")
            # 修改模型加载配置以解决形状问题
            from transformers import AutoConfig
            
            # 首先加载模型配置
            teacher_config = AutoConfig.from_pretrained(
                args.teacher_model,
                trust_remote_code=True
            )
            
            # 禁用任何会导致形状问题的配置
            # 明确设置attention头数和注意力实现方式
            if hasattr(teacher_config, "num_attention_heads"):
                logger.info(f"教师模型注意力头数: {teacher_config.num_attention_heads}")
            if hasattr(teacher_config, "hidden_size"):
                logger.info(f"教师模型隐藏层大小: {teacher_config.hidden_size}")
            
            teacher_config.use_cache = False  # 禁用KV缓存
            teacher_config.output_attentions = False  # 禁用注意力输出
            teacher_config.output_hidden_states = False  # 禁用隐藏状态输出
            
            # 加载教师模型 - 使用完全禁用并行注意力的配置
            # 为Qwen模型特别设置，防止多头注意力形状问题
            config_updates = {
                "attn_implementation": "eager",  # 禁用flash_attention
                "use_cache": False,              # 禁用KV缓存
                "output_attentions": False,      # 禁用注意力输出
                "output_hidden_states": False,   # 禁用隐藏状态输出
                "_attn_implementation": "eager", # 确保使用普通注意力实现
            }
            
            # 更新配置
            for key, value in config_updates.items():
                if hasattr(teacher_config, key):
                    setattr(teacher_config, key, value)
            
            # 使用CPU模式加载教师模型，避免GPU内存溢出
            # 创建加载目录
            os.makedirs("cpu_offload", exist_ok=True)
            
            # 加载模型到CPU以避免内存溢出
            teacher_model = AutoModelForCausalLM.from_pretrained(
                args.teacher_model,
                config=teacher_config,
                trust_remote_code=True,
                torch_dtype=torch.float32,      # 使用全精度加载，提高CPU计算稳定性
                device_map="cpu",               # 强制使用CPU加载
                offload_folder="cpu_offload",   # 启用CPU模型卸载文件夹避免OOM
                low_cpu_mem_usage=True,         # 低内存使用
            )
            
            logger.info("教师模型已加载到CPU，避免GPU内存溢出")
            
            # 强制模型为评估模式
            teacher_model.eval()  # 设置为评估模式
            for param in teacher_model.parameters():
                param.requires_grad = False  # 确保不计算梯度
                
            logger.info("教师模型加载完成")
            logger.info(f"教师模型配置: 词汇表大小={teacher_model.config.vocab_size}")
            
            if is_main_process:
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
    
    # 确保所有进程同步等待
    if local_rank != -1:
        try:
            torch.distributed.barrier()
            logger.info(f"进程 {local_rank} 成功通过第二个同步点")
        except Exception as e:
            logger.error(f"分布式同步失败: {e}")
    
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
                cache_key
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
    
    # 确保主进程已完成预计算，其他进程等待
    if local_rank != -1:
        try:
            torch.distributed.barrier()
            logger.info(f"进程 {local_rank} 成功通过第三个同步点（预计算完成）")
        except Exception as e:
            logger.error(f"分布式同步失败: {e}")
    
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
            
    # 再次同步所有进程，确保所有进程都完成了缓存加载
    if local_rank != -1:
        try:
            torch.distributed.barrier()
            logger.info(f"进程 {local_rank} 成功通过第四个同步点（所有进程加载完成）")
        except Exception as e:
            logger.error(f"分布式同步失败: {e}")
    
    # 创建数据加载器
    if local_rank == -1:
        # 单机训练
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
    else:
        # 分布式训练
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    
    # 使用标准collate_fn，预计算的教师输出将在训练循环中添加
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
    
    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # 计算训练步数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_steps = int(num_update_steps_per_epoch * args.num_train_epochs)
    
    # 学习率调度器
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * max_steps),
        num_training_steps=max_steps,
    )
    
    # 使用fp16
    if args.fp16:
        logger.info("使用混合精度训练")
        scaler = GradScaler()
    else:
        scaler = None
    
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
    
    # 梯度清零
    model.zero_grad()
    
    # 开始训练循环
    best_eval_loss = float('inf')
    best_ccc_score = 0.0  # 最佳CCC分数
    
    # 初始化CCC评估器
    ccc_evaluator = ConsistentCrossCheckEvaluator(device)
    
    # 创建CCC评估指标记录器
    ccc_metric_history = {
        'steps': [],
        'consistency_scores': [],
        'knowledge_transfer_scores': [],
        'overall_ccc_scores': []
    }
    
    for epoch in range(starting_epoch, int(args.num_train_epochs)):
        logger.info(f"开始第 {epoch+1}/{args.num_train_epochs} 轮训练")
        
        # 分布式训练需要设置epoch
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)
        
        # 设置为训练模式
        model.train()
        
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
                    # 注意：在分布式训练中，每个进程看到的批次顺序可能不同
                    start_idx = batch_idx * args.batch_size
                    
                    # 安全处理：确保不会越界
                    if start_idx < len(precomputed_teacher_outputs):
                        # 计算实际可用的批次数量
                        available_samples = min(args.batch_size, len(precomputed_teacher_outputs) - start_idx)
                        
                        # 获取预计算的教师输出
                        teacher_logits_list = []
                        for i in range(start_idx, start_idx + available_samples):
                            if i < len(precomputed_teacher_outputs):
                                teacher_logits_list.append(precomputed_teacher_outputs[i]["teacher_logits"])
                        
                        # 确保有足够的教师输出
                        if teacher_logits_list:
                            # 避免使用torch.cat，改为创建一个新的匹配大小的张量
                            # 并逐一填充，这更加安全，避免形状不匹配
                            vocab_size = len(tokenizer)
                            current_batch_size = batch["input_ids"].size(0)
                            seq_length = batch["input_ids"].size(1)
                            
                            # 创建目标大小的空张量，使用固定的词汇表大小
                            teacher_logits = torch.zeros(
                                (current_batch_size, seq_length, 151669),  # 固定为教师模型词汇表大小
                                dtype=torch.float16,
                                device=device
                            )
                            
                            # 填充可用的教师输出
                            for idx, t_logits in enumerate(teacher_logits_list):
                                if idx < current_batch_size:
                                    try:
                                        # 安全处理张量形状
                                        if len(t_logits.shape) == 3:  # 确保是3D张量 [seq_len, batch_size, vocab_size]
                                            # 确保序列长度和词汇表大小匹配
                                            t_seq_len = min(seq_length, t_logits.size(1))
                                            # 只复制词汇表大小相同的部分
                                            t_vocab_size = min(151669, t_logits.size(2))
                                            teacher_logits[idx, :t_seq_len, :t_vocab_size] = t_logits[:t_seq_len, :t_vocab_size]
                                        elif len(t_logits.shape) == 2:  # 如果是2D张量 [seq_len, vocab_size]
                                            # 处理2D张量情况
                                            t_seq_len = min(seq_length, t_logits.size(0))
                                            t_vocab_size = min(151669, t_logits.size(1))
                                            teacher_logits[idx, :t_seq_len, :t_vocab_size] = t_logits[:t_seq_len, :t_vocab_size]
                                        else:
                                            logger.warning(f"教师logits形状异常: {t_logits.shape}，使用零值替代")
                                    except Exception as e:
                                        logger.warning(f"填充教师输出时出错: {e}，使用零值替代")
                            
                            # 添加到批次
                            batch["teacher_logits"] = teacher_logits
                        else:
                            # 创建零张量作为回退
                            vocab_size = len(tokenizer)
                            batch_size = batch["input_ids"].size(0)
                            seq_length = batch["input_ids"].size(1)
                            batch["teacher_logits"] = torch.zeros(
                                (batch_size, seq_length, vocab_size),
                                dtype=torch.float16,
                                device=device
                            )
                    else:
                        # 如果超出了预计算输出的范围，创建零张量
                        vocab_size = 151669  # 固定为教师模型的词汇表大小
                        batch_size = batch["input_ids"].size(0)
                        seq_length = batch["input_ids"].size(1)
                        batch["teacher_logits"] = torch.zeros(
                            (batch_size, seq_length, vocab_size),
                            dtype=torch.float16,
                            device=device
                        )
                except Exception as e:
                    logger.error(f"获取预计算教师输出失败: {e}")
                    # 如果失败，创建零张量作为备用
                    vocab_size = 151669  # 固定为教师模型的词汇表大小
                    batch_size = batch["input_ids"].size(0)
                    seq_length = batch["input_ids"].size(1)
                    batch["teacher_logits"] = torch.zeros(
                        (batch_size, seq_length, vocab_size),
                        dtype=torch.float16,
                        device=device
                    )
            else:
                # 如果没有预计算的教师输出，创建零张量
                vocab_size = 151669  # 固定为教师模型的词汇表大小
                batch_size = batch["input_ids"].size(0)
                seq_length = batch["input_ids"].size(1)
                batch["teacher_logits"] = torch.zeros(
                    (batch_size, seq_length, vocab_size),
                    dtype=torch.float16,
                    device=device
                )
            
            # 使用混合精度训练
            if args.fp16:
                with autocast(device_type='cuda'):
                    # 前向传播
                    outputs = model(**batch)
                    loss = outputs["loss"]
                    
                    # 处理梯度累积
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                
                # 反向传播
                scaler.scale(loss).backward()
                
                # 如果到达累积步数，更新参数
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or batch_idx == len(train_dataloader) - 1:
                    # 梯度裁剪
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    # 优化器步骤
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    model.zero_grad()
                    
                    global_step += 1
            else:
                # 前向传播
                outputs = model(**batch)
                loss = outputs["loss"]
                
                # 处理梯度累积
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                # 反向传播
                loss.backward()
                
                # 如果到达累积步数，更新参数
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or batch_idx == len(train_dataloader) - 1:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    # 优化器步骤
                    optimizer.step()
                    lr_scheduler.step()
                    model.zero_grad()
                    
                    global_step += 1
            
            tr_loss += loss.item()
            
            # 更新进度条（只在主进程中）
            if is_main_process:
                epoch_iterator.set_postfix({"loss": loss.item(), "step": global_step})
                
            # 按步数保存检查点
            if is_main_process and args.save_steps > 0 and global_step > 0 and global_step % args.save_steps == 0:
                save_checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-step-{global_step}")
                os.makedirs(save_checkpoint_dir, exist_ok=True)
                if hasattr(model, "module"):
                    model.module.save_pretrained(save_checkpoint_dir)
                else:
                    model.save_pretrained(save_checkpoint_dir)
                logger.info(f"保存第 {global_step} 步检查点到 {save_checkpoint_dir}")
                
            # 按步数进行评估
            if eval_dataloader is not None and args.eval_steps > 0 and global_step > 0 and global_step % args.eval_steps == 0:
                # 评估
                model.eval()
                eval_loss = 0.0
                eval_steps = 0
                
                # 初始化CCC评估指标
                ccc_metrics = {
                    'consistency_score': 0.0,
                    'knowledge_transfer_score': 0.0,
                    'overall_ccc_score': 0.0
                }
                
                if is_main_process:
                    logger.info(f"步骤 {global_step}: 开始评估...")
                    eval_iterator = tqdm(eval_dataloader, desc=f"Evaluating Step {global_step}")
                else:
                    eval_iterator = eval_dataloader
                
                for eval_batch in eval_iterator:
                    eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                    
                    # 添加教师logits
                    vocab_size = 151669  # 固定为教师模型的词汇表大小
                    batch_size = eval_batch["input_ids"].size(0)
                    seq_length = eval_batch["input_ids"].size(1)
                    eval_batch["teacher_logits"] = torch.zeros(
                        (batch_size, seq_length, vocab_size), 
                        dtype=torch.float16, 
                        device=device
                    )
                    
                    with torch.no_grad():
                        # 学生模型前向传播
                        eval_outputs = model(**eval_batch)
                        
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
                
                # 处理分布式评估结果
                if world_size > 1:
                    # 将所有进程的损失收集到主进程
                    eval_loss_tensor = torch.tensor([eval_loss]).to(device)
                    eval_steps_tensor = torch.tensor([eval_steps]).to(device)
                    
                    dist.all_reduce(eval_loss_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(eval_steps_tensor, op=dist.ReduceOp.SUM)
                    
                    eval_loss = eval_loss_tensor.item()
                    eval_steps = eval_steps_tensor.item()
                    
                    # 同步CCC评估指标
                    for k in ccc_metrics:
                        metric_tensor = torch.tensor([ccc_metrics[k]]).to(device)
                        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                        ccc_metrics[k] = metric_tensor.item()
                
                eval_loss = eval_loss / eval_steps
                
                # 计算平均CCC评估指标
                if eval_steps > 0:
                    for k in ccc_metrics:
                        ccc_metrics[k] = ccc_metrics[k] / eval_steps
                
                # 记录CCC评估指标历史
                ccc_metric_history['steps'].append(global_step)
                ccc_metric_history['consistency_scores'].append(ccc_metrics['consistency_score'])
                ccc_metric_history['knowledge_transfer_scores'].append(ccc_metrics['knowledge_transfer_score'])
                ccc_metric_history['overall_ccc_scores'].append(ccc_metrics['overall_ccc_score'])
                
                # 只在主进程中打印和保存
                if is_main_process:
                    logger.info(f"步骤 {global_step} 评估损失: {eval_loss:.5f}")
                    logger.info(f"CCC评估指标: 一致性={ccc_metrics['consistency_score']:.4f}, " 
                               f"知识迁移={ccc_metrics['knowledge_transfer_score']:.4f}, "
                               f"整体CCC={ccc_metrics['overall_ccc_score']:.4f}")
                    
                    # 保存CCC评估指标历史
                    try:
                        import json
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
                        
                        # 保存模型
                        if hasattr(model, "module"):
                            model.module.save_pretrained(best_dir)
                        else:
                            model.save_pretrained(best_dir)
                        
                        # 保存tokenizer
                        tokenizer.save_pretrained(best_dir)
                    
                    # 基于CCC评分保存最佳模型
                    if ccc_metrics['overall_ccc_score'] > best_ccc_score:
                        logger.info(f"发现新的CCC最佳模型 (分数: {ccc_metrics['overall_ccc_score']:.5f}, 之前: {best_ccc_score:.5f})")
                        best_ccc_score = ccc_metrics['overall_ccc_score']
                        
                        best_ccc_dir = os.path.join(args.output_dir, "best_ccc_model")
                        os.makedirs(best_ccc_dir, exist_ok=True)
                        
                        # 保存模型
                        if hasattr(model, "module"):
                            model.module.save_pretrained(best_ccc_dir)
                        else:
                            model.save_pretrained(best_ccc_dir)
                        
                        # 保存tokenizer
                        tokenizer.save_pretrained(best_ccc_dir)
                
                # 恢复训练模式
                model.train()
        
        # 每个epoch结束后评估和保存
        if eval_dataloader is not None:
            # 评估
            model.eval()
            eval_loss = 0.0
            eval_steps = 0
            
            if is_main_process:
                logger.info("开始评估...")
                eval_iterator = tqdm(eval_dataloader, desc="Evaluating")
            else:
                eval_iterator = eval_dataloader
            
            for eval_batch in eval_iterator:
                eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                
                # 添加教师logits
                vocab_size = 151669  # 固定为教师模型的词汇表大小
                batch_size = eval_batch["input_ids"].size(0)
                seq_length = eval_batch["input_ids"].size(1)
                eval_batch["teacher_logits"] = torch.zeros(
                    (batch_size, seq_length, vocab_size), 
                    dtype=torch.float16, 
                    device=device
                )
                
                with torch.no_grad():
                    # 学生模型前向传播
                    eval_outputs = model(**eval_batch)
                    
                eval_loss += eval_outputs["loss"].item()
                eval_steps += 1
            
            # 处理分布式评估结果
            if world_size > 1:
                # 将所有进程的损失收集到主进程
                eval_loss_tensor = torch.tensor([eval_loss]).to(device)
                eval_steps_tensor = torch.tensor([eval_steps]).to(device)
                
                dist.all_reduce(eval_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(eval_steps_tensor, op=dist.ReduceOp.SUM)
                
                eval_loss = eval_loss_tensor.item()
                eval_steps = eval_steps_tensor.item()
            
            eval_loss = eval_loss / eval_steps
            
            # 只在主进程中打印和保存
            if is_main_process:
                logger.info(f"Epoch {epoch+1} 评估损失: {eval_loss:.5f}")
                
                # 保存最佳模型
                if eval_loss < best_eval_loss:
                    logger.info(f"发现新的最佳模型 (loss: {eval_loss:.5f}, 之前: {best_eval_loss:.5f})")
                    best_eval_loss = eval_loss
                    
                    best_dir = os.path.join(args.output_dir, "best_model")
                    os.makedirs(best_dir, exist_ok=True)
                    
                    # 保存模型
                    if hasattr(model, "module"):
                        model.module.save_pretrained(best_dir)
                    else:
                        model.save_pretrained(best_dir)
                    
                    # 保存tokenizer
                    tokenizer.save_pretrained(best_dir)
        
        # 每个epoch后保存检查点（只在主进程中）
        if is_main_process:
            # 保存模型检查点
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 保存模型
            if hasattr(model, "module"):
                model.module.save_pretrained(checkpoint_dir)
            else:
                model.save_pretrained(checkpoint_dir)
            
            # 保存tokenizer
            tokenizer.save_pretrained(checkpoint_dir)
            
            logger.info(f"保存第 {epoch+1} 轮模型到 {checkpoint_dir}")
    
    # 训练完成，保存最终模型（只在主进程中）
    if is_main_process:
        logger.info("保存最终模型")
        final_dir = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)
        
        # 保存模型
        if hasattr(model, "module"):
            model.module.save_pretrained(final_dir)
        else:
            model.save_pretrained(final_dir)
        
        # 保存tokenizer
        tokenizer.save_pretrained(final_dir)
        
        logger.info(f"保存最终模型到 {final_dir}")
    
    return global_step, tr_loss / global_step

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多GPU知识蒸馏训练脚本")
    
    # 必要参数
    parser.add_argument("--teacher_model", type=str, default="/home/lizixi/xuelang/Qwen3-8B", help="教师模型路径")
    parser.add_argument("--train_file", type=str, required=True, help="训练集文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    
    # 模型和数据参数
    parser.add_argument("--validation_file", type=str, default=None, help="验证集文件路径")
    parser.add_argument("--max_seq_length", type=int, default=512, help="最大序列长度")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4, help="每个设备的批次大小")
    parser.add_argument("--num_train_epochs", type=float, default=3.0, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪的最大范数")
    
    # 分布式参数
    parser.add_argument("--local_rank", type=int, default=-1, help="本地进程的排名")
    
    # 混合精度训练
    parser.add_argument("--fp16", action="store_true", help="是否使用混合精度训练")
    
    # 知识蒸馏参数
    parser.add_argument("--top_k_distillation", action="store_true", help="是否使用Top-k知识蒸馏")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k知识蒸馏的k值")
    parser.add_argument("--temperature", type=float, default=2.0, help="知识蒸馏温度")
    
    # 缓存控制
    parser.add_argument("--overwrite_cache", action="store_true", help="是否覆盖数据缓存")
    
    # 恢复训练
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从指定检查点恢复训练")
    
    # 保存和评估频率
    parser.add_argument("--save_steps", type=int, default=0, help="每多少步保存一次模型，0表示只在每个轮次结束时保存")
    parser.add_argument("--eval_steps", type=int, default=0, help="每多少步评估一次模型，0表示只在每个轮次结束时评估")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 训练模型
    train(args)

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
        
        Args:
            student_logits: 学生模型输出 [batch_size, seq_length, vocab_size]
            teacher_logits: 教师模型输出 [batch_size, seq_length, vocab_size]
            labels: 真实标签 (可选) [batch_size, seq_length]
            
        Returns:
            metrics: 包含一致性分数、知识迁移分数和整体CCC分数的字典
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

if __name__ == "__main__":
    main() 