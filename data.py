import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os
import json
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import pickle
from tqdm import tqdm

logger = logging.getLogger(__name__)

class CDIALGPTDataset(Dataset):
    """用于CDial-GPT的数据集类"""
    
    def __init__(
        self,
        tokenizer,
        file_path: str,
        block_size: int,
        overwrite_cache=False,
        cache_dir: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        
        # 设置缓存路径
        self.cache_dir = cache_dir if cache_dir is not None else os.path.dirname(file_path)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 创建缓存文件名，基于输入文件和tokenizer
        cache_file_name = f"{os.path.basename(file_path)}_{block_size}_{tokenizer.__class__.__name__}.cache"
        self.cache_file = os.path.join(self.cache_dir, cache_file_name)
        
        # 如果缓存存在且不需要覆盖，则直接加载
        if os.path.exists(self.cache_file) and not overwrite_cache:
            logger.info(f"从缓存加载数据: {self.cache_file}")
            try:
                with open(self.cache_file, 'rb') as f:
                    self.examples = pickle.load(f)
                logger.info(f"成功从缓存加载了 {len(self.examples)} 条数据")
                return
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}，将重新处理数据")
        
        # 没有缓存或需要覆盖缓存，则处理数据
        logger.info(f"处理数据并创建缓存: {file_path}")
        self.examples = self.load_and_process_data(file_path, block_size)
        
        # 保存处理后的数据到缓存
        logger.info(f"保存数据到缓存: {self.cache_file}")
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.examples, f)
            logger.info(f"成功将 {len(self.examples)} 条数据保存到缓存")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
        
    def load_and_process_data(self, file_path, block_size):
        """加载并处理CDial-GPT数据"""
        logger.info(f"从文件加载数据: {file_path}")
        
        # 存储处理后的示例
        examples = []
        
        # 读取JSON文件
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 使用tqdm显示进度
            for conversation in tqdm(data, desc="处理对话数据"):
                # CDial-GPT中每个样本是一个对话列表
                if isinstance(conversation, list):
                    # 将对话转换为单个字符串
                    dialog_text = ""
                    for i, utterance in enumerate(conversation):
                        if i % 2 == 0:
                            # 用户回复
                            dialog_text += f"用户: {utterance} "
                        else:
                            # 系统回复
                            dialog_text += f"系统: {utterance} "
                    
                    # 对文本进行编码 - 使用批处理以加快速度
                    try:
                        encodings = self.tokenizer(dialog_text, max_length=block_size, truncation=True)
                        input_ids = encodings["input_ids"]
                        
                        # 创建示例
                        example = {
                            "input_ids": input_ids,
                            "labels": input_ids.copy(),  # 用于自回归训练
                        }
                        
                        examples.append(example)
                    except Exception as e:
                        logger.warning(f"处理对话时出错，跳过该样本: {e}")
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            
        logger.info(f"成功加载 {len(examples)} 个样本")
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.examples[idx]["input_ids"], dtype=torch.long),
            "labels": torch.tensor(self.examples[idx]["labels"], dtype=torch.long),
        }

def collate_fn(batch):
    """自定义collate函数，处理不同长度的序列"""
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # 填充序列使它们具有相同的长度
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # 使用-100作为忽略的标签
    
    # 创建注意力掩码
    attention_mask = torch.ones_like(input_ids).float()
    attention_mask[input_ids == 0] = 0.0
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask
    }

def get_dataset(tokenizer, file_path, block_size, cache_dir=None, overwrite_cache=False):
    """获取数据集"""
    return CDIALGPTDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
        overwrite_cache=overwrite_cache,
        cache_dir=cache_dir,
    )

def prepare_teacher_student_batch(
    teacher_model,
    batch,
    tokenizer,
    device
):
    """准备教师和学生模型的批次数据"""
    try:
        logger.debug(f"准备批次数据: 批次大小={len(batch['input_ids'])}")
        
        # 将batch放到设备上
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 对教师模型进行前向传播（不计算梯度）
        logger.debug(f"正在获取教师模型logits...")
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                return_dict=True
            )
            
            teacher_logits = teacher_outputs.logits
            logger.debug(f"教师模型logits shape: {teacher_logits.shape}")
        
        # 返回增强后的批次
        return {
            **batch,
            "teacher_logits": teacher_logits,
        }
    except Exception as e:
        logger.error(f"准备批次数据时出错: {e}")
        raise 