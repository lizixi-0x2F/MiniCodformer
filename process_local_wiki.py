#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
本地维基百科中文数据集处理工具
用于处理本地的wikipedia-cn-20230720-filtered.json文件
"""

import os
import argparse
import logging
import json
import random
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def process_local_wiki(input_file, output_dir, max_samples=None, min_length=100):
    """处理本地维基百科JSON数据
    
    Args:
        input_file: 输入JSON文件路径
        output_dir: 输出目录
        max_samples: 最大样本数，None表示全部
        min_length: 最小文本长度，过滤太短的文章
    """
    logger.info(f"开始处理本地维基百科数据: {input_file}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取JSON文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            wiki_data = json.load(f)
        logger.info(f"成功加载数据集，包含 {len(wiki_data)} 条样本")
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        return
    
    # 提取文本
    texts = []
    for item in tqdm(wiki_data, desc="提取文本"):
        if "completion" in item and len(item["completion"]) >= min_length:
            texts.append(item["completion"])
    
    logger.info(f"提取了 {len(texts)} 条有效文本")
    
    # 随机打乱并截取指定数量
    random.shuffle(texts)
    if max_samples is not None and max_samples > 0:
        texts = texts[:max_samples]
        logger.info(f"截取了 {len(texts)} 条样本")
    
    # 写入文本文件
    text_file = os.path.join(output_dir, "wiki_corpus.txt")
    with open(text_file, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text.strip() + "\n\n")
    
    # 写入JSON格式文件
    json_file = os.path.join(output_dir, "wiki_corpus.json")
    with open(json_file, "w", encoding="utf-8") as f:
        for text in texts:
            json_line = json.dumps({"text": text.strip()}, ensure_ascii=False)
            f.write(json_line + "\n")
    
    # 写入小批量样本文件，便于检查数据质量
    sample_file = os.path.join(output_dir, "wiki_samples.txt")
    with open(sample_file, "w", encoding="utf-8") as f:
        for text in texts[:10]:
            f.write(f"{'='*40}\n{text.strip()}\n{'='*40}\n\n")
    
    logger.info(f"处理完成！输出文件:")
    logger.info(f" - 文本文件: {text_file}")
    logger.info(f" - JSON文件: {json_file}")
    logger.info(f" - 样本文件: {sample_file}")
    
    # 返回处理的统计信息
    return {
        "total_samples": len(texts),
        "output_files": [text_file, json_file, sample_file]
    }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="本地维基百科中文数据集处理工具")
    parser.add_argument("--input_file", default="data/wikipedia-cn-20230720-filtered.json", help="输入JSON文件路径")
    parser.add_argument("--output_dir", default="data/corpus/wikipedia", help="输出目录")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数")
    parser.add_argument("--min_length", type=int, default=100, help="最小文本长度")
    args = parser.parse_args()
    
    process_local_wiki(
        input_file=args.input_file,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        min_length=args.min_length
    )

if __name__ == "__main__":
    main() 