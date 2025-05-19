#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import argparse
import time
import json
from transformers import AutoTokenizer
from model import DistillationModel

def parse_args():
    parser = argparse.ArgumentParser(description="MiniCodformer知识蒸馏推理脚本 (CCC+KD版)")
    parser.add_argument("--model_path", type=str, default="./output/distill/final_model", help="模型路径")
    parser.add_argument("--input", type=str, default=None, help="输入文本")
    parser.add_argument("--file", type=str, default=None, help="输入文件路径")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--max_length", type=int, default=100, help="最大生成长度")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda", help="运行设备")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--time_stats", action="store_true", help="显示生成时间统计")
    return parser.parse_args()

def setup_device(device_str):
    """设置并验证设备"""
    if device_str == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA不可用，回退到CPU")
        return torch.device("cpu")
    return torch.device(device_str)

def load_model(model_path, device, debug=False):
    """加载模型并返回模型和tokenizer"""
    try:
        print(f"从 {model_path} 加载模型...")
        
        # 检查模型文件
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
            
        model_files = os.listdir(model_path)
        required_files = ["pytorch_model.bin", "config.json"]
        for file in required_files:
            if file not in model_files:
                raise FileNotFoundError(f"模型目录中缺少必要文件: {file}")
        
        # 加载模型
        model = DistillationModel.from_pretrained(model_path, device=device)
        model.eval()
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if debug:
            print(f"模型词汇表大小: {model.config.vocab_size}")
            print(f"Tokenizer词汇表大小: {len(tokenizer)}")
            print(f"模型配置: {model.config}")
        
        print("模型加载成功!")
        return model, tokenizer
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)

def get_input_prompts(args):
    """从命令行参数获取输入提示"""
    if args.input:
        return [args.input]
        
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"读取输入文件失败: {e}")
            sys.exit(1)
    
    # 从标准输入读取
    print("请输入提示文本 (一次输入一行，输入空行或 Ctrl+D 结束):")
    prompts = []
    try:
        while True:
            line = input("> ")
            if not line:
                if prompts:  # 如果已有输入，空行表示结束
                    break
                continue     # 如果还没有输入，继续等待
            prompts.append(line)
    except (KeyboardInterrupt, EOFError):
        if not prompts:
            print("\n未提供输入，退出程序")
            sys.exit(0)
    
    return prompts

def clean_generated_text(text):
    """清理生成的文本，移除乱码和重复内容"""
    # 简单处理输出，移除重复行和乱码
    lines = text.split("\n")
    clean_lines = []
    seen_lines = set()
    
    for line in lines:
        # 去除含有大量非中文/英文/标点的行（可能是乱码）
        non_standard_chars = sum(1 for c in line if not (
            '\u4e00' <= c <= '\u9fff' or  # 中文
            'a' <= c.lower() <= 'z' or    # 英文
            '0' <= c <= '9' or            # 数字
            c in '.,;:!?()[]{}"\'-+*/=<>%$#@&|\\~`^_ \t\n'  # 常见标点和空白
        ))
        
        # 如果非标准字符过多，跳过该行
        if non_standard_chars > len(line) * 0.3 and len(line) > 5:
            continue
            
        # 去重
        if line not in seen_lines:
            seen_lines.add(line)
            clean_lines.append(line)
    
    # 重新组合文本
    return "\n".join(clean_lines)

def generate_text(model, tokenizer, prompt, args, debug=False):
    """使用模型生成文本"""
    try:
        start_time = time.time()
        
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        
        # 设置生成参数
        generation_config = {
            "max_length": len(input_ids[0]) + args.max_length,
            "repetition_penalty": args.repetition_penalty,
        }
        
        # 可选参数
        if args.temperature != 1.0:
            generation_config["temperature"] = args.temperature
        if args.top_k > 0:
            generation_config["top_k"] = args.top_k
        if args.top_p < 1.0:
            generation_config["top_p"] = args.top_p
        
        if debug:
            print(f"输入token数: {len(input_ids[0])}")
            print(f"生成参数: {generation_config}")
        
        # 生成文本
        encoding_time = time.time() - start_time
        generation_start = time.time()
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids, 
                attention_mask=attention_mask,
                **generation_config
            )
        
        generation_time = time.time() - generation_start
        
        # 解码输出
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # 清理生成的文本
        cleaned_text = clean_generated_text(generated_text)
        
        total_time = time.time() - start_time
        
        # 时间统计
        if args.time_stats:
            tokens_generated = len(output_ids[0]) - len(input_ids[0])
            print(f"编码时间: {encoding_time:.3f}秒")
            print(f"生成时间: {generation_time:.3f}秒")
            print(f"总时间: {total_time:.3f}秒")
            print(f"生成速度: {tokens_generated / generation_time:.2f} tokens/秒")
        
        return cleaned_text
        
    except Exception as e:
        print(f"生成文本时出错: {e}")
        return f"[生成失败: {e}]"

def main():
    """
    推理主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="MiniCodformer知识蒸馏推理脚本 (CCC+KD版)")
    parser.add_argument("--model_path", type=str, default="./output/distill/final_model", help="模型路径")
    parser.add_argument("--input", type=str, default=None, help="输入文本")
    parser.add_argument("--file", type=str, default=None, help="输入文件路径")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--max_length", type=int, default=100, help="最大生成长度")
    args = parse_args()
    
    # 设置设备
    device = setup_device(args.device)
    args.device = device
    print(f"使用设备: {device}")
    
    # 启用调试输出
    if args.debug:
        print("调试模式已启用")
    
    # 加载模型和tokenizer
    model, tokenizer = load_model(args.model_path, device, args.debug)
    
    # 获取输入提示
    prompts = get_input_prompts(args)
    
    # 准备输出
    outputs = []
    
    # 逐个处理输入
    for i, prompt in enumerate(prompts):
        if not prompt:
            continue
            
        print(f"\n======= 输入 {i+1}/{len(prompts)} =======")
        print(f"提示: {prompt}")
        
        # 生成文本
        generated_text = generate_text(model, tokenizer, prompt, args, args.debug)
        
        # 输出结果
        print(f"\n生成结果:")
        print("-" * 50)
        print(generated_text)
        print("-" * 50)
        
        # 保存结果
        outputs.append({
            "prompt": prompt,
            "response": generated_text
        })
    
    # 如果指定了输出文件，保存结果
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                for item in outputs:
                    f.write(f"提示: {item['prompt']}\n")
                    f.write(f"回复: {item['response']}\n")
                    f.write("-" * 50 + "\n")
            print(f"\n结果已保存到: {args.output}")
        except Exception as e:
            print(f"保存结果到文件失败: {e}")
    
    print("\n推理完成!")

if __name__ == "__main__":
    main() 