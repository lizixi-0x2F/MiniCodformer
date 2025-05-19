#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练日志分析脚本 - 用于检测训练崩溃原因
- 分析内存泄漏模式
- 检测异常的梯度或损失值
- 识别学习率问题
- 发现数据批次异常
"""

import os
import re
import sys
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 正则表达式模式
LOSS_PATTERN = r"步骤\s+(\d+):\s+总损失=(\d+\.\d+),\s+学生损失=(\d+\.\d+),\s+KD损失=(\d+\.\d+)"
MEMORY_PATTERN = r"GPU\s+(\d+)\s+已分配内存:\s+(\d+\.\d+)\s+MB"
ERROR_PATTERN = r"(错误|Error|error|OOM|Out of memory|CUDA)"
CUDA_ERROR_PATTERN = r"CUDA\s+error:(.+)$"
GRAD_PATTERN = r"梯度范数:\s+(\d+\.\d+)"
LR_PATTERN = r"学习率:\s+(\d+\.\d+)"

def load_log_file(file_path):
    """加载并解析日志文件"""
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.readlines()
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return None

def parse_timestamps(lines):
    """从日志中解析时间戳"""
    timestamps = []
    last_timestamp = None
    
    # 尝试不同的时间戳格式
    timestamp_patterns = [
        r"(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})",  # MM/DD/YYYY HH:MM:SS
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})",  # YYYY-MM-DD HH:MM:SS
    ]
    
    for i, line in enumerate(lines):
        for pattern in timestamp_patterns:
            match = re.search(pattern, line)
            if match:
                timestamp_str = match.group(1)
                try:
                    # 尝试解析时间戳，支持多种格式
                    if '/' in timestamp_str:
                        timestamp = datetime.datetime.strptime(timestamp_str, "%m/%d/%Y %H:%M:%S")
                    else:
                        timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    
                    timestamps.append((i, timestamp))
                    last_timestamp = timestamp
                    break
                except ValueError:
                    continue
        
        # 如果这一行没有时间戳但我们有之前的时间戳，使用前一个
        if last_timestamp is not None and i > 0 and i not in [t[0] for t in timestamps]:
            timestamps.append((i, last_timestamp))
    
    return timestamps

def analyze_training_progression(lines):
    """分析训练进度和损失值趋势"""
    steps = []
    total_losses = []
    student_losses = []
    kd_losses = []
    
    for line in lines:
        match = re.search(LOSS_PATTERN, line)
        if match:
            step, total_loss, student_loss, kd_loss = match.groups()
            steps.append(int(step))
            total_losses.append(float(total_loss))
            student_losses.append(float(student_loss))
            kd_losses.append(float(kd_loss))
    
    return {
        "steps": steps,
        "total_losses": total_losses,
        "student_losses": student_losses,
        "kd_losses": kd_losses
    }

def analyze_memory_usage(lines):
    """分析GPU内存使用情况"""
    memory_usage = defaultdict(list)
    line_numbers = defaultdict(list)
    
    for i, line in enumerate(lines):
        matches = re.finditer(MEMORY_PATTERN, line)
        for match in matches:
            gpu_id, memory = match.groups()
            memory_usage[int(gpu_id)].append(float(memory))
            line_numbers[int(gpu_id)].append(i)
    
    return memory_usage, line_numbers

def detect_errors(lines):
    """检测日志中的错误"""
    errors = []
    
    for i, line in enumerate(lines):
        if re.search(ERROR_PATTERN, line, re.IGNORECASE):
            # 提取上下文
            context_start = max(0, i - 5)
            context_end = min(len(lines), i + 5)
            context = lines[context_start:context_end]
            
            errors.append({
                "line_number": i,
                "error_text": line.strip(),
                "context": context
            })
    
    return errors

def detect_gradient_anomalies(lines):
    """检测梯度异常"""
    steps = []
    grad_norms = []
    
    for line in lines:
        match = re.search(GRAD_PATTERN, line)
        if match:
            # 尝试从行中提取步骤
            step_match = re.search(r"步骤\s+(\d+)", line)
            step = int(step_match.group(1)) if step_match else len(grad_norms)
            
            norm = float(match.group(1))
            steps.append(step)
            grad_norms.append(norm)
    
    # 检测异常
    anomalies = []
    if grad_norms:
        mean_norm = np.mean(grad_norms)
        std_norm = np.std(grad_norms)
        
        for i, norm in enumerate(grad_norms):
            # 检测异常大或异常小的梯度
            if norm > mean_norm + 3 * std_norm or norm < mean_norm - 3 * std_norm:
                anomalies.append({
                    "step": steps[i],
                    "norm": norm,
                    "z_score": (norm - mean_norm) / (std_norm if std_norm > 0 else 1)
                })
    
    return {
        "steps": steps,
        "grad_norms": grad_norms,
        "anomalies": anomalies
    }

def analyze_learning_rate(lines):
    """分析学习率变化"""
    steps = []
    learning_rates = []
    
    for line in lines:
        match = re.search(LR_PATTERN, line)
        if match:
            # 尝试从行中提取步骤
            step_match = re.search(r"步骤\s+(\d+)", line)
            step = int(step_match.group(1)) if step_match else len(learning_rates)
            
            lr = float(match.group(1))
            steps.append(step)
            learning_rates.append(lr)
    
    return {
        "steps": steps,
        "learning_rates": learning_rates
    }

def analyze_time_between_steps(lines, timestamps):
    """分析步骤之间的时间间隔"""
    step_times = {}
    step_timestamps = {}
    
    # 提取带有步骤信息的行及其时间戳
    for i, line in enumerate(lines):
        step_match = re.search(r"步骤\s+(\d+)", line)
        if step_match:
            step = int(step_match.group(1))
            
            # 查找这一行的时间戳
            for line_num, timestamp in timestamps:
                if line_num == i:
                    step_timestamps[step] = timestamp
                    break
    
    # 计算步骤间的时间间隔
    step_numbers = sorted(step_timestamps.keys())
    for i in range(1, len(step_numbers)):
        prev_step = step_numbers[i-1]
        curr_step = step_numbers[i]
        
        if prev_step in step_timestamps and curr_step in step_timestamps:
            time_diff = (step_timestamps[curr_step] - step_timestamps[prev_step]).total_seconds()
            step_times[curr_step] = time_diff
    
    return step_times

def detect_stalling(step_times):
    """检测训练是否出现停滞"""
    if not step_times:
        return None
    
    steps = sorted(step_times.keys())
    times = [step_times[s] for s in steps]
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    # 检测异常长的步骤时间
    stall_points = []
    for step, time in step_times.items():
        if time > mean_time + 3 * std_time:
            stall_points.append({
                "step": step,
                "time": time,
                "expected_time": mean_time,
                "times_slower": time / mean_time
            })
    
    return {
        "steps": steps,
        "times": times,
        "mean_time": mean_time,
        "stall_points": stall_points
    }

def find_training_crash(lines, timestamps):
    """尝试定位训练崩溃的位置"""
    # 检查是否有明确的错误信息
    errors = detect_errors(lines)
    if errors:
        return {
            "crash_detected": True,
            "crash_type": "explicit_error",
            "error_details": errors[0]["error_text"],
            "line_number": errors[0]["line_number"],
            "context": errors[0]["context"]
        }
    
    # 检查训练是否突然停止（没有正常的结束消息）
    training_progression = analyze_training_progression(lines)
    steps = training_progression["steps"]
    
    if not steps:
        return {
            "crash_detected": False,
            "crash_type": "unknown",
            "message": "未找到训练步骤信息"
        }
    
    # 根据时间戳判断训练是否正常结束
    last_timestamp = None
    for _, timestamp in reversed(timestamps):
        if timestamp:
            last_timestamp = timestamp
            break
    
    if last_timestamp:
        now = datetime.datetime.now()
        time_diff = (now - last_timestamp).total_seconds() / 3600  # 小时
        
        # 如果最后一条日志时间在6小时以内，并且没有正常结束的消息，可能是崩溃了
        last_few_lines = [line.strip() for line in lines[-10:]]
        normal_end_patterns = ["训练完成", "Training finished", "completed", "saved model"]
        normal_end = any(re.search(pattern, ' '.join(last_few_lines), re.IGNORECASE) for pattern in normal_end_patterns)
        
        if time_diff < 6 and not normal_end:
            # 尝试找出最后一个训练步骤
            last_step = steps[-1] if steps else None
            return {
                "crash_detected": True,
                "crash_type": "abrupt_termination",
                "last_step": last_step,
                "last_timestamp": last_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "hours_ago": time_diff
            }
    
    return {
        "crash_detected": False,
        "crash_type": "unknown",
        "message": "未检测到明确的崩溃"
    }

def generate_plots(training_data, memory_data, grad_data, lr_data, step_times, output_dir):
    """生成分析图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 损失曲线
    if training_data["steps"]:
        plt.figure(figsize=(10, 6))
        plt.plot(training_data["steps"], training_data["total_losses"], label='总损失')
        plt.plot(training_data["steps"], training_data["student_losses"], label='学生损失')
        plt.plot(training_data["steps"], training_data["kd_losses"], label='KD损失')
        plt.xlabel('训练步骤')
        plt.ylabel('损失值')
        plt.title('训练损失曲线')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
        plt.close()
    
    # 2. 内存使用曲线
    for gpu_id, memory_values in memory_data.items():
        if memory_values:
            x = range(len(memory_values))
            plt.figure(figsize=(10, 6))
            plt.plot(x, memory_values)
            plt.xlabel('日志条目')
            plt.ylabel('内存使用 (MB)')
            plt.title(f'GPU {gpu_id} 内存使用')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'memory_gpu{gpu_id}.png'))
            plt.close()
    
    # 3. 梯度范数曲线
    if grad_data["steps"]:
        plt.figure(figsize=(10, 6))
        plt.plot(grad_data["steps"], grad_data["grad_norms"])
        
        # 标记异常点
        for anomaly in grad_data["anomalies"]:
            plt.scatter(anomaly["step"], anomaly["norm"], color='red', s=100, marker='x')
        
        plt.xlabel('训练步骤')
        plt.ylabel('梯度范数')
        plt.title('梯度范数变化')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'gradient_norm.png'))
        plt.close()
    
    # 4. 学习率曲线
    if lr_data["steps"]:
        plt.figure(figsize=(10, 6))
        plt.plot(lr_data["steps"], lr_data["learning_rates"])
        plt.xlabel('训练步骤')
        plt.ylabel('学习率')
        plt.title('学习率变化')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'learning_rate.png'))
        plt.close()
    
    # 5. 步骤耗时曲线
    if step_times:
        steps = sorted(step_times.keys())
        times = [step_times[s] for s in steps]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, times)
        plt.xlabel('训练步骤')
        plt.ylabel('时间 (秒)')
        plt.title('每步训练耗时')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'step_times.png'))
        plt.close()

def print_summary(training_data, memory_data, errors, grad_data, crash_info, step_times_analysis):
    """打印分析摘要"""
    print("\n===== 训练日志分析摘要 =====\n")
    
    # 1. 崩溃信息
    print("崩溃分析:")
    if crash_info["crash_detected"]:
        print(f"  检测到训练崩溃! 类型: {crash_info['crash_type']}")
        if "error_details" in crash_info:
            print(f"  错误详情: {crash_info['error_details']}")
        if "last_step" in crash_info:
            print(f"  最后一个步骤: {crash_info['last_step']}")
        if "last_timestamp" in crash_info:
            print(f"  最后时间戳: {crash_info['last_timestamp']} (约 {crash_info['hours_ago']:.2f} 小时前)")
    else:
        print(f"  未检测到明确的崩溃 - {crash_info['message']}")
    
    # 2. 训练进度
    if training_data["steps"]:
        print("\n训练进度:")
        print(f"  总训练步骤: {len(training_data['steps'])}")
        print(f"  最后记录的步骤: {training_data['steps'][-1]}")
        if len(training_data["total_losses"]) > 1:
            first_loss = training_data["total_losses"][0]
            last_loss = training_data["total_losses"][-1]
            print(f"  初始损失: {first_loss:.4f}, 最终损失: {last_loss:.4f}, 减少: {(first_loss - last_loss) / first_loss * 100:.2f}%")
    
    # 3. 内存使用情况
    print("\n内存使用情况:")
    for gpu_id, memory_values in memory_data.items():
        if memory_values:
            print(f"  GPU {gpu_id}:")
            print(f"    最小内存: {min(memory_values):.2f} MB")
            print(f"    最大内存: {max(memory_values):.2f} MB")
            print(f"    平均内存: {np.mean(memory_values):.2f} MB")
            # 计算内存增长率
            if len(memory_values) > 2:
                first_quarter = np.mean(memory_values[:len(memory_values)//4])
                last_quarter = np.mean(memory_values[-len(memory_values)//4:])
                growth = (last_quarter - first_quarter) / first_quarter * 100
                print(f"    内存增长率: {growth:.2f}% (从第一个四分位到最后一个四分位)")
                if growth > 20:
                    print("    警告: 检测到显著的内存增长，可能存在内存泄漏")
    
    # 4. 错误信息
    if errors:
        print("\n检测到的错误:")
        for i, error in enumerate(errors[:5]):  # 仅显示前5个错误
            print(f"  错误 {i+1} (行 {error['line_number']}):")
            print(f"    {error['error_text']}")
        if len(errors) > 5:
            print(f"    ... 还有 {len(errors) - 5} 个错误未显示")
    
    # 5. 梯度异常
    if grad_data["anomalies"]:
        print("\n梯度异常:")
        for anomaly in grad_data["anomalies"][:5]:
            print(f"  步骤 {anomaly['step']}: 梯度范数 = {anomaly['norm']:.4f} (Z分数: {anomaly['z_score']:.2f})")
        if len(grad_data["anomalies"]) > 5:
            print(f"    ... 还有 {len(grad_data['anomalies']) - 5} 个异常未显示")
    
    # 6. 步骤耗时分析
    if step_times_analysis and "stall_points" in step_times_analysis:
        print("\n训练停滞分析:")
        print(f"  平均步骤耗时: {step_times_analysis['mean_time']:.2f} 秒")
        
        if step_times_analysis["stall_points"]:
            print("  检测到的停滞点:")
            for stall in step_times_analysis["stall_points"][:5]:
                print(f"    步骤 {stall['step']}: {stall['time']:.2f} 秒 (正常应为 {stall['expected_time']:.2f} 秒, 慢了 {stall['times_slower']:.2f} 倍)")
            if len(step_times_analysis["stall_points"]) > 5:
                print(f"    ... 还有 {len(step_times_analysis['stall_points']) - 5} 个停滞点未显示")
    
    # 7. 可能的崩溃原因和建议
    print("\n可能的崩溃原因和建议:")
    suggestions = []
    
    # 基于内存
    for gpu_id, memory_values in memory_data.items():
        if memory_values and max(memory_values) > 24000:  # 假设24GB GPU
            suggestions.append(f"- GPU {gpu_id} 内存接近极限 ({max(memory_values):.2f} MB)。考虑减小批量大小或使用梯度累积。")
    
    # 基于错误分析
    cuda_oom = any("out of memory" in e["error_text"].lower() for e in errors)
    cuda_error = any("cuda error" in e["error_text"].lower() for e in errors)
    
    if cuda_oom:
        suggestions.append("- 检测到CUDA内存不足错误。尝试减小批量大小、模型大小或启用梯度检查点技术。")
    
    if cuda_error and not cuda_oom:
        suggestions.append("- 检测到CUDA错误但不是内存不足。可能是硬件问题、驱动版本不兼容或代码中的CUDA操作错误。")
    
    # 基于梯度异常
    if grad_data["anomalies"]:
        if any(a["norm"] > 100 for a in grad_data["anomalies"]):
            suggestions.append("- 检测到非常大的梯度范数。考虑使用梯度裁剪或降低学习率。")
        elif any(a["norm"] < 1e-6 for a in grad_data["anomalies"]):
            suggestions.append("- 检测到异常小的梯度范数。可能遇到了梯度消失问题。")
    
    # 基于训练停滞
    if step_times_analysis and "stall_points" in step_times_analysis and step_times_analysis["stall_points"]:
        suggestions.append("- 训练过程中存在明显的停滞点。可能是由于数据加载瓶颈、系统GC或其他系统进程干扰。")
    
    # 其他常见建议
    if not suggestions:
        suggestions.extend([
            "- 检查训练数据是否有异常样本或极端情况",
            "- 确保批量标准化层在评估模式下正确运行",
            "- 考虑使用混合精度训练降低内存使用",
            "- 检查是否有张量被意外地保留在计算图中（内存泄漏）",
            "- 监控系统温度，确保GPU没有过热导致的故障"
        ])
    
    for suggestion in suggestions:
        print(suggestion)
    
    print("\n提示: 查看生成的图表以获取更详细的分析")

def main():
    parser = argparse.ArgumentParser(description="分析训练日志，检测崩溃原因")
    parser.add_argument("log_file", help="训练日志文件路径")
    parser.add_argument("--output", default="log_analysis", help="输出目录，用于保存图表")
    args = parser.parse_args()
    
    # 加载日志文件
    lines = load_log_file(args.log_file)
    if not lines:
        print(f"无法分析日志文件: {args.log_file}")
        return 1
    
    print(f"正在分析日志文件: {args.log_file} ({len(lines)} 行)")
    
    # 解析时间戳
    timestamps = parse_timestamps(lines)
    print(f"找到 {len(timestamps)} 个时间戳")
    
    # 分析训练进度
    training_data = analyze_training_progression(lines)
    print(f"找到 {len(training_data['steps'])} 个训练步骤")
    
    # 分析内存使用
    memory_data, line_numbers = analyze_memory_usage(lines)
    print(f"找到 {sum(len(v) for v in memory_data.values())} 个内存使用记录，涉及 {len(memory_data)} 个GPU")
    
    # 检测错误
    errors = detect_errors(lines)
    print(f"找到 {len(errors)} 个潜在错误")
    
    # 分析梯度
    grad_data = detect_gradient_anomalies(lines)
    print(f"找到 {len(grad_data['grad_norms'])} 个梯度记录，其中 {len(grad_data['anomalies'])} 个异常")
    
    # 分析学习率
    lr_data = analyze_learning_rate(lines)
    print(f"找到 {len(lr_data['learning_rates'])} 个学习率记录")
    
    # 分析步骤间时间
    step_times = analyze_time_between_steps(lines, timestamps)
    print(f"分析了 {len(step_times)} 个步骤间的时间间隔")
    
    # 检测训练停滞
    step_times_analysis = detect_stalling(step_times)
    if step_times_analysis:
        print(f"平均步骤时间: {step_times_analysis['mean_time']:.2f} 秒")
        print(f"找到 {len(step_times_analysis['stall_points'])} 个训练停滞点")
    
    # 寻找训练崩溃
    crash_info = find_training_crash(lines, timestamps)
    
    # 生成图表
    generate_plots(training_data, memory_data, grad_data, lr_data, step_times, args.output)
    print(f"图表已生成到目录: {args.output}")
    
    # 打印摘要
    print_summary(training_data, memory_data, errors, grad_data, crash_info, step_times_analysis)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 