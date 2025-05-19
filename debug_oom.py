import gc
import torch
import time
import os
import psutil
import argparse
from pathlib import Path

def log_tensor_info(filename):
    """记录当前所有张量的信息"""
    with open(filename, 'a') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 系统内存信息
        process = psutil.Process(os.getpid())
        f.write(f"进程内存使用: {process.memory_info().rss / (1024 * 1024):.2f} MB\n")
        
        # CUDA内存信息
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                f.write(f"GPU {i} 已分配内存: {torch.cuda.memory_allocated(i) / (1024 * 1024):.2f} MB\n")
                f.write(f"GPU {i} 缓存内存: {torch.cuda.memory_reserved(i) / (1024 * 1024):.2f} MB\n")
        
        # 检查所有张量
        total_size = 0
        tensor_count = 0
        largest_tensors = []
        
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and not obj.is_cuda:
                    tensor_count += 1
                    size = obj.numel() * obj.element_size()
                    total_size += size
                    largest_tensors.append((size, obj.shape, obj.dtype))
                elif hasattr(obj, 'data') and torch.is_tensor(obj.data) and not obj.data.is_cuda:
                    tensor_count += 1
                    size = obj.data.numel() * obj.data.element_size()
                    total_size += size
                    largest_tensors.append((size, obj.data.shape, obj.data.dtype))
            except:
                pass
                
        f.write(f"CPU张量数量: {tensor_count}\n")
        f.write(f"CPU张量总内存: {total_size / (1024 * 1024):.2f} MB\n")
        
        # CUDA张量
        cuda_tensor_count = 0
        cuda_total_size = 0
        cuda_largest_tensors = []
        
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    cuda_tensor_count += 1
                    size = obj.numel() * obj.element_size()
                    cuda_total_size += size
                    cuda_largest_tensors.append((size, obj.shape, obj.dtype, obj.device))
                elif hasattr(obj, 'data') and torch.is_tensor(obj.data) and obj.data.is_cuda:
                    cuda_tensor_count += 1
                    size = obj.data.numel() * obj.data.element_size()
                    cuda_total_size += size
                    cuda_largest_tensors.append((size, obj.data.shape, obj.data.dtype, obj.data.device))
            except:
                pass
                
        f.write(f"CUDA张量数量: {cuda_tensor_count}\n")
        f.write(f"CUDA张量总内存: {cuda_total_size / (1024 * 1024):.2f} MB\n")
        
        # 显示最大的张量
        largest_tensors.sort(reverse=True)
        f.write("\n最大的CPU张量:\n")
        for i, (size, shape, dtype) in enumerate(largest_tensors[:10]):
            f.write(f"{i+1}. 大小: {size / (1024 * 1024):.2f} MB, 形状: {shape}, 类型: {dtype}\n")
            
        cuda_largest_tensors.sort(reverse=True)
        f.write("\n最大的CUDA张量:\n")
        for i, (size, shape, dtype, device) in enumerate(cuda_largest_tensors[:10]):
            f.write(f"{i+1}. 大小: {size / (1024 * 1024):.2f} MB, 形状: {shape}, 类型: {dtype}, 设备: {device}\n")

def monitor_script(args):
    """在训练过程中插入使用的监控脚本"""
    log_file = Path(args.log_file)
    
    # 创建日志文件
    log_file.parent.mkdir(exist_ok=True)
    
    with open(log_file, 'w') as f:
        f.write(f"PyTorch内存监控开始于: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"PyTorch版本: {torch.__version__}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA版本: {torch.version.cuda}\n")
            f.write(f"GPU数量: {torch.cuda.device_count()}\n")
            for i in range(torch.cuda.device_count()):
                f.write(f"GPU {i}: {torch.cuda.get_device_name(i)}\n")
    
    # 定期记录内存使用情况
    try:
        while True:
            log_tensor_info(log_file)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"监控已停止，日志保存在: {log_file}")

def add_memory_hooks(model, log_file):
    """为模型的每一层添加前向和后向钩子，以监控内存变化"""
    hooks = []
    
    def forward_hook(module, input, output):
        with open(log_file, 'a') as f:
            f.write(f"\n前向传播 - {module.__class__.__name__}\n")
            # 记录CUDA内存使用情况
            for i in range(torch.cuda.device_count()):
                f.write(f"GPU {i} 内存: {torch.cuda.memory_allocated(i) / (1024 * 1024):.2f} MB\n")
    
    def backward_hook(module, grad_input, grad_output):
        with open(log_file, 'a') as f:
            f.write(f"\n反向传播 - {module.__class__.__name__}\n")
            # 记录CUDA内存使用情况
            for i in range(torch.cuda.device_count()):
                f.write(f"GPU {i} 内存: {torch.cuda.memory_allocated(i) / (1024 * 1024):.2f} MB\n")
    
    # 为模型的每一层添加钩子
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(forward_hook))
        hooks.append(module.register_backward_hook(backward_hook))
    
    return hooks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch内存监控工具")
    parser.add_argument("--log-file", type=str, default="pytorch_memory.log", help="日志文件路径")
    parser.add_argument("--interval", type=int, default=60, help="监控间隔（秒）")
    args = parser.parse_args()
    
    monitor_script(args) 