import os
import torch
import argparse
import logging
import time
import signal
import sys
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cuda_errors.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CUDA错误检测器")

class CUDAErrorDetector:
    def __init__(self, interval=10, pid=None):
        self.interval = interval
        self.running = True
        self.pid = pid or os.getpid()
        self.error_count = 0
        self.last_check_time = time.time()
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
    
    def handle_signal(self, sig, frame):
        logger.info("接收到终止信号，停止监控...")
        self.running = False
        sys.exit(0)
    
    def check_cuda_errors(self):
        """检查CUDA错误"""
        try:
            # 分配一个小的张量来触发任何挂起的CUDA错误
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    try:
                        test_tensor = torch.zeros(1, device=f"cuda:{i}")
                        del test_tensor
                    except Exception as e:
                        self.error_count += 1
                        logger.error(f"GPU {i} 错误: {str(e)}")
                        return False
                return True
            else:
                logger.warning("CUDA不可用")
                return False
        except Exception as e:
            self.error_count += 1
            logger.error(f"检查CUDA错误时发生异常: {str(e)}")
            return False
    
    def get_gpu_info(self):
        """获取GPU信息"""
        if not torch.cuda.is_available():
            return "CUDA不可用"
            
        info = []
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)  # MB
            memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)    # MB
            
            info.append(f"GPU {i}: {device_name}")
            info.append(f"  已分配内存: {memory_allocated:.2f} MB")
            info.append(f"  预留内存: {memory_reserved:.2f} MB")
            
        return "\n".join(info)
    
    def run(self):
        """运行监控"""
        logger.info("开始CUDA错误监控...")
        logger.info(f"监控进程ID: {self.pid}")
        logger.info(f"检查间隔: {self.interval}秒")
        logger.info(self.get_gpu_info())
        
        while self.running:
            current_time = time.time()
            if current_time - self.last_check_time >= self.interval:
                logger.info(f"检查CUDA状态... 已运行时间: {current_time - self.last_check_time:.2f}秒")
                status_ok = self.check_cuda_errors()
                
                if status_ok:
                    logger.info("CUDA状态正常")
                    logger.info(self.get_gpu_info())
                else:
                    logger.error("检测到CUDA错误!")
                
                self.last_check_time = current_time
            
            time.sleep(1)  # 降低CPU使用率

def monitor_process(args):
    """监控指定进程的CUDA错误"""
    detector = CUDAErrorDetector(interval=args.interval, pid=args.pid)
    detector.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CUDA错误监控工具")
    parser.add_argument("--interval", type=int, default=10, help="检查间隔（秒）")
    parser.add_argument("--pid", type=int, help="要监控的进程ID（默认为当前进程）")
    args = parser.parse_args()
    
    monitor_process(args) 