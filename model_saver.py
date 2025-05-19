import os
import torch
from transformers import Trainer
from transformers.trainer import WEIGHTS_NAME
from typing import Dict, Any, Optional, Union

class CustomTrainer(Trainer):
    """自定义训练器，重写保存逻辑以处理共享权重问题"""
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """重写保存模型方法，使用torch.save而不是safetensors"""
        # 获取输出目录
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 打印保存信息
        print(f"正在保存模型到 {output_dir} (使用torch.save处理共享权重)")
        
        # 保存模型配置
        if hasattr(self.model, "config") and self.model.config is not None:
            self.model.config.save_pretrained(output_dir)
        
        # 获取模型状态字典
        state_dict = self.model.state_dict()
        
        # 使用原生torch.save保存模型权重
        torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        
        # 保存tokenizer (如果有)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
            
        # 保存训练器状态
        if self.args.should_save:
            self._save_trainer_state(output_dir)

# 辅助函数，用于处理共享权重模型的加载
def load_model_with_shared_weights(model, model_path: str):
    """加载带有共享权重的模型"""
    # 加载状态字典
    state_dict = torch.load(os.path.join(model_path, WEIGHTS_NAME), map_location="cpu")
    
    # 加载模型权重
    model.load_state_dict(state_dict)
    
    return model 