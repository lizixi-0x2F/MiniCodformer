from dataclasses import dataclass, field
from typing import Optional, List, Union

@dataclass
class ModelConfig:
    """蒸馏模型配置"""
    
    # 基础配置
    vocab_size: int = 151669  # 词汇表大小，匹配Qwen3-8B教师模型的词汇表大小
    hidden_size: int = 256  # 编码器隐藏维度
    num_hidden_layers: int = 3  # 编码器层数
    num_attention_heads: int = 4  # 注意力头数
    intermediate_size: int = 1024  # 前馈网络中间层大小
    hidden_dropout_prob: float = 0.1  # Dropout概率
    attention_probs_dropout_prob: float = 0.1  # 注意力Dropout概率
    max_position_embeddings: int = 2048  # 最大位置编码长度
    
    # LTC-NCP解码器配置
    ltc_ncp_hidden_size: int = 256  # 解码器隐藏维度
    ltc_ncp_num_layers: int = 3  # 解码器层数
    ltc_kernel_size: int = 3  # 卷积核大小，必须为奇数
    
    # 优化器和学习率配置
    optimizer_type: str = "adamw"  # 优化器类型: adamw, adam, sgd
    learning_rate: float = 5e-4  # 基础学习率
    weight_decay: float = 0.01  # 权重衰减系数
    lr_scheduler_type: str = "cosine"  # 学习率调度器: cosine, linear, polynomial
    warmup_steps: int = 1000  # 预热步数
    warmup_ratio: float = 0.1  # 预热比例(相对于总步数)
    layerwise_lr_decay: Optional[float] = 0.8  # 分层学习率衰减率(从高层到低层)
    
    # 蒸馏配置
    distill_supervision: bool = True  # 是否使用辅助监督transformer
    share_embeddings: bool = True  # 是否共享嵌入层权重
    distill_logits: bool = True  # 是否蒸馏logits
    distill_hiddens: bool = True  # 是否蒸馏中间层隐藏状态
    temperature: float = 2.0  # KL散度温度参数
    distill_top_k: int = 20  # Top-K蒸馏，0表示全部词表，>0则只蒸馏教师top-k预测
    ignore_padding: bool = True  # 是否在损失计算中忽略padding
    pad_token_id: int = 0  # Padding token ID
    
    # 训练策略配置
    mixed_precision: bool = True  # 是否使用混合精度训练
    precision_type: str = "bf16"  # 混合精度类型: fp16, bf16
    gradient_accumulation_steps: int = 4  # 梯度累积步数
    gradient_checkpointing: bool = False  # 是否使用梯度检查点
    max_grad_norm: float = 1.0  # 梯度剪裁阈值
    label_smoothing: float = 0.0  # 标签平滑系数
    adv_training: bool = False  # 对抗训练
    fgsm_epsilon: float = 0.3  # FGSM扰动大小
    layer_drop_prob: float = 0.0  # LayerDrop概率
    token_dropout_prob: float = 0.0  # Token级Dropout概率
    
    # 生成策略配置
    use_cache: bool = True  # 是否使用缓存加速生成
    use_causal_mask: bool = True  # 自回归监督时是否使用因果掩码
    enable_tf32: bool = True  # 是否启用TF32加速(Ampere架构)
    eos_token_id: int = 2  # 结束标记token ID
    repetition_penalty: float = 1.0  # 重复惩罚系数
    
    # 其他配置
    teacher_model_name: str = "/home/lizixi/xuelang/Qwen3-8B"  # 教师模型名称
    teacher_device: str = "cuda"  # 教师模型设备
    seed: int = 42  # 随机种子
    
    def __post_init__(self):
        """初始化后检查配置有效性"""
        warnings = []
        
        # 确保卷积核大小为奇数
        if self.ltc_kernel_size % 2 == 0:
            old_size = self.ltc_kernel_size
            self.ltc_kernel_size += 1
            warnings.append(f"卷积核大小必须为奇数，已从 {old_size} 调整为 {self.ltc_kernel_size}")
            
        # 确保distill_top_k合理
        if self.distill_top_k < 0:
            self.distill_top_k = 0
            warnings.append("distill_top_k不能为负数，已设置为0")
            
        # 确保学习率调度器类型有效
        valid_schedulers = ["cosine", "linear", "polynomial", "constant", "constant_with_warmup"]
        if self.lr_scheduler_type not in valid_schedulers:
            warnings.append(f"无效的学习率调度器类型 '{self.lr_scheduler_type}'，已设置为 'cosine'")
            self.lr_scheduler_type = "cosine"
            
        # 确保词汇表大小足够大
        if self.vocab_size < 10000:
            warnings.append(f"词汇表大小({self.vocab_size})过小，这可能不适用于Qwen模型")
            
        # 输出所有警告
        if warnings:
            print("配置警告:")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. 警告: {warning}")

@dataclass
class TrainingConfig:
    output_dir: str = "output"
    overwrite_output_dir: bool = True
    
    do_train: bool = True
    do_eval: bool = True
    
    # 数据参数
    dataset_name: str = "cdial-gpt"
    dataset_config_name: str = None
    train_file: Optional[str] = "./data/processed/train.json"
    validation_file: Optional[str] = "./data/processed/dev.json"
    
    # 训练超参数
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_train_epochs: int = 3
    max_steps: int = -1
    warmup_steps: int = 0
    
    # 分布式训练
    local_rank: int = -1
    deepspeed: Optional[str] = None
    
    # 保存与评估
    save_strategy: str = "steps"
    save_steps: int = 500
    eval_strategy: str = "steps"
    eval_steps: int = 500
    logging_dir: str = "logs"
    logging_steps: int = 100
    
    # 序列长度
    max_seq_length: int = 512
    pad_to_max_length: bool = False
    
    # 多卡训练
    num_gpus: int = 5
    
    # 混合精度训练
    fp16: bool = True
    
    # 梯度累积
    gradient_accumulation_steps: int = 1 