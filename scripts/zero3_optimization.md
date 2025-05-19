# ZeRO-3 优化配置说明

本文档介绍了MiniCodformer知识蒸馏项目中使用的DeepSpeed ZeRO-3优化配置及其作用。

## 主要优化点

- 内存效率大幅提升：通过ZeRO-3的参数分片和CPU卸载
- 避免OOM：通过精细的内存管理策略
- 性能平衡：在内存效率和计算速度之间寻求平衡

## 配置文件详解 (ds_config.json)

### 基础配置

```json
{
  "train_batch_size": 16,
  "gradient_accumulation_steps": 1
}
```
- `train_batch_size`: 全局批处理大小
- `gradient_accumulation_steps`: 梯度累积步数

### ZeRO-3 核心配置

```json
"zero_optimization": {
  "stage": 3,
  "offload_optimizer": {
    "device": "cpu",
    "pin_memory": true,
    "fast_init": true
  },
  "offload_param": {
    "device": "cpu",
    "pin_memory": true,
    "buffer_count": 5,
    "buffer_size": 1e8,
    "max_in_cpu": 1e9
  }
}
```

- `stage`: 设置为3表示启用ZeRO-3（参数分片、梯度分片、优化器状态分片）
- `offload_optimizer`: 将优化器状态卸载到CPU
  - `pin_memory`: 使用固定内存提高CPU-GPU数据传输速度
  - `fast_init`: 加速初始化过程
- `offload_param`: 将参数卸载到CPU
  - `buffer_count`: CPU-GPU之间传输的缓冲区数量
  - `buffer_size`: 每个缓冲区的大小
  - `max_in_cpu`: CPU中可以存储的最大参数量

### 高级内存管理

```json
"stage3_prefetch_bucket_size": 1e7,
"stage3_param_persistence_threshold": 1e5,
"stage3_max_live_parameters": 5e8,
"stage3_max_reuse_distance": 5e8,
"stage3_gather_16bit_weights_on_model_save": true,
"round_robin_gradients": true
```

- `stage3_prefetch_bucket_size`: 预取参数的批次大小
- `stage3_param_persistence_threshold`: 小于此大小的参数层将保持在GPU中
- `stage3_max_live_parameters`: 在任意时间点GPU中的最大参数数量
- `stage3_max_reuse_distance`: 参数在被再次使用前可以被换出的最大距离
- `round_robin_gradients`: 使用轮询策略处理梯度，平衡计算和通信

### 激活检查点

```json
"activation_checkpointing": {
  "partition_activations": true,
  "cpu_checkpointing": true,
  "contiguous_memory_optimization": true,
  "number_checkpoints": 8
}
```

- `partition_activations`: 分区激活以减少内存占用
- `cpu_checkpointing`: 将激活检查点存储在CPU上
- `contiguous_memory_optimization`: 使用连续内存块优化性能
- `number_checkpoints`: 激活检查点的数量

### 其他优化

```json
"memory_efficient_linear": true,
"aio": {
  "block_size": 1048576,
  "queue_depth": 8,
  "single_submit": false,
  "overlap_events": true
},
"use_dynamic_tiling": true
```

- `memory_efficient_linear`: 使用内存效率更高的线性层实现
- `aio`: 异步I/O优化
- `use_dynamic_tiling`: 使用动态平铺优化CUDA操作

## 环境变量优化（在脚本中）

```bash
# DeepSpeed性能优化
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_AUTO_BOOST=0
export DS_PIPE_RESERVE=10
export DS_SKIP_UNUSED_PARAMETERS=true
export DS_REUSE_BUFFERS=1
export DS_USE_CUDA_TILING=1
export TORCH_DISTRIBUTED_DEBUG=OFF
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.9"
```

- `CUDA_DEVICE_MAX_CONNECTIONS`: 限制CUDA设备连接，稳定通信
- `CUDA_AUTO_BOOST`: 关闭GPU自动睿频，稳定性能
- `DS_PIPE_RESERVE`: 提前为管道并行预留内存
- `DS_SKIP_UNUSED_PARAMETERS`: 跳过未使用参数，减少内存占用
- `DS_REUSE_BUFFERS`: 重用内存缓冲区
- `DS_USE_CUDA_TILING`: 启用CUDA平铺优化
- `PYTORCH_CUDA_ALLOC_CONF`: 优化PyTorch CUDA内存分配

## 性能监控

在训练过程中，可以使用以下命令监控GPU使用情况：

```bash
watch -n 1 nvidia-smi
```

内存使用异常时，可以考虑进一步减小批处理大小或调整ZeRO-3参数。 