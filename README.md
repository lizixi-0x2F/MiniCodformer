# MiniCodformer模型知识蒸馏项目

## 项目简介

这个项目实现了MiniCodformer模型从Qwen3-8B模型蒸馏的功能，使用DeepSpeed进行高效训练。

## 项目结构

```
.
├── config.py                  # 模型配置
├── data.py                    # 数据处理
├── data/                      # 训练和验证数据
├── DataSet-Deepseek-110k/     # 数据集
├── inference_ccc_kd.py        # 推理脚本
├── model.py                   # 模型定义
├── output/                    # 输出目录
├── Qwen3-8B/                  # 教师模型
├── requirements.txt           # 依赖包
├── scripts/                   # 脚本目录
│   ├── ds_config.json         # DeepSpeed配置
│   ├── run_distill.sh         # 蒸馏训练脚本
│   ├── run_inference_ccc.sh   # 推理脚本
│   └── zero3_optimization.md  # ZeRO-3优化说明
└── train_ds.py                # DeepSpeed训练代码
```

## 使用方法

### 训练

```bash
cd scripts
bash run_distill.sh
```

### 推理

```bash
cd scripts
bash run_inference_ccc_kd.sh
```

## 配置说明

- 训练配置位于 `scripts/run_distill.sh`，可以根据需要调整参数
- DeepSpeed配置位于 `scripts/ds_config.json`
- 模型配置位于 `config.py`

## ZeRO-3优化

项目使用DeepSpeed的ZeRO-3优化技术，显著降低GPU内存使用，优化包括：

- **参数分片**: 模型参数在多GPU间分片存储
- **优化器状态分片**: 优化器状态也在GPU间分片
- **CPU卸载**: 将部分参数和优化器状态卸载到CPU内存
- **内存管理**: 动态管理GPU内存，避免OOM错误

详细优化配置说明见 `scripts/zero3_optimization.md`

## 硬件要求

- 推荐使用3张或以上的GPU
- 如果只有1-2张GPU，会自动调整配置

# 知识蒸馏文本生成模型

本项目实现了基于LTC-NCP（Linear Transformer with Continuous Non-Causal Processing）的知识蒸馏文本生成模型。通过从大型教师模型（Qwen3-8B）中蒸馏知识到轻量级学生模型，实现高效推理与部署。

## 项目结构

```
.
├── model.py               # 模型架构（包含Encoder+LTC-NCP解码器）
├── data.py                # 数据处理模块
├── config.py              # 模型配置
├── optimized_train.py     # 优化训练脚本（多项蒸馏优化策略）
├── inference.py           # 推理模块
├── api_server.py          # HTTP API服务
├── run_parallel_train.sh  # 多GPU并行训练脚本
├── monitor_training.sh    # 训练监控脚本
├── requirements.txt       # 项目依赖
└── README.md              # 项目说明
```

## 核心特性

- **模型架构**：轻量级Encoder + LTC-NCP解码器，大幅降低参数量
- **生成加速**：实现encoder_outputs缓存和滑动窗口卷积状态缓存
- **增强蒸馏**：支持Top-k KL散度、动态温度、注意力匹配等优化
- **训练优化**：分层学习率、梯度累积、动态硬负例挖掘等
- **多GPU训练**：支持分布式数据并行训练

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动训练

```bash
# 多GPU并行训练
./run_parallel_train.sh
```

### 监控训练

```bash
# 查看训练状态
./monitor_training.sh
```

## 训练步骤

### 1. 准备数据

已处理好的数据位于`data/processed/`目录。如需处理新数据，可使用：

```bash
python process_deepseek_data.py \
    --input_file <input_jsonl_file> \
    --output_dir ./data/processed \
    --train_size 20000 \
    --val_size 2000
```

### 2. 多GPU分布式训练

```bash
./run_multi_gpu.sh ./data/processed/train.json ./data/processed/dev.json output_multi_gpu 3 4 3 4
```

参数说明：
- 第一个参数：训练数据文件路径
- 第二个参数：验证数据文件路径
- 第三个参数：输出目录
- 第四个参数：GPU数量（默认3）
- 第五个参数：每个GPU的批次大小（默认4）
- 第六个参数：训练轮数（默认3）
- 第七个参数：梯度累积步数（默认4）

### 3. 使用screen在后台训练

```bash
./run_multi_gpu_screen.sh ./data/processed/train.json ./data/processed/dev.json output_multi_gpu 3 4 3 4 minicodformer_train
```

参数说明与`run_multi_gpu.sh`相同，最后一个参数是screen会话名称。

训练启动后，可使用以下命令查看进度：
```bash
screen -r minicodformer_train
```

使用`Ctrl+A`然后按`D`从screen会话分离。

## 模型结构

- 教师模型：Qwen3-8B
- 学生模型：
  - 编码器：Mini-Transformer（隐藏层256，层数3，注意力头4）
  - 解码器：LTC-NCP（高效的长距离依赖建模解码器）

## 进度记录
| 任务 | 状态 | 时间 |
|-----|------|------|
| 项目精简，只保留PyTorch多卡训练 | 完成 | 2025-05-18 09:30 | 

# 文本生成模型推理接口使用说明

这个项目提供了用于文本生成模型推理的简单接口。支持命令行、Python API以及Web服务方式使用。

## 模型路径

默认模型路径为 `output_screen/final_model`