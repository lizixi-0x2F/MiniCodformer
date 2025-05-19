# MiniCodformer 知识蒸馏项目

[![GitHub](https://img.shields.io/github/license/lizixi-0x2F/MiniCodformer?color=blue)](https://github.com/lizixi-0x2F/MiniCodformer)

## 项目简介

MiniCodformer 是一个轻量级、高效的知识蒸馏项目，通过从 Qwen3-8B 大模型中蒸馏知识到小模型中，使用 DeepSpeed ZeRO-3 技术进行高效训练。模型采用创新的 LTC-NCP（Linear Transformer with Continuous Non-Causal Processing）架构，显著降低推理延迟。

## 特性

- **轻量化**: 小型模型具有显著降低的参数量，推理速度提升3-5倍
- **高效训练**: 使用 DeepSpeed ZeRO-3 优化，支持多GPU分布式训练
- **创新架构**: 采用 LTC-NCP 解码器架构，优化长序列生成效率
- **灵活部署**: 支持命令行、Python API 和 HTTP API 多种推理方式

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/lizixi-0x2F/MiniCodformer.git
cd MiniCodformer
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备教师模型

下载 Qwen3-8B 模型到项目根目录下的 `Qwen3-8B` 文件夹中。

## 快速开始

### 训练模型

```bash
cd scripts
bash run_distill.sh
```

主要训练参数：
- 批处理大小: 2
- 梯度累积步数: 4
- 训练轮数: 10
- ZeRO 阶段: 3

### 推理

```bash
cd scripts
bash run_inference_ccc_kd.sh --input "请介绍一下自己"
```

## 项目结构

```
.
├── config.py                  # 模型配置
├── data.py                    # 数据处理
├── data/                      # 训练和验证数据
├── inference_ccc_kd.py        # 推理脚本
├── model.py                   # 模型定义（包含Encoder+LTC-NCP解码器）
├── output/                    # 输出目录
├── requirements.txt           # 依赖包
├── scripts/                   # 脚本目录
│   ├── ds_config.json         # DeepSpeed配置
│   ├── run_distill.sh         # 蒸馏训练脚本
│   ├── run_inference_ccc_kd.sh  # 推理脚本
│   └── zero3_optimization.md  # ZeRO-3优化说明
└── train_ds.py                # DeepSpeed训练代码
```

## 模型架构

- **教师模型**: Qwen3-8B
- **学生模型**:
  - **编码器**: Mini-Transformer（隐藏层256，层数3，注意力头4）
  - **解码器**: LTC-NCP（高效的长距离依赖建模解码器）

## ZeRO-3 优化

项目使用 DeepSpeed 的 ZeRO-3 优化技术，显著降低 GPU 内存使用，优化包括：

- **参数分片**: 模型参数在多 GPU 间分片存储
- **优化器状态分片**: 优化器状态也在 GPU 间分片
- **CPU 卸载**: 将部分参数和优化器状态卸载到 CPU 内存
- **内存管理**: 动态管理 GPU 内存，避免 OOM 错误

详细优化配置说明见 `scripts/zero3_optimization.md`

## 训练方法

### 1. 准备数据

已处理好的数据位于 `data/processed/` 目录。如需处理新数据，可使用：

```bash
python process_data.py \
    --input_file <input_jsonl_file> \
    --output_dir ./data/processed \
    --train_size 20000 \
    --val_size 2000
```

### 2. 自定义训练参数

修改 `scripts/run_distill.sh` 文件中的参数：

```bash
MAX_LENGTH=128   # 序列最大长度
BATCH_SIZE=2     # 批处理大小
GRAD_ACCUM=4     # 梯度累积步数
EPOCHS=10        # 训练轮数
ZERO_STAGE=3     # ZeRO 优化阶段
```

### 3. 监控训练

使用以下命令监控 GPU 使用情况：

```bash
watch -n 1 nvidia-smi
```

## 硬件要求

- 推荐使用 3 张或以上的 GPU
- 如果只有 1-2 张 GPU，会自动调整配置
- 推荐每个 GPU 至少 16GB 显存

## 进度记录

| 日期 | 更新内容 |
|------|---------|
| 2025-05-19 | 实现 ZeRO-3 优化，降低内存使用 |
| 2025-05-18 | 项目精简，只保留 PyTorch 多卡训练 |
| 2025-05-17 | 初始版本发布 |

## 许可证

[MIT License](LICENSE)

