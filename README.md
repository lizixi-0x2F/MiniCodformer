# DeepSeek 500M模型蒸馏框架

这个项目提供了一个简洁高效的知识蒸馏框架，专注于将大型语言模型（如Qwen3-8B）的知识蒸馏到中等规模模型(约500M参数)中。项目针对中文语言模型进行了特别优化，并支持使用中文维基百科数据进行预训练和蒸馏。

## 主要功能

- **高效知识蒸馏**：使用优化的蒸馏算法将大模型知识传递到小模型
- **内存优化**：解决词汇表不匹配和CUDA内存溢出问题
- **中文预训练**：支持使用中文维基百科数据进行模型预训练
- **灵活架构**：支持自定义模型架构和多种蒸馏损失函数
- **单GPU训练**：针对有限计算资源环境进行优化

## 快速开始

### 环境准备

```bash
pip install -r requirements.txt
```

### 数据准备

项目使用中文维基百科数据（wikipedia-cn-20230720-filtered.json）：

```bash
# 处理维基百科数据
python process_local_wiki.py \
  --input_file data/wikipedia-cn-20230720-filtered.json \
  --output_dir data/corpus \
  --max_samples 100000
```

### 运行蒸馏

```bash
# 使用内存优化版蒸馏脚本
./run_optimized_distill.sh
```

## 核心文件说明

- `optimized_distill.py`：内存优化版知识蒸馏脚本，解决词汇表不匹配和显存溢出问题
- `run_optimized_distill.sh`：蒸馏脚本的运行配置
- `model.py`：自定义模型架构定义
- `process_local_wiki.py`：维基百科数据处理工具
- `requirements.txt`：项目依赖

## 蒸馏流程

1. 使用教师模型（如Qwen3-8B）获取知识
2. 将预训练的维基百科模型作为学生模型
3. 通过KL散度损失和原始语言模型损失联合优化
4. 自动处理词汇表大小不匹配问题
5. 内存优化确保单卡训练高效稳定

## 主要参数

- `--teacher_model`：教师模型路径，如Qwen3-8B
- `--student_model`：学生模型路径，如预训练的wiki模型
- `--alpha`：蒸馏损失权重（0.0-1.0）
- `--temperature`：蒸馏温度系数，控制软标签平滑程度
- `--fp16`：启用半精度训练，减少显存占用

## 技术说明

- 中等规模模型架构（约500M参数）
  - 隐藏层维度: 2048
  - Transformer编码器层数: 24
  - 注意力头数: 16
  - 前馈网络维度: 8192
  - LTC-NCP解码器层数: 12
- 使用JointKDLoss结合多种蒸馏损失函数
- 解决词汇表大小不匹配问题（151936 vs 151669）
- 使用CPU分块处理来避免CUDA内存溢出
- 支持自定义模型架构的知识蒸馏

