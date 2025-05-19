#!/bin/bash

# 内存优化版知识蒸馏运行脚本
# 解决词汇表大小不匹配问题和显存OOM问题

# 错误处理设置
set -e  # 遇到错误立即退出
trap 'echo "错误: 脚本在第 $LINENO 行出错"; exit 1' ERR

# 设置CUDA设备 - 使用单GPU
export CUDA_VISIBLE_DEVICES=0  # 只使用一个GPU

# 设置PyTorch内存分配优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 设置关键目录和文件
OUTPUT_DIR="output/distilled_wiki_model"
DATA_DIR="data/corpus/wikipedia"
TEACHER_MODEL="/home/lizixi/xuelang/Qwen3-8B"  # 使用Qwen作为教师模型
STUDENT_MODEL="/home/lizixi/xuelang/output/wiki_pretrain/best_model"  # 使用预训练Wiki模型作为学生模型
LOCAL_WIKI_DATA="data/wikipedia-cn-20230720-filtered.json"

# 创建必要的目录
mkdir -p $OUTPUT_DIR
mkdir -p $DATA_DIR

echo "===== 开始优化版知识蒸馏流程 ====="

# 1. 处理本地维基百科数据集
echo "===== 处理本地维基百科中文数据集 ====="
echo "Processing local Chinese Wikipedia dataset..."
python process_local_wiki.py \
  --input_file $LOCAL_WIKI_DATA \
  --output_dir $DATA_DIR \
  --max_samples 100000

# 检查数据准备是否成功
if [ ! -f "$DATA_DIR/wiki_corpus.txt" ]; then
  echo "错误: 数据准备失败，未找到 $DATA_DIR/wiki_corpus.txt"
  exit 1
fi

# 2. 运行内存优化版蒸馏
echo "===== 启动内存优化版知识蒸馏 ====="
echo "Starting memory-efficient knowledge distillation..."
echo "教师模型: $TEACHER_MODEL (Qwen)"
echo "学生模型: $STUDENT_MODEL (预训练Wiki模型)"
echo "Teacher model: $TEACHER_MODEL (Qwen)"
echo "Student model: $STUDENT_MODEL (Pre-trained Wiki model)"

# 运行优化版蒸馏脚本
python optimized_distill.py \
  --teacher_model $TEACHER_MODEL \
  --student_model $STUDENT_MODEL \
  --output_dir $OUTPUT_DIR \
  --train_file $DATA_DIR/wiki_corpus.txt \
  --max_seq_length 512 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --warmup_ratio 0.1 \
  --alpha 0.7 \
  --temperature 2.0 \
  --fp16 \
  --save_steps 2000 \
  --eval_steps 1000 \
  --logging_steps 500 \
  --save_total_limit 3

# 检查蒸馏结果
if [ -d "$OUTPUT_DIR/checkpoint-*" ] || [ -f "$OUTPUT_DIR/pytorch_model.bin" ]; then
  echo "===== 蒸馏训练完成！ ====="
  echo "模型已保存至: $OUTPUT_DIR"
else
  echo "警告: 可能未生成完整的模型检查点"
  echo "请检查日志以获取更多信息"
fi 