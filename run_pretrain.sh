#!/bin/bash

# DeepSeek 500M模型预训练运行脚本
# 在维基百科数据上从头训练模型

# 错误处理设置
set -e  # 遇到错误立即退出
trap 'echo "错误: 脚本在第 $LINENO 行出错"; exit 1' ERR

# 设置CUDA设备 - 使用单GPU
export CUDA_VISIBLE_DEVICES=0  # 只使用一个GPU

# 设置PyTorch内存分配优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 设置关键目录和文件
OUTPUT_DIR="output/wiki_pretrain_500m"
DATA_DIR="data/corpus/wikipedia"
TOKENIZER_PATH="/home/lizixi/xuelang/Qwen3-8B"  # 使用Qwen分词器
LOCAL_WIKI_DATA="data/wikipedia-cn-20230720-filtered.json"

# 创建必要的目录
mkdir -p $OUTPUT_DIR
mkdir -p $DATA_DIR

echo "===== 开始DeepSeek 500M预训练流程 ====="

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

# 2. 运行预训练
echo "===== 启动500M模型预训练 ====="
echo "Starting 500M model pretraining..."

# 运行预训练脚本
python pretrain.py \
  --tokenizer_path $TOKENIZER_PATH \
  --output_dir $OUTPUT_DIR \
  --train_file $DATA_DIR/wiki_corpus.txt \
  --max_seq_length 512 \
  --hidden_size 2048 \
  --num_layers 24 \
  --num_heads 16 \
  --intermediate_size 8192 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-4 \
  --num_train_epochs 5 \
  --warmup_ratio 0.1 \
  --fp16 \
  --save_steps 1000 \
  --eval_steps 500 \
  --logging_steps 100 \
  --save_total_limit 3

# 检查预训练结果
if [ -d "$OUTPUT_DIR/checkpoint-*" ] || [ -f "$OUTPUT_DIR/pytorch_model.bin" ]; then
  echo "===== 预训练完成！ ====="
  echo "模型已保存至: $OUTPUT_DIR"
else
  echo "警告: 可能未生成完整的模型检查点"
  echo "请检查日志以获取更多信息"
fi 