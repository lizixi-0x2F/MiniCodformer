#!/bin/bash

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 时间戳函数
timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

# 检测GPU数量
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo -e "${BLUE}[$(timestamp)] 检测到 ${NUM_GPUS} 个GPU可用${NC}"

# 基本训练配置
TEACHER_MODEL="/home/lizixi/xuelang/Qwen3-8B"
TRAIN_DATA="./data/processed/train.json"
EVAL_DATA="./data/processed/dev.json"
OUTPUT_DIR="./output/distill"
MAX_LENGTH=128
BATCH_SIZE=2
GRAD_ACCUM=4
EPOCHS=10
ZERO_STAGE=3

# 离线模式设置
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 创建输出目录
mkdir -p ${OUTPUT_DIR}/logs ${OUTPUT_DIR}/cache

# 检查必要文件路径
if [ ! -d "${TEACHER_MODEL}" ] || [ ! -f "${TRAIN_DATA}" ] || [ ! -f "${EVAL_DATA}" ]; then
  echo -e "${RED}[$(timestamp)] 错误: 模型或数据文件不存在${NC}"
  exit 1
fi

# DeepSpeed ZeRO-3优化环境变量
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SHM_DISABLE=1
export MASTER_ADDR="localhost"
export MASTER_PORT="29500"

# DeepSpeed性能优化
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_AUTO_BOOST=0  # 关闭GPU自动睿频
export DS_PIPE_RESERVE=10  # 提前分配10%的内存用于动态增长
export DS_SKIP_UNUSED_PARAMETERS=true  # 跳过未使用的参数
export DS_REUSE_BUFFERS=1  # 重用内存缓冲区
export DS_USE_CUDA_TILING=1  # 使用CUDA平铺加速
export TORCH_DISTRIBUTED_DEBUG=OFF  # 减少调试输出，提高性能
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.9" # 内存优化

# GPU分配
if [ $NUM_GPUS -ge 3 ]; then
  TEACHER_GPUS="0,1"
  STUDENT_GPUS=2
elif [ $NUM_GPUS -eq 2 ]; then
  TEACHER_GPUS="0"
  STUDENT_GPUS=1
else
  TEACHER_GPUS="0"
  STUDENT_GPUS=0
fi
STUDENT_GPU_COUNT=1

# 更新训练配置
export CUDA_VISIBLE_DEVICES=0,1,2
export WORLD_SIZE=${STUDENT_GPU_COUNT}

echo -e "${BLUE}[$(timestamp)] 教师模型使用GPU: ${TEACHER_GPUS}, 学生模型使用GPU: ${STUDENT_GPUS}${NC}"
nvidia-smi

# 计算总批次大小
TOTAL_BATCH=$((BATCH_SIZE * GRAD_ACCUM * STUDENT_GPU_COUNT))
echo -e "${BLUE}[$(timestamp)] 总批次大小=${TOTAL_BATCH}${NC}"

# DeepSpeed命令
DEEPSPEED_CMD="deepspeed \
  --include=localhost:${STUDENT_GPUS} \
  train_ds.py \
  --train_file ${TRAIN_DATA} \
  --validation_file ${EVAL_DATA} \
  --teacher_model ${TEACHER_MODEL} \
  --output_dir ${OUTPUT_DIR} \
  --max_seq_length ${MAX_LENGTH} \
  --batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACCUM} \
  --num_train_epochs ${EPOCHS} \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --top_k 5 \
  --temperature 2.0 \
  --save_steps 500 \
  --eval_steps 500 \
  --top_k_distillation \
  --overwrite_cache \
  --seed 42 \
  --deepspeed \
  --zero_stage ${ZERO_STAGE} \
  --fp16 \
  --deepspeed_config ds_config.json \
  --teacher_gpus ${TEACHER_GPUS}"

# 释放缓存
sync && echo 3 > /proc/sys/vm/drop_caches

# 信号处理
trap 'echo -e "${RED}[$(timestamp)] 检测到中断信号...${NC}"; exit 1' SIGINT SIGTERM

# 运行训练
START_TIME=$(date +%s)
echo -e "${YELLOW}[$(timestamp)] 启动DeepSpeed训练...${NC}"
eval ${DEEPSPEED_CMD}
TRAIN_EXIT_STATUS=$?

# 计算训练时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

# 训练结果报告
if [ $TRAIN_EXIT_STATUS -eq 0 ]; then
  echo -e "${GREEN}[$(timestamp)] 训练成功完成! 用时: ${HOURS}h ${MINUTES}m ${SECONDS}s${NC}"
else
  echo -e "${RED}[$(timestamp)] 训练异常退出，状态码: ${TRAIN_EXIT_STATUS}${NC}"
fi

echo -e "${BLUE}[$(timestamp)] 训练完成${NC}"