#!/bin/bash

# ====================================================
# MiniCodformer知识蒸馏推理脚本 (CCC+KD版)
# ====================================================

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # 无颜色

# 时间戳函数
timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

# 使用最新的CCC+KD蒸馏模型
# 默认使用基于CCC的最佳模型（通常效果更好）
MODEL_PATH="./output/distill_kd_ccc_full_10epoch/best_ccc_model"
# 如果CCC模型不存在，尝试使用普通最佳模型
if [ ! -d "$MODEL_PATH" ]; then
  MODEL_PATH="./output/distill_kd_ccc/best_model"
  # 如果普通最佳模型也不存在，使用最终模型
  if [ ! -d "$MODEL_PATH" ]; then
    MODEL_PATH="./output/distill_kd_ccc/final_model"
  fi
fi

MAX_LENGTH=100
TEMPERATURE=0.7
TOP_K=50
TOP_P=0.9
REPETITION_PENALTY=1.2  # 增加重复惩罚以减少重复文本

# 显示基本信息
echo -e "${GREEN}===================================================${NC}"
echo -e "${PURPLE}        MiniCodformer知识蒸馏模型推理 (CCC+KD版)      ${NC}"
echo -e "${GREEN}===================================================${NC}"
echo -e "${BLUE}[$(timestamp)] 使用模型: ${MODEL_PATH}${NC}"
echo -e "${BLUE}[$(timestamp)] 最大长度: ${MAX_LENGTH}${NC}"

# 检查CUDA可见设备
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES=0  # 默认使用第一个GPU
fi
echo -e "${BLUE}[$(timestamp)] CUDA设备: ${CUDA_VISIBLE_DEVICES}${NC}"

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
  echo -e "${RED}[$(timestamp)] 错误: 模型路径不存在: ${MODEL_PATH}${NC}"
  echo -e "${YELLOW}[$(timestamp)] 请先运行训练脚本 run_distill_kd_ccc.sh${NC}"
  exit 1
fi

# 检查Python环境
if ! command -v python &> /dev/null; then
  echo -e "${RED}[$(timestamp)] 错误: 未找到Python命令${NC}"
  exit 1
fi

# 检查inference_ccc_kd.py是否存在
if [ ! -f "inference_ccc_kd.py" ]; then
  echo -e "${RED}[$(timestamp)] 错误: 找不到推理脚本 inference_ccc_kd.py${NC}"
  exit 1
fi

# 确保脚本有执行权限
chmod +x inference_ccc_kd.py

# 构建推理命令
CMD="python inference_ccc_kd.py \
  --model_path ${MODEL_PATH} \
  --max_length ${MAX_LENGTH} \
  --temperature ${TEMPERATURE} \
  --top_k ${TOP_K} \
  --top_p ${TOP_P} \
  --repetition_penalty ${REPETITION_PENALTY} \
  --time_stats"

# 处理命令行参数
if [ "$#" -ge 1 ]; then
  if [ "$1" == "--prompt" ] && [ "$2" != "" ]; then
    # 使用命令行提示
    CMD="${CMD} --prompt \"$2\""
    echo -e "${BLUE}[$(timestamp)] 使用单个提示: $2${NC}"
    
    # 检查是否指定了输出文件
    if [ "$3" == "--output" ] && [ "$4" != "" ]; then
      CMD="${CMD} --output \"$4\""
      echo -e "${BLUE}[$(timestamp)] 输出将保存到: $4${NC}"
    fi
  elif [ "$1" == "--input" ] && [ "$2" != "" ]; then
    # 使用输入文件
    if [ -f "$2" ]; then
      CMD="${CMD} --input \"$2\""
      echo -e "${BLUE}[$(timestamp)] 输入文件: $2${NC}"
      
      # 检查是否指定了输出文件
      if [ "$3" == "--output" ] && [ "$4" != "" ]; then
        CMD="${CMD} --output \"$4\""
        echo -e "${BLUE}[$(timestamp)] 输出将保存到: $4${NC}"
      fi
    else
      echo -e "${RED}[$(timestamp)] 输入文件不存在: $2${NC}"
      exit 1
    fi
  elif [ -f "$1" ]; then
    # 使用输入文件
    CMD="${CMD} --input \"$1\""
    echo -e "${BLUE}[$(timestamp)] 输入文件: $1${NC}"
    
    # 检查是否指定了输出文件
    if [ "$2" == "--output" ] && [ "$3" != "" ]; then
      CMD="${CMD} --output \"$3\""
      echo -e "${BLUE}[$(timestamp)] 输出将保存到: $3${NC}"
    fi
  else
    echo -e "${YELLOW}[$(timestamp)] 警告: 未识别的参数: $1${NC}"
    echo -e "${YELLOW}[$(timestamp)] 使用交互式输入模式...${NC}"
  fi
else
  echo -e "${YELLOW}[$(timestamp)] 使用交互式输入模式...${NC}"
  # 如果没有参数，尝试使用测试提示文件
  if [ -f "test_prompts.txt" ]; then
    echo -e "${YELLOW}[$(timestamp)] 检测到测试提示文件，是否使用? (y/n)${NC}"
    read -r answer
    if [[ $answer =~ ^[Yy]$ ]]; then
      CMD="${CMD} --input test_prompts.txt"
      echo -e "${BLUE}[$(timestamp)] 使用测试提示文件: test_prompts.txt${NC}"
    fi
  fi
fi

echo -e "${GREEN}[$(timestamp)] 开始推理...${NC}"
echo ""

# 运行推理
eval "${CMD}"

if [ $? -eq 0 ]; then
  echo -e "${GREEN}[$(timestamp)] 推理完成!${NC}"
else
  echo -e "${RED}[$(timestamp)] 推理过程中出现错误!${NC}"
  exit 1
fi 