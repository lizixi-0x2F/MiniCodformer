#!/bin/bash

# 设置颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # 恢复默认颜色

echo -e "${GREEN}===== 训练崩溃调试工具 =====${NC}"
echo "此脚本将帮助您诊断训练过程中的崩溃问题"

# 检查是否安装了必要的工具
check_tools() {
    echo -e "${YELLOW}检查必要工具...${NC}"
    local missing_tools=()
    
    # 检查nvidia-smi
    if ! command -v nvidia-smi &> /dev/null; then
        missing_tools+=("nvidia-smi")
    fi
    
    # 检查Python和必要的包
    if ! command -v python3 &> /dev/null; then
        missing_tools+=("python3")
    else
        for pkg in torch psutil; do
            if ! python3 -c "import $pkg" &> /dev/null; then
                missing_tools+=("Python包: $pkg")
            fi
        done
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        echo -e "${RED}缺少以下工具:${NC}"
        for tool in "${missing_tools[@]}"; do
            echo " - $tool"
        done
        return 1
    else
        echo -e "${GREEN}所有必要工具已安装!${NC}"
        return 0
    fi
}

# 获取硬件信息
get_system_info() {
    echo -e "${YELLOW}收集系统信息...${NC}"
    echo "CPU信息:"
    lscpu | grep "Model name\|CPU(s)\|MHz" | sed 's/^/  /'
    
    echo -e "\n内存信息:"
    free -h | sed 's/^/  /'
    
    echo -e "\nGPU信息:"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,power.draw,power.limit --format=csv | sed 's/^/  /'
    
    echo -e "\n磁盘使用情况:"
    df -h | grep -v "tmpfs\|udev" | sed 's/^/  /'
    
    echo -e "\n操作系统信息:"
    cat /etc/os-release | grep "PRETTY_NAME" | sed 's/PRETTY_NAME=//g' | sed 's/"//g' | sed 's/^/  /'
}

# 启动监控
start_monitoring() {
    echo -e "${YELLOW}启动系统监控...${NC}"
    
    # 创建日志目录
    LOG_DIR="training_debug_logs_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$LOG_DIR"
    echo -e "${GREEN}日志将保存在目录: $LOG_DIR${NC}"
    
    # 启动基本系统监控
    ./monitor.sh > "$LOG_DIR/system_monitor.log" 2>&1 &
    MONITOR_PID=$!
    echo "系统监控进程ID: $MONITOR_PID"
    
    # 询问训练进程ID
    echo -e "${YELLOW}请输入您的训练进程ID (如果不知道，可以输入 'find' 查找):${NC}"
    read -r TRAIN_PID
    
    if [ "$TRAIN_PID" = "find" ]; then
        echo "搜索可能的训练进程..."
        ps aux | grep -E "python|train" | grep -v grep
        echo -e "${YELLOW}请从上面的列表中输入训练进程ID:${NC}"
        read -r TRAIN_PID
    fi
    
    # 验证进程ID
    if ! ps -p "$TRAIN_PID" > /dev/null; then
        echo -e "${RED}错误: 进程ID $TRAIN_PID 不存在${NC}"
        echo "请输入正确的进程ID:"
        read -r TRAIN_PID
    fi
    
    echo -e "${GREEN}将监控进程ID: $TRAIN_PID${NC}"
    
    # 启动CUDA错误检测
    python3 check_cuda_errors.py --interval 5 --pid "$TRAIN_PID" > "$LOG_DIR/cuda_errors.log" 2>&1 &
    CUDA_CHECK_PID=$!
    echo "CUDA错误检测进程ID: $CUDA_CHECK_PID"
    
    # 启动内存监控
    python3 debug_oom.py --log-file "$LOG_DIR/memory_profile.log" --interval 30 > /dev/null 2>&1 &
    MEMORY_CHECK_PID=$!
    echo "内存监控进程ID: $MEMORY_CHECK_PID"
    
    # 记录所有监控进程ID
    echo "$MONITOR_PID $CUDA_CHECK_PID $MEMORY_CHECK_PID" > "$LOG_DIR/monitor_pids.txt"
    
    echo -e "${GREEN}所有监控工具已启动!${NC}"
    echo -e "${YELLOW}监控日志将保存在目录: $LOG_DIR${NC}"
    echo -e "${YELLOW}您可以继续运行训练，如果发生崩溃，请运行 $0 analyze $LOG_DIR${NC}"
    echo -e "${YELLOW}要停止监控，请运行 $0 stop $LOG_DIR${NC}"
}

# 停止监控
stop_monitoring() {
    if [ -z "$1" ]; then
        echo -e "${RED}错误: 未指定日志目录${NC}"
        echo "用法: $0 stop <日志目录>"
        return 1
    fi
    
    LOG_DIR="$1"
    if [ ! -d "$LOG_DIR" ]; then
        echo -e "${RED}错误: 日志目录 $LOG_DIR 不存在${NC}"
        return 1
    fi
    
    if [ ! -f "$LOG_DIR/monitor_pids.txt" ]; then
        echo -e "${RED}错误: 找不到监控进程ID文件${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}停止监控...${NC}"
    while read -r pid; do
        if ps -p "$pid" > /dev/null; then
            echo "终止进程 $pid"
            kill "$pid" 2>/dev/null
        fi
    done < "$LOG_DIR/monitor_pids.txt"
    
    echo -e "${GREEN}所有监控进程已停止${NC}"
    echo -e "${YELLOW}现在可以运行 $0 analyze $LOG_DIR 来分析日志${NC}"
}

# 分析日志
analyze_logs() {
    if [ -z "$1" ]; then
        echo -e "${RED}错误: 未指定日志目录${NC}"
        echo "用法: $0 analyze <日志目录>"
        return 1
    fi
    
    LOG_DIR="$1"
    if [ ! -d "$LOG_DIR" ]; then
        echo -e "${RED}错误: 日志目录 $LOG_DIR 不存在${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}分析日志...${NC}"
    
    # 分析系统监控日志
    echo -e "${GREEN}系统监控日志分析:${NC}"
    if [ -f "$LOG_DIR/system_monitor.log" ]; then
        # 检查CPU负载峰值
        echo "CPU负载峰值:"
        grep -E "CPU负载" "$LOG_DIR/system_monitor.log" | sort -k2 -n -r | head -5 | sed 's/^/  /'
        
        # 检查内存使用峰值
        echo "内存使用峰值:"
        grep -E "内存使用" "$LOG_DIR/system_monitor.log" | sort -k3 -n -r | head -5 | sed 's/^/  /'
        
        # 检查GPU使用率峰值
        echo "GPU使用率峰值:"
        grep -E "GPU使用率" "$LOG_DIR/system_monitor.log" | sort -k4 -n -r | head -5 | sed 's/^/  /'
        
        # 检查GPU内存使用峰值
        echo "GPU内存使用峰值:"
        grep -E "GPU内存使用" "$LOG_DIR/system_monitor.log" | sort -k5 -n -r | head -5 | sed 's/^/  /'
    else
        echo -e "${RED}找不到系统监控日志${NC}"
    fi
    
    # 分析CUDA错误日志
    echo -e "\n${GREEN}CUDA错误日志分析:${NC}"
    if [ -f "$LOG_DIR/cuda_errors.log" ]; then
        if grep -q "错误" "$LOG_DIR/cuda_errors.log"; then
            echo -e "${RED}发现CUDA错误:${NC}"
            grep -E "错误|异常" "$LOG_DIR/cuda_errors.log" | sed 's/^/  /'
        else
            echo -e "${GREEN}未发现CUDA错误${NC}"
        fi
    else
        echo -e "${RED}找不到CUDA错误日志${NC}"
    fi
    
    # 分析内存监控日志
    echo -e "\n${GREEN}内存监控日志分析:${NC}"
    if [ -f "$LOG_DIR/memory_profile.log" ]; then
        # 检查最大的张量
        echo "最大的CUDA张量:"
        grep -A 10 "最大的CUDA张量" "$LOG_DIR/memory_profile.log" | grep "大小:" | sort -k2 -n -r | head -5 | sed 's/^/  /'
        
        # 检查内存增长趋势
        echo "内存使用趋势:"
        grep "CUDA张量总内存" "$LOG_DIR/memory_profile.log" | sed 's/^/  /'
    else
        echo -e "${RED}找不到内存监控日志${NC}"
    fi
    
    echo -e "\n${YELLOW}建议:${NC}"
    echo "1. 检查是否有内存持续增长的情况，这可能表明存在内存泄漏"
    echo "2. 观察崩溃前的GPU内存使用情况，是否接近GPU总内存"
    echo "3. 检查是否有CUDA错误，特别是'out of memory'或'illegal memory access'等错误"
    echo "4. 查看CPU和GPU负载是否存在异常峰值"
    echo "5. 如果怀疑是内存泄漏，可能需要检查代码中的张量操作，尤其是大型张量是否被适当释放"
}

# 主函数
main() {
    if ! check_tools; then
        echo -e "${RED}请安装所有必要的工具后再运行此脚本${NC}"
        exit 1
    fi
    
    case "$1" in
        "start")
            get_system_info
            start_monitoring
            ;;
        "stop")
            stop_monitoring "$2"
            ;;
        "analyze")
            analyze_logs "$2"
            ;;
        *)
            echo -e "${GREEN}训练崩溃调试工具${NC}"
            echo "用法:"
            echo "  $0 start             - 启动监控"
            echo "  $0 stop <日志目录>    - 停止监控"
            echo "  $0 analyze <日志目录> - 分析日志"
            ;;
    esac
}

main "$@" 