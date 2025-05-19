#!/bin/bash

LOG_FILE="training_monitor.log"
echo "时间戳,CPU负载,内存使用,GPU使用率,GPU内存使用" > $LOG_FILE

echo "开始监控系统状态，日志保存在 $LOG_FILE"
echo "按 Ctrl+C 停止监控"

# 监控间隔（秒）
INTERVAL=1

# 监控次数（设置为-1表示无限循环直到手动停止）
COUNT=1000

i=0
while [ $COUNT -eq -1 ] || [ $i -lt $COUNT ]; do
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    CPU_LOAD=$(uptime | awk -F'[a-z]:' '{ print $2}' | sed 's/,//g')
    MEM_USAGE=$(free -m | awk 'NR==2{printf "%.2f%%", $3*100/$2 }')
    
    # 获取GPU信息
    GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
    
    echo "$TIMESTAMP,$CPU_LOAD,$MEM_USAGE,$GPU_INFO" >> $LOG_FILE
    
    # 额外记录进程信息
    if [ $((i % 10)) -eq 0 ]; then
        echo "=== 进程信息 $TIMESTAMP ===" >> $LOG_FILE
        ps aux | grep -E 'python|pytorch' | grep -v grep >> $LOG_FILE
        echo "=== NVIDIA进程 ===" >> $LOG_FILE
        nvidia-smi pmon -c 1 >> $LOG_FILE
        echo "" >> $LOG_FILE
    fi
    
    sleep $INTERVAL
    i=$((i+1))
done 