

###
 # @FilePath: /yolo26/ultralytics/qrdet/record_bags.sh
 # @author: wupke
 # @Date: 2026-02-26 10:48:22
 # @Version: 1.0
 # @LastEditors: wupke
 # @LastEditTime: 2026-02-26 10:54:03
 # @Description:       
 # @Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
### 



#!/bin/bash

# =========================================
# 配置参数
# =========================================
TOPIC="/camera/rgb_image_raw"        # 目标话题
DURATION="60"                        # 每个包的持续时间（秒）
PREFIX="rosbagCorner"                # 文件名前缀
rosbagCornerDir="./rosbag_data"      # 保存文件夹路径（可自定义）
COUNTER=1                            # 计数器起始值

# =========================================
# 初始化环境
# =========================================
# 检查文件夹是否存在，不存在则创建
if [ ! -d "$rosbagCornerDir" ]; then
    echo "文件夹 $rosbagCornerDir 不存在，正在创建..."
    mkdir -p "$rosbagCornerDir"
fi

echo "------------------------------------------------"
echo "开始录制 ROSBag 数据包"
echo "目标话题: $TOPIC"
echo "保存位置: $rosbagCornerDir"
echo "保存间隔: ${DURATION}s"
echo "按 Ctrl+C 停止录制"
echo "------------------------------------------------"

# 捕获 Ctrl+C 信号以优雅地退出
trap "echo -e '\n录制已由用户停止。'; exit" SIGINT

# =========================================
# 循环录制逻辑
# =========================================
while true
do
    # 构建完整的文件保存路径
    FILENAME="${rosbagCornerDir}/${PREFIX}_${COUNTER}.bag"
    
    echo "[$(date +%T)] 正在录制第 $COUNTER 个包: $FILENAME ..."
    
    # 执行录制
    # --duration: 指定录制时长
    # -O: 指定输出路径及文件名
    rosbag record $TOPIC --duration=$DURATION -O $FILENAME --quiet

    # 检查上一条指令的退出状态
    # 如果用户按下 Ctrl+C，rosbag 会返回非 0 状态，此时跳出循环
    if [ $? -ne 0 ]; then
        break
    fi

    echo "保存成功: $FILENAME"
    
    # 计数器累加
    ((COUNTER++))
done























#####################   without --saveDir

# #!/bin/bash

# # =========================================
# # 配置参数
# # =========================================
# TOPIC="/camera/rgb_image_raw"    # 目标话题
# DURATION="60"                    # 每个包的持续时间（秒）
# PREFIX="rosbagCorner"            # 文件名前缀
# COUNTER=1                        # 计数器起始值

# echo "------------------------------------------------"
# echo "开始录制 ROSBag 数据包"
# echo "目标话题: $TOPIC"
# echo "保存间隔: ${DURATION}s"
# echo "按 Ctrl+C 停止录制"
# echo "------------------------------------------------"

# # 捕获 Ctrl+C 信号以优雅地退出
# trap "echo -e '\n停止录制。'; exit" SIGINT

# # 循环录制
# while true
# do
#     FILENAME="${PREFIX}_${COUNTER}.bag"
    
#     echo "[$(date +%T)] 正在录制第 $COUNTER 个包: $FILENAME ..."
    
#     # --duration: 指定录制时长
#     # -O: 指定输出文件名 (会覆盖同名文件)
#     # --quiet: 减少输出信息
#     rosbag record $TOPIC --duration=$DURATION -O $FILENAME --quiet

#     # 检查上一个命令执行状态，如果是被手动中止则退出
#     if [ $? -ne 0 ]; then
#         break
#     fi

#     echo "完成保存: $FILENAME"
    
#     # 计数器累加
#     ((COUNTER++))
# done

