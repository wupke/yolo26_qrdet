#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
FilePath: /ultralytics/scripts/ros_yolo_detector2.py
author: wupke
Date: 2026-02-03 13:38:58
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-02-03 14:07:03
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
'''

'''

cd ~/01yuyao/ultralytics/

# 1. 补上 OpenBLAS 库路径
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/openblas-pthread/

# 2. 将 ROS 官方 Python 路径加入 PYTHONPATH（这步很关键）
export PYTHONPATH=$PYTHONPATH:/opt/ros/noetic/lib/python3/dist-packages

# 3. 启动脚本
python -s ros_yolo_detector.py

--------  在 4021 Orin上 直接运行pt推理rostopic 输出  ------------- ：

--- FPS: 34.32 --- | straight: 0.91 | right: 0.80
--- FPS: 34.93 --- | straight: 0.91 | right: 0.83
--- FPS: 34.79 --- | straight: 0.91 | right: 0.83
--- FPS: 34.53 --- | straight: 0.91 | right: 0.82
--- FPS: 35.62 --- | straight: 0.91 | right: 0.81
--- FPS: 36.07 --- | straight: 0.91 | right: 0.81
--- FPS: 31.71 --- | straight: 0.91 | right: 0.83
--- FPS: 28.01 --- | straight: 0.91 | right: 0.82

TODO:
---------------------------
1. 终端显示优化：防止“刷屏”
目前的输出每一帧都会换行，如果你想让终端更像一个监控面板，
可以使用 \r（回车符）让输出保持在同一行，并根据置信度过滤掉不重要的干扰。

# 修改打印部分
# \r 使光标回到行首，end="" 防止换行
print(f"\r{log_str:<100}", end="", flush=True)

----------------------------------------
2. 导出 TensorRT 模型 (.engine) —— 性能翻倍

# 在终端执行
yolo export model=runs/detect/train_s-label2/weights/best.pt format=engine device=0 half=True

转换后，只需将脚本中的 model_path 指向 best.engine 即可，代码无需其他修改。

----------------------------------------

3. 加入“看门狗”逻辑：处理空图像
在 ROS 机器人开发中，相机偶尔会因为带宽或发热断流。可增加一个简单保护，防止脚本因收到空数据而崩溃。

def callback(self, data):
        # 增加对空数据的判断
        if not data.data:
            rospy.logwarn_throttle(5, "Received empty image frame")
            return
            
        start_time = time.time()
        # ... 后续逻辑


'''



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
import cv2
from ultralytics import YOLO
import numpy as np
import time

class YoloRosDetector:
    def __init__(self):
        rospy.init_node('yolo_detector_node', anonymous=True)
        
        # 1. 加载模型
        model_path = "runs/detect/train_s-label2/weights/best.pt"
        self.model = YOLO(model_path)
        
        # 2. 初始化统计变量
        self.prev_time = 0
        self.fps = 0
        
        # 3. 订阅与发布
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.callback)
        self.image_pub = rospy.Publisher("/yolo/detected_image", Image, queue_size=1)
        
        rospy.loginfo("YOLO Detector Node Started.")

    def callback(self, data):
        # 记录开始推理的时间
        start_time = time.time()
        
        try:
            # 1. 图像解析
            img_array = np.frombuffer(data.data, dtype=np.uint8)
            cv_image = img_array.reshape(data.height, data.width, 3)
            # 如果你的画面颜色看起来不对（比如蓝红颠倒），可以尝试取消下面这行的注释
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

            # 2. 执行推理
            results = self.model.predict(cv_image, conf=0.1, device='0', verbose=False)[0]
            
            # --- 🚀 新增：性能与结果输出 ---
            current_time = time.time()
            # 计算帧率 (FPS)
            self.fps = 1.0 / (current_time - start_time)
            
            # 获取检测到的类别名称映射
            names = self.model.names
            
            # 终端打印头部
            log_str = f"--- FPS: {self.fps:.2f} ---"
            
            # 遍历检测框
            if len(results.boxes) > 0:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = names[cls_id]
                    # 打印每个目标的详细信息：类别 [置信度]
                    log_str += f" | {name}: {conf:.2f}"
            else:
                log_str += " | No objects detected."
            
            # 一次性输出到终端，避免多次打印闪烁
            print(log_str)
            # --- 🚀 结束：性能与结果输出 ---

            # 3. 绘制并发布图像
            annotated_frame = results.plot()
            msg = Image()
            msg.header = data.header
            msg.height = annotated_frame.shape[0]
            msg.width = annotated_frame.shape[1]
            msg.encoding = "bgr8"
            msg.step = annotated_frame.shape[1] * 3
            msg.data = annotated_frame.tobytes()
            self.image_pub.publish(msg)
            
        except Exception as e:
            rospy.logerr(f"Prediction Error: {e}")

if __name__ == '__main__':
    try:
        detector = YoloRosDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
