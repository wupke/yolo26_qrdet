#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
FilePath: /ultralytics/scripts/camera_interenceEngine.py
author: wupke
Date: 2026-03-27 14:08:00
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-03-27 14:08:03
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
'''



import time
import cv2
from ultralytics import YOLO

# 1. 加载 TensorRT 模型
model_path = 'runs/detect/train4/weights/best_150epoch.engine'
model = YOLO(model_path, task='detect')

# 2. 打开摄像头 (0 通常是笔记本自带摄像头或第一个 USB 摄像头)
cap = cv2.VideoCapture(0)

print("开始实时推理，按 'q' 键退出...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # --- 核心推理部分 ---
    start_time = time.time()
    
    # stream=True 能够显著减少内存占用，适用于实时视频流
    results = model.predict(frame, conf=0.5, device=0, stream=True)
    
    # 计算耗时
    end_time = time.time()
    latency = (end_time - start_time) * 1000  # 毫秒
    # ------------------

    for r in results:
        # 获取带有标注的图像（numpy 数组）
        annotated_frame = r.plot()
        
        # 在图像上实时打印推理耗时
        cv2.putText(annotated_frame, f"Latency: {latency:.2f}ms", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow("YOLO TensorRT Real-time Detection", annotated_frame)

    # 打印到控制台
    print(f"推理耗时: {latency:.2f} ms | FPS: {1000/latency:.1f}")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()