#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
FilePath: /ultralytics/scripts/interferencePT.py
author: wupke
Date: 2026-02-02 17:56:34
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-02-03 11:17:55
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
'''

######  ---------------------------------------------- 检测视频流（摄像头） ------- ######

import cv2
from PIL import Image
from ultralytics import YOLO

# 1. 加载模型
# model = YOLO("runs/detect/train6_100epoch/weights/best.pt")
model = YOLO("runs/detect/train_s-label2/weights/best.pt")

##### 检测视频流（摄像头） #####
# 2. 打开摄像头（0=默认摄像头）
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open failed")
    exit()

print("Start detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. YOLO 推理
    results = model(frame, conf=0.3)[0]   # 取第一张结果

    # 4. 将检测框画到图像
    annotated_frame = results.plot()  # ⭐ 关键：自动画框+类别+置信度

    # 5. 显示
    cv2.imshow("YOLO Detection", annotated_frame)

    # 6. 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()








# ##### ---------------------------------other version  ------- ######
# from ultralytics import YOLO
# import cv2
# import os

# # === 1. 配置路径 ===
# model_path = 'runs/detect/train6/weights/best.pt' # 刚才生成的最佳权重
# video_path = 'path/to/your/test_video.mp4'      # 待检测视频路径
# output_path = 'runs/detect/inference_result.mp4' # 结果保存路径

# # === 2. 加载模型 ===
# model = YOLO(model_path)

# # === 3. 执行推理 ===
# # stream=True 使用生成器模式，适合长视频，节省内存
# results = model.predict(
#     source=video_path,
#     conf=0.5,         # 置信度阈值，低于0.5的框不显示
#     iou=0.45,         # NMS IOU阈值
#     save=True,        # 自动保存带框的视频片段
#     save_txt=False,   # 是否保存检测结果的txt
#     show=True,        # 运行过程中实时显示画面
#     device='0'        # 使用 GPU 0
# )

# # === 4. 获取保存后的路径 (Ultralytics 默认保存在 runs/detect/predictX) ===
# print(f"✅ 推理完成！")
# print(f"📺 结果已保存至: {model.predictor.save_dir}")

# # --- 进阶：如果你想手动控制保存逻辑或处理每一帧 ---
# """
# # 示例：手动处理帧
# cap = cv2.VideoCapture(video_path)
# while cap.isOpened():
#     success, frame = cap.read()
#     if success:
#         results = model(frame)
#         annotated_frame = results[0].plot() # 绘制框图
#         cv2.imshow("YOLO26 Inference", annotated_frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         break
# cap.release()
# cv2.destroyAllWindows()
# """