#!/usr/bin/env python3
"""
FilePath: /ultralytics/ultralytics/test_predict.py
author: wupke
Date: 2026-01-28 10:26:33
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-01-30 17:36:15
Description:
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
"""


######  ---------------------------------------------- 检测视频流（摄像头） ------- ######

import cv2

from ultralytics import YOLO

# 1. 加载模型
model = YOLO("/home/wupke/Downloads/yolo26s.pt")


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
    results = model(frame, conf=0.3)[0]  # 取第一张结果

    # 4. 将检测框画到图像
    annotated_frame = results.plot()  # ⭐ 关键：自动画框+类别+置信度

    # 5. 显示
    cv2.imshow("YOLO Detection", annotated_frame)

    # 6. 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


# ##### ------------------------------------------ 检测单张图 ------- #####


# import cv2
# from PIL import Image
# from ultralytics import YOLO

# # 1. 加载模型
# model = YOLO("/home/wupke/Downloads/yolo26s.pt")


# ##### 检测图片 #####

# img = cv2.imread("traffic_scene.png")
# results = model(img)[0]

# annotated = results.plot()

# cv2.imshow("Result", annotated)

# print(" -------------- outpus - ok ")

# cv2.waitKey(0)
# cv2.destroyAllWindows()


##### ------------------------------------------ 检测多张图文件夹 ------- #####

import os

import cv2

from ultralytics import YOLO

# ====== 配置 ======
model_path = "/home/wupke/Downloads/yolo26s.pt"
input_dir = "/home/wupke/traffic_det/save_pic"
output_dir = "/home/wupke/traffic_det/save_pic_res"

os.makedirs(output_dir, exist_ok=True)

# ====== 加载模型 ======
model = YOLO(model_path)

print("Start inference on folder...")

img_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

img_files.sort()

for idx, img_name in enumerate(img_files):
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Failed to read {img_path}")
        continue

    # 推理
    results = model(img, verbose=False)[0]

    # 画框
    annotated = results.plot()

    # 保存结果
    save_path = os.path.join(output_dir, img_name)
    cv2.imwrite(save_path, annotated)

    print(f"[{idx + 1}/{len(img_files)}] Saved: {save_path}")

print("\n✅ All images processed.")
