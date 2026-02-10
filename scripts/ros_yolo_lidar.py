#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
FilePath: /ultralytics/scripts/ros_yolo_lidar.py
author: wupke
Date: 2026-02-03 14:31:21
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-02-03 14:32:30
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
'''

'''
最终目标：实现基于YOLO的激光雷达点云目标检测，并集成到ROS系统中，实现实时检测与反馈。

- 输出如： FPS: 32.1 | [straight: 12.45m] | [right: 8.12m]


'''


import message_filters  # 用于时间同步
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2

class YoloLidarFusion:
    def __init__(self):
        # 1. 设置标定参数 (请根据你的实际标定结果修改)
        self.intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]) # 内参
        self.r_vec = np.array([...]) # 旋转向量
        self.t_vec = np.array([...]) # 平移向量
        self.dist_coeffs = np.array([...]) # 畸变参数

        # 2. 时间同步订阅器：同时获取图像和点云
        self.image_sub = message_filters.Subscriber("/camera/image_raw", Image)
        self.lidar_sub = message_filters.Subscriber("/velodyne_points", PointCloud2)
        
        # 0.1秒内的消息被视为同一帧
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.callback)

    def callback(self, img_msg, pc_msg):
        # --- 步骤 1: 解析图像并推理 ---
        img_array = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
        results = self.model.predict(img_array, conf=0.1, device='0', verbose=False)[0]

        # --- 步骤 2: 解析并投影点云 ---
        # 提取点云中的 XYZ
        points_3d = np.array(list(pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True)))
        
        # 只保留相机前方的点 (X > 0)
        points_3d = points_3d[points_3d[:, 0] > 0]

        # 利用 cv2.projectPoints 将 3D 点投影到 2D 平面
        points_2d, _ = cv2.projectPoints(
            points_3d, self.r_vec, self.t_vec, self.intrinsic, self.dist_coeffs
        )
        points_2d = points_2d.reshape(-1, 2)

        # --- 步骤 3: 关联与距离输出 ---
        log_output = f"FPS: {self.fps:.1f} | "
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() # 获取 YOLO 2D 框
            cls_name = self.model.names[int(box.cls[0])]
            
            # 在投影点中寻找位于当前 2D 框内的所有点
            mask = (points_2d[:, 0] >= x1) & (points_2d[:, 0] <= x2) & \
                   (points_2d[:, 1] >= y1) & (points_2d[:, 1] <= y2)
            
            box_points_3d = points_3d[mask]

            if len(box_points_3d) > 0:
                # 计算该目标物体的距离（通常取框内点云的中位数或均值以排除噪声）
                distances = np.linalg.norm(box_points_3d, axis=1)
                dist_to_obj = np.median(distances) 
                log_output += f"[{cls_name}: {dist_to_obj:.2f}m] "
            else:
                log_output += f"[{cls_name}: No Lidar Point] "

        print(f"\r{log_output:<120}", end="")








