#!/usr/bin/env python3
"""
FilePath: /ultralytics/scripts/ros_yolo_livox.py
author: wupke
Date: 2026-02-03 17:10:47
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-02-04 09:05:17
Description:
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import cv2
import message_filters
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf.transformations as tf_trans
from sensor_msgs.msg import Image, PointCloud2

from ultralytics import YOLO


class LivoxYoloFusion:
    def __init__(self):
        rospy.init_node("livox_yolo_fusion", anonymous=True)

        # 1. 加载模型 (Orin GPU)
        self.model = YOLO("runs/detect/train_s-label2/weights/best.pt")

        # 2. 相机参数 (你提供的 K 和 D)
        self.intrinsic = np.array(
            [[818.53994414, 0.0, 289.78608567], [0.0, 818.61550357, 283.78690782], [0.0, 0.0, 1.0]]
        )
        self.dist_coeffs = np.array([0.1096357, -0.3696521, -0.0072587, 0.0002249, 0.0])

        # 3. 初始化外参 (根据你的描述估计)
        self.init_extrinsics()

        # 4. 时间同步订阅
        self.image_sub = message_filters.Subscriber("/camera/image_raw", Image)
        self.lidar_sub = message_filters.Subscriber("/livox/lidar", PointCloud2)

        # slop设为0.06s，适应Mid-360频率
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.lidar_sub], queue_size=10, slop=0.06
        )
        self.ts.registerCallback(self.callback)

        self.last_time = time.time()
        rospy.loginfo("Fusion Node Started: Logic -> Box Center Median Distance")

    def init_extrinsics(self):
        # 轴向转换矩阵: LiDAR(X,Y,Z) -> Camera(Z,-X,-Y)
        R_axes = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        # 补偿雷达倾斜 20 度
        R_pitch = tf_trans.euler_matrix(0, np.radians(-20), 0)[:3, :3]
        R_combined = R_axes @ R_pitch
        self.r_vec, _ = cv2.Rodrigues(R_combined)
        # 平移向量 (相机在雷达系坐标: 前0.05, 左-0.12, 下-0.05)
        t_lidar_in_cam = np.array([0.05, -0.12, -0.05])
        self.t_vec = R_combined @ t_lidar_in_cam

    def callback(self, img_msg, pc_msg):
        time.time()
        try:
            # A. 图像推理
            img_array = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
            results = self.model.predict(img_array, conf=0.3, device="0", verbose=False)[0]

            # B. 点云预处理 (过滤前方 20m)
            # 使用列表推导式或 generator 在 Orin 上通常比 pc2.read_points 略快
            pts_generator = pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True)
            points_3d = np.array([p for p in pts_generator if 0.2 < p[0] < 20.0])

            if points_3d.size == 0:
                return

            # C. 投影
            points_2d, _ = cv2.projectPoints(points_3d, self.r_vec, self.t_vec, self.intrinsic, self.dist_coeffs)
            points_2d = points_2d.reshape(-1, 2)

            # D. 关联与输出
            fps = 1.0 / (time.time() - self.last_time)
            self.last_time = time.time()
            log_str = f"FPS: {fps:.1f}"

            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = self.model.names[int(box.cls[0])]

                # --- 核心优化：计算中心 ROI 区域 ---
                # 取检测框中心 40% 的面积，避开边缘背景
                w, h = x2 - x1, y2 - y1
                cx1, cy1 = x1 + w * 0.3, y1 + h * 0.3
                cx2, cy2 = x1 + w * 0.7, y1 + h * 0.7

                # 筛选落入中心 ROI 的点
                mask = (
                    (points_2d[:, 0] >= cx1)
                    & (points_2d[:, 0] <= cx2)
                    & (points_2d[:, 1] >= cy1)
                    & (points_2d[:, 1] <= cy2)
                )

                roi_points_3d = points_3d[mask]

                if roi_points_3d.size > 0:
                    # 计算到雷达原点的直线距离
                    dists = np.linalg.norm(roi_points_3d, axis=1)
                    # 取中位数，最稳妥地代表目标距离
                    dist_val = np.median(dists)
                    log_str += f" | [{cls}: {dist_val:.2f}m]"
                else:
                    log_str += f" | [{cls}: no_pts]"

            # 终端覆盖输出
            print(f"\r{log_str:<120}", end="", flush=True)

        except Exception as e:
            rospy.logerr(f"Fusion Error: {e}")


if __name__ == "__main__":
    LivoxYoloFusion()
    rospy.spin()


##############    ---- 初版 Livox + YOLOv8 距离融合节点 （适用于 Mid-360） ----    ##############


# #!/usr/bin/env python3
# import rospy
# import message_filters
# from sensor_msgs.msg import Image, PointCloud2
# import sensor_msgs.point_cloud2 as pc2
# import numpy as np
# import cv2
# from ultralytics import YOLO

# class LivoxYoloFusion:
#     def __init__(self):
#         rospy.init_node('livox_yolo_fusion', anonymous=True)

#         # 1. 加载 YOLO 模型 (Orin GPU)
#         self.model = YOLO("runs/detect/train_s-label2/weights/best.pt")

#         # 2. 相机内参 (需替换为您实测的标定参数)
#         self.intrinsic = np.array([
#             [605.12, 0, 320.5],
#             [0, 604.98, 240.2],
#             [0, 0, 1]
#         ])

#         # 3. 外参：Livox 到 相机 的变换 (需替换为您标定的 R 和 T)
#         # rvec: 旋转向量, tvec: 平移向量
#         self.r_vec = np.array([-1.57, 0.0, -1.57]) # 示例值：假设雷达在相机上方
#         self.t_vec = np.array([0.0, -0.05, 0.1])   # 示例值：单位为米
#         self.dist_coeffs = np.zeros(5)             # 假设已做去畸变处理

#         # 4. 时间同步订阅 (图像 + Mid-360 点云)
#         self.image_sub = message_filters.Subscriber("/camera/image_raw", Image)
#         self.lidar_sub = message_filters.Subscriber("/livox/lidar", PointCloud2) # 或 /livox/points

#         # slop 设置为 0.05s 左右，Mid-360 频率通常为 10Hz
#         self.ts = message_filters.ApproximateTimeSynchronizer(
#             [self.image_sub, self.lidar_sub], queue_size=5, slop=0.05
#         )
#         self.ts.registerCallback(self.sync_callback)

#         rospy.loginfo("Livox-YOLO Fusion Node Ready.")

#     def sync_callback(self, img_msg, pc_msg):
#         # --- A. 图像推理 ---
#         img_array = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
#         # 针对 ROS rgb8/bgr8 可能需要转换
#         # cv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
#         results = self.model.predict(img_array, conf=0.3, device='0', verbose=False)[0]

#         # --- B. 点云处理与投影 ---
#         # 1. 提取点云 XYZ
#         raw_points = np.array(list(pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True)))

#         # 2. 空间裁剪：Mid-360 是 360 度的，只保留相机前方区域（例如 X > 0）
#         # 注意：这里的 X 是雷达坐标系方向，需根据安装位置调整
#         front_mask = raw_points[:, 0] > 0.5
#         points_3d = raw_points[front_mask]

#         if len(points_3d) == 0:
#             return

#         # 3. 投影 3D 点到 2D 像素平面
#         points_2d, _ = cv2.projectPoints(points_3d, self.r_vec, self.t_vec, self.intrinsic, self.dist_coeffs)
#         points_2d = points_2d.reshape(-1, 2)

#         # --- C. 关联检测结果 ---
#         log_str = ""
#         for box in results.boxes:
#             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#             label = self.model.names[int(box.cls[0])]

#             # 找到落在 2D 框内的点云索引
#             in_box_mask = (points_2d[:, 0] >= x1) & (points_2d[:, 0] <= x2) & \
#                           (points_2d[:, 1] >= y1) & (points_2d[:, 1] <= y2)

#             box_points_3d = points_3d[in_box_mask]

#             if len(box_points_3d) > 0:
#                 # 计算距离：欧几里得距离的第 20 分位数（取较近的点，排除背景）
#                 dists = np.linalg.norm(box_points_3d, axis=1)
#                 distance = np.percentile(dists, 20)
#                 log_str += f" | {label}: {distance:.2f}m"
#             else:
#                 log_str += f" | {label}: dist unknown"

#         # 终端输出
#         print(f"\r{log_str:<120}", end="")
