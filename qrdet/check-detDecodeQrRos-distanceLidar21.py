#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
FilePath: /ultralytics/qrdet/check-detDecodeQrRos-distanceLidar21.py
author: wupke
Date: 2026-03-27 14:31:48
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-03-27 15:00:54
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
'''


''' 在 11 版本上 进一步优化：

将当前的 cv2.QRCodeDetector 替换为 YOLO 目标检测模型（best_with65.pt），
核心逻辑需要从“盲找二维码”变为“先定位 Box，再在 Box 内解析内容”。
由于 YOLO 推理通常比简单的 OpenCV 算子耗时，我们需要引入 ultralytics 库，并优化点云映射的逻辑。

------------ 修改思路 : 
加载模型：在 __init__ 中初始化 YOLO 模型。

目标检测：每帧图像先跑一次 YOLO，获取二维码的 Bounding Box。

内容解析：将 Box 区域裁剪或直接传递给 cv2.QRCodeDetector 进行 decode。

点云映射：利用 YOLO 提供的更精确的 Box 坐标来定义 ROI，提取激光点云深度。

距离计算：根据激光点云深度和相机内参，计算二维码到相机的距离。

'''

# ----  缺点： 当前代码逻辑只能处理一个二维码，如果场景中存在多个二维码，可能需要增加一个循环来处理每个检测到的 Box，并且需要设计一个机制来关联每个 Box 的内容和对应的距离信息（比如通过中心点距离或其他特征进行匹配）。  --- 当前版本仅处理置信度最高的一个二维码，后续可以考虑增加多目标处理逻辑。

# 如果场景中有两个二维码（A 和 B）：

# 跳变风险：如果 A 的置信度是 0.85，B 是 0.84，当前帧输出 A 的坐标和距离；
# 下一帧如果 B 变成 0.86，输出就会瞬间切换到 B。

# 数据错位风险：对于自动驾驶路径规划或定位来说，这种 ID 和距离的突然跳变会导致下游算法（比如轨迹追踪）出现抖动。



###  代码未测试

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy, cv2, json, numpy as np, message_filters, threading, time
import sensor_msgs.point_cloud2 as pc2
from camera_node.msg import StereoImage
from std_msgs.msg import String
from ultralytics import YOLO  # 需安装 ultralytics

EMA_ALPHA = 0.25

class QRPerceptionNode:
    def __init__(self):
        rospy.init_node("qr_perception_node")

        # 1. 加载 YOLO 模型 (建议放在工程的 weights 目录下)
        self.model = YOLO("best_with65.pt") 
        self.qr_decoder = cv2.QRCodeDetector() # 仅用于解析内容

        # ========= 相机与外参 (保持不变) =========
        self.K = np.array([[809.0, 0, 471.0], [0, 808.0, 355.0], [0, 0, 1.0]], np.float32)
        self.D = np.array([-0.1397, 0.0121, 0.00069, -0.00011, -0.00042], np.float32)
        self.R_lidar2cam = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], np.float32)
        self.t_lidar2cam = np.array([[0.01, 0.0, 0.075]], np.float32)

        # ========= 状态变量 =========
        self.filtered_dist = None
        self.last_time = time.time()
        
        # ========= ROS 通信 =========
        img_sub = message_filters.Subscriber("/camera/stereo_image", StereoImage)
        pc_sub = message_filters.Subscriber("/livox/lidar", pc2.PointCloud2)
        
        # 考虑到 YOLO 耗时，同步容差可适当放宽至 0.05-0.1
        ts = message_filters.ApproximateTimeSynchronizer([img_sub, pc_sub], 10, 0.08)
        ts.registerCallback(self.callback)

        self.pub_result = rospy.Publisher("/qr_perception/result", String, queue_size=10)
        rospy.loginfo("🚀 YOLO QR Perception Node Ready")

    def callback(self, img_msg, pc_msg):
        frame = self.rosimg_to_cv(img_msg.rgb_image)
        stamp = img_msg.header.stamp.to_sec()

        # 2. YOLO 推理 (设定置信度阈值)
        results = self.model.predict(frame, conf=0.5, verbose=False)
        
        # 寻找置信度最高的一个二维码
        best_box = None
        for r in results:
            if len(r.boxes) > 0:
                # 获取最高置信度的索引
                idx = r.boxes.conf.argmax()
                best_box = r.boxes.xyxy[idx].cpu().numpy() # [x1, y1, x2, y2]
                break

        if best_box is None:
            self.publish_result(valid=False, stamp=stamp)
            return

        x1, y1, x2, y2 = map(int, best_box)
        
        # 3. 在 YOLO Box 内解析二维码内容
        # 稍微扩大一点点区域有助于解析边缘
        crop_img = frame[max(0, y1-5):min(frame.shape[0], y2+5), 
                         max(0, x1-5):min(frame.shape[1], x2+5)]
        
        data, _, _ = self.qr_decoder.decodeCurved(crop_img) # decodeCurved 对畸变适应性好一点
        if not data:
            # 如果 decodeCurved 失败，尝试普通 decode
            data, _ = self.qr_decoder.decode(crop_img, np.array([[0,0],[0,0],[0,0],[0,0]]))

        # 4. 点云处理 (基于 YOLO 的 Box 提取距离)
        # 定义更精准的中心 ROI (利用 YOLO 的框)
        bw, bh = x2 - x1, y2 - y1
        rx1, ry1 = x1 + 0.3 * bw, y1 + 0.3 * bh
        rx2, ry2 = x1 + 0.7 * bw, y1 + 0.7 * bh

        pts = np.array(list(pc2.read_points(pc_msg, ("x", "y", "z"), skip_nans=True)), np.float32)
        
        # 这里的过滤条件可以根据实际安装高度微调
        mask = (pts[:, 0] > 0.1) & (pts[:, 0] < 6.0) 
        pts = pts[mask]

        # 投影到图像平面
        pts_cam = (pts @ self.R_lidar2cam.T) + self.t_lidar2cam
        pts_cam = pts_cam[pts_cam[:, 2] > 0.1]
        pts_lidar = pts[pts_cam[:, 2] > 0.1] # 同步保留原始点云用于计算真实距离

        pts2d, _ = cv2.projectPoints(pts_cam, np.zeros(3), np.zeros(3), self.K, self.D)
        pts2d = pts2d.reshape(-1, 2)

        # 筛选落在 YOLO ROI 内的点
        mask_roi = (pts2d[:, 0] >= rx1) & (pts2d[:, 0] <= rx2) & \
                   (pts2d[:, 1] >= ry1) & (pts2d[:, 1] <= ry2)
        
        roi_pts = pts_lidar[mask_roi]

        if roi_pts.shape[0] > 0:
            # 取中值距离并进行平滑
            raw_dist = np.median(np.linalg.norm(roi_pts, axis=1))
            final_dist = self.ema_filter(raw_dist)
        else:
            final_dist = None

        # 5. 解析 JSON 数据并发布
        self.process_and_pub(data, final_dist, stamp, (x1, y1, x2, y2), frame)

    def process_and_pub(self, data, dist, stamp, box, frame):
        qr_info = {"id": None, "x": None, "y": None, "turn": None, "yaw": None}
        if data:
            try:
                qr_info.update(json.loads(data))
            except:
                pass

        valid = True if (data and dist) else False
        
        msg = {
            "valid": valid,
            "id": qr_info["id"],
            "map_x": qr_info["x"],
            "map_y": qr_info["y"],
            "turn": qr_info["turn"],
            "yaw": qr_info["yaw"],
            "distance": round(float(dist), 3) if dist else 0,
            "stamp": stamp
        }
        self.pub_result.publish(json.dumps(msg))

        # 可视化检测框
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        if dist:
            cv2.putText(frame, f"Dist: {dist:.2f}m", (box[0], box[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 辅助函数保持逻辑一致
    def rosimg_to_cv(self, msg):
        h, w = msg.height, msg.width
        return np.frombuffer(msg.data, np.uint8).reshape(h, w, 3).copy()

    def ema_filter(self, dist):
        if self.filtered_dist is None: self.filtered_dist = dist
        else: self.filtered_dist = EMA_ALPHA * dist + (1 - EMA_ALPHA) * self.filtered_dist
        return self.filtered_dist

    def publish_result(self, valid, stamp):
        self.pub_result.publish(json.dumps({"valid": valid, "stamp": stamp}))

    def display_loop(self):
        rospy.spin()

if __name__ == "__main__":
    QRPerceptionNode().display_loop()


