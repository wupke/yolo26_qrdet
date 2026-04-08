#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
FilePath: /ultralytics/qrdet/check-detDecodeQrRos-distanceLidar20.py
author: wupke
Date: 2026-03-24 17:33:55
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-03-27 14:31:37
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.

'''


''' 相比于check-detDecodeQrRos-distanceLidar11-fix.py进行优化：

使用WeChatQRCode 替代原有的 QRCodeDetector

核心优势：
识别率极高：它对强光、反光、模糊和远距离的二维码鲁棒性极强（专门为手机扫码设计）。

格式兼容：它输出的依然是二维码内的原始字符串，完全不影响你现有的 JSON 解析逻辑（id, map_x, map_y 等都会照常读取）。

稳定性：它不会出现 convhull 这种底层断言崩溃，处理流程比原生检测器更稳。

------------------------------------------------------------------------------------------
要安装 opencv-contrib-python 并在本地下载模型文件（微信扫码是基于机器学习的，需要模型权重：

pip3 install opencv-contrib-python



需要从 OpenCV 官方仓库 下载以下 4 个文件并放在脚本同级目录下： 没有发现需要的模型 (代码废弃)

detect.prototxt

detect.caffemodel

sr.prototxt (超分辨率模型)

sr.caffemodel (超分辨率模型)


'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy, cv2, json, numpy as np, message_filters, threading, time, os
import sensor_msgs.point_cloud2 as pc2
from camera_node.msg import StereoImage
from std_msgs.msg import String

DEBUG_PROJECTION = False
EMA_ALPHA = 0.25

class QRPerceptionNode:
    def __init__(self):
        rospy.init_node("qr_perception_node")

        # ========= WeChatQRCode 初始化 =========
        # 确保模型文件路径正确，这里假设放在脚本同级目录
        model_dir = os.path.dirname(os.path.realpath(__file__))
        try:
            self.detector = cv2.wechat_qrcode_WeChatQRCode(
                os.path.join(model_dir, "detect.prototxt"),
                os.path.join(model_dir, "detect.caffemodel"),
                os.path.join(model_dir, "sr.prototxt"),
                os.path.join(model_dir, "sr.caffemodel")
            )
        except Exception as e:
            rospy.logerr(f"❌ Failed to load WeChatQRCode models: {e}")
            rospy.signal_shutdown("Model missing")

        # ========= Camera intrinsics (保持不变) =========
        self.K = np.array([[809.0, 0, 471.0], [0, 808.0, 355.0], [0, 0, 1.0]], np.float32)
        self.D = np.array([-0.1397, 0.0121, 0.00069, -0.00011, -0.00042], np.float32)

        # ========= LiDAR → OpenCV Camera (保持不变) =========
        self.R_lidar2cam = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], np.float32)
        self.t_lidar2cam = np.array([[0.01, 0.0, 0.075]], np.float32)

        # ========= Runtime =========
        self.lock = threading.Lock()
        self.latest_frame = np.zeros((720, 960, 3), np.uint8)
        self.last_time = time.time()
        self.filtered_dist = None
        self.roi_pc_buffer = []
        self.buffer_size = 1 # 实时性优先

        # ========= ROS IO =========
        img_sub = message_filters.Subscriber("/camera/stereo_image", StereoImage)
        pc_sub = message_filters.Subscriber("/livox/lidar", pc2.PointCloud2)
        ts = message_filters.ApproximateTimeSynchronizer([img_sub, pc_sub], 10, 0.05)
        ts.registerCallback(self.callback)

        self.pub_result = rospy.Publisher("/qr_perception/result", String, queue_size=10)
        rospy.loginfo("🚀 QR Perception Node Started (WeChatQRCode Mode)")

    def rosimg_to_cv(self, msg):
        h, w, step = msg.height, msg.width, msg.step
        return np.frombuffer(msg.data, np.uint8).reshape(h, step)[:, :w * 3].reshape(h, w, 3).copy()

    def ema_filter(self, dist):
        if self.filtered_dist is None: self.filtered_dist = dist
        else: self.filtered_dist = (EMA_ALPHA * dist + (1 - EMA_ALPHA) * self.filtered_dist)
        return self.filtered_dist

    def callback(self, img_msg, pc_msg):
        try:
            frame = self.rosimg_to_cv(img_msg.rgb_image)
            if frame is None or frame.size == 0: return
        except Exception as e:
            rospy.logwarn(f"Image conversion failed: {e}")
            return
        
        stamp = img_msg.header.stamp.to_sec()

        # ========= WeChatQRCode 检测与解码 (一步到位) =========
        # WeChatQRCode 返回两个值: (字符串列表, 坐标点列表)
        data_list, points_list = self.detector.detectAndDecode(frame)

        if not data_list or len(data_list) == 0:
            self.publish_result(valid=False, stamp=stamp)
            self._update_fps(frame)
            return

        # 选第一个识别到的码
        data = data_list[0]
        # WeChatQRCode 的 points 是 float32, shape 为 (N, 4, 2)
        pts_qr = points_list[0].reshape(4, 2).astype(np.float32)

        # ========= QR overlay & ROI 计算 =========
        cv2.polylines(frame, [pts_qr.astype(int)], True, (0, 255, 0), 2)
        
        x1, y1 = pts_qr.min(axis=0)
        x2, y2 = pts_qr.max(axis=0)
        w_qr, h_qr = x2 - x1, y2 - y1
        
        # 严格中心 ROI (10% 面积)
        rx1, ry1 = x1 + 0.46 * w_qr, y1 + 0.46 * h_qr
        rx2, ry2 = x1 + 0.53 * w_qr, y1 + 0.53 * h_qr

        # ========= Point cloud 处理 (优化版) =========
        pc_data = pc2.read_points(pc_msg, ("x", "y", "z"), skip_nans=True)
        pts = np.array(list(pc_data), np.float32)

        if pts.shape[0] < 20:
            self.publish_result(False, stamp); return

        # 空间剪裁提速
        mask = (pts[:, 0] > 0.1) & (pts[:, 0] < 5.0) & (np.abs(pts[:, 1]) < 3.0)
        pts = pts[mask]

        # 坐标变换 (LiDAR -> Cam)
        pts_cam = (pts @ self.R_lidar2cam.T) + self.t_lidar2cam
        mask_front = pts_cam[:, 2] > 0.1
        pts_cam = pts_cam[mask_front]
        pts_lidar = pts[mask_front]

        if pts_cam.shape[0] < 10:
            self.publish_result(False, stamp); return

        # 投影到像素坐标
        pts2d, _ = cv2.projectPoints(pts_cam, np.zeros(3), np.zeros(3), self.K, self.D)
        pts2d = pts2d.reshape(-1, 2)

        # ROI 过滤
        mask_roi = (pts2d[:, 0] >= rx1) & (pts2d[:, 0] <= rx2) & \
                   (pts2d[:, 1] >= ry1) & (pts2d[:, 1] <= ry2)

        roi_pts = pts_lidar[mask_roi]
        if roi_pts.shape[0] == 0:
            self.publish_result(False, stamp); return

        # 距离计算
        raw_dist = np.median(np.linalg.norm(roi_pts, axis=1))
        lidar_dist = self.ema_filter(raw_dist)

        # ========= 解析 QR 内 JSON =========
        qr_id, map_x, map_y, qr_turn, yaw = None, None, None, None, None
        try:
            info = json.loads(data)
            qr_id = info.get("id")
            map_x = info.get("x")
            map_y = info.get("y")
            qr_turn = info.get("turn")
            yaw = info.get("yaw")
        except Exception:
            rospy.logwarn("Failed to parse JSON from QR code")

        # ========= 发布结果 =========
        self.publish_result(
            valid=True, stamp=stamp, qr_id=qr_id,
            map_x=map_x, map_y=map_y, turn=qr_turn,
            yaw=yaw, dist=round(float(lidar_dist), 3)
        )

        rospy.loginfo(f"✅ ID:{qr_id} Dist:{lidar_dist:.2f}m")
        self._update_fps(frame)

    def publish_result(self, valid, stamp, qr_id=None, map_x=None, map_y=None, turn=None, yaw=None, dist=None):
        msg = {"valid": valid, "id": qr_id, "map_x": map_x, "map_y": map_y, "turn": turn, "yaw": yaw, "distance": dist, "stamp": stamp}
        self.pub_result.publish(json.dumps(msg))

    def _update_fps(self, frame):
        fps = 1.0 / max(1e-6, time.time() - self.last_time)
        self.last_time = time.time()
        cv2.putText(frame, f"FPS:{fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        with self.lock:
            self.latest_frame = frame

    def display_loop(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown(): rate.sleep()

if __name__ == "__main__":
    QRPerceptionNode().display_loop()