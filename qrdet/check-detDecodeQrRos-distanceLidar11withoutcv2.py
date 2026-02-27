#!/usr/bin/env python3
"""
FilePath: /ultralytics/qrdet/check-detDecodeQrRos-distanceLidar11withoutcv2.py
author: wupke
Date: 2026-02-10 14:21:47
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-02-13 09:14:15
Description:
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time

import cv2
import message_filters
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from camera_node.msg import StereoImage
from std_msgs.msg import String

EMA_ALPHA = 0.25


class QRPerceptionNode:
    def __init__(self):
        rospy.init_node("qr_perception_node")

        self.detector = cv2.QRCodeDetector()

        # ========= Camera intrinsics =========
        self.K = np.array([[809.0, 0, 471.0], [0, 808.0, 355.0], [0, 0, 1.0]], np.float32)

        self.D = np.array([-0.1397, 0.0121, 0.00069, -0.00011, -0.00042], np.float32)

        # ========= LiDAR → OpenCV Camera =========
        self.R_lidar2cam = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], np.float32)

        self.t_lidar2cam = np.array([[0.01, 0.0, 0.075]], np.float32)

        # ========= Runtime =========
        self.last_time = time.time()
        self.filtered_dist = None
        self.roi_pc_buffer = []
        self.buffer_size = 3

        # ========= ROS IO =========
        img_sub = message_filters.Subscriber("/camera/stereo_image", StereoImage)
        pc_sub = message_filters.Subscriber("/livox/lidar", pc2.PointCloud2)

        ts = message_filters.ApproximateTimeSynchronizer([img_sub, pc_sub], 10, 0.05)
        ts.registerCallback(self.callback)

        self.pub_result = rospy.Publisher("/qr_perception/result", String, queue_size=10)

        # 移除窗口初始化
        rospy.loginfo("🚀 QR Perception Node Started (Headless Mode)")

    def rosimg_to_cv(self, msg):
        # 优化：只转换图像数据，不进行 copy 可能更快（取决于后续是否修改原图）
        h, w, step = msg.height, msg.width, msg.step
        return np.frombuffer(msg.data, np.uint8).reshape(h, step)[:, : w * 3].reshape(h, w, 3)

    def ema_filter(self, dist):
        if self.filtered_dist is None:
            self.filtered_dist = dist
        else:
            self.filtered_dist = EMA_ALPHA * dist + (1 - EMA_ALPHA) * self.filtered_dist
        return self.filtered_dist

    def publish_result(self, valid, stamp, qr_id=None, map_x=None, map_y=None, turn=None, yaw=None, dist=None):
        msg = {
            "valid": valid,
            "id": qr_id,
            "map_x": map_x,
            "map_y": map_y,
            "turn": turn,
            "yaw": yaw,
            "distance": dist,
            "stamp": stamp,
        }
        self.pub_result.publish(json.dumps(msg))

    def callback(self, img_msg, pc_msg):
        # 记录开始时间用于计算 FPS
        time.time()
        frame = self.rosimg_to_cv(img_msg.rgb_image)
        stamp = img_msg.header.stamp.to_sec()

        # ========= QR detect =========
        ok, bbox = self.detector.detect(frame)
        if not ok or bbox is None:
            self.publish_result(valid=False, stamp=stamp)
            self._log_fps()
            return

        pts_qr = bbox[0].astype(np.float32)
        if pts_qr.shape != (4, 2) or cv2.contourArea(pts_qr) < 10.0:
            self.publish_result(valid=False, stamp=stamp)
            self._log_fps()
            return

        data, _ = self.detector.decode(frame, bbox)
        if not data:
            self.publish_result(valid=False, stamp=stamp)
            self._log_fps()
            return

        # ========= ROI & Point Cloud (与原逻辑一致) =========
        x1, y1 = pts_qr.min(axis=0)
        x2, y2 = pts_qr.max(axis=0)
        w, h = x2 - x1, y2 - y1
        rx1, ry1 = x1 + 0.2 * w, y1 + 0.2 * h
        rx2, ry2 = x1 + 0.8 * w, y1 + 0.8 * h

        pts = np.array(list(pc2.read_points(pc_msg, ("x", "y", "z"), skip_nans=True)), np.float32)
        if pts.shape[0] < 20:
            self.publish_result(False, stamp)
            return

        mask = (pts[:, 0] > 0.1) & (pts[:, 0] < 5.0) & (np.abs(pts[:, 1]) < 3.0) & (np.abs(pts[:, 2]) < 2.0)
        pts = pts[mask]

        pts_cam = (pts @ self.R_lidar2cam.T) + self.t_lidar2cam
        mask_front = pts_cam[:, 2] > 0.1
        pts_cam = pts_cam[mask_front]
        pts_lidar = pts[mask_front]

        if pts_cam.shape[0] < 10:
            self.publish_result(False, stamp)
            return

        pts2d, _ = cv2.projectPoints(pts_cam, np.zeros(3), np.zeros(3), self.K, self.D)
        pts2d = pts2d.reshape(-1, 2)

        mask_roi = (pts2d[:, 0] >= rx1) & (pts2d[:, 0] <= rx2) & (pts2d[:, 1] >= ry1) & (pts2d[:, 1] <= ry2)
        roi_pts = pts_lidar[mask_roi]

        if roi_pts.shape[0] == 0:
            self.publish_result(False, stamp)
            return

        self.roi_pc_buffer.append(roi_pts)
        if len(self.roi_pc_buffer) > self.buffer_size:
            self.roi_pc_buffer.pop(0)

        roi_all = np.vstack(self.roi_pc_buffer)
        lidar_dist = self.ema_filter(np.median(np.linalg.norm(roi_all, axis=1)))

        # ========= Parse QR & Publish =========
        qr_id, map_x, map_y, qr_turn, yaw = None, None, None, None, None
        try:
            info = json.loads(data)
            qr_id, map_x, map_y, qr_turn, yaw = (
                info.get("id"),
                info.get("x"),
                info.get("y"),
                info.get("turn"),
                info.get("yaw"),
            )
        except:
            pass

        self.publish_result(True, stamp, qr_id, map_x, map_y, qr_turn, yaw, round(float(lidar_dist), 3))

        # 仅在终端打印结果
        rospy.loginfo(f"📦 ID:{qr_id} | Dist:{lidar_dist:.2f}m")
        self._log_fps()

    def _log_fps(self):
        fps = 1.0 / max(1e-6, time.time() - self.last_time)
        self.last_time = time.time()
        # 将 FPS 打印在终端，而不是绘制
        rospy.loginfo_throttle(1.0, f"Processing at {fps:.1f} FPS")


if __name__ == "__main__":
    try:
        QRPerceptionNode()
        rospy.spin()  # 使用标准的阻塞
    except rospy.ROSInterruptException:
        pass

    # ========= 新增：ROI 多尺度 decode =========

    def decode_with_roi_retry(self, frame, pts_qr):
        x1, y1 = pts_qr.min(axis=0)
        x2, y2 = pts_qr.max(axis=0)
        w, h = x2 - x1, y2 - y1
        scales = [0.2, 0.4, 0.6, 0.8]
        for i in range(len(scales) - 1):
            rx1, ry1 = x1 + scales[i] * w, y1 + scales[i] * h
            rx2, ry2 = x1 + scales[i + 1] * w, y1 + scales[i + 1] * h
            roi = frame[int(ry1) : int(ry2), int(rx1) : int(rx2)]
            data, _ = self.detector.decode(roi)
            if data:
                return data
        return None

    def callback(self, img_msg, pc_msg):
        # 记录开始时间用于计算 FPS
        time.time()
        frame = self.rosimg_to_cv(img_msg.rgb_image)
        stamp = img_msg.header.stamp.to_sec()

        # ========= QR detect =========
        ok, bbox = self.detector.detect(frame)
        if not ok or bbox is None:
            self.publish_result(valid=False, stamp=stamp)
            self._log_fps()
            return
