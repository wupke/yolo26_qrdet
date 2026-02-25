#!/usr/bin/env python3
"""
FilePath: /ultralytics/qrdet/check-detDecodeQrRos-distanceLidar12.py
author: wupke
Date: 2026-02-09 16:06:05
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-02-09 16:58:11
Description:
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
"""

"""

仿照手机，从远处拉近二维码，再扫描，增大识别的距离


手机“远处看到框 → 拉近后识别”的能力，本质是：

“先低成本 detect（找可能是二维码的地方），再对该区域做连续跟踪 + 多尺度解码”



-------------------  下游使用的正确姿势  : -------------------

def qr_callback(msg):
    data = json.loads(msg.data)

    if not data["valid"]:
        return  # 忽略

    # 这里的数据 = 当前有效二维码信息
    qr_id = data["id"]
    map_x = data["map_x"]
    map_y = data["map_y"]
    turn = data["turn"]
    yaw = data["yaw"]
    dist = data["distance"]


"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import threading
import time

import cv2
import message_filters
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from camera_node.msg import StereoImage
from std_msgs.msg import String

DEBUG_PROJECTION = False
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
        self.lock = threading.Lock()
        self.latest_frame = np.zeros((720, 960, 3), np.uint8)
        self.last_time = time.time()
        self.filtered_dist = None

        self.roi_pc_buffer = []
        self.buffer_size = 3

        # ⭐ 手机扫码级别：跨帧 QR bbox 缓存
        self.last_qr_bbox = None
        self.last_qr_time = 0
        self.bbox_timeout = 0.6  # 秒

        # ========= ROS IO =========
        img_sub = message_filters.Subscriber("/camera/stereo_image", StereoImage)
        pc_sub = message_filters.Subscriber("/livox/lidar", pc2.PointCloud2)

        ts = message_filters.ApproximateTimeSynchronizer([img_sub, pc_sub], 10, 0.05)
        ts.registerCallback(self.callback)

        self.pub_result = rospy.Publisher("/qr_perception/result", String, queue_size=10)

        cv2.namedWindow("QR_Perception", cv2.WINDOW_NORMAL)
        rospy.loginfo("🚀 QR Perception Node Started (ROI multi-scale decode enabled)")

    def rosimg_to_cv(self, msg):
        h, w, step = msg.height, msg.width, msg.step
        return np.frombuffer(msg.data, np.uint8).reshape(h, step)[:, : w * 3].reshape(h, w, 3).copy()

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

    # ========= 新增：ROI 多尺度 decode =========
    def decode_with_roi_retry(self, frame, pts_qr):
        x1, y1 = np.min(pts_qr, axis=0).astype(int)
        x2, y2 = np.max(pts_qr, axis=0).astype(int)

        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        for scale in [1.5, 2.0, 3.0]:
            roi_up = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            data, _ = self.detector.detectAndDecode(roi_up)
            if data:
                return data
        return None

    def callback(self, img_msg, pc_msg):
        frame = self.rosimg_to_cv(img_msg.rgb_image)
        stamp = img_msg.header.stamp.to_sec()

        data = None
        pts_qr = None

        # ========= 1️⃣ detect =========
        ok, bbox = self.detector.detect(frame)

        if ok and bbox is not None:
            pts_qr = bbox[0].astype(np.float32)
            if pts_qr.shape == (4, 2) and cv2.contourArea(pts_qr) > 10:
                self.last_qr_bbox = pts_qr
                self.last_qr_time = time.time()

                # 先尝试原图 decode
                data, _ = self.detector.decode(frame, bbox)
                if not data:
                    data = self.decode_with_roi_retry(frame, pts_qr)

        # ========= 2️⃣ detect 失败，用历史 bbox =========
        if data is None:
            if self.last_qr_bbox is not None:
                if time.time() - self.last_qr_time < self.bbox_timeout:
                    pts_qr = self.last_qr_bbox
                    data = self.decode_with_roi_retry(frame, pts_qr)

        if data is None or pts_qr is None:
            self.publish_result(valid=False, stamp=stamp)
            self._update_fps(frame)
            return

        # ========= Overlay QR =========
        cv2.polylines(frame, [pts_qr.astype(int)], True, (0, 255, 0), 2)
        _cx, _cy = pts_qr.mean(axis=0)

        # ========= ROI =========
        x1, y1 = pts_qr.min(axis=0)
        x2, y2 = pts_qr.max(axis=0)
        w, h = x2 - x1, y2 - y1
        rx1, ry1 = x1 + 0.2 * w, y1 + 0.2 * h
        rx2, ry2 = x1 + 0.8 * w, y1 + 0.8 * h

        # ========= Point cloud =========
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

        # ========= Parse QR =========
        qr_id = map_x = map_y = qr_turn = yaw = None
        try:
            info = json.loads(data)
            qr_id = info.get("id")
            map_x = info.get("x")
            map_y = info.get("y")
            qr_turn = info.get("turn")
            yaw = info.get("yaw")
        except:
            pass

        # ========= Publish =========
        self.publish_result(
            valid=True,
            stamp=stamp,
            qr_id=qr_id,
            map_x=map_x,
            map_y=map_y,
            turn=qr_turn,
            yaw=yaw,
            dist=round(float(lidar_dist), 3),
        )

        self._update_fps(frame)

    def _update_fps(self, frame):
        fps = 1.0 / max(1e-6, time.time() - self.last_time)
        self.last_time = time.time()
        cv2.putText(frame, f"FPS:{fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        with self.lock:
            self.latest_frame = frame

    def display_loop(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            with self.lock:
                cv2.imshow("QR_Perception", self.latest_frame)
            cv2.waitKey(1)
            rate.sleep()


if __name__ == "__main__":
    QRPerceptionNode().display_loop()
