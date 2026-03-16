#!/usr/bin/env python3
"""
FilePath: /ultralytics/qrdet/check-detDecodeQrRos-distanceLidar7.py
author: wupke
Date: 2026-02-06 11:25:06
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-02-06 12:14:07
Description:
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
"""

"""
将版本6代码实时识别到的二维码信息与距离，如ID:1 Turn:straight 📏 1.11 m发布成标准的rostopic提供给下游使用

- 发布话题： /qr_perception/result

- 消息类型： std_msgs/String
- 消息内容： JSON格式字符串，包含二维码ID、转向指令、距离和时间戳，例如：
{
  "id": 1,
  "turn": "straight",
  "distance": 1.11,
  "stamp": 1700000000.123
}


-----------  下游如何用（示例）:

rostopic echo /qr_perception/result


import json
def cb(msg):
    data = json.loads(msg.data)
    print(data["id"], data["turn"], data["distance"], data["stamp"])




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
QR_SIZE = 0.17
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

        # ========= ROS IO =========
        img_sub = message_filters.Subscriber("/camera/stereo_image", StereoImage)
        pc_sub = message_filters.Subscriber("/livox/lidar", pc2.PointCloud2)

        ts = message_filters.ApproximateTimeSynchronizer([img_sub, pc_sub], 10, 0.05)
        ts.registerCallback(self.callback)

        # ⭐ 发布二维码感知结果
        self.pub_result = rospy.Publisher("/qr_perception/result", String, queue_size=10)

        cv2.namedWindow("QR_Perception", cv2.WINDOW_NORMAL)
        rospy.loginfo("🚀 QR Perception Node Started (Publish Enabled)")

    def rosimg_to_cv(self, msg):
        h, w, step = msg.height, msg.width, msg.step
        return np.frombuffer(msg.data, np.uint8).reshape(h, step)[:, : w * 3].reshape(h, w, 3).copy()

    def ema_filter(self, dist):
        if self.filtered_dist is None:
            self.filtered_dist = dist
        else:
            self.filtered_dist = EMA_ALPHA * dist + (1 - EMA_ALPHA) * self.filtered_dist
        return self.filtered_dist

    def callback(self, img_msg, pc_msg):
        frame = self.rosimg_to_cv(img_msg.rgb_image)

        # ========= QR detect =========
        data, bbox, _ = self.detector.detectAndDecode(frame)
        if not (data and bbox is not None):
            self._update_fps(frame)
            return

        pts_qr = bbox[0].astype(np.float32)
        cv2.polylines(frame, [pts_qr.astype(int)], True, (0, 255, 0), 2)
        cx, cy = pts_qr.mean(axis=0)

        # ========= ROI =========
        x1, y1 = pts_qr.min(axis=0)
        x2, y2 = pts_qr.max(axis=0)
        w, h = x2 - x1, y2 - y1
        rx1, ry1 = x1 + 0.2 * w, y1 + 0.2 * h
        rx2, ry2 = x1 + 0.8 * w, y1 + 0.8 * h

        # ========= Point cloud =========
        pts = np.array(list(pc2.read_points(pc_msg, ("x", "y", "z"), skip_nans=True)), np.float32)

        if pts.shape[0] < 20:
            return

        mask = (pts[:, 0] > 0.1) & (pts[:, 0] < 5.0) & (np.abs(pts[:, 1]) < 3.0) & (np.abs(pts[:, 2]) < 2.0)
        pts = pts[mask]

        pts_cam = (pts @ self.R_lidar2cam.T) + self.t_lidar2cam
        mask_front = pts_cam[:, 2] > 0.1
        pts_cam = pts_cam[mask_front]
        pts_lidar = pts[mask_front]

        if pts_cam.shape[0] < 10:
            return

        pts2d, _ = cv2.projectPoints(pts_cam, np.zeros(3), np.zeros(3), self.K, self.D)
        pts2d = pts2d.reshape(-1, 2)

        mask_roi = (pts2d[:, 0] >= rx1) & (pts2d[:, 0] <= rx2) & (pts2d[:, 1] >= ry1) & (pts2d[:, 1] <= ry2)

        roi_pts = pts_lidar[mask_roi]
        if roi_pts.shape[0] == 0:
            return

        self.roi_pc_buffer.append(roi_pts)
        if len(self.roi_pc_buffer) > self.buffer_size:
            self.roi_pc_buffer.pop(0)

        roi_all = np.vstack(self.roi_pc_buffer)
        lidar_dist = self.ema_filter(np.median(np.linalg.norm(roi_all, axis=1)))

        # ========= Parse QR =========
        qr_id = None
        qr_turn = None
        try:
            info = json.loads(data)
            qr_id = info.get("id", None)
            qr_turn = info.get("turn", None)
        except:
            pass

        # ========= Publish =========
        result = {
            "id": qr_id,
            "turn": qr_turn,
            "distance": round(float(lidar_dist), 3),
            "stamp": rospy.Time.now().to_sec(),
        }
        self.pub_result.publish(json.dumps(result))

        # ========= Overlay =========
        text = f"ID:{qr_id} Turn:{qr_turn}"
        cv2.putText(frame, text, (int(cx) - 120, int(cy) - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(
            frame, f"{lidar_dist:.2f} m", (int(cx) - 60, int(cy) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
        )

        rospy.loginfo(f"📦 {text}  📏 {lidar_dist:.2f} m")

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
