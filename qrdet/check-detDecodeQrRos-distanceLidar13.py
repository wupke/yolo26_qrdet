#!/usr/bin/env python3
"""
FilePath: /ultralytics/qrdet/check-detDecodeQrRos-distanceLidar13.py
author: wupke
Date: 2026-02-13 10:34:45
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-02-13 10:36:13
Description:
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
"""

"""

check-detDecodeQrRos-distanceLidar11withoutcv2.py 的升级版：

无窗口，输出距离数据到控制台，验证 LiDAR 距离测量的稳定性和准确性



要输出对应的 $(x, y, z)$ 坐标，需要从关联到的 roi_pts（雷达坐标系下的点云）中计算出中心位置。
最稳健的方法是对这些点的坐标分别取中位数（Median），这样可以有效排除二维码边缘或背景的离群点，
确保得到的 3D 坐标正好落在二维码的平面中心附近。
修改后的核心逻辑坐标计算：
从 roi_all（缓存后的 ROI 点云）中提取坐标中位数：target_3d = np.median(roi_all, axis=0)。
数据下发：将 x, y, z 加入到 JSON 输出消息中。

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

        rospy.loginfo("🚀 QR Perception Node Started (Output XYZ included)")

    def rosimg_to_cv(self, msg):
        h, w, step = msg.height, msg.width, msg.step
        return np.frombuffer(msg.data, np.uint8).reshape(h, step)[:, : w * 3].reshape(h, w, 3)

    def ema_filter(self, dist):
        if self.filtered_dist is None:
            self.filtered_dist = dist
        else:
            self.filtered_dist = EMA_ALPHA * dist + (1 - EMA_ALPHA) * self.filtered_dist
        return self.filtered_dist

    # 修改 publish_result，增加 x, y, z 参数
    def publish_result(
        self, valid, stamp, qr_id=None, map_x=None, map_y=None, turn=None, yaw=None, dist=None, pos_xyz=None
    ):
        msg = {
            "valid": valid,
            "id": qr_id,
            "map_x": map_x,
            "map_y": map_y,
            "turn": turn,
            "yaw": yaw,
            "distance": dist,
            "pos_lidar": pos_xyz,  # 这里的 xyz 是相对于雷达的坐标
            "stamp": stamp,
        }
        self.pub_result.publish(json.dumps(msg))

    def callback(self, img_msg, pc_msg):
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

        # ========= ROI & Point Cloud =========
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

        # --- 计算中心 3D 坐标 ---
        # 对 ROI 内的所有点求中位数，得到相对于雷达坐标系的 (x, y, z)
        target_xyz = np.median(roi_all, axis=0)

        # 计算欧氏距离
        raw_dist = np.linalg.norm(target_xyz)
        lidar_dist = self.ema_filter(raw_dist)

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

        # 将坐标打包输出，保留3位小数
        pos_output = [round(float(c), 3) for c in target_xyz]

        self.publish_result(
            valid=True,
            stamp=stamp,
            qr_id=qr_id,
            map_x=map_x,
            map_y=map_y,
            turn=qr_turn,
            yaw=yaw,
            dist=round(float(lidar_dist), 3),
            pos_xyz=pos_output,
        )

        rospy.loginfo(f"📦 ID:{qr_id} | Dist:{lidar_dist:.2f}m | XYZ:{pos_output}")
        self._log_fps()

    def _log_fps(self):
        fps = 1.0 / max(1e-6, time.time() - self.last_time)
        self.last_time = time.time()
        rospy.loginfo_throttle(1.0, f"Processing at {fps:.1f} FPS")


if __name__ == "__main__":
    try:
        QRPerceptionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
