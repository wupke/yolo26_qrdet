#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
FilePath: /ultralytics/qrdet/check-detDecodeQrRos-distanceLidar15.py
author: wupke
Date: 2026-03-19 09:05:41
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-03-20 17:42:41
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
'''


# 从11版开始继续优化：

# 1. 输出距离存在抖动，随着小车靠近二维码，距离估计的误差会变大，甚至出现距离突然变远的情况。

    # 2. 添加逻辑判断：如果当前距离比上次距离更远，且两次距离差值超过一定阈值（比如0.2米），则认为当前距离是异常的，过滤掉（不发布）突变的距离值，保持输出距离的连续性和稳定性

    # 3. 看能否缩小选取二维码对应点云区域内的范围，减少干扰点云对距离估计的影响，比如只选取二维码中心点附近的一小块区域，而不是整个二维码区域内的点云

'''
核心修改逻辑：
    - 距离突变检查：对比当前 lidar_dist 与 self.filtered_dist。
    如果当前值比旧值大且差值 $> 0.2m$，则判定为检测到了背景点，丢弃该帧或沿用旧值。
    - 缩小点云 ROI：将原本 60% 的二维码宽度缩减到 20%，只取二维码最中心的点云，这样能有效避开二维码边缘（可能扫到墙面或空气）的干扰。
    - 增加距离有效性检查：只有当距离在合理下降或微弱波动时才发布结果。


-------- 测试后存在的问题：

当前的检测逻辑存在问题，如果正常检测到第一个二维码的数据与距离，后面检测到第二个二维码的时候，
它会与第一个检测的距离进行对比，此时两者的距离有可能就是超过了设定的距离阈值，从而后续会一直报错，提示：
rospy.logwarn(f"⚠️ Detect Distance Jump: {raw_dist:.2f} (prev: {self.last_valid_dist:.2f}), Ignoring...")，
需要修改这个逻辑   
    
'''





#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy, cv2, json, numpy as np, message_filters, threading, time
import sensor_msgs.point_cloud2 as pc2
from camera_node.msg import StereoImage
from std_msgs.msg import String

EMA_ALPHA = 0.25
DIST_JUMP_THRESHOLD = 0.2  # 距离跳变阈值 (米)

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

        # ========= Runtime & Filtering =========
        self.lock = threading.Lock()
        self.latest_frame = np.zeros((720, 960, 3), np.uint8)
        self.display_last_time = time.time()
        
        self.filtered_dist = None  # 经过EMA后的平滑距离
        self.last_valid_dist = None # 上一次发布的合法距离

        self.roi_pc_buffer = []
        self.buffer_size = 3

        # ========= ROS IO =========
        img_sub = message_filters.Subscriber("/camera/stereo_image", StereoImage)
        pc_sub = message_filters.Subscriber("/livox/lidar", pc2.PointCloud2)
        ts = message_filters.ApproximateTimeSynchronizer([img_sub, pc_sub], 10, 0.05)
        ts.registerCallback(self.callback)

        self.pub_result = rospy.Publisher("/qr_perception/result", String, queue_size=10)
        rospy.loginfo("🚀 QR Perception Node Started (Stability Patch Applied)")

    def rosimg_to_cv(self, msg):
        h, w, step = msg.height, msg.width, msg.step
        return np.frombuffer(msg.data, np.uint8).reshape(h, step)[:, :w * 3].reshape(h, w, 3).copy()

    def ema_filter(self, dist):
        if self.filtered_dist is None:
            self.filtered_dist = dist
        else:
            self.filtered_dist = EMA_ALPHA * dist + (1 - EMA_ALPHA) * self.filtered_dist
        return self.filtered_dist

    def callback(self, img_msg, pc_msg):
        # 将ROS图像消息转换为OpenCV格式
        frame = self.rosimg_to_cv(img_msg.rgb_image)
        # 获取时间戳
        stamp = img_msg.header.stamp.to_sec()

        # 1. QR 检测
        ok, bbox = self.detector.detect(frame)
        if not ok or bbox is None:
            self._update_display(frame)
            self.publish_result(False, stamp)
            return

        pts_qr = bbox[0].astype(np.float32)
        if pts_qr.shape != (4, 2) or cv2.contourArea(pts_qr) < 10.0:
            self._update_display(frame)
            self.publish_result(False, stamp)
            return

        data, _ = self.detector.decode(frame, bbox)
        if not data:
            self._update_display(frame)
            self.publish_result(False, stamp)
            return

        # 2. 缩小 ROI (只取中心 20% 区域，减少边缘干扰)
        x1, y1 = pts_qr.min(axis=0)
        x2, y2 = pts_qr.max(axis=0)
        w, h = x2 - x1, y2 - y1
        rx1, ry1 = x1 + 0.4 * w, y1 + 0.4 * h  # 从 0.2 缩减到 0.4
        rx2, ry2 = x1 + 0.6 * w, y1 + 0.6 * h  # 从 0.8 缩减到 0.6

        # 3. 点云投影与过滤
        pts = np.array(list(pc2.read_points(pc_msg, ("x", "y", "z"), skip_nans=True)), np.float32)
        mask = (pts[:, 0] > 0.1) & (pts[:, 0] < 6.0) & (np.abs(pts[:, 1]) < 2.0)
        pts = pts[mask]

        pts_cam = (pts @ self.R_lidar2cam.T) + self.t_lidar2cam
        pts_lidar = pts[pts_cam[:, 2] > 0.1]
        pts_cam = pts_cam[pts_cam[:, 2] > 0.1]

        if pts_cam.shape[0] < 5:
            self._update_display(frame)
            self.publish_result(False, stamp)
            return

        pts2d, _ = cv2.projectPoints(pts_cam, np.zeros(3), np.zeros(3), self.K, self.D)
        pts2d = pts2d.reshape(-1, 2)

        mask_roi = (pts2d[:, 0] >= rx1) & (pts2d[:, 0] <= rx2) & \
                   (pts2d[:, 1] >= ry1) & (pts2d[:, 1] <= ry2)
        roi_pts = pts_lidar[mask_roi]

        # 4. 距离计算与稳定性逻辑
        if roi_pts.shape[0] > 0:
            raw_dist = np.median(np.linalg.norm(roi_pts, axis=1))
            
            # --- 稳定性逻辑判断 ---
            is_valid_jump = True
            if self.last_valid_dist is not None:
                diff = raw_dist - self.last_valid_dist
                # 如果距离变大且超过阈值，可能是扫到了二维码后面的物体
                if diff > DIST_JUMP_THRESHOLD:
                    rospy.logwarn(f"⚠️ Detect Distance Jump: {raw_dist:.2f} (prev: {self.last_valid_dist:.2f}), Ignoring...")
                    is_valid_jump = False
            
            if is_valid_jump:
                self.last_valid_dist = raw_dist
                smooth_dist = self.ema_filter(raw_dist)
            else:
                # 跳变时，不更新 EMA 也不发布新值，直接返回
                self._update_display(frame)
                return
        else:
            self._update_display(frame)
            self.publish_result(False, stamp)
            return

        # 5. 解析并发布
        try:
            info = json.loads(data)
            qr_id, mx, my, turn, yaw = info.get("id"), info.get("x"), info.get("y"), info.get("turn"), info.get("yaw")
        except: 
            self._update_display(frame)
            return

        self.publish_result(True, stamp, qr_id, mx, my, turn, yaw, round(float(smooth_dist), 3))

        # 绘制 UI
        cv2.polylines(frame, [pts_qr.astype(int)], True, (0, 255, 0), 2)
        cv2.rectangle(frame, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (255, 0, 0), 1) # 画出点云采样区
        cv2.putText(frame, f"ID:{qr_id} Dist:{smooth_dist:.2f}m", (int(x1), int(y1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        rospy.loginfo(f"✅ QR Detected: ID={qr_id}, Dist={smooth_dist:.2f}m, MapX={mx}, MapY={my}, Turn={turn}, Yaw={yaw}")
        self._update_display(frame)

    def _update_display(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()

    def publish_result(self, valid, stamp, qr_id=None, map_x=None, map_y=None, turn=None, yaw=None, dist=None):
        msg = {"valid": valid, "id": qr_id, "map_x": map_x, "map_y": map_y, "turn": turn, "yaw": yaw, "distance": dist, "stamp": stamp}
        self.pub_result.publish(json.dumps(msg))

    def display_loop(self):
        cv2.namedWindow("QR_Perception", cv2.WINDOW_NORMAL)
        while not rospy.is_shutdown():
            with self.lock:
                frame = self.latest_frame.copy()
            cv2.imshow("QR_Perception", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    QRPerceptionNode().display_loop()
