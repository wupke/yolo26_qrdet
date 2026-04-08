#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
FilePath: /ultralytics/qrdet/check-detDecodeQrRos-distanceLidar17.py
author: wupke
Date: 2026-03-20 17:29:10
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-03-27 09:44:52
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
'''

# 相比于v16，继续优化：

            # ========= 新增：图像预处理（增强检测稳定性） =========

'''
灰度化处理：二维码检测只需要亮度信息。

应用:自适应直方图均衡化 (CLAHE) ：增强局部对比度。

双重检测策略（可选）：如果原始图像没检测到，再用增强后的图像尝试检测。

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
        self.last_qr_id = None

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
            
            frame = self.rosimg_to_cv(img_msg.rgb_image)
            stamp = img_msg.header.stamp.to_sec()

            ####  ------- 新增数据增强：
        
            # ========= 新增：图像预处理（增强检测稳定性） =========
            # 1. 转为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 2. 创建 CLAHE 对象 (ClipLimit 是对比度限制，TileGridSize 是分块大小)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            
            # 3. (可选) 如果环境极度恶劣，可以尝试二值化（针对过暗环境）
            # _, thresh = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # ========= QR 检测（使用增强后的灰度图） =========
            # 提示：OpenCV 的 detect 可以在灰度图上运行，效果通常比彩色图更稳
            ok, bbox = self.detector.detect(enhanced_gray)
            
            # 如果增强图没检测到，有时原始灰度图反而能行，可以做一个简单的 fallback (后备)
            if not ok:
                ok, bbox = self.detector.detect(gray)

            if not ok or bbox is None:
                # 如果没检测到，在 UI 上显示增强后的图像（方便调试看效果）
                # self._update_display(cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)) 
                self._update_display(frame)
                self.publish_result(False, stamp)
                return


            ### ---------
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

            # --- 新增：提前解析 ID 以便判断是否切换了目标 ---
            try:
                info = json.loads(data)
                qr_id = info.get("id")
                mx, my, turn, yaw = info.get("x"), info.get("y"), info.get("turn"), info.get("yaw")
            except:
                self._update_display(frame)
                return

            # # 2. 缩小 ROI (中心 20% 区域)
            # x1, y1 = pts_qr.min(axis=0)
            # x2, y2 = pts_qr.max(axis=0)
            # w, h = x2 - x1, y2 - y1
            # rx1, ry1 = x1 + 0.4 * w, y1 + 0.4 * h
            # rx2, ry2 = x1 + 0.6 * w, y1 + 0.6 * h

            # 2. 缩小 ROI (中心 10% 区域)
            x1, y1 = pts_qr.min(axis=0)
            x2, y2 = pts_qr.max(axis=0)
            w, h = x2 - x1, y2 - y1
            rx1, ry1 = x1 + 0.46 * w, y1 + 0.46 * h
            rx2, ry2 = x1 + 0.6 * w, y1 + 0.6 * h

            # 3. 点云投影
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

            # 4. 距离计算与 ID 隔离的稳定性逻辑
            if roi_pts.shape[0] > 0:
                raw_dist = np.median(np.linalg.norm(roi_pts, axis=1))
                
                # --- 核心逻辑修改：判断是否为同一个 ID ---
                # 定义 self.last_qr_id 在 __init__ 中初始化为 None
                if not hasattr(self, 'last_qr_id'): self.last_qr_id = None

                is_valid = True
                # 如果是同一个二维码，才执行跳变过滤
                if qr_id == self.last_qr_id and self.last_valid_dist is not None:
                    diff = raw_dist - self.last_valid_dist
                    if diff > DIST_JUMP_THRESHOLD:
                        rospy.logwarn(f"⚠️ [ID:{qr_id}] Jump Detected: {raw_dist:.2f} (prev: {self.last_valid_dist:.2f}), Ignoring...")
                        is_valid = False
                else:
                    # 如果 ID 变了，或者是第一次检测，重置滤波器
                    if qr_id != self.last_qr_id:
                        rospy.loginfo(f"🆕 Switched to New QR: {qr_id}, resetting filters.")
                        self.filtered_dist = raw_dist # 重置 EMA 基准
                    self.last_qr_id = qr_id
                
                if is_valid:
                    self.last_valid_dist = raw_dist
                    smooth_dist = self.ema_filter(raw_dist)
                else:
                    # 如果是跳变噪声，沿用上一次的平滑距离进行显示和发布，避免输出中断
                    smooth_dist = self.filtered_dist 
            else:
                self._update_display(frame)
                self.publish_result(False, stamp)
                return

            # 5. 发布结果
            self.publish_result(True, stamp, qr_id, mx, my, turn, yaw, round(float(smooth_dist), 3))

            # 绘制 UI
            cv2.polylines(frame, [pts_qr.astype(int)], True, (0, 255, 0), 2)
            cv2.rectangle(frame, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (255, 0, 0), 1)
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



