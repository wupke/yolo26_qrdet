#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
FilePath: /ultralytics/qrdet/check-detDecodeQrRos-distanceLidar12.py
author: wupke
Date: 2026-02-09 16:06:05
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-03-24 12:07:24
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
'''

'''
仿照手机，从远处拉近二维码，再扫描，增大识别的距离

手机“远处看到框 → 拉近后识别”的能力，本质是：

“先低成本 detect（找可能是二维码的地方），再对该区域做连续跟踪 + 多尺度解码”

'''


# 基于 check-detDecodeQrRos-distanceLidar11withoutcv11和17版本的结合，继续优化：


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy, cv2, json, numpy as np, message_filters, threading, time
import sensor_msgs.point_cloud2 as pc2
from camera_node.msg import StereoImage
from std_msgs.msg import String

# ========= 参数配置 =========
EMA_ALPHA = 0.25            # 指数移动平均系数，越小越平滑但延迟越高
DIST_JUMP_THRESHOLD = 0.2   # 同一个码距离突变阈值 (米)
BBOX_TIMEOUT = 0.6          # 丢失检测后保留上一帧位置的时间 (秒)

class QRPerceptionNode:
    def __init__(self):
        rospy.init_node("qr_perception_node")
        
        # 初始化 OpenCV 二维码检测器
        self.detector = cv2.QRCodeDetector()
        # 自适应直方图均衡化 (CLAHE)，用于增强对比度
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # ========= 相机内参 (请根据实际标定值微调) =========
        self.K = np.array([[809.0, 0, 471.0], [0, 808.0, 355.0], [0, 0, 1.0]], np.float32)
        self.D = np.array([-0.1397, 0.0121, 0.00069, -0.00011, -0.00042], np.float32)

        # ========= 外参: LiDAR -> Camera =========
        self.R_lidar2cam = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], np.float32)
        self.t_lidar2cam = np.array([[0.01, 0.0, 0.075]], np.float32)

        # ========= 运行时状态变量 =========
        self.lock = threading.Lock()
        self.latest_frame = np.zeros((720, 960, 3), np.uint8)
        self.last_display_time = time.time()
        
        self.filtered_dist = None      # 平滑后的距离
        self.last_valid_dist = None    # 上一次合法的测量值
        self.last_qr_id = None         # 记录上一个 ID 用于隔离逻辑
        
        # 跨帧跟踪缓存 (仿手机扫码)
        self.last_qr_bbox = None
        self.last_qr_time = 0

        # ========= ROS 订阅与发布 =========
        img_sub = message_filters.Subscriber("/camera/stereo_image", StereoImage)
        pc_sub = message_filters.Subscriber("/livox/lidar", pc2.PointCloud2)
        
        # 时间同步器
        ts = message_filters.ApproximateTimeSynchronizer([img_sub, pc_sub], 10, 0.05)
        ts.registerCallback(self.callback)

        self.pub_result = rospy.Publisher("/qr_perception/result", String, queue_size=10)
        
        rospy.loginfo("🚀 QR Perception Node: High-Stability & Multi-Scale Mode Active")

    def decode_with_roi_retry(self, frame, pts_qr):
        """ 仿手机拉近逻辑：截取 ROI，增强对比度并多尺度放大解码 """
        x1, y1 = np.min(pts_qr, axis=0).astype(int)
        x2, y2 = np.max(pts_qr, axis=0).astype(int)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return None

        # 1. 局部增强
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        enhanced_roi = self.clahe.apply(gray_roi)

        # 2. 多尺度尝试 (1.5倍到3倍放大)
        for scale in [1.5, 2.0, 3.0]:
            roi_up = cv2.resize(enhanced_roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            # 💡 关键修复：使用 [0] 获取结果，兼容不同版本返回值数量
            res = self.detector.detectAndDecode(roi_up)
            data = res[0]
            if data: return data
        return None

    def callback(self, img_msg, pc_msg):
        frame = self.rosimg_to_cv(img_msg.rgb_image)
        stamp = img_msg.header.stamp.to_sec()
        
        # 预处理：增强全图对比度用于检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced_gray = self.clahe.apply(gray)

        data, pts_qr = None, None

        # --- 步骤 1: 实时检测 ---
        ok, bbox = self.detector.detect(enhanced_gray)
        if ok and bbox is not None:
            pts_qr = bbox[0].astype(np.float32)
            if pts_qr.shape == (4, 2) and cv2.contourArea(pts_qr) > 10:
                self.last_qr_bbox = pts_qr
                self.last_qr_time = time.time()
                
                # 尝试解码
                decode_res = self.detector.decode(enhanced_gray, bbox)
                data = decode_res[0]
                if not data:
                    data = self.decode_with_roi_retry(frame, pts_qr)

        # --- 步骤 2: 历史 bbox 盲扫 (手机级跟踪) ---
        if data is None and self.last_qr_bbox is not None:
            if time.time() - self.last_qr_time < BBOX_TIMEOUT:
                pts_qr = self.last_qr_bbox
                data = self.decode_with_roi_retry(frame, pts_qr)

        if data is None or pts_qr is None:
            self.publish_result(False, stamp)
            self._update_display(frame)
            return

        # --- 步骤 3: 解析内容与 ID 隔离逻辑 ---
        try:
            info = json.loads(data)
            qr_id = info.get("id")
        except:
            self._update_display(frame)
            return

        # --- 步骤 4: 点云测距逻辑 (ROI 采样 5%) ---
        x1, y1 = pts_qr.min(axis=0)
        x2, y2 = pts_qr.max(axis=0)
        w, h = x2 - x1, y2 - y1
        # 只取中心 5% 面积区域
        rx1, ry1 = x1 + 0.46 * w, y1 + 0.46 * h
        rx2, ry2 = x1 + 0.6 * w, y1 + 0.56 * h

        pts = np.array(list(pc2.read_points(pc_msg, ("x", "y", "z"), skip_nans=True)), np.float32)
        # 点云初筛
        mask = (pts[:, 0] > 0.1) & (pts[:, 0] < 6.0) & (np.abs(pts[:, 1]) < 2.0)
        pts_lidar = pts[mask]

        # 投影到图像
        pts_cam = (pts_lidar @ self.R_lidar2cam.T) + self.t_lidar2cam
        front_mask = pts_cam[:, 2] > 0.1
        pts_cam = pts_cam[front_mask]
        pts_lidar = pts_lidar[front_mask]

        if pts_cam.shape[0] >= 5:
            pts2d, _ = cv2.projectPoints(pts_cam, np.zeros(3), np.zeros(3), self.K, self.D)
            pts2d = pts2d.reshape(-1, 2)
            
            mask_roi = (pts2d[:, 0] >= rx1) & (pts2d[:, 0] <= rx2) & \
                       (pts2d[:, 1] >= ry1) & (pts2d[:, 1] <= ry2)
            roi_pts = pts_lidar[mask_roi]

            if roi_pts.shape[0] > 0:
                raw_dist = np.median(np.linalg.norm(roi_pts, axis=1))
                
                # --- 稳定性校验 ---
                is_valid = True
                if qr_id == self.last_qr_id and self.last_valid_dist is not None:
                    # 如果同一个码距离突然变远，认为是扫到了背景，拦截
                    if (raw_dist - self.last_valid_dist) > DIST_JUMP_THRESHOLD:
                        rospy.logwarn(f"⚠️ [ID:{qr_id}] Jump Detected! Raw:{raw_dist:.2f} Prev:{self.last_valid_dist:.2f}")
                        is_valid = False
                else:
                    # ID 变化，重置 EMA 和基准
                    self.filtered_dist = raw_dist 
                    self.last_qr_id = qr_id

                if is_valid:
                    self.last_valid_dist = raw_dist
                    smooth_dist = self.ema_filter(raw_dist)
                else:
                    # 无效跳变时，沿用上次平滑值，不发布突变数据
                    smooth_dist = self.filtered_dist
            else:
                self._update_display(frame); return
        else:
            self._update_display(frame); return

        # --- 步骤 5: 发布与绘制 ---
        mx, my, turn, yaw = info.get("x"), info.get("y"), info.get("turn"), info.get("yaw")
        self.publish_result(True, stamp, qr_id, mx, my, turn, yaw, round(float(smooth_dist), 3))

        # UI 绘制
        cv2.polylines(frame, [pts_qr.astype(int)], True, (0, 255, 0), 2)
        cv2.rectangle(frame, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (255, 0, 0), 1)
        cv2.putText(frame, f"ID:{qr_id} Dist:{smooth_dist:.2f}m", (int(x1), int(y1)-15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        rospy.loginfo(f"✅ QR Detected: ID={qr_id}, Dist={smooth_dist:.2f}m, MapX={mx}, MapY={my}, Turn={turn}, Yaw={yaw}")

        self._update_display(frame)

    def rosimg_to_cv(self, msg):
        h, w, step = msg.height, msg.width, msg.step
        return np.frombuffer(msg.data, np.uint8).reshape(h, step)[:, :w * 3].reshape(h, w, 3).copy()

    def ema_filter(self, dist):
        if self.filtered_dist is None: self.filtered_dist = dist
        else: self.filtered_dist = EMA_ALPHA * dist + (1 - EMA_ALPHA) * self.filtered_dist
        return self.filtered_dist

    def publish_result(self, valid, stamp, qr_id=None, map_x=None, map_y=None, turn=None, yaw=None, dist=None):
        msg = {
            "valid": valid, "id": qr_id, "map_x": map_x, "map_y": map_y, 
            "turn": turn, "yaw": yaw, "distance": dist, "stamp": stamp
        }
        self.pub_result.publish(json.dumps(msg))

    def _update_display(self, frame):
        # 计算显示 FPS
        now = time.time()
        fps = 1.0 / max(1e-6, now - self.last_display_time)
        self.last_display_time = now
        cv2.putText(frame, f"FPS:{fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        with self.lock:
            self.latest_frame = frame

    def display_loop(self):
        cv2.namedWindow("QR_Perception", cv2.WINDOW_NORMAL)
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            with self.lock:
                cv2.imshow("QR_Perception", self.latest_frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            rate.sleep()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        QRPerceptionNode().display_loop()
    except rospy.ROSInterruptException:
        pass














########################################################################



# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import rospy, cv2, json, numpy as np, message_filters, threading, time
# import sensor_msgs.point_cloud2 as pc2
# from camera_node.msg import StereoImage
# from std_msgs.msg import String

# DEBUG_PROJECTION = False
# EMA_ALPHA = 0.25


# class QRPerceptionNode:
#     def __init__(self):
#         rospy.init_node("qr_perception_node")

#         self.detector = cv2.QRCodeDetector()

#         # ========= Camera intrinsics =========
#         self.K = np.array([
#             [809.0, 0,   471.0],
#             [0,   808.0, 355.0],
#             [0,     0,     1.0]
#         ], np.float32)

#         self.D = np.array(
#             [-0.1397, 0.0121, 0.00069, -0.00011, -0.00042],
#             np.float32
#         )

#         # ========= LiDAR → OpenCV Camera =========
#         self.R_lidar2cam = np.array([
#             [ 0, -1,  0],
#             [ 0,  0, -1],
#             [ 1,  0,  0]
#         ], np.float32)

#         self.t_lidar2cam = np.array([[0.01, 0.0, 0.075]], np.float32)

#         # ========= Runtime =========
#         self.lock = threading.Lock()
#         self.latest_frame = np.zeros((720, 960, 3), np.uint8)
#         self.last_time = time.time()
#         self.filtered_dist = None

#         self.roi_pc_buffer = []
#         self.buffer_size = 3

#         # ⭐ 手机扫码级别：跨帧 QR bbox 缓存
#         self.last_qr_bbox = None
#         self.last_qr_time = 0
#         self.bbox_timeout = 0.6  # 秒

#         # ========= ROS IO =========
#         img_sub = message_filters.Subscriber(
#             "/camera/stereo_image", StereoImage
#         )
#         pc_sub = message_filters.Subscriber(
#             "/livox/lidar", pc2.PointCloud2
#         )

#         ts = message_filters.ApproximateTimeSynchronizer(
#             [img_sub, pc_sub], 10, 0.05
#         )
#         ts.registerCallback(self.callback)

#         self.pub_result = rospy.Publisher(
#             "/qr_perception/result",
#             String,
#             queue_size=10
#         )

#         cv2.namedWindow("QR_Perception", cv2.WINDOW_NORMAL)
#         rospy.loginfo("🚀 QR Perception Node Started (ROI multi-scale decode enabled)")

#     def rosimg_to_cv(self, msg):
#         h, w, step = msg.height, msg.width, msg.step
#         return (
#             np.frombuffer(msg.data, np.uint8)
#             .reshape(h, step)[:, :w * 3]
#             .reshape(h, w, 3)
#             .copy()
#         )

#     def ema_filter(self, dist):
#         if self.filtered_dist is None:
#             self.filtered_dist = dist
#         else:
#             self.filtered_dist = (
#                 EMA_ALPHA * dist +
#                 (1 - EMA_ALPHA) * self.filtered_dist
#             )
#         return self.filtered_dist

#     def publish_result(self, valid, stamp,
#                        qr_id=None, map_x=None, map_y=None,
#                        turn=None, yaw=None, dist=None):
#         msg = {
#             "valid": valid,
#             "id": qr_id,
#             "map_x": map_x,
#             "map_y": map_y,
#             "turn": turn,
#             "yaw": yaw,
#             "distance": dist,
#             "stamp": stamp
#         }
#         self.pub_result.publish(json.dumps(msg))

#     # ========= 新增：ROI 多尺度 decode =========
#     def decode_with_roi_retry(self, frame, pts_qr):
#         x1, y1 = np.min(pts_qr, axis=0).astype(int)
#         x2, y2 = np.max(pts_qr, axis=0).astype(int)

#         h, w = frame.shape[:2]
#         x1, y1 = max(0, x1), max(0, y1)
#         x2, y2 = min(w - 1, x2), min(h - 1, y2)

#         roi = frame[y1:y2, x1:x2]
#         if roi.size == 0:
#             return None

#         for scale in [1.5, 2.0, 3.0]:
#             roi_up = cv2.resize(
#                 roi, None, fx=scale, fy=scale,
#                 interpolation=cv2.INTER_CUBIC
#             )
#             data, _ = self.detector.detectAndDecode(roi_up)
#             if data:
#                 return data
#         return None

#     def callback(self, img_msg, pc_msg):
#         frame = self.rosimg_to_cv(img_msg.rgb_image)
#         stamp = img_msg.header.stamp.to_sec()

#         data = None
#         pts_qr = None

#         # ========= 1️⃣ detect =========
#         ok, bbox = self.detector.detect(frame)

#         if ok and bbox is not None:
#             pts_qr = bbox[0].astype(np.float32)
#             if pts_qr.shape == (4, 2) and cv2.contourArea(pts_qr) > 10:
#                 self.last_qr_bbox = pts_qr
#                 self.last_qr_time = time.time()

#                 # 先尝试原图 decode
#                 data, _ = self.detector.decode(frame, bbox)
#                 if not data:
#                     data = self.decode_with_roi_retry(frame, pts_qr)

#         # ========= 2️⃣ detect 失败，用历史 bbox =========
#         if data is None:
#             if self.last_qr_bbox is not None:
#                 if time.time() - self.last_qr_time < self.bbox_timeout:
#                     pts_qr = self.last_qr_bbox
#                     data = self.decode_with_roi_retry(frame, pts_qr)

#         if data is None or pts_qr is None:
#             self.publish_result(valid=False, stamp=stamp)
#             self._update_fps(frame)
#             return

#         # ========= Overlay QR =========
#         cv2.polylines(frame, [pts_qr.astype(int)], True, (0, 255, 0), 2)
#         cx, cy = pts_qr.mean(axis=0)

#         # ========= ROI =========
#         x1, y1 = pts_qr.min(axis=0)
#         x2, y2 = pts_qr.max(axis=0)
#         w, h = x2 - x1, y2 - y1
#         rx1, ry1 = x1 + 0.2 * w, y1 + 0.2 * h
#         rx2, ry2 = x1 + 0.8 * w, y1 + 0.8 * h

#         # ========= Point cloud =========
#         pts = np.array(list(pc2.read_points(
#             pc_msg, ("x", "y", "z"), skip_nans=True
#         )), np.float32)

#         if pts.shape[0] < 20:
#             self.publish_result(False, stamp)
#             return

#         mask = (
#             (pts[:, 0] > 0.1) & (pts[:, 0] < 5.0) &
#             (np.abs(pts[:, 1]) < 3.0) &
#             (np.abs(pts[:, 2]) < 2.0)
#         )
#         pts = pts[mask]

#         pts_cam = (pts @ self.R_lidar2cam.T) + self.t_lidar2cam
#         mask_front = pts_cam[:, 2] > 0.1
#         pts_cam = pts_cam[mask_front]
#         pts_lidar = pts[mask_front]

#         if pts_cam.shape[0] < 10:
#             self.publish_result(False, stamp)
#             return

#         pts2d, _ = cv2.projectPoints(
#             pts_cam, np.zeros(3), np.zeros(3), self.K, self.D
#         )
#         pts2d = pts2d.reshape(-1, 2)

#         mask_roi = (
#             (pts2d[:, 0] >= rx1) & (pts2d[:, 0] <= rx2) &
#             (pts2d[:, 1] >= ry1) & (pts2d[:, 1] <= ry2)
#         )

#         roi_pts = pts_lidar[mask_roi]
#         if roi_pts.shape[0] == 0:
#             self.publish_result(False, stamp)
#             return

#         self.roi_pc_buffer.append(roi_pts)
#         if len(self.roi_pc_buffer) > self.buffer_size:
#             self.roi_pc_buffer.pop(0)

#         roi_all = np.vstack(self.roi_pc_buffer)
#         lidar_dist = self.ema_filter(
#             np.median(np.linalg.norm(roi_all, axis=1))
#         )

#         # ========= Parse QR =========
#         qr_id = map_x = map_y = qr_turn = yaw = None
#         try:
#             info = json.loads(data)
#             qr_id = info.get("id")
#             map_x = info.get("x")
#             map_y = info.get("y")
#             qr_turn = info.get("turn")
#             yaw = info.get("yaw")
#         except:
#             pass

#         # ========= Publish =========
#         self.publish_result(
#             valid=True,
#             stamp=stamp,
#             qr_id=qr_id,
#             map_x=map_x,
#             map_y=map_y,
#             turn=qr_turn,
#             yaw=yaw,
#             dist=round(float(lidar_dist), 3)
#         )

#         self._update_fps(frame)

#     def _update_fps(self, frame):
#         fps = 1.0 / max(1e-6, time.time() - self.last_time)
#         self.last_time = time.time()
#         cv2.putText(frame, f"FPS:{fps:.1f}",
#                     (20, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1, (255, 255, 0), 2)
#         with self.lock:
#             self.latest_frame = frame

#     def display_loop(self):
#         rate = rospy.Rate(30)
#         while not rospy.is_shutdown():
#             with self.lock:
#                 cv2.imshow("QR_Perception", self.latest_frame)
#             cv2.waitKey(1)
#             rate.sleep()


# if __name__ == "__main__":
#     QRPerceptionNode().display_loop()
