
# ------------------- GPt  ---------------




#######################    v 5   ---  精度准确---包含投影，距离滤波显示 ，但是资源消耗大 ----------------#######



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy, cv2, json, numpy as np, message_filters, threading, time
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from camera_node.msg import StereoImage

DEBUG_PROJECTION = True
QR_SIZE = 0.17
EMA_ALPHA = 0.25


class QRPerceptionNode:
    def __init__(self):
        rospy.init_node("qr_perception_node")

        self.detector = cv2.QRCodeDetector()

        # ========= Camera intrinsics =========
        self.K = np.array([
            [809.0, 0,   471.0],
            [0,   808.0, 355.0],
            [0,     0,     1.0]
        ], np.float32)

        self.D = np.array(
            [-0.1397, 0.0121, 0.00069, -0.00011, -0.00042],
            np.float32
        )

        # ========= LiDAR → OpenCV Camera 坐标变换 =========
        # LiDAR: x前 y左 z上
        # Cam  : x右 y下 z前
        self.R_lidar2cam = np.array([
            [ 0, -1,  0],
            [ 0,  0, -1],
            [ 1,  0,  0]
        ], dtype=np.float32)

        # 平移（单位：米，定义在相机坐标系下）
        self.t_lidar2cam = np.array([[0.01, 0.0, 0.075]], dtype=np.float32)

        # ========= Runtime =========
        self.lock = threading.Lock()
        self.latest_frame = np.zeros((720, 960, 3), dtype=np.uint8)
        self.last_time = time.time()
        self.filtered_dist = None

        # 多帧点云缓存
        self.pc_buffer = []
        self.buffer_size = 3

        img_sub = message_filters.Subscriber(
            "/camera/stereo_image", StereoImage
        )
        pc_sub = message_filters.Subscriber(
            "/livox/lidar", PointCloud2
        )

        ts = message_filters.ApproximateTimeSynchronizer(
            [img_sub, pc_sub], queue_size=10, slop=0.05
        )
        ts.registerCallback(self.callback)

        cv2.namedWindow("QR_Perception", cv2.WINDOW_NORMAL)
        rospy.loginfo("🚀 QR Perception Node Started")

    def rosimg_to_cv(self, img_msg):
        h, w, step = img_msg.height, img_msg.width, img_msg.step
        return (
            np.frombuffer(img_msg.data, np.uint8)
            .reshape(h, step)[:, :w * 3]
            .reshape(h, w, 3)
            .copy()
        )

    def ema_filter(self, dist):
        if self.filtered_dist is None:
            self.filtered_dist = dist
        else:
            self.filtered_dist = (
                EMA_ALPHA * dist +
                (1 - EMA_ALPHA) * self.filtered_dist
            )
        return self.filtered_dist

    def callback(self, img_msg, pc_msg):
        frame = self.rosimg_to_cv(img_msg.rgb_image)
        h_img, w_img = frame.shape[:2]

        # ========= Read point cloud =========
        pts = np.array(
            list(pc2.read_points(
                pc_msg, ("x", "y", "z"), skip_nans=True
            )),
            dtype=np.float32
        )

        if pts.size == 0:
            return

        # 粗裁剪（车体坐标系）
        mask = (
            (pts[:, 0] > 0.1) & (pts[:, 0] < 5.0) &
            (np.abs(pts[:, 1]) < 3.0) &
            (np.abs(pts[:, 2]) < 2.0)
        )
        pts = pts[mask]

        if pts.shape[0] < 10:
            return

        # 多帧融合
        self.pc_buffer.append(pts)
        if len(self.pc_buffer) > self.buffer_size:
            self.pc_buffer.pop(0)

        pts_all = np.vstack(self.pc_buffer)

        # ========= LiDAR → Camera 坐标系 =========
        pts_cam = (pts_all @ self.R_lidar2cam.T) + self.t_lidar2cam

        # 只保留在相机前方的点
        mask_front = pts_cam[:, 2] > 0.1
        pts_cam = pts_cam[mask_front]
        pts_lidar_valid = pts_all[mask_front]

        if pts_cam.shape[0] < 10:
            return

        # ========= Projection =========
        pts2d, _ = cv2.projectPoints(
            pts_cam,
            np.zeros(3), np.zeros(3),
            self.K, self.D
        )
        pts2d = pts2d.reshape(-1, 2)

        if DEBUG_PROJECTION:
            for x, y in pts2d[::3]:
                if np.isfinite(x) and np.isfinite(y):
                    xi, yi = int(x), int(y)
                    if 0 <= xi < w_img and 0 <= yi < h_img:
                        cv2.circle(
                            frame, (xi, yi),
                            2, (0, 0, 255), -1
                        )

        # ========= QR detect =========
        data, bbox, _ = self.detector.detectAndDecode(frame)

        if data and bbox is not None:
            pts_qr = bbox[0].astype(np.float32)
            cv2.polylines(
                frame, [pts_qr.astype(int)],
                True, (0, 255, 0), 2
            )

            cx, cy = pts_qr.mean(axis=0)

            # ========= ROI =========
            x1, y1 = pts_qr.min(axis=0)
            x2, y2 = pts_qr.max(axis=0)
            w, h = x2 - x1, y2 - y1

            rx1, ry1 = x1 + 0.2 * w, y1 + 0.2 * h
            rx2, ry2 = x1 + 0.8 * w, y1 + 0.8 * h

            mask_roi = (
                (pts2d[:, 0] >= rx1) & (pts2d[:, 0] <= rx2) &
                (pts2d[:, 1] >= ry1) & (pts2d[:, 1] <= ry2)
            )

            roi_pts = pts_lidar_valid[mask_roi]

            lidar_dist = None
            if roi_pts.shape[0] > 5:
                lidar_dist = np.median(
                    np.linalg.norm(roi_pts, axis=1)
                )
                lidar_dist = self.ema_filter(lidar_dist)

            # ========= Text =========
            qr_text = data
            try:
                info = json.loads(data)
                qr_text = f"ID:{info['id']} Turn:{info['turn']}"
            except:
                pass

            cv2.putText(
                frame, qr_text,
                (int(cx) - 120, int(cy) - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

            if lidar_dist:
                cv2.putText(
                    frame, f"{lidar_dist:.2f} m",
                    (int(cx) - 60, int(cy) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2
                )
                rospy.loginfo(
                    f"📦 {qr_text}  📏 {lidar_dist:.2f} m"
                )

        # ========= FPS =========
        fps = 1.0 / max(1e-6, time.time() - self.last_time)
        self.last_time = time.time()
        cv2.putText(
            frame, f"FPS:{fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 0), 2
        )

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
    node = QRPerceptionNode()
    node.display_loop()



