#!/usr/bin/env python3
"""
FilePath: /ultralytics/qrdet/detDecodeQrRos-distance.py
author: wupke
Date: 2026-02-05 09:29:11
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-02-05 16:14:56
Description:
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
"""

"""
看到二维码”升级到“知道自己离二维码多远“

接下来如何根据检测到的图像二维码区域或者像素来算二维码与小车的实时真实距离，有没有激光雷达两种情况下的计算方法？



思路：
1. 使用 OpenCV 的 QRCodeDetector 检测二维码并获取其四个顶点坐标。
2. 计算二维码在图像中的像素宽度（通过四个顶点坐标计算）。
3. 通过已知的二维码实际尺寸和相机的焦距，使用相似三角形原理计算出相机到二维码的距离。
4. 输出距离信息。   

"""


# ###########   ------ 【视觉测距版】：  适配 A3 相机RGB检测二维码 并输出距离数据   ------ ###########

# 订阅 /camera/stereo_image

"""

cd /home/01yuyao/ultralytics/scripts/

① 先激活 ROS 环境（系统 Python）
source /opt/ros/noetic/setup.bash
使用camera_node这个节点
source ~/hyl/perception/devel/setup.bash
② 再激活 Conda 环境
conda activate yolo

③ 运行 A3 相机二维码检测脚本

python  ros_qrDet2A3-distancamera.py

------

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/openblas-pthread/
export PYTHONPATH=$PYTHONPATH:/opt/ros/noetic/lib/python3/dist-packages

        # A3-rgb 相机参数
self.K = np.array([
    [809.0, 0.0, 471.0],
    [0.0, 808.0, 355.0],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

self.D = np.array([-0.139726, 0.121056843, 0.000691770362, -0.00010981268, 0.0424173560], dtype=np.float32)



        # ========= 相机内参 =========
        # # A3-rgb 自标定-相机参数 871.2787604019493, 0.0, 473.9694786149157, 0.0, 866.6266039611612, 354.45736769312197, 0.0, 0.0, 1.0
        # self.K = np.array([[871.2787604019493, 0.0, 473.9694786149157],
        #                    [0.0, 866.6266039611612, 354.45736769312197],
        #                    [0.0, 0.0, 1.0]])
        # self.D = np.array([0.13452653490176816, -0.06657173298041671, 0.003443620193850912, 0.0061879804032612664, 0.0])

       # A3-rgb 黄工给的-相机参数 
        self.K = np.array([[809.0, 0, 471.0],
                           [0, 808.0, 355.0],
                           [0.0, 0.0, 1.0]])
        self.D = np.array([-0.139726073, 0.0121056843, 0.000691770362, -0.000109812168, -0.000424173560])



"""
# 相机输出距离相差太大，建议使用视觉PnP测距法


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import threading

import cv2
import numpy as np
import rospy
from camera_node.msg import StereoImage

QR_SIZE = 0.17  # ⭐ 二维码真实边长（米）


class QRDetectorNode:
    def __init__(self):
        rospy.init_node("qr_detector_node", anonymous=True)

        self.detector = cv2.QRCodeDetector()

        # 相机参数
        self.K = np.array(
            [[818.53994414, 0.0, 289.78608567], [0.0, 818.61550357, 283.78690782], [0.0, 0.0, 1.0]], dtype=np.float32
        )

        self.D = np.array([0.1096357, -0.3696521, -0.0072587, 0.0002249, 0.0], dtype=np.float32)

        self.frame_lock = threading.Lock()
        self.latest_frame = None

        rospy.Subscriber("/camera/stereo_image", StereoImage, self.callback, queue_size=1, buff_size=2**24)

        cv2.namedWindow("QR_View", cv2.WINDOW_NORMAL)
        rospy.loginfo("✅ QR Detector Node Started (Visual Distance Mode)")

    def rosimg_to_cv(self, img_msg):
        h, w, step = img_msg.height, img_msg.width, img_msg.step
        np_arr = np.frombuffer(img_msg.data, dtype=np.uint8)
        frame = np_arr.reshape(h, step)[:, : w * 3].reshape(h, w, 3)
        return frame

    # ⭐ 视觉PnP测距函数
    def estimate_pose(self, pts_2d):
        half = QR_SIZE / 2.0
        obj_pts = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]], dtype=np.float32)

        img_pts = pts_2d.astype(np.float32)

        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.K, self.D, flags=cv2.SOLVEPNP_IPPE_SQUARE)

        if success:
            distance = np.linalg.norm(tvec)
            return rvec, tvec, distance
        return None, None, None

    def callback(self, msg):
        try:
            frame = self.rosimg_to_cv(msg.rgb_image)
        except Exception as e:
            rospy.logerr(f"Image conversion failed: {e}")
            return

        data, bbox, _ = self.detector.detectAndDecode(frame)

        if data and bbox is not None:
            pts = bbox[0]
            pts_int = pts.astype(int)

            cv2.polylines(frame, [pts_int], True, (0, 0, 255), 2)

            cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            # ⭐ 计算距离
            _rvec, tvec, dist = self.estimate_pose(pts)

            if dist is not None:
                z_forward = tvec[2][0]
                cv2.putText(
                    frame,
                    f"Dist:{dist:.2f}m Z:{z_forward:.2f}m",
                    (cx - 80, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                rospy.loginfo(f"📏 距离:{dist:.2f}m  前向:{z_forward:.2f}m")

            try:
                info = json.loads(data)
                rospy.loginfo(f"ID:{info['id']} Turn:{info['turn']}")
            except:
                rospy.loginfo(f"Raw QR: {data}")

        with self.frame_lock:
            self.latest_frame = frame.copy()

    def display_loop(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            with self.frame_lock:
                if self.latest_frame is not None:
                    cv2.imshow("QR_View", self.latest_frame)
            cv2.waitKey(1)
            rate.sleep()


if __name__ == "__main__":
    node = QRDetectorNode()
    node.display_loop()
