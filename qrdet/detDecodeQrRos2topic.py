#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
FilePath: /ultralytics/qrdet/detDecodeQrRos2topic.py
author: wupke
Date: 2026-02-04 18:32:28
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-02-04 18:38:56
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
'''

'''

把 detDecodeQrRos.py 的二维码检测结果，发布成 ROS topic 供其他节点使用
思路：
1. 在 detDecodeQrRos.py 中，找到二维码检测结果的相关代码
2. 将检测结果封装成 ROS 消息
3. 创建一个 ROS Publisher，将消息发布到指定的 topic


- 定义一个二维码检测消息，在你的 camera_node/msg 下新建：

QRDetection.msg：
        std_msgs/Header header

        int32 id
        float32 world_x
        float32 world_y
        float32 yaw
        string turn

        int32 pixel_x
        int32 pixel_y

- 然后编译：

    cd ~/hyl/perception
    catkin_make
    source devel/setup.bash

- 在 detDecodeQrRos.py 中添加发布代码：


'''


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import json
import numpy as np
import threading

from sensor_msgs.msg import Image
from camera_node.msg import StereoImage
from camera_node.msg import QRDetection   # ⭐ 新消息


class QRDetectorNode:
    def __init__(self):
        rospy.init_node("qr_detector_node", anonymous=True)

        self.detector = cv2.QRCodeDetector()

        self.frame_lock = threading.Lock()
        self.latest_frame = None

        # ===== 发布器 =====
        self.img_pub = rospy.Publisher("/qr/detected_image", Image, queue_size=1)
        self.qr_pub  = rospy.Publisher("/qr/detections", QRDetection, queue_size=10)

        rospy.Subscriber(
            "/camera/stereo_image",
            StereoImage,
            self.callback,
            queue_size=1,
            buff_size=2**24
        )

        cv2.namedWindow("QR_View", cv2.WINDOW_NORMAL)
        rospy.loginfo("✅ QR Detector Node Started")

    # ROS Image → OpenCV
    def rosimg_to_cv(self, img_msg):
        h, w, step = img_msg.height, img_msg.width, img_msg.step
        np_arr = np.frombuffer(img_msg.data, dtype=np.uint8)
        frame = np_arr.reshape(h, step)[:, :w*3].reshape(h, w, 3)
        return frame

    # OpenCV → ROS Image
    def cv_to_rosimg(self, frame, header):
        msg = Image()
        msg.header = header
        msg.height, msg.width = frame.shape[:2]
        msg.encoding = "bgr8"
        msg.step = msg.width * 3
        msg.data = frame.tobytes()
        return msg

    # ROS 回调
    def callback(self, msg):
        try:
            frame = self.rosimg_to_cv(msg.rgb_image)
        except Exception as e:
            rospy.logerr(f"Image conversion failed: {e}")
            return

        data, bbox, _ = self.detector.detectAndDecode(frame)

        if data and bbox is not None:
            pts = bbox[0].astype(int)
            cv2.polylines(frame, [pts], True, (0, 0, 255), 2)

            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            try:
                info = json.loads(data)

                # ===== 发布结构化检测结果 =====
                det_msg = QRDetection()
                det_msg.header = msg.header
                det_msg.id = info["id"]
                det_msg.world_x = info["x"]
                det_msg.world_y = info["y"]
                det_msg.yaw = info["yaw"]
                det_msg.turn = info["turn"]
                det_msg.pixel_x = cx
                det_msg.pixel_y = cy

                self.qr_pub.publish(det_msg)

                rospy.loginfo(f"📡 发布QR检测 ID:{info['id']}")

            except:
                rospy.loginfo(f"Raw QR data: {data}")

        # 发布标注图像
        self.img_pub.publish(self.cv_to_rosimg(frame, msg.header))

        with self.frame_lock:
            self.latest_frame = frame.copy()

    # GUI线程
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



