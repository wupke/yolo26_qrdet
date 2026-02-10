#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
FilePath: /ultralytics/qrdet/detDecodeQrRos.py
author: wupke
Date: 2026-02-04 15:37:16
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-02-04 18:27:47
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
'''

# ###########   ------  适配 A3 相机RGB检测二维码   （ROS）   ------ ###########


'''

cd /home/01yuyao/ultralytics/scripts/

① 先激活 ROS 环境（系统 Python）
source /opt/ros/noetic/setup.bash
使用camera_node这个节点
source ~/hyl/perception/devel/setup.bash
② 再激活 Conda 环境
conda activate yolo

③ 运行 A3 相机二维码检测脚本

python ros_qrDet2A3-2.py 

------

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/openblas-pthread/
export PYTHONPATH=$PYTHONPATH:/opt/ros/noetic/lib/python3/dist-packages



'''




#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import json
import numpy as np
import threading
from camera_node.msg import StereoImage


class QRDetectorNode:
    def __init__(self):
        rospy.init_node("qr_detector_node", anonymous=True)

        self.detector = cv2.QRCodeDetector()

        # ===== 线程安全图像缓存 =====
        self.frame_lock = threading.Lock()
        self.latest_frame = None

        rospy.Subscriber(
            "/camera/stereo_image",
            StereoImage,
            self.callback,
            queue_size=1,
            buff_size=2**24
        )

        cv2.namedWindow("QR_View", cv2.WINDOW_NORMAL)
        rospy.loginfo("✅ QR Detector Node Started (/camera/stereo_image RGB)")

    # ================= ROS Image → OpenCV =================
    def rosimg_to_cv(self, img_msg):
        h = img_msg.height
        w = img_msg.width
        step = img_msg.step
        ch = 3  # bgr8

        np_arr = np.frombuffer(img_msg.data, dtype=np.uint8)

        frame = np_arr.reshape(h, step)
        frame = frame[:, :w * ch]
        frame = frame.reshape(h, w, ch)

        return frame

    # ================= ROS回调线程 =================
    def callback(self, msg):
        try:
            frame = self.rosimg_to_cv(msg.rgb_image)  # ⭐ 只取 RGB
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

            rospy.loginfo(f"📍 QR pixel center: ({cx}, {cy})")

            try:
                info = json.loads(data)
                rospy.loginfo(
                    f"🧾 ID:{info['id']}  "
                    f"World:({info['x']:.2f},{info['y']:.2f})  "
                    f"Yaw:{info['yaw']}  Turn:{info['turn']}"
                )
            except:
                rospy.loginfo(f"Raw QR data: {data}")

        # ⭐ 只更新缓存，不显示
        with self.frame_lock:
            self.latest_frame = frame.copy()

    # ================= GUI主线程 =================
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
    node.display_loop()   # ⭐ GUI主线程
























# #####  usb-单目版本  ----------- （稳定不崩版）



# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import rospy
# import cv2
# import json
# import numpy as np
# from sensor_msgs.msg import Image
# import threading


# class QRDetectorNode:
#     def __init__(self):
#         rospy.init_node("qr_detector_node", anonymous=True)

#         self.detector = cv2.QRCodeDetector()

#         # 线程安全的图像缓存
#         self.frame_lock = threading.Lock()
#         self.latest_frame = None

#         rospy.Subscriber(
#             "/camera/image_raw",
#             Image,
#             self.callback,
#             queue_size=1,
#             buff_size=2**24
#         )

#         cv2.namedWindow("QR_View", cv2.WINDOW_NORMAL)
#         rospy.loginfo("✅ QR Detector Node Started (/camera/image_raw)")

#     # ================= 图像转换 =================
#     def rosimg_to_cv(self, img_msg):
#         img_array = np.frombuffer(img_msg.data, dtype=np.uint8)
#         frame = img_array.reshape(img_msg.height, img_msg.width, 3)

#         if img_msg.encoding == "rgb8":
#             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#         return frame

#     # ================= ROS回调（只做计算，不显示） =================
#     def callback(self, msg):
#         try:
#             frame = self.rosimg_to_cv(msg)
#         except Exception as e:
#             rospy.logerr(f"Image conversion failed: {e}")
#             return

#         data, bbox, _ = self.detector.detectAndDecode(frame)

#         if data and bbox is not None:
#             pts = bbox[0].astype(int)
#             cv2.polylines(frame, [pts], True, (0, 0, 255), 2)

#             cx = int(np.mean(pts[:, 0]))
#             cy = int(np.mean(pts[:, 1]))
#             cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

#             rospy.loginfo(f"📍 QR pixel center: ({cx}, {cy})")

#             try:
#                 info = json.loads(data)
#                 rospy.loginfo(
#                     f"🧾 ID:{info['id']}  "
#                     f"World:({info['x']:.2f},{info['y']:.2f})  "
#                     f"Yaw:{info['yaw']}  Turn:{info['turn']}"
#                 )
#             except:
#                 rospy.loginfo(f"Raw QR data: {data}")

#         # 只更新共享帧
#         with self.frame_lock:
#             self.latest_frame = frame.copy()

#     # ================= 主线程显示循环 =================
#     def display_loop(self):
#         rate = rospy.Rate(30)
#         while not rospy.is_shutdown():
#             with self.frame_lock:
#                 if self.latest_frame is not None:
#                     cv2.imshow("QR_View", self.latest_frame)

#             cv2.waitKey(1)
#             rate.sleep()


# if __name__ == "__main__":
#     node = QRDetectorNode()
#     node.display_loop()   # ⚠️ 主线程在这里跑GUI


















# ###########   ------  适配 A3 相机识别（ROS）  ------ ###########

''''
① 先激活 ROS 环境（系统 Python）
source /opt/ros/noetic/setup.bash
source ~/hyl/perception/devel/setup.bash

② 再激活 Conda 环境
conda activate yolo

export PYTHONPATH=$PYTHONPATH:/opt/ros/noetic/lib/python3/dist-packages


'''



# #!/usr/bin/env python3
# import rospy
# import cv2
# import json
# import numpy as np
# from camera_node.msg import StereoImage

# class QRDetectorNode:
#     def __init__(self):
#         rospy.init_node("qr_detector_node")

#         # pip 安装的 OpenCV，必须支持 QUIRC
#         self.detector = cv2.QRCodeDetector()

#         rospy.Subscriber(
#             "/camera/stereo_image",
#             StereoImage,
#             self.callback,
#             queue_size=1,
#             buff_size=2**24
#         )

#         cv2.namedWindow("QR_View", cv2.WINDOW_NORMAL)
#         rospy.loginfo("QR Detector Node Started (RGB only)")

#     def rosimg_to_cv(self, img_msg):
#         """不使用cv_bridge，手动解析ROS Image"""
#         h = img_msg.height
#         w = img_msg.width
#         step = img_msg.step  # 每行真实字节数（Jetson通常> w*3）
#         ch = 3  # bgr8

#         np_arr = np.frombuffer(img_msg.data, dtype=np.uint8)

#         frame = np_arr.reshape(h, step)
#         frame = frame[:, :w * ch]   # 裁掉对齐填充
#         frame = frame.reshape(h, w, ch)

#         return frame

#     def callback(self, msg):
#         try:
#             frame = self.rosimg_to_cv(msg.rgb_image)
#         except Exception as e:
#             rospy.logerr(f"Image conversion failed: {e}")
#             return

#         data, bbox, _ = self.detector.detectAndDecode(frame)

#         if data and bbox is not None:
#             pts = bbox[0].astype(int)

#             # 画二维码边框
#             cv2.polylines(frame, [pts], True, (0, 0, 255), 2)

#             # 中心点
#             cx = int(np.mean(pts[:, 0]))
#             cy = int(np.mean(pts[:, 1]))
#             cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

#             rospy.loginfo(f"QR pixel center: ({cx}, {cy})")

#             try:
#                 info = json.loads(data)
#                 rospy.loginfo(
#                     f"ID:{info['id']}  "
#                     f"World:({info['x']:.2f},{info['y']:.2f})  "
#                     f"Yaw:{info['yaw']}  Turn:{info['turn']}"
#                 )
#             except:
#                 rospy.loginfo(f"Raw QR data: {data}")

#         cv2.imshow("QR_View", frame)
#         cv2.waitKey(1)


# if __name__ == "__main__":
#     QRDetectorNode()
#     rospy.spin()
