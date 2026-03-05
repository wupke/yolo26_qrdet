#!/usr/bin/env python3
"""
FilePath: /ultralytics/qrdet/detDecodeQr.py
author: wupke
Date: 2026-02-04 12:07:53
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-02-04 15:32:48
Description:
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
"""

"""

pip install qrcode[pil]


识别并解码图像中的二维码信息:
    最远识别距离 ≈ 二维码边长 × 10~15

10cm 码 → 最远 1–1.5m
20cm 码 → 最远 2–3m

"""
#!/usr/bin/env python3
import json

import cv2
import numpy as np
import rospy
from camera_node.msg import StereoImage  # 你的自定义消息
from cv_bridge import CvBridge


class QRDetectorNode:
    def __init__(self):
        rospy.init_node("qr_detector_node")

        self.bridge = CvBridge()
        self.detector = cv2.QRCodeDetector()

        rospy.Subscriber("/camera/stereo_image", StereoImage, self.image_callback, queue_size=1, buff_size=2**24)

        rospy.loginfo("QR Detector Node Started")
        cv2.namedWindow("camera", cv2.WINDOW_NORMAL)

    def image_callback(self, msg):
        img_msg = msg.rgb_image

        try:
            # 先直接转成numpy（不让cv_bridge自动reshape）
            np_arr = np.frombuffer(img_msg.data, dtype=np.uint8)

            h = img_msg.height
            step = img_msg.step  # 每行真实字节数（含padding）
            width = img_msg.width  # 有效宽度
            channels = 3  # bgr8

            # 先按 step 展开二维
            frame = np_arr.reshape(h, step)

            # 裁掉 padding
            frame = frame[:, : width * channels]

            # 再 reshape 成图像
            frame = frame.reshape(h, width, channels)

        except Exception as e:
            rospy.logerr("Image reshape error: %s", e)
            return

        # ===== QR 检测 =====
        data, bbox, _ = self.detector.detectAndDecode(frame)

        if data and bbox is not None:
            try:
                info = json.loads(data)
            except:
                rospy.logwarn("QR data not JSON: %s", data)
                return

            pts = bbox[0].astype(int)
            cv2.polylines(frame, [pts], True, (0, 0, 255), 2)

            rospy.loginfo("====== 识别到二维码 ======")
            rospy.loginfo("ID: %s", info["id"])
            rospy.loginfo("坐标: (%.2f, %.2f)", info["x"], info["y"])
            rospy.loginfo("朝向: %s", info["yaw"])
            rospy.loginfo("转向指令: %s", info["turn"])

            if info["turn"] == "left":
                rospy.loginfo(">>> 执行左转")
            elif info["turn"] == "right":
                rospy.loginfo(">>> 执行右转")
            elif info["turn"] == "straight":
                rospy.loginfo(">>> 直行")
            elif info["turn"] == "uturn":
                rospy.loginfo(">>> 掉头")
            elif info["turn"] == "stop":
                rospy.loginfo(">>> 停止")

        cv2.imshow("camera", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    QRDetectorNode()
    rospy.spin()


# ###########   --------   订阅ros话题  -------   格式不对


# #!/usr/bin/env python3
# import rospy
# import cv2
# import json
# import numpy as np
# from cv_bridge import CvBridge
# from sensor_msgs.msg import Image
# from camera_node.msg import StereoImage   # ⭐ 你的自定义消息

# class QRDetectorNode:
#     def __init__(self):
#         rospy.init_node("qr_detector_node")

#         self.bridge = CvBridge()
#         self.detector = cv2.QRCodeDetector()

#         rospy.Subscriber("/camera/stereo_image", StereoImage, self.image_callback, queue_size=1)

#         rospy.loginfo("QR Detector Node Started")
#         cv2.namedWindow("camera", cv2.WINDOW_NORMAL)

#     def image_callback(self, msg):
#         try:
#             # ⭐ 只取 RGB 图像
#             frame = self.bridge.imgmsg_to_cv2(msg.rgb_image, desired_encoding="bgr8")
#         except Exception as e:
#             rospy.logerr("CvBridge error: %s", e)
#             return

#         data, bbox, _ = self.detector.detectAndDecode(frame)

#         if data and bbox is not None:
#             info = json.loads(data)

#             pts = bbox[0].astype(int)
#             cv2.polylines(frame, [pts], True, (0, 0, 255), 2)

#             rospy.loginfo("====== 识别到二维码 ======")
#             rospy.loginfo("ID: %s", info["id"])
#             rospy.loginfo("坐标: (%.2f, %.2f)", info["x"], info["y"])
#             rospy.loginfo("朝向: %s", info["yaw"])
#             rospy.loginfo("转向指令: %s", info["turn"])

#             # 控制逻辑示例
#             if info["turn"] == "left":
#                 rospy.loginfo(">>> 执行左转")
#             elif info["turn"] == "right":
#                 rospy.loginfo(">>> 执行右转")
#             elif info["turn"] == "straight":
#                 rospy.loginfo(">>> 直行")
#             elif info["turn"] == "uturn":
#                 rospy.loginfo(">>> 掉头")
#             elif info["turn"] == "stop":
#                 rospy.loginfo(">>> 停止")

#         cv2.imshow("camera", frame)
#         cv2.waitKey(1)


# if __name__ == "__main__":
#     node = QRDetectorNode()
#     rospy.spin()


# ################     --------  直接订阅 video0 视频流检测------
# import cv2
# import json

# detector = cv2.QRCodeDetector()
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     data, bbox, _ = detector.detectAndDecode(frame)


#     if data:
#         info = json.loads(data)
#         print("------------------------二维码边框坐标:\n", bbox)
#         # 画出检测到的二维码边框
#         pts = bbox[0].astype(int)  # ⭐ 转整数！

#         for i in range(4):
#             pt1 = tuple(pts[i])
#             pt2 = tuple(pts[(i + 1) % 4])
#             cv2.line(frame, pt1, pt2, (255, 0, 0), 2)


#         print("\n====== 识别到二维码 ======")
#         print("点位ID:", info["id"])
#         print("坐标: (%.2f, %.2f)" % (info["x"], info["y"]))
#         print("朝向:", info["yaw"])
#         print("转向指令:", info["turn"])

#         # 示例：控制逻辑
#         if info["turn"] == "left":
#             print(">>> 执行左转")
#         elif info["turn"] == "right":
#             print(">>> 执行右转")
#         elif info["turn"] == "straight":
#             print(">>> 直行")
#         elif info["turn"] == "uturn":
#             print(">>> 掉头")
#         elif info["turn"] == "stop":
#             print(">>> 停止")

#     cv2.imshow("camera", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
