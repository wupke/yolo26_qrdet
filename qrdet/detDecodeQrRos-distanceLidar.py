#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
FilePath: /ultralytics/qrdet/detDecodeQrRos-distanceLidar.py
author: wupke
Date: 2026-02-05 10:42:12
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-02-05 16:06:19
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
'''


######   ------ 【视觉+LiDAR融合测距版】：  适配 A3 相机RGB检测二维码 + Livox LiDAR 输出距离数据   ------ ###########
# QR内容识别 + LiDAR测距 融合版节点

'''

输入：相机 + LiDAR
输出：
✔ 二维码内容
✔ 二维码在图像中的位置
✔ 二维码到小车的真实距离（LiDAR）
✔ 可视化调试画面

功能	状态
实时二维码识别	✅
解析二维码JSON内容	✅
图像中画框 + 中心点	✅
LiDAR算真实距离	✅
日志输出识别信息	✅
稳定GUI显示	✅

---------- 下一步可升级

发布结果为ROS Topic（供规划/控制使用）

多二维码同时识别

置信度过滤

卡尔曼滤波稳定距离



'''



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy, cv2, json, numpy as np, message_filters, threading
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from camera_node.msg import StereoImage


class QRPerceptionNode:
    def __init__(self):
        rospy.init_node("qr_perception_node")

        self.detector = cv2.QRCodeDetector()

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




        # ========= 外参 LiDAR → Camera =========
        T = np.array([
            [1, 0, 0, 0.01],
            [0, 1, 0, 0.0],
            [0, 0, 1, 0.075],
            [0, 0, 0, 1]
        ])
        self.R = T[:3, :3]
        self.tvec = T[:3, 3].reshape(3,1)

        # 线程安全图像缓存
        self.lock = threading.Lock()
        self.latest_frame = None

        # 同步订阅
        img_sub = message_filters.Subscriber("/camera/stereo_image", StereoImage)
        pc_sub  = message_filters.Subscriber("/livox/lidar", PointCloud2)

        ts = message_filters.ApproximateTimeSynchronizer([img_sub, pc_sub], 10, 0.05)
        ts.registerCallback(self.callback)

        cv2.namedWindow("QR_Perception", cv2.WINDOW_NORMAL)
        rospy.loginfo("🚀 QR Perception Node Started")

    def rosimg_to_cv(self, img_msg):
        h,w,step = img_msg.height, img_msg.width, img_msg.step
        return np.frombuffer(img_msg.data,np.uint8).reshape(h,step)[:,:w*3].reshape(h,w,3)

    def callback(self, img_msg, pc_msg):

        frame = self.rosimg_to_cv(img_msg.rgb_image)

        data, bbox, _ = self.detector.detectAndDecode(frame)

        if data and bbox is not None:
            pts = bbox[0].astype(int)
            cv2.polylines(frame, [pts], True, (0,0,255), 2)

            cx = int(np.mean(pts[:,0]))
            cy = int(np.mean(pts[:,1]))
            cv2.circle(frame, (cx,cy), 5, (255,0,0), -1)

            # ========= 解析二维码内容 =========
            qr_text = data
            try:
                info = json.loads(data)
                qr_text = f"ID:{info['id']} Turn:{info['turn']}"
            except:
                pass

            # ========= LiDAR测距 =========
            x1,y1 = np.min(pts,axis=0)
            x2,y2 = np.max(pts,axis=0)
            w,h = x2-x1, y2-y1

            cx1,cy1 = x1+w*0.3, y1+h*0.3
            cx2,cy2 = x1+w*0.7, y1+h*0.7

            pts3d = np.array([p for p in pc2.read_points(pc_msg, field_names=("x","y","z"), skip_nans=True)
                              if 0.2<p[0]<15])

            distance = None

            if pts3d.size > 0:
                # LiDAR → Camera
                pts_cam = (pts3d @ self.R.T) + self.tvec.ravel()

                pts2d,_ = cv2.projectPoints(pts3d, cv2.Rodrigues(self.R)[0], self.tvec, self.K, self.D)
                pts2d = pts2d.reshape(-1,2)

                mask = (pts2d[:,0]>=cx1)&(pts2d[:,0]<=cx2)&(pts2d[:,1]>=cy1)&(pts2d[:,1]<=cy2)
                roi_pts = pts3d[mask]

                if roi_pts.size > 0:
                    distance = np.median(np.linalg.norm(roi_pts,axis=1))

            # ========= 画信息 =========
            cv2.putText(frame, qr_text, (cx-120, cy-40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            if distance:
                cv2.putText(frame, f"{distance:.2f} m", (cx-60, cy-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                rospy.loginfo(f"📦 QR:{qr_text}  📏 {distance:.2f} m")
            else:
                rospy.loginfo(f"📦 QR:{qr_text}  (no lidar hit)")

        # ⭐ 永远缓存避免黑屏
        with self.lock:
            self.latest_frame = frame.copy()

    def display_loop(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            with self.lock:
                if self.latest_frame is not None:
                    cv2.imshow("QR_Perception", self.latest_frame)
            cv2.waitKey(1)
            rate.sleep()


if __name__ == "__main__":
    node = QRPerceptionNode()
    node.display_loop()
















# # 完整融合版代码（QR + LiDAR 距离）  -------------  避免黑屏窗口  ------------ ok 

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import rospy
# import cv2
# import numpy as np
# import message_filters
# import sensor_msgs.point_cloud2 as pc2
# from sensor_msgs.msg import PointCloud2
# from camera_node.msg import StereoImage
# import threading


# class QRLidarFusion:
#     def __init__(self):
#         rospy.init_node("qr_lidar_fusion")

#         self.detector = cv2.QRCodeDetector()

#         # 相机参数
#         self.K = np.array([[818.5,0,289.7],
#                            [0,818.6,283.7],
#                            [0,0,1]])
#         self.D = np.array([0.109,-0.369,-0.007,0.0002,0])

#         # 外参 LiDAR → Camera
#         T = np.array([
#             [0, 1, 0, -0.05],
#             [0, 0, -1, -0.5],
#             [-1, 0, 0, 0.1],
#             [0, 0, 0, 1]
#         ])
#         R = T[:3, :3]
#         t = T[:3, 3]
#         self.rvec, _ = cv2.Rodrigues(R)
#         self.tvec = t.reshape(3,1)

#         # ===== 线程安全缓存 =====
#         self.lock = threading.Lock()
#         self.latest_frame = None

#         img_sub = message_filters.Subscriber("/camera/stereo_image", StereoImage)
#         pc_sub = message_filters.Subscriber("/livox/lidar", PointCloud2)

#         ts = message_filters.ApproximateTimeSynchronizer([img_sub, pc_sub], 10, 0.05)
#         ts.registerCallback(self.callback)

#         cv2.namedWindow("fusion", cv2.WINDOW_NORMAL)
#         rospy.loginfo("🚀 QR + LiDAR Fusion Started")

#     def rosimg_to_cv(self, img_msg):
#         h,w,step = img_msg.height, img_msg.width, img_msg.step
#         return np.frombuffer(img_msg.data, np.uint8).reshape(h,step)[:,:w*3].reshape(h,w,3)

#     # 🔵 ROS回调线程（不允许imshow）
#     def callback(self, img_msg, pc_msg):

#         frame = self.rosimg_to_cv(img_msg.rgb_image)
#         data, bbox, _ = self.detector.detectAndDecode(frame)

#         if bbox is not None:
#             pts = bbox[0].astype(int)
#             cv2.polylines(frame, [pts], True, (0,255,0), 2)

#             x1,y1 = np.min(pts,axis=0)
#             x2,y2 = np.max(pts,axis=0)
#             w,h = x2-x1, y2-y1
#             cx1,cy1 = x1+w*0.3, y1+h*0.3
#             cx2,cy2 = x1+w*0.7, y1+h*0.7

#             cv2.rectangle(frame,(int(cx1),int(cy1)),(int(cx2),int(cy2)),(255,0,0),2)

#             pts3d = np.array([p for p in pc2.read_points(pc_msg, field_names=("x","y","z"), skip_nans=True)
#                               if 0.2<p[0]<15])

#             if pts3d.size>0:
#                 pts2d,_ = cv2.projectPoints(pts3d,self.rvec,self.tvec,self.K,self.D)
#                 pts2d = pts2d.reshape(-1,2)

#                 mask = (pts2d[:,0]>=cx1)&(pts2d[:,0]<=cx2)&(pts2d[:,1]>=cy1)&(pts2d[:,1]<=cy2)
#                 roi_pts = pts3d[mask]

#                 if roi_pts.size>0:
#                     dist = np.median(np.linalg.norm(roi_pts,axis=1))
#                     cv2.putText(frame,f"{dist:.2f} m",(int(x1),int(y1)-10),
#                                 cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
#                     rospy.loginfo(f"📏 QR Distance: {dist:.2f} m")

#         # ⭐ 只缓存
#         with self.lock:
#             self.latest_frame = frame.copy()

#     # 🟢 主线程显示
#     def display_loop(self):
#         rate = rospy.Rate(30)
#         while not rospy.is_shutdown():
#             with self.lock:
#                 if self.latest_frame is not None:
#                     cv2.imshow("fusion", self.latest_frame)
#             cv2.waitKey(1)
#             rate.sleep()


# if __name__ == "__main__":
#     node = QRLidarFusion()
#     node.display_loop()









