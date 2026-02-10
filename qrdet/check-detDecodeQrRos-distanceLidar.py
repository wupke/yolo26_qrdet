#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
FilePath: /ultralytics/qrdet/check-detDecodeQrRos-distanceLidar.py
author: wupke
Date: 2026-02-05 12:46:04
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-02-06 08:57:53
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
'''


'''

投影检测到的二维码区域，减少资源计算 → 都变成红点投到图像上

优化后的专业版本（只投影二维码区域点云）


下一步我可以带你做

✔ 自动微调外参
✔ 投影误差热力图
✔ 交互式标定工具

'''





###   ----   QR + LiDAR 融合（ROI视锥优化 + 投影调试 + CPU优化版

# ------------------- GPt  ---------------


















        # T = np.array([[0,1,0,0.01],
        #               [0,0,-1,0],
        #               [-1,0,0,0.075],
        #               [0,0,0,1]],np.float32)



#################     外参不对  

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

        self.K = np.array([[809.0,0,471.0],[0,808.0,355.0],[0,0,1]],np.float32)
        self.D = np.array([-0.1397,0.0121,0.00069,-0.00011,-0.00042],np.float32)

        T = np.array([[1,0,0,0.01],
                      [0,1,0,0],
                      [0,0,1,0.075],
                      [0,0,0,1]],np.float32)
        self.R = T[:3,:3]
        self.tvec = T[:3,3].reshape(3,1)
        self.rvec,_ = cv2.Rodrigues(self.R)

        self.lock = threading.Lock()
        self.latest_frame = np.zeros((720,960,3),dtype=np.uint8)
        self.last_time = time.time()
        self.filtered_dist = None

        # ⭐ 多帧点云缓存
        self.pc_buffer = []
        self.buffer_size = 3

        img_sub = message_filters.Subscriber("/camera/stereo_image", StereoImage)
        pc_sub  = message_filters.Subscriber("/livox/lidar", PointCloud2)
        ts = message_filters.ApproximateTimeSynchronizer([img_sub, pc_sub],10,0.05)
        ts.registerCallback(self.callback)

        cv2.namedWindow("QR_Perception",cv2.WINDOW_NORMAL)
        rospy.loginfo("🚀 QR Perception Node Started")

    def rosimg_to_cv(self, img_msg):
        h,w,step = img_msg.height,img_msg.width,img_msg.step
        return np.frombuffer(img_msg.data,np.uint8).reshape(h,step)[:,:w*3].reshape(h,w,3).copy()

    def ema_filter(self, dist):
        if self.filtered_dist is None:
            self.filtered_dist = dist
        else:
            self.filtered_dist = EMA_ALPHA*dist + (1-EMA_ALPHA)*self.filtered_dist
        return self.filtered_dist

    def callback(self,img_msg,pc_msg):

        frame = self.rosimg_to_cv(img_msg.rgb_image)
        h_img,w_img = frame.shape[:2]

        # ========= 点云 =========
        pts = np.array(list(pc2.read_points(pc_msg,("x","y","z"),skip_nans=True)),dtype=np.float32)
        if pts.size == 0:
            return

        mask = (np.abs(pts[:,0])<5)&(np.abs(pts[:,1])<5)&(0.1<pts[:,2])&(pts[:,2]<5)
        pts = pts[mask]

        self.pc_buffer.append(pts)
        if len(self.pc_buffer) > self.buffer_size:
            self.pc_buffer.pop(0)

        pts_all = np.vstack(self.pc_buffer)

        # ========= 投影 =========
        pts2d,_ = cv2.projectPoints(pts_all, self.rvec, self.tvec, self.K, self.D)
        pts2d = pts2d.reshape(-1,2)

        if DEBUG_PROJECTION:
            for x,y in pts2d[::3]:
                if np.isfinite(x) and np.isfinite(y):
                    x,y=int(x),int(y)
                    if 0<=x<w_img and 0<=y<h_img:
                        cv2.circle(frame,(x,y),2,(0,0,255),-1)

        # ========= QR识别 =========
        data,bbox,_ = self.detector.detectAndDecode(frame)

        if data and bbox is not None:
            pts_qr=bbox[0].astype(np.float32)
            cv2.polylines(frame,[pts_qr.astype(int)],True,(0,255,0),2)

            cx,cy=np.mean(pts_qr[:,0]),np.mean(pts_qr[:,1])

            # ===== ROI区域放大 =====
            x1,y1=np.min(pts_qr,axis=0)
            x2,y2=np.max(pts_qr,axis=0)
            w,h=x2-x1,y2-y1
            rx1,ry1=x1+w*0.2,y1+h*0.2
            rx2,ry2=x1+w*0.8,y1+h*0.8

            mask_roi=(pts2d[:,0]>=rx1)&(pts2d[:,0]<=rx2)&(pts2d[:,1]>=ry1)&(pts2d[:,1]<=ry2)
            roi_pts=pts_all[mask_roi]

            lidar_dist=None
            if roi_pts.shape[0]>5:
                lidar_dist=np.median(np.linalg.norm(roi_pts,axis=1))
                lidar_dist=self.ema_filter(lidar_dist)

            # ===== 距离融合 =====
            if lidar_dist:
                dist = lidar_dist
            else:
                dist = None

            qr_text=data
            try:
                info=json.loads(data)
                qr_text=f"ID:{info['id']} Turn:{info['turn']}"
            except: pass

            # ⭐ 永远显示
            cv2.putText(frame,qr_text,(int(cx)-120,int(cy)-45),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

            if dist:
                cv2.putText(frame,f"{dist:.2f} m",(int(cx)-60,int(cy)-15),
                            cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)
                rospy.loginfo(f"📦 {qr_text}  📏 {dist:.2f} m")

        # ========= FPS =========
        fps=1/(time.time()-self.last_time)
        self.last_time=time.time()
        cv2.putText(frame,f"FPS:{fps:.1f}",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

        with self.lock:
            self.latest_frame=frame

    def display_loop(self):
        rate=rospy.Rate(30)
        while not rospy.is_shutdown():
            with self.lock:
                cv2.imshow("QR_Perception",self.latest_frame)
            cv2.waitKey(1)
            rate.sleep()


if __name__=="__main__":
    node=QRPerceptionNode()
    node.display_loop()


















## ----- GPt  -----对应check-detDecodeQrRos-distanceLidar3.py：  
# 累计多帧投影，查看外参 没有检测



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy, cv2, json, numpy as np, message_filters, threading, time
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from camera_node.msg import StereoImage

DEBUG_PROJECTION = True
QR_SIZE = 0.17          # ⭐ 二维码真实边长（米）
EMA_ALPHA = 0.25        # ⭐ 距离时间滤波强度


class QRPerceptionNode:
    def __init__(self):
        rospy.init_node("qr_perception_node")

        self.detector = cv2.QRCodeDetector()

        self.K = np.array([[809.0,0,471.0],[0,808.0,355.0],[0,0,1]],np.float32)
        self.D = np.array([-0.1397,0.0121,0.00069,-0.00011,-0.00042],np.float32)

        T = np.array([[1,0,0,0.01],[0,1,0,0],[0,0,1,0.075],[0,0,0,1]],np.float32)
        self.R = T[:3,:3]
        self.tvec = T[:3,3].reshape(3,1)
        self.rvec,_ = cv2.Rodrigues(self.R)

        self.lock = threading.Lock()
        self.latest_frame = np.zeros((720,960,3),dtype=np.uint8)
        self.last_time = time.time()

        self.filtered_dist = None  # ⭐ EMA缓存

        img_sub = message_filters.Subscriber("/camera/stereo_image", StereoImage)
        pc_sub  = message_filters.Subscriber("/livox/lidar", PointCloud2)
        ts = message_filters.ApproximateTimeSynchronizer([img_sub, pc_sub],10,0.05)
        ts.registerCallback(self.callback)

        cv2.namedWindow("QR_Perception",cv2.WINDOW_NORMAL)
        rospy.loginfo("🚀 QR Perception Node Started (Advanced Mode)")

    def rosimg_to_cv(self, img_msg):
        h,w,step = img_msg.height,img_msg.width,img_msg.step
        return np.frombuffer(img_msg.data,np.uint8).reshape(h,step)[:,:w*3].reshape(h,w,3).copy()

    def ema_filter(self, dist):
        if self.filtered_dist is None:
            self.filtered_dist = dist
        else:
            self.filtered_dist = EMA_ALPHA*dist + (1-EMA_ALPHA)*self.filtered_dist
        return self.filtered_dist

    def callback(self,img_msg,pc_msg):

        frame = self.rosimg_to_cv(img_msg.rgb_image)
        h_img,w_img = frame.shape[:2]

        # ===== 读取当前帧点云 =====
        pts = np.array(list(pc2.read_points(pc_msg,
                                            field_names=("x","y","z"),
                                            skip_nans=True)), dtype=np.float32)

        if pts.size > 0:
            mask = (np.abs(pts[:,0]) < 5) & \
                (np.abs(pts[:,1]) < 5) & \
                (0.1 < pts[:,2]) & (pts[:,2] < 5)
            pts = pts[mask]

        # ===== 存入缓存 =====
        if pts.size > 0:
            self.pc_buffer.append(pts)
            if len(self.pc_buffer) > self.buffer_size:
                self.pc_buffer.pop(0)

        # ===== 融合多帧 =====
        if len(self.pc_buffer) == 0:
            return

        pts_all = np.vstack(self.pc_buffer)   # ⭐ 融合后的点云


        pts2d=None
        pts2d,_ = cv2.projectPoints(pts_all, self.rvec, self.tvec, self.K, self.D)
        pts2d = pts2d.reshape(-1,2)


        # ===== 全局外参验证投影 =====
        if DEBUG_PROJECTION:
            for x, y in pts2d[::3]:   # 稍微降采样避免太密
                x, y = int(x), int(y)
                if 0 <= x < w_img and 0 <= y < h_img:
                    cv2.circle(frame, (x, y), 2, (0,0,255), -1)



        data,bbox,_=self.detector.detectAndDecode(frame)

        if data and bbox is not None:
            pts_qr=bbox[0].astype(np.float32)
            cv2.polylines(frame,[pts_qr.astype(int)],True,(0,255,0),2)

            cx,cy=np.mean(pts_qr[:,0]),np.mean(pts_qr[:,1])

            # ========== PnP 求二维码平面 ==========
            half=QR_SIZE/2
            obj_pts=np.array([[-half, half,0],[half,half,0],[half,-half,0],[-half,-half,0]],np.float32)

            ok,rvec,tvec=cv2.solvePnP(obj_pts,pts_qr,self.K,self.D,flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                R_qr,_=cv2.Rodrigues(rvec)
                normal=R_qr[:,2]          # 平面法向量
                P0=tvec.reshape(3)        # 平面上一点

            # ROI区域
            x1,y1=np.min(pts_qr,axis=0)
            x2,y2=np.max(pts_qr,axis=0)
            w,h=x2-x1,y2-y1
            rx1,ry1=x1+w*0.3,y1+h*0.3
            rx2,ry2=x1+w*0.7,y1+h*0.7

            qr_text=data
            try:
                info=json.loads(data)
                qr_text=f"ID:{info['id']} Turn:{info['turn']}"
            except: pass

            # ========== LiDAR距离 + 残差热力图 ==========
            if pts2d is not None:
                mask=(pts2d[:,0]>=rx1)&(pts2d[:,0]<=rx2)&(pts2d[:,1]>=ry1)&(pts2d[:,1]<=ry2)
                roi_pts=pts[mask]
                roi_2d=pts2d[mask]

                if roi_pts.size>0:
                    # 距离
                    dist=np.median(np.linalg.norm(roi_pts,axis=1))
                    dist=self.ema_filter(dist)

                    # 平面残差
                    residuals=np.abs((roi_pts-P0)@normal)

                    # ⭐ 热力图
                    if DEBUG_PROJECTION:
                        for (x,y),r in zip(roi_2d,residuals):
                            x,y=int(x),int(y)
                            if 0<=x<w_img and 0<=y<h_img:
                                c=min(int(r*255/0.05),255)  # 5cm误差映射
                                # frame[y,x]=(0,c,255-c)
                                color = (0, c, 255-c)   # 你的热力图颜色
                                cv2.circle(frame, (x, y), 3, color, -1)  # 半径3像素实心圆



                    cv2.putText(frame,f"{dist:.2f} m",(int(cx)-60,int(cy)-20),
                                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
                    rospy.loginfo(f"📦 {qr_text}  📏 {dist:.2f} m")

            cv2.putText(frame,qr_text,(int(cx)-120,int(cy)-45),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
            

        fps=1/(time.time()-self.last_time)
        self.last_time=time.time()
        cv2.putText(frame,f"FPS:{fps:.1f}",(20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

        with self.lock:
            self.latest_frame=frame

    def display_loop(self):
        rate=rospy.Rate(30)
        while not rospy.is_shutdown():
            with self.lock:
                cv2.imshow("QR_Perception",self.latest_frame)
            cv2.waitKey(1)
            rate.sleep()


if __name__=="__main__":
    node=QRPerceptionNode()
    node.display_loop()

















# ## ----- GPt  -----对应check-detDecodeQrRos-distanceLidar2.py：  
# # 实际1m,检测结果为2-3.5m，误差较大


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import rospy, cv2, json, numpy as np, message_filters, threading, time
# import sensor_msgs.point_cloud2 as pc2
# from sensor_msgs.msg import PointCloud2
# from camera_node.msg import StereoImage

# DEBUG_PROJECTION = True
# QR_SIZE = 0.17          # ⭐ 二维码真实边长（米）
# EMA_ALPHA = 0.25        # ⭐ 距离时间滤波强度


# class QRPerceptionNode:
#     def __init__(self):
#         rospy.init_node("qr_perception_node")

#         self.detector = cv2.QRCodeDetector()

#         self.K = np.array([[809.0,0,471.0],[0,808.0,355.0],[0,0,1]],np.float32)
#         self.D = np.array([-0.1397,0.0121,0.00069,-0.00011,-0.00042],np.float32)

#         T = np.array([[1,0,0,0.01],[0,1,0,0],[0,0,1,0.075],[0,0,0,1]],np.float32)
#         self.R = T[:3,:3]
#         self.tvec = T[:3,3].reshape(3,1)
#         self.rvec,_ = cv2.Rodrigues(self.R)

#         self.lock = threading.Lock()
#         self.latest_frame = np.zeros((720,960,3),dtype=np.uint8)
#         self.last_time = time.time()

#         self.filtered_dist = None  # ⭐ EMA缓存

#         img_sub = message_filters.Subscriber("/camera/stereo_image", StereoImage)
#         pc_sub  = message_filters.Subscriber("/livox/lidar", PointCloud2)
#         ts = message_filters.ApproximateTimeSynchronizer([img_sub, pc_sub],10,0.05)
#         ts.registerCallback(self.callback)

#         cv2.namedWindow("QR_Perception",cv2.WINDOW_NORMAL)
#         rospy.loginfo("🚀 QR Perception Node Started (Advanced Mode)")

#     def rosimg_to_cv(self, img_msg):
#         h,w,step = img_msg.height,img_msg.width,img_msg.step
#         return np.frombuffer(img_msg.data,np.uint8).reshape(h,step)[:,:w*3].reshape(h,w,3).copy()

#     def ema_filter(self, dist):
#         if self.filtered_dist is None:
#             self.filtered_dist = dist
#         else:
#             self.filtered_dist = EMA_ALPHA*dist + (1-EMA_ALPHA)*self.filtered_dist
#         return self.filtered_dist

#     def callback(self,img_msg,pc_msg):

#         frame = self.rosimg_to_cv(img_msg.rgb_image)
#         h_img,w_img = frame.shape[:2]

#         pts = np.array(list(pc2.read_points(pc_msg,field_names=("x","y","z"),skip_nans=True)),dtype=np.float32)
#         if pts.size>0:
#             mask=(np.abs(pts[:,0])<5)&(np.abs(pts[:,1])<5)&(0.1<pts[:,2])&(pts[:,2]<5)
#             pts=pts[mask]

#         pts2d=None
#         if pts.size>0:
#             pts2d,_=cv2.projectPoints(pts,self.rvec,self.tvec,self.K,self.D)
#             pts2d=pts2d.reshape(-1,2)

#         data,bbox,_=self.detector.detectAndDecode(frame)

#         if data and bbox is not None:
#             pts_qr=bbox[0].astype(np.float32)
#             cv2.polylines(frame,[pts_qr.astype(int)],True,(0,255,0),2)

#             cx,cy=np.mean(pts_qr[:,0]),np.mean(pts_qr[:,1])

#             # ========== PnP 求二维码平面 ==========
#             half=QR_SIZE/2
#             obj_pts=np.array([[-half, half,0],[half,half,0],[half,-half,0],[-half,-half,0]],np.float32)

#             ok,rvec,tvec=cv2.solvePnP(obj_pts,pts_qr,self.K,self.D,flags=cv2.SOLVEPNP_IPPE_SQUARE)
#             if ok:
#                 R_qr,_=cv2.Rodrigues(rvec)
#                 normal=R_qr[:,2]          # 平面法向量
#                 P0=tvec.reshape(3)        # 平面上一点

#             # ROI区域
#             x1,y1=np.min(pts_qr,axis=0)
#             x2,y2=np.max(pts_qr,axis=0)
#             w,h=x2-x1,y2-y1
#             rx1,ry1=x1+w*0.3,y1+h*0.3
#             rx2,ry2=x1+w*0.7,y1+h*0.7

#             qr_text=data
#             try:
#                 info=json.loads(data)
#                 qr_text=f"ID:{info['id']} Turn:{info['turn']}"
#             except: pass

#             # ========== LiDAR距离 + 残差热力图 ==========
#             if pts2d is not None:
#                 mask=(pts2d[:,0]>=rx1)&(pts2d[:,0]<=rx2)&(pts2d[:,1]>=ry1)&(pts2d[:,1]<=ry2)
#                 roi_pts=pts[mask]
#                 roi_2d=pts2d[mask]

#                 if roi_pts.size>0:
#                     # 距离
#                     dist=np.median(np.linalg.norm(roi_pts,axis=1))
#                     dist=self.ema_filter(dist)

#                     # 平面残差
#                     residuals=np.abs((roi_pts-P0)@normal)

#                     # ⭐ 热力图
#                     if DEBUG_PROJECTION:
#                         for (x,y),r in zip(roi_2d,residuals):
#                             x,y=int(x),int(y)
#                             if 0<=x<w_img and 0<=y<h_img:
#                                 c=min(int(r*255/0.05),255)  # 5cm误差映射
#                                 # frame[y,x]=(0,c,255-c)
#                                 color = (0, c, 255-c)   # 你的热力图颜色
#                                 cv2.circle(frame, (x, y), 3, color, -1)  # 半径3像素实心圆



#                     cv2.putText(frame,f"{dist:.2f} m",(int(cx)-60,int(cy)-20),
#                                 cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
#                     rospy.loginfo(f"📦 {qr_text}  📏 {dist:.2f} m")

#             cv2.putText(frame,qr_text,(int(cx)-120,int(cy)-45),
#                         cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
            

#         fps=1/(time.time()-self.last_time)
#         self.last_time=time.time()
#         cv2.putText(frame,f"FPS:{fps:.1f}",(20,40),
#                     cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

#         with self.lock:
#             self.latest_frame=frame

#     def display_loop(self):
#         rate=rospy.Rate(30)
#         while not rospy.is_shutdown():
#             with self.lock:
#                 cv2.imshow("QR_Perception",self.latest_frame)
#             cv2.waitKey(1)
#             rate.sleep()


# if __name__=="__main__":
#     node=QRPerceptionNode()
#     node.display_loop()
















#######   -------- GPt  -----对应 check-detDecodeQrRos-distanceLidar.py  ： 可以跑通，但是误差大，需要调试外参  ###########

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import rospy, cv2, json, numpy as np, message_filters, threading, time
# import sensor_msgs.point_cloud2 as pc2
# from sensor_msgs.msg import PointCloud2
# from camera_node.msg import StereoImage

# DEBUG_PROJECTION = True   # 外参可视化验证


# class QRPerceptionNode:
#     def __init__(self):
#         rospy.init_node("qr_perception_node")

#         self.detector = cv2.QRCodeDetector()

#         self.K = np.array([[809.0,0,471.0],[0,808.0,355.0],[0,0,1]],np.float32)
#         self.D = np.array([-0.1397,0.0121,0.00069,-0.00011,-0.00042],np.float32)

#         T = np.array([[1,0,0,0.01],[0,1,0,0],[0,0,1,0.075],[0,0,0,1]],np.float32)
#         self.R = T[:3,:3]
#         self.tvec = T[:3,3].reshape(3,1)
#         self.rvec,_ = cv2.Rodrigues(self.R)

#         self.lock = threading.Lock()
#         self.latest_frame = np.zeros((720,960,3),dtype=np.uint8)
#         self.last_time = time.time()

#         img_sub = message_filters.Subscriber("/camera/stereo_image", StereoImage)
#         pc_sub  = message_filters.Subscriber("/livox/lidar", PointCloud2)
#         ts = message_filters.ApproximateTimeSynchronizer([img_sub, pc_sub],10,0.05)
#         ts.registerCallback(self.callback)

#         cv2.namedWindow("QR_Perception",cv2.WINDOW_NORMAL)
#         rospy.loginfo("🚀 QR Perception Node Started (Stable)")

#     # def rosimg_to_cv(self,img_msg):
#     #     h,w,step = img_msg.height,img_msg.width,img_msg.step
#     #     return np.frombuffer(img_msg.data,np.uint8).reshape(h,step)[:,:w*3].reshape(h,w,3)

#     def rosimg_to_cv(self, img_msg):
#         h, w, step = img_msg.height, img_msg.width, img_msg.step

#         frame = np.frombuffer(img_msg.data, np.uint8) \
#                     .reshape(h, step)[:, :w*3] \
#                     .reshape(h, w, 3)

#         return frame.copy()   # ⭐ 关键：变成可写内存



#     def callback(self,img_msg,pc_msg):

#         frame = self.rosimg_to_cv(img_msg.rgb_image)
#         h_img,w_img = frame.shape[:2]

#         # ========= 点云读取（超快写法） =========
#         pts = np.array(list(pc2.read_points(pc_msg,field_names=("x","y","z"),skip_nans=True)),dtype=np.float32)
#         if pts.size > 0:
#             mask = (np.abs(pts[:,0])<5)&(np.abs(pts[:,1])<5)&(0.1<pts[:,2])&(pts[:,2]<5)
#             pts = pts[mask]

#         # ========= 投影（只算一次） =========
#         pts2d = None
#         if pts.size > 0:
#             pts2d,_ = cv2.projectPoints(pts,self.rvec,self.tvec,self.K,self.D)
#             pts2d = pts2d.reshape(-1,2)

#             # 调试模式：降采样显示
#             if DEBUG_PROJECTION:
#                 for x,y in pts2d[::5]:
#                     x,y=int(x),int(y)
#                     if 0<=x<w_img and 0<=y<h_img:
#                         frame[y,x]=(0,0,255)

#         # ========= QR检测 =========
#         data,bbox,_ = self.detector.detectAndDecode(frame)

#         if data and bbox is not None:
#             pts_qr = bbox[0].astype(int)
#             cv2.polylines(frame,[pts_qr],True,(0,255,0),2)

#             cx,cy = np.mean(pts_qr[:,0]),np.mean(pts_qr[:,1])

#             x1,y1 = np.min(pts_qr,axis=0)
#             x2,y2 = np.max(pts_qr,axis=0)
#             w,h = x2-x1,y2-y1
#             rx1,ry1 = x1+w*0.3,y1+h*0.3
#             rx2,ry2 = x1+w*0.7,y1+h*0.7

#             qr_text = data
#             try:
#                 info=json.loads(data)
#                 qr_text=f"ID:{info['id']} Turn:{info['turn']}"
#             except: pass

#             # ========= ROI点云测距 =========
#             if pts2d is not None:
#                 mask = (pts2d[:,0]>=rx1)&(pts2d[:,0]<=rx2)&(pts2d[:,1]>=ry1)&(pts2d[:,1]<=ry2)
#                 roi_pts = pts[mask]
#                 if roi_pts.size>0:
#                     dist = np.median(np.linalg.norm(roi_pts,axis=1))
#                     cv2.putText(frame,f"{dist:.2f} m",(int(cx)-60,int(cy)-20),
#                                 cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
#                     rospy.loginfo(f"📦 {qr_text}  📏 {dist:.2f} m")

#             cv2.putText(frame,qr_text,(int(cx)-120,int(cy)-45),
#                         cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

#         # ========= FPS =========
#         fps=1/(time.time()-self.last_time)
#         self.last_time=time.time()
#         cv2.putText(frame,f"FPS:{fps:.1f}",(20,40),
#                     cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

#         # ⭐ 永远更新缓存（关键）
#         with self.lock:
#             self.latest_frame = frame

#     def display_loop(self):
#         rate=rospy.Rate(30)
#         while not rospy.is_shutdown():
#             with self.lock:
#                 cv2.imshow("QR_Perception",self.latest_frame)
#             cv2.waitKey(1)
#             rate.sleep()


# if __name__=="__main__":
#     node=QRPerceptionNode()
#     node.display_loop()





















# ------------------------ Gemini -------------------  待测试

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

        # ========= 相机参数 =========
        self.K = np.array([[809.0, 0, 471.0], [0, 808.0, 355.0], [0.0, 0.0, 1.0]])
        self.D = np.array([-0.139726073, 0.0121056843, 0.000691770362, -0.000109812168, -0.000424173560])

        # ========= 外参 (注意：Rodrigues 转换) =========
        self.R_mat = np.eye(3) 
        self.tvec = np.array([[0.01], [0.0], [0.075]])
        self.rvec, _ = cv2.Rodrigues(self.R_mat)

        self.lock = threading.Lock()
        self.latest_frame = None

        # 增加 slop 到 0.1s 以防同步失败，queue_size 调大
        img_sub = message_filters.Subscriber("/camera/stereo_image", StereoImage)
        pc_sub  = message_filters.Subscriber("/livox/lidar", PointCloud2)

        # 建议尝试将 slop 从 0.05 放大到 0.1
        ts = message_filters.ApproximateTimeSynchronizer([img_sub, pc_sub], 20, 0.1)
        ts.registerCallback(self.callback)

        cv2.namedWindow("QR_Perception", cv2.WINDOW_NORMAL)
        rospy.loginfo("🚀 Optimized QR Perception Node Started")

    def rosimg_to_cv(self, img_msg):
        try:
            h, w, step = img_msg.height, img_msg.width, img_msg.step
            # 更加健壮的解析方式
            raw_data = np.frombuffer(img_msg.data, np.uint8)
            # 兼容有些设备 step != w*3 的情况
            img = raw_data.reshape(h, step // (raw_data.size // (h * (step // (raw_data.size // h)))), -1)
            return img[:, :w, :3].copy()
        except Exception as e:
            rospy.logerr(f"Image Convert Error: {e}")
            return None

    def callback(self, img_msg, pc_msg):
        # 1. 优先获取图像，保证“不黑屏”
        frame = self.rosimg_to_cv(img_msg.rgb_image)
        if frame is None: return
        
        display_frame = frame.copy()

        # 2. 处理点云
        try:
            gen = pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True)
            pts3d_raw = np.array(list(gen))
            
            if pts3d_raw.size > 0:
                # 空间预裁剪：只看前方 5m
                mask_roi = (pts3d_raw[:, 0] > 0.1) & (pts3d_raw[:, 0] < 5.0) & \
                           (pts3d_raw[:, 1] > -3.0) & (pts3d_raw[:, 1] < 3.0)
                pts3d = pts3d_raw[mask_roi]

                if pts3d.size > 0:
                    # 投影验证点（下采样）
                    pts_sub = pts3d[::5]
                    pts2d_val, _ = cv2.projectPoints(pts_sub, self.rvec, self.tvec, self.K, self.D)
                    pts2d_val = pts2d_val.reshape(-1, 2).astype(int)
                    
                    for p in pts2d_val:
                        if 0 <= p[0] < frame.shape[1] and 0 <= p[1] < frame.shape[0]:
                            cv2.circle(display_frame, tuple(p), 1, (0, 255, 0), -1)
            else:
                pts3d = np.array([])
        except Exception as e:
            rospy.logwarn(f"Lidar Proc Warning: {e}")
            pts3d = np.array([])

        # 3. 二维码逻辑
        data, bbox, _ = self.detector.detectAndDecode(frame)
        if data and bbox is not None:
            pts = bbox[0].astype(int)
            cv2.polylines(display_frame, [pts], True, (0, 0, 255), 2)

            if pts3d.size > 0:
                # 计算中心 ROI 关联
                x1, y1 = np.min(pts, axis=0)
                x2, y2 = np.max(pts, axis=0)
                w, h = x2 - x1, y2 - y1
                
                full_pts2d, _ = cv2.projectPoints(pts3d, self.rvec, self.tvec, self.K, self.D)
                full_pts2d = full_pts2d.reshape(-1, 2)
                
                dist_mask = (full_pts2d[:,0] >= x1+w*0.3) & (full_pts2d[:,0] <= x1+w*0.7) & \
                            (full_pts2d[:,1] >= y1+h*0.3) & (full_pts2d[:,1] <= y1+h*0.7)
                
                roi_pts_3d = pts3d[dist_mask]
                if roi_pts_3d.size > 0:
                    distance = np.median(np.linalg.norm(roi_pts_3d, axis=1))
                    cv2.putText(display_frame, f"{distance:.2f}m", (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 更新全局变量
        with self.lock:
            self.latest_frame = display_frame

    def display_loop(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            with self.lock:
                if self.latest_frame is not None:
                    cv2.imshow("QR_Perception", self.latest_frame)
                else:
                    # 如果还是黑屏，在控制台提示
                    pass 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            rate.sleep()

if __name__ == "__main__":
    try:
        node = QRPerceptionNode()
        node.display_loop()
    except rospy.ROSInterruptException:
        pass