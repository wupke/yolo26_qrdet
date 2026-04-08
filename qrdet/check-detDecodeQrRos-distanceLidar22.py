#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
FilePath: /ultralytics/qrdet/check-detDecodeQrRos-distanceLidar22.py
author: wupke
Date: 2026-03-27 14:55:38
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-03-27 15:11:54
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
'''

'''
#如果在同一帧中检测到多个二维码，在根据置信度进行切换的时候，对应的距离信息会不会出现跳变？
# 如果是的话，是否需要引入一些平滑机制（比如 EMA）来稳定距离输出？ --- 已引入 EMA 平滑机制，减少距离跳变的影响。   
#在实际中需要将二维码的数据与距离信息进行关联，确保发布的消息中包含正确的二维码 ID 和对应的距离，否则会导致接收到的距离信息混乱。 
具体来说，在处理每个二维码时，会同时提取其对应的距离信息，并将二者一起打包到发布的消息中。
---------：

- 消除跳变：
游节点（如决策层）现在可以同时看到 ID=1 和 ID=2 的坐标。它们可以根据 ID 进行平滑跟踪，而不是被迫在两个目标间反复横跳。

- 更强的空间感知：
在自动驾驶中，如果你在路口看到两个二维码，你实际上获得了一个更准确的定位参考（类似多点定位）。

- 计算效率：
由于最耗时的点云投影操作（projectPoints）是在循环外完成的，即使画面中有 3-5 个二维码，
增加的计算量也仅仅是几次简单的 numpy 掩码计算和 QR 解码，性能损耗极低。

'''

# 代码未测试，并且可以继续优化：要给 EMA_FILTER 增加一个 dict 维护，针对每个 ID 独立进行距离平滑，互不干扰。




#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy, cv2, json, numpy as np, message_filters, threading, time
import sensor_msgs.point_cloud2 as pc2
from camera_node.msg import StereoImage
from std_msgs.msg import String
from ultralytics import YOLO  # 需安装 ultralytics

EMA_ALPHA = 0.25

class QRPerceptionNode:
    def __init__(self):
        rospy.init_node("qr_perception_node")

        # 1. 加载 YOLO 模型 (建议放在工程的 weights 目录下)
        self.model = YOLO("best_with65.pt") 
        self.qr_decoder = cv2.QRCodeDetector() # 仅用于解析内容

        # ========= 相机与外参 (保持不变) =========
        self.K = np.array([[809.0, 0, 471.0], [0, 808.0, 355.0], [0, 0, 1.0]], np.float32)
        self.D = np.array([-0.1397, 0.0121, 0.00069, -0.00011, -0.00042], np.float32)
        self.R_lidar2cam = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], np.float32)
        self.t_lidar2cam = np.array([[0.01, 0.0, 0.075]], np.float32)

        # ========= 状态变量 =========
        self.filtered_dist = None
        self.last_time = time.time()
        
        # ========= ROS 通信 =========
        img_sub = message_filters.Subscriber("/camera/stereo_image", StereoImage)
        pc_sub = message_filters.Subscriber("/livox/lidar", pc2.PointCloud2)
        
        # 考虑到 YOLO 耗时，同步容差可适当放宽至 0.05-0.1
        ts = message_filters.ApproximateTimeSynchronizer([img_sub, pc_sub], 10, 0.08)
        ts.registerCallback(self.callback)

        self.pub_result = rospy.Publisher("/qr_perception/result", String, queue_size=10)
        rospy.loginfo("🚀 YOLO QR Perception Node Ready")

    def callback(self, img_msg, pc_msg):
            t0 = time.time()
            frame = self.rosimg_to_cv(img_msg.rgb_image)
            stamp = img_msg.header.stamp.to_sec()

            # 1. GPU 推理获取所有结果
            results = self.model.predict(frame, conf=0.5, device=0, stream=True, verbose=False)
            
            detected_items = [] # 用于存储当前帧所有识别到的 QR 信息
            
            # 预先处理点云（投影到图像坐标系，避免在循环内重复计算）
            pts_raw = np.array(list(pc2.read_points(pc_msg, ("x", "y", "z"), skip_nans=True)), np.float32)
            mask_near = (pts_raw[:, 0] > 0.1) & (pts_raw[:, 0] < 8.0) # 适当扩大检测距离
            pts_filtered = pts_raw[mask_near]
            
            pts_cam = (pts_filtered @ self.R_lidar2cam.T) + self.t_lidar2cam
            z_mask = pts_cam[:, 2] > 0.1
            pts_cam = pts_cam[z_mask]
            pts_lidar_world = pts_filtered[z_mask]
            
            pts2d, _ = cv2.projectPoints(pts_cam, np.zeros(3), np.zeros(3), self.K, self.D)
            pts2d = pts2d.reshape(-1, 2)

            # 2. 遍历 YOLO 检测到的每一个框
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    
                    # --- A. 解析该框内的二维码内容 ---
                    roi_img = frame[max(0, y1-5):min(frame.shape[0], y2+5), 
                                    max(0, x1-5):min(frame.shape[1], x2+5)]
                    qr_data, _ = self.qr_decoder.decode(roi_img)
                    
                    # --- B. 计算该框对应的点云距离 ---
                    bw, bh = x2 - x1, y2 - y1
                    cx1, cy1 = x1 + 0.3 * bw, y1 + 0.3 * bh
                    cx2, cy2 = x1 + 0.7 * bw, y1 + 0.7 * bh

                    mask_roi = (pts2d[:, 0] >= cx1) & (pts2d[:, 0] <= cx2) & \
                            (pts2d[:, 1] >= cy1) & (pts2d[:, 1] <= cy2)
                    
                    target_pts = pts_lidar_world[mask_roi]
                    
                    dist = 0.0
                    if target_pts.shape[0] > 5: # 确保有点云落入
                        dist = float(np.median(np.linalg.norm(target_pts, axis=1)))
                    
                    # --- C. 封装单个 QR 数据 ---
                    item_info = {
                        "conf": round(conf, 2),
                        "box": [x1, y1, x2, y2],
                        "distance": round(dist, 3),
                        "raw_qr": qr_data
                    }
                    
                    # 尝试解析内容 JSON
                    if qr_data:
                        try:
                            item_info.update(json.loads(qr_data))
                        except:
                            pass
                    
                    detected_items.append(item_info)

            # 3. 统一发布结果列表
            latency = (time.time() - t0) * 1000
            output = {
                "stamp": stamp,
                "latency_ms": round(latency, 2),
                "count": len(detected_items),
                "results": detected_items
            }
            self.pub_result.publish(json.dumps(output))
            
            if len(detected_items) > 0:
                rospy.loginfo(f"🔍 Found {len(detected_items)} QR(s) | Latency: {latency:.1f}ms")


    def process_and_pub(self, data, dist, stamp, box, frame):
        qr_info = {"id": None, "x": None, "y": None, "turn": None, "yaw": None}
        if data:
            try:
                qr_info.update(json.loads(data))
            except:
                pass

        valid = True if (data and dist) else False
        
        msg = {
            "valid": valid,
            "id": qr_info["id"],
            "map_x": qr_info["x"],
            "map_y": qr_info["y"],
            "turn": qr_info["turn"],
            "yaw": qr_info["yaw"],
            "distance": round(float(dist), 3) if dist else 0,
            "stamp": stamp
        }
        self.pub_result.publish(json.dumps(msg))

        # 可视化检测框
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        if dist:
            cv2.putText(frame, f"Dist: {dist:.2f}m", (box[0], box[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 辅助函数保持逻辑一致
    def rosimg_to_cv(self, msg):
        h, w = msg.height, msg.width
        return np.frombuffer(msg.data, np.uint8).reshape(h, w, 3).copy()

    def ema_filter(self, dist):
        if self.filtered_dist is None: self.filtered_dist = dist
        else: self.filtered_dist = EMA_ALPHA * dist + (1 - EMA_ALPHA) * self.filtered_dist
        return self.filtered_dist

    def publish_result(self, valid, stamp):
        self.pub_result.publish(json.dumps({"valid": valid, "stamp": stamp}))

    def display_loop(self):
        rospy.spin()

if __name__ == "__main__":
    QRPerceptionNode().display_loop()



