#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
FilePath: /ultralytics/qrdet/check-detDecodeQrRos-distanceLidar24.py
author: wupke
Date: 2026-03-31 11:05:02
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-03-31 11:18:38
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
'''





# 代码未测试  ---- 基于v23版本进行优化：


'''
- 添加打印信息：距离，耗时，帧率等，方便调试和性能评估。
- 增加对小二维码的鲁棒性：在裁剪 ROI 时增加 padding ，并且在解码前先进行灰度化处理，减少 OpenCV 内部 floodFill 对杂色的误判。
- 统一发布格式：将二维码 ID、距离、地图坐标、朝向等信息打包成一个 JSON 对象，方便后续处理和扩展。

'''


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy, cv2, json, numpy as np, message_filters, threading, time
import sensor_msgs.point_cloud2 as pc2
from camera_node.msg import StereoImage
from std_msgs.msg import String
from ultralytics import YOLO

class QRPerceptionNode:
    def __init__(self):
        rospy.init_node("qr_perception_node")

        # 1. 加载 TensorRT 模型 (GPU 加速)
        # model_path = 'runs/detect/train4/weights/best_150epoch.engine'
        # self.model = YOLO(model_path, task='detect')
        # 请确保使用你在 export 成功后输出的绝对路径或正确相对路径
        model_path = '/home/nvidia/01yuyao/ultralytics/runs/detect/train4/weights/best_150epoch.onnx'
        # self.model = YOLO(model_path, task='detect')
        try:
            # YOLO 会自动识别 .onnx 后缀并加载 onnxruntime 后端
            self.model = YOLO(model_path, task='detect')
            rospy.loginfo(f"✅ Loaded ONNX model from {model_path}")
        except Exception as e:
            rospy.logerr(f"❌ Failed to load ONNX: {e}")

        self.qr_decoder = cv2.QRCodeDetector()

        # ========= 基于 ID 的独立滤波器存储 =========
        self.filter_dict = {}  # 格式: { "id_1": last_dist_1, "id_2": last_dist_2 }
        self.ema_alpha = 0.3   # 平滑系数，越小越平滑，越大实时性越好

        # ========= 相机与外参 (保持不变) =========
        self.K = np.array([[809.0, 0, 471.0], [0, 808.0, 355.0], [0, 0, 1.0]], np.float32)
        self.D = np.array([-0.1397, 0.0121, 0.00069, -0.00011, -0.00042], np.float32)
        self.R_lidar2cam = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], np.float32)
        self.t_lidar2cam = np.array([[0.01, 0.0, 0.075]], np.float32)

        # ========= ROS 通信 =========
        img_sub = message_filters.Subscriber("/camera/stereo_image", StereoImage)
        pc_sub = message_filters.Subscriber("/livox/lidar", pc2.PointCloud2)
        ts = message_filters.ApproximateTimeSynchronizer([img_sub, pc_sub], 10, 0.06)
        ts.registerCallback(self.callback)

        self.pub_result = rospy.Publisher("/qr_perception/result", String, queue_size=10)

        # - -- 新增：FPS 统计变量 ---
        self.prev_time = time.time()
        self.fps_list = []

        rospy.loginfo("🚀 QR Multi-ID Tracker with TensorRT Started")

    def apply_id_filter(self, qr_id, current_dist):
        """核心：针对每个 ID 独立的 EMA 滤波器"""
        if qr_id not in self.filter_dict:
            # 第一次见到该 ID，直接记录
            self.filter_dict[qr_id] = current_dist
        else:
            # 针对该 ID 进行平滑计算
            self.filter_dict[qr_id] = (self.ema_alpha * current_dist + 
                                       (1 - self.ema_alpha) * self.filter_dict[qr_id])
        
        return round(float(self.filter_dict[qr_id]), 3)

    def callback(self, img_msg, pc_msg):
        t_start = time.time()
        frame = self.rosimg_to_cv(img_msg.rgb_image)
        stamp = img_msg.header.stamp.to_sec()

        # --- A. 点云预处理 (一次性投影) ---
        pts_raw = np.array(list(pc2.read_points(pc_msg, ("x", "y", "z"), skip_nans=True)), np.float32)
        mask = (pts_raw[:, 0] > 0.1) & (pts_raw[:, 0] < 10.0)
        pts_filtered = pts_raw[mask]
        
        pts_cam = (pts_filtered @ self.R_lidar2cam.T) + self.t_lidar2cam
        z_mask = pts_cam[:, 2] > 0.1
        pts_cam = pts_cam[z_mask]
        pts_lidar = pts_filtered[z_mask]

        pts2d, _ = cv2.projectPoints(pts_cam, np.zeros(3), np.zeros(3), self.K, self.D)
        pts2d = pts2d.reshape(-1, 2)
        t_preprocess = (time.time() - t_start) * 1000 # 预处理耗时

        # --- B. GPU 推理检测 ---
        t_infer_start = time.time()
        results = self.model.predict(frame, conf=0.5, device=0, stream=True, verbose=False)
        t_inference = (time.time() - t_infer_start) * 1000 # 仅推理耗时

        current_frame_results = []

        for r in results:
            for box in r.boxes:
                # # 1. 提取框坐标
                # x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # # 2. 视觉解码 (Data Binding 开始)
                # roi_img = frame[max(0, y1-2):min(frame.shape[0], y2+2), 
                #                 max(0, x1-2):min(frame.shape[1], x2+2)]
                # # qr_content, _ = self.qr_decoder.decode(roi_img)
                # qr_content, points, _ = self.qr_decoder.detectAndDecode(roi_img)
                
                # 1. 提取框坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # --- 核心修复：更安全的 ROI 裁剪 ---
                # 增加 5 像素的 padding，防止框太紧导致 OpenCV 报错
                pad = 5
                img_h, img_w = frame.shape[:2]
                
                start_y = max(0, y1 - pad)
                end_y = min(img_h, y2 + pad)
                start_x = max(0, x1 - pad)
                end_x = min(img_w, x2 + pad)
                
                roi_img = frame[start_y:end_y, start_x:end_x]

                # 校验 ROI 尺寸：如果太小（比如小于 10x10），OpenCV 的解码器大概率会崩
                if roi_img.shape[0] < 10 or roi_img.shape[1] < 10:
                    continue

                # 3. 尝试解码
                try:

                    # # 使用 detectAndDecode，并增加一个 try-except 保护，防止 OpenCV 内部越界
                    # qr_content, points, _ = self.qr_decoder.detectAndDecode(roi_img)

                    # --- 预处理：灰度化处理 ---
                    # 灰度化不仅能提速，还能减少算法内部 floodFill 对杂色的误判
                    roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

                    # (可选) 增加对比度或轻微模糊，视现场光照环境而定
                    roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)

                    # 使用 detectAndDecode，传入灰度图
                    qr_content, points, _ = self.qr_decoder.detectAndDecode(roi_gray)

                except cv2.error as e:
                    rospy.logwarn(f"⚠️ OpenCV QR error: {e}")
                    qr_content = "" # 解码失败
                
                # 3. 空间测距
                bw, bh = x2 - x1, y2 - y1
                cx1, cy1, cx2, cy2 = x1+0.46*bw, y1+0.46*bh, x1+0.6*bw, y1+0.6*bh
                mask_roi = (pts2d[:, 0] >= cx1) & (pts2d[:, 0] <= cx2) & \
                           (pts2d[:, 1] >= cy1) & (pts2d[:, 1] <= cy2)
                
                target_pts = pts_lidar[mask_roi]
                
                if target_pts.shape[0] > 3 and qr_content:
                    raw_dist = np.median(np.linalg.norm(target_pts, axis=1))
                    
                    # --- 核心逻辑：数据解析与 ID 绑定 ---
                    try:
                        info = json.loads(qr_content)
                        qr_id = str(info.get("id", "unknown"))
                        
                        # 4. 执行基于 ID 的独立滤波
                        smooth_dist = self.apply_id_filter(qr_id, raw_dist)
                        
                        # 5. 打包强关联数据
                        item = {
                            "id": qr_id,
                            "distance": smooth_dist,
                            "map_x": info.get("x"),
                            "map_y": info.get("y"),
                            "yaw": info.get("yaw"),
                            "turn": info.get("turn"),
                            "conf": round(float(box.conf[0]), 2)
                        }
                        current_frame_results.append(item)
                    except Exception:
                        pass # 格式不符则跳过

        # --- C. 统一发布消息 ---
        # latency = (time.time() - t_start) * 1000
        # --- C. 计算 FPS 与 耗时 ---

        t_end = time.time()
        total_latency = (t_end - t_start) * 1000 # 总延迟 (ms)  
        
        # 计算 FPS
        duration = t_end - self.prev_time
        current_fps = 1.0 / duration if duration > 0 else 0
        self.prev_time = t_end
        
        # 平滑 FPS 显示
        self.fps_list.append(current_fps)
        if len(self.fps_list) > 10: self.fps_list.pop(0)
        avg_fps = sum(self.fps_list) / len(self.fps_list)

        # --- D. 统一发布与打印 ---        

        output = {
            "stamp": stamp,
            "latency_ms": round(total_latency, 2),
            "count": len(current_frame_results),
            "data": current_frame_results
        }
        self.pub_result.publish(json.dumps(output))

        if current_frame_results:
            # rospy.loginfo(f"📊 Frame Sync: {len(current_frame_results)} QRs | Top ID: {current_frame_results[0]['id']}")
            top_qr = current_frame_results[0]
            rospy.loginfo(
                f"📊 [FPS: {avg_fps:4.1f}] | "
                f"Total: {total_latency:5.1f}ms (Infer: {t_inference:4.1f}ms) | "
                f"Detected: {len(current_frame_results)} | "
                f"Top ID: {top_qr['id']} | Dist: {top_qr['distance']:.3f}m"
            )
        else:
            # 即使没检测到也打印一下 FPS 方便观察性能
            if int(time.time() * 10) % 20 == 0: # 降低无检测时的打印频率
                rospy.loginfo(f"🔎 Scanning... [FPS: {avg_fps:4.1f}] | Latency: {total_latency:5.1f}ms")

    def rosimg_to_cv(self, msg):
        return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3).copy()

if __name__ == "__main__":
    QRPerceptionNode()
    rospy.spin()



