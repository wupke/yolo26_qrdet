#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
FilePath: /ultralytics/scripts/ros_yolo_detector1.py
author: wupke
Date: 2026-02-03 12:48:17
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-02-03 13:51:29
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
    订阅相机话题 "/camera/image_raw",进行推理
    
    发布检测后的图像"/yolo/detected_image", 可以在 rviz 中可视化查看检测结果

"""





import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from ultralytics import YOLO
import numpy as np

class YoloRosDetector:
    def __init__(self):
        # 1. 初始化 ROS 节点
        rospy.init_node('yolo_detector_node', anonymous=True)
        
        # 2. 加载模型 (确保路径正确)
        model_path = "runs/detect/train_s-label2/weights/best.pt"
        self.model = YOLO(model_path)
        
        # 3. 初始化 CvBridge
        self.bridge = CvBridge()
        
        # 4. 订阅者：订阅原始图像
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.callback)
        
        # 5. 发布者：发布检测后的图像（方便在 rviz 查看）
        self.image_pub = rospy.Publisher("/yolo/detected_image", Image, queue_size=1)
        
        rospy.loginfo("YOLO Detector Node Started. Subscribing to /camera/image_raw...")

    def callback(self, data):
        try:
            # 1. 彻底绕过 cv_bridge，直接将 ROS Image 转换为 Numpy 数组
            # 假设你的 /camera/image_raw 是常见的 bgr8 格式
            img_array = np.frombuffer(data.data, dtype=np.uint8)
            cv_image = img_array.reshape(data.height, data.width, 3)
            # 如果你的画面颜色看起来不对（比如蓝红颠倒），可以尝试取消下面这行的注释
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            # 2. 执行 YOLO 推理
            results = self.model.predict(cv_image, conf=0.3, device='0', verbose=False)[0]
            # 3. 绘制检测框
            annotated_frame = results.plot()
            # 4. 手动构造 ROS 图像消息回传 (同样避开 cv_bridge)
            msg = Image()
            msg.header = data.header
            msg.height = annotated_frame.shape[0]
            msg.width = annotated_frame.shape[1]
            msg.encoding = "bgr8"
            msg.is_bigendian = 0
            msg.step = annotated_frame.shape[1] * 3
            msg.data = annotated_frame.tobytes()

            self.image_pub.publish(msg)
            
        except Exception as e:
            rospy.logerr(f"Prediction Error: {e}")

if __name__ == '__main__':
    try:
        detector = YoloRosDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
