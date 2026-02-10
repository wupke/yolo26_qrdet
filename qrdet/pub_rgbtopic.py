#!/usr/bin/env python3
"""
FilePath: /ultralytics/qrdet/pub_rgbtopic.py
author: wupke
Date: 2026-02-05 11:11:04
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-02-05 11:14:20
Description:
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
"""

#!/usr/bin/env python3
import rospy
from camera_node.msg import StereoImage
from sensor_msgs.msg import Image


class RGBRelayNode:
    def __init__(self):
        rospy.init_node("rgb_relay_node")

        # 发布标准图像话题
        self.pub = rospy.Publisher("/camera/rgb_image_raw", Image, queue_size=1)

        # 订阅自定义消息
        rospy.Subscriber("/camera/stereo_image", StereoImage, self.callback, queue_size=1, buff_size=2**24)

        rospy.loginfo("✅ RGB relay started: /camera/stereo_image → /camera/rgb_image_raw")

    def callback(self, msg):
        # 直接把 rgb_image 原样转发
        self.pub.publish(msg.rgb_image)


if __name__ == "__main__":
    RGBRelayNode()
    rospy.spin()
