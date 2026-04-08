#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
FilePath: /ultralytics/qrdet/extract_image_from_bag.py
author: wupke
Date: 2026-01-30 16:10:03
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-03-26 10:43:20
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
'''



#####  ---------------     version 2  for rosbags ---------------  #####

import os
import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import get_typestore, Stores
from pathlib import Path

# ================= 配置区域 =================
input_dir = Path('/media/wupke/d3fe9cd3-5ca2-4f15-bcd1-b335370b92b81/workspace_m2/yolo26/ultralytics/rosbag_data_Qr')  # 包含 .bag 文件的文件夹路径
topic_name = 'camera/rgb_image_raw20hz'  # 要提取的图像话题名称
save_root_dir = input_dir/'save_pic_Qr'  # 保存图片的根目录
# ===========================================

os.makedirs(save_root_dir, exist_ok=True)
typestore = get_typestore(Stores.ROS1_NOETIC)

# 获取文件夹下所有 .bag 文件并排序
bag_files = sorted(input_dir.glob('*.bag'))

if not bag_files:
    print(f"在 {input_dir} 中未找到任何 .bag 文件！")
    exit()

total_bags = len(bag_files)
global_count = 0
global_save_count = 0

print(f"找到 {total_bags} 个文件，开始处理...")

for idx, bag_path in enumerate(bag_files):
    print(f"\n[{idx+1}/{total_bags}] 正在处理: {bag_path.name}")
    
    count = 0
    save_count = 0
    
    # 获取不带后缀的文件名作为前缀
    bag_prefix = bag_path.stem 

    try:
        with AnyReader([bag_path], default_typestore=typestore) as reader:
            connections = [x for x in reader.connections if x.topic == topic_name]
            
            if not connections:
                print(f"跳过：话题 {topic_name} 在此包中不存在")
                continue

            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                count += 1

                # 每 2 帧取 1 帧 (保持原逻辑)
                if count % 2 != 0:
                    continue

                height = msg.height
                width = msg.width
                encoding = msg.encoding

                # 转换图像数据
                img = np.frombuffer(msg.data, dtype=np.uint8)

                if encoding == 'rgb8':
                    img = img.reshape((height, width, 3))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif encoding == 'bgr8':
                    img = img.reshape((height, width, 3))
                elif encoding == 'mono8':
                    img = img.reshape((height, width))
                else:
                    # 处理一些常见的其他格式，如压缩格式或不同位数
                    continue

                # 文件名：前缀_时间戳.jpg
                filename = os.path.join(save_root_dir, f"{bag_prefix}_{timestamp}.jpg")
                cv2.imwrite(filename, img)

                save_count += 1
                if save_count % 50 == 0:
                    print(f"  已从当前包保存 {save_count} 张图片...")

        print(f"完成 {bag_path.name}: 读取 {count} 帧, 保存 {save_count} 帧")
        global_count += count
        global_save_count += save_count

    except Exception as e:
        print(f"处理文件 {bag_path.name} 时出错: {e}")

print("-" * 30)
print(f"全部处理完成！")
print(f"总计处理包数: {total_bags}")
print(f"总计读取帧数: {global_count}")
print(f"总计保存图片: {global_save_count}")
















# ############################# version 1 for rosbag ##############
# import os
# import cv2
# import numpy as np
# from rosbags.highlevel import AnyReader
# from rosbags.typesys import get_typestore, Stores
# from pathlib import Path


# bag_path = Path('traffic3.bag')

# topic_name = '/camera/image_raw_10hz'
# save_dir = 'save_pic'

# os.makedirs(save_dir, exist_ok=True)

# typestore = get_typestore(Stores.ROS1_NOETIC)

# count = 0
# save_count = 0

# with AnyReader([bag_path], default_typestore=typestore) as reader:
#     connections = [x for x in reader.connections if x.topic == topic_name]

#     for connection, timestamp, rawdata in reader.messages(connections=connections):
#         msg = reader.deserialize(rawdata, connection.msgtype)
#         count += 1

#         if count % 2 != 0:
#             continue

#         height = msg.height
#         width = msg.width
#         encoding = msg.encoding

#         img = np.frombuffer(msg.data, dtype=np.uint8)

#         if encoding == 'rgb8':
#             img = img.reshape((height, width, 3))
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         elif encoding == 'bgr8':
#             img = img.reshape((height, width, 3))
#         elif encoding == 'mono8':
#             img = img.reshape((height, width))
#         else:
#             print(f"Unsupported encoding: {encoding}")
#             continue

#         filename = os.path.join(save_dir, f"{timestamp}.jpg")
#         cv2.imwrite(filename, img)

#         save_count += 1
#         print(f"Saved {filename}")

# print(f"Done. Total frames: {count}, Saved: {save_count}")
