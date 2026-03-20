#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
FilePath: /yolo26/ultralytics/qrdet/qrcreate.py
author: wupke
Date: 2026-02-04 11:34:22
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-03-09 10:57:49
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
'''

'''
生成预设的二维码信息图像

'''


########  使用A4纸打印 


# import qrcode
# import json
# import os

# print("-------------------------脚本开始执行-------------------------")

# points = [
#     {"id": 1, "x": 4.5, "y": 1.55, "turn": "right", "yaw": 90},
#     {"id": 2, "x": 3.75, "y": -6.5, "turn": "right", "yaw": 180},
#     {"id": 3, "x": 3.75, "y": 2.1, "turn": "right", "yaw": 90},
#     {"id": 4, "x":9.0, "y": 1.55, "turn": "right", "yaw": 90},
#     {"id": 5, "x": 8.25, "y": -6.0, "turn": "right", "yaw": 180},
#     {"id": 6, "x": 8.25, "y": 1.55, "turn": "right", "yaw": 90},
#     {"id": 7, "x": 13.5, "y": 1.55, "turn": "right", "yaw": 90},
#     {"id": 8, "x": 0, "y": 0, "turn": "stop", "yaw": 0},

# ]

# os.makedirs("qr_codes20260227", exist_ok=True)

# for p in points:
#     data = json.dumps(p)
    
#     qr = qrcode.QRCode(
#         version=5,  # 小数据够用, 
#         # Version 3 二维码约 29×29 模块：29 × 10 = 290 px 290 px 打印在 A4 上太小 ❌
#         # Version 4 二维码约 41×41 模块：41 × 10 = 410 px 
#         # Version 5 二维码约 53×53 模块：53 × 10 = 530 px 
#         error_correction=qrcode.constants.ERROR_CORRECT_H,
#         box_size=30,    # box_size = 30, 29 × 30 = 870 px 打印出来大约 12~15cm，合适 ✅

#         border=4,
#     )
    
#     qr.add_data(data)
#     qr.make(fit=True)
#     img = qr.make_image(fill_color="black", back_color="white")
    
#     filename = f'qr_codes20260227/point_{p["id"]}.png'
#     img.save(filename)
#     print("生成:", filename, "内容:", data)






########  修改为 使用A3纸打印 

import qrcode
import json
import os

print("-------------------------脚本开始执行-------------------------")

points = [
    {"id": 1, "x": 4.5, "y": 1.55, "turn": "right", "yaw": 90},
    {"id": 2, "x": 3.75, "y": -6.5, "turn": "right", "yaw": 180},
    {"id": 3, "x": 3.75, "y": 2.1, "turn": "right", "yaw": 90},
    {"id": 4, "x":9.0, "y": 1.55, "turn": "right", "yaw": 90},
    {"id": 5, "x": 8.25, "y": -6.0, "turn": "right", "yaw": 180},
    {"id": 6, "x": 8.25, "y": 1.55, "turn": "right", "yaw": 90},
    {"id": 7, "x": 13.5, "y": 1.55, "turn": "right", "yaw": 90},
    {"id": 8, "x": 0, "y": 0, "turn": "stop", "yaw": 0},
]

os.makedirs("qr_codes20260309", exist_ok=True)

for p in points:
    data = json.dumps(p)
    
    qr = qrcode.QRCode(
        version=4,  # 小数据够用, 
        # Version 3 二维码约 29×29 模块：29 × 10 = 290 px 290 px 打印在 A4 上太小 ❌
        # Version 4 二维码约 41×41 模块：41 × 10 = 410 px 
        # Version 5 二维码约 53×53 模块：53 × 10 = 530 px 
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=40,    # box_size = 30, 29 × 30 = 870 px 打印出来大约 12~15cm，合适 ✅

        border=4,
    )
    
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    filename = f'qr_codes20260309/point_{p["id"]}.png'
    img.save(filename)
    print("生成:", filename, "内容:", data)











