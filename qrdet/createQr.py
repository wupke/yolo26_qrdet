#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
FilePath: /ultralytics/qrdet/createQr.py
author: wupke
Date: 2026-04-07 17:31:32
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-04-07 22:18:35
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
'''



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

os.makedirs("qr", exist_ok=True)

for p in points:
    data = json.dumps(p)
    
    qr = qrcode.QRCode(
        version=5,  
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=30,   

        border=4,
    )
    
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    filename = f'qr/point_{p["id"]}.png'
    img.save(filename)
    print("生成:", filename, "内容:", data)





