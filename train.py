#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
FilePath: /ultralytics/train.py
author: wupke
Date: 2026-02-02 15:43:33
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-02-02 15:45:19
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
'''


# -*- coding: utf-8 -*-
"""
FilePath: /ultralytics/ultralytics/train.py
author: wupke
Date: 2026-01-28 10:40:21
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-02-02 15:42:59
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
"""

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
 
if __name__ == '__main__':
    # 加载模型结构
    model = YOLO(model='/root/workspace/ultralytics-8.4.2/ultralytics/cfg/models/26/yolo26.yaml')
    
    # 加载预训练权重（可选）
    model.load('yolo26n.pt')  # 初次训练可加载，若做消融实验建议不加载
    
    # 开始训练
    model.train(data=r'data.yaml',
                imgsz=640,           # 输入图像尺寸
                epochs=200,          # 训练轮数
                batch=128,           # 批次大小
                workers=8,           # 数据加载线程数
                device='0',          # 使用GPU 0
                optimizer='SGD',     # 优化器类型
                close_mosaic=10,     # 最后10轮关闭Mosaic增强
                resume=False,        # 不从中断处继续
                project='runs/train',
                name='exp',          # 实验名称
                single_cls=False,    # 是否单类别训练
                cache=False,         # 是否缓存数据到内存
                )

