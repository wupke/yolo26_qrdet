#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
FilePath: /ultralytics/scripts/split_yolo_oneClasses.py
author: wupke
Date: 2026-03-26 16:44:58
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-03-26 16:55:46
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
'''


# ######  ---------------------   split_data: train val -----    ##################### 

# # ----------    1类组合分布划分    ----------


import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm
from pathlib import Path

# === 1. 配置 ===
# 输入路径：包含 .jpg 和 .txt 的原始文件夹
src_dir = '/media/wupke/d3fe9cd3-5ca2-4f15-bcd1-b335370b92b81/workspace_m2/yolo26/ultralytics/rosbag_data_Qr/save_pic_Qr/'
# 输出路径：生成的 YOLO 格式数据集目录
save_dir = '/media/wupke/d3fe9cd3-5ca2-4f15-bcd1-b335370b92b81/workspace_m2/yolo26/ultralytics/rosbag_data_Qr/dataset_qr/'

train_ratio = 0.85
image_exts = ('.jpg', '.jpeg', '.png')

# === 2. 扫描并过滤数据 ===
valid_ids = []
labels_meta = []

print("🔍 正在扫描数据并分析标签...")
# 使用 Path 更好处理路径
src_path = Path(src_dir)

for img_file in src_path.iterdir():
    if img_file.suffix.lower() in image_exts:
        fid = img_file.stem
        txt_path = src_path / f"{fid}.txt"
        
        # 必须同时存在图片和标签文件
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                content = [line.split()[0] for line in f.readlines() if line.strip()]
            
            # 如果你有背景图（无标签），可以自行决定是否包含
            # 这里默认只包含有 qr 目标的图
            if len(content) > 0:
                valid_ids.append(fid)
                # 因为只有一类，组合固定为 "0"
                labels_meta.append("class_0") 

if not valid_ids:
    print("❌ 未发现有效的图片+标签对，请检查路径！")
    exit()

print(f"📊 找到有效样本数: {len(valid_ids)}")

# === 3. 核心划分 ===
# 虽然只有一类，stratify 依然可以保证数据被打散得更均匀
train_ids, val_ids = train_test_split(
    valid_ids, 
    test_size=(1 - train_ratio), 
    random_state=42, 
    stratify=labels_meta 
)

# === 4. 执行文件分发 ===
def prepare_dirs():
    if os.path.exists(save_dir):
        print(f"🧹 清理旧目录: {save_dir}")
        shutil.rmtree(save_dir)
    for d in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(save_dir, d), exist_ok=True)

prepare_dirs()

def dispatch(ids, split_name):
    print(f"🚚 正在分发 {split_name} 数据 (共 {len(ids)} 组)...")
    for fid in tqdm(ids):
        # 匹配图片后缀
        img_name = None
        for ex in image_exts:
            if (src_path / (fid + ex)).exists():
                img_name = fid + ex
                break
        
        if img_name:
            # 复制图片
            shutil.copy2(src_path / img_name, os.path.join(save_dir, 'images', split_name, img_name))
            # 复制标签
            shutil.copy2(src_path / f"{fid}.txt", os.path.join(save_dir, 'labels', split_name, f"{fid}.txt"))

dispatch(train_ids, 'train')
dispatch(val_ids, 'val')

# === 5. 最终质量检查 ===
def verify_instance_dist(split_name):
    counts = Counter()
    label_dir = os.path.join(save_dir, 'labels', split_name)
    for f in os.listdir(label_dir):
        with open(os.path.join(label_dir, f), 'r') as tf:
            for line in tf:
                cls_id = line.split()[0]
                counts[cls_id] += 1
    return counts

train_dist = verify_instance_dist('train')
val_dist = verify_instance_dist('val')

print("\n✅ 最终划分结果（类别 0: qr）：")
t_c = train_dist.get('0', 0)
v_c = val_dist.get('0', 0)
total = t_c + v_c
ratio = v_c / total if total > 0 else 0
print(f"训练集实例数: {t_c:<4}")
print(f"验证集实例数: {v_c:<4}")
print(f"验证集占比: {ratio:.2%}")

# 生成数据后提示
print(f"\n🚀 数据集已就绪！路径: {save_dir}")



