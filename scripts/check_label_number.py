#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
FilePath: /ultralytics/scripts/check_label_number.py
author: wupke
Date: 2026-02-02 15:22:59
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-03-26 16:50:47
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
'''

import os
from collections import Counter

src_dir = '/home/wupke/traffic_det/save_pic_checked'
image_exts = ('.jpg', '.jpeg', '.png')

# 统计
file_count = 0
class_dist = Counter()
multi_label_files = 0

print(f"🕵️ 正在深度核查目录: {src_dir}")

files = [f for f in os.listdir(src_dir) if f.lower().endswith(image_exts)]

for f in files:
    fid = os.path.splitext(f)[0]
    t_path = os.path.join(src_dir, fid + '.txt')
    
    if os.path.exists(t_path):
        with open(t_path, 'r') as tf:
            lines = [l.strip() for l in tf.readlines() if l.strip()]
            if not lines: continue
            
            file_count += 1
            # 获取该文件中所有的类别ID
            labels_in_file = [line.split()[0] for line in lines]
            
            # 记录多标签情况
            if len(set(labels_in_file)) > 1:
                multi_label_files += 1
            
            # 使用脚本之前的判定逻辑：取出现次数最多的
            main_cls = max(set(labels_in_file), key=labels_in_file.count)
            class_dist[main_cls] += 1

print("\n--- 🔍 核查报告 ---")
print(f"✅ 有效图文配对总数: {file_count}")
print(f"📊 识别到的类别分布: {dict(sorted(class_dist.items()))}")
print(f"⚠️ 包含多种类别的复杂图片数: {multi_label_files}")

# 抽样打印前5个文件的解析结果，请手动比对
print("\n📝 抽样验证 (前5个文件):")
for f in files[:5]:
    fid = os.path.splitext(f)[0]
    print(f"文件: {fid}.txt -> 判定类别: {max(set([l.split()[0] for l in open(os.path.join(src_dir, fid+'.txt')).readlines()]), key=[l.split()[0] for l in open(os.path.join(src_dir, fid+'.txt')).readlines()].count)}")






### ------  统计单个类别的数量 ------- ### 


import os
from collections import Counter
from tqdm import tqdm

# === 配置路径与类别映射 ===
# src_dir = '/home/wupke/traffic_det/save_pic_checked'
src_dir = '/media/wupke/d3fe9cd3-5ca2-4f15-bcd1-b335370b92b81/workspace_m2/yolo26/ultralytics/rosbag_data_Qr/dataset_qr/'
class_map = {'0': 'right', '1': 'left', '2': 'straight'}

def count_yolo_labels(data_dir):
    label_counts = Counter()
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.txt') and f != 'classes.txt']
    
    print(f"🔍 正在扫描目录: {data_dir}")
    print(f"总计发现 {len(file_list)} 个标注文件\n")

    for file_name in tqdm(file_list, desc="统计中"):
        file_path = os.path.join(data_dir, file_name)
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) > 0:
                        cls_id = parts[0]
                        label_counts[cls_id] += 1
        except Exception as e:
            print(f"⚠️ 无法读取文件 {file_name}: {e}")

    # === 输出结果 ===
    print("\n" + "="*30)
    print(f"{'类别ID':<10} | {'类别名称':<10} | {'实例总数':<10}")
    print("-" * 30)
    
    total_instances = 0
    # 按照 0, 1, 2 的顺序打印
    for i in ['0', '1', '2']:
        count = label_counts.get(i, 0)
        name = class_map.get(i, 'Unknown')
        print(f"{i:<12} | {name:<10} | {count:<10}")
        total_instances += count
        
    print("-" * 30)
    print(f"{'总计':<12} | {'-':<10} | {total_instances:<10}")
    print("="*30)

if __name__ == "__main__":
    count_yolo_labels(src_dir)




# 🔍 正在扫描目录: /home/wupke/traffic_det/save_pic_checked
# 总计发现 353 个标注文件

# 统计中: 100%|████████████████████████████████████████████████████████████████████████████████████████| 353/353 [00:00<00:00, 37137.29it/s]

# ==============================
# 类别ID       | 类别名称       | 实例总数      
# ------------------------------
# 0            | right      | 216       
# 1            | left       | 182       
# 2            | straight   | 150       
# ------------------------------
# 总计           | -          | 548       
# ==============================



    