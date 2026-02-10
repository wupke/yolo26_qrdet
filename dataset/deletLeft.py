#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
FilePath: /ultralytics/dataset/deletLeft.py
author: wupke
Date: 2026-02-02 18:30:43
Version: 1.0
LastEditors: wupke
LastEditTime: 2026-02-02 18:30:48
Description:       
Copyright: Copyright (c) 2026 by ${git_name} email: ${git_email}, All Rights Reserved.
'''


import os
from tqdm import tqdm
import shutil

# === 1. 配置路径 ===
# 建议先在 dataset 目录下操作，或者直接在源目录操作（记得备份）
label_dir = 'dataset/labels' # 包含 train 和 val 的父目录

def filter_labels(target_dir):
    # 类别映射逻辑：
    # 原 0 (right)    -> 保持 0
    # 原 1 (left)     -> 删除
    # 原 2 (straight) -> 变为 1
    
    txt_files = []
    for root, dirs, files in os.walk(target_dir):
        for f in files:
            if f.endswith('.txt') and f != 'classes.txt':
                txt_files.append(os.path.join(root, f))

    print(f"🚀 开始清理 {len(txt_files)} 个标注文件...")

    for file_path in tqdm(txt_files):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            
            cls_id = parts[0]
            
            if cls_id == '0':
                # 保持 right 为 0
                new_lines.append(line)
            elif cls_id == '2':
                # 将 straight 从 2 改为 1
                parts[0] = '1'
                new_lines.append(" ".join(parts) + "\n")
            # 如果 cls_id == '1'，则直接跳过，不加入 new_lines

        # 将修改后的内容写回文件
        with open(file_path, 'w') as f:
            f.writelines(new_lines)

if __name__ == "__main__":
    # 执行清理
    filter_labels(label_dir)
    print("\n✅ 清理完成！已删除所有 left(1) 标签，并将 straight(2) 重映射为 1。")




















