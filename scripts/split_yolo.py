
######  ---------------------   split_data: train val -----    ##################### 

# ----------    3类组合分布划分（更细粒度的分层抽样）    ----------

import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

# === 1. 配置 ===
# src_dir = '/home/wupke/traffic_det/save_pic_checked'
# save_dir = '/home/wupke/traffic_det/dataset'

src_dir = '/media/wupke/d3fe9cd3-5ca2-4f15-bcd1-b335370b92b81/workspace_m2/yolo26/ultralytics/rosbag_data_Qr/save_pic_Qr/'
save_dir = '/media/wupke/d3fe9cd3-5ca2-4f15-bcd1-b335370b92b81/workspace_m2/yolo26/ultralytics/rosbag_data_Qr/save_pic_Qr/dataset/'

train_ratio = 0.85

# === 2. 获取每张图的标签组合 (Meta-class) ===
def get_label_combination(txt_path):
    try:
        with open(txt_path, 'r') as f:
            # 读取所有类别ID，去重并排序，如 [0, 1] 变成 "0_1"
            labels = sorted(list(set([line.split()[0] for line in f.readlines() if line.strip()])))
            return "_".join(labels) if labels else None
    except:
        return None

# === 3. 扫描数据 ===
image_exts = ('.jpg', '.jpeg', '.png')
valid_ids = []
meta_classes = []

print("🔍 正在分析标签组合分布...")
for f in os.listdir(src_dir):
    if f.lower().endswith(image_exts):
        fid = os.path.splitext(f)[0]
        t_path = os.path.join(src_dir, fid + '.txt')
        if os.path.exists(t_path):
            comb = get_label_combination(t_path)
            if comb:
                valid_ids.append(fid)
                meta_classes.append(comb)

# 打印组合分布，让你心里有数
print(f"📊 标签组合分布: {dict(Counter(meta_classes))}")

# === 4. 核心划分：基于组合进行分层 ===
# 这样能保证 [0,1,2] 同时出现的图片在两个集中比例一致
train_ids, val_ids = train_test_split(
    valid_ids, 
    test_size=(1 - train_ratio), 
    random_state=42, 
    stratify=meta_classes # 关键：按组合划分
)

# === 5. 执行分发 ===
def prepare_dirs():
    if os.path.exists(save_dir): shutil.rmtree(save_dir)
    for d in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(save_dir, d), exist_ok=True)

prepare_dirs()

def dispatch(ids, split_name):
    print(f"🚚 正在分发 {split_name} 数据...")
    for fid in tqdm(ids):
        img_name = next((fid + ex for ex in image_exts if os.path.exists(os.path.join(src_dir, fid + ex))), None)
        if img_name:
            shutil.copy2(os.path.join(src_dir, img_name), os.path.join(save_dir, 'images', split_name, img_name))
            shutil.copy2(os.path.join(src_dir, fid + '.txt'), os.path.join(save_dir, 'labels', split_name, fid + '.txt'))

dispatch(train_ids, 'train')
dispatch(val_ids, 'val')

# === 6. 最终质量检查 ===
def verify_instance_dist(split_name):
    counts = Counter()
    label_dir = os.path.join(save_dir, 'labels', split_name)
    for f in os.listdir(label_dir):
        with open(os.path.join(label_dir, f), 'r') as tf:
            for line in tf: counts[line.split()[0]] += 1
    return counts

train_dist = verify_instance_dist('train')
val_dist = verify_instance_dist('val')

print("\n✅ 最终划分结果（实例级别）：")
for i in ['0', '1', '2']:
    t_c = train_dist.get(i, 0)
    v_c = val_dist.get(i, 0)
    ratio = v_c / (t_c + v_c) if (t_c + v_c) > 0 else 0
    print(f"类别 {i} -> 训练集: {t_c:<4} | 验证集: {v_c:<4} | 验证集占比: {ratio:.2%}")




















################    ---------------------   split_data: train val -----    #####################

# import os
# import shutil
# from sklearn.model_selection import train_test_split
# from collections import Counter
# from tqdm import tqdm

# # === 1. 配置路径与参数 ===
# src_dir = '/home/wupke/traffic_det/save_pic_checked'
# save_dir = '/home/wupke/traffic_det/dataset' # 输出的根目录

# # 仅保留 train 和 val，两者之和需为 1.0
# train_ratio = 0.85
# val_ratio = 0.15 

# # === 2. 创建目录结构 ===
# def create_dirs(base_path):
#     # 只创建 train 和 val 相关的文件夹
#     sub_dirs = [
#         'images/train', 'images/val', 
#         'labels/train', 'labels/val'
#     ]
#     for d in sub_dirs:
#         os.makedirs(os.path.join(base_path, d), exist_ok=True)

# create_dirs(save_dir)

# # === 3. 提取主类别（用于分层抽样） ===
# def get_main_class(txt_path):
#     """读取 YOLO txt 第一行第一个数字作为主类别标签"""
#     try:
#         with open(txt_path, 'r') as f:
#             lines = f.readlines()
#             if not lines: return None
#             # 提取每行开头的 class_id
#             classes = [line.split()[0] for line in lines if line.strip()]
#             # 返回出现次数最多的类别 ID，用于代表这张图的特征
#             return max(set(classes), key=classes.count) if classes else None
#     except Exception as e:
#         print(f"⚠️ 无法读取 {txt_path}: {e}")
#         return None

# # === 4. 扫描并配对数据 ===
# print("🔍 正在扫描原始数据...")
# all_files = os.listdir(src_dir)
# image_exts = ('.jpg', '.jpeg', '.png', '.bmp')

# valid_ids = []
# valid_labels = []

# for f in all_files:
#     if f.lower().endswith(image_exts):
#         file_id = os.path.splitext(f)[0]
#         txt_path = os.path.join(src_dir, file_id + '.txt')
        
#         if os.path.exists(txt_path):
#             main_cls = get_main_class(txt_path)
#             if main_cls is not None:
#                 valid_ids.append(file_id)
#                 valid_labels.append(main_cls)

# print(f"📊 发现有效配对数据: {len(valid_ids)} 组")

# # === 5. 一次性划分 Train 和 Val ===
# # 使用 stratify 确保 train 和 val 中的类别比例与原始数据一致
# train_ids, val_ids = train_test_split(
#     valid_ids, 
#     test_size=val_ratio, 
#     random_state=42, 
#     stratify=valid_labels
# )

# # === 6. 执行文件分发 (复制) ===
# def dispatch_files(ids, split_name):
#     print(f"🚚 正在分发 {split_name} 数据 (共 {len(ids)} 组)...")
#     for file_id in tqdm(ids):
#         # 匹配图片名（处理可能的后缀）
#         img_name = ""
#         for ext in image_exts:
#             if os.path.exists(os.path.join(src_dir, file_id + ext)):
#                 img_name = file_id + ext
#                 break
        
#         if not img_name: continue
        
#         # 定义源和目标路径
#         src_img = os.path.join(src_dir, img_name)
#         src_txt = os.path.join(src_dir, file_id + '.txt')
        
#         dst_img = os.path.join(save_dir, 'images', split_name, img_name)
#         dst_txt = os.path.join(save_dir, 'labels', split_name, file_id + '.txt')
        
#         # 执行复制
#         shutil.copy2(src_img, dst_img)
#         shutil.copy2(src_txt, dst_txt)

# dispatch_files(train_ids, 'train')
# dispatch_files(val_ids, 'val')

# # === 7. 统计输出 ===
# def print_stats(ids, name):
#     cls_counts = Counter()
#     for fid in ids:
#         cls = get_main_class(os.path.join(src_dir, fid + '.txt'))
#         cls_counts[cls] += 1
#     print(f"  {name}: {len(ids)} samples, Class distribution: {dict(sorted(cls_counts.items()))}")

# print("\n✅ 数据集划分完成！")
# print_stats(train_ids, "Train")
# print_stats(val_ids, "Val")
# print(f"\n📂 结果目录结构如下：")
# print(f"{save_dir}/\n├── images/\n│   ├── train/\n│   └── val/\n└── labels/\n    ├── train/\n    └── val/")



























################    ---------------------   split_data: train val test -----    #####################
# import os
# import shutil
# from sklearn.model_selection import train_test_split
# from collections import Counter
# from tqdm import tqdm

# # === 1. 配置路径与参数 ===
# src_dir = '/home/wupke/traffic_det/save_pic_checked'
# save_dir = '/home/wupke/traffic_det/dataset' # 输出的根目录

# train_percent = 0.85
# val_percent = 0.10
# test_percent = 0.05 # 如果不需要 test，可将此设为 0，并将比例加给 val

# # === 2. 创建目录结构 ===
# def create_dirs(base_path):
#     sub_dirs = ['images/train', 'images/val', 'images/test', 
#                 'labels/train', 'labels/val', 'labels/test']
#     for d in sub_dirs:
#         os.makedirs(os.path.join(base_path, d), exist_ok=True)

# create_dirs(save_dir)

# # === 3. 提取主类别（用于分层抽样） ===
# def get_main_class(txt_path):
#     """读取 YOLO txt 第一行第一个数字作为主类别"""
#     try:
#         with open(txt_path, 'r') as f:
#             lines = f.readlines()
#             if not lines: return None
#             # 提取每行开头的 class_id，取出现次数最多的
#             classes = [line.split()[0] for line in lines if line.strip()]
#             return max(set(classes), key=classes.count) if classes else None
#     except Exception as e:
#         print(f"⚠️ 无法读取 {txt_path}: {e}")
#         return None

# # === 4. 扫描数据配对 ===
# print("🔍 正在扫描原始数据...")
# all_files = os.listdir(src_dir)
# image_exts = ('.jpg', '.jpeg', '.png', '.bmp')

# valid_ids = []
# valid_labels = []

# for f in all_files:
#     if f.lower().endswith(image_exts):
#         file_id = os.path.splitext(f)[0]
#         txt_path = os.path.join(src_dir, file_id + '.txt')
        
#         if os.path.exists(txt_path):
#             main_cls = get_main_class(txt_path)
#             if main_cls is not None:
#                 valid_ids.append(file_id)
#                 valid_labels.append(main_cls)

# print(f"📊 发现有效配对数据: {len(valid_ids)} 组")

# # === 5. 数据划分 ===
# # 第一次划分：Train vs Temp(Val + Test)
# temp_ratio = val_percent + test_percent
# train_ids, temp_ids, y_train, y_temp = train_test_split(
#     valid_ids, valid_labels, test_size=temp_ratio, random_state=42, stratify=valid_labels
# )

# # 第二次划分：Val vs Test
# test_size_in_temp = test_percent / temp_ratio
# val_ids, test_ids = train_test_split(
#     temp_ids, test_size=test_size_in_temp, random_state=42, stratify=y_temp
# )

# # === 6. 执行文件分发 (复制/移动) ===
# def dispatch_files(ids, split_name):
#     print(f"🚚 正在分发 {split_name} 数据...")
#     for file_id in tqdm(ids):
#         # 处理图片 (寻找可能的后缀)
#         img_name = ""
#         for ext in image_exts:
#             if os.path.exists(os.path.join(src_dir, file_id + ext)):
#                 img_name = file_id + ext
#                 break
        
#         if not img_name: continue
        
#         # 定义目标路径
#         src_img = os.path.join(src_dir, img_name)
#         src_txt = os.path.join(src_dir, file_id + '.txt')
        
#         dst_img = os.path.join(save_dir, 'images', split_name, img_name)
#         dst_txt = os.path.join(save_dir, 'labels', split_name, file_id + '.txt')
        
#         # 执行复制 (建议先用 copy，确认无误后再改 move)
#         shutil.copy2(src_img, dst_img)
#         shutil.copy2(src_txt, dst_txt)

# dispatch_files(train_ids, 'train')
# dispatch_files(val_ids, 'val')
# dispatch_files(test_ids, 'test')

# # === 7. 最终统计 ===
# def print_stats(ids, name):
#     cls_counts = Counter()
#     for fid in ids:
#         cls = get_main_class(os.path.join(src_dir, fid + '.txt'))
#         cls_counts[cls] += 1
#     print(f"  {name}: {len(ids)} samples, Class distribution: {dict(sorted(cls_counts.items()))}")

# print("\n✅ 数据划分与分发完成！")
# print_stats(train_ids, "Train")
# print_stats(val_ids, "Val")
# print_stats(test_ids, "Test")
# print(f"\n📂 结果已保存至: {save_dir}")import os
# import shutil
# from sklearn.model_selection import train_test_split
# from collections import Counter
# from tqdm import tqdm

# # === 1. 配置路径与参数 ===
# src_dir = '/home/wupke/traffic_det/save_pic_checked'
# save_dir = '/home/wupke/traffic_det/dataset' # 输出的根目录

# train_percent = 0.85
# val_percent = 0.10
# test_percent = 0.05 # 如果不需要 test，可将此设为 0，并将比例加给 val

# # === 2. 创建目录结构 ===
# def create_dirs(base_path):
#     sub_dirs = ['images/train', 'images/val', 'images/test', 
#                 'labels/train', 'labels/val', 'labels/test']
#     for d in sub_dirs:
#         os.makedirs(os.path.join(base_path, d), exist_ok=True)

# create_dirs(save_dir)

# # === 3. 提取主类别（用于分层抽样） ===
# def get_main_class(txt_path):
#     """读取 YOLO txt 第一行第一个数字作为主类别"""
#     try:
#         with open(txt_path, 'r') as f:
#             lines = f.readlines()
#             if not lines: return None
#             # 提取每行开头的 class_id，取出现次数最多的
#             classes = [line.split()[0] for line in lines if line.strip()]
#             return max(set(classes), key=classes.count) if classes else None
#     except Exception as e:
#         print(f"⚠️ 无法读取 {txt_path}: {e}")
#         return None

# # === 4. 扫描数据配对 ===
# print("🔍 正在扫描原始数据...")
# all_files = os.listdir(src_dir)
# image_exts = ('.jpg', '.jpeg', '.png', '.bmp')

# valid_ids = []
# valid_labels = []

# for f in all_files:
#     if f.lower().endswith(image_exts):
#         file_id = os.path.splitext(f)[0]
#         txt_path = os.path.join(src_dir, file_id + '.txt')
        
#         if os.path.exists(txt_path):
#             main_cls = get_main_class(txt_path)
#             if main_cls is not None:
#                 valid_ids.append(file_id)
#                 valid_labels.append(main_cls)

# print(f"📊 发现有效配对数据: {len(valid_ids)} 组")

# # === 5. 数据划分 ===
# # 第一次划分：Train vs Temp(Val + Test)
# temp_ratio = val_percent + test_percent
# train_ids, temp_ids, y_train, y_temp = train_test_split(
#     valid_ids, valid_labels, test_size=temp_ratio, random_state=42, stratify=valid_labels
# )

# # 第二次划分：Val vs Test
# test_size_in_temp = test_percent / temp_ratio
# val_ids, test_ids = train_test_split(
#     temp_ids, test_size=test_size_in_temp, random_state=42, stratify=y_temp
# )

# # === 6. 执行文件分发 (复制/移动) ===
# def dispatch_files(ids, split_name):
#     print(f"🚚 正在分发 {split_name} 数据...")
#     for file_id in tqdm(ids):
#         # 处理图片 (寻找可能的后缀)
#         img_name = ""
#         for ext in image_exts:
#             if os.path.exists(os.path.join(src_dir, file_id + ext)):
#                 img_name = file_id + ext
#                 break
        
#         if not img_name: continue
        
#         # 定义目标路径
#         src_img = os.path.join(src_dir, img_name)
#         src_txt = os.path.join(src_dir, file_id + '.txt')
        
#         dst_img = os.path.join(save_dir, 'images', split_name, img_name)
#         dst_txt = os.path.join(save_dir, 'labels', split_name, file_id + '.txt')
        
#         # 执行复制 (建议先用 copy，确认无误后再改 move)
#         shutil.copy2(src_img, dst_img)
#         shutil.copy2(src_txt, dst_txt)

# dispatch_files(train_ids, 'train')
# dispatch_files(val_ids, 'val')
# dispatch_files(test_ids, 'test')

# # === 7. 最终统计 ===
# def print_stats(ids, name):
#     cls_counts = Counter()
#     for fid in ids:
#         cls = get_main_class(os.path.join(src_dir, fid + '.txt'))
#         cls_counts[cls] += 1
#     print(f"  {name}: {len(ids)} samples, Class distribution: {dict(sorted(cls_counts.items()))}")

# print("\n✅ 数据划分与分发完成！")
# print_stats(train_ids, "Train")
# print_stats(val_ids, "Val")
# print_stats(test_ids, "Test")
# print(f"\n📂 结果已保存至: {save_dir}")