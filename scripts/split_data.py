import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from collections import Counter
from tqdm import tqdm

# === 可调参数 ===
train_percent = 0.85
val_percent = 0.10
test_percent = 0.05

assert abs(train_percent + val_percent + test_percent - 1.0) < 1e-6, "比例之和必须为 1.0"

# === 配置路径 ===
xml_dir = '/media/backup/wupke/apollo-model-yolox/datasets/VOC4/annotations'
save_dir = '/media/backup/wupke/apollo-model-yolox/datasets/VOC4/ImageSets/Main'
os.makedirs(save_dir, exist_ok=True)

# === 提取每张图像的主类别 ===
def extract_main_class(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        classes = [obj.find('name').text for obj in root.findall('object') if obj.find('name') is not None]
        return max(set(classes), key=classes.count) if classes else None
    except Exception as e:
        print(f"⚠️ 无法解析 {xml_path}: {e}")
        return None

# === 安全分层划分函数 ===
def safe_stratified_split(X, y, test_size=0.5, random_state=42):
    label_counts = Counter(y)
    too_few = [cls for cls, count in label_counts.items() if count < 2]
    if too_few:
        print(f"⚠️ 类别样本不足（<2），无法进行 StratifiedSplit，使用 train_test_split：{too_few}")
        X_train, X_test, _, _ = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)
        return X_train, X_test
    else:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y))
        return [X[i] for i in train_idx], [X[i] for i in test_idx]

# === 类别统计函数（主类别）===
def count_main_classes(split_ids):
    cls_counter = Counter()
    for name in split_ids:
        xml_path = os.path.join(xml_dir, name + '.xml')
        label = extract_main_class(xml_path)
        if label:
            cls_counter[label] += 1
    return cls_counter

# === 所有样本及其主类别提取 ===
xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
valid_ids, valid_labels = [], []
for f in xml_files:
    xml_path = os.path.join(xml_dir, f)
    label = extract_main_class(xml_path)
    if label:
        valid_ids.append(os.path.splitext(f)[0])
        valid_labels.append(label)

assert len(valid_ids) == len(valid_labels), "图像与标签数量不一致"

# === 第一次划分：train vs temp(val+test) ===
temp_ratio = val_percent + test_percent
train_ids, temp_ids = safe_stratified_split(valid_ids, valid_labels, test_size=temp_ratio, random_state=42)
temp_labels = [extract_main_class(os.path.join(xml_dir, name + '.xml')) for name in temp_ids]

# === 第二次划分：val vs test ===
val_ratio_in_temp = val_percent / temp_ratio
val_ids, test_ids = safe_stratified_split(temp_ids, temp_labels, test_size=(1 - val_ratio_in_temp), random_state=42)

# === 保存划分结果 ===
def save_split(file_list, out_path):
    with open(out_path, 'w') as f:
        for name in file_list:
            f.write(name + '\n')

save_split(train_ids, os.path.join(save_dir, 'train.txt'))
save_split(val_ids, os.path.join(save_dir, 'val.txt'))
save_split(test_ids, os.path.join(save_dir, 'test.txt'))

# === 输出划分统计信息 ===
print(f"\n✅ 数据划分完成：")
print(f"  Train: {len(train_ids)} samples")
print(f"    Class dist: {dict(sorted(count_main_classes(train_ids).items()))}")
print(f"  Val:   {len(val_ids)} samples")
print(f"    Class dist: {dict(sorted(count_main_classes(val_ids).items()))}")
print(f"  Test:  {len(test_ids)} samples")
print(f"    Class dist: {dict(sorted(count_main_classes(test_ids).items()))}")

# === 输出所有类别总计 ===
def count_all_classes(xml_dir):
    cls_counter = Counter()
    for f in tqdm(os.listdir(xml_dir), desc="📊 统计所有类别"):
        if f.endswith('.xml'):
            xml_path = os.path.join(xml_dir, f)
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name = obj.find('name')
                    if name is not None:
                        cls_counter[name.text] += 1
            except Exception as e:
                print(f"⚠️ 无法解析 {f}: {e}")
    return cls_counter

all_class_counts = count_all_classes(xml_dir)
print(f"\n📊 所有 XML 中类别总计:")
for cls, count in sorted(all_class_counts.items()):
    print(f"  {cls}: {count}")


