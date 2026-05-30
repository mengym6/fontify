import os
import json
import random

random.seed(42)

# ============ 手动修改区 ============
# new 下每个字体文件夹 -> ttf 下对应的字体文件夹（image_path 从这里取）
# 按你的实际对应关系修改下面的右值即可
folder_to_ttf = {
    "DongqcBF":  "MaShanZheng-Regular",
    "DongqcJT":  "MaShanZheng-Regular",
    "LiugqBF":   "MaShanZheng-Regular",
    "LiugqJT":   "MaShanZheng-Regular",
    "OuyxBF":    "ZhiMangXing-Regular",
    "OuyxJT":    "ZhiMangXing-Regular",
    "SushBF":    "ZhiMangXing-Regular",
    "SushJT":    "ZhiMangXing-Regular",
    "YanzqBF":   "MaShanZheng-Regular",
    "YanzqJT":   "MaShanZheng-Regular",
    "ZhaomfBF":  "ZhiMangXing-Regular",
    "ZhaomfJT":  "ZhiMangXing-Regular",
}
ttf_base = "ttf"
# ===================================

new_base = "font/train/new"
train_output_dir = "train_json_new"
val_output_dir = "val_json_new"
val_ratio = 0.15

abs_root = os.path.dirname(os.path.abspath(__file__))

os.makedirs(os.path.join(abs_root, train_output_dir), exist_ok=True)
os.makedirs(os.path.join(abs_root, val_output_dir), exist_ok=True)

new_abs = os.path.join(abs_root, new_base)
for folder in sorted(os.listdir(new_abs)):
    wb_dir = os.path.join(new_abs, folder, "images_white_bg")
    if not os.path.isdir(wb_dir):
        continue

    ttf_folder = folder_to_ttf.get(folder)
    if ttf_folder is None:
        print(f"{folder}: 未在 folder_to_ttf 中配置，跳过")
        continue
    source_dir = f"{ttf_base}/{ttf_folder}"
    source_files = set(os.listdir(os.path.join(abs_root, source_dir)))

    pairs = []
    for f in sorted(os.listdir(wb_dir)):
        name, ext = os.path.splitext(f)
        source_name = name[0] + ext
        if source_name not in source_files:
            continue
        pairs.append({
            "image_path": f"{source_dir}/{source_name}",
            "target_path": f"{new_base}/{folder}/images_white_bg/{f}",
            "type": f"font_{folder}"
        })

    random.shuffle(pairs)
    val_count = max(1, int(len(pairs) * val_ratio))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    train_path = os.path.join(abs_root, train_output_dir, f"font_train_{folder}.json")
    val_path = os.path.join(abs_root, val_output_dir, f"font_val_{folder}.json")
    with open(train_path, "w", encoding="utf-8") as fp:
        json.dump(train_pairs, fp, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as fp:
        json.dump(val_pairs, fp, ensure_ascii=False, indent=2)
    print(f"{folder} <- {ttf_folder}: train {len(train_pairs)}, val {len(val_pairs)}")
