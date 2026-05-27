import os
import json
import random
import shutil

random.seed(42)

DATA_ROOT = "fontdata_example"
new_base = "font/train/new"
source_dir = "ttf/SourceHanSansSC-VF"
train_output_dir = "train_json_new"
val_output_dir = "val_json_new"
val_ratio = 0.15

abs_root = os.path.dirname(os.path.abspath(__file__))
source_files = set(os.listdir(os.path.join(abs_root, DATA_ROOT, source_dir)))

train_abs = os.path.join(abs_root, DATA_ROOT, train_output_dir)
val_abs = os.path.join(abs_root, DATA_ROOT, val_output_dir)
if os.path.exists(train_abs):
    shutil.rmtree(train_abs)
if os.path.exists(val_abs):
    shutil.rmtree(val_abs)
os.makedirs(train_abs)
os.makedirs(val_abs)

new_abs = os.path.join(abs_root, DATA_ROOT, new_base)
for folder in sorted(os.listdir(new_abs)):
    wb_dir = os.path.join(new_abs, folder, "images_white_bg_mask_denoised")
    if not os.path.isdir(wb_dir):
        continue

    pairs = []
    missing = []
    for f in sorted(os.listdir(wb_dir)):
        name, ext = os.path.splitext(f)
        source_name = name[0] + ext
        if source_name not in source_files:
            missing.append(f)
            continue
        pairs.append({
            "image_path": f"{source_dir}/{source_name}",
            "target_path": f"{new_base}/{folder}/images_white_bg_mask_denoised/{f}",
            "type": f"font_{folder}"
        })

    random.shuffle(pairs)
    val_count = max(1, int(len(pairs) * val_ratio))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    train_path = os.path.join(abs_root, DATA_ROOT, train_output_dir, f"font_train_{folder}.json")
    val_path = os.path.join(abs_root, DATA_ROOT, val_output_dir, f"font_val_{folder}.json")
    with open(train_path, "w", encoding="utf-8") as fp:
        json.dump(train_pairs, fp, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as fp:
        json.dump(val_pairs, fp, ensure_ascii=False, indent=2)
    print(f"{folder}: train {len(train_pairs)}, val {len(val_pairs)}, 未配对 {len(missing)}")
    if missing:
        print(f"  未配对文件: {missing}")
