import os
import json
import random

random.seed(42)

new_base = "font/train/new"
source_dir = "font/train/source"
train_output_dir = "train_json_new"
val_output_dir = "val_json_new"
val_ratio = 0.15

abs_root = os.path.dirname(os.path.abspath(__file__))
source_files = set(os.listdir(os.path.join(abs_root, source_dir)))

os.makedirs(os.path.join(abs_root, train_output_dir), exist_ok=True)
os.makedirs(os.path.join(abs_root, val_output_dir), exist_ok=True)

new_abs = os.path.join(abs_root, new_base)
for folder in sorted(os.listdir(new_abs)):
    wb_dir = os.path.join(new_abs, folder, "images_white_bg")
    if not os.path.isdir(wb_dir):
        continue

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
    print(f"{folder}: train {len(train_pairs)}, val {len(val_pairs)}")
