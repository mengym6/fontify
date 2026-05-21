# -*- coding: utf-8 -*-
# 用法：python gen_jsons.py
# 在 Fontify-main 根目录下运行
import os
import json
import re

DATA_ROOT = "fontdata_example"

# (split 子目录, 输出目录)：train -> train_json，val 用 test_unknown_content -> val_json
SPLITS = [
    ("train",                 "train_json"),
    ("test_unknown_content",  "val_json"),
]


def gen_one_split(split_name, out_subdir):
    source_dir  = os.path.join(DATA_ROOT, "font", split_name, "source")
    chinese_dir = os.path.join(DATA_ROOT, "font", split_name, "chinese")
    out_dir     = os.path.join(DATA_ROOT, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(source_dir) or not os.path.isdir(chinese_dir):
        print(f"[skip] {split_name}: 缺 source 或 chinese 子目录")
        return

    source_chars = {f for f in os.listdir(source_dir) if f.endswith(".png")}
    font_dirs = sorted(
        d for d in os.listdir(chinese_dir)
        if os.path.isdir(os.path.join(chinese_dir, d))
    )

    n_font, n_total = 0, 0
    used_fonts = []  # 实际生成了 json 的字体名（按字母序）
    for font_name in font_dirs:
        target_dir = os.path.join(chinese_dir, font_name)
        target_chars = {f for f in os.listdir(target_dir) if f.endswith(".png")}
        # 取交集：source 必须有，target 也必须有
        common = sorted(source_chars & target_chars)
        if not common:
            continue
        pairs = [{
            "image_path":  f"font/{split_name}/source/{c}",
            "target_path": f"font/{split_name}/chinese/{font_name}/{c}",
            "type":        f"font_{font_name}",
        } for c in common]

        # 文件名去掉空格/特殊字符，只留字母数字
        safe = re.sub(r'[^A-Za-z0-9]', '', font_name)
        out_path = os.path.join(out_dir, f"font_{out_subdir.split('_')[0]}_{safe}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)

        n_font += 1
        n_total += len(pairs)
        used_fonts.append((font_name, len(pairs)))

    # 把实际收录的字体名写到 txt：每行 "字体名\t样本数"
    list_path = os.path.join(DATA_ROOT, f"fonts_{split_name}.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        f.write(f"# split={split_name}  total_fonts={n_font}  total_pairs={n_total}\n")
        for name, n in used_fonts:
            f.write(f"{name}\t{n}\n")

    print(f"[{split_name}] 生成 {n_font} 个字体 json，共 {n_total} 条样本 -> {out_dir}/")
    print(f"[{split_name}] 字体清单 -> {list_path}")


if __name__ == "__main__":
    for split, out in SPLITS:
        gen_one_split(split, out)
