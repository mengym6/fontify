import cv2
from pathlib import Path

from preprocess_common import DEFAULT_NEW_DIR, image_files, iter_font_dirs

# ============================================================
# 路径配置
# ============================================================
NEW_DIR = DEFAULT_NEW_DIR
INPUT_SUBDIR = "images_white_bg_v1"

# ============================================================
# 主程序
# ============================================================
def process_folder(input_dir: Path):
    files = image_files(input_dir)

    if not files:
        print(f"未找到图片: {input_dir}")
        return 0

    print(f"输入目录: {input_dir}")
    print(f"图片数量: {len(files)}")
    print("-" * 40)

    success = 0
    for i, path in enumerate(files, 1):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  [跳过] {path.name} (读取失败)")
            continue
        img_inv = 255 - img
        cv2.imwrite(str(path), img_inv)
        success += 1
        if i % 10 == 0 or i == len(files):
            print(f"  进度: {i}/{len(files)}")

    print("-" * 40)
    print(f"完成，已反转 {success}/{len(files)} 张图片（覆盖原文件）")
    return success


def main():
    total = 0
    for font_dir in iter_font_dirs(NEW_DIR):
        input_dir = font_dir / INPUT_SUBDIR
        if not input_dir.is_dir():
            print(f"[跳过] {font_dir.name}: 缺少 {INPUT_SUBDIR}")
            continue
        print(f"[{font_dir.name}]")
        total += process_folder(input_dir)
    print(f"全部完成，共反转 {total} 张图片。")


if __name__ == "__main__":
    main()
