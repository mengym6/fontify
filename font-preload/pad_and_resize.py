"""
将输入文件夹的图片补白成正方形，再缩放到64x64，输出到新文件夹。
"""

from pathlib import Path
from PIL import Image

from preprocess_common import DEFAULT_NEW_DIR, image_files, iter_font_dirs


NEW_DIR = DEFAULT_NEW_DIR
SUBFOLDER = "images_white_bg_mask_denoised"
TARGET_SIZE = 448


def pad_and_resize(img: Image.Image, target_size: int = 64) -> Image.Image:
    w, h = img.size
    max_side = max(w, h)
    # 白色背景正方形
    square = Image.new("RGB", (max_side, max_side), (255, 255, 255))
    # 居中粘贴
    offset_x = (max_side - w) // 2
    offset_y = (max_side - h) // 2
    square.paste(img, (offset_x, offset_y))
    return square.resize((target_size, target_size), Image.LANCZOS)


def main():
    total = 0
    for font_dir in iter_font_dirs(NEW_DIR):
        input_dir = font_dir / SUBFOLDER
        if not input_dir.is_dir():
            print(f"[跳过] {font_dir.name}: 缺少 {SUBFOLDER}")
            continue
        files = image_files(input_dir)
        if not files:
            print(f"[跳过] {font_dir.name}: {SUBFOLDER} 内无图片")
            continue
        print(f"[{font_dir.name}] 找到 {len(files)} 张图片，开始处理...")
        for f in files:
            img = Image.open(f).convert("RGB")
            result = pad_and_resize(img, TARGET_SIZE)
            result.save(f)
        total += len(files)
        print(f"[{font_dir.name}] 完成")

    print(f"全部处理完成，共处理 {total} 张。")


if __name__ == "__main__":
    main()
