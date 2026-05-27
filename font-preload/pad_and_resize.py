"""
将输入文件夹的图片补白成正方形，再缩放到64x64，输出到新文件夹。
"""

from pathlib import Path
from PIL import Image

# ===== 在这里设置路径 =====
NEW_DIR = Path("/Users/root1/Desktop/Fontify-main/fontdata_example/font/train/new")
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
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    font_dirs = [d for d in NEW_DIR.iterdir() if d.is_dir()]

    for font_dir in font_dirs:
        input_dir = font_dir / SUBFOLDER
        if not input_dir.is_dir():
            continue
        files = [f for f in input_dir.iterdir() if f.suffix.lower() in exts]
        if not files:
            continue
        print(f"[{font_dir.name}] 找到 {len(files)} 张图片，开始处理...")
        for f in files:
            img = Image.open(f).convert("RGB")
            result = pad_and_resize(img, TARGET_SIZE)
            result.save(f)
        print(f"[{font_dir.name}] 完成")

    print("全部处理完成。")


if __name__ == "__main__":
    main()
