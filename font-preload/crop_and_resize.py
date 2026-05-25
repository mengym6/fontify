"""
将文件夹中的图片按短边中心裁剪为正方形，再 resize 为 448x448。
输出到同级目录 <原文件夹名>_448/
支持中文路径，批量处理常见图片格式。
"""

from pathlib import Path
from PIL import Image

# ===== 在这里修改输入路径 =====
INPUT_DIR = "/Users/root1/Desktop/Fontify-main/fontdata_example/font/train/new/颜真卿结体/images_white_bg"
# ==============================

TARGET_SIZE = 448
SUPPORTED_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def process_folder(input_dir: str):
    input_path = Path(input_dir).resolve()
    if not input_path.is_dir():
        print(f"错误：路径不存在或不是文件夹 -> {input_path}")
        return

    output_path = input_path.parent / f"{input_path.name}_{TARGET_SIZE}"
    output_path.mkdir(exist_ok=True)

    files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
    ]

    if not files:
        print(f"未找到支持的图片文件: {input_path}")
        return

    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    print(f"共 {len(files)} 张图片，目标尺寸 {TARGET_SIZE}x{TARGET_SIZE}")

    for f in files:
        img = Image.open(f).convert("RGB")
        img = center_crop_square(img)
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        img.save(output_path / f.name)

    print("处理完成。")


if __name__ == "__main__":
    process_folder(INPUT_DIR)
