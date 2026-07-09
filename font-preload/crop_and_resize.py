"""
将文件夹中的图片按短边中心裁剪为正方形，再 resize 为 448x448。
输出到同级目录 <原文件夹名>_448/
支持中文路径，批量处理常见图片格式。
"""

from pathlib import Path
from PIL import Image

from preprocess_common import DEFAULT_NEW_DIR, image_files, iter_font_dirs, prepare_output_dir


NEW_DIR = DEFAULT_NEW_DIR
INPUT_SUBDIR = "images_white_bg"
OUTPUT_SUBDIR = "images_white_bg_448"
CLEAR_OUTPUT = True

TARGET_SIZE = 448


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def process_folder(input_dir: Path, output_path: Path):
    input_path = Path(input_dir).resolve()
    if not input_path.is_dir():
        print(f"错误：路径不存在或不是文件夹 -> {input_path}")
        return 0

    output_path = prepare_output_dir(output_path, clear=CLEAR_OUTPUT)
    files = image_files(input_path)

    if not files:
        print(f"未找到支持的图片文件: {input_path}")
        return 0

    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    print(f"共 {len(files)} 张图片，目标尺寸 {TARGET_SIZE}x{TARGET_SIZE}")

    for f in files:
        img = Image.open(f).convert("RGB")
        img = center_crop_square(img)
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        img.save(output_path / f.name)

    print("处理完成。")
    return len(files)


def process_all_fonts(new_dir=NEW_DIR):
    total = 0
    for font_dir in iter_font_dirs(new_dir):
        input_dir = font_dir / INPUT_SUBDIR
        if not input_dir.is_dir():
            print(f"[跳过] {font_dir.name}: 缺少 {INPUT_SUBDIR}")
            continue
        total += process_folder(input_dir, font_dir / OUTPUT_SUBDIR)
    print(f"全部处理完成，共处理 {total} 张。")


if __name__ == "__main__":
    process_all_fonts()
