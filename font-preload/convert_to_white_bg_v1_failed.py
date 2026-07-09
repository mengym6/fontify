import cv2
import numpy as np
from pathlib import Path

from preprocess_common import DEFAULT_NEW_DIR, image_files, iter_font_dirs, prepare_output_dir


NEW_DIR = DEFAULT_NEW_DIR
INPUT_SUBDIR = "images"
OUTPUT_SUBDIR = "images_white_bg_v1"
CLEAR_OUTPUT = True


def process_single(img_path: Path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape
    blurred = cv2.medianBlur(img, 19)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    min_area = h * w * 0.002
    filtered = np.zeros_like(closed)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255

    circ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, circ)
    smooth = cv2.GaussianBlur(filtered, (7, 7), 0)
    return 255 - smooth


def process_font_dir(font_dir: Path):
    input_dir = font_dir / INPUT_SUBDIR
    files = image_files(input_dir)
    if not files:
        print(f"[跳过] {font_dir.name}: 未找到图片文件 {input_dir}")
        return 0, 0

    output_dir = prepare_output_dir(font_dir / OUTPUT_SUBDIR, clear=CLEAR_OUTPUT)
    success = 0
    print(f"[{font_dir.name}] 输入 {len(files)} 张 -> {output_dir}")
    for idx, path in enumerate(files, 1):
        result = process_single(path)
        if result is None:
            print(f"  [跳过] {path.name} (读取失败)")
            continue
        cv2.imwrite(str(output_dir / path.name), result)
        success += 1
        if idx % 50 == 0 or idx == len(files):
            print(f"  进度: {idx}/{len(files)}")
    return success, len(files)


def process_all_fonts(new_dir=NEW_DIR):
    total_success = 0
    total_files = 0
    font_dirs = iter_font_dirs(new_dir)
    print(f"根目录: {Path(new_dir)}")
    print(f"发现 {len(font_dirs)} 个字体目录")
    print("=" * 50)
    for font_dir in font_dirs:
        success, count = process_font_dir(font_dir)
        total_success += success
        total_files += count
    print("=" * 50)
    print(f"全部完成，共处理 {total_success}/{total_files} 张")


def main():
    process_all_fonts()


if __name__ == "__main__":
    main()
