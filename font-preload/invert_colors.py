from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from preprocess_common import DEFAULT_NEW_DIR, IMAGE_EXTS, REPO_ROOT, image_files, iter_font_dirs


NEW_DIR = DEFAULT_NEW_DIR
V1_SUBDIR = "images_white_bg_v1"
NOISY_SUBDIR = "images_white_bg"
DENOISED_SUBDIR = "images_white_bg_mask_denoised"
CSV_PATH_COLUMN = "relative_path"
TARGET_SIZE = 448

DENOISE_PARAMS = {
    "bin_threshold": 0,
    "dilate_kernel_size": 7,
    "dilate_iterations": 4,
    "feather_sigma": 10.0,
    "hole_max_area": 1500,
    "hole_fill_kernel": 5,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Invert font images with the original 255-pixel rule. "
            "With --csv, invert the matching v1 images first, then regenerate "
            "the matching denoised images from v1 + images_white_bg."
        )
    )
    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "CSV report containing denoised image paths to repair. "
            "The matching image name under images_white_bg_v1 is inverted in place."
        ),
    )
    parser.add_argument(
        "--path-column",
        default=CSV_PATH_COLUMN,
        help=f"Column name in --csv that stores denoised image paths. Default: {CSV_PATH_COLUMN}",
    )
    parser.add_argument(
        "--new-dir",
        default=str(NEW_DIR),
        help="Root font/train/new directory used by the legacy folder mode.",
    )
    parser.add_argument(
        "--v1-subdir",
        default=V1_SUBDIR,
        help="Subdirectory under each font folder that contains v1 images.",
    )
    parser.add_argument(
        "--noisy-subdir",
        default=NOISY_SUBDIR,
        help="Subdirectory under each font folder that contains images_white_bg images.",
    )
    parser.add_argument(
        "--denoised-subdir",
        default=DENOISED_SUBDIR,
        help="Subdirectory under each font folder to overwrite with regenerated denoised images.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=TARGET_SIZE,
        help=f"Final square image size after pad-and-resize. Default: {TARGET_SIZE}",
    )
    parser.add_argument(
        "--force-invert-v1",
        action="store_true",
        help="Invert v1 images even when their border already looks white.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be overwritten without writing files.",
    )
    return parser.parse_args()


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def invert_image(img: Image.Image) -> Image.Image:
    """Original invert_colors behavior: new_pixel = 255 - old_pixel."""
    arr = np.asarray(img)
    inv = 255 - arr
    return Image.fromarray(inv.astype(np.uint8), mode=img.mode)


def invert_file(src_path: Path, dst_path: Path, *, dry_run: bool = False) -> str:
    if not src_path.is_file():
        return "missing_source"
    if src_path.suffix.lower() not in IMAGE_EXTS:
        return "not_image"

    try:
        img = Image.open(src_path)
        img.load()
    except OSError:
        return "read_failed"

    if dry_run:
        return "would_overwrite"

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        invert_image(img).save(dst_path)
    except OSError:
        return "write_failed"
    return "overwritten"


def process_folder(input_dir: Path, *, dry_run: bool = False):
    files = image_files(input_dir)

    if not files:
        print(f"未找到图片: {input_dir}")
        return 0

    print(f"输入目录: {input_dir}")
    print(f"图片数量: {len(files)}")
    print("-" * 40)

    success = 0
    for i, path in enumerate(files, 1):
        status = invert_file(path, path, dry_run=dry_run)
        if status not in {"overwritten", "would_overwrite"}:
            print(f"  [跳过] {path.name} ({status})")
            continue
        success += 1
        if i % 10 == 0 or i == len(files):
            print(f"  进度: {i}/{len(files)}")

    print("-" * 40)
    action = "将反转" if dry_run else "已反转"
    print(f"完成，{action} {success}/{len(files)} 张图片（覆盖原文件）")
    return success


def csv_repair_records(
    csv_path: Path,
    path_column: str,
    v1_subdir: str,
    noisy_subdir: str,
    denoised_subdir: str,
):
    seen = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames or path_column not in reader.fieldnames:
            available = ", ".join(reader.fieldnames or [])
            raise ValueError(
                f"CSV missing column {path_column!r}. Available columns: {available}"
            )

        for row in reader:
            raw = (row.get(path_column) or "").strip()
            if not raw:
                continue
            csv_target = resolve_repo_path(raw)
            font_dir = csv_target.parent.parent
            denoised_path = font_dir / denoised_subdir / csv_target.name
            key = denoised_path.resolve()
            if key in seen:
                continue
            seen.add(key)
            yield {
                "v1": font_dir / v1_subdir / csv_target.name,
                "noisy": font_dir / noisy_subdir / csv_target.name,
                "denoised": denoised_path,
            }


def border_mean(gray: np.ndarray) -> float:
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    return float(border.mean())


def should_invert_v1(v1_path: Path, *, force: bool):
    if not v1_path.is_file():
        return False, "missing_v1"
    try:
        gray = np.asarray(Image.open(v1_path).convert("L"))
    except OSError:
        return False, "read_v1_failed"
    if force:
        return True, "force_invert_v1"
    if border_mean(gray) < 128:
        return True, "dark_v1"
    return False, "already_white_v1"


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)

    sum_bg = 0.0
    weight_bg = 0.0
    max_between = -1.0
    threshold = 0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > max_between:
            max_between = between
            threshold = t

    return threshold


def binarize_clean(gray: np.ndarray, threshold: int) -> np.ndarray:
    if threshold == 0:
        threshold = otsu_threshold(gray)
    return (gray <= threshold).astype(np.uint8) * 255


def ellipse_kernel(size: int) -> np.ndarray:
    size = size if size % 2 == 1 else size + 1
    if size <= 1:
        return np.ones((1, 1), dtype=bool)
    radius = (size - 1) / 2.0
    yy, xx = np.ogrid[:size, :size]
    return ((yy - radius) ** 2 + (xx - radius) ** 2) <= radius**2


def dilate_mask(binary: np.ndarray, params: dict) -> np.ndarray:
    kernel = ellipse_kernel(params["dilate_kernel_size"])
    mask = binary > 0
    for _ in range(params["dilate_iterations"]):
        mask = ndimage.binary_dilation(mask, structure=kernel)
    return mask.astype(np.uint8) * 255


def feather_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    mask_f = mask.astype(np.float32) / 255.0
    if sigma > 0:
        mask_f = ndimage.gaussian_filter(mask_f, sigma=sigma, mode="nearest")
    return np.clip(mask_f, 0.0, 1.0)


def apply_mask(noisy_gray: np.ndarray, mask_soft: np.ndarray) -> np.ndarray:
    noisy_f = noisy_gray.astype(np.float32)
    result = noisy_f * mask_soft + 255.0 * (1.0 - mask_soft)
    return np.clip(result, 0, 255).astype(np.uint8)


def fill_small_holes(result: np.ndarray, stroke_mask: np.ndarray, params: dict) -> np.ndarray:
    if params["hole_max_area"] <= 0:
        return result

    kernel = ellipse_kernel(params["hole_fill_kernel"])
    inner_mask = ndimage.binary_erosion(stroke_mask > 0, structure=kernel, iterations=1)
    holes = inner_mask & (result > 50)
    result[holes] = 0
    return result


def pad_and_resize_array(gray: np.ndarray, target_size: int) -> Image.Image:
    img = Image.fromarray(gray.astype(np.uint8), mode="L").convert("RGB")
    w, h = img.size
    max_side = max(w, h)
    square = Image.new("RGB", (max_side, max_side), (255, 255, 255))
    offset_x = (max_side - w) // 2
    offset_y = (max_side - h) // 2
    square.paste(img, (offset_x, offset_y))
    return square.resize((target_size, target_size), Image.Resampling.LANCZOS)


def regenerate_denoised(v1_path: Path, noisy_path: Path, output_path: Path, *, target_size: int):
    if not v1_path.is_file():
        return "missing_v1"
    if not noisy_path.is_file():
        return "missing_noisy"

    try:
        clean_img = Image.open(v1_path).convert("L")
        noisy_img = Image.open(noisy_path).convert("L")
        clean_img.load()
        noisy_img.load()
    except OSError:
        return "read_failed"

    if clean_img.size != noisy_img.size:
        clean_img = clean_img.resize(noisy_img.size, Image.Resampling.LANCZOS)

    clean = np.asarray(clean_img)
    noisy = np.asarray(noisy_img)

    binary = binarize_clean(clean, DENOISE_PARAMS["bin_threshold"])
    dilated = dilate_mask(binary, DENOISE_PARAMS)
    mask_soft = feather_mask(dilated, DENOISE_PARAMS["feather_sigma"])
    result = apply_mask(noisy, mask_soft)
    result = fill_small_holes(result, binary, DENOISE_PARAMS)
    final_img = pad_and_resize_array(result, target_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        final_img.save(output_path)
    except OSError:
        return "write_failed"
    return "denoised_written"


def process_csv(
    csv_path: Path,
    *,
    path_column: str,
    v1_subdir: str,
    noisy_subdir: str,
    denoised_subdir: str,
    target_size: int,
    force_invert_v1: bool,
    dry_run: bool,
):
    records = list(
        csv_repair_records(csv_path, path_column, v1_subdir, noisy_subdir, denoised_subdir)
    )
    print(f"CSV: {csv_path}")
    print(f"路径列: {path_column}")
    print(f"v1 子目录: {v1_subdir}")
    print(f"noisy 子目录: {noisy_subdir}")
    print(f"denoised 子目录: {denoised_subdir}")
    print(f"最终尺寸: {target_size}x{target_size}")
    print(f"图片数量: {len(records)}")
    print("-" * 40)

    counts = {}
    for i, record in enumerate(records, 1):
        v1_path = record["v1"]
        noisy_path = record["noisy"]
        denoised_path = record["denoised"]

        should_invert, reason = should_invert_v1(v1_path, force=force_invert_v1)
        counts[reason] = counts.get(reason, 0) + 1

        if should_invert:
            if dry_run:
                counts["would_invert_v1"] = counts.get("would_invert_v1", 0) + 1
            else:
                status = invert_file(v1_path, v1_path, dry_run=False)
                counts[status] = counts.get(status, 0) + 1
                if status != "overwritten":
                    print(f"  [v1跳过] {v1_path} ({status})")

        if dry_run:
            if v1_path.is_file() and noisy_path.is_file():
                counts["would_write_denoised"] = counts.get("would_write_denoised", 0) + 1
            else:
                if not v1_path.is_file():
                    counts["missing_v1_for_denoise"] = counts.get("missing_v1_for_denoise", 0) + 1
                if not noisy_path.is_file():
                    counts["missing_noisy"] = counts.get("missing_noisy", 0) + 1
        else:
            status = regenerate_denoised(
                v1_path,
                noisy_path,
                denoised_path,
                target_size=target_size,
            )
            counts[status] = counts.get(status, 0) + 1
            if status != "denoised_written":
                print(f"  [denoise跳过] v1={v1_path} noisy={noisy_path} ({status})")

        if i % 25 == 0 or i == len(records):
            print(f"  进度: {i}/{len(records)}")

    print("-" * 40)
    for status in sorted(counts):
        print(f"{status}: {counts[status]}")
    return counts


def main():
    args = parse_args()

    if args.csv:
        csv_path = resolve_repo_path(args.csv)
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        counts = process_csv(
            csv_path,
            path_column=args.path_column,
            v1_subdir=args.v1_subdir,
            noisy_subdir=args.noisy_subdir,
            denoised_subdir=args.denoised_subdir,
            target_size=args.target_size,
            force_invert_v1=args.force_invert_v1,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print(
                "全部完成，dry-run 未写入；"
                f"将反转 v1 {counts.get('would_invert_v1', 0)} 张，"
                f"将重写 denoised {counts.get('would_write_denoised', 0)} 张。"
            )
        else:
            print(
                "全部完成，"
                f"已反转 v1 {counts.get('overwritten', 0)} 张，"
                f"已重写 denoised {counts.get('denoised_written', 0)} 张。"
            )
        return

    total = 0
    new_dir = Path(args.new_dir)
    for font_dir in iter_font_dirs(new_dir):
        input_dir = font_dir / args.v1_subdir
        if not input_dir.is_dir():
            print(f"[跳过] {font_dir.name}: 缺少 {args.v1_subdir}")
            continue
        print(f"[{font_dir.name}]")
        total += process_folder(input_dir, dry_run=args.dry_run)
    action = "将反转" if args.dry_run else "已反转"
    print(f"全部完成，{action} {total} 张图片。")


if __name__ == "__main__":
    main()
