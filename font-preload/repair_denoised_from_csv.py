#!/usr/bin/env python3
"""Repair wrongly inverted font images from a CSV report.

Workflow for each CSV target:
1. Locate the same file under images_white_bg_v1.
2. If the v1 image still has a dark background, invert it in place with 255 - pixel.
3. Regenerate images_white_bg_mask_denoised from the repaired v1 image and images_white_bg.
4. Pad-and-resize the final denoised image to 448x448 and overwrite the old file.

The script is intentionally standalone so it can be copied to a server together
with the CSV file. It does not import project-local helper modules.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from scipy import ndimage  # type: ignore
except Exception:
    ndimage = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
DEFAULT_CSV = "reports/fontdata_non_white_black_report.csv"
DEFAULT_PATH_COLUMN = "relative_path"
DEFAULT_V1_SUBDIR = "images_white_bg_v1"
DEFAULT_NOISY_SUBDIR = "images_white_bg"
DEFAULT_DENOISED_SUBDIR = "images_white_bg_mask_denoised"
DEFAULT_TARGET_SIZE = 448

DENOISE_PARAMS = {
    "bin_threshold": 0,
    "dilate_kernel_size": 7,
    "dilate_iterations": 4,
    "feather_sigma": 10.0,
    "hole_max_area": 1500,
    "hole_fill_kernel": 5,
}


@dataclass(frozen=True)
class RepairRecord:
    font_name: str
    filename: str
    v1_path: Path
    noisy_path: Path
    denoised_path: Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Invert CSV-selected v1 images and regenerate their denoised outputs."
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help=f"CSV file path. Default: {DEFAULT_CSV}",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Project root on the server. Run from repo root and keep this as '.'.",
    )
    parser.add_argument(
        "--path-column",
        default=DEFAULT_PATH_COLUMN,
        help=f"CSV column containing denoised image paths. Default: {DEFAULT_PATH_COLUMN}",
    )
    parser.add_argument(
        "--v1-subdir",
        default=DEFAULT_V1_SUBDIR,
        help=f"v1 folder name. Default: {DEFAULT_V1_SUBDIR}",
    )
    parser.add_argument(
        "--noisy-subdir",
        default=DEFAULT_NOISY_SUBDIR,
        help=f"images_white_bg folder name. Default: {DEFAULT_NOISY_SUBDIR}",
    )
    parser.add_argument(
        "--denoised-subdir",
        default=DEFAULT_DENOISED_SUBDIR,
        help=f"denoised output folder name. Default: {DEFAULT_DENOISED_SUBDIR}",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help=f"Final square image size. Default: {DEFAULT_TARGET_SIZE}",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cv2", "scipy"),
        default="auto",
        help="Denoise backend. auto prefers cv2 and falls back to scipy.",
    )
    parser.add_argument(
        "--force-invert-v1",
        action="store_true",
        help="Invert v1 even when its border already looks white. Use carefully.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be changed; do not write files.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the final verification scan.",
    )
    parser.add_argument(
        "--sample-sheet",
        default="",
        help="Optional output path for a visual sample sheet, e.g. reports/repair_samples.jpg.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=16,
        help="Number of samples to place in --sample-sheet. Default: 16.",
    )
    return parser.parse_args()


def resolve_path(repo_root: Path, path_value: str) -> Path:
    """Resolve relative CSV paths and rescue absolute paths copied from another machine."""
    raw = Path(path_value)
    if not raw.is_absolute():
        return repo_root / raw

    raw_parts = raw.parts
    anchors = [
        ("fontdata_example", "font", "train", "new"),
        ("fontdata_example",),
    ]
    for anchor in anchors:
        n = len(anchor)
        for i in range(0, len(raw_parts) - n + 1):
            if raw_parts[i : i + n] == anchor:
                return repo_root.joinpath(*raw_parts[i:])
    return raw


def load_records(
    csv_path: Path,
    repo_root: Path,
    path_column: str,
    v1_subdir: str,
    noisy_subdir: str,
    denoised_subdir: str,
) -> list[RepairRecord]:
    records: list[RepairRecord] = []
    seen: set[Path] = set()

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
            csv_target = resolve_path(repo_root, raw)
            font_dir = csv_target.parent.parent
            denoised_path = font_dir / denoised_subdir / csv_target.name
            key = denoised_path.resolve()
            if key in seen:
                continue
            seen.add(key)
            records.append(
                RepairRecord(
                    font_name=font_dir.name,
                    filename=csv_target.name,
                    v1_path=font_dir / v1_subdir / csv_target.name,
                    noisy_path=font_dir / noisy_subdir / csv_target.name,
                    denoised_path=denoised_path,
                )
            )

    return records


def invert_image(img: Image.Image) -> Image.Image:
    arr = np.asarray(img)
    inv = 255 - arr
    return Image.fromarray(inv.astype(np.uint8), mode=img.mode)


def invert_file_in_place(path: Path, *, dry_run: bool) -> str:
    if not path.is_file():
        return "missing_v1"
    if path.suffix.lower() not in IMAGE_EXTS:
        return "not_image"
    try:
        img = Image.open(path)
        img.load()
    except OSError:
        return "read_v1_failed"

    if dry_run:
        return "would_invert_v1"

    try:
        invert_image(img).save(path)
    except OSError:
        return "write_v1_failed"
    return "inverted_v1"


def gray_array(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    img.load()
    return np.asarray(img)


def border_mean(gray: np.ndarray) -> float:
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    return float(border.mean())


def v1_needs_invert(path: Path, *, force: bool) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing_v1"
    try:
        gray = gray_array(path)
    except OSError:
        return False, "read_v1_failed"
    if force:
        return True, "force_invert_v1"
    if border_mean(gray) < 128:
        return True, "dark_v1"
    return False, "already_white_v1"


def choose_backend(name: str) -> str:
    if name == "cv2":
        if cv2 is None:
            raise RuntimeError("backend=cv2 was requested, but opencv-python is not installed.")
        return "cv2"
    if name == "scipy":
        if ndimage is None:
            raise RuntimeError("backend=scipy was requested, but scipy is not installed.")
        return "scipy"
    if cv2 is not None:
        return "cv2"
    if ndimage is not None:
        return "scipy"
    raise RuntimeError("Neither cv2 nor scipy is available. Install opencv-python or scipy.")


def read_gray_cv2(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)  # type: ignore[union-attr]


def denoise_cv2(v1_path: Path, noisy_path: Path) -> np.ndarray | None:
    clean = read_gray_cv2(v1_path)
    noisy = read_gray_cv2(noisy_path)
    if clean is None or noisy is None:
        return None

    h, w = noisy.shape
    if clean.shape != noisy.shape:
        clean = cv2.resize(clean, (w, h), interpolation=cv2.INTER_AREA)  # type: ignore[union-attr]

    threshold = DENOISE_PARAMS["bin_threshold"]
    if threshold == 0:
        _, binary = cv2.threshold(  # type: ignore[union-attr]
            clean, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU  # type: ignore[union-attr]
        )
    else:
        _, binary = cv2.threshold(  # type: ignore[union-attr]
            clean, threshold, 255, cv2.THRESH_BINARY_INV  # type: ignore[union-attr]
        )

    ksize = DENOISE_PARAMS["dilate_kernel_size"]
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))  # type: ignore[union-attr]
    dilated = cv2.dilate(  # type: ignore[union-attr]
        binary, kernel, iterations=DENOISE_PARAMS["dilate_iterations"]
    )

    mask_soft = dilated.astype(np.float32) / 255.0
    sigma = DENOISE_PARAMS["feather_sigma"]
    if sigma > 0:
        blur_ksize = int(np.ceil(sigma * 3)) * 2 + 1
        mask_soft = cv2.GaussianBlur(mask_soft, (blur_ksize, blur_ksize), sigma)  # type: ignore[union-attr]

    result = noisy.astype(np.float32) * mask_soft + 255.0 * (1.0 - mask_soft)
    result = np.clip(result, 0, 255).astype(np.uint8)

    if DENOISE_PARAMS["hole_max_area"] > 0:
        erode_k = DENOISE_PARAMS["hole_fill_kernel"]
        erode_kernel = cv2.getStructuringElement(  # type: ignore[union-attr]
            cv2.MORPH_ELLIPSE, (erode_k, erode_k)  # type: ignore[union-attr]
        )
        inner_mask = cv2.erode(binary, erode_kernel, iterations=1)  # type: ignore[union-attr]
        holes = (inner_mask == 255) & (result > 50)
        result[holes] = 0

    return result


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


def ellipse_kernel(size: int) -> np.ndarray:
    size = size if size % 2 == 1 else size + 1
    if size <= 1:
        return np.ones((1, 1), dtype=bool)
    radius = (size - 1) / 2.0
    yy, xx = np.ogrid[:size, :size]
    return ((yy - radius) ** 2 + (xx - radius) ** 2) <= radius**2


def denoise_scipy(v1_path: Path, noisy_path: Path) -> np.ndarray | None:
    clean_img = Image.open(v1_path).convert("L")
    noisy_img = Image.open(noisy_path).convert("L")
    clean_img.load()
    noisy_img.load()

    if clean_img.size != noisy_img.size:
        clean_img = clean_img.resize(noisy_img.size, Image.Resampling.LANCZOS)

    clean = np.asarray(clean_img)
    noisy = np.asarray(noisy_img)

    threshold = DENOISE_PARAMS["bin_threshold"] or otsu_threshold(clean)
    binary = (clean <= threshold)

    dilate_k = ellipse_kernel(DENOISE_PARAMS["dilate_kernel_size"])
    dilated = binary
    for _ in range(DENOISE_PARAMS["dilate_iterations"]):
        dilated = ndimage.binary_dilation(dilated, structure=dilate_k)  # type: ignore[union-attr]

    mask_soft = dilated.astype(np.float32)
    sigma = DENOISE_PARAMS["feather_sigma"]
    if sigma > 0:
        mask_soft = ndimage.gaussian_filter(mask_soft, sigma=sigma, mode="nearest")  # type: ignore[union-attr]
    mask_soft = np.clip(mask_soft, 0.0, 1.0)

    result = noisy.astype(np.float32) * mask_soft + 255.0 * (1.0 - mask_soft)
    result = np.clip(result, 0, 255).astype(np.uint8)

    if DENOISE_PARAMS["hole_max_area"] > 0:
        erode_k = ellipse_kernel(DENOISE_PARAMS["hole_fill_kernel"])
        inner_mask = ndimage.binary_erosion(binary, structure=erode_k, iterations=1)  # type: ignore[union-attr]
        holes = inner_mask & (result > 50)
        result[holes] = 0

    return result


def pad_and_resize(gray: np.ndarray, target_size: int) -> Image.Image:
    img = Image.fromarray(gray.astype(np.uint8), mode="L").convert("RGB")
    w, h = img.size
    max_side = max(w, h)
    square = Image.new("RGB", (max_side, max_side), (255, 255, 255))
    offset_x = (max_side - w) // 2
    offset_y = (max_side - h) // 2
    square.paste(img, (offset_x, offset_y))
    return square.resize((target_size, target_size), Image.Resampling.LANCZOS)


def regenerate_denoised(
    record: RepairRecord,
    *,
    backend: str,
    target_size: int,
    dry_run: bool,
) -> str:
    if not record.v1_path.is_file():
        return "missing_v1_for_denoise"
    if not record.noisy_path.is_file():
        return "missing_noisy"

    if dry_run:
        return "would_write_denoised"

    if backend == "cv2":
        result = denoise_cv2(record.v1_path, record.noisy_path)
    else:
        result = denoise_scipy(record.v1_path, record.noisy_path)

    if result is None:
        return "read_failed"

    final_img = pad_and_resize(result, target_size)
    record.denoised_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        final_img.save(record.denoised_path)
    except OSError:
        return "write_denoised_failed"
    return "denoised_written"


def repair_records(
    records: Iterable[RepairRecord],
    *,
    backend: str,
    target_size: int,
    force_invert_v1: bool,
    dry_run: bool,
) -> Counter:
    counts: Counter = Counter()
    records = list(records)

    for i, record in enumerate(records, 1):
        should_invert, reason = v1_needs_invert(record.v1_path, force=force_invert_v1)
        counts[reason] += 1

        if should_invert:
            status = invert_file_in_place(record.v1_path, dry_run=dry_run)
            counts[status] += 1
            if status not in {"inverted_v1", "would_invert_v1"}:
                print(f"  [v1跳过] {record.v1_path} ({status})")

        status = regenerate_denoised(
            record,
            backend=backend,
            target_size=target_size,
            dry_run=dry_run,
        )
        counts[status] += 1
        if status not in {"denoised_written", "would_write_denoised"}:
            print(
                f"  [denoise跳过] v1={record.v1_path} "
                f"noisy={record.noisy_path} ({status})"
            )

        if i % 25 == 0 or i == len(records):
            print(f"  进度: {i}/{len(records)}")

    return counts


def verify(records: list[RepairRecord], target_size: int) -> Counter:
    counts: Counter = Counter()
    bad_examples = []

    for record in records:
        if not record.v1_path.is_file():
            counts["verify_missing_v1"] += 1
            continue
        if not record.denoised_path.is_file():
            counts["verify_missing_denoised"] += 1
            continue

        v1 = gray_array(record.v1_path)
        v1_border = border_mean(v1)
        if v1_border > 191:
            counts["v1_white_border"] += 1
        elif v1_border < 64:
            counts["v1_dark_border"] += 1
        else:
            counts["v1_mid_border"] += 1

        img = Image.open(record.denoised_path)
        counts[f"denoised_size_{img.size[0]}x{img.size[1]}"] += 1
        counts[f"denoised_mode_{img.mode}"] += 1
        if img.size != (target_size, target_size):
            counts["bad_size"] += 1

        arr = np.asarray(img.convert("L"))
        white_ratio = float((arr > 223).mean())
        dark_ratio = float((arr < 32).mean())
        border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
        border_white = float((border > 223).mean())

        bad = False
        if white_ratio < 0.50 and dark_ratio > 0.50:
            bad = True
            counts["remaining_low_white_high_dark"] += 1
        if border_white < 0.80:
            bad = True
            counts["remaining_dark_border_low_white"] += 1
        if bad and len(bad_examples) < 20:
            bad_examples.append(
                f"{record.font_name}/{record.filename} "
                f"white={white_ratio:.3f} dark={dark_ratio:.3f} "
                f"border_white={border_white:.3f}"
            )

    remaining = (
        counts["remaining_low_white_high_dark"]
        + counts["remaining_dark_border_low_white"]
    )
    counts["remaining_issue_count"] = remaining
    if bad_examples:
        print("问题样例:")
        for item in bad_examples:
            print(f"  {item}")
    return counts


def make_sample_sheet(records: list[RepairRecord], output_path: Path, sample_count: int):
    if not records or sample_count <= 0:
        return
    selected = records[:sample_count]
    thumb = 160
    label_h = 28
    cols = min(4, len(selected))
    rows = (len(selected) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)

    for i, record in enumerate(selected):
        if not record.denoised_path.is_file():
            continue
        img = Image.open(record.denoised_path).convert("RGB")
        img = img.resize((thumb, thumb), Image.Resampling.LANCZOS)
        x = (i % cols) * thumb
        y = (i // cols) * (thumb + label_h)
        sheet.paste(img, (x, y))
        draw.text((x + 4, y + thumb + 4), f"{record.font_name}/{record.filename}", fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def print_counts(title: str, counts: Counter):
    print(title)
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")


def main():
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    csv_path = resolve_path(repo_root, args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    backend = choose_backend(args.backend)
    records = load_records(
        csv_path,
        repo_root,
        args.path_column,
        args.v1_subdir,
        args.noisy_subdir,
        args.denoised_subdir,
    )
    if not records:
        raise RuntimeError("No repair records were loaded from the CSV.")

    print(f"repo_root: {repo_root}")
    print(f"csv: {csv_path}")
    print(f"backend: {backend}")
    print(f"records: {len(records)}")
    print(f"target_size: {args.target_size}x{args.target_size}")
    print("-" * 40)

    counts = repair_records(
        records,
        backend=backend,
        target_size=args.target_size,
        force_invert_v1=args.force_invert_v1,
        dry_run=args.dry_run,
    )
    print("-" * 40)
    print_counts("处理计数:", counts)

    if args.dry_run:
        print(
            "dry-run 未写入；"
            f"将反色 v1 {counts['would_invert_v1']} 张，"
            f"将重写 denoised {counts['would_write_denoised']} 张。"
        )
        return

    if not args.no_verify:
        print("-" * 40)
        verify_counts = verify(records, args.target_size)
        print_counts("验证计数:", verify_counts)

    if args.sample_sheet:
        sample_path = resolve_path(repo_root, args.sample_sheet)
        make_sample_sheet(records, sample_path, args.sample_count)
        print(f"sample_sheet: {sample_path}")

    print(
        "完成；"
        f"已反色 v1 {counts['inverted_v1']} 张，"
        f"已重写 denoised {counts['denoised_written']} 张。"
    )


if __name__ == "__main__":
    main()
