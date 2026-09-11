#!/usr/bin/env python3
"""Batch denoise font images with COCO ``text`` annotation masks.

The ``text`` category is treated as the whole-glyph contour. By default this
script traverses every font folder under ``fontdata_example/font/train/new``:

  <font>/annotations/instances_default.json
  <font>/images/*.png
  -> <font>/images_text_mask_denoised/*.png

Default mode preserves original pixels inside the text mask and sets everything
outside the mask to white. Set ``MODE = "mask"`` to rebuild pure black glyphs
from the annotation mask.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, UnidentifiedImageError

try:
    from preprocess_common import IMAGE_EXTS
except ImportError:
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


# ============================================================
# 路径配置：通常只需要改这里
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

# 批量字体根目录。脚本会遍历 NEW_DIR 下每个字体文件夹。
NEW_DIR = Path('/Users/root1/Desktop/Fontify-main/fontdata_example/font/train/new')

# 只处理指定字体时填列表，例如 ["CaoqbBF", "DongqcBF"]；None 表示处理所有字体。
FONT_NAMES: list[str] | None = None

# 每个字体目录内部的固定子路径。
ANNOTATIONS_SUBDIR = "annotations"
ANNOTATIONS_FILENAME = "instances_default.json"
IMAGE_SUBDIR = "images"
OUTPUT_SUBDIR = "images_text_mask_denoised"


# ============================================================
# 标注与去噪参数：通常只需要改这里
# ============================================================

# CVAT/COCO 中整字轮廓类别名。
CATEGORY_NAME = "text"

# preserve: mask 内保留原图像素，mask 外变白。
# mask: 直接用 text 标注重建纯黑字形，背景为白。
MODE = "mask"

# text mask 扩张像素数。标注贴边过紧时可设为 1~3。
DILATE_PX = 0

# mask 边缘羽化半径。想保留抗锯齿过渡可设为 0.5~1.5。
FEATHER_RADIUS = 0.0

# 输出目录已存在时是否覆盖其中同名文件。
OVERWRITE = True

# True 只统计，不写图。
DRY_RUN = False

# True 遇到缺图/缺 text 标注/不支持的标注格式立即报错。
STRICT = False

# 每处理多少张图打印一次进度。
PROGRESS_EVERY = 50


@dataclass(frozen=True)
class Config:
    new_dir: Path = NEW_DIR
    font_names: tuple[str, ...] | None = None
    annotations_subdir: str = ANNOTATIONS_SUBDIR
    annotations_filename: str = ANNOTATIONS_FILENAME
    image_subdir: str = IMAGE_SUBDIR
    output_subdir: str = OUTPUT_SUBDIR
    category_name: str = CATEGORY_NAME
    mode: str = MODE
    dilate_px: int = DILATE_PX
    feather_radius: float = FEATHER_RADIUS
    overwrite: bool = OVERWRITE
    dry_run: bool = DRY_RUN
    strict: bool = STRICT
    progress_every: int = PROGRESS_EVERY


def default_config() -> Config:
    names = tuple(FONT_NAMES) if FONT_NAMES else None
    return Config(font_names=names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch denoise font images using COCO category named text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--new-dir", type=Path, default=None)
    parser.add_argument(
        "--font",
        action="append",
        default=None,
        help="Font folder name to process. Repeat for multiple fonts.",
    )
    parser.add_argument("--image-subdir", default=None)
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument("--annotations-subdir", default=None)
    parser.add_argument("--annotations-filename", default=None)
    parser.add_argument("--category-name", default=None)
    parser.add_argument("--mode", choices=("preserve", "mask"), default=None)
    parser.add_argument("--dilate-px", type=int, default=None)
    parser.add_argument("--feather-radius", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> Config:
    base = default_config()
    overwrite = base.overwrite
    if args.overwrite:
        overwrite = True
    if args.no_overwrite:
        overwrite = False

    return Config(
        new_dir=args.new_dir or base.new_dir,
        font_names=tuple(args.font) if args.font else base.font_names,
        annotations_subdir=args.annotations_subdir or base.annotations_subdir,
        annotations_filename=args.annotations_filename or base.annotations_filename,
        image_subdir=args.image_subdir or base.image_subdir,
        output_subdir=args.output_subdir or base.output_subdir,
        category_name=args.category_name or base.category_name,
        mode=args.mode or base.mode,
        dilate_px=base.dilate_px if args.dilate_px is None else args.dilate_px,
        feather_radius=(
            base.feather_radius if args.feather_radius is None else args.feather_radius
        ),
        overwrite=overwrite,
        dry_run=base.dry_run or args.dry_run,
        strict=base.strict or args.strict,
        progress_every=base.progress_every,
    )


def load_coco(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_category_id(categories: list[dict[str, Any]], category_name: str) -> int:
    matches = [cat["id"] for cat in categories if cat.get("name") == category_name]
    if not matches:
        names = ", ".join(str(cat.get("name")) for cat in categories)
        raise ValueError(f"Category {category_name!r} not found. Available: {names}")
    if len(matches) > 1:
        raise ValueError(f"Category {category_name!r} appears multiple times")
    return int(matches[0])


def annotations_by_image(
    annotations: list[dict[str, Any]], category_id: int
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        if int(ann.get("category_id", -1)) == category_id:
            grouped[int(ann["image_id"])].append(ann)
    return grouped


def decode_uncompressed_rle_mask(segmentation: dict[str, Any]) -> Image.Image:
    """Decode uncompressed COCO RLE segmentation to an L mask."""
    counts = segmentation.get("counts")
    mask_size = segmentation.get("size")
    if not isinstance(counts, list) or not isinstance(mask_size, list):
        raise NotImplementedError(
            "Only uncompressed COCO RLE with counts as a list is supported."
        )

    height, width = int(mask_size[0]), int(mask_size[1])
    flat = np.zeros(height * width, dtype=np.uint8)
    cursor = 0
    foreground = False

    for count in counts:
        count = int(count)
        next_cursor = min(cursor + count, flat.size)
        if foreground and next_cursor > cursor:
            flat[cursor:next_cursor] = 255
        cursor = next_cursor
        foreground = not foreground
        if cursor >= flat.size:
            break

    # COCO RLE is stored in column-major order.
    arr = flat.reshape((width, height)).T
    return Image.fromarray(arr, mode="L")


def render_segmentation_mask(segmentation: Any, size: tuple[int, int]) -> Image.Image:
    """Render COCO polygon or uncompressed RLE segmentation to an L mask.

    ``size`` is (width, height), matching PIL's coordinate convention.
    """
    if isinstance(segmentation, dict):
        return decode_uncompressed_rle_mask(segmentation)

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    if not segmentation:
        return mask

    polygons = segmentation
    if isinstance(segmentation[0], (int, float)):
        polygons = [segmentation]

    for poly in polygons:
        if len(poly) < 6:
            continue
        if len(poly) % 2 != 0:
            poly = poly[:-1]
        points = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly), 2)]
        draw.polygon(points, fill=255)
    return mask


def render_text_mask(image_info: dict[str, Any], anns: list[dict[str, Any]]) -> Image.Image:
    width = int(image_info["width"])
    height = int(image_info["height"])
    combined = Image.new("L", (width, height), 0)
    for ann in anns:
        ann_mask = render_segmentation_mask(ann.get("segmentation"), (width, height))
        if ann_mask.size != (width, height):
            ann_mask = ann_mask.resize((width, height), Image.Resampling.NEAREST)
        combined = ImageChops.lighter(combined, ann_mask)
    return combined


def prepare_mask(
    mask: Image.Image,
    size: tuple[int, int],
    dilate_px: int,
    feather_radius: float,
) -> Image.Image:
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    if dilate_px > 0:
        kernel_size = max(3, 2 * dilate_px + 1)
        mask = mask.filter(ImageFilter.MaxFilter(kernel_size))
    if feather_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))
    return mask


def flatten_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A"))
        return background
    return image.convert("RGB")


def apply_text_mask(image: Image.Image, mask: Image.Image, mode: str) -> Image.Image:
    if mode == "mask":
        result = Image.new("RGB", image.size, "white")
        black = Image.new("RGB", image.size, "black")
        result.paste(black, mask=mask)
        return result

    source = flatten_to_rgb(image)
    result = Image.new("RGB", source.size, "white")
    result.paste(source, mask=mask)
    return result


def iter_font_dirs(config: Config) -> list[Path]:
    if not config.new_dir.is_dir():
        raise FileNotFoundError(f"NEW_DIR does not exist: {config.new_dir}")

    if config.font_names:
        return [config.new_dir / name for name in config.font_names]

    return sorted(
        p
        for p in config.new_dir.iterdir()
        if p.is_dir() and not p.name.startswith(("_", "."))
    )


def output_path_for(output_dir: Path, file_name: str) -> Path:
    out_path = output_dir / file_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def process_font_dir(font_dir: Path, config: Config) -> tuple[int, int]:
    ann_path = font_dir / config.annotations_subdir / config.annotations_filename
    image_dir = font_dir / config.image_subdir
    output_dir = font_dir / config.output_subdir

    if not ann_path.is_file() or not image_dir.is_dir():
        message = f"Missing annotation or image directory: {font_dir}"
        if config.strict:
            raise FileNotFoundError(message)
        print(f"[skip-font] {message}")
        return 0, 0

    data = load_coco(ann_path)
    text_cat_id = find_category_id(data["categories"], config.category_name)
    text_anns = annotations_by_image(data["annotations"], text_cat_id)

    if not config.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for image_info in data["images"]:
        file_name = image_info["file_name"]
        if Path(file_name).suffix.lower() not in IMAGE_EXTS:
            skipped += 1
            continue

        img_path = image_dir / file_name
        out_path = output_dir / file_name
        anns = text_anns.get(int(image_info["id"]), [])

        if not img_path.is_file() or not anns:
            message = f"Missing image or text annotation: {font_dir.name}/{file_name}"
            if config.strict:
                raise FileNotFoundError(message)
            print(f"[skip] {message}")
            skipped += 1
            continue

        if out_path.exists() and not config.overwrite:
            skipped += 1
            continue

        if config.dry_run:
            processed += 1
            continue

        try:
            with Image.open(img_path) as image:
                image.load()
                mask = render_text_mask(image_info, anns)
                mask = prepare_mask(
                    mask,
                    image.size,
                    dilate_px=max(0, config.dilate_px),
                    feather_radius=max(0.0, config.feather_radius),
                )
                result = apply_text_mask(image, mask, config.mode)
                out_path = output_path_for(output_dir, file_name)
                result.save(out_path)
        except (OSError, UnidentifiedImageError, NotImplementedError) as exc:
            if config.strict:
                raise
            print(f"[skip] Cannot process {font_dir.name}/{file_name}: {exc}")
            skipped += 1
            continue

        processed += 1
        if config.progress_every > 0 and processed % config.progress_every == 0:
            print(f"  {font_dir.name}: processed {processed} images...")

    return processed, skipped


def process_all_fonts(config: Config) -> tuple[int, int, int]:
    font_dirs = iter_font_dirs(config)
    total_processed = 0
    total_skipped = 0
    fonts_done = 0

    print(f"NEW_DIR: {config.new_dir}")
    print(f"Fonts: {len(font_dirs)}")
    print(f"Image subdir: {config.image_subdir}")
    print(f"Output subdir: {config.output_subdir}")
    print(f"Category: {config.category_name!r}")
    print(f"Mode: {config.mode}")
    print(f"Dry run: {config.dry_run}")
    print("=" * 60)

    for index, font_dir in enumerate(font_dirs, 1):
        print(f"[{index}/{len(font_dirs)}] {font_dir.name}")
        processed, skipped = process_font_dir(font_dir, config)
        total_processed += processed
        total_skipped += skipped
        if processed > 0:
            fonts_done += 1
        print(f"  done: processed={processed}, skipped={skipped}")

    return fonts_done, total_processed, total_skipped


def main() -> None:
    config = config_from_args(parse_args())
    fonts_done, processed, skipped = process_all_fonts(config)
    print("=" * 60)
    print(f"Fonts processed: {fonts_done}")
    print(f"Images processed: {processed}")
    print(f"Images skipped: {skipped}")


if __name__ == "__main__":
    main()
