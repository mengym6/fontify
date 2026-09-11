#!/usr/bin/env python3
"""Evaluate generated glyphs against GT glyphs grouped by font style."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# Edit these defaults if you prefer running the script without command-line args.
# Directory structure must be:
#   GEN_ROOT/<font_style>/<image_name>
#   GT_ROOT/<font_style>/<GT_IMAGE_SUBDIR>/<image_name>
GEN_ROOT = "/Users/root1/Desktop/fontify/generate/cv_fold_3_stroke_mask/40"
GT_ROOT = "/Users/root1/Desktop/Fontify-main/fontdata_example/font/train/new"
GT_IMAGE_SUBDIR = "images_white_bg_mask_denoised"
OUT_DIR = "/Users/root1/Desktop/fontify/generate/cv_fold_3_stroke_mask/40"
PIXEL_MODE = "gray"  # "gray" or "rgb"
LPIPS_BACKBONE = "vgg"  # "alex", "vgg", or "squeeze"
DEVICE = "cuda"  # "auto", "cpu", or "cuda"
STRICT_PAIRING = False
WRITE_IMAGE_METRICS = False
EVAL_SIZE = 448  # Resize both generated and GT glyphs to this square size.
RESIZE_GT_TO_GEN = True


@dataclass(frozen=True)
class ImageMetrics:
    style: str
    name: str
    psnr: float
    ssim: float
    mae: float
    lpips: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute PSNR, SSIM, MAE, and LPIPS for generated glyph images. "
            "Images are paired as gen/<style>/<image> and gt/<style>/<image>. "
            "The main CSV contains one row per font style and a final average row."
        )
    )
    parser.add_argument("--gen", default=GEN_ROOT, help="Generated glyph root directory.")
    parser.add_argument("--gt", default=GT_ROOT, help="Ground-truth glyph root directory.")
    parser.add_argument(
        "--gt_image_subdir",
        default=GT_IMAGE_SUBDIR,
        help=(
            "Subdirectory under each GT style directory that contains GT images. "
            "Pass an empty string if images are directly under each style directory."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["gray", "rgb"],
        default=PIXEL_MODE,
        help="Pixel mode for PSNR/SSIM/MAE. Font glyphs usually use gray.",
    )
    parser.add_argument(
        "--lpips_net",
        choices=["alex", "vgg", "squeeze"],
        default=LPIPS_BACKBONE,
        help="LPIPS backbone. Default follows the requested VGG backbone.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=DEVICE,
        help="Device used for LPIPS.",
    )
    parser.add_argument(
        "--out_dir",
        default=OUT_DIR,
        help="Directory for metrics_by_font.csv.",
    )
    parser.add_argument(
        "--write_image_metrics",
        action="store_true",
        default=WRITE_IMAGE_METRICS,
        help="Also write per-image metrics to image_metrics.csv for debugging.",
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        default=EVAL_SIZE,
        help=(
            "Resize both generated and GT images to this square size before metrics. "
            "Use 0 to disable fixed-size resizing."
        ),
    )
    parser.add_argument(
        "--resize_gt_to_gen",
        action="store_true",
        default=RESIZE_GT_TO_GEN,
        help=(
            "When --eval_size 0 is used, resize GT images to each generated image "
            "size before computing metrics."
        ),
    )
    parser.add_argument(
        "--no_resize_gt_to_gen",
        action="store_false",
        dest="resize_gt_to_gen",
        help="Fail instead of resizing when generated and GT image sizes differ.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=STRICT_PAIRING,
        help="Fail instead of skipping when images do not match within a matched style.",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_false",
        dest="strict",
        help="Skip unmatched images within a matched style.",
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def style_dirs(root: Path) -> dict[str, Path]:
    return {path.name: path for path in sorted(root.iterdir()) if path.is_dir()}


def image_files(style_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(style_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files[path.relative_to(style_dir).as_posix()] = path
    return files


def load_image(path: Path, mode: str, size: tuple[int, int] | None = None) -> np.ndarray:
    pil_mode = "L" if mode == "gray" else "RGB"
    image = Image.open(path).convert(pil_mode)
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image).astype(np.float32) / 255.0


def lpips_tensor(image: np.ndarray, mode: str, device: torch.device) -> torch.Tensor:
    if mode == "gray":
        image = np.repeat(image[..., None], 3, axis=2)
    image = np.ascontiguousarray(image)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor * 2.0 - 1.0


def compute_pair(
    style: str,
    name: str,
    gen_path: Path,
    gt_path: Path,
    mode: str,
    lpips_fn: torch.nn.Module,
    device: torch.device,
    eval_size: int,
    resize_gt_to_gen: bool,
) -> ImageMetrics:
    fixed_size = (eval_size, eval_size) if eval_size > 0 else None
    gen = load_image(gen_path, mode, size=fixed_size)
    gt_size = fixed_size
    if gt_size is None and resize_gt_to_gen:
        gt_size = (gen.shape[1], gen.shape[0])
    gt = load_image(gt_path, mode, size=gt_size)

    if gen.shape != gt.shape:
        raise RuntimeError(
            f"Shape mismatch for {style}/{name}: gen={gen.shape}, gt={gt.shape}"
        )

    psnr = peak_signal_noise_ratio(gt, gen, data_range=1.0)
    if mode == "gray":
        ssim = structural_similarity(gt, gen, data_range=1.0)
    else:
        ssim = structural_similarity(gt, gen, channel_axis=-1, data_range=1.0)
    mae = float(np.mean(np.abs(gen - gt)))

    with torch.no_grad():
        gen_tensor = lpips_tensor(gen, mode, device)
        gt_tensor = lpips_tensor(gt, mode, device)
        lpips_value = float(lpips_fn(gen_tensor, gt_tensor).item())

    return ImageMetrics(
        style=style,
        name=name,
        psnr=float(psnr),
        ssim=float(ssim),
        mae=mae,
        lpips=lpips_value,
    )


def mean_metrics(records: list[ImageMetrics]) -> dict[str, float]:
    return {
        "count": float(len(records)),
        "psnr": float(np.mean([record.psnr for record in records])),
        "ssim": float(np.mean([record.ssim for record in records])),
        "mae": float(np.mean([record.mae for record in records])),
        "lpips": float(np.mean([record.lpips for record in records])),
    }


def write_image_csv(path: Path, records: list[ImageMetrics]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["style", "name", "psnr", "ssim", "mae", "lpips"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "style": record.style,
                    "name": record.name,
                    "psnr": record.psnr,
                    "ssim": record.ssim,
                    "mae": record.mae,
                    "lpips": record.lpips,
                }
            )


def write_summary_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Font", "PSNR", "SSIM", "MAE", "LPIPS"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Font": row["style"],
                    "PSNR": f"{row['psnr']:.4f}",
                    "SSIM": f"{row['ssim']:.4f}",
                    "MAE": f"{row['mae']:.4f}",
                    "LPIPS": f"{row['lpips']:.4f}",
                }
            )


def fail_or_warn(strict: bool, message: str) -> None:
    if strict:
        raise RuntimeError(message)
    print(f"Warning: {message}")


def main() -> None:
    args = parse_args()
    if not args.gen:
        raise RuntimeError("Set GEN_ROOT at the top of this script, or pass --gen.")
    if not args.gt:
        raise RuntimeError("Set GT_ROOT at the top of this script, or pass --gt.")

    gen_root = Path(args.gen).expanduser()
    gt_root = Path(args.gt).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    if not gen_root.is_dir():
        raise RuntimeError(f"Generated root is not a directory: {gen_root}")
    if not gt_root.is_dir():
        raise RuntimeError(f"GT root is not a directory: {gt_root}")

    gen_styles = style_dirs(gen_root)
    gt_styles = style_dirs(gt_root)
    common_styles = sorted(set(gen_styles) & set(gt_styles))
    if not common_styles:
        raise RuntimeError("No matched style directories found.")

    missing_gen_styles = sorted(set(gt_styles) - set(gen_styles))
    missing_gt_styles = sorted(set(gen_styles) - set(gt_styles))
    if missing_gen_styles:
        print(f"Skipping {len(missing_gen_styles)} styles that exist only in GT.")
    if missing_gt_styles:
        print(f"Skipping {len(missing_gt_styles)} styles that exist only in generated results.")

    device = choose_device(args.device)
    lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device).eval()

    all_records: list[ImageMetrics] = []
    style_rows: list[dict[str, float | str]] = []

    for style in common_styles:
        gen_images = image_files(gen_styles[style])
        gt_image_dir = (
            gt_styles[style] / args.gt_image_subdir
            if args.gt_image_subdir
            else gt_styles[style]
        )
        if not gt_image_dir.is_dir():
            fail_or_warn(args.strict, f"{style}: GT image directory not found: {gt_image_dir}")
            continue

        gt_images = image_files(gt_image_dir)
        common_images = sorted(set(gen_images) & set(gt_images))

        if not common_images:
            fail_or_warn(args.strict, f"No matched images under style {style}.")
            continue

        missing_gen_images = sorted(set(gt_images) - set(gen_images))
        missing_gt_images = sorted(set(gen_images) - set(gt_images))
        if missing_gen_images:
            fail_or_warn(args.strict, f"{style}: {len(missing_gen_images)} images exist only in GT.")
        if missing_gt_images:
            fail_or_warn(args.strict, f"{style}: {len(missing_gt_images)} images exist only in generated results.")

        style_records = []
        for name in common_images:
            try:
                style_records.append(
                    compute_pair(
                        style=style,
                        name=name,
                        gen_path=gen_images[name],
                        gt_path=gt_images[name],
                        mode=args.mode,
                        lpips_fn=lpips_fn,
                        device=device,
                        eval_size=args.eval_size,
                        resize_gt_to_gen=args.resize_gt_to_gen,
                    )
                )
            except (FileNotFoundError, UnidentifiedImageError) as error:
                if args.strict:
                    raise
                print(f"Warning: Skipping unreadable image pair {style}/{name}: {error}")

        if not style_records:
            fail_or_warn(args.strict, f"No readable matched images under style {style}.")
            continue

        all_records.extend(style_records)

        summary = mean_metrics(style_records)
        style_rows.append({"style": style, **summary})

    if not all_records:
        raise RuntimeError("No image pairs were evaluated.")

    overall = mean_metrics(all_records)
    style_rows.append({"style": "Average", **overall})

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "metrics_by_font.csv"
    write_summary_csv(summary_csv, style_rows)
    if args.write_image_metrics:
        write_image_csv(out_dir / "image_metrics.csv", all_records)

    print(f"Styles evaluated: {len(style_rows) - 1}")
    print(f"Image pairs evaluated: {len(all_records)}")
    print(f"LPIPS backbone: {args.lpips_net}")
    print(f"Pixel mode: {args.mode}")
    if args.eval_size > 0:
        print(f"Evaluation size: {args.eval_size}x{args.eval_size}")
    else:
        print("Evaluation size: native generated size")
    print(
        "Overall: "
        f"PSNR={overall['psnr']:.4f}, "
        f"SSIM={overall['ssim']:.4f}, "
        f"MAE={overall['mae']:.4f}, "
        f"LPIPS={overall['lpips']:.4f}"
    )
    print(f"Wrote: {summary_csv}")
    if args.write_image_metrics:
        print(f"Wrote: {out_dir / 'image_metrics.csv'}")


if __name__ == "__main__":
    main()
