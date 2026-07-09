#!/usr/bin/env python3
"""Run the full preprocessing flow for every font folder under font/train/new.

Default flow:
1. images -> images_white_bg, using convert_to_white_bg_cuda.py
2. images -> images_white_bg_v1
3. images_white_bg + images_white_bg_v1 -> images_white_bg_mask_denoised
4. resize images_white_bg_mask_denoised in place to 448x448
5. annotations/*.json -> semantic_masks/*.npy
6. regenerate train_json_new/ and val_json_new/
"""

from __future__ import annotations

import argparse
import importlib.util
import importlib
from pathlib import Path

from preprocess_common import DEFAULT_NEW_DIR, REPO_ROOT


STAGES = ("white-bg", "white-bg-v1", "denoise", "resize", "semantic", "json")


def run_generate_json():
    script = REPO_ROOT / "fontdata_example" / "generate_new_json.py"
    spec = importlib.util.spec_from_file_location("fontdata_generate_new_json", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.main()


def parse_args():
    parser = argparse.ArgumentParser(description="Batch preprocess all font folders.")
    parser.add_argument("--new_dir", default=str(DEFAULT_NEW_DIR), help="font/train/new directory")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=list(STAGES),
        choices=STAGES,
        help="subset of stages to run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    new_dir = Path(args.new_dir)
    stages = set(args.stages)

    print(f"new_dir={new_dir}")
    print(f"stages={', '.join(args.stages)}")

    if "white-bg" in stages:
        print("\n=== 1. images -> images_white_bg (CUDA) ===")
        convert_to_white_bg_cuda = importlib.import_module("convert_to_white_bg_cuda")
        convert_to_white_bg_cuda.process_all_fonts(new_dir)

    if "white-bg-v1" in stages:
        print("\n=== 2. images -> images_white_bg_v1 ===")
        convert_to_white_bg_v1_failed = importlib.import_module("convert_to_white_bg_v1_failed")
        convert_to_white_bg_v1_failed.process_all_fonts(new_dir)

    if "denoise" in stages:
        print("\n=== 3. images_white_bg_mask_denoised ===")
        mask_denoise = importlib.import_module("mask_denoise")
        mask_denoise.NEW_DIR = str(new_dir)
        mask_denoise.main()

    if "resize" in stages:
        print("\n=== 4. resize final images to 448 ===")
        pad_and_resize = importlib.import_module("pad_and_resize")
        pad_and_resize.NEW_DIR = new_dir
        pad_and_resize.main()

    if "semantic" in stages:
        print("\n=== 5. annotations JSON -> semantic_masks ===")
        render_semantic_mask = importlib.import_module("render_semantic_mask")
        render_semantic_mask.NEW_DIR = new_dir
        render_semantic_mask.batch_mode(verbose=True)

    if "json" in stages:
        print("\n=== 6. regenerate train/val JSON ===")
        run_generate_json()


if __name__ == "__main__":
    main()
