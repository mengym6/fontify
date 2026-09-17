"""Build fixed 1:1 chinese/CalliPhase JSON manifests for stage 2.

The source JSONs are read from the existing train_json(_new)/val_json manifests;
paths remain relative to ``--data-root``.  All CalliPhase samples are retained
and an equal-size, deterministic sample is drawn from the chinese split.
Chinese source images are rewritten to ``ttf/source`` so the mixed manifests
do not inherit the larger SourceHanSans source directory used by stage 1.
"""
import argparse
import json
import random
from pathlib import Path


def read_items(paths, source):
    items = []
    for path in sorted(paths):
        with path.open(encoding="utf-8") as f:
            cur = json.load(f)
        for item in cur:
            item = dict(item)
            item["source_dataset"] = source
            items.append(item)
    return items


def rewrite_chinese_source(items, source_dir, data_root):
    """Point chinese samples at ttf/source using the target basename."""
    source_dir = source_dir.resolve()
    try:
        source_rel = source_dir.relative_to(data_root.resolve())
    except ValueError:
        raise RuntimeError(
            f"chinese source directory must be inside data root for relative paths: {source_dir}"
        )

    missing = []
    for item in items:
        filename = Path(item["target_path"]).name
        source_path = source_dir / filename
        if not source_path.is_file():
            missing.append((item.get("target_path"), item.get("image_path")))
            continue
        item["image_path"] = (source_rel / filename).as_posix()
    if missing:
        sample = "、".join(str(item[0]) for item in missing[:5])
        raise RuntimeError(
            f"{len(missing)} chinese samples lack a ttf/source image, "
            f"for example: {sample}"
        )


def build_split(chinese_dir, calli_dir, output_dir, split, rng, chinese_source_dir, data_root):
    chinese = read_items(chinese_dir.glob("*.json"), "chinese")
    calli = read_items(calli_dir.glob("*.json"), "calliphase")
    if not calli:
        raise RuntimeError(f"{split}: no CalliPhase samples found in {calli_dir}")
    if len(chinese) < len(calli):
        raise RuntimeError(f"{split}: chinese samples ({len(chinese)}) < CalliPhase ({len(calli)})")
    chinese = rng.sample(chinese, len(calli))
    rewrite_chinese_source(chinese, chinese_source_dir, data_root)
    merged = chinese + calli
    rng.shuffle(merged)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"stage2_{split}_mixed.json"
    with output.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"{split}: chinese={len(chinese)}, calliphase={len(calli)}, total={len(merged)}")
    print(f"output: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "fontdata_example")
    parser.add_argument(
        "--chinese-source-dir",
        type=Path,
        default=None,
        help="chinese source image directory, default: data-root/ttf/source",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = args.data_root
    chinese_source_dir = args.chinese_source_dir or root / "ttf" / "source"
    if not chinese_source_dir.is_dir():
        raise RuntimeError(f"chinese source directory does not exist: {chinese_source_dir}")
    rng = random.Random(args.seed)
    build_split(
        root / "train_json",
        root / "train_json_new",
        root / "train_json_mix",
        "train",
        rng,
        chinese_source_dir,
        root,
    )
    build_split(
        root / "val_json",
        root / "val_json_new",
        root / "val_json_mix",
        "val",
        rng,
        chinese_source_dir,
        root,
    )


if __name__ == "__main__":
    main()
