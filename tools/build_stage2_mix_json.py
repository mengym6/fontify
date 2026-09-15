"""Build fixed 1:1 chinese/CalliPhase JSON manifests for stage 2.

The source JSONs are read from the existing train_json(_new)/val_json manifests;
paths remain relative to ``--data-root``.  All CalliPhase samples are retained
and an equal-size, deterministic sample is drawn from the chinese split.
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


def build_split(chinese_dir, calli_dir, output_dir, split, rng):
    chinese = read_items(chinese_dir.glob("*.json"), "chinese")
    calli = read_items(calli_dir.glob("*.json"), "calliphase")
    if not calli:
        raise RuntimeError(f"{split}: no CalliPhase samples found in {calli_dir}")
    if len(chinese) < len(calli):
        raise RuntimeError(f"{split}: chinese samples ({len(chinese)}) < CalliPhase ({len(calli)})")
    chinese = rng.sample(chinese, len(calli))
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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = args.data_root
    rng = random.Random(args.seed)
    build_split(root / "train_json", root / "train_json_new", root / "train_json_mix", "train", rng)
    build_split(root / "val_json", root / "val_json_new", root / "val_json_mix", "val", rng)


if __name__ == "__main__":
    main()
