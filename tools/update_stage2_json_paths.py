"""Point existing train/val manifests at text-denoised CalliPhase images."""
import argparse
import json
from pathlib import Path


def update_dir(path: Path) -> tuple[int, int]:
    changed = missing = 0
    for json_path in sorted(path.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for item in data:
            target = item.get("target_path", "")
            if "font/train/new/" not in target:
                continue
            parts = target.split("/")
            try:
                idx = parts.index("new")
                font = parts[idx + 1]
                old_subdir = parts[idx + 2]
                filename = "/".join(parts[idx + 3:])
            except (ValueError, IndexError):
                continue
            if old_subdir.startswith("images_"):
                parts[idx + 2] = "images_text_denoised"
                new_target = "/".join(parts)
                if not Path("fontdata_example") .joinpath(new_target).is_file():
                    missing += 1
                if new_target != target:
                    item["target_path"] = new_target
                    changed += 1
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("fontdata_example"))
    args = parser.parse_args()
    for name in ("train_json_new", "val_json_new", "train_json_mix", "val_json_mix"):
        changed, missing = update_dir(args.root / name)
        print(f"{name}: updated={changed}, missing={missing}")


if __name__ == "__main__":
    main()
