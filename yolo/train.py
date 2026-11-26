"""Train the standalone Calli stroke-phase/JT-component instance segmenter.

Run ``python train.py --dry-run`` to validate a portable dataset without torch.
The model is trained only when main() is explicitly invoked without --dry-run.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_dataset(path: Path) -> dict:
    """Resolve data.yaml relative to its own location, not the launch directory."""
    import yaml

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("data.yaml must contain a mapping")
    names = data.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    elif isinstance(names, dict):
        names = {int(key): value for key, value in names.items()}
    else:
        raise ValueError("data.yaml must define names as a list or mapping")
    if not names or sorted(names) != list(range(len(names))):
        raise ValueError("Class IDs must be contiguous integers starting at zero")
    root = Path(str(data.get("path", "."))).expanduser()
    root = root.resolve() if root.is_absolute() else (path.parent / root).resolve()
    resolved = {"path": str(root), "names": names, "nc": len(names)}
    for split in ("train", "val", "test"):
        value = data.get(split)
        if not value:
            if split != "test":
                raise ValueError(f"Missing {split} split in data.yaml")
            continue
        entries = value if isinstance(value, list) else [value]
        paths = []
        for entry in entries:
            item = Path(str(entry)).expanduser()
            paths.append(str(item.resolve() if item.is_absolute() else (root / item).resolve()))
        resolved[split] = paths if isinstance(value, list) else paths[0]
    return resolved


def split_images(value: str | list[str]) -> list[Path]:
    """Expand image directories or image-list files using Ultralytics conventions."""
    result = []
    entries = value if isinstance(value, list) else [value]
    for entry in entries:
        path = Path(entry)
        if path.is_dir():
            result.extend(p.resolve() for p in sorted(path.rglob("*"))
                          if p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                image = Path(line).expanduser()
                if not image.is_absolute() and not line.startswith("./"):
                    raise ValueError(
                        f"Relative image-list entries must start with ./ to match "
                        f"Ultralytics path handling: {path}: {line}"
                    )
                image = image if image.is_absolute() else path.parent / image
                if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                    raise ValueError(f"Invalid image-list entry: {image}")
                result.append(image.resolve())
        else:
            raise ValueError(f"Split must be an image directory or .txt list: {path}")
    if not result:
        raise ValueError(f"Empty dataset split: {value}")
    if len(set(result)) != len(result):
        raise ValueError(f"Duplicate image paths within split: {value}")
    return result


def label_path(image: Path) -> Path:
    parts = list(image.parts)
    positions = [i for i, item in enumerate(parts) if item == "images"]
    if not positions:
        raise ValueError(f"Image path must contain an images/ directory: {image}")
    parts[positions[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def validate_dataset(data: dict) -> dict:
    """Check paired segmentation labels, IDs, coordinates, and path leakage."""
    report = {}
    previous = set()
    class_count = len(data["names"])
    for split in ("train", "val", "test"):
        if split not in data:
            continue
        images = split_images(data[split])
        overlap = previous.intersection(images)
        if overlap:
            raise ValueError(f"Image paths overlap between splits: {next(iter(overlap))}")
        previous.update(images)
        counts = [0] * class_count
        empty = 0
        for image in images:
            label = label_path(image)
            if not label.is_file():
                raise FileNotFoundError(f"Missing segmentation label: {label}")
            rows = [line.split() for line in label.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            empty += not bool(rows)
            for row in rows:
                if len(row) < 7 or len(row) % 2 != 1:
                    raise ValueError(f"Expected class and at least 3 polygon points: {label}")
                class_id = int(row[0])
                if not 0 <= class_id < class_count:
                    raise ValueError(f"Class ID outside data.yaml names: {label}")
                coords = [float(value) for value in row[1:]]
                if any(not math.isfinite(value) or not 0 <= value <= 1 for value in coords):
                    raise ValueError(f"Nonfinite or nonnormalized polygon coordinate: {label}")
                if len(set(zip(coords[::2], coords[1::2]))) < 3:
                    raise ValueError(f"Polygon has fewer than 3 distinct points: {label}")
                counts[class_id] += 1
        if not sum(counts):
            raise ValueError(f"No instances in {split} split")
        report[split] = {
            "images": len(images), "instances": sum(counts),
            "empty_labels": empty, "instances_per_class": counts,
        }
    # Byte/image-identity grouping is recorded by the dataset builder. This
    # inexpensive launcher check deliberately does not claim to re-audit it.
    return report


def validate_resume_checkpoint(checkpoint: object) -> None:
    """Reject completed/stripped weights before Ultralytics can start a new run."""
    if not isinstance(checkpoint, dict):
        raise ValueError("--resume requires a full training checkpoint dictionary")
    epoch = checkpoint.get("epoch")
    if not isinstance(epoch, int) or epoch < 0 or checkpoint.get("optimizer") is None:
        raise ValueError(
            "Cannot resume this checkpoint: its epoch or optimizer state is missing, "
            "or training has completed and the optimizer was stripped. "
            "Use an unfinished last.pt to resume, or start a separate fine-tuning run "
            "with --model PATH_TO_WEIGHTS --name NEW_RUN_NAME."
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path, default=ROOT / "dataset" / "data.yaml")
    result.add_argument("--model", default="yolo26s-seg.pt")
    result.add_argument("--epochs", type=int, default=100)
    result.add_argument("--batch", type=int, default=8)
    result.add_argument("--imgsz", type=int, default=640)
    result.add_argument("--patience", type=int, default=25)
    result.add_argument("--device", default="0", help="Colab GPU: 0; local validation only: cpu")
    result.add_argument("--workers", type=int, default=2)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--project", type=Path, default=ROOT / "runs")
    result.add_argument("--name", default="calli_phase_jieti_yolo26s")
    result.add_argument("--resume", type=Path, metavar="LAST_PT",
                        help="Resume the original run from last.pt, preserving its schedule")
    result.add_argument("--resume-output", type=Path,
                        help="Resume output directory; defaults to LAST_PT's run/ directory")
    result.add_argument("--dry-run", action="store_true",
                        help="Validate data and print settings; no model import/download/update")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32 or args.patience < 0:
        raise ValueError("epochs/batch must be positive, imgsz >= 32, patience >= 0")
    data = read_dataset(args.data)
    report = validate_dataset(data)
    if args.resume and not args.resume.expanduser().is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume}")
    if args.resume_output and not args.resume:
        raise ValueError("--resume-output requires --resume")
    resume_output = None
    if args.resume:
        resume_output = (
            args.resume_output.expanduser().resolve() if args.resume_output
            else args.resume.expanduser().resolve().parent.parent
        )
    model_path = Path(args.model).expanduser()
    if len(model_path.parts) == 1 and not model_path.is_file():
        model_path = ROOT / "models" / model_path
    model_path = model_path.resolve()
    settings = {
        "epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz,
        "patience": args.patience, "device": args.device, "workers": args.workers,
        "seed": args.seed, "project": str(args.project.expanduser().resolve()),
        "name": args.name, "overlap_mask": False, "fliplr": 0.0, "flipud": 0.0,
        "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0, "degrees": 0.0,
        "translate": 0.03, "scale": 0.10, "shear": 0.0, "perspective": 0.0,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.1,
        "save": True, "save_period": 10, "plots": True,
    }
    plan = {
        "model": str(model_path), "resume": str(args.resume) if args.resume else None,
        "resume_output": str(resume_output) if resume_output else None,
        "resolved_data": data, "dataset_check": report, "train_settings": settings,
        "note": "Resume restores the checkpoint schedule; new-run defaults are not reapplied."
        if args.resume else "COCO initialization is not a trained Calli recognizer.",
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    import yaml
    from ultralytics import YOLO

    if not args.resume:
        model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = str(args.resume.expanduser().resolve()) if args.resume else str(model_path)
    model = YOLO(checkpoint)
    if args.resume:
        validate_resume_checkpoint(model.ckpt)
    if model.task != "segment":
        raise ValueError("This dataset requires an instance-segmentation (-seg) model")
    config_dir = Path(settings["project"]) / "_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    runtime_yaml = config_dir / f"dataset_{stamp}.yaml"
    runtime_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (config_dir / f"launch_{stamp}.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.resume:
        model.train(
            resume=True, data=str(runtime_yaml), device=args.device, workers=args.workers,
            save_dir=str(resume_output),
        )
    else:
        model.train(data=str(runtime_yaml), **settings)
    print(f"Training finished. Actual output directory: {model.trainer.save_dir}")


if __name__ == "__main__":
    main()
