"""Train the standalone Calli stroke-phase/JT-component instance segmenter.

Run ``python train.py --dry-run`` to validate a portable dataset without torch.
The model is trained only when main() is explicitly invoked without --dry-run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
    if data.get("annotation_availability"):
        availability = Path(str(data["annotation_availability"])).expanduser()
        resolved["annotation_availability"] = str(
            availability.resolve() if availability.is_absolute() else (root / availability).resolve()
        )
        if not isinstance(data.get("split_id"), str) or not data["split_id"]:
            raise ValueError("Partially labelled data must have an explicit split_id")
        resolved["split_id"] = data["split_id"]
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
    from partial_supervision import image_task_valid, load_annotation_availability
    availability = load_annotation_availability(data)
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
        known_counts = {"BF": 0, "JT": 0}
        for image in images:
            has_bf, has_jt = image_task_valid(image, data["path"], availability)
            if not (has_bf or has_jt):
                raise ValueError(f"Image has no available supervision: {image}")
            known_counts["BF"] += has_bf
            known_counts["JT"] += has_jt
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
                if availability is not None and not (has_bf if class_id < 50 else has_jt):
                    raise ValueError(f"Label belongs to a task marked unknown: {label}")
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
            "labelled_images": known_counts,
        }
    # Byte/image-identity grouping is recorded by the dataset builder. This
    # inexpensive launcher check deliberately does not claim to re-audit it.
    return report


def validate_initialization_split(model_names: dict, checkpoint: Path, data: dict) -> None:
    """Keep old Calli weights that saw new validation images out of a new split."""
    if not data.get("split_id") or dict(model_names) != data["names"]:
        return
    sidecar = checkpoint.resolve().parent.parent / "partial_supervision.json"
    if not sidecar.is_file():
        raise ValueError(
            "This Calli checkpoint has no new-split provenance. Use the original "
            "COCO yolo26s-seg.pt; old Calli weights may have seen the new validation images."
        )
    source = json.loads(sidecar.read_text(encoding="utf-8"))
    availability_sha = hashlib.sha256(Path(data["annotation_availability"]).read_bytes()).hexdigest()
    if (source.get("schema") != "calli-partial-supervision-v1"
            or source.get("split_id") != data["split_id"]
            or source.get("availability_sha256") != availability_sha):
        raise ValueError("Checkpoint and dataset split/task protocols differ; start from COCO")


def create_partial_trainer(*args, staged_training=None, **kwargs):
    """Lazy factory keeps metadata checking usable without importing torch."""
    from partial_supervision import PartialSegmentationTrainer
    return PartialSegmentationTrainer(*args, staged_training=staged_training, **kwargs)


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


def validate_hyperparameters(args) -> None:
    semantic_weight = 1.0 if args.sem_weight is None else args.sem_weight
    checks = (
        ("lr0", args.lr0, 0 < args.lr0 <= 1, "0 < lr0 <= 1"),
        ("lrf", args.lrf, 0 <= args.lrf <= 1, "0 <= lrf <= 1"),
        ("weight-decay", args.weight_decay, 0 <= args.weight_decay <= 1, "0 <= weight-decay <= 1"),
        ("warmup-epochs", args.warmup_epochs, args.warmup_epochs >= 0, "warmup-epochs >= 0"),
        ("warmup-bias-lr", args.warmup_bias_lr, 0 <= args.warmup_bias_lr <= 1, "0 <= warmup-bias-lr <= 1"),
        ("momentum", args.momentum, 0 <= args.momentum < 1, "0 <= momentum < 1"),
        ("box-weight", args.box_weight, args.box_weight >= 0, "box-weight >= 0"),
        ("cls-weight", args.cls_weight, args.cls_weight >= 0, "cls-weight >= 0"),
        ("dfl-weight", args.dfl_weight, args.dfl_weight >= 0, "dfl-weight >= 0"),
        ("sem-weight", semantic_weight, semantic_weight >= 0, "sem-weight >= 0"),
        ("translate", args.translate, 0 <= args.translate <= 1, "0 <= translate <= 1"),
        ("scale", args.scale, 0 <= args.scale <= 1, "0 <= scale <= 1"),
        ("shear", args.shear, 0 <= args.shear < 90, "0 <= shear < 90 degrees"),
        ("degrees", args.degrees, 0 <= args.degrees <= 180, "0 <= degrees <= 180"),
        ("perspective", args.perspective, 0 <= args.perspective <= 1, "0 <= perspective <= 1"),
        ("hsv-h", args.hsv_h, 0 <= args.hsv_h <= 1, "0 <= hsv-h <= 1"),
        ("hsv-s", args.hsv_s, 0 <= args.hsv_s <= 1, "0 <= hsv-s <= 1"),
        ("hsv-v", args.hsv_v, 0 <= args.hsv_v <= 1, "0 <= hsv-v <= 1"),
        ("mosaic", args.mosaic, 0 <= args.mosaic <= 1, "0 <= mosaic <= 1"),
    )
    for name, value, in_range, description in checks:
        if not math.isfinite(value) or not in_range:
            raise ValueError(f"{name} must be finite and satisfy {description}")
    if args.save_period < -1:
        raise ValueError("save-period must be -1, 0, or a positive number of epochs")
    if args.freeze < 0 or args.mask_ratio < 1:
        raise ValueError("freeze must be nonnegative and mask-ratio must be positive")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path, default=ROOT / "dataset_full" / "data.yaml")
    result.add_argument("--model", default="yolo26s-seg.pt")
    result.add_argument("--epochs", type=int, default=60)
    result.add_argument("--batch", type=int, default=24)
    result.add_argument("--imgsz", type=int, default=640)
    result.add_argument("--patience", type=int, default=0, choices=(0,),
                        help="Early stopping is disabled; stop manually or reach the epoch limit")
    result.add_argument("--optimizer", default="auto",
                        choices=("auto", "SGD", "MuSGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp"),
                        help="auto selects optimizer and lr0 itself; choose AdamW/SGD to set lr0 manually")
    result.add_argument("--lr0", type=float, default=.01,
                        help="Initial learning rate; ignored by optimizer=auto")
    result.add_argument("--lrf", type=float, default=.01, help="Final learning-rate fraction")
    result.add_argument("--weight-decay", type=float, default=.0005)
    result.add_argument("--warmup-epochs", type=float, default=3.0)
    result.add_argument("--warmup-bias-lr", type=float, default=.1,
                        help="Initial bias learning rate during warmup; auto optimizer resets it to 0")
    result.add_argument("--momentum", type=float, default=.937,
                        help="SGD momentum/Adam beta1; optimizer=auto selects its own value")
    result.add_argument("--save-period", type=int, default=-1,
                        help="Periodic checkpoint interval; -1 or 0 disables periodic copies")
    result.add_argument("--cos-lr", action="store_true", help="Use cosine LR decay for a new run")
    result.add_argument("--no-amp", action="store_true", help="Disable mixed precision for a new run")
    result.add_argument("--box-weight", type=float, default=7.5,
                        help="Native box gain; YOLO26 also applies it to segmentation/semantic terms")
    result.add_argument("--cls-weight", type=float, default=.5)
    result.add_argument("--dfl-weight", type=float, default=1.5,
                        help="Native dfl gain; YOLO26 reg_max=1 uses it for L1 regression")
    result.add_argument("--sem-weight", type=float, default=None,
                        help="Both E2E branches' semantic multiplier; new-run default 1, resume inherits saved config")
    result.add_argument("--translate", type=float, default=.1)
    result.add_argument("--scale", type=float, default=.5)
    result.add_argument("--shear", type=float, default=0.0, help="Maximum absolute shear angle in degrees")
    result.add_argument("--degrees", type=float, default=0.0, help="Maximum absolute rotation angle")
    result.add_argument("--perspective", type=float, default=0.0, help="Perspective transform magnitude")
    result.add_argument("--bf-only-after", type=int, default=None, help="Joint epochs before BF-only training; new-run default 40, 0 disables")
    result.add_argument("--stage-freeze", type=int, default=None, help="Backbone modules frozen at switch; new-run default 4")
    result.add_argument("--freeze", type=int, default=0, help="Number of initial model layers to freeze")
    result.add_argument("--mask-ratio", type=int, default=4, help="Training-mask downsampling ratio")
    result.add_argument("--hsv-h", type=float, default=.015, help="Hue augmentation magnitude")
    result.add_argument("--hsv-s", type=float, default=.7, help="Saturation augmentation magnitude")
    result.add_argument("--hsv-v", type=float, default=.4, help="Brightness augmentation magnitude")
    result.add_argument("--mosaic", type=float, default=1.0, help="Same-annotation-pool Mosaic probability")
    result.add_argument("--device", default="0", help="AutoDL GPU: 0; CPU: cpu")
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
    result.add_argument("--preview-every", type=int, default=1, choices=(1,),
                        help="Write full validation metrics and fixed previews every epoch")
    result.add_argument("--preview-count", type=int, default=60, choices=(60,),
                        help="Exactly 60 fixed validation images in one TensorBoard image card")
    result.add_argument("--no-tensorboard", action="store_true",
                        help="Explicitly disable TensorBoard scalars/images (metrics.csv remains)")
    return result


def main() -> None:
    args = parser().parse_args()
    validate_hyperparameters(args)
    if args.epochs < 1 or args.batch < 1 or args.imgsz < 32 or args.patience < 0:
        raise ValueError("epochs/batch must be positive, imgsz >= 32, patience >= 0")
    if args.preview_every < 1 or args.preview_count < 0:
        raise ValueError("preview-every must be positive and preview-count must be nonnegative")
    if "," in args.device and "LOCAL_RANK" not in os.environ:
        raise ValueError("Use one GPU or launch this script with torchrun for multi-GPU callbacks")
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
    from calli_loss import resolve_semantic_weight
    args.sem_weight = resolve_semantic_weight(args.sem_weight, resume_checkpoint=args.resume)
    from staged_training import resolve_stages
    stage_config = resolve_stages(args.bf_only_after, args.stage_freeze, args.resume)
    model_path = Path(args.model).expanduser()
    if len(model_path.parts) == 1 and not model_path.is_file():
        model_path = ROOT / "models" / model_path
    model_path = model_path.resolve()
    settings = {
        "epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz,
        "patience": args.patience, "device": args.device, "workers": args.workers,
        "seed": args.seed, "project": str(args.project.expanduser().resolve()),
        "name": args.name, "overlap_mask": False, "fliplr": 0.0, "flipud": 0.0,
        "mosaic": args.mosaic, "mixup": 0.0, "copy_paste": 0.0, "degrees": args.degrees,
        "translate": args.translate, "scale": args.scale, "shear": args.shear, "perspective": args.perspective,
        "hsv_h": args.hsv_h, "hsv_s": args.hsv_s, "hsv_v": args.hsv_v,
        "optimizer": args.optimizer, "lr0": args.lr0, "lrf": args.lrf,
        "weight_decay": args.weight_decay, "warmup_epochs": args.warmup_epochs,
        "warmup_bias_lr": args.warmup_bias_lr, "momentum": args.momentum,
        "cos_lr": args.cos_lr, "amp": not args.no_amp,
        "box": args.box_weight, "cls": args.cls_weight, "dfl": args.dfl_weight,
        "freeze": args.freeze, "mask_ratio": args.mask_ratio,
        "save": True, "save_period": args.save_period, "plots": False,
    }
    plan = {
        "model": str(model_path), "resume": str(args.resume) if args.resume else None,
        "resume_output": str(resume_output) if resume_output else None,
        "resolved_data": data, "dataset_check": report, "train_settings": settings,
        "auxiliary_loss": {"sem_weight": args.sem_weight},
        "staged_training": stage_config,
        "visualization": {"tensorboard": not args.no_tensorboard,
                          "tensorboard_root": str((resume_output.parent if resume_output else Path(settings["project"]))
                                                  / "tensorboard"),
                          "preview_every": args.preview_every, "preview_count": args.preview_count,
                          "region_metrics_scope": "full validation",
                          "region_metrics_schedule": "every epoch",
                          "region_conf_threshold": .25, "region_iou_threshold": .5},
        "note": "Resume restores the checkpoint optimizer and training schedule; "
                "the printed new-run hyperparameters are not applied to resume."
        if args.resume else "COCO initialization is not a trained Calli recognizer.",
        "optimizer_note": (
            "Ultralytics optimizer=auto selects the optimizer, initial learning rate, and momentum, "
            "ignoring the supplied lr0/momentum. Choose AdamW or SGD for manual settings."
        ) if args.optimizer == "auto" else "Explicit optimizer and lr0 apply to a new run only.",
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    import yaml
    from functools import partial
    trainer_factory = partial(create_partial_trainer, staged_training=stage_config)
    from ultralytics import YOLO
    from live_visualization import LiveVisualization, isolated_logging_integrations
    from calli_loss import make_semantic_loss_callback

    live = LiveVisualization(data, split_images(data["val"]), args.preview_every,
                             args.preview_count, args.seed, not args.no_tensorboard)

    if not args.resume:
        model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = str(args.resume.expanduser().resolve()) if args.resume else str(model_path)
    model = YOLO(checkpoint)
    if args.resume:
        validate_resume_checkpoint(model.ckpt)
    if model.task != "segment":
        raise ValueError("This dataset requires an instance-segmentation (-seg) model")
    if data.get("split_id"):
        validate_initialization_split(model.names, Path(checkpoint), data)
    model.add_callback("on_train_start", make_semantic_loss_callback(args.sem_weight))
    live.attach(model)
    config_dir = Path(settings["project"]) / "_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    runtime_yaml = config_dir / f"dataset_{stamp}.yaml"
    runtime_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    try:
        with isolated_logging_integrations():
            if args.resume:
                model.train(
                    trainer=trainer_factory,
                    resume=True, data=str(runtime_yaml), device=args.device, workers=args.workers, patience=0,
                    save_dir=str(resume_output),
                )
            else:
                model.train(trainer=trainer_factory, data=str(runtime_yaml), **settings)
    finally:
        live.close()
    print(f"Training finished. Actual output directory: {model.trainer.save_dir}")


if __name__ == "__main__":
    main()
