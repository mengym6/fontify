"""Evaluate BF/JT geometric recall, label-aware precision, and mean mask IoU.

This is post-training inference only. Native mAP histories and checkpoint
selection are unchanged. Recall ignores fine-class errors within BF/JT;
precision requires the fine label to be correct and counts every prediction
in its denominator. They intentionally do not use one common TP definition.
IoU is averaged over geometrically matched masks, not bounding boxes.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def write_gallery(output: Path, images: list[dict], checkpoint: str) -> None:
    groups = defaultdict(list)
    for item in images:
        if item.get("preview"):
            groups[item["author"]].append(item)
    sections = []
    for author, items in sorted(groups.items()):
        cards = []
        for item in items:
            image = html.escape(item["preview"], quote=True)
            label = html.escape(item["image"])
            cards.append(f'<figure><figcaption>{label}</figcaption>'
                         f'<a href="{image}"><img loading="lazy" src="{image}"></a></figure>')
        sections.append(f'<h2>{html.escape(author)} ({len(items)})</h2>' + "".join(cards))
    document = ('<!doctype html><html lang="zh"><meta charset="utf-8">'
                '<title>笔法／结体完整预览</title><style>'
                'body{font:16px sans-serif;margin:28px;background:#f3f5f8;color:#243645}'
                'figure{margin:12px 0;padding:12px;background:white;border-radius:8px}'
                'img{width:100%;height:auto}figcaption{margin-bottom:8px}'
                '</style><h1>笔法阶段与结体实例预览</h1>'
                '<p>依次为原图、笔法真值、笔法预测、结体真值、结体预测。点击图片查看原尺寸。</p>'
                f'<p>权重：{html.escape(checkpoint)}</p>' + "".join(sections) + '</html>')
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "dataset/data.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=.25)
    parser.add_argument("--match-iou", type=float, default=.5)
    parser.add_argument("--checkpoint-epoch", type=int, default=0,
                        help="One-based epoch label from training records; 0 means unknown")
    parser.add_argument("--save-previews", action="store_true",
                        help="Save every image, grouped by source; does not sample only four")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.conf <= 1 or not 0 < args.match_iou <= 1:
        raise ValueError("conf must be in [0,1] and match-iou in (0,1]")
    if args.checkpoint_epoch < 0:
        raise ValueError("checkpoint-epoch must be nonnegative")
    weights, output = args.weights.expanduser().resolve(), args.output.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new evaluation directory: {output}")
    tensorboard_dir = output.parent / "dashboard" / output.name
    if (not args.no_tensorboard and tensorboard_dir.exists()
            and any(tensorboard_dir.iterdir())):
        raise FileExistsError(f"Use a new evaluation name; dashboard already exists: {tensorboard_dir}")

    import numpy as np
    from PIL import Image
    from ultralytics import YOLO
    from train import read_dataset, split_images
    from live_visualization import (
        gt_instances, label_for, panel_image, prediction_instances, sha256,
        summary_writer_class,
    )
    from region_metrics import (
        aggregate_region_metrics, deduplicate_gt_instances,
        evaluate_regions, tensorboard_scalars,
    )

    data = read_dataset(args.data)
    if args.split not in data:
        raise ValueError(f"No {args.split} split in data YAML")
    images = split_images(data[args.split])
    model = YOLO(str(weights))
    if model.task != "segment" or dict(model.names) != data["names"] or len(model.names) != 70:
        raise ValueError("Checkpoint must match the dataset's 50 phase + 20 component classes")
    output.mkdir(parents=True)
    writer = None if args.no_tensorboard else summary_writer_class()(str(tensorboard_dir))
    records, author_reports = [], defaultdict(list)
    checkpoint_sha = sha256(weights)
    protocol = {
        "unit": "instance_count", "confidence_threshold": args.conf,
        "mask_iou_threshold": args.match_iou,
        "matching": "within BF/JT; ignore fine class; maximum cardinality then maximum summed IoU",
        "recall": "geometrically matched instances / GT instances",
        "precision": "fine-label-correct geometric matches / all predictions after confidence filtering",
        "precision_json_field": "label_aware_precision",
        "iou": "sum of matched mask IoUs / geometric matches; not bbox IoU",
        "unmatched_gt": "reduces recall; excluded from the matched-IoU mean",
        "fine_class_error": "within BF/JT, penalizes precision but not geometric recall or matched IoU",
        "cross_group_error": "miss in the GT task and extra prediction in the other task",
        "metric_convention": "custom separate recall/precision numerators, not a standard shared-TP precision/recall pair",
        "legacy_json_fields": {
            "region_precision": "geometric matches / predictions; retained for compatibility, not displayed precision",
            "label_accuracy": "correct fine labels / geometric matches; retained only in JSON",
            "miss_rate": "missed GT instances / GT instances; retained only in JSON",
        },
        "aggregation": "micro: sum counts before calculating rates",
        "gt_deduplication": "same class and identical original-resolution binary mask",
        "zero_denominator": "null in JSON; omitted from TensorBoard",
        "tensorboard_main_rates": "metrics/BF and metrics/JT: recall_pct, precision_pct, iou_pct only",
        "tensorboard_directory": str(tensorboard_dir) if writer else None,
    }
    try:
        for index, source in enumerate(images):
            result_list = model.predict(
                source=str(source), device=args.device, imgsz=args.imgsz,
                conf=args.conf, max_det=300, retina_masks=True, batch=1,
                save=False, verbose=False, quantize=32,
            )
            if len(result_list) != 1 or Path(result_list[0].path).resolve() != source.resolve():
                raise RuntimeError("Inference source/result mismatch")
            with Image.open(source) as image:
                image = image.convert("RGB")
            truth = gt_instances(label_for(source), image.height, image.width)
            predictions = prediction_instances(result_list[0])
            counts = evaluate_regions(truth, predictions, conf_threshold=args.conf,
                                      iou_threshold=args.match_iou, deduplicate_gt=True)
            author = source.stem.split("_", 1)[0]
            row = {"image": str(source.relative_to(data["path"])), "author": author,
                   "image_sha256": sha256(source), "label_sha256": sha256(label_for(source)),
                   "metrics": counts}
            if args.save_previews:
                unique_truth, _ = deduplicate_gt_instances(truth)
                panel = panel_image(image, unique_truth, predictions, data["names"],
                                    args.checkpoint_epoch, source.stem,
                                    checkpoint_label=f"{weights.name} | post-training evaluation",
                                    confidence_threshold=args.conf)
                relative = Path("previews") / author / (source.stem + ".png")
                (output / relative).parent.mkdir(parents=True, exist_ok=True)
                panel.save(output / relative)
                row["preview"] = str(relative)
                if writer:
                    writer.add_image(f"previews/{author}/{source.stem}", np.asarray(panel),
                                     args.checkpoint_epoch, dataformats="HWC")
            records.append(row)
            author_reports[author].append(counts)
            del result_list, truth, predictions, image
            if (index + 1) % 25 == 0 or index + 1 == len(images):
                print(f"Evaluated {index + 1}/{len(images)} {args.split} images", flush=True)

        summary = {
            "schema": "calli-region-count-evaluation-v2", "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(weights), "checkpoint_sha256": checkpoint_sha,
            "checkpoint_epoch_label": args.checkpoint_epoch,
            "checkpoint_epoch_label_source": "caller supplied from training records; not inferred from stripped checkpoint",
            "source_files_sha256": {name: sha256(ROOT / name) for name in
                                    ("evaluate_regions.py", "region_metrics.py", "live_visualization.py")},
            "data_yaml_sha256": sha256(args.data.expanduser().resolve()),
            "split": args.split, "images": len(images), "protocol": protocol,
            "groups": aggregate_region_metrics([row["metrics"] for row in records]),
            "by_author": {author: {"images": len(reports), "groups": aggregate_region_metrics(reports)}
                          for author, reports in sorted(author_reports.items())},
        }
        if sha256(weights) != checkpoint_sha:
            raise RuntimeError("Checkpoint changed during evaluation")
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        (output / "per_image.json").write_text(json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        if writer:
            for key, value in tensorboard_scalars(summary["groups"]).items():
                writer.add_scalar(key, value, args.checkpoint_epoch)
            writer.add_text("protocol/definition", json.dumps(protocol, ensure_ascii=False, indent=2), args.checkpoint_epoch)
            writer.flush()
        if args.save_previews:
            write_gallery(output, records, f"{weights.name} | epoch {args.checkpoint_epoch} | {checkpoint_sha}")
        print(json.dumps({"output": str(output), "images": len(images),
                          "metrics_percent": tensorboard_scalars(summary["groups"])},
                         ensure_ascii=False, indent=2))
    finally:
        if writer:
            writer.close()


if __name__ == "__main__":
    main()
