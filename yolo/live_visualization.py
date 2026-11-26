"""Epoch scalars, periodic full-val region counts, and fixed BF/JT previews.

Preview inference loads the current last.pt into a separate YOLO object. The
training model, EMA, optimizer, and their mode/dtype are never passed to predict.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path


COMMON_LOSS_NAMES = ("box_loss", "seg_loss", "cls_loss", "l1_loss")
DASHBOARD_SCALAR_TAGS = {
    "loss/train_combined", "loss/val_combined",
    *{f"metrics/{group}/{metric}_pct" for group in ("BF", "JT")
      for metric in ("recall", "precision", "iou")},
}


def dashboard_directory(run_directory: Path) -> Path:
    run_directory = Path(run_directory)
    return run_directory.parent / "dashboard" / run_directory.name


def common_loss_totals(train_losses: dict, val_metrics: dict) -> dict:
    """Sum comparable, gain-weighted one-to-one task terms, not E2E objectives.

    YOLO26's displayed loss_items describe its one-to-one branch. Semantic
    loss is train-only and is excluded from both displayed totals. Missing or
    nonfinite components yield None; never display a partial sum as a total.
    """
    output = {}
    for prefix, values in (("train", train_losses), ("val", val_metrics)):
        keys = [f"{prefix}/{name}" for name in COMMON_LOSS_NAMES]
        complete = all(key in values and values[key] is not None for key in keys)
        numbers = [float(values[key]) for key in keys] if complete else []
        output[f"loss/{prefix}_combined"] = (
            sum(numbers) if complete and all(math.isfinite(value) for value in numbers) else None
        )
    return output


def primary_rank() -> bool:
    return int(os.environ.get("RANK", "-1")) in {-1, 0}


def should_preview(epoch: int, total: int, every: int, first: int = 1,
                   stopping: bool = False) -> bool:
    """Epoch arguments are one-based; first includes the first resumed epoch."""
    return epoch == first or epoch % every == 0 or epoch >= total or stopping


def checkpoint_matches_epoch(checkpoint: object, epoch: int) -> None:
    if not isinstance(checkpoint, dict) or checkpoint.get("epoch") != epoch:
        found = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
        raise RuntimeError(
            f"Preview requires the current last.pt epoch={epoch}; found {found}. "
            "Checkpoint may be stale or stripped; preview was not mislabeled."
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label_for(image: Path) -> Path:
    parts = list(image.parts)
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def validation_authors(data_root: Path, images: list[Path]) -> dict[Path, str]:
    """Prefer the dataset's source author; filenames are a portable fallback."""
    authors = {}
    source = data_root / "manifest.json"
    if source.is_file():
        for row in json.loads(source.read_text(encoding="utf-8")).get("images", []):
            if row.get("image") and row.get("author"):
                authors[(data_root / row["image"]).resolve()] = str(row["author"])
    return {image: authors.get(image, image.stem.rsplit("_", 1)[0]) for image in images}


def fixed_samples(data_root: Path, candidates: list[Path], count: int,
                  seed: int, path: Path) -> dict:
    """Keep sample identity and image/label bytes fixed across epochs/resumes."""
    data_root = Path(data_root).resolve()
    available = sorted({Path(item).resolve() for item in candidates})
    count = min(count, len(available))
    if path.is_file():
        # Existing runs, including original 4-image v1 manifests, keep their
        # exact samples and manifest bytes when the new default becomes 45.
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        authors = validation_authors(data_root, available)
        groups = defaultdict(list)
        for image in available:
            groups[authors[image]].append(image)
        rng = random.Random(seed)
        order = sorted(groups)
        rng.shuffle(order)
        for group in groups.values():
            rng.shuffle(group)
        chosen = []
        while len(chosen) < count:
            for author in order:
                if groups[author] and len(chosen) < count:
                    chosen.append(groups[author].pop())
        chosen.sort()
        manifest = {
            "schema": "calli-fixed-validation-preview-v2", "count": count,
            "sampling_strategy": "author_stratified_round_robin",
            "author_source": "dataset manifest author; filename prefix fallback",
            "target_per_author": count // max(len(order), 1),
            "available_per_author": dict(sorted(Counter(authors.values()).items())),
            "selected_per_author": dict(sorted(Counter(authors[image] for image in chosen).items())),
            "seed": seed, "samples": [
                {"image": os.path.relpath(image, data_root),
                 "label": os.path.relpath(label_for(image), data_root),
                 "author": authors[image],
                 "image_sha256": sha256(image), "label_sha256": sha256(label_for(image))}
                for image in chosen
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    allowed = set(available)
    for row in manifest["samples"]:
        image, label = data_root / row["image"], data_root / row["label"]
        if image.resolve() not in allowed:
            raise ValueError(f"Fixed preview sample is no longer in validation: {image}")
        if sha256(image) != row["image_sha256"] or sha256(label) != row["label_sha256"]:
            raise ValueError(f"Fixed preview image/label changed: {image}")
    return manifest


@contextmanager
def preserve_rng():
    """Inference may consume RNG during model setup; restore every generator."""
    import numpy as np
    import torch

    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@contextmanager
def isolated_logging_integrations():
    """Disable native graph tracing/W&B in memory, without saving user settings."""
    from ultralytics.utils import SETTINGS

    previous = {key: SETTINGS[key] for key in ("tensorboard", "wandb")}
    try:
        for key in previous:
            dict.__setitem__(SETTINGS, key, False)
        yield
    finally:
        for key, value in previous.items():
            dict.__setitem__(SETTINGS, key, value)


def summary_writer_class():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except (ImportError, RuntimeError, AttributeError, TypeError) as error:
        raise RuntimeError(
            "TensorBoard logging is enabled but its dependency could not load. "
            "Install yolo/requirements.txt (including tensorboard), or explicitly "
            "pass --no-tensorboard to disable it."
        ) from error
    return SummaryWriter


def gt_instances(label: Path, height: int, width: int) -> list[dict]:
    import cv2
    import numpy as np

    output = []
    for line in label.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        # Match Ultralytics and build_dataset.serialized_label_iou exactly:
        # float32 normalized coordinates -> scale -> int32 truncation -> fillPoly.
        # Labels use (pixel + 0.5)/size; rounding would shift odd boundaries +1.
        coords = np.asarray(fields[1:], dtype=np.float32).reshape(-1, 2)
        coords = (coords * np.asarray([width, height], dtype=np.float32)).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [coords], 1)
        output.append({"class_id": int(fields[0]), "mask": mask.astype(bool)})
    return output


def prediction_instances(result) -> list[dict]:
    if result.boxes is None or not len(result.boxes):
        return []
    if result.masks is None or len(result.masks) != len(result.boxes):
        raise RuntimeError("Preview detections and segmentation masks are not aligned")
    masks = result.masks.data.detach().cpu().numpy() > 0.5
    if tuple(masks.shape[1:]) != tuple(result.orig_shape):
        raise RuntimeError("Preview retina masks must have original-image dimensions")
    classes = result.boxes.cls.detach().cpu().tolist()
    confidence = result.boxes.conf.detach().cpu().tolist()
    return [{"class_id": int(cls), "confidence": float(conf), "mask": mask}
            for cls, conf, mask in zip(classes, confidence, masks)]


def panel_image(image, truth: list[dict], predicted: list[dict], names: dict,
                epoch: int, sample: str, checkpoint_label: str = "current last.pt EMA",
                confidence_threshold: float = .25):
    """One source and four grouped overlays, all from the same prediction."""
    import colorsys
    import numpy as np
    from PIL import Image, ImageDraw

    image = image.convert("RGB")
    width, height = image.size
    header, footer = 62, 26
    canvas = Image.new("RGB", (width * 5, height + header + footer), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), f"Epoch {epoch:03d} | {sample} | {checkpoint_label}", fill="black")
    columns = [("INPUT", []), ("BF GT [0:50]", [v for v in truth if v["class_id"] < 50]),
               ("BF PRED [0:50]", [v for v in predicted if v["class_id"] < 50]),
               ("JT GT [50:70]", [v for v in truth if 50 <= v["class_id"] < 70]),
               ("JT PRED [50:70]", [v for v in predicted if 50 <= v["class_id"] < 70])]
    for column, (title, instances) in enumerate(columns):
        pixels = np.array(image, dtype=np.float32)
        label_positions = []
        for index, item in enumerate(instances):
            mask = item["mask"]
            if mask.shape != (height, width):
                raise ValueError("Preview mask and source image dimensions differ")
            if not mask.any():
                continue
            cls = item["class_id"]
            color = np.array(colorsys.hsv_to_rgb((cls * .618 + index * .17) % 1, .75, .95)) * 255
            pixels[mask] = .55 * pixels[mask] + .45 * color
            y, x = np.nonzero(mask)
            label = str(names.get(cls, cls)).replace("phase_", "").replace("component_", "")
            label = label.replace("beginning", "B").replace("middle", "M").replace("ending", "E")
            label = label.encode("ascii", "replace").decode("ascii")
            if "confidence" in item:
                label += f" {item['confidence']:.2f}"
            label_positions.append((int(x.min()), int(y.min()), label))
        panel = Image.fromarray(pixels.clip(0, 255).astype(np.uint8))
        annotate = ImageDraw.Draw(panel)
        for x, y, label in label_positions:
            box = annotate.textbbox((x, y), label)
            annotate.rectangle(box, fill="white")
            annotate.text((x, y), label, fill="black")
        canvas.paste(panel, (column * width, header))
        draw.text((column * width + 8, 30), f"{title} | n={len(instances)}", fill="black")
    draw.text((8, height + header + 7),
              f"BF: stroke phases; JT: components; B/M/E: beginning/middle/ending; confidence >= {confidence_threshold:g}",
              fill="black")
    return canvas


class LiveVisualization:
    def __init__(self, data: dict, val_images: list[Path], every: int = 5,
                 count: int = 45, seed: int = 42, tensorboard: bool = True):
        self.data, self.val_images = data, val_images
        self.every, self.count, self.seed = every, count, seed
        self.writer_type = summary_writer_class() if tensorboard and primary_rank() else None
        self.writer = None
        self.manifest = None
        self.active_epoch = None
        self.last_fit_epoch = None
        if count and len(data["names"]) != 70:
            raise ValueError("BF/JT previews require 50 phase + 20 component classes")

    def attach(self, model) -> None:
        model.add_callback("on_pretrain_routine_end", self.on_start)
        model.add_callback("on_train_epoch_start", self.on_epoch_start)
        model.add_callback("on_fit_epoch_end", self.on_fit_end)
        model.add_callback("teardown", self.close)

    def on_start(self, trainer) -> None:
        if not primary_rank():
            return
        if self.count:
            self.manifest = fixed_samples(
                Path(self.data["path"]), self.val_images, self.count, self.seed,
                Path(trainer.save_dir) / "previews" / "fixed_samples.json",
            )
            if self.manifest["count"] != min(self.count, len(self.val_images)):
                print(f"Keeping existing fixed preview manifest: {self.manifest['count']} images "
                      f"(new-run request: {self.count})", flush=True)
        if self.writer_type:
            event_dir = dashboard_directory(Path(trainer.save_dir))
            self.writer = self.writer_type(str(event_dir))
            protocol = {
                "schema": "calli-minimal-dashboard-v1", "event_directory": str(event_dir),
                "scalar_tags": sorted(DASHBOARD_SCALAR_TAGS),
                "common_loss_components": list(COMMON_LOSS_NAMES),
                "loss_definition": (
                    "train_combined and val_combined sum the four common gain-weighted one-to-one "
                    "branch loss items reported by Ultralytics. They exclude train-only sem_loss "
                    "and are not the complete E2E backpropagation objective. Missing components "
                    "are omitted, never replaced by a partial sum."
                ),
                "details": "Native loss components, mAP, and LR remain in results.csv; region counts remain in JSON.",
                "history": "Existing event files under the old run/tensorboard directory are unchanged.",
            }
            (Path(trainer.save_dir) / "dashboard_protocol.json").write_text(
                json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"TensorBoard minimal dashboard enabled: {event_dir}", flush=True)

    def on_epoch_start(self, trainer) -> None:
        self.active_epoch = trainer.epoch

    def _scalars(self, values: dict, step: int, prefix: str = "") -> None:
        if self.writer:
            for key, value in values.items():
                tag = prefix + key
                if value is not None and tag in DASHBOARD_SCALAR_TAGS:
                    self.writer.add_scalar(tag, float(value), step)

    def on_fit_end(self, trainer) -> None:
        if not primary_rank():
            return
        if trainer.epoch != self.active_epoch:
            # Ultralytics final_eval invokes this hook for best.pt with epoch+1.
            # Its losses describe a different checkpoint, so do not mix them
            # into the current-epoch train/val totals or add native mAP tags.
            if self.writer:
                self.writer.flush()
            return
        if self.last_fit_epoch == trainer.epoch:
            return
        step = trainer.epoch + 1
        totals = common_loss_totals(trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.metrics)
        self._scalars(totals, step)
        if should_preview(
            step, trainer.epochs, self.every, trainer.start_epoch + 1, bool(trainer.stop)
        ):
            self._preview(trainer, step)
        self.last_fit_epoch = trainer.epoch
        if self.writer:
            self.writer.flush()

    def _preview(self, trainer, step: int) -> None:
        import numpy as np
        from PIL import Image
        from ultralytics import YOLO
        from region_metrics import (
            RegionMetricsAccumulator, deduplicate_gt_instances, tensorboard_scalars,
        )

        checkpoint = Path(trainer.last)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Current epoch preview checkpoint missing: {checkpoint}")
        data_root = Path(self.data["path"]).resolve()
        if self.count:
            self.manifest = fixed_samples(
                data_root, self.val_images, self.count, self.seed,
                Path(trainer.save_dir) / "previews" / "fixed_samples.json",
            )
        rows = self.manifest["samples"] if self.count else []
        preview_lookup = {(data_root / row["image"]).resolve(): (index, row)
                          for index, row in enumerate(rows)}
        sources = sorted({Path(source).resolve() for source in self.val_images})
        authors = validation_authors(data_root, sources)
        counts = RegionMetricsAccumulator(conf_threshold=.25, iou_threshold=.5, deduplicate_gt=True)
        with preserve_rng():
            predictor = YOLO(str(checkpoint))
            checkpoint_matches_epoch(predictor.ckpt, trainer.epoch)
            folder = Path(trainer.save_dir) / "previews" / f"epoch_{step:03d}"
            folder.mkdir(parents=True, exist_ok=True)
            provenance = {
                "epoch": step, "checkpoint_epoch_zero_based": trainer.epoch,
                "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                "checkpoint_note": "last.pt is overwritten later; SHA identifies the weights used here",
                "source": "current epoch last.pt EMA", "conf_threshold": .25, "iou_threshold": .5,
                "evaluated_split": "val", "full_validation": True,
                "evaluated_images": len(sources), "preview_images": len(rows),
                "evaluation_interval_epochs": self.every,
                "evaluation_schedule": "first, interval multiples, final/early-stop epochs only",
                "ground_truth_policy": "deduplicate equal class_id and pixel-identical binary mask",
                "native_duplicate_note": (
                    "Native Ultralytics removes duplicate label rows during loading. This report "
                    "independently deduplicates class+binary mask; raw/removed counts are measured, "
                    "not assumed equal to the native loader warning."
                ),
            }
            metadata = {**provenance, "samples": []}
            # Evaluate each validation image once, then reuse the same result for
            # its fixed preview if selected. Never retain all 171 GPU mask sets.
            for source in sources:
                results = predictor.predict(
                    source=str(source), device=trainer.device, imgsz=trainer.args.imgsz,
                    conf=.25, max_det=300, retina_masks=True, batch=1,
                    save=False, verbose=False, quantize=32,
                )
                if len(results) != 1:
                    raise RuntimeError("A single validation path must produce exactly one result")
                result = results[0]
                if Path(result.path).resolve() != source:
                    raise RuntimeError("Region-count prediction returned the wrong source image")
                with Image.open(source) as image:
                    image = image.convert("RGB")
                truth = gt_instances(label_for(source), image.height, image.width)
                predicted = prediction_instances(result)
                relative = os.path.relpath(source, data_root)
                counts.update(truth, predicted, image_id=relative)
                if source in preview_lookup:
                    index, row = preview_lookup[source]
                    unique_truth, removed = deduplicate_gt_instances(truth)
                    preview = panel_image(image, unique_truth, predicted, self.data["names"], step, source.stem)
                    filename = f"{index:02d}_{source.stem}.png"
                    preview.save(folder / filename)
                    if self.writer:
                        self.writer.add_image(f"previews/{index:02d}_{source.stem}",
                                              np.asarray(preview), step, dataformats="HWC")
                    metadata["samples"].append({
                        "image": row["image"], "author": row.get("author", authors[source]),
                        "preview": filename, "gt_instances_raw": len(truth),
                        "gt_instances": len(unique_truth), "gt_duplicates_removed": removed,
                        "pred_instances": len(predicted),
                    })
                del results, result, truth, predicted
            summary = counts.summary()
            metrics_dir = Path(trainer.save_dir) / "region_metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = metrics_dir / f"epoch_{step:03d}.json"
            metrics_path.write_text(json.dumps(
                {**provenance, "summary": summary, "per_image": counts.per_image},
                ensure_ascii=False, indent=2, allow_nan=False,
            ), encoding="utf-8")
            self._scalars(tensorboard_scalars(summary), step)
            metadata["region_metrics"] = str(metrics_path)
            metadata["samples"].sort(key=lambda row: row["preview"])
            (folder / "preview.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            del predictor
        print(f"Full-val region counts ({len(sources)} images): {metrics_path}; "
              f"fixed previews ({len(rows)} images): {folder}", flush=True)

    def close(self, trainer=None) -> None:
        if self.writer:
            self.writer.flush()
            self.writer.close()
            self.writer = None
