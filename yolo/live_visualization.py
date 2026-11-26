"""Grouped losses, full-val metrics, and one 60-sample image card per epoch.

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
    *{f"{group}/loss/{split}_combined" for group in ("BF", "JT") for split in ("train", "val")},
    *{f"{group}/metrics/{metric}_pct" for group in ("BF", "JT")
      for metric in ("recall", "precision", "iou")},
}
PREVIEW_IMAGE_TAG_TEMPLATE = "previews/epoch_{epoch:03d}"


def preview_image_tag(epoch: int) -> str:
    return PREVIEW_IMAGE_TAG_TEMPLATE.format(epoch=epoch)


def dashboard_directory(run_directory: Path) -> Path:
    run_directory = Path(run_directory)
    return run_directory.parent / "dashboard_full" / run_directory.name


def group_loss_totals(train_groups: dict, val_groups: dict) -> dict:
    """Display actual criterion sidechannels, never split an overall loss."""
    totals = {}
    for prefix, values in (("train", train_groups), ("val", val_groups)):
        for task in ("BF", "JT"):
            value = values.get(task)
            totals[f"{task}/loss/{prefix}_combined"] = (
                float(value) if value is not None and math.isfinite(float(value)) else None
            )
    return totals


def _loss_model(trainer, validation=False):
    return (getattr(getattr(trainer, "ema", None), "ema", None) if validation
            else getattr(trainer, "model", None))


def _reset_group_losses(model):
    if model is not None:
        from partial_supervision import reset_group_loss_accumulators
        reset_group_loss_accumulators(model)


def _read_group_losses(model):
    if model is None:
        return {"BF": None, "JT": None}
    from partial_supervision import read_group_loss_accumulators
    return read_group_loss_accumulators(model)


def image_tensor_summary(tag: str, png_paths: list[Path], *, epoch: int | None = None):
    """Make one v2 image tensor: [width, height, png0, ...], not a mosaic.

    TensorBoard Images exposes len(string_val)-2 samples in this tag's card.
    Each epoch has its own tag; sample n selects string_val[n+2].
    """
    from PIL import Image
    from tensorboard.compat.proto import summary_pb2, tensor_pb2, tensor_shape_pb2, types_pb2
    from tensorboard.plugins.image.metadata import create_summary_metadata

    if not png_paths:
        raise ValueError("An image card requires at least one PNG")
    encoded, dimensions = [], None
    for path in png_paths:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError(f"Image card requires PNG input: {path}")
            if dimensions is not None and image.size != dimensions:
                raise ValueError("All image-card samples must have the same dimensions")
            dimensions = image.size
        encoded.append(Path(path).read_bytes())
    width, height = dimensions
    values = [str(width).encode("ascii"), str(height).encode("ascii"), *encoded]
    tensor = tensor_pb2.TensorProto(
        dtype=types_pb2.DT_STRING,
        tensor_shape=tensor_shape_pb2.TensorShapeProto(
            dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=len(values))]),
        string_val=values,
    )
    metadata = create_summary_metadata(
        display_name=f"Epoch {epoch:03d} · fixed validation samples" if epoch is not None else tag,
        description=(f"Epoch {epoch:03d}. " if epoch is not None else "") +
                    f"Select sample 0..{len(png_paths) - 1} from the fixed sample manifest. "
                    "Each sample shows input, BF GT/prediction, JT GT/prediction. "
                    "Other epochs appear in separate cards with the same sample ordering.",
    )
    metadata.data_class = summary_pb2.DATA_CLASS_BLOB_SEQUENCE
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=tag, tensor=tensor, metadata=metadata)])


def write_image_card(writer, png_paths: list[Path], step: int) -> None:
    summary = image_tensor_summary(preview_image_tag(step), png_paths, epoch=step)
    writer.file_writer.add_summary(summary, global_step=step)


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


def annotation_tasks(data: dict, images: list[Path]) -> dict[Path, dict[str, bool]]:
    """Read explicit per-image annotation availability; old dual sets default known."""
    root = Path(data["path"]).resolve()
    configured = data.get("annotation_availability")
    path = Path(configured) if configured else root / "annotation_availability.json"
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        if configured:
            raise FileNotFoundError(f"Annotation availability is missing: {path}")
        return {Path(image).resolve(): {"BF": True, "JT": True} for image in images}
    rows = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for image in images:
        image = Path(image).resolve()
        relative = image.relative_to(root).as_posix()
        row = rows.get(relative)
        if row is None or any(not isinstance(row.get(key), bool) for key in ("has_bf", "has_jt")):
            raise ValueError(f"Missing or invalid task availability: {relative}")
        result[image] = {"BF": row["has_bf"], "JT": row["has_jt"]}
    return result


def fixed_samples(data_root: Path, candidates: list[Path], count: int,
                  seed: int, path: Path, *, split_id: str | None = None) -> dict:
    """Keep sample identity and image/label bytes fixed across epochs/resumes."""
    data_root = Path(data_root).resolve()
    available = sorted({Path(item).resolve() for item in candidates})
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if split_id is not None and manifest.get("split_id") != split_id:
            raise ValueError("Preview manifest belongs to another dataset split; start a new run")
        if split_id is not None and (manifest.get("count") != count or manifest.get("seed") != seed):
            raise ValueError("Fixed preview count/seed changed for this dataset split")
    else:
        if len(available) < count:
            raise ValueError(f"Need {count} fixed validation images, but only {len(available)} are available")
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
            "schema": "calli-fixed-validation-preview-v3", "count": count, "split_id": split_id,
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
                confidence_threshold: float = .25, known_tasks=None):
    """One source and four grouped overlays, all from the same prediction."""
    import colorsys
    import numpy as np
    from PIL import Image, ImageDraw
    from region_metrics import normalize_known_tasks
    tasks = normalize_known_tasks(known_tasks)

    image = image.convert("RGB")
    width, height = image.size
    header, footer = 62, 26
    canvas = Image.new("RGB", (width * 5, height + header + footer), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), f"Epoch {epoch:03d} | {sample} | {checkpoint_label}", fill="black")
    columns = [("INPUT", []),
               ("BF GT [0:50]" if "BF" in tasks else "BF GT UNANNOTATED",
                [v for v in truth if v["class_id"] < 50] if "BF" in tasks else []),
               ("BF PRED [0:50]", [v for v in predicted if v["class_id"] < 50]),
               ("JT GT [50:70]" if "JT" in tasks else "JT GT UNANNOTATED",
                [v for v in truth if 50 <= v["class_id"] < 70] if "JT" in tasks else []),
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
        heading = title if "UNANNOTATED" in title else f"{title} | n={len(instances)}"
        draw.text((column * width + 8, 30), heading, fill="black")
    draw.text((8, height + header + 7),
              f"BF: stroke phases; JT: components; B/M/E: beginning/middle/ending; confidence >= {confidence_threshold:g}",
              fill="black")
    return canvas


class LiveVisualization:
    def __init__(self, data: dict, val_images: list[Path], every: int = 1,
                 count: int = 60, seed: int = 42, tensorboard: bool = True):
        self.data, self.val_images = data, val_images
        self.every, self.count, self.seed = every, count, seed
        self.writer_type = summary_writer_class() if tensorboard and primary_rank() else None
        self.writer = None
        self.manifest = None
        self.active_epoch = None
        self.last_fit_epoch = None
        self.trainer = None
        if count and len(data["names"]) != 70:
            raise ValueError("BF/JT previews require 50 phase + 20 component classes")

    def attach(self, model) -> None:
        model.add_callback("on_pretrain_routine_end", self.on_start)
        model.add_callback("on_train_epoch_start", self.on_epoch_start)
        model.add_callback("on_val_start", self.on_val_start)
        model.add_callback("on_fit_epoch_end", self.on_fit_end)
        model.add_callback("teardown", self.close)

    def on_start(self, trainer) -> None:
        self.trainer = trainer
        if not primary_rank():
            return
        if self.count:
            self.manifest = fixed_samples(
                Path(self.data["path"]), self.val_images, self.count, self.seed,
                Path(trainer.save_dir) / "previews" / "fixed_samples.json",
                split_id=self.data.get("split_id"),
            )
            if self.manifest["count"] != self.count:
                print(f"Keeping existing fixed preview manifest: {self.manifest['count']} images "
                      f"(new-run request: {self.count})", flush=True)
        if self.writer_type:
            event_dir = dashboard_directory(Path(trainer.save_dir))
            self.writer = self.writer_type(str(event_dir))
            protocol = {
                "schema": "calli-grouped-dashboard-v3", "event_directory": str(event_dir),
                "scalar_tags": sorted(DASHBOARD_SCALAR_TAGS),
                "common_loss_components": list(COMMON_LOSS_NAMES),
                "loss_definition": (
                    "BF/JT train_combined and val_combined read actual one-to-one criterion accumulators "
                    "for four common weighted main terms, using the original shared denominators. "
                    "These are additive task contributions, not the complete E2E objective; "
                    "train-only sem_loss is excluded and missing groups are omitted."
                ),
                "image_tag_template": PREVIEW_IMAGE_TAG_TEMPLATE,
                "image_card_policy": "one independent card per epoch; fixed sample order within each card",
                "image_encoding": "v2 DT_STRING [width,height,png0,...]",
                "preview_schedule": "every epoch", "requested_preview_count": self.count,
                "details": "Native loss components, mAP, and LR remain in results.csv; region counts remain in JSON.",
                "history": "Existing event files under the old run/tensorboard directory are unchanged.",
            }
            (Path(trainer.save_dir) / "dashboard_protocol.json").write_text(
                json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"TensorBoard minimal dashboard enabled: {event_dir}", flush=True)

    def on_epoch_start(self, trainer) -> None:
        self.trainer = trainer
        self.active_epoch = trainer.epoch
        _reset_group_losses(_loss_model(trainer))

    def on_val_start(self, validator) -> None:
        if self.trainer is not None:
            _reset_group_losses(_loss_model(self.trainer, validation=True))

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
        totals = group_loss_totals(_read_group_losses(_loss_model(trainer)),
                                  _read_group_losses(_loss_model(trainer, validation=True)))
        self._scalars(totals, step)
        self._preview(trainer, step)
        stop_request = Path(trainer.save_dir) / "STOP"
        if stop_request.is_file():
            trainer.stop = True
            stop_request.rename(Path(trainer.save_dir) / f"STOP.handled_epoch_{step:03d}")
            print(f"Manual stop requested; finishing cleanly after epoch {step}", flush=True)
        self.last_fit_epoch = trainer.epoch
        if self.writer:
            self.writer.flush()

    def _preview(self, trainer, step: int) -> None:
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
                split_id=self.data.get("split_id"),
            )
        rows = self.manifest["samples"] if self.count else []
        preview_lookup = {(data_root / row["image"]).resolve(): (index, row)
                          for index, row in enumerate(rows)}
        sources = sorted({Path(source).resolve() for source in self.val_images})
        authors = validation_authors(data_root, sources)
        known = annotation_tasks(self.data, sources)
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
                "data_split_id": self.data.get("split_id"),
                "evaluated_images": len(sources), "preview_images": len(rows),
                "evaluation_interval_epochs": 1,
                "evaluation_schedule": "every epoch",
                "unknown_task_policy": "exclude unannotated task from metrics; show both predictions",
                "tensorboard_image_tag": preview_image_tag(step),
                "tensorboard_image_card_policy": "one card per epoch; same fixed sample manifest",
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
                counts.update(truth, predicted, image_id=relative, known_tasks=known[source])
                if source in preview_lookup:
                    index, row = preview_lookup[source]
                    unique_truth, removed = deduplicate_gt_instances(truth)
                    preview = panel_image(image, unique_truth, predicted, self.data["names"], step,
                                          source.stem, known_tasks=known[source])
                    filename = f"{index:02d}_{source.stem}.png"
                    preview.save(folder / filename)
                    metadata["samples"].append({
                        "image": row["image"], "author": row.get("author", authors[source]),
                        "preview": filename, "gt_instances_raw": len(truth),
                        "gt_instances": len(unique_truth), "gt_duplicates_removed": removed,
                        "pred_instances": len(predicted),
                        "known_tasks": known[source],
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
            if self.writer and rows:
                if len(metadata["samples"]) != len(rows):
                    raise RuntimeError("Not all fixed preview samples were exported")
                write_image_card(self.writer, [folder / row["preview"] for row in metadata["samples"]], step)
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
