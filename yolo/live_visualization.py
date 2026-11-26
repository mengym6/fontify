"""Epoch scalars and fixed BF/JT previews, independent of the training model.

Preview inference loads the current last.pt into a separate YOLO object. The
training model, EMA, optimizer, and their mode/dtype are never passed to predict.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from contextlib import contextmanager
from pathlib import Path


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


def fixed_samples(data_root: Path, candidates: list[Path], count: int,
                  seed: int, path: Path) -> dict:
    """Keep sample identity and image/label bytes fixed across epochs/resumes."""
    data_root = Path(data_root).resolve()
    available = sorted({Path(item).resolve() for item in candidates})
    count = min(count, len(available))
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("count") != count or manifest.get("seed") != seed:
            raise ValueError("Existing preview manifest count/seed changed; use the original settings")
    else:
        chosen = sorted(random.Random(seed).sample(available, count))
        manifest = {
            "schema": "calli-fixed-validation-preview-v1", "count": count,
            "seed": seed, "samples": [
                {"image": os.path.relpath(image, data_root),
                 "label": os.path.relpath(label_for(image), data_root),
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
                epoch: int, sample: str):
    """One source and four grouped overlays, all from the same prediction."""
    import colorsys
    import numpy as np
    from PIL import Image, ImageDraw

    image = image.convert("RGB")
    width, height = image.size
    header, footer = 62, 26
    canvas = Image.new("RGB", (width * 5, height + header + footer), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), f"Epoch {epoch:03d} | {sample} | current last.pt EMA", fill="black")
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
              "BF: stroke phases; JT: components; B/M/E: beginning/middle/ending; confidence >= 0.25",
              fill="black")
    return canvas


class LiveVisualization:
    def __init__(self, data: dict, val_images: list[Path], every: int = 5,
                 count: int = 4, seed: int = 42, tensorboard: bool = True):
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
        if self.writer_type:
            self.writer = self.writer_type(str(Path(trainer.save_dir) / "tensorboard"))
            print(f"TensorBoard enabled: {Path(trainer.save_dir) / 'tensorboard'}", flush=True)

    def on_epoch_start(self, trainer) -> None:
        self.active_epoch = trainer.epoch

    def _scalars(self, values: dict, step: int, prefix: str = "") -> None:
        if self.writer:
            for key, value in values.items():
                self.writer.add_scalar(prefix + key, float(value), step)

    def on_fit_end(self, trainer) -> None:
        if not primary_rank():
            return
        if trainer.epoch != self.active_epoch:
            # Ultralytics final_eval invokes this hook for best.pt with epoch+1.
            self._scalars(trainer.metrics, trainer.epoch + 1, "final_best/")
            if self.writer:
                self.writer.flush()
            return
        if self.last_fit_epoch == trainer.epoch:
            return
        step = trainer.epoch + 1
        self._scalars(trainer.label_loss_items(trainer.tloss, prefix="train"), step)
        self._scalars(trainer.lr, step)
        self._scalars(trainer.metrics, step)
        if self.count and should_preview(
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

        checkpoint = Path(trainer.last)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Current epoch preview checkpoint missing: {checkpoint}")
        # Recheck the pinned validation bytes rather than silently changing GT.
        self.manifest = fixed_samples(
            Path(self.data["path"]), self.val_images, self.count, self.seed,
            Path(trainer.save_dir) / "previews" / "fixed_samples.json",
        )
        with preserve_rng():
            predictor = YOLO(str(checkpoint))
            checkpoint_matches_epoch(predictor.ckpt, trainer.epoch)
            sources = [str(Path(self.data["path"]) / row["image"])
                       for row in self.manifest["samples"]]
            # A path list uses LoadPilAndNumpy in 8.4.218, which ignores batch=1
            # and stacks all images. Reuse one independent model but predict one
            # path at a time so preview memory does not scale with sample count.
            results = []
            for source in sources:
                result = predictor.predict(
                    source=source, device=trainer.device, imgsz=trainer.args.imgsz,
                    conf=.25, max_det=300, retina_masks=True, batch=1,
                    save=False, verbose=False, quantize=32,
                )
                if len(result) != 1:
                    raise RuntimeError("A single preview path must produce exactly one result")
                results.extend(result)
            if len(results) != len(sources):
                raise RuntimeError("Fixed preview output count does not match its manifest")
            folder = Path(trainer.save_dir) / "previews" / f"epoch_{step:03d}"
            folder.mkdir(parents=True, exist_ok=True)
            metadata = {"epoch": step, "checkpoint_epoch_zero_based": trainer.epoch,
                        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                        "source": "current epoch last.pt EMA", "samples": []}
            for index, (row, result) in enumerate(zip(self.manifest["samples"], results)):
                source = Path(self.data["path"]) / row["image"]
                if Path(result.path).resolve() != source.resolve():
                    raise RuntimeError("Fixed preview inference changed sample ordering")
                with Image.open(source) as image:
                    image = image.convert("RGB")
                truth = gt_instances(Path(self.data["path"]) / row["label"], image.height, image.width)
                predicted = prediction_instances(result)
                preview = panel_image(image, truth, predicted, self.data["names"], step, source.stem)
                filename = f"{index:02d}_{source.stem}.png"
                preview.save(folder / filename)
                if self.writer:
                    self.writer.add_image(f"previews/{index:02d}_{source.stem}",
                                          np.asarray(preview), step, dataformats="HWC")
                metadata["samples"].append({"image": row["image"], "preview": filename,
                                             "gt_instances": len(truth), "pred_instances": len(predicted)})
            (folder / "preview.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            del results, predictor
        print(f"Fixed validation previews saved: {folder}", flush=True)

    def close(self, trainer=None) -> None:
        if self.writer:
            self.writer.flush()
            self.writer.close()
            self.writer = None
