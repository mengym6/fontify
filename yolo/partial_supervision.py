"""Partially annotated Calli instance segmentation for Ultralytics 8.4.218.

The short criterion/dataset overrides below adapt the pinned Ultralytics
AGPL-3.0 implementation (https://github.com/ultralytics/ultralytics). Model
structure, assigners, E2E scheduling, and native primary-loss gains are kept.
Torch/Ultralytics imports are lazy for metadata checks and --dry-run.
"""

from __future__ import annotations

import json
import hashlib
import os
import random
from collections import defaultdict
from pathlib import Path


GROUPS = ("BF", "JT")
PRIMARY_NAMES = ("box_loss", "seg_loss", "cls_loss", "l1_loss")
PINNED_VERSION = "8.4.218"


def load_annotation_availability(data: dict):
    value = data.get("annotation_availability")
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path(data["path"]) / path
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("annotation_availability must map relative image paths to task flags")
    for name, row in mapping.items():
        if not isinstance(row, dict) or any(not isinstance(row.get(key), bool) for key in ("has_bf", "has_jt")):
            raise ValueError(f"Invalid annotation-availability flags: {name}")
    return mapping


def image_task_valid(image: str | Path, data_root: str | Path, availability) -> tuple[bool, bool]:
    if availability is None:
        return True, True
    relative = Path(image).resolve().relative_to(Path(data_root).resolve()).as_posix()
    if relative not in availability:
        raise KeyError(f"Image missing from annotation_availability: {relative}")
    row = availability[relative]
    return row["has_bf"], row["has_jt"]


def reject_mixed_image_augmentation(hyp, augment=True) -> None:
    if augment:
        for name in ("mixup", "copy_paste", "cutmix"):
            if float(getattr(hyp, name, 0) or 0) != 0:
                raise ValueError(f"Partial annotation does not support cross-image {name}; set it to 0")
    if getattr(hyp, "overlap_mask", False):
        raise ValueError("Partial BF/JT supervision requires overlap_mask=False to preserve each instance")
    if getattr(hyp, "single_cls", False) or getattr(hyp, "classes", None) is not None:
        raise ValueError("Partial BF/JT supervision requires the complete 70-class label space")


def _same_task_mosaic_class(native_class):
    """Change donor sampling only; retain the native mosaic geometry and RNG."""
    class SameTaskMosaic(native_class):
        def get_params(self, labels):
            previous = getattr(self, "_active_task_valid", None)
            self._active_task_valid = image_task_valid(
                labels["im_file"], self.dataset.availability_root, self.dataset.annotation_availability,
            )
            try:
                return super().get_params(labels)
            finally:
                self._active_task_valid = previous

        def get_indexes(self):
            flags = self._active_task_valid
            pool = self.dataset.mosaic_task_pools[flags]
            if self.buffer_enabled:
                buffered = [index for index in self.dataset.buffer
                            if self.dataset.mosaic_task_flags[index] == flags]
                return random.choices(buffered or pool, k=self.n - 1)
            return [random.choice(pool) for _ in range(self.n - 1)]

    SameTaskMosaic.__qualname__ = "SameTaskMosaic"
    return SameTaskMosaic


def _mosaic_class():
    if "SameTaskMosaic" not in globals():
        from ultralytics.data.augment import Mosaic
        globals()["SameTaskMosaic"] = _same_task_mosaic_class(Mosaic)
    return globals()["SameTaskMosaic"]


def replace_mosaic_sources(transforms):
    """Replace Mosaic nodes in this dataset's Compose; no global monkeypatch."""
    from ultralytics.data.augment import Compose, Mosaic
    for index, transform in enumerate(transforms.transforms):
        if isinstance(transform, Mosaic):
            transforms.transforms[index] = _mosaic_class()(
                transform.dataset, imgsz=transform.imgsz, p=transform.p, n=transform.n,
            )
        elif isinstance(transform, Compose):
            replace_mosaic_sources(transform)
    return transforms


def _branches(model):
    model = getattr(model, "module", model)
    criterion = getattr(model, "criterion", None)
    if criterion is None:
        return []
    return [criterion.one2many, criterion.one2one] if hasattr(criterion, "one2one") else [criterion]


def reset_group_loss_accumulators(model) -> None:
    """Clear existing criteria only; never create a criterion on model or EMA."""
    for branch in _branches(model):
        branch._calli_group_accumulator = None


def read_group_loss_accumulators(model) -> dict:
    """Read the one-to-one branch's actual common-term batch averages.

    These are local-rank criterion-call means, matching native tloss/val-loss
    batch averaging. No implicit distributed collective runs from a rank-0 hook.
    """
    branches = _branches(model)
    state = getattr(branches[-1], "_calli_group_accumulator", None) if branches else None
    result = {"BF": None, "JT": None, "batches": 0,
              "known_samples": {"BF": 0, "JT": 0}, "components": {}, "common_total": None,
              "branch": "one2one" if len(branches) == 2 else "single",
              "aggregation": "arithmetic mean over actual criterion calls on this rank"}
    if not state or not state["batches"]:
        return result
    result["batches"] = state["batches"]
    result["common_total"] = sum(sum(row) for row in state["sums"]) / state["batches"]
    for index, group in enumerate(GROUPS):
        result["known_samples"][group] = state["known_samples"][index]
        values = [value / state["batches"] for value in state["sums"][index]]
        result["components"][group] = dict(zip(PRIMARY_NAMES, values))
        result[group] = sum(values) if state["known_samples"][index] else None
    return result


def save_partial_protocol(trainer) -> dict:
    """Write the supervised-domain contract; reject changed resume identities."""
    data = trainer.data
    availability = data.get("annotation_availability")
    path = None
    if availability:
        path = Path(availability)
        if not path.is_absolute():
            path = Path(data["path"]) / path
    protocol = {
        "schema": "calli-partial-supervision-v1",
        "annotation_availability": str(path) if path else None,
        "availability_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path else None,
        "split_id": data.get("split_id"),
        "mode": "availability-masked" if path else "all-known-legacy",
        "staged_training": getattr(trainer, "staged_training", None),
        "task_valid": "bool[B,2], columns BF/JT; known empty regions remain supervised negatives",
        "class_groups": {"BF": [0, 50], "JT": [50, 70]},
        "semantic_targets": "per-class instance unions, allowing overlapping BF/JT multi-hot targets",
        "semantic_denominator": "native B*70*H*W BCE and B*70 Dice; unknown channels have zero contribution",
        "primary_group_denominators": "shared native target_scores_sum (box/cls/L1), foreground count (mask)",
        "group_epoch_reduction": "one-to-one criterion-call arithmetic means on each local rank",
        "validation": "unknown-task predictions excluded only from native metrics, not model output",
    }
    destination = Path(trainer.save_dir) / "partial_supervision.json"
    if destination.is_file():
        old = json.loads(destination.read_text(encoding="utf-8"))
        for key in ("schema", "availability_sha256", "split_id", "mode", "staged_training"):
            if old.get(key) != protocol[key]:
                raise ValueError(f"Partial supervision identity changed for existing run: {key}")
    if int(os.environ.get("RANK", "-1")) in {-1, 0}:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    return protocol


def filter_unknown_task_predictions(predictions, task_valid):
    """Return filtered copies solely for metric accounting; keep full outputs."""
    if task_valid is None:
        return predictions
    import torch
    filtered = []
    for index, prediction in enumerate(predictions):
        classes = prediction["cls"]
        known = torch.as_tensor(task_valid[index], device=classes.device, dtype=torch.bool)
        keep = ((classes < 50) & known[0]) | ((classes >= 50) & (classes < 70) & known[1])
        filtered.append({key: value[keep] for key, value in prediction.items()})
    return filtered


def task_channel_mask(task_valid, *, device=None, dtype=None):
    import torch
    flags = torch.as_tensor(task_valid, device=device, dtype=torch.bool)
    if flags.ndim != 2 or flags.shape[1] != 2:
        raise ValueError("task_valid must have shape [batch, 2] for BF/JT")
    return torch.cat((flags[:, :1].expand(-1, 50), flags[:, 1:].expand(-1, 20)), 1).to(dtype=dtype)


def multihot_semantic_targets(masks, batch_idx, classes, batch_size: int, shape):
    import torch
    import torch.nn.functional as F
    masks = masks.float()
    if tuple(masks.shape[-2:]) != tuple(shape) and len(masks):
        masks = F.interpolate(masks.unsqueeze(1), shape, mode="nearest").squeeze(1)
    target = masks.new_zeros(batch_size * 70, *shape)
    if len(masks):
        indices = batch_idx.reshape(-1).long() * 70 + classes.reshape(-1).long()
        target.index_add_(0, indices, masks)
    return target.clamp_(0, 1).view(batch_size, 70, *shape)


def masked_semantic_loss(prediction, target, task_valid):
    """Native 0.5 BCE + 0.5 Dice, with unknown sample/channels contributing zero.

    The fixed N*70 denominator is retained; unknown tasks do not upweight known
    tasks. Full-known, nonoverlapping inputs reproduce the native expression.
    """
    import torch.nn.functional as F
    valid = task_channel_mask(task_valid, device=prediction.device, dtype=prediction.dtype)
    bce = (F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
           * valid[:, :, None, None]).mean()
    probabilities = prediction.sigmoid()
    intersection = (probabilities * target).sum((2, 3))
    union = probabilities.sum((2, 3)) + target.sum((2, 3))
    dice = ((1 - (2 * intersection + 1) / (union + 1)) * valid).mean()
    return .5 * bce + .5 * dice


def connected_zero(*tensors):
    """Keep zero-gradient graph paths without overflowing an FP16 reduction."""
    return sum((tensor * 0).sum() for tensor in tensors)


def _partial_segment_class(native_class):
    import torch
    import torch.nn.functional as F
    from ultralytics.utils.loss import (
        bbox2dist, bbox_iou, crop_mask, interpolate_bilinear, make_anchors, xyxy2xywh,
    )

    class PartialSegmentLoss26(native_class):
        _calli_partial = True

        def _record_groups(self, groups, task_valid):
            self._last_group_components = groups.detach()
            state = getattr(self, "_calli_group_accumulator", None)
            if state is None:
                state = {"sums": [[0.] * 4 for _ in GROUPS], "batches": 0, "known_samples": [0, 0]}
                self._calli_group_accumulator = state
            values = groups.detach().double().cpu().tolist()
            known = task_valid.sum(0).cpu().tolist()
            state["batches"] += 1
            for group in range(2):
                state["known_samples"][group] += int(known[group])
                for index in range(4):
                    state["sums"][group][index] += values[group][index]

        def loss(self, preds, batch):
            if self.nc != 70 or self.overlap or self.use_dfl:
                raise ValueError("Partial loss requires YOLO26, nc=70, reg_max=1, overlap_mask=False")
            scores = preds["scores"].permute(0, 2, 1).contiguous()
            distribution = preds["boxes"].permute(0, 2, 1).contiguous()
            coefficients = preds["mask_coefficient"].permute(0, 2, 1).contiguous()
            proto = preds["proto"]
            proto, semantic = proto if isinstance(proto, tuple) else (proto, None)
            bs = scores.shape[0]
            if getattr(self, "_calli_bf_only", False):
                from staged_training import bf_only_targets
                batch = bf_only_targets(batch, bs)
            valid = torch.as_tensor(batch.get("task_valid", torch.ones(bs, 2)), device=self.device, dtype=torch.bool)
            channel_valid = task_channel_mask(valid, device=self.device, dtype=scores.dtype)
            if valid.shape[0] != bs:
                raise ValueError("task_valid batch dimension mismatch")
            batch_idx = batch["batch_idx"].reshape(-1).long().to(self.device)
            classes = batch["cls"].reshape(-1).long().to(self.device)
            if len(classes) and (not bool(((classes >= 0) & (classes < 70)).all())
                                 or not bool(valid[batch_idx, (classes >= 50).long()].all())):
                raise ValueError("GT class belongs to an unavailable annotation task")
            masks = batch["masks"].to(self.device).float()
            anchors, strides = make_anchors(preds["feats"], self.stride, .5)
            imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=scores.dtype) * self.stride[0]
            raw_targets = torch.cat((batch_idx[:, None], classes[:, None], batch["bboxes"].to(self.device)), 1)
            targets = self.preprocess(raw_targets.to(self.device), bs, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_boxes = targets.split((1, 4), 2)
            mask_gt = gt_boxes.sum(2, keepdim=True).gt_(0)
            predicted_boxes = self.bbox_decode(anchors, distribution)

            # Same mask-aware task alignment as native SegmentLoss26. Unknown
            # logits are not zeroed here, so known-task assignment is unchanged.
            detached_proto = proto.detach()
            mh, mw = detached_proto.shape[-2:]
            sizes = torch.bincount(batch_idx, minlength=bs)
            assignment_masks = masks
            if len(masks) and tuple(masks.shape[-2:]) != (mh, mw):
                assignment_masks = F.interpolate(masks[None], (mh, mw), mode="nearest")[0]
            gt_masks = masks.new_zeros(bs, int(sizes.max()), mh, mw)
            if len(masks):
                within = torch.arange(len(batch_idx), device=self.device) - (sizes.cumsum(0) - sizes)[batch_idx]
                gt_masks[batch_idx, within] = assignment_masks
            _, target_boxes, target_scores, fg, target_idx = self.assigner(
                scores.detach().sigmoid(), (predicted_boxes.detach() * strides).type(gt_boxes.dtype),
                anchors * strides, gt_labels, gt_boxes, mask_gt,
                pd_coef=coefficients.detach(),
                mask_scale=(mh / (preds["feats"][0].shape[2] * self.stride[0])).expand(bs, 1),
                proto=detached_proto, gt_masks=gt_masks > .5,
            )
            normalizer = target_scores.sum().clamp(min=1)
            if gt_labels.shape[1]:
                assigned_class = gt_labels[..., 0].long().gather(1, target_idx.long())
            else:
                assigned_class = torch.zeros_like(target_idx)
            assigned_group = (assigned_class >= 50).long()

            # Grouped primary terms share the original total denominators.
            # No group-specific rescaling, sample-based guessing, or halving.
            # Native loss vectors are float32 even when model outputs are AMP
            # half/bfloat16. Do not round grouped numerators back to AMP dtype.
            groups = torch.zeros(2, 4, device=self.device)
            bce = self.bce(scores, target_scores.to(scores.dtype)) * channel_valid[:, None, :]
            if self.class_weights is not None:
                bce = bce * self.class_weights
            groups[0, 2] = bce[..., :50].sum() / normalizer * self.hyp.cls
            groups[1, 2] = bce[..., 50:].sum() / normalizer * self.hyp.cls
            positive_group = assigned_group[fg]
            weight = target_scores[fg].sum(-1, keepdim=True)
            boxes_grid = target_boxes / strides
            iou = bbox_iou(predicted_boxes[fg], boxes_grid[fg], xywh=False, CIoU=True)
            box_terms = ((1 - iou) * weight).reshape(-1)
            target_distance = bbox2dist(anchors, boxes_grid) * strides
            target_distance[..., 0::2] /= imgsz[1]
            target_distance[..., 1::2] /= imgsz[0]
            pred_distance = distribution * strides
            pred_distance[..., 0::2] /= imgsz[1]
            pred_distance[..., 1::2] /= imgsz[0]
            l1_terms = (F.l1_loss(pred_distance[fg], target_distance[fg], reduction="none")
                        .mean(-1, keepdim=True) * weight).reshape(-1)
            for group in range(2):
                groups[group, 0] = box_terms[positive_group == group].sum() / normalizer * self.hyp.box
                groups[group, 3] = l1_terms[positive_group == group].sum() / normalizer * self.hyp.dfl

            semantic_loss = connected_zero(proto, coefficients)
            if bool(fg.any()):
                mask_proto = proto
                if tuple(masks.shape[-2:]) != tuple(proto.shape[-2:]):
                    mask_proto = interpolate_bilinear(proto, masks.shape[-2:])
                mask_h, mask_w = mask_proto.shape[-2:]
                normalized_boxes = target_boxes / imgsz[[1, 0, 1, 0]]
                areas = xyxy2xywh(normalized_boxes)[..., 2:].prod(2)
                boxes_mask = normalized_boxes * proto.new_tensor([mask_w, mask_h, mask_w, mask_h])
                mask_sums = [connected_zero(mask_proto, coefficients) for _ in GROUPS]
                for image in range(bs):
                    selected = fg[image].nonzero(as_tuple=False).flatten()
                    if not len(selected):
                        continue
                    gt_mask = masks[batch_idx == image][target_idx[image, selected]]
                    logits = torch.einsum("in,nhw->ihw", coefficients[image, selected], mask_proto[image])
                    terms = F.binary_cross_entropy_with_logits(logits, gt_mask, reduction="none")
                    terms = crop_mask(terms, boxes_mask[image, selected]).mean((1, 2)) / areas[image, selected]
                    for group in range(2):
                        mask_sums[group] = mask_sums[group] + terms[assigned_group[image, selected] == group].sum()
                for group in range(2):
                    groups[group, 1] = mask_sums[group] / fg.sum() * self.hyp.box
                if semantic is not None:
                    semantic_targets = multihot_semantic_targets(masks, batch_idx, classes, bs, semantic.shape[-2:])
                    semantic_loss = masked_semantic_loss(semantic, semantic_targets, valid) * self.hyp.box
            else:
                for group in range(2):
                    groups[group, 1] = connected_zero(proto, coefficients)
            if semantic is not None:
                semantic_loss = semantic_loss + connected_zero(semantic)
            semantic_loss = semantic_loss * getattr(self, "_calli_semantic_weight", 1.0)
            common = groups.sum(0)
            loss = torch.cat((common, semantic_loss.reshape(1)))
            self._record_groups(groups, valid)
            return loss * bs, dict(zip(self.loss_names, loss.detach()))

    return PartialSegmentLoss26


def install_partial_supervision(model, semantic_weight=None) -> dict:
    import ultralytics
    from ultralytics.utils.loss import SegmentLoss26
    from ultralytics.utils.torch_utils import unwrap_model
    if ultralytics.__version__ != PINNED_VERSION:
        raise RuntimeError(f"Partial supervision requires ultralytics=={PINNED_VERSION}")
    model = unwrap_model(model)
    criterion = getattr(model, "criterion", None)
    if criterion is None:
        criterion = model.init_criterion()
    model.criterion = criterion
    partial_class = _partial_segment_class(SegmentLoss26)
    for parent, attribute in ((criterion, "one2many"), (criterion, "one2one")) if hasattr(criterion, "one2one") else ((model, "criterion"),):
        old = getattr(parent, attribute)
        if not isinstance(old, SegmentLoss26):
            raise TypeError("Expected native YOLO26 segmentation criterion")
        if getattr(old, "_calli_partial", False):
            wrapped = old
        else:
            wrapped = object.__new__(partial_class)
            wrapped.__dict__ = old.__dict__.copy()
            wrapped._calli_group_accumulator = None
        if semantic_weight is not None:
            wrapped._calli_semantic_weight = float(semantic_weight)
        setattr(parent, attribute, wrapped)
    return {"partial_supervision": True, "branches": len(_branches(model))}


def _training_classes():
    from copy import copy
    import torch
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.build import get_split_fraction
    from ultralytics.models.yolo.segment import SegmentationTrainer
    from ultralytics.models.yolo.segment import SegmentationValidator
    from ultralytics.utils import colorstr
    from ultralytics.utils.torch_utils import unwrap_model

    class PartialYOLODataset(YOLODataset):
        def __init__(self, *args, data=None, **kwargs):
            self.availability_root = Path(data["path"]).resolve()
            self.annotation_availability = load_annotation_availability(data)
            reject_mixed_image_augmentation(kwargs["hyp"], augment=kwargs.get("augment", False))
            super().__init__(*args, data=data, **kwargs)

        def build_transforms(self, hyp=None):
            transforms = super().build_transforms(hyp)
            if self.augment:
                self.mosaic_task_flags = [
                    image_task_valid(image, self.availability_root, self.annotation_availability)
                    for image in self.im_files
                ]
                self.mosaic_task_pools = defaultdict(list)
                for index, flags in enumerate(self.mosaic_task_flags):
                    self.mosaic_task_pools[flags].append(index)
                replace_mosaic_sources(transforms)
            return transforms

        def __getitem__(self, index):
            item = super().__getitem__(index)
            item["task_valid"] = torch.tensor(
                image_task_valid(item["im_file"], self.availability_root, self.annotation_availability),
                dtype=torch.bool,
            )
            return item

        @staticmethod
        def collate_fn(batch):
            flags = torch.stack([item["task_valid"] for item in batch])
            output = YOLODataset.collate_fn([{k: v for k, v in item.items() if k != "task_valid"} for item in batch])
            output["task_valid"] = flags
            return output

    class PartialSegmentationTrainer(SegmentationTrainer):
        def __init__(self, *args, staged_training=None, **kwargs):
            self.staged_training = staged_training
            super().__init__(*args, **kwargs)
            self.add_callback("on_train_start", self._install_partial_criteria)
            if staged_training:
                from staged_training import apply_training_stage, save_joint_checkpoint
                self.add_callback("on_train_epoch_start", apply_training_stage)
                self.add_callback("on_model_save", save_joint_checkpoint)

        @staticmethod
        def _install_partial_criteria(trainer):
            install_partial_supervision(trainer.model)
            ema = getattr(getattr(trainer, "ema", None), "ema", None)
            if ema is not None:
                install_partial_supervision(ema)
            save_partial_protocol(trainer)

        def build_dataset(self, img_path, mode="train", batch=None):
            cfg = self.args
            reject_mixed_image_augmentation(cfg, augment=mode == "train")
            if self.data["nc"] != 70:
                raise ValueError("Partial trainer requires 50 BF + 20 JT classes")
            stride = max(int(unwrap_model(self.model).stride.max()), 32)
            fraction = 1.0 if self.data.get("complete") else get_split_fraction(cfg.fraction, "train" if mode == "train" else cfg.split)
            return PartialYOLODataset(
                img_path=img_path, imgsz=cfg.imgsz, batch_size=batch, augment=mode == "train", hyp=cfg,
                rect=cfg.rect or mode == "val", cache=cfg.cache or None, single_cls=False, stride=stride,
                pad=0.0 if mode == "train" else .5, prefix=colorstr(f"{mode}: "), task=cfg.task,
                classes=None, data=self.data, fraction=fraction, tile=cfg.tile,
            )

        def get_validator(self):
            return PartialSegmentationValidator(
                self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks,
            )

    class PartialSegmentationValidator(SegmentationValidator):
        def update_metrics(self, predictions, batch):
            filtered = filter_unknown_task_predictions(predictions, batch.get("task_valid"))
            return super().update_metrics(filtered, batch)

        def get_stats(self):
            from official_metrics import grouped_mask_map_scalars
            stats = super().get_stats()
            # Sidechannel only: keep the native CSV schema and returned stats
            # exactly intact, including its box.map + seg.map fitness.
            self.grouped_mask_map = grouped_mask_map_scalars(self.metrics.seg)
            return stats

        def build_dataset(self, img_path, mode="val", batch=None):
            cfg = self.args
            fraction = get_split_fraction(cfg.fraction, cfg.split or "val")
            return PartialYOLODataset(
                img_path=img_path, imgsz=cfg.imgsz, batch_size=batch, augment=False, hyp=cfg,
                rect=cfg.rect, cache=cfg.cache or None, single_cls=False, stride=self.stride,
                pad=.5, prefix=colorstr(f"{mode}: "), task=cfg.task,
                classes=None, data=self.data, fraction=fraction, tile=cfg.tile,
            )

    classes = {"PartialYOLODataset": PartialYOLODataset,
               "PartialSegmentationTrainer": PartialSegmentationTrainer,
               "PartialSegmentationValidator": PartialSegmentationValidator}
    for name, cls in classes.items():
        cls.__qualname__ = name  # worker-spawn pickling resolves the lazy module-level attribute
    return classes


def __getattr__(name):
    if name == "SameTaskMosaic":
        return _mosaic_class()
    if name in {"PartialYOLODataset", "PartialSegmentationTrainer", "PartialSegmentationValidator"}:
        classes = _training_classes()
        globals().update(classes)
        return classes[name]
    raise AttributeError(name)
