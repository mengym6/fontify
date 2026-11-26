"""Count-based BF/JT instance metrics, independent of torch and Ultralytics.

Predictions below ``conf_threshold`` are discarded first. BF classes 0..49 and
JT classes 50..69 are matched separately by mask geometry, ignoring the fine
class label. Within each group, matching first maximizes the number of pairs
whose IoU reaches the threshold, then their total IoU. A cross-BF/JT class error
therefore counts as a miss in the GT group and a false positive in the other
group. A wrong fine label within one group retains its geometric match.

The three displayed metrics are deliberately not standard precision/recall
with one common TP definition: recall = geometric matches / GT; label-aware
precision = label-correct geometric matches / all predictions; IoU = the mean
mask IoU over geometric matches. Wrong fine labels within BF or JT affect
precision, not geometric recall or IoU. Extra predictions lower precision.
Unmatched GT regions lower recall and are not given zero IoU in that mean.

Legacy JSON fields remain available: region_precision = geometric matches /
predictions; label_accuracy = correct labels / geometric matches; miss_rate =
missed / GT. They are not displayed as the new precision. All denominators
count regions, not pixel areas. Zero denominators yield None. Aggregation sums
counts and matched IoUs first, never averages per-image percentages.

GT deduplication is opt-in and removes only same-class, pixel-identical masks.
Different classes or overlapping instances are retained. Callers should use
one confidence/IoU/deduplication policy for every report they aggregate.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


GROUPS = ("BF", "JT", "all")
COUNT_FIELDS = (
    "raw_gt_count", "duplicate_gt_count", "removed_duplicate_gt_count",
    "gt_count", "pred_count", "matched_count", "label_correct_count",
    "missed_count", "false_positive_count",
)


def _threshold(value: float, name: str, *, positive: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1 or (positive and value == 0):
        bound = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{name} must be finite and in {bound}")
    return value


def _prepare(instances: Sequence[dict]) -> list[dict]:
    """Validate one image's instances without modifying input objects."""
    output = []
    shape = None
    for instance in instances:
        class_id = instance["class_id"]
        if (isinstance(class_id, (bool, np.bool_))
                or not isinstance(class_id, (int, np.integer))
                or not 0 <= int(class_id) < 70):
            raise ValueError("class_id must be an integer in 0..69")
        mask = np.asarray(instance["mask"])
        if mask.ndim != 2 or 0 in mask.shape:
            raise ValueError("Instance masks must have a nonempty HxW shape")
        if mask.dtype != np.bool_ and not np.isin(mask, (0, 1)).all():
            raise ValueError("Instance masks must be boolean or binary 0/1 arrays")
        mask = mask.astype(bool, copy=False)
        if shape is not None and mask.shape != shape:
            raise ValueError("All masks for one image must have the same HxW shape")
        shape = mask.shape
        y, x = np.nonzero(mask)
        bounds = (int(y.min()), int(y.max()) + 1, int(x.min()), int(x.max()) + 1) if y.size else None
        output.append({"class_id": int(class_id), "mask": mask,
                       "area": int(y.size), "bounds": bounds})
    return output


def _unique_indices(instances: Sequence[dict]) -> list[int]:
    seen = set()
    indices = []
    for index, instance in enumerate(instances):
        mask = instance["mask"]
        key = (instance["class_id"], mask.shape, np.packbits(mask, axis=None).tobytes())
        if key not in seen:
            indices.append(index)
            seen.add(key)
    return indices


def deduplicate_gt_instances(instances: Sequence[dict]) -> tuple[list[dict], int]:
    """Return first occurrences and removed count; input dictionaries stay intact.

    Use these unique instances for a GT overlay when metrics explicitly enable
    deduplication. Still pass the original GT list to ``evaluate_regions`` or
    the accumulator so that its duplicate counters remain informative.
    """
    prepared = _prepare(instances)
    indices = _unique_indices(prepared)
    return [instances[index] for index in indices], len(instances) - len(indices)


def _iou_matrix(truth: Sequence[dict], predictions: Sequence[dict]) -> np.ndarray:
    overlaps = np.zeros((len(truth), len(predictions)), dtype=np.float64)
    for gi, gt in enumerate(truth):
        if gt["bounds"] is None:
            continue
        gy0, gy1, gx0, gx1 = gt["bounds"]
        for pi, pred in enumerate(predictions):
            if pred["bounds"] is None:
                continue
            py0, py1, px0, px1 = pred["bounds"]
            y0, y1 = max(gy0, py0), min(gy1, py1)
            x0, x1 = max(gx0, px0), min(gx1, px1)
            if y0 >= y1 or x0 >= x1:
                continue
            intersection = int(np.count_nonzero(
                gt["mask"][y0:y1, x0:x1] & pred["mask"][y0:y1, x0:x1]
            ))
            union = gt["area"] + pred["area"] - intersection
            overlaps[gi, pi] = intersection / union if union else 0.0
    return overlaps


def _match_iou(overlaps: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Maximum-cardinality, then maximum-total-IoU thresholded matching."""
    gt_count, pred_count = overlaps.shape
    if not gt_count or not pred_count:
        return []
    # Each GT can choose a real prediction or its own available dummy column.
    # The bonus exceeds every possible change in total IoU, so gaining one
    # valid match always dominates improving any number of existing matches.
    bonus = min(gt_count, pred_count) + 1.0
    cost = np.zeros((gt_count, pred_count + gt_count), dtype=np.float64)
    valid = overlaps >= threshold
    cost[:, :pred_count] = np.where(valid, -(bonus + overlaps), 1.0)
    rows, columns = linear_sum_assignment(cost)
    return [(int(gi), int(pi)) for gi, pi in zip(rows, columns)
            if pi < pred_count and valid[gi, pi]]


def _ratio(numerator: int | float, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _with_rates(counts: dict) -> dict:
    result = {key: int(counts[key]) for key in COUNT_FIELDS}
    result["matched_iou_sum"] = float(counts["matched_iou_sum"])
    result.update(
        region_recall=_ratio(result["matched_count"], result["gt_count"]),
        label_accuracy=_ratio(result["label_correct_count"], result["matched_count"]),
        miss_rate=_ratio(result["missed_count"], result["gt_count"]),
        region_precision=_ratio(result["matched_count"], result["pred_count"]),
        label_aware_precision=_ratio(result["label_correct_count"], result["pred_count"]),
        mean_matched_iou=_ratio(result["matched_iou_sum"], result["matched_count"]),
    )
    return result


def _sum_groups(groups: Iterable[dict]) -> dict:
    counts = dict.fromkeys(COUNT_FIELDS, 0)
    counts["matched_iou_sum"] = 0.0
    for group in groups:
        for key in COUNT_FIELDS:
            counts[key] += group[key]
        counts["matched_iou_sum"] += group["matched_iou_sum"]
    return _with_rates(counts)


def evaluate_regions(
    gt_instances: Sequence[dict],
    pred_instances: Sequence[dict],
    *,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
    deduplicate_gt: bool = False,
) -> dict:
    """Evaluate one image; return JSON-serializable BF, JT, and all summaries.

    Instances contain ``class_id`` and a boolean HxW ``mask``. Prediction
    confidence defaults to 1.0 when absent. Filtering occurs before geometric
    matching. ``duplicate_gt_count`` reports exact duplicates even when kept;
    ``removed_duplicate_gt_count`` is nonzero only with explicit deduplication.
    """
    conf_threshold = _threshold(conf_threshold, "conf_threshold")
    iou_threshold = _threshold(iou_threshold, "iou_threshold", positive=True)
    filtered = []
    for instance in pred_instances:
        confidence = _threshold(instance.get("confidence", 1.0), "confidence")
        if confidence >= conf_threshold:
            filtered.append(instance)
    truth, predictions = _prepare(gt_instances), _prepare(filtered)
    if truth and predictions and truth[0]["mask"].shape != predictions[0]["mask"].shape:
        raise ValueError("GT and prediction masks must share the same HxW shape")
    result = {}
    for name, low, high in (("BF", 0, 50), ("JT", 50, 70)):
        raw = [item for item in truth if low <= item["class_id"] < high]
        unique_indices = _unique_indices(raw)
        duplicate_count = len(raw) - len(unique_indices)
        selected = [raw[index] for index in unique_indices] if deduplicate_gt else raw
        predicted = [item for item in predictions if low <= item["class_id"] < high]
        overlaps = _iou_matrix(selected, predicted)
        pairs = _match_iou(overlaps, iou_threshold)
        matched = len(pairs)
        result[name] = _with_rates({
            "raw_gt_count": len(raw), "duplicate_gt_count": duplicate_count,
            "removed_duplicate_gt_count": duplicate_count if deduplicate_gt else 0,
            "gt_count": len(selected), "pred_count": len(predicted),
            "matched_count": matched,
            "label_correct_count": sum(selected[gi]["class_id"] == predicted[pi]["class_id"]
                                       for gi, pi in pairs),
            "missed_count": len(selected) - matched,
            "false_positive_count": len(predicted) - matched,
            "matched_iou_sum": float(sum(overlaps[gi, pi] for gi, pi in pairs)),
        })
    result["all"] = _sum_groups((result["BF"], result["JT"]))
    return result


def aggregate_region_metrics(per_image_reports: Iterable[dict]) -> dict:
    """Micro-average compatible per-image reports; ignore optional image_id."""
    reports = list(per_image_reports)
    result = {name: _sum_groups(report[name] for report in reports) for name in ("BF", "JT")}
    result["all"] = _sum_groups((result["BF"], result["JT"]))
    return result


def tensorboard_scalars(report: dict) -> dict[str, float]:
    """Return only BF/JT recall, label-aware precision and mask IoU percentages.

    Undefined values are omitted rather than displayed as zero or 100 percent.
    Accepts the output of either ``evaluate_regions`` or the accumulator's
    ``summary``. Legacy rates and diagnostic counts stay in JSON, not the UI.
    There is no dependency on a TensorBoard writer or torch.
    """
    output = {}
    primary = {
        "recall": "region_recall",
        "precision": "label_aware_precision",
        "iou": "mean_matched_iou",
    }
    for name in ("BF", "JT"):
        group = report[name]
        for display_name, key in primary.items():
            value = group.get(key)
            if value is not None:
                output[f"metrics/{name}/{display_name}_pct"] = 100.0 * float(value)
    return output


class RegionMetricsAccumulator:
    """Keep scalar per-image records without retaining masks or GPU tensors."""

    def __init__(self, *, conf_threshold: float = 0.25, iou_threshold: float = 0.5,
                 deduplicate_gt: bool = False):
        self.options = {
            "conf_threshold": _threshold(conf_threshold, "conf_threshold"),
            "iou_threshold": _threshold(iou_threshold, "iou_threshold", positive=True),
            "deduplicate_gt": deduplicate_gt,
        }
        self.per_image: list[dict] = []

    def update(self, gt_instances: Sequence[dict], pred_instances: Sequence[dict],
               image_id: str | None = None) -> dict:
        report = evaluate_regions(gt_instances, pred_instances, **self.options)
        report["image_id"] = str(image_id) if image_id is not None else None
        self.per_image.append(report)
        return report

    def summary(self) -> dict:
        return aggregate_region_metrics(self.per_image)
