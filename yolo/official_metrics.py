"""Read BF/JT mask mAP from native Ultralytics per-GT-class AP results.

This module does not perform matching, confidence filtering, AP integration,
or checkpoint selection. In particular, Metric.maps is not used: it fills
classes absent from GT with the global mean and would bias group averages.
"""

from __future__ import annotations

import numpy as np


def grouped_mask_map_scalars(seg_metric) -> dict[str, float]:
    """Return native mask AP averages as fractions, only over classes with GT.

    The pinned native ap_per_class includes GT classes with no predictions as
    zero AP rows. Keep those rows; omit only groups with no GT classes at all.
    all_ap has ten columns for IoU 0.50, 0.55, ..., 0.95.
    """
    classes = np.asarray(seg_metric.ap_class_index, dtype=np.int64).reshape(-1)
    if not len(classes):
        return {}
    ap = np.asarray(seg_metric.all_ap, dtype=np.float64)
    if ap.shape != (len(classes), 10):
        raise ValueError("Expected native mask all_ap shape [GT classes, 10] aligned with ap_class_index")
    output = {}
    for group, lower, upper in (("BF", 0, 50), ("JT", 50, 70)):
        selected = ap[(classes >= lower) & (classes < upper)]
        if len(selected):
            output[f"metrics/{group}/mAP50(M)"] = float(selected[:, 0].mean())
            output[f"metrics/{group}/mAP50-95(M)"] = float(selected.mean())
    return output
