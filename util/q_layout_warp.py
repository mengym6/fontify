"""Read-only, single-instance spatial interventions on lower feature grids.

All features use B,H,W,C (production: 1,28,28,C). Coordinates are cell
centres in [-1,1], align_corners=False. Nothing in this module accepts GT.
"""

import math

import torch
from torch.nn import functional as F


def moments(mass):
    """Return centroid and sigma in grid-cell units for (...,H,W) mass."""
    height, width = mass.shape[-2:]
    x = torch.arange(width, device=mass.device, dtype=mass.dtype) + 0.5
    y = torch.arange(height, device=mass.device, dtype=mass.dtype) + 0.5
    total = mass.sum((-1, -2)).clamp_min(1e-12)
    cx = (mass * x).sum((-1, -2)) / total
    cy = (mass * y[:, None]).sum((-1, -2)) / total
    vx = (mass * (x - cx[..., None, None]).square()).sum((-1, -2)) / total
    vy = (mass * (y[:, None] - cy[..., None, None]).square()).sum((-1, -2)) / total
    return torch.stack((cx, cy), -1), torch.stack((vx, vy), -1).clamp_min(1e-12).sqrt()


def sampling_grid(support, axis, ratio, output_shape=None):
    """Inverse affine grid around the unmodified soft-support centroid."""
    if support.ndim != 3 or axis not in (0, 1):
        raise ValueError("Need B,H,W support and x=0 or y=1")
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("Scale must be positive and finite")
    height, width = output_shape or support.shape[-2:]
    center, _ = moments(support)
    center = center / support.new_tensor(support.shape[-2:][::-1]) * 2 - 1
    x = 2 * (torch.arange(width, device=support.device) + 0.5) / width - 1
    y = 2 * (torch.arange(height, device=support.device) + 0.5) / height - 1
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((xx, yy), -1)[None].expand(support.shape[0], -1, -1, -1)
    factors = support.new_ones(2)
    factors[axis] = ratio
    return center[:, None, None] + (grid - center[:, None, None]) / factors


def resample(values, grid):
    """Sample B,H,W,C in FP32, using zero extension outside the domain."""
    return F.grid_sample(
        values.float().permute(0, 3, 1, 2),
        grid.float(),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).permute(0, 2, 3, 1)


def scale_support(support, axis, ratio):
    if ratio == 1:
        return support
    return resample(support[..., None], sampling_grid(support, axis, ratio))[..., 0]


def warp_features(features, support, axis, ratio):
    """Soft remove/deposit one instance; normalize overlapping contributions.

    X' = ((1-m)X + sample(mX, inverse_grid)) / max(1,1-m+sample(m)).
    Zero is the vacated feature value. This is a diagnostic operation, not a
    learned source/target correspondence. Bilinear soft masks avoid hard cuts.
    """
    if features.shape[:3] != support.shape or features.ndim != 4:
        raise ValueError("Need aligned B,H,W,C features and B,H,W support")
    if ratio == 1:
        return features
    grid = sampling_grid(support, axis, ratio)
    moved = resample(support[..., None], grid)
    deposit = resample(features * support[..., None], grid)
    retained = 1 - support[..., None]
    return ((retained * features + deposit) / (retained + moved).clamp_min(1)).to(
        features.dtype
    )


def predicted_supports(head):
    """Use the same raw soft occupancy for Q/F and G-only geometry.

    Existence filters instances, but does not change their support. Only one
    instance moves at a time; overlap normalization is in ``warp_features``.
    """
    return head["occupancy_logits"].float().sigmoid()


def slot_validity(head, supports, minimum_existence=0.5, minimum_area=1.0):
    """Prediction-only validity; no matching or GT is permitted here."""
    existence = head["existence_logits"].sigmoid()
    center, sigma = moments(supports)
    area = supports.sum((-1, -2))
    finite = torch.isfinite(center).all(-1) & torch.isfinite(sigma).all(-1)
    valid = (existence >= minimum_existence) & (area >= minimum_area) & finite
    return valid, existence, area


def compose_g(head, occupancy):
    """Change only occupancy; preserve slot vectors, classes, Q and existence."""
    gates = occupancy * head["existence_logits"].sigmoid()[..., None, None]
    information = torch.cat((head["slots"], head["class_logits"].softmax(-1)), -1)
    field = torch.einsum("bkhw,bkd->bhwd", gates, information) / occupancy.shape[1]
    query = head["query"].reshape(*field.shape[:3], -1)
    return torch.cat((field, query), -1)


def intervention(fusion, head, lower, support, slot, axis, ratio, group):
    """Freeze the complementary normalized slice, including joint-LN coupling.

    A1/A4/A5/AO: warp Q *after* the original joint LayerNorm, before Linear.
    A2: rescale one raw occupancy, recompute joint LN, use only changed G
    slice and keep original normalized Q. Thus the effective Q path is fixed.
    A3/A4: warp F_full, keeping the original additive residual for A3.
    """
    if group not in ("A1", "A2", "A3", "A4", "A5", "AO"):
        raise ValueError("Unknown intervention group")
    normalized = fusion.norm(head["layout"])
    g_width = head["slots"].shape[-1] + head["class_logits"].shape[-1]
    g_fixed, q_fixed = normalized[..., :g_width], normalized[..., g_width:]
    if ratio == 1:
        return lower, fusion.projection(normalized), g_fixed, q_fixed
    g_changed, q_changed, changed_lower = g_fixed, q_fixed, lower
    if group == "A2":
        occupancy = head["occupancy_logits"].sigmoid().clone()
        occupancy[:, slot] = scale_support(occupancy[:, slot], axis, ratio)
        g_changed = fusion.norm(compose_g(head, occupancy))[..., :g_width]
    if group in ("A1", "A4", "A5", "AO"):
        q_changed = warp_features(q_fixed, support, axis, ratio)
    if group in ("A3", "A4"):
        changed_lower = warp_features(lower, support, axis, ratio)
    residual = fusion.projection(torch.cat((g_changed, q_changed), -1))
    return changed_lower, residual, g_changed, q_changed


def support_diagnostics(support, other, axis, ratio):
    """Measure actual support sigma, mass clipping and overlap, never infer it."""
    changed = scale_support(support, axis, ratio)
    _, before_sigma = moments(support)
    _, after_sigma = moments(changed)
    actual = float((after_sigma / before_sigma)[0, axis])
    height, width = support.shape[-2:]
    padded = F.pad(support, (width, width, height, height))
    expected_mass = float(scale_support(padded, axis, ratio).sum())
    clipping = max(0.0, 1 - float(changed.sum()) / max(expected_mass, 1e-12))
    overlap = float(torch.minimum(changed, other.clamp(0, 1)).sum()) / max(
        float(changed.sum()), 1e-12
    )
    boundary = changed.clone()
    boundary[:, 1:-1, 1:-1] = 0
    edge_mass = float(boundary.sum()) / max(float(changed.sum()), 1e-12)
    return changed, {
        "support_ratio": actual,
        "support_mass_ratio": float(changed.sum() / support.sum().clamp_min(1e-12)),
        "mass_clipping_fraction": clipping,
        "boundary_mass_fraction": edge_mass,
        "overlap_fraction": overlap,
        "support_sigma_before": float(before_sigma[0, axis]),
        "support_sigma_after": float(after_sigma[0, axis]),
    }
