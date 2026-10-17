"""Run C v2 supervision with one slot assignment and explicit reductions.

Each optimized term averages valid parts inside a character, then averages
the complete batch. A character without eligible parts contributes zero.
Thus equal-size microbatches/ranks compose exactly; unequal-size batches must
be weighted by their actual character counts by the runner. Metric sums and
counts additionally expose instance-weighted evaluation without changing the
training objective. No old Run B/Run C v1 coefficient is imported here.
"""

import math
from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from util.dual_loss import InstanceHeadLoss
from util.jieti_loss import JietiLoss


@dataclass(frozen=True)
class RunCV2LossWeights:
    """Unit coefficients are smoke candidates, never calibrated weights."""

    head: float = 1.0
    head_geo: float = 1.0
    head_logsigma: float = 1.0
    head_centroid: float = 1.0
    head_shape: float = 1.0
    density: float = 1.0
    occupancy: float = 1.0
    occupancy_dice: float = 0.0
    j_coarse: float = 1.0
    j_centroid: float = 1.0
    j_logsigma: float = 1.0
    j_shape: float = 1.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite nonnegative number")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")


def _weights(value):
    if value is None:
        return RunCV2LossWeights()
    if isinstance(value, RunCV2LossWeights):
        return value
    if isinstance(value, Mapping):
        return RunCV2LossWeights(**value)
    raise TypeError("Expected RunCV2LossWeights or a mapping")


def _part_average(value, mask):
    """Average parts per word; retain empty words in the batch denominator."""
    mask = mask.to(value.dtype)
    return (value * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def _reduction(value, valid):
    """Expose separate per-word training and per-instance reporting domains."""
    per_sample = _part_average(value, valid)
    return {
        "mean": per_sample.mean(),
        "sum": (value * valid.to(value.dtype)).sum(),
        "count": valid.sum(),
        "per_sample": per_sample,
    }


class _MassGeometry(JietiLoss):
    """Moment/descriptor helpers on explicit soft ink, never normalized RGB."""

    def moments(self, mass):
        if mass.ndim != 4 or mass.shape[-2:] != (self.pool, self.pool):
            raise ValueError("Soft mass must have shape (B,K,pool,pool)")
        x = self.coord.view(1, 1, 1, self.pool)
        y = self.coord.view(1, 1, self.pool, 1)
        total = mass.sum((-1, -2))
        safe = total.clamp_min(self.eps)
        cx = (mass * x).sum((-1, -2)) / safe
        cy = (mass * y).sum((-1, -2)) / safe
        vx = (mass * (x - cx[..., None, None]).square()).sum((-1, -2))
        vy = (mass * (y - cy[..., None, None]).square()).sum((-1, -2))
        sx = (vx / safe + self.eps).sqrt()
        sy = (vy / safe + self.eps).sqrt()
        return total, cx, cy, sx, sy

    def descriptor(self, mass, stats):
        """8x8 normalized shape; reference moments stop gradient, ink does not.

        This follows the original J descriptor's reference-frame policy. The
        Head uses independent 28-grid slot ink, not the 224-grid Voronoi ink,
        so these descriptors are not advertised as numerically equal to old J.
        """
        _, cx, cy, sx, sy = stats
        bins = self.shape_bins
        axis = self.coord.view(1, 1, self.pool)
        bx = ((axis - cx.detach().unsqueeze(-1))
              / (sx.detach().unsqueeze(-1) + self.eps) + self.shape_range)
        by = ((axis - cy.detach().unsqueeze(-1))
              / (sy.detach().unsqueeze(-1) + self.eps) + self.shape_range)
        wx = self._tri(bx / (2 * self.shape_range) * bins)
        wy = self._tri(by / (2 * self.shape_range) * bins)
        tmp = torch.einsum("bkyx,bkxi->bkyi", mass, wx)
        hist = torch.einsum("bkyi,bkyj->bkji", tmp, wy)
        return hist / hist.sum((-1, -2), keepdim=True).clamp_min(self.eps)

    def errors(self, prediction, target, valid, normalizer_mass=None):
        """Compare scale axes, absolute/relative centroids and shape."""
        ps, ts = self.moments(prediction), self.moments(target)
        mp, px, py, psx, psy = ps
        mt, tx, ty, tsx, tsy = ts
        float_valid = valid.to(prediction.dtype)
        pm, tm = mp * float_valid, mt * float_valid
        pcx = (px * pm).sum(-1) / pm.sum(-1).clamp_min(self.eps)
        pcy = (py * pm).sum(-1) / pm.sum(-1).clamp_min(self.eps)
        tcx = (tx * tm).sum(-1) / tm.sum(-1).clamp_min(self.eps)
        tcy = (ty * tm).sum(-1) / tm.sum(-1).clamp_min(self.eps)
        dx = ((px - pcx[:, None]) - (tx - tcx[:, None]))
        dy = ((py - pcy[:, None]) - (ty - tcy[:, None]))
        # Exact zero at equality with finite derivative, unlike sqrt(d^2).
        distance = (dx.square() + dy.square() + self.eps).sqrt()
        distance = distance - math.sqrt(self.eps)
        x = self.coord.view(1, 1, 1, self.pool)
        y = self.coord.view(1, 1, self.pool, 1)
        if normalizer_mass is None:
            normalizer_mass = target * float_valid[..., None, None]
        normalizer = normalizer_mass.sum(1, keepdim=True)
        normalizer_total = normalizer.sum((1, 2, 3)).clamp_min(self.eps)
        ncx = (normalizer * x).sum((1, 2, 3)) / normalizer_total
        ncy = (normalizer * y).sum((1, 2, 3)) / normalizer_total
        variance = (normalizer * (
            (x - ncx[:, None, None, None]).square()
            + (y - ncy[:, None, None, None]).square()
        )).sum((1, 2, 3)) / normalizer_total
        sigma = (variance + self.eps).sqrt().detach()
        relation_valid = valid & (valid.sum(-1) >= 2)[:, None]
        return {
            "logsigma_x": (psx.log() - tsx.log()).abs(),
            "logsigma_y": (psy.log() - tsy.log()).abs(),
            "centroid_absolute": (
                (px - tx).abs() + (py - ty).abs()
            ) / max(self.pool - 1, 1),
            "centroid_relative": distance / sigma[:, None].clamp_min(self.eps),
            "shape": (
                self.descriptor(prediction, ps) - self.descriptor(target, ts)
            ).abs().sum((-1, -2)),
            "valid": valid,
            "relative_valid": relation_valid,
            "prediction_mass": mp,
            "target_mass": mt,
        }


class MatchedHeadGeometry(nn.Module):
    """Geometry on the same discrete assignment as InstanceHeadLoss.

    Prediction mass never gates validity. A near-empty predicted slot still
    incurs scale/position/shape losses and the original BCE/Dice/existence.
    """

    def __init__(self, pool=28, shape_bins=8, eps=1e-6):
        super().__init__()
        self.geometry = _MassGeometry(
            pool=pool, shape_bins=shape_bins, eps=eps
        )

    def forward(self, head, occupancy, has_jt, matching):
        probability = head["occupancy_logits"].float().sigmoid()
        target = occupancy.to(probability).detach()
        assignment = matching["assignment"].to(probability.device)
        target = target.gather(
            1, assignment[:, :, None, None].expand_as(target)
        )
        valid = matching["matched_slots"].to(probability.device).bool()
        valid = valid & has_jt.to(probability.device).bool()[:, None]
        empty_target = valid & (target.sum((-1, -2)) <= self.geometry.eps)
        valid = valid & ~empty_target
        with torch.autocast(
            device_type=probability.device.type, enabled=False
        ):
            result = self.geometry.errors(probability, target, valid)
        result["empty_target_count"] = empty_target.sum()
        result["assignment"] = assignment
        result["target"] = target
        result["probability"] = probability
        result["reductions"] = {
            name: _reduction(result[name], result[
                "relative_valid" if name == "centroid_relative" else "valid"
            ])
            for name in ("logsigma_x", "logsigma_y", "centroid_absolute",
                         "centroid_relative", "shape")
        }
        return result


class CoarseJietiLoss(nn.Module):
    """J on predicted/oracle density at exactly the same coarse coordinates.

    GT Voronoi ownership is a detached label map. The default legacy mass
    shield remains explicit and counted; density/occupancy still penalize an
    empty prediction. No RGB normalization or resize occurs inside this loss.
    Relative distance uses a zero-at-equality epsilon stabilization, so this
    is not claimed to equal the old 224-grid J bit for bit.
    """

    def __init__(self, pool=112, slots=4, shape_bins=8, eps=1e-6,
                 pred_mass_ratio=0.1):
        super().__init__()
        if not 0 <= pred_mass_ratio <= 1:
            raise ValueError("pred_mass_ratio must lie in [0,1]")
        self.slots = slots
        self.pred_mass_ratio = pred_mass_ratio
        self.geometry = _MassGeometry(
            pool=pool, shape_bins=shape_bins, eps=eps
        )

    def forward(self, prediction, target, voro, valid, has_jt, area=None):
        if prediction.shape != target.shape or prediction.shape[1] != 1:
            raise ValueError("Coarse density tensors must share (B,1,H,W)")
        prediction, target = prediction.float(), target.detach().float()
        pool = self.geometry.pool
        batch = prediction.shape[0]
        if prediction.shape[-2:] != (pool, pool):
            raise ValueError("Coarse geometry pool differs from density grid")
        if voro.shape != (batch, pool, pool):
            raise ValueError("Coarse Voronoi must have shape (B,pool,pool)")
        if valid.shape != (batch, self.slots):
            raise ValueError("Coarse valid parts must have shape (B,slots)")
        voro = voro.to(prediction.device).detach()
        if ((voro < 0) | (voro > self.slots)).any():
            raise ValueError("Voronoi labels outside 0..slots")
        part = torch.stack([
            voro == index + 1 for index in range(self.slots)
        ], 1).to(prediction.dtype)
        if area is not None:
            part = part * area.to(prediction).detach()
        pred_mass, target_mass = prediction * part, target * part
        gt_has_mass = target_mass.sum((-1, -2)) > self.geometry.eps
        base_valid = valid.to(prediction.device).bool() & gt_has_mass
        base_valid = base_valid & has_jt.to(prediction.device).bool()[:, None]
        pred_ok = pred_mass.sum((-1, -2)) >= (
            self.pred_mass_ratio * target_mass.sum((-1, -2))
        )
        effective = base_valid & pred_ok
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            # Preserve original J's GT whole-character scale normalizer even
            # when predicted-mass shielding excludes a part from the relation.
            details = self.geometry.errors(
                pred_mass, target_mass, effective, normalizer_mass=target_mass
            )
        details["shield_count"] = (base_valid & ~pred_ok).sum()
        details["base_valid_count"] = base_valid.sum()
        details["reductions"] = {
            name: _reduction(details[name], details[
                "relative_valid" if name == "centroid_relative" else "valid"
            ])
            for name in ("logsigma_x", "logsigma_y", "centroid_absolute",
                         "centroid_relative", "shape")
        }
        return details


def _lower(value):
    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError("Expected RGB batch")
    height, width = value.shape[-2:]
    if height == width:
        return value
    if height == 2 * width:
        return value[:, :, height // 2:]
    raise ValueError("Expected a square target or vertical two-row canvas")


def _masked_mean(error, area):
    area = area.to(error).expand_as(error)
    return (error * area).sum((1, 2, 3)) / area.sum((1, 2, 3)).clamp_min(1)


def _coarse_targets(batch, density):
    if "density_target" in batch and "occupancy_target" in batch:
        targets = {
            name: batch[name].to(density).detach()
            for name in ("density_target", "occupancy_target")
        }
        targets["valid_coarse"] = batch.get("valid_coarse")
    else:
        from data.runc_v2_dataset import coarse_targets_from_normalized

        target = _lower(batch["target"]).to(density).detach()
        valid = _lower(batch["valid"]).to(density).detach()
        targets = coarse_targets_from_normalized(target, valid=valid)
    if targets["valid_coarse"] is None:
        if "valid" in batch:
            targets["valid_coarse"] = F.adaptive_avg_pool2d(
                _lower(batch["valid"]).to(density).mean(1, keepdim=True),
                density.shape[-2:],
            )
        else:
            targets["valid_coarse"] = torch.ones_like(density)
    targets = {
        key: value.to(density).detach() for key, value in targets.items()
    }
    for name, value in targets.items():
        if value.shape != density.shape:
            raise ValueError(f"{name} differs from predicted density shape")
        if (not torch.isfinite(value).all()
                or ((value < 0) | (value > 1)).any()):
            raise ValueError(f"{name} must be finite in [0,1]")
    return targets


def runc_v2_losses(outputs, batch, weights=None, stage="IA",
                   head_criterion=None, head_geometry=None,
                   coarse_criterion=None):
    """Return phase-gated losses, leaf probes and sample/instance metrics.

    Matching is computed exactly once by the unchanged original Head loss and
    reused for geometry. ``IA`` needs only ``outputs['head']``. Frozen module
    selection belongs to the model/optimizer; this function only selects the
    objective. Callers must use equally sized DDP ranks or scale a local mean
    by ``world_size * local_count / global_count``. Across accumulation, scale
    by each microbatch count divided by the actual window count (including a
    partial final window), not unconditionally by a fixed accumulation factor.
    """
    stage = stage.upper().replace("-", "")
    stage = "IA" if stage in ("IA0", "IA1") else stage
    if stage not in ("IA", "IB", "IC"):
        raise ValueError("Run C v2 stage must be IA/IA0/IA1, IB or IC")
    weights = _weights(weights)
    head = outputs["head"]
    occupancy = batch["instance_occupancy"]
    head_criterion = head_criterion or InstanceHeadLoss(
        slots=head["occupancy_logits"].shape[1]
    )
    raw_head, matching = head_criterion(
        head, occupancy, batch["instance_valid"], batch["instance_classes"],
        batch["has_jt"],
    )
    head_geometry = head_geometry or MatchedHeadGeometry(
        pool=head["occupancy_logits"].shape[-1]
    ).to(head["occupancy_logits"].device)
    geometry = head_geometry(head, occupancy, batch["has_jt"], matching)
    reduced = geometry["reductions"]
    zero = raw_head * 0
    result = {
        "raw_head": raw_head,
        "raw_head_logsigma_x": reduced["logsigma_x"]["mean"],
        "raw_head_logsigma_y": reduced["logsigma_y"]["mean"],
        "raw_head_centroid": reduced["centroid_relative"]["mean"],
        "raw_head_centroid_absolute": reduced["centroid_absolute"]["mean"],
        "raw_head_shape": reduced["shape"]["mean"],
        "head_parts": matching,
        "head_geometry": geometry,
        "count_batch": raw_head.new_tensor(head["occupancy_logits"].shape[0]),
        "count_jt": batch["has_jt"].to(raw_head.device).sum(),
        "count_head_parts": geometry["valid"].sum(),
    }
    result["raw_head_logsigma"] = (
        result["raw_head_logsigma_x"] + result["raw_head_logsigma_y"]
    )
    result["raw_head_geo"] = (
        weights.head_logsigma * result["raw_head_logsigma"]
        + weights.head_centroid * result["raw_head_centroid"]
        + weights.head_shape * result["raw_head_shape"]
    )
    for name in ("density", "occupancy", "occupancy_bce", "occupancy_dice",
                 "j_coarse", "j_centroid", "j_logsigma", "j_shape"):
        result[f"raw_{name}"] = zero
    if stage in ("IB", "IC"):
        density = outputs["density"].float()
        logits = outputs["occupancy_logits"].float()
        if logits.shape != density.shape:
            raise ValueError("Density and coarse occupancy logits must align")
        targets = _coarse_targets(batch, density)
        oracle = targets["density_target"]
        target_occupancy = targets["occupancy_target"]
        area = targets["valid_coarse"]
        per_density = _masked_mean(
            F.smooth_l1_loss(density, oracle, reduction="none"), area
        )
        per_bce = _masked_mean(
            F.binary_cross_entropy_with_logits(
                logits, target_occupancy, reduction="none"
            ), area,
        )
        probability = logits.sigmoid()
        intersection = (probability * target_occupancy * area).sum((1, 2, 3))
        mass = ((probability + target_occupancy) * area).sum((1, 2, 3))
        per_dice = 1 - (2 * intersection + 1e-6) / (mass + 1e-6)
        per_dice = per_dice * (area.sum((1, 2, 3)) > 0).to(per_dice)
        per_occupancy = per_bce + weights.occupancy_dice * per_dice
        pool = density.shape[-1]
        voro = batch.get("coarse_voro")
        if voro is None:
            voro = F.interpolate(
                batch["voro"].to(density.device).float()[:, None],
                size=(pool, pool), mode="nearest",
            )[:, 0].long()
        else:
            voro = voro.to(density.device)
        valid_parts = batch.get("valid_parts_coarse", batch["valid_parts"])
        coarse_criterion = coarse_criterion or CoarseJietiLoss(
            pool=pool, slots=occupancy.shape[1]
        ).to(density.device)
        coarse = coarse_criterion(
            density, oracle, voro, valid_parts, batch["has_jt"], area=area,
        )
        cr = coarse["reductions"]
        j_centroid = cr["centroid_relative"]["mean"]
        j_logsigma = cr["logsigma_x"]["mean"] + cr["logsigma_y"]["mean"]
        j_shape = cr["shape"]["mean"]
        result.update({
            "raw_density": per_density.mean(),
            "raw_occupancy": per_occupancy.mean(),
            "raw_occupancy_bce": per_bce.mean(),
            "raw_occupancy_dice": per_dice.mean(),
            "raw_j_centroid": j_centroid,
            "raw_j_logsigma": j_logsigma,
            "raw_j_shape": j_shape,
            "raw_j_coarse": (weights.j_centroid * j_centroid
                             + weights.j_logsigma * j_logsigma
                             + weights.j_shape * j_shape),
            "coarse_geometry": coarse,
            "coarse_targets": targets,
            "coarse_voro": voro,
            "density_per_sample": per_density,
            "occupancy_per_sample": per_occupancy,
            "occupancy_foreground_fraction": _masked_mean(
                target_occupancy, area
            ),
            "count_j_coarse_parts": coarse["valid"].sum(),
            "count_j_coarse_shield": coarse["shield_count"],
            "count_coarse_partition_lost": (
                batch["valid_parts"].to(voro.device).bool()
                & ~torch.stack([
                    (voro == index + 1).any((-1, -2))
                    for index in range(occupancy.shape[1])
                ], 1)
            ).sum(),
        })
    head_total = (weights.head * raw_head
                  + weights.head_geo * result["raw_head_geo"])
    coarse_total = (weights.density * result["raw_density"]
                    + weights.occupancy * result["raw_occupancy"]
                    + weights.j_coarse * result["raw_j_coarse"])
    result["total"] = (head_total if stage == "IA" else coarse_total
                       if stage == "IB" else head_total + coarse_total)
    probe_names = ("head", "head_logsigma", "head_centroid", "head_shape")
    if stage == "IB":
        probe_names = ()
    if stage in ("IB", "IC"):
        probe_names += ("density", "occupancy_bce", "occupancy_dice",
                        "j_centroid", "j_logsigma", "j_shape")
    result["probe_terms"] = {
        name: result[f"raw_{name}"] for name in probe_names
    }
    result["stage"] = stage
    return result


def head_metric_sums(head, batch, result):
    """Matched-instance metrics and raw arrays for grouped/tail evaluation."""
    geometry, matching = result["head_geometry"], result["head_parts"]
    probability, target = geometry["probability"], geometry["target"]
    valid = geometry["valid"]
    intersection = (probability * target).sum((-1, -2))
    union = (probability + target - probability * target).sum((-1, -2))
    iou = intersection / union.clamp_min(1e-6)
    assignment = matching["assignment"]
    classes = batch["instance_classes"].to(assignment.device).gather(
        1, assignment
    )
    correct = head["class_logits"].argmax(-1).eq(classes)
    exists = head["existence_logits"] >= 0
    metrics = {
        "soft_iou": iou,
        "role_correct": correct.to(probability.dtype),
        "existence_recall": exists.to(probability.dtype),
        **{name: geometry[name] for name in (
            "logsigma_x", "logsigma_y", "centroid_absolute",
            "centroid_relative", "shape",
        )},
    }
    output = {"per_instance": metrics, "valid": valid,
              "relative_valid": geometry["relative_valid"]}
    for name, values in metrics.items():
        mask = (geometry["relative_valid"]
                if name == "centroid_relative" else valid)
        output[f"head_{name}_sum"] = (values * mask.to(values.dtype)).sum()
        output[f"head_{name}_count"] = mask.sum()
    output["head_instance_count"] = valid.sum()
    return output
