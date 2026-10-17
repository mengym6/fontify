"""Run C losses with separate coarse/refine targets and explicit reductions.

All defaults below are unit/smoke candidates, not calibrated training weights.
Smooth-L1 and edge average valid pixels per sample, then average the entire
batch. Empty local regions contribute zero without changing that denominator.
J and head supervision retain the original full-batch reduction and JT gates.
Gram sees the complete lower character, at 224 square, already normalized once.
"""

import math
from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class StructureLossWeights:
    """Uncalibrated coefficients, independent of old Run B weights."""

    lambda_low: float = 1.0
    lambda_j0: float = 1.0
    lambda_h: float = 1.0
    lambda_img1: float = 1.0
    lambda_bf: float = 1.0
    lambda_j1: float = 1.0
    gram: float = 1.0
    edge: float = 1.0
    edge_local: float = 1.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(
                    f"Run C coefficient {name} must be finite >= 0"
                )


@dataclass(frozen=True)
class LowpassConfig:
    """Loss-target Gaussian settings; the target never enters model inputs."""

    kernel_size: int = 7
    sigma: float = 1.5
    boundary: str = "reflect"

    def __post_init__(self):
        if (not isinstance(self.kernel_size, int)
                or isinstance(self.kernel_size, bool)
                or self.kernel_size < 1 or self.kernel_size % 2 != 1):
            raise ValueError(
                "Lowpass kernel_size must be a positive odd integer"
            )
        if not math.isfinite(float(self.sigma)) or self.sigma <= 0:
            raise ValueError("Lowpass sigma must be finite and positive")
        boundaries = ("reflect", "replicate", "constant", "circular")
        if self.boundary not in boundaries:
            raise ValueError("Unsupported lowpass boundary mode")


def _configuration(value, cls):
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        return cls(**value)
    raise TypeError(f"Expected {cls.__name__} or a mapping")


@torch.no_grad()
def lowpass_target(target, config=None):
    """Filter a detached target using a recorded, channelwise Gaussian."""
    config = _configuration(config, LowpassConfig)
    if target.ndim != 4:
        raise ValueError("Lowpass target must have shape (B,C,H,W)")
    target = target.detach().float()
    radius = config.kernel_size // 2
    if radius == 0:
        return target.clone()
    if (config.boundary == "reflect"
            and min(target.shape[-2:]) <= radius):
        raise ValueError("Reflect lowpass radius exceeds target dimensions")
    axis = torch.arange(
        -radius, radius + 1, device=target.device, dtype=target.dtype
    )
    kernel_1d = torch.exp(-axis.square() / (2 * config.sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel.expand(target.shape[1], 1, -1, -1).contiguous()
    padded = F.pad(target, (radius,) * 4, mode=config.boundary)
    return F.conv2d(padded, kernel, groups=target.shape[1])


def masked_per_sample(error, area):
    """Normalize by effective channel-pixels, retaining zero-mask samples."""
    if error.ndim != 4 or area.ndim != 4:
        raise ValueError("Masked reduction requires (B,C,H,W) tensors")
    try:
        area = area.to(device=error.device, dtype=error.dtype).expand_as(error)
    except RuntimeError as exc:
        raise ValueError("Mask cannot broadcast to error shape") from exc
    if not torch.isfinite(area).all() or ((area < 0) | (area > 1)).any():
        raise ValueError("Effective masks must be finite and within [0,1]")
    denominator = area.sum((1, 2, 3))
    safe_denominator = torch.where(
        denominator > 0, denominator, torch.ones_like(denominator)
    )
    value = (error * area).sum((1, 2, 3)) / safe_denominator
    return value, denominator


def _lower(images):
    if images.ndim != 4 or images.shape[-2] != 2 * images.shape[-1]:
        raise ValueError("Run C expects a vertical two-character RGB canvas")
    return images[:, :, images.shape[-2] // 2:]


def _local_support(batch, prediction):
    """Expand the actual MAE patch mask rather than the smaller BF polygon."""
    if "bf_pixel_mask" in batch:
        mask = batch["bf_pixel_mask"].to(prediction)
        if mask.ndim == 3:
            mask = mask[:, None]
    elif "bf_patch_mask" in batch:
        mask = batch["bf_patch_mask"].to(prediction)
        if mask.ndim != 3:
            raise ValueError("bf_patch_mask must have shape (B,h,w)")
        if (prediction.shape[-2] % mask.shape[-2]
                or prediction.shape[-1] % mask.shape[-1]):
            raise ValueError("BF patch mask does not align to the output")
        mask = mask.repeat_interleave(
            prediction.shape[-2] // mask.shape[-2], -2
        ).repeat_interleave(prediction.shape[-1] // mask.shape[-1], -1)
        mask = mask[:, None]
    else:
        raise ValueError(
            "BF refinement requires the actual BF patch/pixel mask"
        )
    if (mask.shape[0] != prediction.shape[0]
            or mask.shape[-2:] != prediction.shape[-2:]):
        raise ValueError(
            "BF pixel mask differs from the lower output geometry"
        )
    if mask.shape[1] not in (1, prediction.shape[1]):
        raise ValueError("BF pixel mask must have one or RGB channels")
    if not torch.isfinite(mask).all() or ((mask != 0) & (mask != 1)).any():
        raise ValueError("Actual BF patch support must be a binary mask")
    return mask


def structure_losses(model, outputs, batch, weights=None, stage="coarse",
                     lowpass=None):
    """Compute Run C terms; GT is used only in this supervision function.

    ``p0``/``p1`` are full two-row predictions. Refinement rejects all non-BF
    samples. Its J gate is ``has_jt & has_bf``; head matching only monitors.
    Local edge computes both complete lower edge maps before applying support,
    so no artificial edge from multiplying an image by its mask is introduced.
    """
    if stage not in ("coarse", "refine"):
        raise ValueError("Run C loss stage must be coarse or refine")
    weights = _configuration(weights, StructureLossWeights)
    prediction = outputs["p0" if stage == "coarse" else "p1"]
    target = batch["target"].to(prediction).detach()
    valid = batch["valid"].to(prediction).detach()
    if target.shape != prediction.shape or valid.shape != prediction.shape:
        raise ValueError(
            "Run C prediction, GT and valid must have equal shapes"
        )
    pred_lower, tgt_lower, valid_lower = map(
        _lower, (prediction, target, valid)
    )
    batch_size = prediction.shape[0]
    has_jt = batch["has_jt"].to(device=prediction.device, dtype=torch.bool)
    has_bf = batch["has_bf"].to(device=prediction.device, dtype=torch.bool)
    if has_jt.shape != (batch_size,) or has_bf.shape != (batch_size,):
        raise ValueError("has_jt/has_bf must each have shape (B,)")
    if stage == "refine" and not bool(has_bf.all()):
        raise ValueError("JT-only/non-BF samples cannot enter BF refinement")
    gate = has_jt if stage == "coarse" else has_jt & has_bf
    jieti, jparts, shield = model.dual_jieti_loss(
        prediction, target, batch["voro"], batch["valid_parts"], gate
    )
    head = outputs["head"]
    head_args = (
        head, batch["instance_occupancy"], batch["instance_valid"],
        batch["instance_classes"], gate,
    )
    if stage == "coarse":
        head_loss, head_parts = model.dual_head_loss(*head_args)
    else:
        with torch.no_grad():
            head_loss, head_parts = model.dual_head_loss(*head_args)
    zero = prediction.sum() * 0
    result = {name: zero for name in (
        "low", "j0", "img1", "bf_local", "j1", "raw_low", "raw_j0",
        "raw_img_smooth", "raw_img_gram", "raw_img_edge", "raw_bf_smooth",
        "raw_bf_edge", "raw_j1",
    )}
    result.update({
        "head": head_loss,
        "raw_head": head_loss,
        "head_parts": head_parts,
        "j_centroid": jparts["centroid"],
        "j_logsigma": jparts["logsigma"],
        "j_shape": jparts["shape"],
        "j_shield": shield,
        "count_batch": prediction.new_tensor(batch_size),
        "count_jt": has_jt.sum(),
        "count_bf": has_bf.sum(),
        "count_dual": (has_jt & has_bf).sum(),
        "count_jt_parts": (
            batch["valid_parts"].to(prediction.device).bool() & gate[:, None]
        ).sum(),
        "count_local_nonempty": prediction.new_zeros(()),
        "count_local_empty": prediction.new_zeros(()),
        "count_local_channel_pixels": prediction.new_zeros(()),
        "local_channel_pixels_per_sample": prediction.new_zeros(batch_size),
    })
    if stage == "coarse":
        filtered = lowpass_target(tgt_lower, lowpass)
        low, _ = masked_per_sample(
            F.smooth_l1_loss(pred_lower.float(), filtered, reduction="none"),
            valid_lower,
        )
        result.update({
            "low": low.mean(), "raw_low": low.mean(),
            "j0": jieti, "raw_j0": jieti,
            "lowpass_target": filtered,
        })
        result["total"] = (
            weights.lambda_low * result["low"]
            + weights.lambda_j0 * jieti + weights.lambda_h * head_loss
        )
    else:
        smooth_error = F.smooth_l1_loss(
            pred_lower.float(), tgt_lower.float(), reduction="none"
        )
        smooth, _ = masked_per_sample(smooth_error, valid_lower)
        # The current VGG forward accepts ImageNet-normalized tensors as-is.
        # It must not receive either a second normalization or a masked image.
        gram = model.vgg_loss(
            F.interpolate(pred_lower.float(), (224, 224), mode="bilinear",
                          align_corners=False, antialias=True),
            F.interpolate(tgt_lower.float(), (224, 224), mode="bilinear",
                          align_corners=False, antialias=True),
            per_sample=True,
        ).mean()
        edge_error = (
            model.improved_edge_detection(pred_lower.float())
            - model.improved_edge_detection(tgt_lower.float())
        ).abs()
        # The original edge map is one channel. Effective RGB validity is
        # averaged, preserving fractional/channel validity without mutation.
        edge_valid = valid_lower.mean(1, keepdim=True)
        edge, _ = masked_per_sample(edge_error, edge_valid)
        local_mask = _local_support(batch, pred_lower)
        local_area = local_mask * valid_lower
        local_smooth, local_denominator = masked_per_sample(
            smooth_error, local_area
        )
        local_edge, _ = masked_per_sample(
            edge_error, (local_mask * valid_lower).mean(1, keepdim=True)
        )
        img1 = smooth.mean() + weights.gram * gram + weights.edge * edge.mean()
        bf_local = local_smooth.mean() + weights.edge_local * local_edge.mean()
        result.update({
            "img1": img1, "bf_local": bf_local,
            "j1": jieti, "raw_j1": jieti,
            "raw_img_smooth": smooth.mean(), "raw_img_gram": gram,
            "raw_img_edge": edge.mean(),
            "raw_bf_smooth": local_smooth.mean(),
            "raw_bf_edge": local_edge.mean(),
            "count_local_nonempty": (local_denominator > 0).sum(),
            "count_local_empty": (local_denominator == 0).sum(),
            "count_local_channel_pixels": local_denominator.sum(),
            "local_channel_pixels_per_sample": local_denominator,
        })
        result["total"] = (
            weights.lambda_img1 * img1 + weights.lambda_bf * bf_local
            + weights.lambda_j1 * jieti
        )
    return result
