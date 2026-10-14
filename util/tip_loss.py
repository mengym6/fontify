"""Masked, per-character pixel and signed-difference supervision.

The two vertically stacked characters are independent averaging units. Only
adjacent pixels inside the same character and supervision domain form a pair.
"""


import math

import torch

MEAN = (0.485, 0.456, 0.406)


STD = (0.229, 0.224, 0.225)


GRAY = (0.299, 0.587, 0.114)


EPSILON = 1e-6


STAT_NAMES = (
    "pixel", "dx", "dy", "loss", "pixels", "pairs_x", "pairs_y",
    "valid_words", "ink_fraction",
)


POOL_NAMES = ("dual", "bf_only", "jt_only")


def tip_weight_at_epoch(final, epoch, start=10, duration=8):
    """Use exactly the replaced edge's epoch-based linear ramp."""
    if not all(math.isfinite(value) for value in (final, epoch, start, duration)):
        raise ValueError("tip schedule values must be finite")
    if min(final, epoch, start, duration) < 0:
        raise ValueError("tip weight/epoch/start/duration must be nonnegative")
    if epoch < start:
        return 0.0
    if duration > 0 and epoch < start + duration:
        return final * (epoch - start) / duration
    return float(final)


def masked_tip_loss(prediction, target, domain):
    """Return FP32 loss and (B, 2) word statistics, retaining the full 2B mean.

    Inputs are ImageNet-normalized (B, 3, 2H, W) tensors and the actual expanded
    mask*valid. A grayscale pixel is valid only when all RGB channels are valid.
    Foreground threshold 0.5 is diagnostic only; it never weights the loss.
    Prediction is deliberately not clamped. No buffers or parameters are added.
    """
    if prediction.shape != target.shape or prediction.shape != domain.shape:
        raise ValueError("tip prediction, target and domain shapes must match")
    if prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("tip requires RGB BCHW tensors")
    batch, _, height, width = prediction.shape
    if batch < 1 or height < 2 or height % 2 or width < 1:
        raise ValueError("tip requires two equal, nonempty stacked characters")
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        mean = prediction.new_tensor(MEAN, dtype=torch.float32)[None, :, None, None]
        std = prediction.new_tensor(STD, dtype=torch.float32)[None, :, None, None]
        weights = prediction.new_tensor(GRAY, dtype=torch.float32)[None, :, None, None]
        pred = ((prediction.float() * std + mean) * weights).sum(1)
        truth = ((target.float() * std + mean) * weights).sum(1)
        shape = (batch, 2, height // 2, width)
        pred, truth = pred.reshape(shape), truth.reshape(shape)
        active = (domain > 0).all(1).reshape(shape)
        count = active.sum((-1, -2)).float()
        pixel = ((pred - truth).abs() * active).sum((-1, -2))
        pixel = pixel / (count + EPSILON)
        pair_x = active[..., 1:] & active[..., :-1]
        pair_y = active[..., 1:, :] & active[..., :-1, :]
        count_x = pair_x.sum((-1, -2)).float()
        count_y = pair_y.sum((-1, -2)).float()
        dx = ((pred[..., 1:] - pred[..., :-1])
              - (truth[..., 1:] - truth[..., :-1])).abs()
        dy = ((pred[..., 1:, :] - pred[..., :-1, :])
              - (truth[..., 1:, :] - truth[..., :-1, :])).abs()
        dx = (dx * pair_x).sum((-1, -2)) / (count_x + EPSILON)
        dy = (dy * pair_y).sum((-1, -2)) / (count_y + EPSILON)
        per_word = 0.5 * pixel + 0.25 * dx + 0.25 * dy
        ink = ((truth < 0.5) & active).sum((-1, -2)).float()
        stats = {
            "pixel": pixel, "dx": dx, "dy": dy, "loss": per_word,
            "pixels": count, "pairs_x": count_x, "pairs_y": count_y,
            "valid_words": (count > 0).float(),
            "ink_fraction": ink / (count + EPSILON),
        }
        return per_word.mean(), stats


def pool_stat_sums(stats, has_bf, has_jt):
    """Pack fixed-size per-pool sums/counts for consistent DDP collectives."""
    has_bf, has_jt = has_bf.bool(), has_jt.bool()
    if has_bf.shape != has_jt.shape or has_bf.shape != stats["loss"].shape[:1]:
        raise ValueError("tip pool flags must have shape (B,)")
    if not (has_bf | has_jt).all():
        raise ValueError("tip samples must belong to one of the three pools")
    rows = []
    for gate in (has_bf & has_jt, has_bf & ~has_jt, ~has_bf & has_jt):
        words = gate[:, None].expand_as(stats["loss"])
        values = [(stats[name].detach() * words).sum() for name in STAT_NAMES]
        rows.append(torch.stack([words.sum().float(), *values]))
    return torch.stack(rows)
