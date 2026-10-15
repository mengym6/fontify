"""Explicit instance sigma prediction and differentiable G-field scaling."""

import math

import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = "g-instance-sigma-v1"


def map_geometry(probability):
    """Return normalized cell-center centroid and physical 448px sigma."""
    _, _, height, width = probability.shape
    x = 2 * (torch.arange(width, device=probability.device) + 0.5) / width - 1
    y = 2 * (torch.arange(height, device=probability.device) + 0.5) / height - 1
    mass = probability.sum((-1, -2)).clamp_min(1e-12)
    cx = (probability * x).sum((-1, -2)) / mass
    cy = (probability * y[:, None]).sum((-1, -2)) / mass
    vx = (probability * (x - cx[..., None, None]).square()).sum((-1, -2)) / mass
    vy = (probability * (y[:, None] - cy[..., None, None]).square()).sum(
        (-1, -2)
    ) / mass
    center = torch.stack((cx, cy), -1)
    sigma = torch.stack((vx, vy), -1).clamp_min(1e-12).sqrt() * 224
    return center, sigma


def affine_scale(probability, factors):
    """Scale B,K,H,W occupancy around each predicted soft centroid.

    O + sample(O, scaled_grid) - sample(O, identity_grid) is exactly O for
    factor=1 and still has a nonzero scale derivative there. No mass rescaling.
    """
    batch, slots, height, width = probability.shape
    if factors.shape != (batch, slots, 2):
        raise ValueError("Need horizontal and vertical factors for every instance")
    center, _ = map_geometry(probability)
    x = 2 * (torch.arange(width, device=probability.device) + 0.5) / width - 1
    y = 2 * (torch.arange(height, device=probability.device) + 0.5) / height - 1
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((xx, yy), -1)[None, None]
    center = center[:, :, None, None]
    changed_grid = grid + (grid - center) * (1 / factors[:, :, None, None] - 1)
    with torch.autocast(device_type=probability.device.type, enabled=False):
        values = probability.float().reshape(batch * slots, 1, height, width)
        changed = F.grid_sample(
            values,
            changed_grid.float().reshape(batch * slots, height, width, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).reshape_as(probability)
        identity = F.grid_sample(
            values,
            grid.expand(batch, slots, -1, -1, -1).reshape(
                batch * slots, height, width, 2
            ),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).reshape_as(probability)
    return (probability + (changed - identity)).clamp(0, 1)


class InstanceScaleAdapter(nn.Module):
    """Predict log-sigma from frozen slots and adapt only the new G pathway."""

    def __init__(self, width=256, classes=20, hidden=128, max_factor=2.0):
        super().__init__()
        if max_factor <= 1 or not math.isfinite(max_factor):
            raise ValueError("Scale bound must be finite and greater than one")
        self.max_log_factor = math.log(max_factor)
        self.predictor = nn.Sequential(
            nn.LayerNorm(width + classes + 4),
            nn.Linear(width + classes + 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        self.adapter = nn.Linear(2, width + classes, bias=False)
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)
        nn.init.zeros_(self.adapter.weight)

    def forward(self, head):
        occupancy = head["occupancy_logits"].float().sigmoid()
        center, base_sigma = map_geometry(occupancy)
        information = torch.cat((head["slots"], head["class_logits"].softmax(-1)), -1)
        features = torch.cat(
            (information.float(), center, base_sigma.clamp_min(1).log()), -1
        )
        delta = self.predictor(features).float().tanh() * self.max_log_factor
        scaled = affine_scale(occupancy, delta.exp())
        gates = scaled * head["existence_logits"].float().sigmoid()[..., None, None]
        base_gates = (
            occupancy * head["existence_logits"].float().sigmoid()[..., None, None]
        )
        change = torch.einsum(
            "bkhw,bkd->bhwd", gates - base_gates, information.float()
        ) + torch.einsum("bkhw,bkd->bhwd", gates, self.adapter(delta).float())
        g_width = information.shape[-1]
        original = head["layout"]
        field = original[..., :g_width].float() + change / occupancy.shape[1]
        query = original[..., g_width:]
        return {
            "layout": torch.cat((field.to(query.dtype), query), -1),
            "log_sigma": base_sigma.clamp_min(1).log() + delta,
            "factors": delta.exp(),
            "occupancy": scaled,
        }


def enable_g_scale(model, config=None):
    config = dict(config or {"schema": SCHEMA})
    if config.get("schema") != SCHEMA or not hasattr(model, "g_control"):
        raise ValueError("Scale prediction requires the G spatial module and schema")
    allowed = {"schema", "width", "classes", "hidden", "max_factor"}
    if set(config) - allowed or hasattr(model, "g_scale"):
        raise ValueError("Unknown scale configuration or duplicate module")
    config = {
        "schema": SCHEMA,
        "width": 256,
        "classes": 20,
        "hidden": 128,
        "max_factor": 2.0,
        **config,
    }
    model.g_scale = InstanceScaleAdapter(
        **{k: v for k, v in config.items() if k != "schema"}
    )
    model.g_scale_config = config
    return model.g_scale


def scale_supervision(model, head, target, targets, matching):
    """Supervise sigma in GT RGB-ink/Voronoi, never use labels in prediction.

    Per-word matched-instance mean followed by full-batch mean, like the
    original H reduction. BF and absent/invalid instances contribute zero.
    """
    jmod = model.dual_jieti_loss
    gt = target[:, :, target.shape[2] // 2 :].float()
    foreground = F.adaptive_avg_pool2d(
        jmod._soft_fg(gt)[:, None], (jmod.pool, jmod.pool)
    )[:, 0]
    part = torch.stack([(targets["voro"] == k + 1).float() for k in range(4)], 1)
    mass, _, _, sx, sy = jmod._part_stats(foreground, part)
    sigma = torch.stack((sx, sy), -1) * (448 / jmod.pool)
    indices = matching["assignment"]
    assigned = sigma.gather(1, indices[..., None].expand(-1, -1, 2)).clamp_min(1).log()
    valid = targets["valid_parts"].bool() & (mass > jmod.eps)
    valid = valid.gather(1, indices) & matching["matched_slots"]
    valid = valid & targets["has_jt"].bool()[:, None]
    errors = (head["scale_log_sigma"] - assigned.detach()).abs().sum(-1)
    per_sample = (errors * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    return per_sample.mean()
