"""Optional G-only spatial control while preserving the original Q projection."""

import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = "g-spatial-control-v1"
SLOT_PREFIXES = (
    "slot_queries",
    "slot_norm.",
    "token_norm.",
    "slot_attention.",
    "slot_ffn_norm.",
    "slot_ffn.",
    "existence.",
    "classifier.",
)


def route_control_head(head, has_jt=None):
    """Preserve predictions while blocking BF image gradients into the JT head."""
    if has_jt is None:
        return head
    result = {}
    for name, value in head.items():
        active = has_jt.to(device=value.device, dtype=torch.bool)
        if active.shape != (value.shape[0],):
            raise ValueError("JT availability must match the head batch")
        active = active.reshape(value.shape[0], *([1] * (value.ndim - 1)))
        result[name] = torch.where(active, value, value.detach())
    return result


def protected_prefix_exclusions(stage):
    """List only tensors permitted to change in each continuation stage."""
    if stage == 2:
        return ("g_control.",)
    if stage == 3:
        return ("g_control.", "decoder_embed.", "decoder_pred.")
    if stage == 4:
        return ("g_scale.",)
    if stage == 5:
        return ("g_control.", "g_scale.", "decoder_pred.") + tuple(
            "dual_head." + prefix for prefix in SLOT_PREFIXES
        )
    raise ValueError("Unsupported training stage")


class GSpatialControl(nn.Module):
    """Predict a bounded sampling displacement from G, then warp lower taps.

    Only the four-tap encoder features are warped. The original joint G/Q
    projection remains a separate, unmodified additive path. Zero flow gives
    exactly zero correction, including grid_sample identity-roundoff.
    """

    def __init__(self, g_width=276, hidden=128, max_displacement=4.0):
        super().__init__()
        if g_width < 1 or hidden < 16 or hidden % 16 or max_displacement <= 0:
            raise ValueError("Invalid G spatial control dimensions or flow bound")
        self.g_width = g_width
        self.max_displacement = float(max_displacement)
        self.norm = nn.LayerNorm(g_width)
        self.flow = nn.Sequential(
            nn.Conv2d(g_width, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1),
            nn.GroupNorm(8, hidden // 2),
            nn.GELU(),
            nn.Conv2d(hidden // 2, 2, 3, padding=1),
        )
        nn.init.zeros_(self.flow[-1].weight)
        nn.init.zeros_(self.flow[-1].bias)

    def forward(self, lower_features, layout):
        if lower_features.shape[:3] != layout.shape[:3]:
            raise ValueError("G and lower features must share B,H,W")
        g_field = self.norm(layout[..., : self.g_width])
        flow = self.flow(g_field.permute(0, 3, 1, 2)).tanh()
        flow = flow * self.max_displacement
        _, height, width, _ = lower_features.shape
        # CUDA grid_sample does not accept bf16. Preserve gradients through the
        # explicit float conversion, and return to the decoder feature dtype.
        with torch.autocast(device_type=lower_features.device.type, enabled=False):
            features = lower_features.float().permute(0, 3, 1, 2)
            x = 2 * (torch.arange(width, device=features.device) + 0.5) / width - 1
            y = 2 * (torch.arange(height, device=features.device) + 0.5) / height - 1
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            grid = torch.stack((xx, yy), -1)[None]
            offset = flow.float().permute(0, 2, 3, 1)
            offset = offset * offset.new_tensor([2 / width, 2 / height])
            identity = F.grid_sample(
                features,
                grid.expand(features.shape[0], -1, -1, -1),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            shifted = F.grid_sample(
                features,
                grid + offset,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            correction = (shifted - identity).permute(0, 2, 3, 1)
        return correction.to(lower_features.dtype)


def enable_g_control(model, config=None):
    """Attach only the explicitly versioned optional module before strict load."""
    config = dict(config or {"schema": SCHEMA})
    if config.get("schema") != SCHEMA or not getattr(model, "dual_view", False):
        raise ValueError("G control requires dual-view and a known schema")
    allowed = {"schema", "g_width", "hidden", "max_displacement"}
    if set(config) - allowed:
        raise ValueError("Unknown G control configuration fields")
    if hasattr(model, "g_control"):
        raise ValueError("G control already attached")
    config = {
        "schema": SCHEMA,
        "g_width": 276,
        "hidden": 128,
        "max_displacement": 4.0,
        **config,
    }
    model.g_control = GSpatialControl(
        **{key: value for key, value in config.items() if key != "schema"}
    )
    model.g_control_config = config
    return model.g_control


def configure_stage(model, stage, decoder_phase=0):
    """Select stage-specific modules while preserving the ViT and Q network."""
    if stage not in (2, 3, 4, 5) or decoder_phase not in (0, 1, 2):
        raise ValueError("Unsupported G training stage or decoder phase")
    if not hasattr(model, "g_control"):
        raise ValueError("G module must exist before selecting trainable parameters")
    model.requires_grad_(False)
    if stage == 5:
        if not hasattr(model, "g_scale"):
            raise ValueError("Stage 5 requires the trained scale branch")
        model.g_control.requires_grad_(True)
        model.g_scale.requires_grad_(True)
        model.decoder_pred.requires_grad_(True)
        for name, parameter in model.dual_head.named_parameters():
            if name.startswith(SLOT_PREFIXES):
                parameter.requires_grad_(True)
        return [name for name, value in model.named_parameters() if value.requires_grad]
    if stage == 4:
        if not hasattr(model, "g_scale"):
            raise ValueError("Stage 4 requires the scale branch")
        model.g_scale.requires_grad_(True)
        return [name for name, value in model.named_parameters() if value.requires_grad]
    model.g_control.requires_grad_(True)
    if stage == 3:
        if decoder_phase == 0:
            model.decoder_pred[-1].requires_grad_(True)
        else:
            model.decoder_pred.requires_grad_(True)
        if decoder_phase == 2:
            model.decoder_embed.requires_grad_(True)
    return [name for name, value in model.named_parameters() if value.requires_grad]
