"""Reference-only, identity-initialized conditioning for Fontify."""

import torch
from torch import nn
from torch.nn import functional as F


class ReferenceConditioner(nn.Module):
    """Encode visible upper reference pixels; never access the query target."""

    def __init__(self, channels, mode="reference"):
        super().__init__()
        if mode not in ("reference", "constant"):
            raise ValueError(f"Unknown conditioning mode: {mode}")
        self.mode = mode
        layers = []
        previous = 4
        for width in (32, 64, 128):
            layers.extend(
                [
                    nn.Conv2d(previous, width, 3, stride=2, padding=1),
                    nn.GroupNorm(8, width),
                    nn.GELU(),
                ]
            )
            previous = width
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.heads = nn.ModuleList([nn.Linear(128, 2 * channels) for _ in range(3)])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, target, mask, patch_size):
        height, width = target.shape[-2:]
        visible = 1 - mask.reshape(
            target.shape[0], 1, height // patch_size, width // patch_size
        ).to(target.dtype)
        visible = F.interpolate(visible, size=(height, width), mode="nearest")
        visible = visible[:, :, : height // 2]
        mean = target.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = target.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        reference = target[:, :, : height // 2] * std + mean
        reference = reference * visible + (1 - visible)
        inputs = torch.cat((reference, visible), dim=1)
        if self.mode == "constant":
            inputs = torch.ones_like(inputs)
        code = self.encoder(inputs).flatten(1)
        return [head(code).chunk(2, dim=1) for head in self.heads]


def enable_conditioning(model, mode):
    """Attach after legacy initialization so zero heads stay zero."""
    if mode not in ("off", "reference", "constant"):
        raise ValueError(mode)
    if mode != "off":
        channels = model.segment_token_x.shape[-1]
        model.style_conditioner = ReferenceConditioner(channels, mode)
