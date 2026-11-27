"""GT masks define query identity; outputs never require GT-side re-detection."""

from copy import deepcopy
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from yolo.pairing.model import YoloFeatures, PairRegions, load_pair_checkpoint


class GTCorrespondence(nn.Module):
    def __init__(self, encoder, width=128, layers=3, heads=4, token_grid=12):
        super().__init__()
        self.encoder = encoder
        self.config = dict(
            width=width, layers=layers, heads=heads, token_grid=token_grid
        )
        self.project = nn.Conv2d(encoder.channels, width, 1)
        self.position = nn.Linear(2, width)
        self.side = nn.Parameter(torch.randn(2, width) * 0.02)
        self.group_embedding = nn.Embedding(2, width)
        self.geometry = nn.Sequential(
            nn.Linear(5, width), nn.GELU(), nn.Linear(width, width)
        )
        self.query_norm = nn.LayerNorm(width)
        layer = nn.TransformerDecoderLayer(
            width, heads, width * 4, dropout=0.0, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, layers, nn.LayerNorm(width))
        self.coefficients = nn.Linear(width, encoder.mask_channels)
        self.mask_bias = nn.Linear(width, 1)
        self.matchability = nn.Linear(width, 1)
        self.architecture = None
        self.init_sha256 = None
        self.encoder_frozen = False

    @classmethod
    def from_init(cls, checkpoint):
        saved = torch.load(checkpoint, map_location="cpu")
        if saved.get("schema") == "calli-pair-regions-v1":
            old = load_pair_checkpoint(checkpoint)
        else:
            old = PairRegions.from_baseline(checkpoint)
        config = {k: old.config[k] for k in ("width", "layers", "heads", "token_grid")}
        result = cls(deepcopy(old.encoder), **config)
        result.architecture = deepcopy(old.architecture)
        # A YOLO init transfers only trained encoder/prototypes, not random v8 heads.
        if saved.get("schema") == "calli-pair-regions-v1":
            for name in ("project", "position", "decoder"):
                getattr(result, name).load_state_dict(getattr(old, name).state_dict())
            with torch.no_grad():
                result.side.copy_(old.side)
                n = old.encoder.mask_channels
                result.coefficients.weight.copy_(old.coefficients.weight[n:])
                result.coefficients.bias.copy_(old.coefficients.bias[n:])
                result.mask_bias.weight.copy_(old.mask_bias.weight[1:])
                result.mask_bias.bias.copy_(old.mask_bias.bias[1:])
                result.matchability.weight.copy_(old.presence.weight[1:])
                result.matchability.bias.copy_(old.presence.bias[1:])
        with open(checkpoint, "rb") as f:
            result.init_sha256 = hashlib.sha256(f.read()).hexdigest()
        return result

    def freeze_encoder(self, frozen=True):
        self.encoder_frozen = frozen
        self.encoder.requires_grad_(not frozen)
        if frozen:
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def forward(self, images, gt_masks, groups):
        if images.ndim != 5 or images.shape[1:3] != (2, 3):
            raise ValueError("Expected image pairs [B,2,3,H,W]")
        b, _, _, h, w = images.shape
        if h % 32 or w % 32 or len(gt_masks) != b or len(groups) != b:
            raise ValueError("Invalid image dimensions or GT batch")
        feat, proto = self.encoder(images.reshape(b * 2, 3, h, w))
        feature = self.project(feat).reshape(
            b, 2, self.config["width"], *feat.shape[-2:]
        )
        k = self.config["token_grid"]
        memory = (
            F.adaptive_avg_pool2d(feature.flatten(0, 1), (k, k))
            .flatten(2)
            .transpose(1, 2)
            .reshape(b, 2, k * k, -1)
        )
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, k, device=images.device),
            torch.linspace(-1, 1, k, device=images.device),
            indexing="ij",
        )
        memory = (
            memory
            + self.position(torch.stack((xx, yy), -1).reshape(k * k, 2))[None, None]
            + self.side[None, :, None]
        )
        proto = proto.reshape(b, 2, self.encoder.mask_channels, *proto.shape[-2:])
        results = []
        for i in range(b):
            masks = gt_masks[i].to(images.device).float()
            group = groups[i].to(images.device)
            n = len(masks)
            if (
                masks.ndim != 3
                or group.shape != (n,)
                or ((group < 0) | (group > 1)).any()
            ):
                raise ValueError("Each GT mask needs a BF/JT group")
            if not n:
                results.append(
                    {
                        "mask_logits": proto.new_empty(0, *proto.shape[-2:]),
                        "match_logits": proto.new_empty(0),
                    }
                )
                continue
            if (masks.flatten(1).sum(1) <= 0).any():
                raise ValueError("GT anchor mask cannot be empty")
            weights = F.interpolate(masks[:, None], feature.shape[-2:], mode="area")[
                :, 0
            ].flatten(1)
            pooled = (
                weights
                @ feature[i, 0].flatten(1).T
                / weights.sum(1, keepdim=True).clamp_min(1e-6)
            )
            mh, mw = masks.shape[-2:]
            gy, gx = torch.meshgrid(
                torch.linspace(0, 1, mh, device=images.device),
                torch.linspace(0, 1, mw, device=images.device),
                indexing="ij",
            )
            area = masks.sum((1, 2)).clamp_min(1e-6)
            cx = (masks * gx).sum((1, 2)) / area
            cy = (masks * gy).sum((1, 2)) / area
            sx = ((masks * (gx - cx[:, None, None]).square()).sum((1, 2)) / area).sqrt()
            sy = ((masks * (gy - cy[:, None, None]).square()).sum((1, 2)) / area).sqrt()
            geom = torch.stack((cx, cy, area / (mh * mw), sx, sy), 1)
            q = self.query_norm(
                pooled + self.geometry(geom) + self.group_embedding(group)
            )
            decoded = self.decoder(q[None], memory[i].flatten(0, 1)[None])[0]
            masks_out = (
                torch.einsum("nc,chw->nhw", self.coefficients(decoded), proto[i, 1])
                + self.mask_bias(decoded)[:, :, None]
            )
            results.append(
                {
                    "mask_logits": masks_out,
                    "match_logits": self.matchability(decoded).squeeze(-1),
                }
            )
        return results


def load_checkpoint(path, device="cpu"):
    from ultralytics.nn.tasks import SegmentationModel

    saved = torch.load(path, map_location="cpu")
    if saved.get("schema") != "calli-gt2gen-v9":
        raise ValueError("Expected a v9 checkpoint")
    base = SegmentationModel(cfg=saved["architecture"], ch=3, nc=70, verbose=False)
    model = GTCorrespondence(YoloFeatures(base), **saved["config"])
    model.load_state_dict(saved["model"], strict=True)
    model.architecture = saved["architecture"]
    model.init_sha256 = saved["init_sha256"]
    return model.to(device)
