"""Transfer YOLO26 features and prototypes, jointly decode paired region queries."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class YoloFeatures(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.layers = nn.ModuleList(list(model.model)[:-1])
        head = model.model[-1]
        self.outputs = list(head.f)
        self.proto = deepcopy(head.proto)
        # The 70-class auxiliary training head is not used for correspondence.
        self.proto.semseg = nn.Identity()
        self.channels = self.proto.cv1.conv.in_channels
        self.mask_channels = head.nm
        self.requires_grad_(True)

    def forward(self, images):
        saved, x = [], images
        for module in self.layers:
            if module.f != -1:
                x = (
                    saved[module.f]
                    if isinstance(module.f, int)
                    else [x if j == -1 else saved[j] for j in module.f]
                )
            x = module(x)
            saved.append(x)
        features = [saved[i] for i in self.outputs]
        return features[0], self.proto(features, return_semantic=False)


class PairRegions(nn.Module):
    """images [B,2,3,H,W] -> masks [B,Q,2,H/4,W/4], presence [B,Q,2].

    Each query owns both masks. No detection, NMS or post-hoc image matching.
    First bf_queries slots are BF, remaining jt_queries slots are JT.
    """

    def __init__(
        self,
        encoder,
        width=128,
        bf_queries=64,
        jt_queries=16,
        layers=3,
        heads=4,
        token_grid=12,
    ):
        super().__init__()
        if min(bf_queries, jt_queries, token_grid) < 1 or width % heads:
            raise ValueError("Invalid query counts, token grid or attention width")
        self.encoder = encoder
        self.config = dict(
            width=width,
            bf_queries=bf_queries,
            jt_queries=jt_queries,
            layers=layers,
            heads=heads,
            token_grid=token_grid,
        )
        self.project = nn.Conv2d(encoder.channels, width, 1)
        self.position = nn.Linear(2, width)
        self.side = nn.Parameter(torch.randn(2, width) * 0.02)
        self.queries = nn.Embedding(bf_queries + jt_queries, width)
        block = nn.TransformerDecoderLayer(
            width, heads, width * 4, dropout=0.0, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(block, layers, nn.LayerNorm(width))
        self.coefficients = nn.Linear(width, 2 * encoder.mask_channels)
        self.mask_bias = nn.Linear(width, 2)
        self.presence = nn.Linear(width, 2)
        self.register_buffer(
            "groups", torch.tensor([0] * bf_queries + [1] * jt_queries)
        )
        self.architecture = None
        self.baseline_sha256 = None
        self.encoder_frozen = False

    @classmethod
    def from_baseline(cls, checkpoint, **config):
        import hashlib
        import ultralytics
        from ultralytics import YOLO

        if ultralytics.__version__ != "8.4.218":
            raise RuntimeError("This adapter requires ultralytics==8.4.218")
        base = YOLO(str(checkpoint)).model.float()
        result = cls(YoloFeatures(base), **config)
        result.architecture = deepcopy(base.yaml)
        with open(checkpoint, "rb") as f:
            result.baseline_sha256 = (
                hashlib.file_digest(f, "sha256").hexdigest()
                if hasattr(hashlib, "file_digest")
                else hashlib.sha256(f.read()).hexdigest()
            )
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

    def forward(self, images):
        if images.ndim != 5 or images.shape[1:3] != (2, 3):
            raise ValueError("Expected [batch,2,3,height,width]")
        b, _, _, h, w = images.shape
        if h % 32 or w % 32:
            raise ValueError("Image dimensions must be divisible by 32")
        features, proto = self.encoder(images.reshape(b * 2, 3, h, w))
        k = self.config["token_grid"]
        memory = F.adaptive_avg_pool2d(self.project(features), (k, k))
        memory = memory.flatten(2).transpose(1, 2).reshape(b, 2, k * k, -1)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, k, device=images.device),
            torch.linspace(-1, 1, k, device=images.device),
            indexing="ij",
        )
        xy = torch.stack((xx, yy), -1).reshape(k * k, 2).to(memory.dtype)
        memory = memory + self.position(xy)[None, None] + self.side[None, :, None]
        q = self.queries.weight[None].expand(b, -1, -1)
        decoded = self.decoder(q, memory.flatten(1, 2))
        coefficients = self.coefficients(decoded).reshape(
            b, -1, 2, self.encoder.mask_channels
        )
        proto = proto.reshape(b, 2, self.encoder.mask_channels, *proto.shape[-2:])
        masks = torch.einsum("bqsc,bschw->bqshw", coefficients, proto)
        masks = masks + self.mask_bias(decoded)[..., None, None]
        return {
            "masks": masks,
            "presence": self.presence(decoded),
            "groups": self.groups,
        }


def load_pair_checkpoint(path, device="cpu"):
    from ultralytics.nn.tasks import SegmentationModel

    saved = torch.load(path, map_location="cpu")
    if saved.get("schema") != "calli-pair-regions-v1":
        raise ValueError("Not a joint-region checkpoint")
    base = SegmentationModel(cfg=saved["architecture"], ch=3, nc=70, verbose=False)
    model = PairRegions(YoloFeatures(base), **saved["config"])
    model.load_state_dict(saved["model"], strict=True)
    model.architecture = saved["architecture"]
    model.baseline_sha256 = saved["baseline_sha256"]
    return model.to(device)
