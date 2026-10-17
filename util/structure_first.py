"""Run C: predicted structure/coarse image followed by masked BF rebuilding.

The optional modules are attached explicitly, after legacy initialization. The
frozen second pass sees the model's own coarse pixels, never target GT pixels.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from util.g_control import route_control_head

SCHEMA = "structure-first-bifa-v1"
MODULE_PREFIXES = ("structure_adapter.", "bf_refiner.")


class StructureCrossAttention(nn.Module):
    """One pre-normalized cross attention/FFN condition-interaction block."""

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, query, context):
        context = self.context_norm(context)
        update, _ = self.attention(
            self.query_norm(query), context, context, need_weights=False,
        )
        query = query + update
        return query + self.ffn(self.ffn_norm(query))


class StructureConditioner(nn.Module):
    """Cross-attend from F tokens to Q plus predicted spatial slot S tokens.

    S is the existence-weighted occupancy mixture of slot embeddings and class
    probabilities. The visible upper reference contributes an extra KV token.
    """

    def __init__(self, feature_width, query_width, classes, hidden_dim,
                 num_heads, num_layers):
        super().__init__()
        self.feature = nn.Sequential(
            nn.LayerNorm(feature_width), nn.Linear(feature_width, hidden_dim),
        )
        self.query = nn.Sequential(
            nn.LayerNorm(query_width), nn.Linear(query_width, hidden_dim),
        )
        self.slot = nn.Linear(query_width + classes, hidden_dim)
        self.blocks = nn.ModuleList([
            StructureCrossAttention(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.projection = nn.Linear(hidden_dim, feature_width)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, lower, reference, head):
        batch, height, width, _ = lower.shape
        query = head["query"]
        if query.shape[:2] != (batch, height * width):
            raise ValueError("Query and lower feature grids disagree")
        gates = (head["occupancy_logits"].sigmoid().flatten(2)
                 * head["existence_logits"].sigmoid().unsqueeze(-1))
        metadata = torch.cat(
            (head["slots"], head["class_logits"].softmax(-1)), dim=-1,
        )
        slot_tokens = self.slot(metadata)
        spatial_s = torch.bmm(gates.transpose(1, 2), slot_tokens)
        spatial_s = spatial_s / gates.shape[1]
        spatial_context = self.query(query) + spatial_s
        reference_token = self.feature(reference.mean((1, 2))).unsqueeze(1)
        context = torch.cat((spatial_context, reference_token), dim=1)
        feature_query = self.feature(lower).flatten(1, 2)
        for block in self.blocks:
            feature_query = block(feature_query, context)
        return self.projection(feature_query).reshape(batch, height, width, -1)


class BFRefiner(nn.Module):
    """New trainable features and RGB residual around a frozen MAE/decoder."""

    def __init__(self, feature_width, query_width, classes, hidden_dim,
                 num_heads, num_layers, rgb_hidden):
        super().__init__()
        self.adapter = StructureConditioner(
            feature_width, query_width, classes, hidden_dim,
            num_heads, num_layers,
        )
        self.rgb = nn.Sequential(
            nn.Conv2d(8, rgb_hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(rgb_hidden, 3, 3, padding=1),
        )
        nn.init.zeros_(self.rgb[-1].weight)
        nn.init.zeros_(self.rgb[-1].bias)

    def rgb_residual(self, coarse, reference, head, pixel_mask):
        gates = (head["occupancy_logits"].sigmoid()
                 * head["existence_logits"].sigmoid()[:, :, None, None])
        gate_map = gates.mean(1, keepdim=True)
        gate_map = F.interpolate(
            gate_map, coarse.shape[-2:], mode="bilinear", align_corners=False,
        )
        return self.rgb(torch.cat(
            (coarse, reference, gate_map, pixel_mask), dim=1,
        ))


def enable_structure_first(model, config=None):
    """Attach the Run C architecture with explicit JSON-compatible metadata."""
    if not getattr(model, "dual_view", False):
        raise ValueError("Structure-first requires Run B dual view")
    if hasattr(model, "structure_adapter") or hasattr(model, "bf_refiner"):
        raise ValueError("Structure-first modules are already attached")
    if hasattr(model, "g_control") or hasattr(model, "g_scale"):
        raise ValueError("Run C requires the original Run B Fix Edge model")
    supplied = dict(config or {})
    dimensions = {
        "feature_width": model.decoder_embed.in_features,
        "query_width": model.dual_head.width,
        "classes": model.dual_head.classes,
        "grid_size": model.dual_head.grid_size,
        "patch_size": model.patch_size,
    }
    defaults = {
        "schema": SCHEMA, "experiment": "runc", "hidden_dim": 256,
        "num_heads": 4, "num_layers": 1,
        "rgb_hidden": 32, "coarse_decoder": "full", "stage": "coarse",
        "auto_candidates": 32, "auto_threshold": 0.1, "auto_rounds": 1,
        **dimensions,
    }
    if set(supplied) - set(defaults):
        raise ValueError("Unknown structure-first configuration fields")
    config = {**defaults, **supplied}
    if config["schema"] != SCHEMA or config["experiment"] != "runc":
        raise ValueError("Unknown structure-first schema or experiment")
    if any(config[key] != value for key, value in dimensions.items()):
        raise ValueError("Structure-first metadata and dimensions disagree")
    for name in ("hidden_dim", "num_heads", "num_layers", "rgb_hidden",
                 "auto_candidates", "auto_rounds"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config["hidden_dim"] % config["num_heads"]:
        raise ValueError("hidden_dim must be divisible by num_heads")
    threshold = config["auto_threshold"]
    if (not isinstance(threshold, (int, float)) or not math.isfinite(threshold)
            or not 0 <= threshold <= 1):
        raise ValueError("auto_threshold must be finite and within [0, 1]")
    config["auto_threshold"] = float(threshold)
    if config["coarse_decoder"] not in ("full", "pred"):
        raise ValueError("coarse_decoder must be full or pred")
    arguments = {
        key: config[key]
        for key in ("feature_width", "query_width", "classes", "hidden_dim",
                    "num_heads", "num_layers")
    }
    device = next(model.parameters()).device
    model.structure_adapter = StructureConditioner(**arguments).to(device)
    model.bf_refiner = BFRefiner(
        **arguments, rgb_hidden=config["rgb_hidden"],
    ).to(device)
    model.structure_first_config = config
    model.structure_first_bifa = True
    set_structure_stage(model, config["stage"])
    return model


def _set_module_modes(model, stage):
    """Keep frozen dropout/statistics in eval after runner model.train()."""
    training = model.training
    for child in model.children():
        child.eval()
    if stage == "coarse":
        for name in ("dual_head", "dual_fusion", "structure_adapter",
                     "decoder_pred"):
            getattr(model, name).train(training)
        if model.structure_first_config["coarse_decoder"] == "full":
            model.decoder_embed.train(training)
    elif stage == "refine":
        model.bf_refiner.train(training)


def set_structure_stage(model, stage):
    """Select the independently trainable parameters for each stage."""
    if stage not in ("coarse", "refine", "infer"):
        raise ValueError("Stage must be coarse, refine, or infer")
    if not hasattr(model, "structure_first_config"):
        raise ValueError("Enable structure-first before selecting a stage")
    model.zero_grad(set_to_none=True)
    model.requires_grad_(False)
    if stage == "coarse":
        for name in ("dual_head", "dual_fusion", "structure_adapter",
                     "decoder_pred"):
            getattr(model, name).requires_grad_(True)
        if model.structure_first_config["coarse_decoder"] == "full":
            model.decoder_embed.requires_grad_(True)
    elif stage == "refine":
        model.bf_refiner.requires_grad_(True)
    model.structure_first_stage = stage
    model.structure_first_config["stage"] = stage
    _set_module_modes(model, stage)
    return [name for name, value in model.named_parameters()
            if value.requires_grad]


def _safe_deployment_target(tgts):
    """Remove lower GT before patch embedding, including NaN/Inf values."""
    half = tgts.shape[-2] // 2
    if tgts.ndim != 4 or tgts.shape[-2] != 2 * tgts.shape[-1]:
        raise ValueError("Targets must be vertically stacked squares")
    return torch.cat(
        (tgts[:, :, :half], torch.zeros_like(tgts[:, :, half:])), 2,
    )


def _coarse_prediction(model, imgs, tgts, has_jt):
    mask = model.full_target_mask(imgs)
    safe_target = _safe_deployment_target(tgts)
    with torch.no_grad():
        taps, quadrants = model.forward_encoder(
            imgs, safe_target, mask, return_quadrants=True,
        )
    features = torch.cat(taps, dim=-1)
    half = features.shape[1] // 2
    head = model.dual_head(quadrants, features[:, half:])
    routed = route_control_head(head, has_jt)
    residual = model.dual_fusion(routed["layout"])
    residual = residual + model.structure_adapter(
        features[:, half:], features[:, :half], routed,
    )
    prediction = model.forward_decoder(taps, lower_residual=residual)
    return prediction, head, mask


def normalize_bf_mask(mask, imgs, patch_size):
    """Accept the existing full-grid BF output or its lower half unchanged."""
    batch, _, height, width = imgs.shape
    gh, gw = height // patch_size, width // patch_size
    if mask is None:
        raise ValueError("Refine training requires an existing BF semantic mask")
    mask = mask.to(device=imgs.device)
    if mask.shape == (batch, gh * gw):
        mask = mask.reshape(batch, gh, gw)
    if mask.shape == (batch, gh, gw):
        if torch.any(mask[:, :gh // 2] != 0):
            raise ValueError("BF mask must keep the upper reference visible")
        mask = mask[:, gh // 2:]
    if mask.shape != (batch, gh // 2, gw):
        raise ValueError("BF mask must be a full or lower patch grid")
    if not torch.isfinite(mask).all() or torch.any((mask != 0) & (mask != 1)):
        raise ValueError("BF patch masks must contain only zero and one")
    return mask.bool()


def auto_structure_mask(model, prediction, head):
    """Rank predicted structure/ink patches; stable ties use raster index."""
    config = model.structure_first_config
    lower = prediction[:, :, prediction.shape[-2] // 2:]
    mean = lower.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = lower.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    foreground = (1 - (lower * std + mean).mean(1, keepdim=True)).clamp(0, 1)
    ink = F.avg_pool2d(foreground, model.patch_size).squeeze(1)
    gates = (head["occupancy_logits"].sigmoid()
             * head["existence_logits"].sigmoid()[:, :, None, None])
    score = ink * gates.amax(1)
    score = torch.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
    flat = score.flatten(1)
    # stable=True preserves the original ascending patch index on score ties.
    ranking = torch.argsort(flat, dim=1, descending=True, stable=True)
    selected = torch.zeros_like(flat, dtype=torch.bool)
    count = min(config["auto_candidates"], flat.shape[1])
    for index in range(flat.shape[0]):
        candidates = ranking[index, :count]
        active = candidates[
            flat[index, candidates] >= config["auto_threshold"]
        ]
        # Empty/background predictions still exercise one reproducible patch.
        if not active.numel():
            active = candidates[:1]
        selected[index, active] = True
    return selected.reshape_as(score)


def _refinement_pass(model, imgs, tgts, current, head, lower_mask):
    pixel_mask = lower_mask[:, None].repeat_interleave(
        model.patch_size, 2,
    ).repeat_interleave(model.patch_size, 3)
    half = current.shape[-2] // 2
    coarse_lower = current[:, :, half:]
    masked_lower = torch.where(
        pixel_mask, torch.zeros_like(coarse_lower), coarse_lower,
    )
    # GT lower pixels have already been replaced by P0; no arithmetic with GT.
    second_target = torch.cat((tgts[:, :, :half], masked_lower), dim=2)
    full_mask = torch.cat((torch.zeros_like(lower_mask), lower_mask), dim=1)
    with torch.no_grad():
        taps = model.forward_encoder(imgs, second_target, full_mask.flatten(1))
        features = torch.cat(taps, dim=-1)
        feature_half = features.shape[1] // 2
        base_decode = model.forward_decoder(taps)
    residual = model.bf_refiner.adapter(
        features[:, feature_half:], features[:, :feature_half], head,
    )
    # This frozen decoder transports gradients into the new adapter.
    changed_decode = model.forward_decoder(taps, lower_residual=residual)
    feature_update = changed_decode - base_decode
    rgb_update = model.bf_refiner.rgb_residual(
        masked_lower, tgts[:, :, :half], head, pixel_mask.to(current.dtype),
    )
    full_rgb_update = torch.cat(
        (torch.zeros_like(rgb_update), rgb_update), dim=2,
    )
    refined = current + feature_update + full_rgb_update
    masked_p0 = torch.cat((current[:, :, :half], masked_lower), dim=2)
    return refined, masked_p0, full_mask.flatten(1)


def structure_forward(model, imgs, tgts, stage, has_jt=None, bf_mask=None):
    """Compute P0, then rebuild masked model pixels using frozen stage I."""
    if not getattr(model, "structure_first_bifa", False):
        raise ValueError("Structure-first reconstruction has not been enabled")
    if stage not in ("coarse", "refine", "infer"):
        raise ValueError("Unknown structure-first execution stage")
    if stage == "refine" and model.structure_first_stage != "refine":
        raise ValueError("Select the frozen refine stage before BF training")
    _set_module_modes(model, stage)
    if stage == "coarse":
        p0, head, full_mask = _coarse_prediction(model, imgs, tgts, has_jt)
    else:
        with torch.no_grad():
            p0, head, full_mask = _coarse_prediction(model, imgs, tgts, None)
    batch, _, height, width = imgs.shape
    empty = torch.zeros(
        batch, height // (2 * model.patch_size), width // model.patch_size,
        device=imgs.device, dtype=torch.bool,
    )
    result = {
        "p0": p0, "head": head, "full_mask": full_mask,
        "mask": empty, "masked_p0": p0, "masks": [], "intermediates": [p0],
    }
    if stage == "coarse":
        return result
    current = p0
    rounds = (model.structure_first_config["auto_rounds"]
              if stage == "infer" else 1)
    for _ in range(rounds):
        if stage == "infer":
            lower_mask = auto_structure_mask(model, current, head)
        else:
            lower_mask = normalize_bf_mask(bf_mask, imgs, model.patch_size)
        current, masked, actual_full_mask = _refinement_pass(
            model, imgs, tgts, current, head, lower_mask,
        )
        result["masks"].append(lower_mask)
        result["intermediates"].append(current)
        result.update(
            mask=lower_mask, masked_p0=masked, full_mask=actual_full_mask,
        )
    result["p1"] = current
    return result
