"""Run C v2: predicted JT structure queries and a dedicated density renderer.

The opt-in path starts from a legacy Fontify checkpoint. Ground-truth target
pixels in the lower half are removed before patch embedding. Four predicted
instance slots are pooled as an unordered set; only their resulting spatial
structure field supplies the renderer queries. Content/reference features and
the JT spatial representation Q supply keys and values, never an RGB shortcut.
"""

import hashlib
import math
from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from util.dual_view import DualViewHead
from util.structure_first import (
    StructureCrossAttention,
    auto_structure_mask,
    normalize_bf_mask,
)

SCHEMA = "runc-v2-jt-structure-v1"
TAP_INDICES = (2, 5, 8, 11)
STAGES = ("ia0", "ia1", "ib", "ic", "infer", "refine")
MODULE_PREFIXES = ("dual_head.", "runc_v2_coarse.", "bf_refiner.")
STRUCTURE_KEYS = ("occupancy_logits", "existence_logits", "class_logits")


class StructureQueryEncoder(nn.Module):
    """Encode a permutation-invariant, existence-weighted spatial slot set."""

    def __init__(self, classes, hidden_dim, grid_size):
        super().__init__()
        self.classes = classes
        self.grid_size = grid_size
        self.projection = nn.Sequential(
            nn.Linear(classes + 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.row_embedding = nn.Parameter(torch.empty(grid_size, hidden_dim))
        self.column_embedding = nn.Parameter(
            torch.empty(grid_size, hidden_dim),
        )
        nn.init.trunc_normal_(self.row_embedding, std=0.02)
        nn.init.trunc_normal_(self.column_embedding, std=0.02)

    def forward(self, head):
        occupancy = head["occupancy_logits"].sigmoid()
        existence = head["existence_logits"].sigmoid()
        classes = head["class_logits"].softmax(-1)
        if occupancy.ndim != 4 or occupancy.shape[-2:] != (
                self.grid_size, self.grid_size):
            raise ValueError("Predicted structure occupancy has a wrong grid")
        if existence.shape != occupancy.shape[:2]:
            raise ValueError("Existence and occupancy dimensions disagree")
        if classes.shape != (*occupancy.shape[:2], self.classes):
            raise ValueError("Role probabilities and occupancy slots disagree")
        gates = occupancy * existence[:, :, None, None]
        count = gates.shape[1]
        # Shared encoding and reductions cannot assign meaning to slot ID.
        role_field = torch.einsum("bkhw,bkc->bhwc", gates, classes) / count
        mass = gates.sum(1).unsqueeze(-1) / count
        second_moment = gates.square().sum(1).unsqueeze(-1) / count
        field = torch.cat((role_field, mass, second_moment), dim=-1)
        position = (self.row_embedding[:, None]
                    + self.column_embedding[None, :])
        return (self.projection(field) + position).flatten(1, 2)


class StructureDensityRenderer(nn.Module):
    """S queries attend to F_full/Q/reference KV, then predict D0 and O0."""

    def __init__(self, feature_width, query_width, classes, hidden_dim=256,
                 num_heads=4, num_layers=1, grid_size=28, density_size=112):
        super().__init__()
        self.grid_size = grid_size
        self.density_size = density_size
        self.structure_query = StructureQueryEncoder(
            classes, hidden_dim, grid_size,
        )
        self.feature = nn.Sequential(
            nn.LayerNorm(feature_width), nn.Linear(feature_width, hidden_dim),
        )
        self.reference = nn.Sequential(
            nn.LayerNorm(feature_width), nn.Linear(feature_width, hidden_dim),
        )
        self.spatial_query = nn.Sequential(
            nn.LayerNorm(query_width), nn.Linear(query_width, hidden_dim),
        )
        self.source_embedding = nn.Parameter(torch.empty(3, hidden_dim))
        nn.init.trunc_normal_(self.source_embedding, std=0.02)
        self.blocks = nn.ModuleList([
            StructureCrossAttention(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        width1 = max(hidden_dim // 2, 8)
        width2 = max(hidden_dim // 4, 8)
        self.spatial = nn.Sequential(
            nn.Conv2d(hidden_dim, width1, 3, padding=1), nn.GELU(),
            nn.Conv2d(width1, width2, 3, padding=1), nn.GELU(),
        )
        self.density = nn.Conv2d(width2, 1, 1)
        self.occupancy = nn.Conv2d(width2, 1, 1)

    def forward(self, lower, reference, head, structure=None):
        batch, height, width, _ = lower.shape
        if (height, width) != (self.grid_size, self.grid_size):
            raise ValueError("Renderer F_full does not match its target grid")
        if reference.shape != lower.shape:
            raise ValueError("Reference and target feature grids must match")
        if head["query"].shape[:2] != (batch, height * width):
            raise ValueError("JT Q and target feature grids disagree")
        # Only S supplies queries. Overrides never replace actual encoder Q/F.
        query = self.structure_query(head if structure is None else structure)
        position = (self.structure_query.row_embedding[:, None]
                    + self.structure_query.column_embedding[None, :])
        position = position.flatten(0, 1)[None]
        context = torch.cat((
            self.feature(lower).flatten(1, 2) + position
            + self.source_embedding[0],
            self.spatial_query(head["query"]) + position
            + self.source_embedding[1],
            self.reference(reference).flatten(1, 2) + position
            + self.source_embedding[2],
        ), dim=1)
        for block in self.blocks:
            query = block(query, context)
        spatial = query.transpose(1, 2).reshape(batch, -1, height, width)
        # Production 28 -> 56 -> 112; tiny CPU fixtures retain size semantics.
        middle_size = max(self.density_size // 2, 1)
        spatial = F.interpolate(
            spatial, (middle_size, middle_size), mode="bilinear",
            align_corners=False,
        )
        spatial = self.spatial[:2](spatial)
        spatial = F.interpolate(
            spatial, (self.density_size, self.density_size), mode="bilinear",
            align_corners=False,
        )
        spatial = self.spatial[2:](spatial)
        return {
            "density": self.density(spatial).sigmoid(),
            "occupancy_logits": self.occupancy(spatial),
        }


class RuncV2BFRefiner(nn.Module):
    """Fresh residual head on frozen second-pass features and masked P0 RGB.

    The final convolution starts at zero, so an untrained refiner is exactly
    the stage-I prediction. No legacy Fontify decoder or fusion is used.
    """

    def __init__(self, feature_width, hidden_dim=32):
        super().__init__()
        self.feature = nn.Sequential(
            nn.LayerNorm(feature_width), nn.Linear(feature_width, hidden_dim),
        )
        self.visible_pixels = nn.Conv2d(8, hidden_dim, 3, padding=1)
        self.output = nn.Conv2d(hidden_dim, 3, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, lower_features, masked_p0, reference, head, pixel_mask):
        gates = (head["occupancy_logits"].sigmoid()
                 * head["existence_logits"].sigmoid()[:, :, None, None])
        gate_map = F.interpolate(
            gates.mean(1, keepdim=True), masked_p0.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        features = self.feature(lower_features).permute(0, 3, 1, 2)
        features = F.interpolate(
            features, masked_p0.shape[-2:], mode="bilinear",
            align_corners=False,
        )
        visible = self.visible_pixels(torch.cat((
            masked_p0, reference, gate_map, pixel_mask.to(masked_p0.dtype),
        ), dim=1))
        return self.output(F.gelu(features + visible))


def _configuration(model, supplied):
    encoder_width = model.decoder_embed.in_features // 4
    grid_size = math.isqrt(model.patch_embed.num_patches // 2)
    if (model.depth != 12
            or 2 * grid_size ** 2 != model.patch_embed.num_patches):
        raise ValueError("Run C v2 requires four tapped, stacked ViT-B grids")
    output_size = grid_size * model.patch_size
    if output_size % 4:
        raise ValueError("Run C v2 targets must support exact 4x downsampling")
    dimensions = {
        "encoder_width": encoder_width, "feature_width": 4 * encoder_width,
        "grid_size": grid_size, "patch_size": model.patch_size,
        "output_size": output_size, "density_size": output_size // 4,
        "tap_indices": list(TAP_INDICES),
    }
    defaults = {
        "schema": SCHEMA, "experiment": "runc_v2", "stage": "ia0",
        "head_width": 256, "head_heads": 4, "slots": 4, "classes": 20,
        "hidden_dim": 256, "num_heads": 4, "num_layers": 1,
        "unfreeze_blocks": list(TAP_INDICES), **dimensions,
    }
    if set(supplied) - set(defaults):
        raise ValueError("Unknown Run C v2 configuration fields")
    config = {**defaults, **supplied}
    if config["schema"] != SCHEMA or config["experiment"] != "runc_v2":
        raise ValueError("Unknown Run C v2 schema or experiment")
    if any(config[key] != value for key, value in dimensions.items()):
        raise ValueError("Run C v2 metadata and model dimensions disagree")
    for name in ("head_width", "head_heads", "slots", "classes", "hidden_dim",
                 "num_heads", "num_layers"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config["head_width"] % config["head_heads"]:
        raise ValueError("Head width must be divisible by head heads")
    if config["hidden_dim"] % config["num_heads"]:
        raise ValueError("Renderer hidden width must be divisible by heads")
    if config["slots"] != 4:
        raise ValueError("Run C v2 uses four unordered JT slots")
    selected = config["unfreeze_blocks"]
    if (not isinstance(selected, (list, tuple)) or not selected
            or any(type(index) is not int or index not in TAP_INDICES
                   for index in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("Unfrozen blocks must be a distinct subset of taps")
    config["unfreeze_blocks"] = sorted(selected)
    if config["stage"] not in STAGES:
        raise ValueError("Unknown Run C v2 stage")
    return config


def enable_runc_v2(model, config=None, refiner_config=None):
    """Attach fresh JT/renderer modules to a strictly legacy Fontify model."""
    conflicting = (
        "dual_head", "dual_fusion", "structure_adapter", "bf_refiner",
        "g_control", "g_scale", "style_conditioner", "runc_v2_coarse",
    )
    if any(hasattr(model, name) for name in conflicting):
        raise ValueError("Run C v2 must attach to a legacy, unadapted model")
    config = _configuration(model, dict(config or {}))
    stage = config["stage"]
    if stage == "refine" and refiner_config is None:
        raise ValueError("A refine checkpoint requires explicit BF metadata")
    device = next(model.parameters()).device
    model.dual_head = DualViewHead(
        encoder_width=config["encoder_width"], width=config["head_width"],
        grid_size=config["grid_size"], slots=config["slots"],
        classes=config["classes"], heads=config["head_heads"],
    ).to(device)
    model.runc_v2_coarse = StructureDensityRenderer(
        feature_width=config["feature_width"],
        query_width=config["head_width"],
        classes=config["classes"], hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"], num_layers=config["num_layers"],
        grid_size=config["grid_size"], density_size=config["density_size"],
    ).to(device)
    model.runc_v2_config = config
    model.runc_v2 = True
    set_runc_v2_stage(model, "infer" if stage == "refine" else stage)
    if refiner_config is not None:
        enable_runc_v2_refiner(model, refiner_config)
        set_runc_v2_stage(model, stage)
    return model


def _set_module_modes(model):
    training = model.training
    stage = model.runc_v2_stage
    for child in model.children():
        child.eval()
    if stage in ("ia0", "ia1", "ic"):
        model.dual_head.train(training)
    if stage in ("ia1", "ic"):
        for index in model.runc_v2_config["unfreeze_blocks"]:
            model.blocks[index].train(training)
        model.norm.train(training)
    if stage in ("ib", "ic"):
        model.runc_v2_coarse.train(training)
    if stage == "refine":
        model.bf_refiner.train(training)


def set_runc_v2_stage(model, stage):
    """Apply the stage freezing matrix and return exact trainable names."""
    stage = stage.lower()
    if stage not in STAGES or not getattr(model, "runc_v2", False):
        raise ValueError("Enable Run C v2 and select a valid stage")
    if stage == "refine" and not hasattr(model, "runc_v2_refiner_config"):
        raise ValueError("Stage II requires new BF Refiner attachment")
    model.zero_grad(set_to_none=True)
    model.requires_grad_(False)
    if stage in ("ia0", "ia1", "ic"):
        model.dual_head.requires_grad_(True)
    if stage in ("ia1", "ic"):
        for index in model.runc_v2_config["unfreeze_blocks"]:
            model.blocks[index].requires_grad_(True)
        model.norm.requires_grad_(True)
    if stage in ("ib", "ic"):
        model.runc_v2_coarse.requires_grad_(True)
    if stage == "refine":
        model.bf_refiner.requires_grad_(True)
    model.runc_v2_stage = stage
    model.runc_v2_config["stage"] = stage
    _set_module_modes(model)
    return [name for name, parameter in model.named_parameters()
            if parameter.requires_grad]


def runc_v2_parameter_report(model):
    """Return auditable, JSON-compatible stage groups and parameter names."""
    groups = {name: {"names": [], "count": 0, "trainable_count": 0}
              for name in ("vit", "head", "coarse", "refiner",
                           "legacy_unused")}
    for name, value in model.named_parameters():
        if name.startswith("dual_head."):
            group = "head"
        elif name.startswith("runc_v2_coarse."):
            group = "coarse"
        elif name.startswith("bf_refiner."):
            group = "refiner"
        elif name.startswith(("blocks.", "norm.", "patch_embed.")) or name in (
                "mask_token", "segment_token_x", "segment_token_y",
                "pos_embed"):
            group = "vit"
        else:
            group = "legacy_unused"
        groups[group]["count"] += value.numel()
        if value.requires_grad:
            groups[group]["trainable_count"] += value.numel()
            groups[group]["names"].append(name)
    return {"stage": model.runc_v2_stage, "groups": groups,
            "tap_indices": list(TAP_INDICES),
            "unfreeze_blocks": model.runc_v2_config["unfreeze_blocks"]}


def render_runc_v2(model, lower, reference, head, structure=None):
    """Render fixed real F/Q with optional S override for condition probes."""
    if structure is not None:
        if set(structure) != set(STRUCTURE_KEYS):
            raise ValueError("Structure override requires the three S heads")
    result = model.runc_v2_coarse(lower, reference, head, structure=structure)
    pixels = 1 - F.interpolate(
        result["density"], (model.runc_v2_config["output_size"],) * 2,
        mode="bilinear", align_corners=False,
    )
    rgb = pixels.expand(-1, 3, -1, -1)
    mean = rgb.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = rgb.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    result["p0_lower"] = (rgb - mean) / std
    return result


def runc_v2_forward(model, imgs, tgts, render=None, head_override=None,
                    return_features=False):
    """Run a label-free deployment forward; lower GT is never embedded."""
    if not getattr(model, "runc_v2", False):
        raise ValueError("Run C v2 has not been enabled")
    if imgs.shape != tgts.shape or imgs.ndim != 4:
        raise ValueError("Source and visible-reference shapes must match")
    if imgs.shape[-2] != 2 * imgs.shape[-1]:
        raise ValueError("Run C v2 expects vertically stacked square images")
    if imgs.shape[-1] != model.runc_v2_config["output_size"]:
        raise ValueError("Input size differs from Run C v2 configuration")
    _set_module_modes(model)
    stage = model.runc_v2_stage
    if render is None:
        render = stage in ("ib", "ic", "infer", "refine")
    half_pixels = tgts.shape[-2] // 2
    safe_target = torch.cat((
        tgts[:, :, :half_pixels], torch.zeros_like(tgts[:, :, half_pixels:]),
    ), dim=2)
    mask = model.full_target_mask(imgs)
    # Autograd must remain active across frozen blocks in IA1/IC, because they
    # transport gradients from later taps to the explicitly selected blocks.
    with torch.set_grad_enabled(torch.is_grad_enabled()
                                and stage in ("ia1", "ic")):
        taps, quadrants = model.forward_encoder(
            imgs, safe_target, mask, return_quadrants=True,
        )
        features = torch.cat(taps, dim=-1)
    half = features.shape[1] // 2
    lower, reference = features[:, half:], features[:, :half]
    with torch.set_grad_enabled(torch.is_grad_enabled()
                                and stage in ("ia0", "ia1", "ic")):
        head = model.dual_head(quadrants, lower)
    result = {"head": head, "full_mask": mask}
    if render:
        result.update(render_runc_v2(
            model, lower, reference, head, structure=head_override,
        ))
        result["p0"] = torch.cat((
            tgts[:, :, :half_pixels], result["p0_lower"],
        ), dim=2)
    elif head_override is not None:
        raise ValueError("Structure overrides require rendering")
    if return_features:
        result.update(lower_features=lower, reference_features=reference)
    return result


def enable_runc_v2_refiner(model, config=None):
    """Attach a new stage-II residual module for interface checks."""
    if not getattr(model, "runc_v2", False) or hasattr(model, "bf_refiner"):
        raise ValueError("Attach one new BF Refiner to an enabled Run C v2")
    defaults = {
        "schema": "runc-v2-bf-interface-v1", "hidden_dim": 32,
        "bf_annotation_layers": 11, "aggregate_bf_types": False,
        "auto_candidates": 32, "auto_threshold": 0.1, "auto_rounds": 1,
    }
    supplied = dict(config or {})
    if set(supplied) - set(defaults):
        raise ValueError("Unknown Run C v2 BF Refiner configuration")
    config = {**defaults, **supplied}
    if (config["schema"] != defaults["schema"]
            or config["bf_annotation_layers"] != 11
            or config["aggregate_bf_types"] is not False):
        raise ValueError("Stage II keeps eleven unaggregated BF annotations")
    for name in ("hidden_dim", "auto_candidates", "auto_rounds"):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    threshold = config["auto_threshold"]
    if (not isinstance(threshold, (int, float)) or not math.isfinite(threshold)
            or not 0 <= threshold <= 1):
        raise ValueError("Auto threshold must be finite and in [0, 1]")
    model.bf_refiner = RuncV2BFRefiner(
        model.runc_v2_config["feature_width"], config["hidden_dim"],
    ).to(next(model.parameters()).device)
    model.runc_v2_refiner_config = config
    set_runc_v2_stage(model, "refine")
    return model


def runc_v2_refine_forward(model, imgs, tgts, bf_mask=None, inference=False):
    """Use frozen generated P0 and second MAE encoding, never lower GT.

    A training caller supplies the existing semantic patch mask generated by
    selecting eleven original BF annotations. This interface neither samples
    nor merges BF categories; it preserves the supplied binary patch mask.
    Inference reuses the existing automatic structure-mask implementation.
    """
    if (getattr(model, "runc_v2_stage", None) != "refine"
            or not hasattr(model, "runc_v2_refiner_config")):
        raise ValueError("Select the frozen stage-II refiner interface first")
    _set_module_modes(model)
    with torch.no_grad():
        result = runc_v2_forward(model, imgs, tgts, render=True)
    current = result["p0"]
    half = tgts.shape[-2] // 2
    config = model.runc_v2_refiner_config
    rounds = config["auto_rounds"] if inference else 1
    result.update(masks=[], intermediates=[current])
    for _ in range(rounds):
        if inference:
            proxy = SimpleNamespace(
                patch_size=model.patch_size, structure_first_config=config,
            )
            lower_mask = auto_structure_mask(proxy, current, result["head"])
        else:
            lower_mask = normalize_bf_mask(bf_mask, imgs, model.patch_size)
        pixel_mask = lower_mask[:, None].repeat_interleave(
            model.patch_size, 2,
        ).repeat_interleave(model.patch_size, 3)
        masked_lower = torch.where(
            pixel_mask, torch.zeros_like(current[:, :, half:]),
            current[:, :, half:],
        )
        second_target = torch.cat((tgts[:, :, :half], masked_lower), dim=2)
        full_mask = torch.cat(
            (torch.zeros_like(lower_mask), lower_mask), dim=1,
        )
        with torch.no_grad():
            taps = model.forward_encoder(
                imgs, second_target, full_mask.flatten(1),
            )
            features = torch.cat(taps, dim=-1)
            lower_features = features[:, features.shape[1] // 2:]
        residual = model.bf_refiner(
            lower_features, masked_lower, tgts[:, :, :half], result["head"],
            pixel_mask,
        )
        current = torch.cat((
            current[:, :, :half], current[:, :, half:] + residual,
        ), dim=2)
        result["masks"].append(lower_mask)
        result["intermediates"].append(current)
        result.update(mask=lower_mask, masked_p0=second_target,
                      full_mask=full_mask.flatten(1))
    result["p1"] = current
    return result


def runc_v2_checkpoint_metadata(model):
    """Snapshot architecture metadata without aliases to mutable state."""
    if not getattr(model, "runc_v2", False):
        raise ValueError("Metadata requires an enabled Run C v2")
    result = {
        "experiment": "runc_v2",
        "runc_v2_config": deepcopy(model.runc_v2_config),
    }
    if hasattr(model, "runc_v2_refiner_config"):
        result["runc_v2_refiner_config"] = deepcopy(
            model.runc_v2_refiner_config,
        )
    return result


def restore_runc_v2(model, checkpoint):
    """Rebuild optional stages from metadata, then strictly load tensors."""
    config = checkpoint.get("runc_v2_config")
    if checkpoint.get("experiment") != "runc_v2" or not config:
        raise ValueError("Run C v2 restoration requires lineage metadata")
    if getattr(model, "runc_v2", False):
        raise ValueError("Restore onto a legacy, unadapted model")
    verified = _configuration(model, dict(config))
    if set(config) != set(verified):
        missing = sorted(set(verified) - set(config))
        raise ValueError(f"Incomplete Run C v2 metadata: {missing}")
    state = checkpoint["model"]
    for prefix in ("dual_head.", "runc_v2_coarse."):
        if not any(name.startswith(prefix) for name in state):
            raise ValueError(f"Run C v2 checkpoint is missing module {prefix}")
    has_refiner = any(name.startswith("bf_refiner.") for name in state)
    refiner = checkpoint.get("runc_v2_refiner_config")
    if has_refiner != bool(refiner):
        raise ValueError("BF Refiner metadata and tensor keys disagree")
    if config.get("stage") == "refine" and not refiner:
        raise ValueError("A refine checkpoint requires explicit BF metadata")
    if refiner:
        required_refiner = {
            "schema", "hidden_dim", "bf_annotation_layers",
            "aggregate_bf_types", "auto_candidates", "auto_threshold",
            "auto_rounds",
        }
        if set(refiner) != required_refiner:
            raise ValueError("Incomplete or unknown BF Refiner metadata")
    enable_runc_v2(model, deepcopy(config), refiner_config=deepcopy(refiner))
    expected = model.state_dict()
    changed_dtypes = {
        name: {"checkpoint": str(value.dtype),
               "model": str(expected[name].dtype)}
        for name, value in state.items()
        if name in expected and value.dtype != expected[name].dtype
    }
    if changed_dtypes:
        raise ValueError(f"Run C v2 tensor dtype mismatch: {changed_dtypes}")
    model.load_state_dict(state, strict=True)
    return model


def state_tensor_digest(state):
    """Hash sorted tensor names, dtypes, shapes and exact contiguous bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().view(
            torch.uint8,
        ).numpy().tobytes())
    return digest.hexdigest()


def load_runc_v2_pretrained(model, checkpoint):
    """Load compatible legacy weights exactly, before fresh module attachment.

    This deliberately has no permissive decoder/backbone shape migration. The
    verified font-pretraining Base/dec64 checkpoint has matching tensor shapes;
    a differing checkpoint requires an explicit audited implementation change.
    """
    if getattr(model, "runc_v2", False) or getattr(model, "dual_view", False):
        raise ValueError("Load font-pretraining into a legacy model first")
    state = checkpoint["model"]
    if any(name.startswith(("dual_", "structure_adapter.", "bf_refiner.",
                            "g_control.", "g_scale.", "runc_v2_"))
           for name in state):
        raise ValueError("Run C v2 initialization cannot use adapted weights")
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = {name: {"checkpoint": list(value.shape),
                         "model": list(expected[name].shape)}
                  for name, value in state.items()
                  if name in expected and value.shape != expected[name].shape}
    if missing or unexpected or mismatched:
        raise ValueError(f"Incompatible pretrained keys: missing={missing}, "
                         f"unexpected={unexpected}, shapes={mismatched}")
    model.load_state_dict(state, strict=True)
    return {
        "source_model_tensor_sha256": state_tensor_digest(state),
        "loaded_keys": sorted(state), "loaded_key_count": len(state),
        "loaded_parameter_count": sum(
            value.numel() for value in state.values()
        ),
        "missing": missing, "unexpected": unexpected,
        "shape_mismatch": mismatched,
        "new_modules": ["dual_head", "runc_v2_coarse"],
        "new_initialization": "fresh constructors after legacy strict load",
    }
