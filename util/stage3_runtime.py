"""Shared stage-3 checkpoint, optimizer and evaluation implementation."""

import math
from collections import defaultdict

import torch
from torch.nn import functional as F

from util.style_conditioning import enable_conditioning


def load_model(path, config=None):
    """Reconstruct saved stage-3 settings, or migrate an exact legacy model."""
    import models_train

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = checkpoint.get("stage3_config")
    config = dict(saved or {}) if config is None else dict(config)
    model = models_train.vit_base_patch16_input896x448_win_dec64_8glb_sl1()
    enable_conditioning(model, config.get("style_mode", "off"))
    result = model.load_state_dict(checkpoint["model"], strict=False)
    allowed = (
        {name for name in model.state_dict() if name.startswith("style_conditioner.")}
        if not saved
        else set()
    )
    if set(result.missing_keys) - allowed or result.unexpected_keys:
        raise ValueError(f"Incompatible checkpoint: {result}")
    configure(model, config)
    return model, config


def configure(model, config):
    """Experiment 2 detail definition is an invariant of stage 3."""
    if config.get("detail_per_sample_normalize", False):
        raise ValueError("Stage 3 forbids per-sample detail normalization")
    for key, default in (
        ("structure_loss_weight", 0.2),
        ("detail_loss_weight", 0.2),
        ("structure_common_scale", 1.0),
    ):
        value = float(config.get(key, default))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {key}")
        setattr(model, key, value)
    coefficients = config.get("structure_coefficients", [1.0] * 4)
    if len(coefficients) != 4 or any(
        not math.isfinite(v) or v <= 0 for v in coefficients
    ):
        raise ValueError("Four finite positive structure coefficients required")
    model.structure_coefficients = coefficients
    model.detail_per_sample_normalize = False
    model.detail_gradient_ratio = 0.1
    model.detail_kernel_size = 5
    model.detail_sigma = 1.0
    model.semantic_only_epochs = 0
    model.vgg_input_mode = config.get("vgg_input_mode", "legacy")
    if model.vgg_input_mode not in ("legacy", "rgb"):
        raise ValueError("Invalid VGG input mode")
    model.stage3_update = 40


def load_inference_checkpoint(model, checkpoint):
    """Strictly reconstruct conditioning in the existing inference model."""
    config = checkpoint["stage3_config"]
    enable_conditioning(model, config.get("style_mode", "off"))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.stage3_inference = True
    return model


def freeze(model):
    """Freeze the original first nine blocks and non-generator networks."""
    model.discriminator.requires_grad_(False)
    model.vgg_loss.requires_grad_(False)
    for name in (
        "patch_embed",
        "mask_token",
        "segment_token_x",
        "segment_token_y",
        "pos_embed",
    ):
        value = getattr(model, name)
        if value is not None:
            value.requires_grad_(False)
    for block in model.blocks[:9]:
        block.requires_grad_(False)


def optimizer_groups(model):
    """Layer decay for legacy weights, dedicated LR for the new module."""
    from util.lr_decay import get_layer_id_for_vit

    groups = defaultdict(list)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        layer = get_layer_id_for_vit(name, len(model.blocks) + 1)
        rate = 1e-4 * 0.8 ** (len(model.blocks) + 1 - layer)
        if name.startswith("style_conditioner."):
            rate = 3e-4
        decay = 0.0 if parameter.ndim == 1 else 0.05
        groups[(rate, decay)].append(parameter)
    return [
        {"params": parameters, "lr": rate, "base_lr": rate, "weight_decay": decay}
        for (rate, decay), parameters in groups.items()
    ]


def parameter_report(model, optimizer=None, before=None):
    """Report norms by actual parameter group, excluding frozen auxiliaries."""
    rates = {}
    if optimizer is not None:
        rates = {
            id(p): group["lr"]
            for group in optimizer.param_groups
            for p in group["params"]
        }
    report = defaultdict(
        lambda: {
            "total": 0,
            "trainable": 0,
            "grad_sq": 0.0,
            "update_sq": 0.0,
            "learning_rates": set(),
        }
    )
    for name, parameter in model.named_parameters():
        if name.startswith(("vgg_loss.", "discriminator.")):
            continue
        group = (
            ".".join(name.split(".")[:2])
            if name.startswith("blocks.")
            else name.split(".")[0]
        )
        row = report[group]
        row["total"] += parameter.numel()
        if parameter.requires_grad:
            row["trainable"] += parameter.numel()
        if parameter.grad is not None:
            row["grad_sq"] += parameter.grad.float().square().sum().item()
        if before is not None and name in before:
            delta = parameter.detach().float().cpu() - before[name]
            row["update_sq"] += delta.square().sum().item()
        if id(parameter) in rates:
            row["learning_rates"].add(rates[id(parameter)])
    for row in report.values():
        row["gradient_norm"] = math.sqrt(row.pop("grad_sq"))
        row["update_norm"] = math.sqrt(row.pop("update_sq"))
        row["learning_rates"] = sorted(row["learning_rates"])
    return dict(report)


def to_rgb(image):
    mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (image.float() * std + mean).clamp(0, 1)


def metrics(prediction, target):
    """Fixed full-query metrics. No training-mask or per-sample loss scaling."""
    prediction, target = to_rgb(prediction), to_rgb(target)
    weights = prediction.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]
    p = (prediction * weights).sum(1, keepdim=True)
    t = (target * weights).sum(1, keepdim=True)
    axis = torch.arange(-2, 3, device=p.device, dtype=p.dtype)
    kernel = torch.exp(-axis.square() / 2)
    kernel = kernel[:, None] * kernel[None, :]
    kernel = (kernel / kernel.sum())[None, None]
    high = lambda x: x - F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), kernel)
    sobel = p.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel = torch.stack((sobel, sobel.T))[:, None] / 8
    grad = lambda x: F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), sobel)
    pg, tg = grad(p), grad(t)
    pe = pg.square().sum(1, keepdim=True).sqrt() > 0.1
    te = tg.square().sum(1, keepdim=True).sqrt() > 0.1
    tp = (pe & te).sum().float()
    f1 = (2 * tp + 1e-8) / (pe.sum() + te.sum() + 1e-8)
    pf, tf = (p < 0.9).float(), (t < 0.9).float()

    def geometry(foreground):
        mass = foreground.sum().clamp_min(1)
        height, width = foreground.shape[-2:]
        ys = torch.linspace(-1, 1, height, device=p.device)[None, None, :, None]
        xs = torch.linspace(-1, 1, width, device=p.device)[None, None, None, :]
        coords = foreground[0, 0].nonzero()
        aspect = 0.0
        if coords.numel():
            extent = coords.max(0).values - coords.min(0).values + 1
            aspect = (extent[1].float() / extent[0]).item()
        return ((foreground * xs).sum() / mass, (foreground * ys).sum() / mass, aspect)

    px, py, pa = geometry(pf)
    tx, ty, ta = geometry(tf)
    return {
        "highpass": (high(p) - high(t)).abs().mean().item(),
        "gradient": (pg - tg).abs().mean().item(),
        "edge_f1": f1.item(),
        "centroid": ((px - tx).abs() + (py - ty).abs()).item(),
        "row": (pf.mean(3) - tf.mean(3)).abs().mean().item(),
        "col": (pf.mean(2) - tf.mean(2)).abs().mean().item(),
        "area": (pf.mean() - tf.mean()).abs().item(),
        "aspect": abs(pa - ta),
    }
