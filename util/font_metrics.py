"""Image-only glyph metrics, independent of training losses."""

import torch
from torch.nn import functional as F


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
