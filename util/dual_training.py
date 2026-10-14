"""Training adapters for the opt-in dual-view run B path."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from util import misc


def complete_accumulation_steps(micro_batches, accumulation):
    """Keep each run B update a full effective batch without changing sampling."""
    if accumulation < 1 or micro_batches < accumulation:
        raise ValueError('an epoch must contain a complete accumulation window')
    return micro_batches - micro_batches % accumulation


def unpack_batch(batch, device):
    """Move supervision separately from the four image/mask inputs."""
    if isinstance(batch, dict):
        inputs = [batch[key] for key in ("image", "target", "mask", "valid")]
        labels = {
            key: batch[key].to(device, non_blocking=True)
            for key in (
                "has_jt", "instance_occupancy", "instance_valid",
                "instance_classes", "voro", "valid_parts",
            )
        }
        if 'has_bf' in batch:
            labels['has_bf'] = batch['has_bf'].to(device, non_blocking=True)
        return (*inputs, {"dual_targets": labels})
    inputs = list(batch[:4])
    labels = {}
    if len(batch) > 4:
        labels = {
            "voro": batch[4].to(device, non_blocking=True),
            "valid_parts": batch[5].to(device, non_blocking=True),
            "is_jt": batch[6].to(device, non_blocking=True),
        }
    return (*inputs, labels)


def save_probe_snapshot(model, args, epoch):
    """Save model-only ep2/25 snapshots without touching RNG or optimizers."""
    if epoch not in (2, 25) or not misc.is_main_process():
        return
    raw = model.module if hasattr(model, "module") else model
    destination = Path(args.output_dir) / f"checkpoint-probe-{epoch}.pth"
    misc.save_on_master({"model": raw.state_dict(), "args": args,
                         "epoch": epoch, "probe_only": True}, destination)


@torch.no_grad()
def evaluate_dual(data_loader, model, device, epoch, args, log_writer=None):
    """Evaluate every fixed-val item once using the shared evaluation J.

    Call the unwrapped model: rank shards may have different lengths. There
    are no DDP forward collectives here; sum/count reductions happen once.
    J is evaluated per image to preserve the standalone evaluator's contract.
    """
    from util.dual_loss import LowerJietiLoss

    raw = model.module if hasattr(model, "module") else model
    raw.eval()
    coefficients = json.loads(
        Path(args.dual_eval_calibration_json).read_text(encoding="utf-8")
    )["coef"]
    jmod = LowerJietiLoss(
        pool=args.jieti_pool, k_max=args.jieti_k_max,
        soft_fg=args.jieti_soft_fg,
        pred_mass_ratio=args.jieti_pred_mass_ratio,
        w_centroid=coefficients["centroid"],
        w_logsigma=coefficients["logsigma"],
        w_shape=coefficients["shape"],
    ).to(device)
    sums = torch.zeros(8, dtype=torch.float64, device=device)
    frames = []
    for batch in data_loader:
        image = batch["image"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device).flatten(1)
        # Evaluation uses fp32, matching eval_s_baseline.py.
        pred = raw.predict_full(image, target)
        pixel_mask = raw.unpatchify(
            mask[..., None].float().repeat(1, 1, raw.patch_size ** 2 * 3)
        )
        valid = batch["valid"].to(device)
        area = pixel_mask * valid
        l1 = ((pred - target).abs() * area).sum((1, 2, 3))
        l1 = l1 / (area.sum((1, 2, 3)) + 1e-2)
        has_jt = batch["has_jt"].to(device).bool()
        sums[0] += l1.sum()
        sums[1] += len(l1)
        sums[2] += l1[has_jt].sum()
        sums[3] += has_jt.sum()
        sums[4] += l1[~has_jt].sum()
        sums[5] += (~has_jt).sum()
        for index in range(len(l1)):
            if has_jt[index]:
                value, _, _ = jmod(
                    pred[index:index + 1], target[index:index + 1],
                    batch["voro"][index:index + 1].to(device),
                    batch["valid_parts"][index:index + 1].to(device),
                    has_jt[index:index + 1],
                )
                sums[6] += value.double()
                sums[7] += 1
        if misc.is_main_process() and getattr(args, "vis_every_epoch", False):
            for index in range(min(4 - len(frames), len(l1))):
                frame = torch.cat(
                    (image[index], target[index] * (1 - pixel_mask[index]),
                     pred[index], target[index]), dim=2,
                ).permute(1, 2, 0).float().cpu().numpy()
                frame = np.clip(
                    (frame * np.array([0.229, 0.224, 0.225])
                     + np.array([0.485, 0.456, 0.406])) * 255, 0, 255,
                ).astype(np.uint8)
                frames.append(frame)
    if misc.is_dist_avail_and_initialized():
        torch.distributed.all_reduce(sums)
    if not torch.isfinite(sums).all():
        raise FloatingPointError('nonfinite dual-view validation values')
    if any(sums[index] <= 0 for index in (1, 3, 5, 7)):
        raise ValueError("dual-view fixed validation must contain JT and BF")
    result = {
        "loss": (sums[0] / sums[1]).item(),
        "L1_JT": (sums[2] / sums[3]).item(),
        "L1_BF": (sums[4] / sums[5]).item(),
        "jieti_J": (sums[6] / sums[7]).item(),
        "n_jt": sums[3].item(), "n_bf": sums[5].item(),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError('nonfinite dual-view validation result')
    if frames:
        vis_dir = Path(args.output_dir) / "vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        grid = np.concatenate(frames, axis=0)
        Image.fromarray(grid).save(vis_dir / f"epoch_{epoch:03d}.png")
        if log_writer is not None:
            log_writer.add_image("dual/val_full", grid, epoch, dataformats="HWC")
    return result
