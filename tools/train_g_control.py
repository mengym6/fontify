"""Train G fusion, progressively unfreeze decoder, then fit explicit sigma.

This is a separate continuation protocol, not a reproduction of the original
51-epoch Run B training. Old losses and their checkpoint coefficients are kept.
The frozen encoder/head run in eval mode throughout all three stages.
"""

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dual_dataset import DualDataset  # noqa: E402
from data.sampler import DistributedSamplerWrapper  # noqa: E402
from tools.dual_view_protocol import hash_files, seed_all  # noqa: E402
from tools.eval_s_baseline import (  # noqa: E402
    DUAL_SCHEMA,
    absolute_s_score,
    build_dataset,
    eval_metadata,
    evaluate,
    file_sha256,
    validate_eval_coefficients,
    validate_s_baseline,
)
from util.checkpoint import build_checkpoint_model, checkpoint_args  # noqa: E402
from util.dual_loss import LowerJietiLoss  # noqa: E402
from util.g_control import (  # noqa: E402
    configure_stage,
    enable_g_control,
    protected_prefix_exclusions,
)
from util.g_scale import enable_g_scale  # noqa: E402
from util.masking_generator import MaskingGenerator  # noqa: E402


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def state_digest(model, excluded=()):
    """Digest all original model tensors that must remain bitwise frozen."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith(excluded):
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def decoder_phase(epoch, epochs):
    """Final RGB layer -> convolution decoder -> pixel projection and decoder."""
    return min(2, 3 * epoch // epochs)


def configure_optimizer(model, stage, phase, lr_g, lr_decoder, lr_head=1e-5):
    names = configure_stage(model, stage, phase)
    if stage == 5:
        groups = []
        for name, lr in (
            ("g_control", lr_g),
            ("g_scale", lr_g),
            ("decoder_pred", lr_decoder),
            ("dual_head", lr_head),
        ):
            parameters = [
                p for p in getattr(model, name).parameters() if p.requires_grad
            ]
            groups.append({"params": parameters, "lr": lr, "base_lr": lr, "name": name})
        return torch.optim.AdamW(groups, weight_decay=0.05, betas=(0.9, 0.999)), names
    if stage == 4:
        groups = [
            {
                "params": list(model.g_scale.parameters()),
                "lr": lr_g,
                "base_lr": lr_g,
                "name": "g_scale",
            }
        ]
        return torch.optim.AdamW(groups, weight_decay=0.05, betas=(0.9, 0.999)), names
    # Keep all possible decoder parameters in the optimizer from the beginning
    # so phase transitions preserve the earlier parameters' Adam moments.
    groups = [
        {
            "params": list(model.g_control.parameters()),
            "lr": lr_g,
            "base_lr": lr_g,
            "name": "g_control",
        }
    ]
    if stage == 3:
        for name in ("decoder_pred", "decoder_embed"):
            groups.append(
                {
                    "params": list(getattr(model, name).parameters()),
                    "lr": lr_decoder,
                    "base_lr": lr_decoder,
                    "name": name,
                }
            )
    return torch.optim.AdamW(groups, weight_decay=0.05, betas=(0.9, 0.999)), names


def checkpoint_payload(model, source, args, optimizer, epoch, step):
    result = copy.copy(source)
    result.pop("optimizer_d", None)
    result.pop("scaler", None)
    result["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    result["args"] = {**checkpoint_args(source), "output_dir": str(args.output)}
    result["g_control_config"] = model.g_control_config
    if hasattr(model, "g_scale_config"):
        result["g_scale_config"] = model.g_scale_config
    result["g_training_config"] = {
        **vars(args),
        "output": str(args.output),
        "checkpoint": str(args.checkpoint),
    }
    result["epoch"] = epoch
    result["optimizer"] = optimizer.state_dict()
    result["g_training_step"] = step
    result["parent_checkpoint_sha256"] = args.parent_sha256
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", type=int, choices=(2, 3, 4, 5), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr-g", type=float, default=3e-4)
    parser.add_argument("--lr-decoder", type=float, default=3e-5)
    parser.add_argument("--lr-head", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--effective-batch", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if args.epochs < 3 or args.effective_batch % (args.batch_size * world):
        raise ValueError("Need >=3 epochs and divisible effective batch")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.set_num_threads(1)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    seed_all(args.seed)
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    args.parent_sha256 = file_sha256(args.checkpoint)
    if args.stage == 2 and args.parent_sha256 != (
        "57ac8276e3468ad92ce2aff033736128b33862a4f922618c51910717166ae852"
    ):
        raise ValueError("Stage 2 must use the sealed Run B Fix ep50 weights")
    if args.stage == 3 and source.get("g_training_config", {}).get("stage") != 2:
        raise ValueError("Stage 3 must continue a stage 2 result")
    if args.stage == 4 and source.get("g_training_config", {}).get("stage") != 3:
        raise ValueError("Stage 4 must continue a stage 3 result")
    if args.stage == 5 and source.get("g_training_config", {}).get("stage") != 4:
        raise ValueError("Stage 5 must continue a stage 4 result")
    if not args.smoke and source.get("g_training_config", {}).get("smoke"):
        raise ValueError("Diagnostic checkpoints cannot initialize a formal run")
    model = build_checkpoint_model(source).to(device).eval()
    if args.stage == 2:
        if hasattr(model, "g_control"):
            raise ValueError("Stage 2 must start from unmodified Run B Fix")
        enable_g_control(model).to(device)
    elif not hasattr(model, "g_control"):
        raise ValueError("Stage 3 requires the completed stage 2 checkpoint")
    if args.stage == 4:
        enable_g_scale(model).to(device)
    if args.stage == 5 and not hasattr(model, "g_scale"):
        raise ValueError("Stage 5 requires the trained scale module")
    control_module = model.g_scale if args.stage == 4 else model.g_control
    saved_args = checkpoint_args(source)
    for name, default in (
        ("style_weight", 14.73),
        ("edge_weight_final", 0.2),
        ("edge_warmup_epochs", 10),
        ("loss_warmup_duration", 8),
    ):
        setattr(model, name, saved_args.get(name, default))
    if saved_args.get("tip_weight") is not None:
        raise ValueError("This experiment requires the original edge Run B Fix")
    optimizer, trainable = configure_optimizer(
        model, args.stage, 0, args.lr_g, args.lr_decoder, args.lr_head
    )
    protected = protected_prefix_exclusions(args.stage)
    frozen_digest = state_digest(model, protected)
    data = ROOT / "fontdata_example"
    prep = ROOT / "models/dual_view_pad_20261014"
    mapped = argparse.Namespace(
        data_path=str(data),
        val_json_path=sorted((data / "val_json_clean_v2").glob("*.json")),
        fixed_pair_path=str(data / "val_pairs_fixed_clean.json"),
        dual_target_cache=str(data / "dual_targets_v1"),
        j_definition=DUAL_SCHEMA,
    )
    metadata = eval_metadata(mapped)
    coef_file = prep / "eval_coef.json"
    coef = validate_eval_coefficients(json.loads(coef_file.read_text()), metadata)
    denominator = json.loads((prep / "s_baseline_dual.json").read_text())
    validate_s_baseline(
        denominator, {**metadata, "eval_coef_sha256": file_sha256(coef_file)}
    )
    validation = build_dataset(mapped)
    train_manifest = data / "train_json_dual_v1/train_dual.json"
    training = DualDataset(
        data,
        [train_manifest],
        mapped.dual_target_cache,
        MaskingGenerator((56, 28), num_masking_patches=784, max_num_patches=392),
    )
    sampler = DistributedSamplerWrapper(
        WeightedRandomSampler(training.weights, len(training), replacement=True),
        num_replicas=world,
        rank=rank,
        shuffle=True,
    )
    loader = DataLoader(
        training,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    accumulation = args.effective_batch // (args.batch_size * world)
    usable = len(loader) // accumulation * accumulation
    if usable == 0:
        raise ValueError("Not enough full batches for one optimizer step")
    evaluation_j = LowerJietiLoss(**{f"w_{k}": v for k, v in coef.items()}).to(device)
    writer = SummaryWriter(str(args.output / "logs")) if rank == 0 else None
    manifest = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "world_size": world,
        "accum_iter": accumulation,
        "steps_per_epoch": usable // accumulation,
        "train_length": len(training),
        "train_manifest_sha256": hash_files([train_manifest]),
        **metadata,
        "eval_coef_sha256": file_sha256(coef_file),
        "frozen_tensors_sha256_before": frozen_digest,
        "original_Q_projection_frozen": True,
        "trainable_parameter_counts": {
            name: sum(
                p.numel() for p in getattr(model, name).parameters() if p.requires_grad
            )
            for name in ("g_control", "dual_head", "decoder_pred", "decoder_embed")
        },
        "ViT_and_raw_Q_network_frozen": True,
        "decoder_embed_frozen": args.stage != 3,
        "head_image_gradient_route": "JT only; BF inputs detached before new G paths",
        "frozen_backbone_mode": "eval",
        "loss_epoch": 50,
        "image_edge_weight": model.get_dynamic_loss_weights(50)[1],
        "loss_weights": {
            "lambda_full": model.dual_lambda_full,
            "w_j": model.jieti_loss_weight,
            "w_h": model.dual_head_weight,
            "style": model.style_weight,
        },
        "G_control_config": model.g_control_config,
        "budget_origin": "exploratory continuation; not established optimal",
        "source_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "tools/train_g_control.py",
                "models_train.py",
                "util/checkpoint.py",
                "util/g_control.py",
                "util/g_scale.py",
                "util/dual_view.py",
            )
        },
    }
    if rank == 0:
        write_json(args.output / "manifest.json", manifest)
        write_json(args.output / "trainable_phase_0.json", trainable)
        for name in manifest["source_sha256"]:
            destination = args.output / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
    image_sigma_grad = None
    if args.smoke and args.stage == 4:
        # Measure the image/J path separately from direct sigma supervision,
        # before DDP installs reducer hooks. Frozen G and decoder must still
        # transmit derivatives to the scale predictor's inputs.
        index = next(i for i, pair in enumerate(training.pairs) if pair["has_jt"])
        probe = training.get_pair(index, augmentation_seed=97)
        probe = {
            k: v[None].to(device) if torch.is_tensor(v) else v for k, v in probe.items()
        }
        with torch.autocast("cuda", dtype=torch.bfloat16):
            values = model(
                probe["image"],
                probe["target"],
                probe["mask"].flatten(1),
                probe["valid"],
                epoch=50,
                no_gan=True,
                dual_targets=probe,
            )
            image_j = (
                model.dual_lambda_full * values[8]["L_full"]
                + model.jieti_loss_weight * values[8]["J"]
            )
        gradients = torch.autograd.grad(
            image_j, tuple(model.g_scale.parameters()), allow_unused=True
        )
        image_sigma_grad = math.sqrt(
            sum(float(g.float().square().sum()) for g in gradients if g is not None)
        )
        if image_sigma_grad <= 0:
            raise RuntimeError(
                "Scale branch has no image/J gradient through frozen modules"
            )
    wrapped = None
    current_phase = None
    best_s, step = math.inf, 0
    grad_evidence = []
    start = time.time()
    for epoch in range(args.epochs):
        phase = decoder_phase(epoch, args.epochs) if args.stage == 3 else 0
        if args.smoke and args.stage == 3:
            phase = epoch
        if phase != current_phase:
            if wrapped is not None:
                del wrapped
            trainable = configure_stage(model, args.stage, phase)
            model.eval()
            control_module.train()
            if args.stage in (3, 5):
                model.decoder_pred.train()
            wrapped = (
                DistributedDataParallel(
                    model,
                    device_ids=[local],
                    broadcast_buffers=False,
                )
                if world > 1
                else model
            )
            current_phase = phase
            if rank == 0:
                write_json(args.output / f"trainable_phase_{phase}.json", trainable)
        control_module.train()
        if args.stage in (3, 5):
            model.decoder_pred.train()
        sampler.set_epoch(epoch)
        seed_all(args.seed + epoch * 1009 + rank)
        optimizer.zero_grad(set_to_none=True)
        epoch_sums, epoch_count = {}, 0
        epoch_start_step = step
        for micro, batch in enumerate(loader):
            if micro >= usable:
                break
            progress = epoch + micro / usable
            # G warms up over two continuation epochs. Decoder phases each
            # warm up over one epoch when new parameters are made trainable.
            for group in optimizer.param_groups:
                warmup = min(1.0, (progress + 1 / usable) / 2)
                if args.stage == 3 and group["name"] not in ("g_control", "g_scale"):
                    phase_start = math.ceil(phase * args.epochs / 3)
                    warmup = min(1.0, progress - phase_start + 1 / usable)
                cosine = (
                    0.1 + 0.9 * (1 + math.cos(math.pi * progress / args.epochs)) / 2
                )
                group["lr"] = group["base_lr"] * warmup * cosine
            batch = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            update = (micro + 1) % accumulation == 0
            sync = (
                contextlib.nullcontext() if update or world == 1 else wrapped.no_sync()
            )
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = wrapped(
                        batch["image"],
                        batch["target"],
                        batch["mask"].flatten(1),
                        batch["valid"],
                        epoch=50,
                        no_gan=True,
                        dual_targets=batch,
                    )
                    loss = result[0]
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Nonfinite loss epoch {epoch} micro {micro}"
                    )
                (loss / accumulation).backward()
            metrics = {
                "total": loss,
                "l1l2": result[1],
                "vgg": result[2],
                "edge": result[3],
                "J": result[8]["J"],
                "H": result[8]["H"],
            }
            if "sigma" in result[8]:
                metrics["sigma"] = result[8]["sigma"]
            for key, value in metrics.items():
                epoch_sums[key] = epoch_sums.get(key, 0.0) + float(value.detach())
            epoch_count += 1
            if update:
                active = [p for p in model.parameters() if p.requires_grad]
                grad = torch.nn.utils.clip_grad_norm_(active, 3)
                if not torch.isfinite(grad):
                    raise FloatingPointError("Nonfinite gradient norm")
                g_grad = math.sqrt(
                    sum(
                        float(p.grad.float().square().sum())
                        for p in control_module.parameters()
                        if p.grad is not None
                    )
                )
                assert all(
                    p.grad is None
                    for n, p in model.named_parameters()
                    if not p.requires_grad
                ), "Frozen parameter received gradient"
                if len(grad_evidence) < 3 or args.smoke:
                    module_norms = {}
                    if args.stage == 5:
                        for name in (
                            "g_control",
                            "g_scale",
                            "dual_head",
                            "decoder_pred",
                        ):
                            module_norms[name] = math.sqrt(
                                sum(
                                    float(p.grad.float().square().sum())
                                    for p in getattr(model, name).parameters()
                                    if p.grad is not None
                                )
                            )
                    grad_evidence.append(
                        {
                            "epoch": epoch,
                            "step": step,
                            "G_grad_post_clip": g_grad,
                            "pre_clip_grad_norm": float(grad),
                            "module_post_clip_norms": module_norms,
                        }
                    )
                if g_grad <= 0:
                    raise RuntimeError(
                        "Trainable control branch has no training gradient"
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if writer:
                    writer.add_scalar("grad/pre_clip", float(grad), step)
                    writer.add_scalar("grad/G_post_clip", g_grad, step)
            if args.smoke and step - epoch_start_step >= 2:
                break
        totals = torch.tensor([epoch_count] + list(epoch_sums.values()), device=device)
        if world > 1:
            dist.all_reduce(totals)
        train_stats = {
            k: float(totals[i + 1] / totals[0]) for i, k in enumerate(epoch_sums)
        }
        frozen_after = state_digest(model, protected)
        if frozen_after != frozen_digest:
            raise RuntimeError("Protected model tensors changed")
        if world > 1:
            dist.barrier()
        if rank == 0:
            model.eval()
            if args.smoke:
                result = {
                    "status": "passed",
                    "optimizer_steps": step,
                    "gradients": grad_evidence,
                    "protected_tensors_unchanged": True,
                    "max_memory_GiB": torch.cuda.max_memory_allocated() / 2**30,
                    "image_J_to_scale_gradient_norm": image_sigma_grad,
                }
                write_json(args.output / "smoke.json", result)
                torch.save(
                    checkpoint_payload(model, source, args, optimizer, epoch, step),
                    args.output / "diagnostic-only.pth",
                )
            else:
                # Evaluation uses FP32, batch1, and the same 240-pair evaluator.
                seed_all(0)
                result, per_item = evaluate(
                    model, validation, evaluation_j, device, DUAL_SCHEMA
                )
                result["S"], result["S_components"] = absolute_s_score(
                    result, denominator
                )
                result.update(
                    epoch=epoch,
                    optimizer_steps=step,
                    per_item=per_item,
                    parent_checkpoint_sha256=args.parent_sha256,
                    **metadata,
                )
                write_json(args.output / f"eval_{epoch:02d}.json", result)
                payload = checkpoint_payload(
                    model, source, args, optimizer, epoch, step
                )
                if result["S"] < best_s:
                    best_s = result["S"]
                    torch.save(payload, args.output / "checkpoint-best.pth")
                    write_json(args.output / "best_eval.json", result)
                torch.save(payload, args.output / "checkpoint-last.pth")
                for key, value in train_stats.items():
                    writer.add_scalar("train_loss_detail/" + key, value, epoch)
                for key in ("L1_JT", "L1_BF", "J", "S"):
                    writer.add_scalar("val/" + key, result[key], epoch)
                with (args.output / "train.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "phase": phase,
                                "step": step,
                                "train": train_stats,
                                "eval": {
                                    k: result[k] for k in ("L1_JT", "L1_BF", "J", "S")
                                },
                            }
                        )
                        + "\n"
                    )
            write_json(
                args.output / "status.json",
                {
                    "status": "complete"
                    if args.smoke or epoch == args.epochs - 1
                    else "running",
                    "stage": args.stage,
                    "completed_epochs": epoch + 1,
                    "optimizer_steps": step,
                    "best_S": best_s if math.isfinite(best_s) else None,
                    "elapsed_seconds": time.time() - start,
                    "protected_tensors_unchanged": True,
                },
            )
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "step": step,
                        "train": train_stats,
                        "S": result.get("S"),
                        "best_S": best_s,
                    }
                ),
                flush=True,
            )
            if writer:
                writer.flush()
        if world > 1:
            dist.barrier()
        if args.smoke and (args.stage != 3 or epoch == 2):
            break
    if writer:
        write_json(args.output / "gradient_evidence.json", grad_evidence)
        writer.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
