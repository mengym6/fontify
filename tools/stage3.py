"""Stage-3 audit, calibration, controlled training and reference evaluation."""

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    """Write a reviewable report with nonfinite values rejected."""
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def config_from_args(args):
    config = {
        "style_mode": args.style_mode,
        "structure_loss_weight": args.structure_weight,
        "detail_loss_weight": args.detail_weight,
        "structure_coefficients": [1.0] * 4,
        "structure_common_scale": 1.0,
        "structure_projection_mode": args.structure_projection_mode,
        "vgg_input_mode": args.vgg_input_mode,
        "detail_per_sample_normalize": False,
        "detail_gradient_ratio": 0.1,
        "detail_kernel_size": 5,
        "detail_sigma": 1.0,
    }
    if args.coefficients:
        calibration = json.loads(Path(args.coefficients).read_text())
        if calibration.get("status") != "ok":
            raise ValueError("Calibration is not approved for use")
        if calibration.get("structure_projection_mode", "legacy") != (
            args.structure_projection_mode
        ):
            raise ValueError("Calibration projection mode does not match run")
        config.update(
            {
                key: calibration[key]
                for key in ("structure_coefficients", "structure_common_scale")
            }
        )
    return config


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def balanced_records(records, count, unique=False):
    """Cycle styles in stable order, then characters within each style."""
    records = list({(r["style_id"], r["target_path"]): r for r in records}.values())
    styles = sorted({r["style_id"] for r in records})
    pools = {
        style: sorted(
            [r for r in records if r["style_id"] == style],
            key=lambda r: (r["character"], r["target_path"]),
        )
        for style in styles
    }
    result = []
    if unique:
        for offset in range(max(len(pool) for pool in pools.values())):
            for style in styles:
                if offset < len(pools[style]):
                    result.append(pools[style][offset])
                    if len(result) == count:
                        return result
        return result
    for index in range(count):
        style = styles[index % len(styles)]
        pool = pools[style]
        result.append(pool[(index // len(styles)) % len(pool)])
    return result


def fixed_batch(root, targets, references, device):
    import torch

    from util.stage3_data import fixed_sample, reference_cases

    samples = [
        fixed_sample(root, target, reference_cases(references, target)[0][1])
        for target in targets
    ]
    return tuple(torch.stack(items).to(device) for items in zip(*samples))


def evaluate(model, rows, root, output, device, all_cases=True, limit=32):
    """Keep per-character records and image strips, never select a winner."""
    import numpy as np
    import torch
    from PIL import Image

    from util.stage3_data import fixed_sample, reference_cases
    from util.stage3_runtime import metrics, to_rgb

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    results = []
    try:
        with torch.no_grad():
            for split, records in rows.items():
                targets = balanced_records(
                    records, min(limit, len(records)), unique=True
                )
                for index, target in enumerate(targets):
                    correct_prediction = None
                    cases = reference_cases(records, target)
                    if not all_cases:
                        cases = cases[:1]
                    for case, ref, blank in cases:
                        source, context, truth, mask = fixed_sample(
                            root, target, ref, blank
                        )
                        source, context, truth, mask = [
                            x.unsqueeze(0).to(device)
                            for x in (source, context, truth, mask)
                        ]
                        pred = model.forward_decoder(
                            model.forward_encoder(source, context, mask.flatten(1))
                        )
                        query = pred[:, :, 448:]
                        gt = truth[:, :, 448:]
                        if case == "correct":
                            correct_prediction = to_rgb(query).clone()
                        row = {
                            "split": split,
                            "case": case,
                            "style_id": target["style_id"],
                            "character": target["character"],
                            "target": target["target_path"],
                            "reference": ref["target_path"],
                            "metrics": metrics(query, gt),
                            "reference_output_delta": (
                                to_rgb(query) - correct_prediction
                            )
                            .abs()
                            .mean()
                            .item(),
                        }
                        # Wrong/blank reference errors are diagnostic only.
                        strip = torch.cat(
                            (to_rgb(context[:, :, :448]), to_rgb(query), to_rgb(gt)),
                            dim=3,
                        )
                        pixels = strip[0].permute(1, 2, 0).cpu().numpy()
                        filename = f"{split}-{index:04d}-{case}.png"
                        Image.fromarray((pixels * 255).round().astype(np.uint8)).save(
                            output / filename
                        )
                        row["image"] = filename
                        results.append(row)
        write_json(output / "metrics.json", results)
    finally:
        model.train(was_training)
    return results


def calibrate(model, rows, args):
    """Measure a 4x4 parameter Gram matrix; save no large gradient arrays."""
    import torch

    from util.stage3_runtime import freeze

    freeze(model)
    model.eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    records = balanced_records(rows["train"], 64)
    measurements = []
    for batch in range(32):
        source, context, truth, mask = fixed_batch(
            args.data_root,
            records[2 * batch : 2 * batch + 2],
            rows["train"],
            args.device,
        )
        pred = model.forward_decoder(
            model.forward_encoder(source, context, mask.flatten(1))
        )
        model.forward_loss(
            source,
            pred,
            truth,
            mask.flatten(1),
            torch.ones_like(truth),
            no_gan=True,
            keep_loss_graph=True,
        )
        losses = [
            model.last_loss_graph[f"structure_{key}"]
            for key in ("row", "col", "centroid", "area")
        ]
        gradients, pixel_norms = [], []
        for loss in losses:
            pixel = torch.autograd.grad(loss, pred, retain_graph=True)[0]
            pixel_norms.append(pixel[:, :, 448:].float().norm().item())
            values = torch.autograd.grad(
                loss, parameters, retain_graph=True, allow_unused=True
            )
            gradients.append(
                [None if g is None else g.detach().float() for g in values]
            )
        gram = [
            [
                sum(
                    (a * b).sum().item()
                    for a, b in zip(left, right)
                    if a is not None and b is not None
                )
                for right in gradients
            ]
            for left in gradients
        ]
        all_norms = pixel_norms + [value for row in gram for value in row]
        if not all(math.isfinite(v) for v in all_norms):
            write_json(
                Path(args.output) / "calibration.json",
                {
                    "status": "blocked",
                    "reason": "Nonfinite calibration gradient",
                    "batch": batch,
                },
            )
            return
        cosines = [
            [
                gram[i][j] / max(math.sqrt(max(0, gram[i][i] * gram[j][j])), 1e-30)
                for j in range(4)
            ]
            for i in range(4)
        ]
        measurements.append(
            {
                "baseline_coefficients": (
                    [1.0 / pred.shape[-2], 1.0 / pred.shape[-1], 1.0, 1.0]
                    if model.structure_projection_mode == "sum"
                    else [1.0] * 4
                ),
                "batch": batch,
                "targets": [
                    r["target_path"] for r in records[2 * batch : 2 * batch + 2]
                ],
                "values": [loss.item() for loss in losses],
                "pixel_norms": pixel_norms,
                "parameter_gram": gram,
                "parameter_cosines": cosines,
                "parameter_norms": [math.sqrt(max(0, gram[i][i])) for i in range(4)],
                "combined_parameter_norm": math.sqrt(max(0, sum(map(sum, gram)))),
            }
        )
        del gradients, losses, pred
        model.last_loss_graph = {}
    from util.stage3_calibration import calibrate_measurements

    report = calibrate_measurements(measurements)
    report["structure_projection_mode"] = model.structure_projection_mode
    report["measurements"] = measurements
    report["checkpoint"] = args.checkpoint
    write_json(Path(args.output) / "calibration.json", report)
    print(report["status"], report.get("reason", ""))


def tensorboard_writer(args, rank):
    """Rank-0 mirror of train.jsonl and eval strips; jsonl stays the record."""
    if rank != 0 or not args.tensorboard:
        return None
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=str(Path(args.output) / "tensorboard"))


def log_train(writer, entry, optimizer, step):
    writer.add_scalar("train/loss_rank0", entry["loss_rank0"], step)
    writer.add_scalar("train/grad_pre_clip", entry["grad_pre_clip"], step)
    writer.add_scalar("train/grad_post_clip", entry["grad_post_clip"], step)
    writer.add_scalar("lr/max", max(g["lr"] for g in optimizer.param_groups), step)
    for key, value in entry["last_microbatch_losses"].items():
        writer.add_scalar(f"last_microbatch_losses/{key}", value, step)
    for group, row in entry.get("parameters", {}).items():
        writer.add_scalar(f"gradient_norm/{group}", row["gradient_norm"], step)
        writer.add_scalar(f"update_norm/{group}", row["update_norm"], step)


def log_eval(writer, results, directory, step, images=8):
    """Correct-reference metric means per split plus the first fixed strips."""
    import numpy as np
    from PIL import Image

    per_split = {}
    for row in results:
        if row["case"] == "correct":
            per_split.setdefault(row["split"], []).append(row)
    for split, rows in per_split.items():
        for key in rows[0]["metrics"]:
            mean = statistics.mean(r["metrics"][key] for r in rows)
            writer.add_scalar(f"eval_{split}/{key}", mean, step)
        for index, row in enumerate(rows[:images]):
            with Image.open(Path(directory) / row["image"]) as image:
                pixels = np.array(image.convert("RGB"))
            writer.add_image(
                f"eval_{split}/{index:02d}", pixels, step, dataformats="HWC"
            )


def train(model, config, rows, paths, args):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler

    from data.pairdataset import PairDataset
    from tools.check_loss_gradients import build_transform
    from util.masking_generator import MaskingGenerator
    from util.stage3_runtime import freeze, optimizer_groups, parameter_report

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if args.batch_size * args.accum_iter * world != 128:
        raise ValueError("batch_size * accum_iter * world_size must be 128")
    freeze(model)
    optimizer = torch.optim.AdamW(optimizer_groups(model), betas=(0.9, 0.999))
    writer = tensorboard_writer(args, rank)
    raw = model
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[args.local_rank])
    fit_records = balanced_records(rows["train"], 32, unique=True)
    if len({r["target_path"] for r in fit_records}) != 32 and args.fit32:
        raise ValueError("fit32 requires 32 distinct target images")
    if args.fit32:
        updates = min(args.updates, 300)
        loader = None
    else:
        updates = args.updates
        transform = build_transform()
        dataset = PairDataset(
            args.data_root,
            paths["train"],
            transform=transform,
            transform2=transform,
            transform3=transform,
            masked_position_generator=MaskingGenerator(
                (56, 28), num_masking_patches=784, max_num_patches=392
            ),
            half_mask_ratio=0.5,
            semantic_only_epochs=0,
            mask_mix_probs=[0.8, 0.0, 0.2],
            semantic_mask_dir=args.semantic_mask_dir,
            num_mask_annotations_bf=11,
            num_mask_annotations_jt=1,
            mask_coverage_threshold=0.1,
            strict_style_pairing=True,
        )
        sampler = DistributedSampler(
            dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed
        )
        loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=0,
        )
        if len(loader) == 0:
            raise ValueError("Not enough training samples")
        iterator = iter(loader)
    epoch = 0
    parameters = [p for p in raw.parameters() if p.requires_grad]
    for update in range(updates):
        seed_all(args.seed + 100003 * update + rank)
        model.train()
        raw.stage3_update = update
        schedule = (
            update / 40
            if update < 40
            else 0.5 * (1 + math.cos(math.pi * (update - 40) / max(1, updates - 40)))
        )
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * schedule
        optimizer.zero_grad(set_to_none=True)
        monitor = (update + 1) % 5 == 0 or update == 0
        before = (
            {
                n: p.detach().float().cpu().clone()
                for n, p in raw.named_parameters()
                if p.requires_grad
            }
            if monitor and rank == 0
            else None
        )
        mean_loss = 0.0
        for micro in range(args.accum_iter):
            if args.fit32:
                offset = (
                    update * 128
                    + rank * args.batch_size
                    + micro * args.batch_size * world
                )
                selected = [
                    fit_records[(offset + i) % 32] for i in range(args.batch_size)
                ]
                source, _context, target, mask = fixed_batch(
                    args.data_root, selected, rows["train"], args.device
                )
                # forward masks query GT; the style branch sees upper only.
                inputs = (source, target, mask, torch.ones_like(target))
            else:
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    sampler.set_epoch(epoch)
                    iterator = iter(loader)
                    batch = next(iterator)
                inputs = tuple(x.to(args.device) for x in batch)
            source, target, mask, valid = inputs
            sync = (
                model.no_sync()
                if world > 1 and micro + 1 < args.accum_iter
                else contextlib.nullcontext()
            )
            with sync:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=args.device.startswith("cuda"),
                ):
                    loss = model(source, target, mask, valid, no_gan=True)[0]
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                (loss / args.accum_iter).backward()
                mean_loss += loss.item() / args.accum_iter
        pre = torch.nn.utils.clip_grad_norm_(
            parameters, 3.0, error_if_nonfinite=True
        ).item()
        post = math.sqrt(
            sum(
                p.grad.float().square().sum().item()
                for p in parameters
                if p.grad is not None
            )
        )
        optimizer.step()
        if rank == 0:
            entry = {
                "update": update + 1,
                "loss_rank0": mean_loss,
                "grad_pre_clip": pre,
                "grad_post_clip": post,
                "detail_per_sample_normalize": False,
            }
            entry["last_microbatch_losses"] = {
                key: float(value) for key, value in raw.last_loss_components.items()
            }
            if monitor:
                entry["parameters"] = parameter_report(raw, optimizer, before)
            with (Path(args.output) / "train.jsonl").open("a") as handle:
                handle.write(json.dumps(entry, allow_nan=False) + "\n")
            if writer is not None:
                log_train(writer, entry, optimizer, update + 1)
        if (update + 1) % 50 == 0 or update + 1 == updates:
            if world > 1:
                dist.barrier()
            if rank == 0:
                torch.save(
                    {
                        "model": raw.state_dict(),
                        "stage3_config": config,
                        "update": update + 1,
                        "optimizer": optimizer.state_dict(),
                        "run_args": vars(args),
                    },
                    Path(args.output) / f"checkpoint-update-{update + 1}.pth",
                )
                eval_dir = Path(args.output) / f"eval-{update + 1}"
                results = evaluate(
                    raw, rows, args.data_root, eval_dir, args.device, all_cases=False
                )
                if writer is not None:
                    log_eval(writer, results, eval_dir, update + 1)
            if world > 1:
                dist.barrier()
    if writer is not None:
        writer.close()
    # Avoid accidental reuse of epoch-based continuation for these checkpoints.


def summarize(paths, output):
    """Aggregate correct-reference metrics, retaining seed/run dispersion."""
    groups = {}
    for path in paths:
        records = json.loads(Path(path).read_text())
        for row in records:
            if row["case"] != "correct":
                continue
            key = (str(path), row["split"])
            groups.setdefault(key, []).append(row["metrics"])
    report = []
    for (path, split), values in groups.items():
        report.append(
            {
                "run": path,
                "split": split,
                "count": len(values),
                "mean": {k: statistics.mean(v[k] for v in values) for k in values[0]},
                "std": {
                    k: statistics.stdev([v[k] for v in values])
                    if len(values) > 1
                    else 0.0
                    for k in values[0]
                },
            }
        )
    matched = {}
    for result in report:
        metric_path = Path(result["run"]).resolve()
        candidates = [
            metric_path.parent / "run.json",
            metric_path.parent.parent / "run.json",
        ]
        metadata_path = next((p for p in candidates if p.is_file()), None)
        if metadata_path is None:
            continue
        metadata = json.loads(metadata_path.read_text())
        seed = metadata["args"]["seed"]
        identity = json.dumps(
            {
                "config": metadata["config"],
                "initialization": metadata["args"]["checkpoint"],
                "data": metadata["data_sha256"],
                "budget": metadata["args"]["updates"],
                "evaluation": metric_path.parent.name,
                "split": result["split"],
            },
            sort_keys=True,
        )
        matched.setdefault(identity, {})[seed] = result["mean"]
    across_seeds = []
    for identity, seed_values in matched.items():
        values = list(seed_values.values())
        across_seeds.append(
            {
                "identity": json.loads(identity),
                "seeds": sorted(seed_values),
                "mean": {k: statistics.mean(v[k] for v in values) for k in values[0]},
                "seed_std": {
                    k: statistics.stdev(v[k] for v in values)
                    if len(values) > 1
                    else None
                    for k in values[0]
                },
            }
        )
    write_json(
        output,
        {
            "runs": report,
            "across_seeds": across_seeds,
            "decision": "Requires matched seeds and blind review",
        },
    )


def smoke(model, config, rows, args):
    """Full-model forward/backward and strict checkpoint reload on real data."""
    import torch

    from util.stage3_runtime import freeze, load_model, optimizer_groups

    freeze(model)
    model.eval()
    source, context, truth, mask = fixed_batch(
        args.data_root, rows["train"][:1], rows["train"], args.device
    )
    optimizer = torch.optim.AdamW(optimizer_groups(model))
    loss = model(source, truth, mask, torch.ones_like(truth), no_gan=True)[0]
    if not torch.isfinite(loss):
        raise FloatingPointError("Smoke loss is nonfinite")
    loss.backward()
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(params, 3.0, error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        expected = model.forward_decoder(
            model.forward_encoder(source, context, mask.flatten(1))
        )
    path = Path(args.output) / "smoke.pth"
    torch.save({"model": model.state_dict(), "stage3_config": config}, path)
    model.cpu()
    restored, _ = load_model(path)
    restored.to(args.device).eval()
    with torch.no_grad():
        actual = restored.forward_decoder(
            restored.forward_encoder(source, context, mask.flatten(1))
        )
    torch.testing.assert_close(expected, actual)
    write_json(
        Path(args.output) / "smoke.json",
        {
            "passed": True,
            "device": args.device,
            "loss": loss.item(),
            "pre_clip_grad_norm": norm.item(),
            "checkpoint_roundtrip": True,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["audit", "calibrate", "train", "evaluate", "summarize", "smoke"],
    )
    parser.add_argument("--manifest")
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", nargs="+")
    parser.add_argument(
        "--style-mode", choices=["off", "reference", "constant"], default="off"
    )
    parser.add_argument("--structure-weight", type=float, default=0.2)
    parser.add_argument("--detail-weight", type=float, default=0.2)
    parser.add_argument("--coefficients")
    parser.add_argument(
        "--structure-projection-mode", choices=["legacy", "sum"], default="sum"
    )
    parser.add_argument("--vgg-input-mode", choices=["rgb", "legacy"], default="legacy")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum-iter", type=int, default=32)
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--semantic-mask-dir")
    parser.add_argument("--fit32", action="store_true")
    parser.add_argument("--no-tensorboard", dest="tensorboard", action="store_false")
    args = parser.parse_args()
    if args.updates < 1 or args.batch_size < 1 or args.accum_iter < 1:
        parser.error("updates, batch-size and accum-iter must be positive")
    if args.command == "summarize":
        if not args.metrics:
            parser.error("--metrics required")
        summarize(args.metrics, args.output)
        return
    if not args.manifest or not args.data_root:
        parser.error("--manifest and --data-root required")
    import torch
    import torch.distributed as dist

    from util.stage3_data import audit, read_manifest
    from util.stage3_runtime import load_model

    rows, paths = read_manifest(args.manifest)
    audit_report = audit(rows, args.data_root)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    output = Path(args.output)
    if args.command == "audit":
        if world > 1:
            parser.error("audit must run in a single process")
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "audit.json", audit_report)
        return
    if not args.checkpoint:
        parser.error("--checkpoint required")
    if args.command == "train" and not torch.cuda.is_available():
        raise RuntimeError("Full training requires CUDA; run unit tests on CPU")
    if world > 1:
        if args.command != "train":
            raise ValueError("Only train supports torchrun")
        torch.cuda.set_device(args.local_rank)
        args.device = f"cuda:{args.local_rank}"
        dist.init_process_group("nccl")
    output_error = [None]
    if rank == 0:
        try:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "audit.json", audit_report)
        except OSError as error:
            output_error[0] = str(error)
    if world > 1:
        dist.broadcast_object_list(output_error, src=0)
    if output_error[0]:
        if dist.is_initialized():
            dist.destroy_process_group()
        raise RuntimeError(output_error[0])
    seed_all(args.seed)
    config = None if args.command == "evaluate" else config_from_args(args)
    model, config = load_model(args.checkpoint, config)
    model.to(args.device)
    if rank == 0:
        fingerprint = hashlib.sha256()
        for split in paths.values():
            for path in split:
                fingerprint.update(Path(path).read_bytes())
        write_json(
            output / "run.json",
            {
                "args": vars(args),
                "config": config,
                "manifest_sha256": fingerprint.hexdigest(),
                "data_sha256": audit_report["image_and_metadata_sha256"],
                "semantic_masks_hashed": False,
            },
        )
    if args.command == "evaluate":
        evaluate(model, rows, args.data_root, output, args.device)
    elif args.command == "calibrate":
        calibrate(model, rows, args)
    elif args.command == "smoke":
        smoke(model, config, rows, args)
    else:
        train(model, config, rows, paths, args)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
