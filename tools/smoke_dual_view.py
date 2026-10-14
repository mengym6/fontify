"""Real-data CUDA/DDP run B smoke; execute only after the GPU is available.

This is an isolated diagnostic run: it never saves or updates its input
checkpoint. Every distributed optimizer update averages 64 target characters;
mixed updates rotate across all three pools. Gradient-route probes use small
independent batches without optimizer updates.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EFFECTIVE_BATCH_SIZE = 64
POOL_TYPES = ("bf_only", "jt_only", "dual")


def update_batch_layout(world_size, batch_size):
    """Return physical microbatch counts for one global 64-target update."""
    if type(world_size) is not int or world_size not in (1, 2, 4):
        raise ValueError("world-size must be an integer in {1, 2, 4}")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch-size must be a positive integer")
    count = EFFECTIVE_BATCH_SIZE // world_size
    return [min(batch_size, count - start)
            for start in range(0, count, batch_size)]


def microbatch_fractions(counts):
    """Weight local mean losses by their share of the local target count."""
    if not counts or any(type(count) is not int or count < 1 for count in counts):
        raise ValueError("microbatch counts must be positive integers")
    total = sum(counts)
    return [count / total for count in counts]


def pool_sequence(category, count, iteration=0, rank=0):
    """Keep the exact target count and rotate mixed-update pool imbalance."""
    if category not in (*POOL_TYPES, "mixed"):
        raise ValueError(f"Unknown smoke pool: {category}")
    if type(count) is not int or count < 1:
        raise ValueError("target count must be a positive integer")
    if category != "mixed":
        return [category] * count
    offset = (iteration + rank) % len(POOL_TYPES)
    return [POOL_TYPES[(offset + index) % len(POOL_TYPES)]
            for index in range(count)]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-path", required=True)
    result.add_argument("--json-path", nargs="+", required=True)
    result.add_argument("--target-cache", required=True)
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--train-calibration")
    result.add_argument("--allow-new-modules", action="store_true",
                        help="Explicitly migrate a backbone-only checkpoint")
    result.add_argument("--output-json", required=True)
    result.add_argument("--batch-size", type=int, default=2)
    result.add_argument("--iterations", type=int, default=2)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument('--epoch', type=int, default=0,
                        help='use 18 to exercise the final tip strength')
    return result


def gradient_groups(model):
    groups = {name: [] for name in ("encoder", "decoder", "fusion", "head")}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("dual_head."):
            group = "head"
        elif name.startswith("dual_fusion."):
            group = "fusion"
        elif name.startswith("decoder_"):
            group = "decoder"
        elif name.startswith(("discriminator.", "vgg_loss.")):
            continue
        else:
            group = "encoder"
        groups[group].append(parameter)
    return groups


def route_probe(model, batch, forward_terms):
    """Probe the unwrapped model BEFORE constructing DDP's reducer."""
    import torch

    extra = forward_terms(model, batch)
    groups = gradient_groups(model)
    names = list(groups)
    parameters = [parameter for name in names for parameter in groups[name]]
    active = bool(batch["has_jt"].any())
    expected = {
        "L_pen": (True, True, False, False),
        "L_full": (True, True, True, active),
        "J": (active, active, active, active),
        "L_H": (active, False, False, active),
    }
    records = {}
    for index, (term, wanted) in enumerate(expected.items()):
        gradients = torch.autograd.grad(
            extra[term], parameters, retain_graph=index < 3, allow_unused=True,
        )
        offset, norms = 0, {}
        for group in names:
            length = len(groups[group])
            current = gradients[offset:offset + length]
            total = sum(gradient.detach().double().square().sum()
                        for gradient in current if gradient is not None)
            norms[group] = float(torch.sqrt(total)) if torch.is_tensor(total) else 0
            offset += length
        for group, should_have_gradient in zip(names, wanted):
            if not torch.isfinite(torch.tensor(norms[group])):
                raise RuntimeError(f"Nonfinite {term}/{group} probe")
            if (norms[group] > 0) != should_have_gradient:
                raise RuntimeError(
                    f"Gradient route mismatch {term}/{group}: {norms[group]}; "
                    "check shielding and whether the fusion projection opened"
                )
        records[term] = norms
    return records


def main(argv=None):
    args = parser().parse_args(argv)
    if args.batch_size < 1 or args.iterations < 2:
        raise ValueError("batch-size >= 1 and at least two iterations required")
    destination = Path(args.output_json)
    if destination.exists():
        raise FileExistsError(destination)
    import numpy as np
    import torch

    from tools.dual_view_protocol import (
        apply_weights,
        build_dataset,
        cache_hash,
        forward_terms,
        hash_files,
        load_dual_model,
        seed_all,
        validate_calibration,
    )
    base_forward_terms = forward_terms

    def forward_terms(model, batch):
        return base_forward_terms(model, batch, epoch=args.epoch)

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Real-model smoke requires CUDA with bf16 support")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    counts = update_batch_layout(world, args.batch_size)
    samples_per_rank = sum(counts)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(
        "nccl", init_method="env://" if world > 1 else
        f"tcp://127.0.0.1:{os.environ.get('MASTER_PORT', '29591')}",
        rank=rank, world_size=world,
    )
    try:
        seed_all(args.seed + rank)
        dataset = build_dataset(args)
        model, _ = load_dual_model(
            args.checkpoint, device, args.seed,
            allow_new=args.allow_new_modules,
        )
        calibrated = False
        if args.train_calibration:
            calibration = validate_calibration(args.train_calibration, {
                "train_manifest_sha256": hash_files(args.json_path),
                "target_cache_sha256": cache_hash(args.target_cache),
            })
            apply_weights(model, calibration)
            calibrated = True
        pools = {
            name: [index for index, record in enumerate(dataset.pairs)
                   if record["pool_type"] == name]
            for name in POOL_TYPES
        }
        if any(not indices for indices in pools.values()):
            raise ValueError("Smoke requires nonempty BF-only, JT-only and dual pools")
        rng = random.Random(args.seed + rank)

        def make_batches(category, iteration, count):
            kinds = pool_sequence(category, count, iteration, rank)
            for start in range(0, count, args.batch_size):
                rows = []
                for offset, kind in enumerate(
                    kinds[start:start + args.batch_size], start=start
                ):
                    index = rng.choice(pools[kind])
                    record = dataset.pairs[index]
                    references = dataset.pair_type_dict[
                        (record["calligrapher"], record["pool_type"])
                    ]
                    rows.append(dataset.get_pair(
                        index, rng.choice(references),
                        args.seed + rank * 10000 + iteration * 100 + offset,
                    ))
                batch = torch.utils.data.default_collate(rows)
                yield {key: value.to(device) if torch.is_tensor(value) else value
                       for key, value in batch.items()}

        model.train()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1.95699e-3, betas=(0.9, 0.999), weight_decay=0.1,
        )
        # This diagnostic opens the zero projection before DDP construction.
        # Each rank accumulates 64/world targets, but there is no all-reduce:
        # the true local opening batch is 64/world, not a global batch of 64.
        optimizer.zero_grad(set_to_none=True)
        for batch, fraction in zip(
            make_batches("dual", 0, samples_per_rank), microbatch_fractions(counts)
        ):
            opened = forward_terms(model, batch)
            total = (opened["L_pen"] + model.dual_lambda_full * opened["L_full"]
                     + model.jieti_loss_weight * opened["J"]
                     + model.dual_head_weight * opened["L_H"])
            if not torch.isfinite(total):
                raise RuntimeError("Nonfinite fusion-opening loss")
            (total * fraction).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if not torch.count_nonzero(model.dual_fusion.projection.weight):
            raise RuntimeError("Fusion projection remained zero after an update")
        probes = {}
        for category in (*pools, "mixed"):
            probe_count = 3 if category == "mixed" else args.batch_size
            probes[category] = [route_probe(model, batch, forward_terms)
                                for batch in make_batches(category, 1, probe_count)]

        # Every rank now starts from rank zero's updated parameters; DDP's
        # construction broadcasts them. The optimizer state remains local
        # smoke state and does not enter any formal-training checkpoint.
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1.95699e-3, betas=(0.9, 0.999), weight_decay=0.1,
        )
        wrapped = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], static_graph=True,
        )
        records = []
        for category in (*pools, "mixed"):
            for iteration in range(args.iterations):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                value = 0.0
                seen = 0
                for batch, fraction in zip(
                    make_batches(category, iteration + 2, samples_per_rank),
                    microbatch_fractions(counts),
                ):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        output = wrapped(
                            batch["image"].to(torch.bfloat16),
                            batch["target"].to(torch.bfloat16),
                            batch["mask"].flatten(1).to(torch.bfloat16),
                            batch["valid"].to(torch.bfloat16),
                            no_gan=True, dual_targets=batch, epoch=args.epoch,
                        )
                    if not torch.isfinite(output[0]):
                        raise RuntimeError("Nonfinite smoke loss")
                    seen += len(batch["image"])
                    (output[0] * fraction).backward()
                    value += float(output[0].detach()) * fraction
                if seen * world != EFFECTIVE_BATCH_SIZE:
                    raise RuntimeError("Smoke update did not cover exactly 64 targets")
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
                if not torch.isfinite(norm):
                    raise RuntimeError("Nonfinite smoke gradient norm")
                optimizer.step()
                torch.cuda.synchronize(device)
                record = {
                    "category": category, "iteration": iteration,
                    "microbatches": len(counts), "samples_per_rank": seen,
                    "effective_batch_size": seen * world,
                    "pool_counts_per_rank": {
                        name: pool_sequence(
                            category, seen, iteration + 2, rank
                        ).count(name) for name in POOL_TYPES
                    },
                    "loss": value, "seconds": time.perf_counter() - started,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "pre_clip_norm": float(norm),
                    "clip_ratio": min(1.0, 3.0 / (float(norm) + 1e-6)),
                    "clipped": bool(norm > 3),
                }
                records.append(record)
                print(json.dumps({"rank": rank, **record}), flush=True)
        ratios = [record["clip_ratio"] for record in records]
        report = {
            "status": "passed", "rank": rank, "world_size": world,
            "real_model": True, "real_dataset": True, "ddp_static_graph": True,
            "weights_calibrated": calibrated, "checkpoint": args.checkpoint,
            "batch_size": args.batch_size, "probes_after_fusion_open": probes,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "fusion_opening": {
                "diagnostic_only": True, "all_reduce": False,
                "samples_per_rank": samples_per_rank,
                "nominal_targets_across_ranks": EFFECTIVE_BATCH_SIZE,
                "local_microbatches": len(counts),
            },
            "route_probes": {"diagnostic_only": True, "optimizer_updates": 0},
            "records": records,
            "clip_trigger_rate": sum(row["clipped"] for row in records) / len(records),
            "clip_ratio_median": float(np.median(ratios)),
            "clip_ratio_p10": float(np.quantile(ratios, 0.1)),
        }
        gathered = [None] * world if rank == 0 else None
        torch.distributed.gather_object(report, gathered, dst=0)
        if rank == 0:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as handle:
                json.dump({"status": "passed", "ranks": gathered}, handle,
                          ensure_ascii=False, indent=2)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
