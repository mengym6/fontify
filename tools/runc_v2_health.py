"""Fixed real-JT tiny overfit and exact frozen-state checks for Run C v2.

This is a disposable correctness run. It writes no model checkpoint and cannot
be used as the initialization of formal training. The two target/reference
pairs are sampled once and held fixed for all updates and before/after eval.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import runc_v2_runtime as runtime  # noqa: E402


def state_group(name):
    if name.startswith("dual_head."):
        return "head"
    if name.startswith("runc_v2_coarse."):
        return "coarse"
    if name.startswith("bf_refiner."):
        return "refiner"
    if name.startswith(("blocks.", "norm.", "patch_embed.")) or name in (
        "mask_token",
        "segment_token_x",
        "segment_token_y",
        "pos_embed",
    ):
        return "vit"
    return "legacy_unused"


def state_fingerprints(model):
    """Hash every parameter/buffer, including exact bytes of frozen tensors."""
    records = {}
    for kind, values in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, value in values:
            key = f"{kind}:{name}"
            records[key] = {
                "sha256": runtime.tensor_sha256(value),
                "group": state_group(name),
                "kind": kind,
                "elements": value.numel(),
                "trainable": bool(kind == "parameter" and value.requires_grad),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
    groups = {}
    for group in sorted({record["group"] for record in records.values()}):
        members = {key: row for key, row in records.items() if row["group"] == group}
        subsets = {}
        for subset in ("all", "frozen", "trainable"):
            selected = {
                key: row
                for key, row in members.items()
                if subset == "all" or row["trainable"] == (subset == "trainable")
            }
            digest = hashlib.sha256()
            for key, row in sorted(selected.items()):
                digest.update(key.encode("utf-8"))
                digest.update(bytes.fromhex(row["sha256"]))
            subsets[subset] = {
                "sha256": digest.hexdigest(),
                "tensor_count": len(selected),
                "elements": sum(row["elements"] for row in selected.values()),
            }
        groups[group] = subsets
    return {"groups": groups, "tensors": records}


def compare_fingerprints(before, after):
    if set(before["tensors"]) != set(after["tensors"]):
        raise RuntimeError(
            "Model parameter/buffer membership changed during health run"
        )
    changes = defaultdict(list)
    frozen_changed = []
    for key, initial in before["tensors"].items():
        final = after["tensors"][key]
        if initial["trainable"] != final["trainable"]:
            raise RuntimeError(f"Trainability changed mid-run: {key}")
        if initial["sha256"] != final["sha256"]:
            changes[initial["group"]].append(key)
            if not initial["trainable"]:
                frozen_changed.append(key)
    return {
        "all_frozen_parameters_and_buffers_unchanged": not frozen_changed,
        "frozen_changed": frozen_changed,
        "changed_tensor_names_by_group": dict(changes),
        "changed_tensor_counts_by_group": {
            key: len(value) for key, value in changes.items()
        },
    }


def fixed_author_indices(dataset, count=2):
    """Choose distinct source authors deterministically, without val targets."""
    authors = {}
    for index, pair in enumerate(dataset.pairs):
        if pair.get("has_jt"):
            authors.setdefault(pair["calligrapher"], index)
    if count < 2 or len(authors) < count:
        raise ValueError("Tiny health requires at least two different JT authors")
    return [authors[author] for author in sorted(authors)[:count]]


def slice_batch(batch, start, stop):
    """Slice only the collated batch dimension while retaining metadata."""
    return {
        key: value[start:stop]
        if torch.is_tensor(value) or isinstance(value, (list, tuple))
        else value
        for key, value in batch.items()
    }


def iter_microbatches(batch, size):
    count = int(batch["image"].shape[0])
    if size < 1:
        raise ValueError("Microbatch size must be positive")
    for start in range(0, count, size):
        stop = min(start + size, count)
        yield slice_batch(batch, start, stop), (stop - start) / count


def amp_context(device, amp):
    if device.type == "cuda" and amp:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def evaluate_fixed(model, wrapped, batch, config, device, microbatch_size):
    """Evaluate exactly the same held input tensors, without new augmentation."""
    model.eval()
    metrics = defaultdict(float)
    render = config["stage"] in ("ib", "ic")
    with torch.no_grad():
        for microbatch, weight in iter_microbatches(batch, microbatch_size):
            with amp_context(device, config["training"]["amp"]):
                outputs = runtime.call_model(wrapped, microbatch, render=render)
                losses = runtime.loss_for(outputs, microbatch, config)
            for key, value in runtime.scalar_losses(losses).items():
                metrics[key] += weight * value
            instance = runtime._head_instance_metrics(outputs, microbatch, losses)
            for key, values in instance.items():
                if values:
                    metrics[f"instances_{key}"] += weight * sum(values) / len(values)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise FloatingPointError("Nonfinite fixed-batch evaluation")
    return dict(metrics)


def run_updates(model, wrapped, batch, config, device, steps, microbatch_size):
    """Accumulate by actual sample counts, then perform one AdamW update."""
    optimizer, parameter_names = runtime.build_optimizer(model, config["training"])
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    render = config["stage"] in ("ib", "ic")
    logs = []
    for step in range(1, steps + 1):
        runtime.training_mode(model, config["stage"])
        optimizer.zero_grad(set_to_none=True)
        values = defaultdict(float)
        start = time.monotonic()
        for microbatch, weight in iter_microbatches(batch, microbatch_size):
            with amp_context(device, config["training"]["amp"]):
                outputs = runtime.call_model(wrapped, microbatch, render=render)
                losses = runtime.loss_for(outputs, microbatch, config)
            if not bool(torch.isfinite(losses["total"])):
                raise FloatingPointError("Nonfinite tiny objective")
            (losses["total"] * weight).backward()
            for key, value in runtime.scalar_losses(losses).items():
                values[key] += weight * value
        groups_before = runtime.grouped_gradient_norms(model)
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, config["training"]["clip_grad"]
        )
        if not torch.isfinite(norm) or norm <= 0:
            raise FloatingPointError("Tiny health gradient must be finite and nonzero")
        groups_after = runtime.grouped_gradient_norms(model)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row = {
            "optimizer_step": step,
            "losses": dict(values),
            "gradient_pre_clip": float(norm),
            "gradient_groups_pre_clip": groups_before,
            "gradient_groups_post_clip": groups_after,
            "seconds": time.monotonic() - start,
        }
        logs.append(row)
        print(
            json.dumps(
                {
                    "health_step": step,
                    "total": values["total"],
                    "head": values["raw_head"],
                    "head_geo": values["raw_head_geo"],
                    "gradient_norm": float(norm),
                }
            ),
            flush=True,
        )
    optimizer.zero_grad(set_to_none=True)
    return logs, parameter_names


def shape_tree(value):
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {key: shape_tree(child) for key, child in value.items()}
    return type(value).__name__


def profile_full_forward(model, wrapped, batch, config, device):
    """Measure a real full S/D0/O0/P0 forward; profiler FLOPs are incomplete."""
    model.eval()
    sample = slice_batch(batch, 0, 1)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    with (
        torch.no_grad(),
        torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            with_flops=True,
            profile_memory=True,
        ) as profiler,
    ):
        with amp_context(device, config["training"]["amp"]):
            outputs = runtime.call_model(wrapped, sample, render=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    events = profiler.key_averages(group_by_input_shape=True)
    operations = [
        {
            "operator": event.key,
            "flops": int(event.flops),
            "input_shapes": event.input_shapes,
        }
        for event in events
        if event.flops
    ]
    operations.sort(key=lambda item: -item["flops"])
    counted = sum(item["flops"] for item in operations)
    result = {
        "batch_size": 1,
        "render": True,
        "output_shapes": shape_tree(outputs),
        "profiler_counted_operator_flops": counted,
        "counted_flops_div2_mac_equivalent": counted / 2,
        "operator_rows": operations,
        "limitations": [
            "Counted profiler operations are a lower bound, "
            "not complete model FLOPs/MACs.",
            "Fused SDPA, softmax, LayerNorm, activations, interpolation "
            "and other operators "
            "may have zero profiler FLOP attribution.",
            "One full deployment forward is profiled; this is not training throughput.",
        ],
    }
    if device.type == "cuda":
        result.update(
            {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
        )
    del outputs
    return result


def save_plot(path, before, after, logs):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    x = [0] + [row["optimizer_step"] for row in logs]
    for key, label in (
        ("total", "total"),
        ("raw_head", "Head"),
        ("raw_head_geo", "Head geometry"),
    ):
        axes[0].plot(
            x,
            [before[key]] + [row["losses"][key] for row in logs],
            marker=".",
            label=label,
        )
    axes[0].set_title("Fixed two-target objective")
    axes[0].legend()
    for key, label in (
        ("raw_head_logsigma_x", "log sigma x"),
        ("raw_head_logsigma_y", "log sigma y"),
        ("raw_head_centroid", "relative centroid"),
        ("raw_head_shape", "shape"),
    ):
        axes[1].plot(
            [0, logs[-1]["optimizer_step"]],
            [before[key], after[key]],
            marker="o",
            label=label,
        )
    axes[1].set_title("Same-input eval before / after")
    axes[1].legend()
    axes[2].plot(
        [row["optimizer_step"] for row in logs],
        [row["gradient_pre_clip"] for row in logs],
        marker=".",
    )
    axes[2].set_title("Pre-clip parameter gradient norm")
    for axis in axes:
        axis.set_xlabel("Optimizer updates")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run(config, output, device, steps=12, microbatch_size=1, profile=True):
    """Run once into a new output directory and persist every health result."""
    from util.runc_v2 import runc_v2_parameter_report

    if steps < 1 or steps > 50:
        raise ValueError("Health updates must be between 1 and 50")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Health output must be empty; no overwrite/resume")
    output.mkdir(parents=True, exist_ok=True)
    runtime.write_json(output / "config.json", config)
    runtime.configure_backend(config)
    model, _, loading = runtime.load_model(config, device)
    runtime.write_json(output / "checkpoint_loading.json", loading)
    dataset = runtime.build_dataset(config)
    indices = fixed_author_indices(dataset)
    batch = runtime.collate_indices(
        dataset, indices, device, seed=config["training"]["seed"] + 18000
    )
    wrapped = runtime.wrap_forward(model)
    fixed = {
        "indices": indices,
        "authors": batch["author"],
        "characters": batch["character"],
        "target_paths": batch["target_path"],
        "reference_paths": batch["reference_path"],
        "geometry_crop": batch["geometry_crop"].cpu().tolist(),
        "image_sha256": runtime.tensor_sha256(batch["image"]),
        "input_target_sha256": runtime.tensor_sha256(batch["input_target"]),
        "target_sha256": runtime.tensor_sha256(batch["target"]),
        "supervision_is_model_input": False,
    }
    runtime.write_json(output / "fixed_batch_manifest.json", fixed)
    parameter_report = runc_v2_parameter_report(model)
    runtime.write_json(output / "parameter_report.json", parameter_report)
    initial = state_fingerprints(model)
    before = evaluate_fixed(model, wrapped, batch, config, device, microbatch_size)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    logs, parameter_names = run_updates(
        model, wrapped, batch, config, device, steps, microbatch_size
    )
    training_memory = None
    if device.type == "cuda":
        training_memory = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    after = evaluate_fixed(model, wrapped, batch, config, device, microbatch_size)
    profile_report = (
        profile_full_forward(model, wrapped, batch, config, device)
        if profile
        else {"status": "not_requested"}
    )
    final = state_fingerprints(model)
    comparison = compare_fingerprints(initial, final)
    runtime.write_json(
        output / "freeze_sha.json",
        {
            "before": initial,
            "after": final,
            "comparison": comparison,
        },
    )
    runtime.write_json(output / "profile.json", profile_report)
    decreasing = {
        key: after[key] < before[key] for key in ("total", "raw_head", "raw_head_geo")
    }
    frozen_ok = comparison["all_frozen_parameters_and_buffers_unchanged"]
    expected = {
        "ia0": ("head",),
        "ia1": ("head", "vit"),
        "ib": ("coarse",),
        "ic": ("head", "vit", "coarse"),
    }
    updated = comparison["changed_tensor_counts_by_group"]
    trained_groups_changed = all(
        updated.get(group, 0) > 0 for group in expected[config["stage"]]
    )
    trend_ok = all(decreasing.values()) if config["stage"] == "ia0" else None
    report = {
        "schema": "runc-v2-fixed-tiny-health-v1",
        "stage": config["stage"],
        "status": (
            "passed"
            if frozen_ok and trained_groups_changed and trend_ok is not False
            else "needs_diagnosis"
        ),
        "optimizer_steps": steps,
        "fixed_target_count": len(indices),
        "microbatch_size": microbatch_size,
        "effective_samples_per_update": len(indices),
        "precision": runtime.backend_record(device),
        "metadata": runtime.source_record(config),
        "before_fixed_eval": before,
        "after_fixed_eval": after,
        "decreasing": decreasing,
        "ia0_required_trend_passed": trend_ok,
        "frozen_state_passed": frozen_ok,
        "trained_groups_changed": trained_groups_changed,
        "optimizer_parameter_names": parameter_names,
        "steps": logs,
        "training_cuda_memory": training_memory,
        "purpose": "Correctness only; no checkpoint written, no formal initialization.",
        "limitations": [
            "Two train targets, one seed; no validation/generalization evidence.",
            "Before/after metrics reuse one fixed target/reference/augmentation batch.",
            "Single-process health does not validate DDP; use separate NCCL smoke.",
        ],
    }
    runtime.write_json(output / "smoke.json", report)
    save_plot(output / "tiny_overfit.png", before, after, logs)
    (output / "smoke.md").write_text(
        "# Run C v2 fixed real-JT tiny health\n\n"
        f"Stage={config['stage']}, status={report['status']}, optimizer steps={steps}. "
        f"Two targets/authors: {fixed['authors']} / {fixed['characters']}. "
        "Reference and augmentation were sampled once and the same tensor batch "
        "was reused throughout. Each microbatch contributes its actual target "
        "count divided by two. No checkpoint is saved or reused.\n\n"
        f"Before objective/H/Hgeo: {before['total']:.6f} / "
        f"{before['raw_head']:.6f} / {before['raw_head_geo']:.6f}; "
        f"after: {after['total']:.6f} / {after['raw_head']:.6f} / "
        f"{after['raw_head_geo']:.6f}. Decreases: {decreasing}.\n\n"
        f"Frozen parameters and buffers unchanged={frozen_ok}; expected active "
        f"groups changed={trained_groups_changed}. Exact group/tensor hashes "
        "are in freeze_sha.json; gradient norms and loss axes are in smoke.json.\n\n"
        "Full S/D0/O0/P0 forward shapes, parameter counts, measured CUDA peak "
        "memory and profiler-counted operator FLOPs are recorded separately. "
        "Fused SDPA and unsupported operators may be uncounted; this is not "
        "a complete MACs/FLOPs claim. The health run provides no quality "
        "or independent-generalization conclusion.\n",
        encoding="utf-8",
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--stage", choices=("ia0", "ia1", "ib", "ic"))
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--no-profile", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.stage:
        config["stage"] = args.stage
    config = runtime.validate_config(config, "smoke")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    report = run(
        config,
        Path(args.output),
        device,
        args.steps,
        args.microbatch_size,
        not args.no_profile,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "stage",
                    "optimizer_steps",
                    "decreasing",
                    "frozen_state_passed",
                )
            }
        ),
        flush=True,
    )
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
