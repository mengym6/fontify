"""Independent full-model Run C CUDA correctness checks, without long training.

Run each phase in a fresh process to bound peak GPU memory. These measurements
verify implementation contracts; conditioner correctness steps do not establish
training benefit. The script never authorizes or launches a formal experiment.
"""

import argparse
import copy
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASES = ("legacy", "coarse_grad", "refine_grad", "safety", "condition")
PREFIX_GROUPS = {
    "head": ("dual_head.",),
    "conditioner": ("structure_adapter.",),
    "fusion": ("dual_fusion.",),
    "decoder": ("decoder_embed.", "decoder_pred."),
    "refiner": ("bf_refiner.",),
    "vit": ("blocks.", "patch_embed.", "norm."),
}


def require(condition, message):
    """Fail a scientific contract instead of only recording a warning."""
    if not condition:
        raise AssertionError(message)


def tensor_digest(value):
    """Hash exact bytes, dtype and dimensions, including bfloat16 tensors."""
    import torch

    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_digest(model, exclude=()):
    """Hash parameters and persistent buffers without retaining a model copy."""
    digest = hashlib.sha256()
    count = 0
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(exclude):
            continue
        digest.update(name.encode())
        digest.update(bytes.fromhex(tensor_digest(value)))
        count += value.numel()
    return {"sha256": digest.hexdigest(), "elements": count}


def gradient_measurement(value, model):
    """Compute raw parameter gradients, grouping by the actual named modules."""
    import torch

    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    gradients = (
        torch.autograd.grad(
            value,
            [parameter for _, parameter in named],
            retain_graph=True,
            allow_unused=True,
        )
        if value.requires_grad
        else [None] * len(named)
    )
    squares = dict.fromkeys(PREFIX_GROUPS, 0.0)
    nonzero = dict.fromkeys(PREFIX_GROUPS, 0)
    used = dict.fromkeys(PREFIX_GROUPS, 0)
    all_square = 0.0
    for (name, _), gradient in zip(named, gradients):
        if gradient is None:
            continue
        require(
            bool(torch.isfinite(gradient).all()),
            f"nonfinite gradient for {name}",
        )
        square = float(gradient.float().square().sum())
        all_square += square
        for group, prefixes in PREFIX_GROUPS.items():
            if name.startswith(prefixes):
                squares[group] += square
                used[group] += 1
                nonzero[group] += int(bool(torch.count_nonzero(gradient)))
    return {
        "norm": math.sqrt(all_square),
        "group_norm": {
            key: math.sqrt(value) for key, value in squares.items()
        },
        "used_parameter_tensors": used,
        "nonzero_parameter_tensors": nonzero,
    }


def backend_record():
    import torch

    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "gpu": (
            torch.cuda.get_device_name() if torch.cuda.is_available() else None
        ),
    }


def configure_backend():
    import torch

    # Match the sealed pre-pull native FP32 captures exactly.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False


def autocast(device, enabled):
    import torch

    return (
        torch.autocast(device.type, dtype=torch.bfloat16)
        if enabled
        else (nullcontext())
    )


def common_model(config, stage, device):
    from tools.structure_first import load_model, training_mode

    config = copy.deepcopy(config)
    config["stage"] = stage
    model, saved = load_model(config, device)
    training_mode(model, stage)
    return config, model, saved


def data_and_cases(config, device, stage):
    from tools.structure_first import (
        build_dataset,
        collate_indices,
        pool_indices,
    )

    config = copy.deepcopy(config)
    config["stage"] = stage
    dataset = build_dataset(config)
    pools = pool_indices(dataset)
    required = (
        ("dual", "bf_only", "jt_only")
        if stage == "coarse"
        else (
            "dual",
            "bf_only",
        )
    )
    chosen = {}
    for pool in required:
        require(bool(pools[pool]), f"missing real-data pool: {pool}")
        for index in pools[pool]:
            batch = collate_indices(dataset, [index], device, seed=1729)
            if pool == "bf_only" or int(batch["valid_parts"].sum()) >= 2:
                chosen[pool] = index
                break
        require(pool in chosen, f"no {pool} sample with two usable JT parts")
    cases = [(pool, [index]) for pool, index in chosen.items()]
    if stage == "coarse":
        cases.append(
            ("mixed", [chosen["bf_only"], chosen["jt_only"], chosen["dual"]])
        )
    else:
        cases.append(("mixed", [chosen["bf_only"], chosen["dual"]]))
    return dataset, cases, {key: len(value) for key, value in pools.items()}


def legacy_check(config, output, device, args):
    import torch

    from tools.structure_first import BASELINE_SHA256, sha256
    from util.checkpoint import build_checkpoint_model
    from util.structure_first import enable_structure_first, structure_forward

    require(not args.amp, "legacy native FP32 comparison cannot use AMP")
    require(
        sha256(config["checkpoint"]) == BASELINE_SHA256,
        "legacy parity requires the sealed Run B Fix checkpoint",
    )
    capture = Path(args.legacy_dir)
    record = json.loads((capture / "metadata.json").read_text())
    actual_backend = backend_record()
    for key in (
        "torch",
        "cuda",
        "cudnn",
        "matmul_tf32",
        "cudnn_tf32",
        "benchmark",
    ):
        require(
            actual_backend[key] == record[key],
            f"legacy backend differs for {key}",
        )
    saved = torch.load(
        config["checkpoint"], map_location="cpu", weights_only=False
    )
    model = build_checkpoint_model(saved).to(device).eval()
    require(
        not hasattr(model, "structure_adapter"),
        "legacy-off model unexpectedly has new module parameters",
    )
    rows = []
    with torch.no_grad():
        for row in record["rows"]:
            tensors = torch.load(
                capture / f"legacy_{row['index']:03d}.pt",
                map_location="cpu",
                weights_only=False,
            )
            images, target = tensors["image"].to(device), tensors["target"].to(
                device
            )
            prediction = model.predict_full(images, target).cpu()
            expected = tensors["prediction"]
            require(
                torch.equal(prediction, expected),
                f"legacy-off changed native prediction index={row['index']}",
            )
            rows.append(
                {
                    "index": row["index"],
                    "legacy_off_equal": True,
                    "max_abs": float((prediction - expected).abs().max()),
                    "prediction_tensor_sha256": tensor_digest(prediction),
                }
            )
    enable_structure_first(model, config.get("model", {}))
    model.eval()
    with torch.no_grad():
        for row in rows:
            tensors = torch.load(
                capture / f"legacy_{row['index']:03d}.pt",
                map_location="cpu",
                weights_only=False,
            )
            images, target = tensors["image"].to(device), tensors["target"].to(
                device
            )
            prediction = structure_forward(model, images, target, "coarse")[
                "p0"
            ]
            require(
                torch.equal(prediction.cpu(), tensors["prediction"]),
                "zero-initialized Run C conditioner changed Run B",
            )
            row["zero_conditioner_equal"] = True
    return {
        "checkpoint_sha256": BASELINE_SHA256,
        "before_pull_metadata": record,
        "checks": rows,
    }


def gradient_check(config, output, device, args, stage):
    import torch

    from tools.structure_first import (
        call_model,
        collate_indices,
        loss_for,
        save_visual,
    )

    config, model, _ = common_model(config, stage, device)
    dataset, cases, pool_counts = data_and_cases(config, device, stage)
    terms = (
        ("raw_low", "raw_j0", "raw_head")
        if stage == "coarse"
        else (
            "raw_img_smooth",
            "raw_img_gram",
            "raw_img_edge",
            "raw_bf_smooth",
            "raw_bf_edge",
            "raw_j1",
        )
    )
    records = []
    for pool, indices in cases:
        model.zero_grad(set_to_none=True)
        batch = collate_indices(dataset, indices, device, seed=1729)
        with autocast(device, args.amp):
            outputs = call_model(model, batch, stage)
            losses = loss_for(model, outputs, batch, config)
        measurements = {}
        for term in (*terms, "total"):
            require(
                bool(torch.isfinite(losses[term])), f"nonfinite {pool}/{term}"
            )
            measurements[term] = gradient_measurement(losses[term], model)
        if stage == "coarse":
            require(
                measurements["raw_low"]["group_norm"]["decoder"] > 0,
                "coarse image reconstruction has no decoder gradient",
            )
            require(
                measurements["raw_low"]["group_norm"]["conditioner"] > 0,
                "coarse image reconstruction has no conditioner gradient",
            )
            if pool == "bf_only":
                require(
                    measurements["raw_low"]["group_norm"]["head"] == 0,
                    "BF-only image loss updates the JT head",
                )
                for term in ("raw_j0", "raw_head"):
                    require(
                        float(losses[term].detach()) == 0
                        and measurements[term]["norm"] == 0,
                        f"BF-only {term} must be zero",
                    )
            else:
                require(
                    measurements["raw_head"]["group_norm"]["head"] > 0,
                    "JT supervision has no head gradient",
                )
                require(
                    measurements["raw_j0"]["norm"] > 0,
                    "JT geometry has no coarse gradient",
                )
        else:
            for term in terms[:5]:
                require(
                    measurements[term]["group_norm"]["refiner"] > 0,
                    f"refine {term} has no effective refiner gradient",
                )
            require(
                not outputs["p0"].requires_grad,
                "stage-II P0 must be generated under no_grad",
            )
            require(
                all(
                    not value.requires_grad
                    for value in outputs["head"].values()
                ),
                "stage-II predicted S/Q must be detached",
            )
            require(
                all(
                    not p.requires_grad
                    for name, p in model.named_parameters()
                    if not name.startswith("bf_refiner.")
                ),
                "stage-II unfroze a stage-I parameter",
            )
            jnorm = measurements["raw_j1"]["group_norm"]["refiner"]
            require(
                jnorm == 0 if pool == "bf_only" else jnorm > 0,
                f"incorrect stage-II J gate/gradient for {pool}",
            )
        losses["total"].backward()
        require(
            all(
                p.grad is None or not bool(torch.count_nonzero(p.grad))
                for name, p in model.named_parameters()
                if (
                    not name.startswith("bf_refiner.")
                    if stage == "refine"
                    else name.startswith("bf_refiner.")
                )
            ),
            "gradient appeared in the inactive stage",
        )
        require(
            all(
                p.grad is None or bool(torch.isfinite(p.grad).all())
                for p in model.parameters()
            ),
            "nonfinite total-loss backward",
        )
        if pool != "mixed":
            save_visual(
                output / f"{stage}-{pool}.png",
                outputs,
                batch,
                config["lowpass"],
            )
        mask_stats = batch.get("bf_mask_stats", {})
        records.append(
            {
                "pool": pool,
                "indices": indices,
                "target_path": batch.get("target_path"),
                "reference_path": batch.get("reference_path"),
                "count_jt": int(batch["has_jt"].sum()),
                "count_bf": int(batch["has_bf"].sum()),
                "count_valid_parts": int(batch["valid_parts"].sum()),
                "losses": {
                    term: float(losses[term].detach())
                    for term in (*terms, "total")
                },
                "gradients": measurements,
                "bf_mask_stats": {
                    key: value.detach().cpu().tolist()
                    for key, value in mask_stats.items()
                },
            }
        )
        print(
            json.dumps({"stage": stage, "pool": pool, "status": "pass"}),
            flush=True,
        )
        del outputs, losses, batch
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "stage": stage,
        "pool_counts": pool_counts,
        "checks": records,
        "weights_status": "correctness candidates, not calibrated",
    }


def compare_predictions(left, right):
    """Use exact equality for model outputs and selected masks."""
    import torch

    for key in ("p0", "p1", "mask", "masked_p0"):
        if key in left:
            require(
                torch.equal(left[key], right[key]),
                f"GT perturbation or repeat changed {key}",
            )
    for key, value in left["head"].items():
        require(
            torch.equal(value, right["head"][key]),
            f"GT perturbation or repeat changed head.{key}",
        )
    require(len(left["masks"]) == len(right["masks"]), "different mask rounds")
    for a, b in zip(left["masks"], right["masks"]):
        require(torch.equal(a, b), "different per-round reconstruction mask")


def safety_check(config, output, device, args):
    import torch

    from tools.structure_first import (
        assert_optimizer,
        collate_indices,
        loss_for,
        save_visual,
    )
    from util.checkpoint import build_checkpoint_model
    from util.structure_first import structure_forward

    config, model, saved = common_model(config, "refine", device)
    dataset, cases, pool_counts = data_and_cases(config, device, "refine")
    index = dict(cases)["dual"]
    batch = collate_indices(dataset, index, device, seed=1729)
    before = state_digest(model, exclude=("bf_refiner.",))
    trainable = [
        name for name, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.01,
    )
    assert_optimizer(model, optimizer, "refine")
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, args.amp):
            outputs = structure_forward(
                model,
                batch["image"],
                batch["target"],
                "refine",
                bf_mask=batch["bf_patch_mask"],
            )
            losses = loss_for(model, outputs, batch, config)
        if step == 0:
            require(
                torch.equal(outputs["p0"], outputs["p1"]),
                "zero-initialized BF Refiner changed P0 before optimization",
            )
        losses["total"].backward()
        require(
            all(
                p.grad is None
                for name, p in model.named_parameters()
                if not name.startswith("bf_refiner.")
            ),
            "stage-I parameter received gradient",
        )
        require(
            any(
                p.grad is not None and bool(torch.count_nonzero(p.grad))
                for p in model.bf_refiner.parameters()
            ),
            "BF Refiner did not receive effective gradient",
        )
        torch.nn.utils.clip_grad_norm_(
            model.bf_refiner.parameters(), 3.0, error_if_nonfinite=True
        )
        optimizer.step()
    after = state_digest(model, exclude=("bf_refiner.",))
    require(
        before == after, "stage-I parameter/buffer bytes changed after AdamW"
    )
    model.eval()
    # Deliberately bypass the runner's pre-sanitized input_target. The model API
    # itself must reject leakage even if lower target pixels contain NaN/Inf.
    checks = []
    with torch.no_grad(), autocast(device, args.amp):
        for mode in ("coarse", "refine", "infer"):
            kwargs = (
                {"bf_mask": batch["bf_patch_mask"]} if mode == "refine" else {}
            )
            baseline = structure_forward(
                model, batch["image"], batch["target"], mode, **kwargs
            )
            repeat = structure_forward(
                model, batch["image"], batch["target"], mode, **kwargs
            )
            compare_predictions(baseline, repeat)
            for perturbation in ("nan", "random"):
                changed = batch["target"].clone()
                lower = changed[:, :, changed.shape[-2] // 2 :]
                lower.copy_(
                    torch.full_like(lower, float("nan"))
                    if (perturbation == "nan")
                    else torch.randn_like(lower) * 100
                )
                actual = structure_forward(
                    model, batch["image"], changed, mode, **kwargs
                )
                compare_predictions(baseline, actual)
                checks.append(
                    {
                        "stage": mode,
                        "perturbation": perturbation,
                        "exact_equal": True,
                        "annotation_inputs": (
                            ["bf_patch_mask"] if mode == "refine" else []
                        ),
                    }
                )
            if mode == "infer":
                save_visual(
                    output / "annotation-free-inference.png", baseline, batch
                )
                expected = baseline["p1"].detach().cpu()
                expected_mask = baseline["mask"].cpu()
    checkpoint = output / "strict-roundtrip.pth"
    payload = {
        "model": model.state_dict(),
        "args": saved.get("args"),
        "dual_view_config": saved["dual_view_config"],
        "structure_first_config": dict(model.structure_first_config),
    }
    with checkpoint.open("xb") as stream:
        torch.save(payload, stream)
    expected_state = state_digest(model)
    del (
        outputs,
        losses,
        baseline,
        repeat,
        actual,
        model,
        saved,
        payload,
        optimizer,
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    saved_again = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    restored = build_checkpoint_model(saved_again).to(device).eval()
    require(
        state_digest(restored) == expected_state,
        "strict roundtrip changed state bytes",
    )
    with torch.no_grad(), autocast(device, args.amp):
        restored_output = structure_forward(
            restored, batch["image"], batch["target"], "infer"
        )
    require(
        torch.equal(restored_output["p1"].cpu(), expected),
        "strict checkpoint roundtrip changed prediction",
    )
    require(
        torch.equal(restored_output["mask"].cpu(), expected_mask),
        "strict checkpoint roundtrip changed automatic mask",
    )
    return {
        "pool_counts": pool_counts,
        "stage_one_state_before": before,
        "stage_one_state_after": after,
        "stage_one_exact_unchanged": True,
        "optimizer_steps": 2,
        "trainable_parameters": trainable,
        "checks": checks,
        "strict_roundtrip_equal": True,
        "strict_roundtrip_state": expected_state,
        "checkpoint": str(checkpoint),
        "annotation_free_inference": True,
    }


def condition_check(config, output, device, args):
    import torch
    from torch.nn import functional as F

    from tools.structure_first import collate_indices
    from util.structure_first_loss import lowpass_target

    _, model, _ = common_model(config, "coarse", device)
    dataset, cases, pool_counts = data_and_cases(config, device, "coarse")
    batch = collate_indices(dataset, dict(cases)["dual"], device, seed=1729)
    model.eval()
    model.requires_grad_(False)
    model.structure_adapter.requires_grad_(True)
    with torch.no_grad(), autocast(device, args.amp):
        safe = batch["target"].clone()
        safe[:, :, safe.shape[-2] // 2 :] = 0
        taps, quadrants = model.forward_encoder(
            batch["image"],
            safe,
            model.full_target_mask(batch["image"]),
            return_quadrants=True,
        )
        features = torch.cat(taps, -1)
        half = features.shape[1] // 2
        lower, reference = features[:, half:], features[:, :half]
        head = {
            key: value.detach()
            for key, value in model.dual_head(
                quadrants,
                lower,
            ).items()
        }
        base_residual = model.dual_fusion(head["layout"])
    variants = {}
    for key in ("occupancy_logits", "existence_logits"):
        changed = dict(head)
        value = head[key]
        changed[key] = -value + 4.0
        variants[key] = changed

    def responses():
        with torch.no_grad(), autocast(device, args.amp):
            base = model.structure_adapter(lower, reference, head)
            values = {}
            for key, changed in variants.items():
                require(
                    torch.equal(head["query"], changed["query"]),
                    "Q changed in S test",
                )
                changed_output = model.structure_adapter(
                    lower, reference, changed
                )
                delta = (changed_output.float() - base.float()).abs()
                values[key] = {
                    "max_abs": float(delta.max()),
                    "mean_abs": float(delta.mean()),
                }
            return values

    initial = responses()
    require(
        all(value["max_abs"] == 0 for value in initial.values()),
        "zero-initialized conditioner must have zero S response",
    )
    optimizer = torch.optim.AdamW(
        model.structure_adapter.parameters(), lr=1e-4, weight_decay=0.0
    )
    target = lowpass_target(batch["target"][:, :, 448:], config["lowpass"])
    steps = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, args.amp):
            residual = model.structure_adapter(lower, reference, head)
            prediction = model.forward_decoder(
                taps, lower_residual=base_residual + residual
            )
            loss = F.smooth_l1_loss(prediction[:, :, 448:].float(), target)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.structure_adapter.parameters(), 3.0, error_if_nonfinite=True
        )
        require(
            float(norm) > 0, "conditioner correctness step has no gradient"
        )
        optimizer.step()
        steps.append(
            {
                "step": step + 1,
                "lowpass_smooth_l1": float(loss.detach()),
                "gradient_preclip": float(norm),
            }
        )
    trained = responses()
    require(
        all(
            math.isfinite(value["max_abs"]) and value["max_abs"] > 0
            for value in trained.values()
        ),
        "predicted S has no measurable effect with F/Q held fixed",
    )
    return {
        "pool_counts": pool_counts,
        "initial_zero_response": initial,
        "after_correctness_steps_response": trained,
        "steps": steps,
        "fixed_inputs": [
            "F_full",
            "Q",
            "reference",
            "slot embeddings",
            "class logits",
        ],
        "perturbed_S": list(variants),
        "interpretation": (
            "condition-dependency correctness only; no training benefit claimed"
        ),
    }


def execute_phase(args):
    import torch

    from tools.structure_first import metadata, sha256, write_json

    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    configure_backend()
    device = torch.device(args.device)
    require(not args.amp or device.type == "cuda", "bf16 check requires CUDA")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    report = {
        "schema": "runc-independent-runtime-v1",
        "phase": args.phase,
        "status": "running",
        "backend": backend_record(),
        "amp": args.amp,
        "autocast_dtype": "bfloat16" if args.amp else None,
        "configuration_sha256": sha256(args.config),
        "metadata": metadata(config),
        "purpose": "correctness, not formal training or generation quality evidence",
    }
    started = time.monotonic()
    try:
        if args.phase == "legacy":
            checks = legacy_check(config, output, device, args)
        elif args.phase in ("coarse_grad", "refine_grad"):
            checks = gradient_check(
                config,
                output,
                device,
                args,
                "coarse" if args.phase == "coarse_grad" else "refine",
            )
        elif args.phase == "safety":
            checks = safety_check(config, output, device, args)
        else:
            checks = condition_check(config, output, device, args)
        report.update(status="pass", result=checks)
    except Exception as error:
        report.update(
            status="fail",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            report["gpu_peak_allocated_bytes"] = (
                torch.cuda.max_memory_allocated(device)
            )
            report["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(
                device
            )
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "runtime-check.json", report)
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "phase",
                        "amp",
                        "status",
                        "elapsed_seconds",
                    )
                }
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=(*PHASES, "all"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--legacy-dir", default="outputs/runc_20261017_legacy_before_pull"
    )
    args = parser.parse_args()
    if args.phase != "all":
        execute_phase(args)
        return
    require(not args.amp, "all automatically runs native FP32 and CUDA bf16")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    for phase in PHASES:
        for amp in (False,) if phase == "legacy" else (False, True):
            name = f"{phase}-{'bf16' if amp else 'fp32'}"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--phase",
                phase,
                "--config",
                args.config,
                "--output",
                str(root / name),
                "--device",
                args.device,
                "--legacy-dir",
                args.legacy_dir,
            ]
            if amp:
                command.append("--amp")
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
