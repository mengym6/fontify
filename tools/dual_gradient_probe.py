"""Offline train/bf16 per-loss probes on warm-end, ep2, ep25 and ep50.

No optimizer step or formal-training RNG change is performed.  Every matrix is
computed from one shared forward graph using separate ``autograd.grad`` calls.
Raw J components and the actual composed J are both saved.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dual_view_protocol import (  # noqa: E402
    MODULES,
    SCHEMA,
    TERMS,
    apply_weights,
    build_dataset,
    check_samples,
    fixed_batch,
    forward_terms,
    load_dual_model,
    seed_all,
    sha256_file,
    validate_calibration,
    weight_summary,
    write_new_json,
)


def parameter_groups(model):
    """Exhaustively partition the trainable generator; ignore frozen VGG/D."""
    result = {key: [] for key in MODULES}
    names = {key: [] for key in MODULES}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("dual_head."):
            group = "head"
        elif name.startswith("dual_fusion."):
            group = "fusion"
        elif name.startswith(("decoder_embed.", "decoder_pred.")):
            group = "decoder"
        elif name.startswith(("patch_embed.", "blocks.", "norm.")) or name in (
            "mask_token",
            "segment_token_x",
            "segment_token_y",
            "pos_embed",
        ):
            group = "vit"
        else:
            raise ValueError(f"unclassified trainable parameter: {name}")
        result[group].append(parameter)
        names[group].append(name)
    if any(not parameters for parameters in result.values()):
        raise ValueError("probe needs trainable ViT, decoder, fusion and head")
    return result, names


def _gram_from_grads(torch, gradients, parameters, chunk=1 << 18):
    matrix = torch.zeros(len(gradients), len(gradients), dtype=torch.float64)
    for position, parameter in enumerate(parameters):
        pieces = [row[position] for row in gradients]
        if all(piece is None for piece in pieces):
            continue
        flats = [
            piece.detach().reshape(-1).float().cpu()
            if piece is not None
            else None
            for piece in pieces
        ]
        for start in range(0, parameter.numel(), chunk):
            length = min(chunk, parameter.numel() - start)
            block = torch.stack(
                [
                    piece[start : start + length].double()
                    if piece is not None
                    else torch.zeros(length, dtype=torch.float64)
                    for piece in flats
                ]
            )
            matrix += block @ block.T
    if not torch.isfinite(matrix).all():
        raise ValueError("nonfinite per-loss gradient matrix")
    return matrix.tolist()


def measure_graph(model, extra):
    """Save six raw terms plus the actual J, not a sum of their norms."""
    import torch

    groups, _ = parameter_groups(model)
    parameters = [
        parameter for module in MODULES for parameter in groups[module]
    ]
    losses = [extra[key] for key in (*TERMS, "J")]
    gradients = [
        torch.autograd.grad(
            loss,
            parameters,
            retain_graph=index < len(losses) - 1,
            allow_unused=True,
        )
        for index, loss in enumerate(losses)
    ]
    matrices = {}
    actual = {}
    start = 0
    for module in MODULES:
        selected = groups[module]
        end = start + len(selected)
        matrix = _gram_from_grads(
            torch, [row[start:end] for row in gradients], selected
        )
        matrices[module] = [row[:6] for row in matrix[:6]]
        # Direct gradients of all four objective terms from the same graph.
        positions = (0, 1, 6, 5)
        actual[module] = [[matrix[i][j] for j in positions] for i in positions]
        start = end
    return matrices, actual


def slot_stats(extra):
    """Aggregate raw predictions without interpreting them as efficacy."""
    import torch

    occupancy = extra["head_occupancy_logits"].detach().sigmoid().float()
    exists = extra["head_existence_logits"].detach().sigmoid().float()
    result = {
        "exist_probability_mean": exists.mean(0).cpu().tolist(),
        "occupancy_probability_mean": occupancy.mean((0, 2, 3)).cpu().tolist(),
        "active_slots_mean": float((exists >= 0.5).sum(1).float().mean()),
    }
    binary = occupancy >= 0.5
    active = exists >= 0.5
    ink = (binary & active[:, :, None, None]).any(1)
    result["empty_glyph_fraction"] = float(
        (~ink.flatten(1).any(1)).float().mean()
    )
    result["full_glyph_fraction"] = float(ink.flatten(1).all(1).float().mean())
    intersection = (binary[:, :, None] & binary[:, None]).sum((-1, -2))
    union = (binary[:, :, None] | binary[:, None]).sum((-1, -2))
    pair_active = active[:, :, None] & active[:, None]
    pairs = [
        (i, j)
        for i in range(binary.shape[1])
        for j in range(i + 1, binary.shape[1])
    ]
    valid_iou = [
        intersection[:, i, j][pair_active[:, i, j]].float()
        / union[:, i, j][pair_active[:, i, j]].clamp_min(1)
        for i, j in pairs
    ]
    values = [value for value in valid_iou if value.numel()]
    result["active_slot_pair_iou_mean"] = (
        float(torch.cat(values).mean()) if values else None
    )
    for key in (
        "assignments",
        "head_assignment",
        "head_matches",
        "head_match",
    ):
        if key in extra:
            value = extra[key]
            result["matching"] = (
                value.detach().cpu().tolist()
                if hasattr(value, "detach")
                else value
            )
            break
    return result


def measure_batches(
    model, dataset, samples, device, epoch=0, groups=("jt", "bf_only", "mixed")
):
    model.train()
    records = []
    size = samples["batch_size"]
    if size <= 0:
        raise ValueError("fixed probe batch size must be positive")
    for group in groups:
        rows = samples["groups"][group]
        for start in range(0, len(rows), size):
            current = rows[start : start + size]
            seed_all(current[0]["forward_seed"])
            batch = fixed_batch(dataset, current, device)
            extra = forward_terms(model, batch, epoch=epoch)
            grams, actual = measure_graph(model, extra)
            record = {
                "group": group,
                "batch_start": start,
                "target_indices": [row["target_index"] for row in current],
                "forward_seed": current[0]["forward_seed"],
                "n": len(current),
                "n_jt": int(batch["has_jt"].sum()),
                "losses": {
                    key: float(extra[key].detach()) for key in (*TERMS, "J")
                },
                "grams": grams,
                "objective_grams": actual,
                "slots": slot_stats(extra),
            }
            records.append(record)
            del extra, batch
    return records


def check_fusion_open(model):
    """Check that ep2 has moved away from the zero fusion projection."""
    for module in reversed(list(model.dual_fusion.modules())):
        if hasattr(module, "weight") and hasattr(module, "bias"):
            return bool(
                module.weight.detach().abs().sum() > 0
                or (
                    module.bias is not None
                    and module.bias.detach().abs().sum() > 0
                )
            )
    raise ValueError("fusion final projection not found")


def compare_partitions(
    prediction,
    target,
    legacy_voro,
    legacy_valid,
    is_jt,
    new_voro,
    new_valid,
    coefficients,
):
    """Compare J value AND pixel gradients on the exact same fixed prediction.

    Historical J averages two halves/valid parts; new J averages per lower
    glyph over the full batch.  This command intentionally preserves both
    definitions, labels that distinction, and never concludes they are equal.
    """
    import torch

    from util.dual_loss import DualJietiLoss
    from util.jieti_loss import JietiLoss

    values, gradients = {}, {}
    for name, module, voro, valid in (
        ("legacy", JietiLoss(**coefficients), legacy_voro, legacy_valid),
        ("instance_lower", DualJietiLoss(**coefficients), new_voro, new_valid),
    ):
        pred = prediction.detach().clone().requires_grad_(True)
        value, _, _ = module.to(pred.device)(pred, target, voro, valid, is_jt)
        (grad,) = torch.autograd.grad(value, pred)
        values[name] = float(value)
        gradients[name] = grad
    old, new = gradients["legacy"], gradients["instance_lower"]
    denominator = old.norm() * new.norm()
    return {
        "values": values,
        "value_difference": values["instance_lower"] - values["legacy"],
        "gradient_norms": {
            name: float(grad.norm()) for name, grad in gradients.items()
        },
        "gradient_difference_norm": float((new - old).norm()),
        "gradient_cosine": float((old * new).sum() / denominator)
        if denominator > 0
        else None,
        "definitions_differ": (
            "legacy both halves/part aggregation vs instance lower/per-glyph"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--point", choices=("warm_end", "ep2", "ep25", "ep50"), required=True
    )
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--json-path", nargs="+", required=True)
    parser.add_argument("--target-cache", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    weights = validate_calibration(args.calibration)
    samples = check_samples(args.samples, args)
    if sha256_file(args.samples) != weights["sample_manifest_sha256"]:
        raise ValueError(
            "probe must use the originally calibrated fixed samples"
        )
    model, checkpoint = load_dual_model(
        args.checkpoint, args.device, args.seed
    )
    if (
        args.point == "warm_end"
        and sha256_file(args.checkpoint) != weights["warm_checkpoint_sha256"]
    ):
        raise ValueError("warm-end probe checkpoint mismatch")
    if args.point != "warm_end":
        expected_epoch = int(args.point[2:])
        if checkpoint.get("epoch") != expected_epoch:
            raise ValueError("probe checkpoint epoch mismatch")
    fusion_open = check_fusion_open(model)
    if args.point == "ep2" and not fusion_open:
        write_new_json(
            args.output,
            {
                "schema": SCHEMA,
                "purpose": "gradient_probe",
                "point": args.point,
                "status": "graph_blocked",
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "calibration_sha256": sha256_file(args.calibration),
                "sample_manifest_sha256": sha256_file(args.samples),
                "reason": "fusion final projection still exactly zero",
            },
        )
        raise RuntimeError(
            "ep2 fusion projection is zero: diagnose the graph "
            "before interpreting probes"
        )
    apply_weights(model, weights)
    epoch = 0 if args.point == "warm_end" else int(args.point[2:])
    records = measure_batches(
        model, build_dataset(args), samples, args.device, epoch
    )
    write_new_json(
        args.output,
        {
            "schema": SCHEMA,
            "purpose": "gradient_probe",
            "status": "measured",
            "point": args.point,
            "epoch": epoch,
            "fusion_open": fusion_open,
            "terms": list(TERMS),
            "objective_terms": ["L_pen", "L_full", "J", "L_H"],
            "train_mode": True,
            "autocast": "bfloat16",
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "calibration_sha256": sha256_file(args.calibration),
            "sample_manifest_sha256": sha256_file(args.samples),
            "summary": weight_summary(records, weights),
            "measurements": records,
        },
    )
    print(
        json.dumps(
            {
                "point": args.point,
                "batches": len(records),
                "fusion_open": fusion_open,
            }
        )
    )


if __name__ == "__main__":
    main()
