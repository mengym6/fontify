"""Paired old/new J values and prediction gradients on identical fixed images.

Each prediction is generated once, detached, hashed and reused for both metrics.
Raw term comparisons distinguish definition changes from coefficient changes.
Legacy and instance partitions receive identical, deterministic image geometry.
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dual_view_protocol import write_new_json  # noqa: E402
from tools.eval_s_baseline import (  # noqa: E402
    DUAL_SCHEMA,
    TERMS,
    build_dataset,
    eval_metadata,
    file_sha256,
    load_model,
    make_jieti_loss,
    validate_eval_coefficients,
)


def paired_gradient_metrics(left, right):
    """Compare gradient vectors directly, retaining zeros as zero/null."""
    import torch

    left, right = left.double(), right.double()
    left_norm, right_norm = float(left.norm()), float(right.norm())
    return {
        "old_norm": left_norm,
        "new_norm": right_norm,
        "cosine": float((left * right).sum()) / (left_norm * right_norm)
        if left_norm > 0 and right_norm > 0
        else None,
        "new_over_old": right_norm / left_norm if left_norm > 0 else None,
        "difference_norm": float((right - left).norm()),
        "old_upper_max": float(left[:, :, : left.shape[2] // 2].abs().max()),
        "new_upper_max": float(right[:, :, : right.shape[2] // 2].abs().max()),
        "finite": bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
    }


def compare_prediction(
    prediction,
    target,
    old_partition,
    old_valid,
    new_partition,
    new_valid,
    old_loss,
    new_loss,
):
    """Compute both metrics and per-term gradients from one prediction tensor."""
    import torch

    prediction = prediction.detach().float().requires_grad_(True)
    active = torch.ones(prediction.shape[0], dtype=torch.bool, device=prediction.device)
    half = target.shape[2] // 2
    composite = torch.cat((target[:, :, :half], prediction[:, :, half:]), dim=2)
    old_value, old_parts, old_shield = old_loss(
        composite, target, old_partition, old_valid, active
    )
    new_value, new_parts, new_shield = new_loss(
        prediction, target, new_partition, new_valid, active
    )
    output = {
        "old_J": float(old_value.detach()),
        "new_J": float(new_value.detach()),
        "old_shield": float(old_shield),
        "new_shield": float(new_shield),
        "terms": {},
    }
    for name in ("J", *TERMS):
        left = old_value if name == "J" else old_parts[name]
        right = new_value if name == "J" else new_parts[name]
        left_gradient = torch.autograd.grad(left, prediction, retain_graph=True)[0]
        right_gradient = torch.autograd.grad(right, prediction, retain_graph=True)[0]
        output["terms"][name] = {
            "old": float(left.detach()),
            "new": float(right.detach()),
            **paired_gradient_metrics(left_gradient, right_gradient),
        }
    return output


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--val_json_path", nargs="+", required=True)
    parser.add_argument("--semantic_mask_dir")
    parser.add_argument("--fixed_pair_path", required=True)
    parser.add_argument("--dual_target_cache", required=True)
    parser.add_argument("--eval_coef_json", required=True)
    parser.add_argument("--legacy_calibration_json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", required=True)
    return parser


def main():
    args = get_parser().parse_args()
    import numpy as np
    import torch

    if Path(args.output_json).exists():
        raise FileExistsError(args.output_json)
    args.j_definition = DUAL_SCHEMA
    metadata = eval_metadata(args)
    new_payload = json.loads(Path(args.eval_coef_json).read_text())
    coefficients = validate_eval_coefficients(new_payload, metadata)
    old_coefficients = json.loads(Path(args.legacy_calibration_json).read_text())[
        "coef"
    ]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = load_model(args.checkpoint).to(args.device).eval()
    new_dataset = build_dataset(args)
    old_dataset = build_dataset(args, "legacy", identity=True)
    old_by_reference = {
        row["target_path"]: index for index, row in enumerate(old_dataset.pairs)
    }
    old_loss = make_jieti_loss("legacy", old_coefficients).to(args.device)
    new_loss = make_jieti_loss(DUAL_SCHEMA, coefficients).to(args.device)
    records = []
    for index in range(len(new_dataset)):
        sample = new_dataset[index]
        if not bool(sample["has_jt"]):
            continue
        reference = new_dataset.pairs[index]["target_path"]
        old = old_dataset[old_by_reference[reference]]
        if not torch.allclose(
            old[0], sample["image"], atol=1e-6, rtol=0
        ) or not torch.allclose(old[1], sample["target"], atol=1e-6, rtol=0):
            raise ValueError(f"Paired old/new images differ for fixed sample {index}")
        targets = sample["target"][None].to(args.device)
        with torch.no_grad():
            prediction = model.predict_full(
                sample["image"][None].to(args.device), targets
            )
        # This hash identifies the exact normalized floating-point prediction,
        # independent of display clipping or subsequent metric computation.
        prediction_hash = hashlib.sha256(
            prediction.detach().float().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        comparison = compare_prediction(
            prediction,
            targets,
            old[4][None].to(args.device),
            old[5][None].to(args.device),
            sample["voro"][None].to(args.device),
            sample["valid_parts"][None].to(args.device),
            old_loss,
            new_loss,
        )
        records.append(
            {
                "index": index,
                "target_path": sample["query_target_path"],
                "reference_path": reference,
                "prediction_sha256": prediction_hash,
                **comparison,
            }
        )
        print(f"[paired J] {len(records)} JT predictions compared", flush=True)
    if not records:
        raise ValueError("No paired JT predictions")
    differences = [row["new_J"] - row["old_J"] for row in records]
    result = {
        "schema": DUAL_SCHEMA,
        "purpose": "paired_old_new_j",
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": model._checkpoint_sha256,
        "eval_coef_sha256": file_sha256(args.eval_coef_json),
        "legacy_coef_sha256": file_sha256(args.legacy_calibration_json),
        "old_coef": old_coefficients,
        "new_coef": coefficients,
        "prediction_reuse": "identical_fp32_tensor",
        "geometry": "identity_448",
        "n_jt": len(records),
        "seed": args.seed,
        "per_item": records,
        "new_minus_old_J": {
            "mean": float(np.mean(differences)),
            "median": float(np.median(differences)),
            "min": min(differences),
            "max": max(differences),
        },
        **metadata,
    }
    write_new_json(args.output_json, result)


if __name__ == "__main__":
    main()
