"""Calibrate shared instance/lower/full evaluation J on ctrl_g12 best ep46.

Uses the old parameter-gradient balancing rule with the new targets and fully
hidden query: coefficient_i = mean(norm(shape)) / mean(norm(term_i)).
This is an evaluation-only calibration, distinct from run B training calibration.
"""

import argparse
import json
import math
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
    load_model,
    make_jieti_loss,
    require_source_model,
)


def parameter_gradient_gram(losses, parameters, chunk_size=1 << 20):
    """Measure true parameter-gradient inner products, including cancellation."""
    import torch

    if not parameters:
        raise ValueError("No trainable parameters for evaluation calibration")
    gradients = [
        torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
        for value in losses
    ]
    gram = torch.zeros(
        len(losses), len(losses), dtype=torch.float64, device=parameters[0].device
    )
    for index, parameter in enumerate(parameters):
        columns = [row[index] for row in gradients]
        if all(column is None for column in columns):
            continue
        for offset in range(0, parameter.numel(), chunk_size):
            vectors = torch.stack(
                [
                    torch.zeros(
                        min(chunk_size, parameter.numel() - offset),
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                    if value is None
                    else value.flatten()[offset : offset + chunk_size]
                    for value in columns
                ]
            ).double()
            gram += vectors @ vectors.T
    if not torch.isfinite(gram).all():
        raise ValueError("Nonfinite evaluation J gradient Gram matrix")
    return gram.detach().cpu().tolist()


def derive_eval_coefficients(grams):
    """Ratio of mean batch norms, rather than a ratio of squared norms."""
    if not grams:
        raise ValueError("No JT calibration batches")
    norms = {
        term: [math.sqrt(max(0, gram[index][index])) for gram in grams]
        for index, term in enumerate(TERMS)
    }
    means = {term: sum(values) / len(values) for term, values in norms.items()}
    if any(not math.isfinite(value) or value <= 0 for value in means.values()):
        raise ValueError("Zero/nonfinite new J gradient; diagnose before calibration")
    return {term: means["shape"] / means[term] for term in TERMS}, norms


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--val_json_path", nargs="+", required=True)
    parser.add_argument("--fixed_pair_path", required=True)
    parser.add_argument("--dual_target_cache", required=True)
    parser.add_argument("--output_json", default="models/dual_view/eval_coef.json")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser


def main():
    args = get_parser().parse_args()
    import numpy as np
    import torch

    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if Path(args.output_json).exists():
        raise FileExistsError(args.output_json)
    args.j_definition = DUAL_SCHEMA
    metadata = eval_metadata(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = load_model(args.checkpoint).to(args.device).eval()
    require_source_model(model, "ctrl_g12_best")
    dataset = build_dataset(args)
    indices = [index for index in range(len(dataset)) if dataset[index]["has_jt"]]
    parameters = [value for value in model.parameters() if value.requires_grad]
    loss = make_jieti_loss(DUAL_SCHEMA, {term: 1.0 for term in TERMS}).to(args.device)
    records = []
    for cursor in range(0, len(indices), args.batch_size):
        selected = indices[cursor : cursor + args.batch_size]
        samples = [dataset[index] for index in selected]
        batch = torch.utils.data.default_collate(samples)
        images, targets = (
            batch["image"].to(args.device),
            batch["target"].to(args.device),
        )
        prediction = model.predict_full(images, targets)
        _, parts, _ = loss(
            prediction,
            targets,
            batch["voro"].to(args.device),
            batch["valid_parts"].to(args.device),
            batch["has_jt"].to(args.device),
        )
        gram = parameter_gradient_gram([parts[term] for term in TERMS], parameters)
        records.append(
            {
                "indices": selected,
                "gram": gram,
                "target_paths": [sample["query_target_path"] for sample in samples],
            }
        )
        print(
            f"[eval calibration] {cursor + len(selected)}/{len(indices)} JT items",
            flush=True,
        )
        del prediction, parts, batch
    coefficients, norms = derive_eval_coefficients([row["gram"] for row in records])
    result = {
        "schema": DUAL_SCHEMA,
        "purpose": "evaluation",
        "status": "validated",
        "source_role": "ctrl_g12_best",
        "source_epoch": 46,
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": model._checkpoint_sha256,
        "coef": coefficients,
        "calibration_space": "model_parameters",
        "norm_rule": "ratio_of_mean_batch_norms_shape_reference",
        "model_mode": "eval",
        "autocast": False,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "n_jt": len(indices),
        "n_batches": len(records),
        "gradient_norms": norms,
        "gradient_records": records,
        **metadata,
    }
    write_new_json(args.output_json, result)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("gradient_norms", "gradient_records")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
