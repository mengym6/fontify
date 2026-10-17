"""Derive and verify runc coarse candidates from measured parameter Grams."""

import argparse
import json
import math
from pathlib import Path
from statistics import median


def dump_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def summary(values):
    values = sorted(values)
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    return {
        "count": len(values),
        "min": values[0],
        "median": median(values),
        "max": values[-1],
        "mean": mean,
        "population_std": std,
        "coefficient_of_variation": std / mean if mean else None,
    }


def norm(row, coefficients):
    data = row["gradients"]["all_trainable"]
    indices = {name: i for i, name in enumerate(data["terms"])}
    square = sum(
        left * right * data["gram"][indices[a]][indices[b]]
        for a, left in coefficients.items()
        for b, right in coefficients.items()
    )
    if square < -1e-6:
        raise ValueError(f"negative Gram norm square {square}")
    return math.sqrt(max(square, 0.0))


def calculate_rows(probe, jcoef, lambda_j0, lambda_h):
    out = []
    for row in probe["measurements"]:
        norms = row["gradients"]["all_trainable"]["norms"]
        jnorm = norm(row, {f"j_{key}": value for key, value in jcoef.items()})
        low, head = norms["raw_low"], norms["raw_head"]
        measured_j = norms["raw_j0"]
        out.append(
            {
                "batch": row["batch"],
                "pool": row["pool"],
                "indices": row["indices"],
                "batch_size": row["batch_size"],
                "raw_low_norm": low,
                "j_composed_norm": jnorm,
                "measured_raw_j0_norm": measured_j,
                "raw_head_norm": head,
                "weighted_component_norms": {
                    key: value * norms[f"j_{key}"] for key, value in jcoef.items()
                },
                "weighted_j0_norm": lambda_j0 * jnorm,
                "weighted_head_norm": lambda_h * head,
                "weighted_j_over_low": lambda_j0 * jnorm / low if low else None,
                "weighted_head_over_j": lambda_h * head / (lambda_j0 * jnorm)
                if jnorm
                else None,
                "actual_weighted_j_over_low": lambda_j0 * measured_j / low
                if low
                else None,
                "actual_weighted_head_over_j": lambda_h
                * head
                / (lambda_j0 * measured_j)
                if measured_j
                else None,
                "j_cancellation_ratio": jnorm
                / sum(value * norms[f"j_{key}"] for key, value in jcoef.items())
                if jnorm
                else None,
                "cosine": row["gradients"]["all_trainable"]["cosine"],
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    probe = json.loads(Path(args.probe).read_text())
    config = json.loads(Path(args.source_config).read_text())
    rows = [r for r in probe["measurements"] if r["pool"] in ("dual", "jt_only")]
    if len(rows) < 4:
        raise ValueError("need multiple pure JT-available real batches")
    medians = {
        name: median(
            r["gradients"]["all_trainable"]["norms"][f"j_{name}"] for r in rows
        )
        for name in ("centroid", "logsigma", "shape")
    }
    if args.verify:
        jcoef = config["j_coefficients"]
        lambda_j0 = config["weights"]["lambda_j0"]
        lambda_h = config["weights"]["lambda_h"]
    else:
        if not all(math.isfinite(v) and v > 0 for v in medians.values()):
            raise ValueError("invalid raw J gradient median")
        jcoef = {
            "centroid": medians["shape"] / medians["centroid"],
            "logsigma": 2.0 * medians["shape"] / medians["logsigma"],
            "shape": 1.0,
        }
        composed = [norm(r, {f"j_{k}": v for k, v in jcoef.items()}) for r in rows]
        ratios = [
            n / r["gradients"]["all_trainable"]["norms"]["raw_low"]
            for r, n in zip(rows, composed)
        ]
        lambda_j0 = 0.5 / median(ratios)
        head_ratios = [
            r["gradients"]["all_trainable"]["norms"]["raw_head"] / (lambda_j0 * n)
            for r, n in zip(rows, composed)
        ]
        lambda_h = 0.5 / median(head_ratios)
    if not all(
        math.isfinite(v) and 0 < v <= 1e4
        for v in [*jcoef.values(), lambda_j0, lambda_h]
    ):
        raise ValueError("coefficient nonfinite/nonpositive/above 1e4")
    measured = calculate_rows(probe, jcoef, lambda_j0, lambda_h)
    fit_rows = [r for r in measured if r["pool"] in ("dual", "jt_only")]
    config["j_coefficients"] = jcoef
    config["weights"].update(lambda_low=1.0, lambda_j0=lambda_j0, lambda_h=lambda_h)
    config["weights_status"] = (
        "reprobed_initial_parameter_gradient_ratios_not_quality_optimal"
        if args.verify
        else "initial_parameter_gradient_ratio_candidates_pending_reprobe"
    )
    report = {
        "schema": config["schema"],
        "experiment": "runc",
        "status": "reprobe_verified" if args.verify else "derived_pending_reprobe",
        "source_probe": str(Path(args.probe)),
        "source_config": str(Path(args.source_config)),
        "policy": {
            "fit_pools": ["dual", "jt_only"],
            "j_component_median_norm_ratio_centroid_logsigma_shape": [1, 2, 1],
            "weighted_j_over_low_median_per_batch_ratio": 0.5,
            "weighted_head_over_j_median_per_batch_ratio": 0.5,
            "lambda_low": 1.0,
            "limitation": (
                "initial calibration policy, not optimal weights "
                "or evidence of training quality"
            ),
            "bf_only": "J and H absent; report zero gradients separately",
        },
        "raw_j_component_parameter_norm_medians": medians,
        "j_coefficients": jcoef,
        "weights": {
            k: config["weights"][k] for k in ("lambda_low", "lambda_j0", "lambda_h")
        },
        "all_batch_measurements": measured,
        "fit_pool_ratio_summary": {
            key: summary([r[key] for r in fit_rows])
            for key in (
                "weighted_j_over_low",
                "weighted_head_over_j",
                "j_cancellation_ratio",
            )
        },
        "weighted_j_component_parameter_norm_summary": {
            key: summary([r["weighted_component_norms"][key] for r in fit_rows])
            for key in jcoef
        },
        "source_precision": probe["precision"],
        "source_metadata": probe["metadata"],
    }
    if args.verify:
        errors = [
            abs(r["j_composed_norm"] - r["measured_raw_j0_norm"])
            / max(r["j_composed_norm"], 1e-12)
            for r in fit_rows
        ]
        report["measured_composite_vs_gram_max_relative_error"] = max(errors)
        report["measured_composite_vs_gram_relative_error_per_fit_batch"] = errors
        report["measured_composite_vs_gram_within_5pct_reference"] = max(errors) <= 0.05
        report["actual_autograd_fit_ratio_summary"] = {
            key: summary([r[key] for r in fit_rows])
            for key in ("actual_weighted_j_over_low", "actual_weighted_head_over_j")
        }
        report["status"] = "reprobed_requires_review_of_actual_ratios"
    authors = sorted(
        {
            Path(path).parts[-3][:-2]
            for row in probe["measurements"]
            for path in row.get("target_path", [])
        }
    )
    report["observed_target_authors"] = authors
    if len(authors) < 9:
        raise ValueError("Formal runc calibration must cover all nine sources")
    dump_new(args.report, report)
    dump_new(args.config, config)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "j_coefficients",
                    "weights",
                    "fit_pool_ratio_summary",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
