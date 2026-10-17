"""Calibrate initial runc refinement weights from real parameter gradients.

Derivation consumes a refine ``structure_first.py probe`` report. ``--verify``
replays its exact indices/seeds with the candidate configuration and measures
the combined losses by autograd, including AMP backward rounding. Neither mode
trains a model, accepts coarse quality, or authorizes stage-II training.
"""

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

J_NAMES = ("centroid", "logsigma", "shape")
RAW_NAMES = (
    "raw_img_smooth", "raw_img_gram", "raw_img_edge", "raw_bf_smooth",
    "raw_bf_edge", "raw_j1", "j_centroid", "j_logsigma", "j_shape",
)
COMBINED_NAMES = ("combined_img1", "combined_bf_local", "combined_j1")


def dump_new(path, value):
    """Create one record without replacing an earlier scientific artifact."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def summary(values):
    """Describe variation across batches without treating it as replication."""
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("summary requires nonempty finite measurements")
    mean = sum(values) / len(values)
    std = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
    return {
        "count": len(values), "min": min(values), "median": median(values),
        "max": max(values), "mean": mean, "population_std": std,
        "coefficient_of_variation": std / mean if mean else None,
    }


def gram_norm(row, coefficients):
    """Compose a norm using cross terms, retaining gradient cancellation."""
    data = row["gradients"]["refiner"]
    indices = {name: index for index, name in enumerate(data["terms"])}
    contributions = [
        left * right * data["gram"][indices[a]][indices[b]]
        for a, left in coefficients.items()
        for b, right in coefficients.items()
    ]
    square = sum(contributions)
    tolerance = 1e-7 * max(1.0, sum(abs(value) for value in contributions))
    if not math.isfinite(square) or square < -tolerance:
        raise ValueError("nonfinite or negative Gram-composed norm square")
    return math.sqrt(max(square, 0.0))


def positive_ratio_scale(rows, numerator, denominator, target):
    ratios = []
    for row in rows:
        top, bottom = numerator(row), denominator(row)
        if not math.isfinite(top) or not math.isfinite(bottom) or bottom <= 0:
            raise ValueError(f"invalid ratio denominator in batch {row['batch']}")
        ratios.append(top / bottom)
    middle = median(ratios)
    if not math.isfinite(middle) or middle <= 0:
        raise ValueError("zero/nonfinite median gradient ratio prevents calibration")
    return target / middle


def observed_authors(rows):
    authors = set()
    for row in rows:
        for path in row.get("target_path", []):
            parts = Path(path).parts
            if len(parts) < 3 or not parts[-3].endswith(("BF", "JT")):
                raise ValueError("cannot resolve source author from target_path")
            authors.add(parts[-3][:-2])
    return sorted(authors)


def validate_probe(probe, config):
    """Reject missing pools, JT-only refinement, and leaked trainable groups."""
    if config.get("stage") != "refine" or config.get("experiment") != "runc":
        raise ValueError("calibration requires an explicit runc refine config")
    if config.get("lowpass") != {
        "kernel_size": 7, "sigma": 1.5, "boundary": "reflect",
    }:
        raise ValueError("refine calibration preserves Gaussian 7/1.5/reflect")
    rows = probe["measurements"]
    pure = [row for row in rows if row["pool"] in ("dual", "bf_only")]
    dual = [row for row in pure if row["pool"] == "dual"]
    bf_only = [row for row in pure if row["pool"] == "bf_only"]
    if min(len(dual), len(bf_only)) < 4:
        raise ValueError("need at least four pure dual and four BF-only batches")
    if len(observed_authors(pure)) != 9:
        raise ValueError("formal calibration requires all nine target sources")
    for row in rows:
        if row["pool"] not in ("dual", "bf_only", "mixed"):
            raise ValueError("JT-only or unknown pool cannot enter refinement")
        bf = row.get("sample_has_bf", [])
        jt = row.get("sample_has_jt", [])
        if (len(bf) != row["batch_size"] or len(jt) != row["batch_size"]
                or not bf or not all(bf)):
            raise ValueError("all refinement samples require real BF annotation")
        if (row["pool"] == "dual" and not all(jt)) or (
            row["pool"] == "bf_only" and any(jt)
        ):
            raise ValueError("pool name disagrees with JT annotation gates")
        groups = row["gradients"]
        refiner = groups["refiner"]
        if not set(RAW_NAMES).issubset(refiner["terms"]):
            raise ValueError("probe omits required refinement raw gradient terms")
        for name, value in refiner["norms"].items():
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid measured Refiner gradient norm")
            other = groups["all_trainable"]["norms"].get(name)
            if other is None or not math.isclose(value, other, abs_tol=1e-10,
                                               rel_tol=1e-6):
                raise ValueError("all-trainable gradients differ from Refiner")
        if "head" in groups and any(groups["head"]["norms"].values()):
            raise ValueError("frozen JT Head has an effective parameter gradient")
        if row["pool"] == "bf_only" and any(
            refiner["norms"][name] != 0 for name in
            ("raw_j1", "j_centroid", "j_logsigma", "j_shape")
        ):
            raise ValueError("BF-only J is invalid and must have zero gradient")
        if row["pool"] == "bf_only" and any(
            row.get("losses", {}).get(name, 0.0) != 0
            for name in ("raw_j1", "raw_head")
        ):
            raise ValueError("BF-only J and H must be absent, not invented labels")
    return pure, dual


def components(config):
    weights = config["weights"]
    return {
        "img1": {"raw_img_smooth": 1.0, "raw_img_gram": weights["gram"],
                 "raw_img_edge": weights["edge"]},
        "bf_local": {"raw_bf_smooth": 1.0,
                     "raw_bf_edge": weights["edge_local"]},
        "j1": {f"j_{name}": value
               for name, value in config["j_coefficients"].items()},
    }


def derive_config(probe, source_config):
    pure, dual = validate_probe(probe, source_config)
    config = deepcopy(source_config)
    def norms(row):
        return row["gradients"]["refiner"]["norms"]

    medians = {
        name: median(norms(row)[f"j_{name}"] for row in dual) for name in J_NAMES
    }
    if any(value <= 0 for value in medians.values()):
        raise ValueError("dual J component median has zero parameter gradient")
    config["j_coefficients"] = {
        "centroid": medians["shape"] / medians["centroid"],
        "logsigma": 2.0 * medians["shape"] / medians["logsigma"],
        "shape": 1.0,
    }
    weights = config["weights"]
    for key, top, bottom, target in (
        ("gram", "raw_img_gram", "raw_img_smooth", 0.5),
        ("edge", "raw_img_edge", "raw_img_smooth", 0.1),
        ("edge_local", "raw_bf_edge", "raw_bf_smooth", 0.1),
    ):
        weights[key] = positive_ratio_scale(
            pure, lambda row: norms(row)[top], lambda row: norms(row)[bottom],
            target,
        )
    groups = components(config)
    weights["lambda_img1"] = 1.0
    weights["lambda_bf"] = positive_ratio_scale(
        pure, lambda row: gram_norm(row, groups["bf_local"]),
        lambda row: gram_norm(row, groups["img1"]), 0.5,
    )
    weights["lambda_j1"] = positive_ratio_scale(
        dual, lambda row: gram_norm(row, groups["j1"]),
        lambda row: gram_norm(row, groups["img1"]), 0.25,
    )
    candidates = [*config["j_coefficients"].values(), *(
        weights[key] for key in ("gram", "edge", "edge_local", "lambda_img1",
                                 "lambda_bf", "lambda_j1")
    )]
    if any(not math.isfinite(value) or not 0 < value <= 1e4 for value in candidates):
        raise ValueError("candidate coefficient nonfinite/nonpositive/above 1e4")
    config["weights_status"] = (
        "refine_initial_parameter_gradient_candidates_pending_actual_reprobe"
    )
    return config


def batch_measurement(row, config):
    norms = row["gradients"]["refiner"]["norms"]
    groups = components(config)
    composed = {name: gram_norm(row, value) for name, value in groups.items()}
    weights = config["weights"]
    ratios = {
        "gram_over_smooth": weights["gram"] * norms["raw_img_gram"]
        / norms["raw_img_smooth"],
        "edge_over_smooth": weights["edge"] * norms["raw_img_edge"]
        / norms["raw_img_smooth"],
        "local_edge_over_smooth": weights["edge_local"] * norms["raw_bf_edge"]
        / norms["raw_bf_smooth"],
        "bf_over_img": weights["lambda_bf"] * composed["bf_local"]
        / composed["img1"],
        "j_over_img": weights["lambda_j1"] * composed["j1"] / composed["img1"],
    }
    result = {
        "batch": row["batch"], "pool": row["pool"], "indices": row["indices"],
        "composed_norms": composed, "gram_composed_ratios": ratios,
        "weighted_j_component_norms": {
            name: config["j_coefficients"][name] * norms[f"j_{name}"]
            for name in J_NAMES
        },
        "cancellation_ratio": {
            name: composed[name] / sum(value * norms[term]
                                       for term, value in coefficients.items())
            if composed[name] else None
            for name, coefficients in groups.items()
        },
        "cosine": row["gradients"]["refiner"]["cosine"],
    }
    if set(COMBINED_NAMES).issubset(norms):
        actual = {name: norms[f"combined_{name}"] for name in groups}
        result["actual_combined_norms"] = actual
        result["composite_vs_gram_relative_error"] = {
            name: abs(actual[name] - composed[name]) / max(composed[name], 1e-12)
            for name in groups
        }
        weighted_names = {f"weighted_{name}" for name in groups}
        if weighted_names.issubset(norms):
            result["actual_weighted_norms"] = {
                name: norms[f"weighted_{name}"] for name in groups
            }
            result["actual_autograd_ratios"] = {
                "bf_over_img": norms["weighted_bf_local"] / norms["weighted_img1"],
                "j_over_img": norms["weighted_j1"] / norms["weighted_img1"],
            }
            total_coefficients = {}
            for name, scale in (("img1", weights["lambda_img1"]),
                                ("bf_local", weights["lambda_bf"]),
                                ("j1", weights["lambda_j1"])):
                total_coefficients.update({key: value * scale
                                           for key, value in groups[name].items()})
            total_norm = gram_norm(row, total_coefficients)
            result["weighted_total_composed_norm"] = total_norm
            result["actual_weighted_total_norm"] = norms["weighted_total"]
            result["weighted_total_vs_gram_relative_error"] = abs(
                norms["weighted_total"] - total_norm
            ) / max(total_norm, 1e-12)
    return result


def make_report(probe, config, verified=False):
    pure, dual = validate_probe(probe, config)
    measured = [batch_measurement(row, config) for row in probe["measurements"]]
    fitted = [row for row in measured if row["pool"] in ("dual", "bf_only")]
    fitted_dual = [row for row in fitted if row["pool"] == "dual"]
    report = {
        "schema": config["schema"], "experiment": "runc", "stage": "refine",
        "status": ("reprobed_requires_review_of_actual_ratios" if verified
                   else "derived_pending_actual_autograd_reprobe"),
        "policy": {
            "image_and_local_fit_pools": ["dual", "bf_only"],
            "j_fit_pool": "dual", "mixed_pool": "diagnostic_only",
            "gram_over_full_smooth": 0.5, "full_edge_over_full_smooth": 0.1,
            "local_edge_over_local_smooth": 0.1, "lambda_img1": 1.0,
            "bf_over_img_median_per_batch_ratio": 0.5,
            "j_over_img_dual_median_per_batch_ratio": 0.25,
            "j_component_median_norm_ratio_centroid_logsigma_shape": [1, 2, 1],
            "limitation": (
                "Initial gradient-norm policy, not optimal weights, independent "
                "replications, optimization contribution shares, or quality proof."
            ),
        },
        "fit_batch_counts": {"all_bf": len(pure), "dual": len(dual)},
        "observed_target_authors": observed_authors(pure),
        "j_coefficients": config["j_coefficients"],
        "weights": {key: config["weights"][key] for key in (
            "lambda_img1", "lambda_bf", "lambda_j1", "gram", "edge", "edge_local"
        )},
        "all_batch_measurements": measured,
        "gram_composed_fit_ratio_summary": {
            key: summary([row["gram_composed_ratios"][key] for row in (
                fitted_dual if key == "j_over_img" else fitted
            )]) for key in fitted[0]["gram_composed_ratios"]
        },
        "weighted_j_component_parameter_norm_summary": {
            key: summary([row["weighted_j_component_norms"][key]
                          for row in fitted_dual]) for key in J_NAMES
        },
        "source_precision": probe["precision"],
        "source_metadata": probe["metadata"],
        "raw_gradient_measurements": probe["measurements"],
    }
    if verified:
        if any("actual_autograd_ratios" not in row for row in measured):
            raise ValueError(
                "verification needs real combined-loss autograd measurements"
            )
        report["actual_autograd_fit_ratio_summary"] = {
            key: summary([row["actual_autograd_ratios"][key] for row in (
                fitted_dual if key == "j_over_img" else fitted
            )]) for key in ("bf_over_img", "j_over_img")
        }
        report["composite_vs_gram_relative_error_summary"] = {
            key: summary([row["composite_vs_gram_relative_error"][key]
                          for row in (fitted_dual if key == "j1" else fitted)])
            for key in ("img1", "bf_local", "j1")
        }
        report["weighted_total_vs_gram_relative_error_summary"] = summary([
            row["weighted_total_vs_gram_relative_error"] for row in fitted
        ])
    return report


def remeasure(probe, config, device_name):
    """Replay real inputs and autograd without an optimizer or parameter updates."""
    import torch

    from tools import structure_first as runtime

    runtime.validate_config(config, "probe")
    device = torch.device(device_name or ("cuda:0" if torch.cuda.is_available()
                                           else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, _ = runtime.load_model(config, device)
    runtime.training_mode(model, "refine")
    parameters = [(name, value) for name, value in model.named_parameters()
                  if value.requires_grad]
    if not parameters or any(not name.startswith("bf_refiner.")
                             for name, _ in parameters):
        raise ValueError("phase II must train only the independent BF Refiner")
    dataset = runtime.build_dataset(config)
    current_metadata = runtime.metadata(config)
    for key in ("parent_checkpoint_sha256", "train_manifest_sha256",
                "target_cache_sha256"):
        if current_metadata.get(key) != probe["metadata"].get(key):
            raise ValueError(f"reprobe source changed: {key}")
    amp = bool(config["training"]["amp"])
    if amp != probe["precision"]["forward_autocast"]:
        raise ValueError("reprobe AMP precision must match source measurements")
    records = []
    for row in probe["measurements"]:
        batch = runtime.collate_indices(dataset, row["indices"], device,
                                        seed=row["batch"])
        for key in ("target_path", "reference_path"):
            if row.get(key) != batch.get(key):
                raise ValueError(f"reprobe dataset indices changed: {key}")
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            outputs = runtime.call_model(model, batch, "refine")
            losses = runtime.loss_for(model, outputs, batch, config)
        terms = {name: losses[name] for name in RAW_NAMES}
        terms.update({"combined_img1": losses["img1"],
                      "combined_bf_local": losses["bf_local"],
                      "combined_j1": losses["j1"],
                      "weighted_img1": config["weights"]["lambda_img1"]
                      * losses["img1"],
                      "weighted_bf_local": config["weights"]["lambda_bf"]
                      * losses["bf_local"],
                      "weighted_j1": config["weights"]["lambda_j1"]
                      * losses["j1"],
                      "weighted_total": losses["total"],
                      "head_monitor": losses["raw_head"]})
        grams = runtime.parameter_grams(terms, parameters)
        if grams["norms"]["head_monitor"] != 0:
            raise ValueError("stage-II frozen head monitor unexpectedly has gradient")
        item = deepcopy(row)
        item.update({"gradients": {"refiner": grams, "all_trainable": grams},
                     "losses": runtime.scalar_losses(losses)})
        records.append(item)
        print(json.dumps({"remeasured_batch": row["batch"], "pool": row["pool"]}),
              flush=True)
    result = deepcopy(probe)
    result.update({"measurements": records, "metadata": current_metadata,
                   "refiner_trainable_parameter_count": sum(
                       value.numel() for _, value in parameters
                   ), "stage_one_requires_grad": False,
                   "actual_composites_measured_by_autograd": True})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--device")
    args = parser.parse_args()
    for path in (args.report, args.config):
        if Path(path).exists():
            raise FileExistsError(f"refusing to replace {path}")
    probe = json.loads(Path(args.probe).read_text(encoding="utf-8"))
    config = json.loads(Path(args.source_config).read_text(encoding="utf-8"))
    validate_probe(probe, config)
    if args.verify:
        probe = remeasure(probe, config, args.device)
        config["weights_status"] = (
            "refine_reprobed_initial_gradient_ratios_requires_quality_review"
        )
    else:
        config = derive_config(probe, config)
    report = make_report(probe, config, verified=args.verify)
    report.update({"source_probe": args.probe, "source_config": args.source_config})
    for key in ("refiner_trainable_parameter_count", "stage_one_requires_grad",
                "actual_composites_measured_by_autograd"):
        if key in probe:
            report[key] = probe[key]
    dump_new(args.report, report)
    dump_new(args.config, config)
    print(json.dumps({key: report[key] for key in (
        "status", "j_coefficients", "weights", "gram_composed_fit_ratio_summary"
    )}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
