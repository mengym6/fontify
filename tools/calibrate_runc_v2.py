"""Select Run C v2 starting weights from measured parameter-gradient Grams.

This CPU-only calculation does not load a model, run training, or use raw loss
values. The declared 2:1:1 geometry policy is a starting experiment policy,
not an optimality claim. Every composite norm uses the measured Gram matrix.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

POLICY_SCHEMA = "runc-v2-gradient-selection-v1"
PROBE_SCHEMA = "runc-v2-runtime-v1"
REFERENCE_POLICY = "distinct-original-reference-v1"
ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite probe/policy value")
    if isinstance(value, dict):
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _finite_tree(child)


def _number(value, name, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value)):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(
            f"{name} must be {'positive' if positive else '>= 0'}")
    return float(value)


def _summary(values):
    values = list(values)
    if not values:
        return {"count": 0, "median": None, "min": None, "max": None}
    return {
        "count": len(values), "median": statistics.median(values),
        "min": min(values), "max": max(values),
    }


def _validated_gram(measurement):
    """Reject inconsistent or non-PSD Grams before selecting weights."""
    import numpy as np

    names = measurement["terms"]
    gram = measurement["gram"]
    if not names or len(set(names)) != len(names):
        raise ValueError("Gram terms must be a nonempty unique list")
    size = len(names)
    if len(gram) != size or any(len(row) != size for row in gram):
        raise ValueError("Gram dimensions differ from term names")
    matrix = np.asarray(gram, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite Gram")
    scale = max(float(np.abs(matrix).max()), 1e-30)
    tolerance = scale * 1e-5
    if not np.allclose(matrix, matrix.T, atol=scale * 1e-8, rtol=1e-8):
        raise ValueError("Gram is not symmetric")
    if np.linalg.eigvalsh(matrix).min() < -tolerance:
        raise ValueError("Gram is not positive semidefinite")
    norms = {}
    for index, name in enumerate(names):
        if matrix[index, index] < 0:
            raise ValueError("Negative squared gradient norm")
        norm = math.sqrt(float(matrix[index, index]))
        reported = _number(measurement["norms"][name], f"norm.{name}")
        if not math.isclose(norm, reported, rel_tol=1e-5, abs_tol=1e-12):
            raise ValueError(
                f"Gram diagonal differs from reported norm: {name}")
        norms[name] = norm
    return {"terms": list(names), "gram": matrix.tolist(), "norms": norms}


def _quadratic(measurement, coefficients):
    names, gram = measurement["terms"], measurement["gram"]
    values = [coefficients.get(name, 0.0) for name in names]
    square = math.fsum(
        values[i] * gram[i][j] * values[j]
        for i in range(len(names)) for j in range(len(names))
    )
    sum_norms = math.fsum(
        abs(values[index]) * measurement["norms"][name]
        for index, name in enumerate(names)
    )
    if square < -max(sum_norms ** 2 * 1e-5, 1e-30):
        raise ValueError("Selected Gram quadratic form is negative")
    return math.sqrt(max(square, 0.0)), sum_norms


def _diagnostics(measurement, coefficients):
    """Report weighted Gram, cosines and cancellation per parameter group."""
    names, gram = measurement["terms"], measurement["gram"]
    norm, sum_norms = _quadratic(measurement, coefficients)
    weighted_norms = {
        name: abs(coefficients.get(name, 0)) * measurement["norms"][name]
        for name in names
    }
    cosine = {}
    for i, left in enumerate(names):
        for j, right in enumerate(names[:i]):
            denominator = measurement["norms"][left] * \
                measurement["norms"][right]
            cosine[f"{left}:{right}"] = (
                max(-1.0, min(1.0, gram[i][j] / denominator))
                if denominator > 0 else None
            )
    return {
        "terms": names,
        "actual_gram": gram,
        "coefficients": {name: coefficients.get(name, 0.0) for name in names},
        "weighted_gram": [
            [gram[i][j] * coefficients.get(left, 0)
             * coefficients.get(right, 0) for j, right in enumerate(names)]
            for i, left in enumerate(names)
        ],
        "weighted_term_norms": weighted_norms,
        "pair_cosine": cosine,
        "composite_norm": norm,
        "sum_term_norms": sum_norms,
        "composite_over_sum_norms": norm / sum_norms if sum_norms else None,
    }


def _coefficient(value, name, maximum):
    value = _number(value, name, positive=True)
    if value > maximum:
        raise ValueError(f"{name}={value:g} exceeds safety bound {maximum:g}")
    return value


def _positive_norm(row, term, epsilon):
    value = row["measurements"]["all_trainable"]["norms"].get(term)
    if value is None or value <= epsilon:
        raise ValueError(
            f"Eligible batch {row['batch']} has zero/tiny gradient for {term}"
        )
    return value


def _geometry_policy(rows, prefix, epsilon, maximum):
    eligible_key = ("head_relative_eligible_parts" if prefix == "head"
                    else "coarse_relative_eligible_parts")
    count_key = ("count_head_parts" if prefix == "head"
                 else "count_j_coarse_parts")
    geometric = [row for row in rows if row["losses"].get(count_key, 0) > 0]
    relative = [row for row in geometric if row.get(eligible_key, 0) >= 2]
    if len(geometric) < 2 or len(relative) < 2:
        raise ValueError(
            "Geometry selection needs >=2 valid/multipart batches")
    sigma, centroid, shape = [f"{prefix}_{name}" for name in (
        "logsigma", "centroid", "shape"
    )]
    scale_ratios = [
        2 * _positive_norm(row, shape, epsilon)
        / _positive_norm(row, sigma, epsilon) for row in geometric
    ]
    centroid_ratios = [
        _positive_norm(row, shape, epsilon)
        / _positive_norm(row, centroid, epsilon) for row in relative
    ]
    coefficients = {
        sigma: _coefficient(statistics.median(scale_ratios), sigma, maximum),
        centroid: _coefficient(
            statistics.median(centroid_ratios), centroid, maximum
        ),
        shape: 1.0,
    }
    return coefficients, {
        "shape_anchor": 1.0, "target_norm_ratio": [2, 1, 1],
        "valid_geometry_batches": [row["batch"] for row in geometric],
        "relative_eligible_batches": [row["batch"] for row in relative],
        "per_batch_sigma_anchor_ratios": scale_ratios,
        "per_batch_centroid_anchor_ratios": centroid_ratios,
        "coefficient_rule": (
            "median(target_ratio * raw_shape_norm / raw_term_norm)"
        ),
    }


def _validate_context(probe, expected_config):
    if expected_config is None:
        return
    if (expected_config.get("data", {}).get("reference_policy")
            != REFERENCE_POLICY):
        raise ValueError("Expected reference policy is absent or obsolete")
    if (expected_config["data"]["reference_policy"]
            != probe["metadata"]["reference_policy"]):
        raise ValueError("Expected reference policy differs from probe")
    if expected_config["stage"] != probe["stage"]:
        raise ValueError("Expected configuration stage differs from probe")
    if (expected_config["initial_checkpoint_sha256"]
            != probe["metadata"]["initial_checkpoint_sha256"]):
        raise ValueError("Expected initialization lineage differs from probe")
    if sha256_file(expected_config["checkpoint"]
                   ) != probe["metadata"]["checkpoint_sha256"]:
        raise ValueError("Expected starting checkpoint differs from probe")
    if expected_config["training"]["amp"] != probe["amp"]:
        raise ValueError("Expected AMP setting differs from probe")
    for name, value in expected_config.get("model", {}).items():
        if name != "stage" and probe["model_config"].get(name) != value:
            raise ValueError(f"Expected model setting differs: {name}")
    for name, value in expected_config.get("backend", {}).items():
        if name in probe.get(
                "backend", {}) and probe["backend"][name] != value:
            raise ValueError(f"Expected backend setting differs: {name}")


def _validate_reference_rows(probe):
    """Reject old/self-reference probes before any mathematical selection."""
    if probe.get("metadata", {}).get("reference_policy") != REFERENCE_POLICY:
        raise ValueError("Probe reference policy is absent or obsolete")
    for row in probe["rows"]:
        query = row.get("query_original_identity")
        reference = row.get("reference_original_identity")
        if (not isinstance(query, list) or not isinstance(reference, list)
                or not query or len(query) != len(reference)):
            raise ValueError("Probe lacks aligned query/reference identities")
        if any(not isinstance(item, str) or not item for item in
               query + reference):
            raise ValueError("Probe query/reference identities must be nonempty")
        for name in ("target_path", "reference_path"):
            if (not isinstance(row.get(name), list)
                    or len(row[name]) != len(query)):
                raise ValueError("Probe identity/path batch lengths differ")
        policy = row.get("reference_policy")
        if isinstance(policy, list):
            valid_policy = (len(policy) == len(query)
                            and all(item == REFERENCE_POLICY for item in policy))
        else:
            valid_policy = policy == REFERENCE_POLICY
        if not valid_policy:
            raise ValueError("Probe row reference policy is absent or obsolete")
        if any(left == right for left, right in zip(query, reference)):
            raise ValueError("Probe contains a self-reference original identity")
        for kind in ("original", "target"):
            qkey, rkey = f"query_{kind}_sha256", f"reference_{kind}_sha256"
            qhash, rhash = row.get(qkey), row.get(rkey)
            if (not isinstance(qhash, list) or not isinstance(rhash, list)
                    or len(qhash) != len(query) or len(rhash) != len(query)):
                raise ValueError("Probe query/reference SHA lengths differ")
            if any(not isinstance(item, str) or not item
                   for item in qhash + rhash):
                raise ValueError("Probe query/reference SHA must be nonempty")
            if any(len(item) != 64 or any(
                    char not in "0123456789abcdef" for char in item
            ) for item in qhash + rhash):
                raise ValueError("Probe query/reference SHA must be SHA256")
            if any(left == right for left, right in zip(qhash, rhash)):
                raise ValueError(f"Probe contains identical {kind} image SHA")
        for side in ("query", "reference"):
            for identity, checksum in zip(
                    row[f"{side}_original_identity"],
                    row[f"{side}_original_sha256"]):
                if identity.rsplit(":", 1)[-1] != checksum:
                    raise ValueError("Probe original identity and SHA disagree")


def _portable_probe_reference(requested, resolved):
    if requested.is_absolute():
        try:
            return str(resolved.relative_to(ROOT))
        except ValueError:
            pass
    return str(requested)


def select_weights(probe_path, expected_config=None, max_coefficient=1000.0,
                   min_gradient=1e-12):
    """Select IA/IB policy and return a SHA-bound, auditable record."""
    requested_probe_path = Path(probe_path)
    probe_path = requested_probe_path.resolve()
    probe_bytes = probe_path.read_bytes()
    probe = json.loads(probe_bytes)
    _finite_tree(probe)
    _number(max_coefficient, "max_coefficient", positive=True)
    _number(min_gradient, "min_gradient", positive=True)
    if max_coefficient < 1:
        raise ValueError("Safety bound must admit unit anchors")
    if (probe.get("schema") != PROBE_SCHEMA
            or probe.get("status") != "measured"):
        raise ValueError("Expected an actual measured Run C v2 probe")
    stage = probe.get("stage")
    if stage not in ("ia0", "ia1", "ib"):
        raise ValueError("Automatic starting policy supports IA0/IA1/IB only")
    for name in ("initial_checkpoint_sha256", "checkpoint_sha256"):
        value = probe.get("metadata", {}).get(name, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Probe lacks a valid {name}")
    if type(probe.get("amp")) is not bool or not probe.get("model_config"):
        raise ValueError("Probe must record AMP and model configuration")
    _validate_reference_rows(probe)
    authors = set(probe.get("authors", []))
    if len(authors) != 9 or {row.get("author")
                             for row in probe["rows"]} != authors:
        raise ValueError("Measured rows must cover the same nine authors")
    _validate_context(probe, expected_config)
    rows = []
    for row in probe["rows"]:
        row = dict(row)
        row["measurements"] = {
            name: _validated_gram(measurement)
            for name, measurement in row["gradients"].items()
        }
        if "all_trainable" not in row["measurements"]:
            raise ValueError("Missing all_trainable parameter-gradient Gram")
        rows.append(row)
    prefix = "head" if stage.startswith("ia") else "j"
    geometry, selection = _geometry_policy(
        rows, prefix, min_gradient, max_coefficient
    )
    base_term = "head" if prefix == "head" else "density"
    ratio = 1.0 if prefix == "head" else 0.5
    common_candidates = []
    for row in rows:
        base = _positive_norm(row, base_term, min_gradient)
        composite, sum_norms = _quadratic(
            row["measurements"]["all_trainable"], geometry
        )
        # Zero-part batches have no geometry signal and cannot calibrate a
        # common geometry multiplier. They remain in final full-loss reports.
        count_key = ("count_head_parts" if prefix == "head"
                     else "count_j_coarse_parts")
        if not row["losses"].get(count_key, 0):
            continue
        if composite <= min_gradient or composite <= sum_norms * 1e-8:
            raise ValueError(
                "Geometry gradients cancel or have no finite signal")
        common_candidates.append(ratio * base / composite)
    if len(common_candidates) < 2:
        raise ValueError("Composite geometry selection needs multiple batches")
    common = _coefficient(
        statistics.median(
            common_candidates), f"{prefix}_common", max_coefficient
    )
    weights = {
        "head": 1.0 if prefix == "head" else 0.0,
        "head_geo": common if prefix == "head" else 0.0,
        "head_logsigma": geometry.get("head_logsigma", 1.0),
        "head_centroid": geometry.get("head_centroid", 1.0),
        "head_shape": 1.0,
        "density": 0.0 if prefix == "head" else 1.0,
        "occupancy": 0.0,
        "occupancy_dice": 0.0,
        "j_coarse": common if prefix == "j" else 0.0,
        "j_centroid": geometry.get("j_centroid", 1.0),
        "j_logsigma": geometry.get("j_logsigma", 1.0),
        "j_shape": 1.0,
    }
    leaf = {name: value * common for name, value in geometry.items()}
    leaf[base_term] = 1.0
    if prefix == "j":
        occupancy_candidates = [
            _positive_norm(row, "density", min_gradient)
            / _positive_norm(row, "occupancy_bce", min_gradient)
            for row in rows
        ]
        occupancy = _coefficient(
            statistics.median(
                occupancy_candidates), "occupancy", max_coefficient
        )
        weights["occupancy"] = occupancy
        leaf["occupancy_bce"] = occupancy
        selection["occupancy_candidates"] = occupancy_candidates
    # Effective coefficients can exceed a bound even if both factors pass it.
    for name, value in leaf.items():
        _coefficient(value, f"effective_{name}", max_coefficient)
    diagnostics = []
    ratios = {"geometry_to_anchor": [], "sigma_to_shape": [],
              "centroid_to_shape_eligible": [], "occupancy_to_density": []}
    clip = (None if expected_config is None
            else expected_config["training"].get("clip_grad"))
    for row in rows:
        measurements = row["measurements"]
        all_gradient = measurements["all_trainable"]
        selected = _diagnostics(all_gradient, leaf)
        family = _diagnostics(all_gradient, {
            name: value * common for name, value in geometry.items()
        })
        norms = all_gradient["norms"]
        base = norms[base_term]
        ratios["geometry_to_anchor"].append(family["composite_norm"] / base)
        if norms[f"{prefix}_shape"] > min_gradient:
            ratios["sigma_to_shape"].append(
                geometry[f"{prefix}_logsigma"] * norms[f"{prefix}_logsigma"]
                / norms[f"{prefix}_shape"]
            )
            eligible_key = ("head_relative_eligible_parts" if prefix == "head"
                            else "coarse_relative_eligible_parts")
            if row.get(eligible_key, 0) >= 2:
                ratios["centroid_to_shape_eligible"].append(
                    geometry[f"{prefix}_centroid"] *
                    norms[f"{prefix}_centroid"]
                    / norms[f"{prefix}_shape"]
                )
        if prefix == "j":
            ratios["occupancy_to_density"].append(
                weights["occupancy"] * norms["occupancy_bce"] / base
            )
        diagnostics.append({
            "batch": row["batch"], "author": row["author"],
            "target_path": row.get("target_path", []),
            "reference_path": row.get("reference_path", []),
            "query_original_identity": row["query_original_identity"],
            "reference_original_identity": row["reference_original_identity"],
            **{f"{side}_{kind}_sha256": row[f"{side}_{kind}_sha256"]
               for side in ("query", "reference")
               for kind in ("original", "target")},
            "reference_policy": REFERENCE_POLICY,
            "geometry": family,
            "parameter_groups": {
                name: _diagnostics(measurement, leaf)
                for name, measurement in measurements.items()
            },
            "selected_preclip_norm": selected["composite_norm"],
            "clip_threshold": clip,
            "predicted_clip_ratio": (
                min(1.0, clip / max(selected["composite_norm"], min_gradient))
                if clip is not None else None
            ),
        })
    rationale = (
        "CPU selection from measured parameter-gradient Grams only. "
        "Geometry uses shape=1 and cross-batch median raw-norm ratios for "
        "logsigma:relative-centroid:shape approximately 2:1:1; centroid uses "
        "eligible multi-part batches only. The common multiplier uses each "
        "batch's actual Gram composite, targeting geometry/Head=1 for IA or "
        "J/density=0.5 for IB. IB targets occupancy-BCE/density=1 "
        "and leaves Dice disabled. These are conservative starting policies, "
        "not demonstrated optimal weights or evidence of training benefit."
    )
    return {
        "schema": POLICY_SCHEMA,
        "status": "selected_from_measured_gradients",
        "stage": stage,
        "source_probe": _portable_probe_reference(
            requested_probe_path, probe_path
        ),
        "source_probe_sha256": hashlib.sha256(probe_bytes).hexdigest(),
        "initial_checkpoint_sha256": probe["metadata"][
            "initial_checkpoint_sha256"
        ],
        "starting_checkpoint_sha256": probe["metadata"]["checkpoint_sha256"],
        "reference_policy": REFERENCE_POLICY,
        "source_sha": probe["metadata"].get("source_sha"),
        "amp": probe["amp"], "model_config": probe["model_config"],
        "backend": probe.get("backend", {}),
        "selected_weights": weights,
        "selected_leaf_coefficients": leaf,
        "rationale": rationale,
        "policy": {
            "geometry_ratio": [2, 1, 1], "geometry_anchor": "shape",
            "geometry_to_anchor": ratio,
            "anchor": base_term,
            "occupancy_to_density": 1.0 if prefix == "j" else None,
            "max_coefficient": max_coefficient,
            "min_gradient": min_gradient,
            "bound_interpretation": (
                "numerical stop bound, not an optimum range"
            ),
        },
        "selection_details": {
            **selection, "common_candidates": common_candidates
        },
        "achieved_ratios": {
            name: _summary(values) for name, values in ratios.items()
        },
        "authors": sorted(authors), "rows": diagnostics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config",
        help="Optional expected training/probe config")
    parser.add_argument("--max-coefficient", type=float, default=1000.0)
    parser.add_argument("--min-gradient", type=float, default=1e-12)
    args = parser.parse_args()
    expected = (json.loads(Path(args.config).read_text(encoding="utf-8"))
                if args.config else None)
    result = select_weights(
        args.probe, expected, max_coefficient=args.max_coefficient,
        min_gradient=args.min_gradient,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(
            result,
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "status": result["status"], "stage": result["stage"],
        "selected_weights": result["selected_weights"],
        "achieved_ratios": result["achieved_ratios"], "output": str(output),
    }, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
