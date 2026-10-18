"""Analytic gradient-vector tests for the CPU Run C v2 selector."""

import copy
import hashlib
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from tools.calibrate_runc_v2 import (
    POLICY_SCHEMA,
    REFERENCE_POLICY,
    select_weights,
    sha256_file,
)
from tools.runc_v2_runtime import validate_calibration


def measurement(vectors):
    names = list(vectors)
    matrix = np.asarray([vectors[name] for name in names], dtype=float)
    gram = matrix @ matrix.T
    return {
        "terms": names,
        "gram": gram.tolist(),
        "norms": {
            name: float(np.linalg.norm(matrix[index]))
            for index, name in enumerate(names)
        },
        "cosine": {}, "parameter_tensors": 1,
    }


def probe(stage="ia1", mixed_centroid=False, aligned=False):
    if stage.startswith("ia"):
        vectors = {
            "head": [2, 0, 0, 0],
            "head_logsigma": [0, 4, 0, 0],
            "head_centroid": [0, 0, 1, 0],
            "head_shape": [0, 0, 0, 2],
        }
        count_key = "count_head_parts"
        eligibility = "head_relative_eligible_parts"
    else:
        vectors = {
            "density": [2, 0, 0, 0, 0],
            "occupancy_bce": [0, 4, 0, 0, 0],
            "occupancy_dice": [0, 1, 0, 0, 0],
            "j_logsigma": [0, 0, 4, 0, 0],
            "j_centroid": [0, 0, 0, 1, 0],
            "j_shape": [0, 0, 0, 0, 2],
        }
        count_key = "count_j_coarse_parts"
        eligibility = "coarse_relative_eligible_parts"
    if aligned:
        vectors = {name: [np.linalg.norm(value), 0, 0, 0]
                   for name, value in vectors.items()}
    rows = []
    for index in range(9):
        local = copy.deepcopy(vectors)
        eligible = not mixed_centroid or index < 3
        if not eligible:
            key = "head_centroid" if stage.startswith("ia") else "j_centroid"
            local[key] = [0] * len(local[key])
        rows.append({
            "batch": index, "author": str(index),
            "target_path": [f"author-{index}/a.png"],
            "reference_path": [f"author-{index}/b.png"],
            "query_original_identity": [f"author-{index}:" + hashlib.sha256(
                f"{index}:query:original".encode()).hexdigest()],
            "reference_original_identity": [f"author-{index}:" + hashlib.sha256(
                f"{index}:reference:original".encode()).hexdigest()],
            **{f"{side}_{kind}_sha256": [hashlib.sha256(
                f"{index}:{side}:{kind}".encode()).hexdigest()]
               for side in ("query", "reference")
               for kind in ("original", "target")},
            "reference_policy": REFERENCE_POLICY,
            "losses": {count_key: 2, "raw_head": 10000 + index},
            eligibility: 2 if eligible else 0,
            "gradients": {"all_trainable": measurement(local)},
        })
    names = rows[0]["gradients"]["all_trainable"]["terms"]
    return {
        "schema": "runc-v2-runtime-v1", "status": "measured", "stage": stage,
        "authors": [str(index) for index in range(9)],
        "metadata": {
            "source_sha": "b" * 40, "initial_checkpoint_sha256": "a" * 64,
            "checkpoint_sha256": "c" * 64,
            "reference_policy": REFERENCE_POLICY,
        },
        "model_config": {"hidden_dim": 256, "stage": stage},
        "backend": {"matmul_tf32": False}, "amp": True,
        "rows": rows,
        "median_raw_norm": {
            name: float(np.median([
                row["gradients"]["all_trainable"]["norms"][name]
                for row in rows
            ])) for name in names
        },
    }


def write_probe(tmp_path, value):
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(value))
    return path


def median_ratio(result, name):
    return result["achieved_ratios"][name]["median"]


def test_legacy_probe_without_reference_policy_is_rejected(tmp_path):
    value = probe()
    value["metadata"].pop("reference_policy")
    with pytest.raises(ValueError, match="reference policy"):
        select_weights(write_probe(tmp_path, value))


@pytest.mark.parametrize("kind", ["original", "target"])
def test_equal_reference_image_sha_is_rejected(tmp_path, kind):
    value = probe()
    row = value["rows"][0]
    row[f"reference_{kind}_sha256"] = row[f"query_{kind}_sha256"]
    with pytest.raises(ValueError, match="identical.*image SHA"):
        select_weights(write_probe(tmp_path, value))


def test_missing_reference_sha_cannot_bypass_identity_check(tmp_path):
    value = probe()
    row = value["rows"][0]
    row.pop("reference_original_sha256")
    row.pop("query_original_sha256")
    with pytest.raises(ValueError, match="SHA lengths"):
        select_weights(write_probe(tmp_path, value))


def test_same_original_identity_is_rejected_even_with_different_hashes(tmp_path):
    value = probe()
    value["rows"][0]["reference_original_identity"] = (
        value["rows"][0]["query_original_identity"]
    )
    with pytest.raises(ValueError, match="self-reference"):
        select_weights(write_probe(tmp_path, value))


def test_relative_probe_reference_remains_portable(tmp_path, monkeypatch):
    path = write_probe(tmp_path, probe())
    monkeypatch.chdir(tmp_path)
    result = select_weights(path.name)
    assert result["source_probe"] == path.name
    assert result["source_probe_sha256"] == sha256_file(path)


def test_ia_policy_matches_analytic_orthogonal_gradients(tmp_path):
    result = select_weights(write_probe(tmp_path, probe()))
    weights = result["selected_weights"]
    assert weights["head"] == 1
    assert weights["head_logsigma"] == pytest.approx(1)
    assert weights["head_centroid"] == pytest.approx(2)
    assert weights["head_shape"] == 1
    assert weights["head_geo"] == pytest.approx(2 / math.sqrt(24))
    assert median_ratio(result, "geometry_to_anchor") == pytest.approx(1)
    assert median_ratio(result, "sigma_to_shape") == pytest.approx(2)
    relative_ratio = median_ratio(result, "centroid_to_shape_eligible")
    assert relative_ratio == pytest.approx(1)
    row = result["rows"][0]
    assert row["selected_preclip_norm"] == pytest.approx(math.sqrt(8))
    assert row["geometry"]["composite_over_sum_norms"] == pytest.approx(
        math.sqrt(24) / 8)
    assert weights["density"] == weights["occupancy"] == 0
    assert weights["j_coarse"] == 0


def test_actual_gram_not_sum_of_norms_controls_common_multiplier(tmp_path):
    result = select_weights(write_probe(tmp_path, probe(aligned=True)))
    assert result["selected_weights"]["head_geo"] == pytest.approx(0.25)
    fraction = result["rows"][0]["geometry"]["composite_over_sum_norms"]
    assert fraction == pytest.approx(1)


def test_centroid_calibration_excludes_ineligible_single_part_batches(
        tmp_path):
    result = select_weights(write_probe(tmp_path, probe(mixed_centroid=True)))
    assert result["selected_weights"]["head_centroid"] == pytest.approx(2)
    assert result["selection_details"]["relative_eligible_batches"] == [
        0, 1, 2]
    summary = result["achieved_ratios"]["centroid_to_shape_eligible"]
    assert summary["count"] == 3
    assert summary["median"] == 1


def test_ib_density_anchor_and_conservative_composite_j_ratio(tmp_path):
    result = select_weights(write_probe(tmp_path, probe("ib")))
    weights = result["selected_weights"]
    assert weights["density"] == 1
    assert weights["occupancy"] == pytest.approx(0.5)
    assert weights["occupancy_dice"] == 0
    assert weights["j_coarse"] == pytest.approx(1 / math.sqrt(24))
    assert median_ratio(result, "geometry_to_anchor") == pytest.approx(0.5)
    assert median_ratio(result, "occupancy_to_density") == pytest.approx(1)
    assert weights["head"] == weights["head_geo"] == 0


def test_selection_is_deterministic_and_does_not_use_raw_loss_values(tmp_path):
    value = probe()
    path = write_probe(tmp_path, value)
    first = select_weights(path)
    assert select_weights(path) == first
    for row in value["rows"]:
        row["losses"]["raw_head"] = 1e-20
    second = select_weights(write_probe(tmp_path, value))
    assert first["selected_weights"] == second["selected_weights"]
    assert first["source_probe_sha256"] != second["source_probe_sha256"]


def test_cross_batch_ratios_use_median_not_pooled_mean(tmp_path):
    value = probe()
    sigma_norms = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for row, sigma in zip(value["rows"], sigma_norms):
        row["gradients"]["all_trainable"] = measurement({
            "head": [2, 0, 0, 0],
            "head_logsigma": [0, sigma, 0, 0],
            "head_centroid": [0, 0, 1, 0],
            "head_shape": [0, 0, 0, 2],
        })
    result = select_weights(write_probe(tmp_path, value))
    assert result["selected_weights"]["head_logsigma"] == pytest.approx(0.25)
    assert median_ratio(result, "sigma_to_shape") == pytest.approx(2)
    assert median_ratio(result, "geometry_to_anchor") == pytest.approx(1)


def test_expected_context_and_runner_validation_accept_same_bound_selection(
        tmp_path):
    value = probe()
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"verified mock checkpoint bytes")
    value["metadata"]["checkpoint_sha256"] = sha256_file(checkpoint)
    config = {
        "stage": "ia1", "checkpoint": str(checkpoint),
        "initial_checkpoint_sha256": "a" * 64,
        "training": {"amp": True, "clip_grad": 1},
        "model": {"hidden_dim": 256}, "backend": {"matmul_tf32": False},
        "data": {"reference_policy": REFERENCE_POLICY},
    }
    result = select_weights(write_probe(tmp_path, value), config)
    assert result["starting_checkpoint_sha256"] == sha256_file(checkpoint)
    assert result["rows"][0]["predicted_clip_ratio"] == pytest.approx(
        1 / math.sqrt(8))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(result))
    config["calibration_record"] = str(selection)
    config["weights"] = result["selected_weights"]
    assert validate_calibration(
        config)["selected_weights"] == result["selected_weights"]
    config["training"]["amp"] = False
    with pytest.raises(ValueError, match="AMP"):
        select_weights(tmp_path / "probe.json", config)


@pytest.mark.parametrize("change", ["stage", "checkpoint", "model", "backend"])
def test_context_mismatches_rejected(tmp_path, change):
    value = probe()
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"one")
    value["metadata"]["checkpoint_sha256"] = sha256_file(checkpoint)
    config = {
        "stage": "ia1", "checkpoint": str(checkpoint),
        "initial_checkpoint_sha256": "a" * 64,
        "training": {"amp": True}, "model": {"hidden_dim": 256},
        "backend": {"matmul_tf32": False},
        "data": {"reference_policy": REFERENCE_POLICY},
    }
    if change == "stage":
        config["stage"] = "ib"
    elif change == "checkpoint":
        checkpoint.write_bytes(b"two")
    elif change == "model":
        config["model"]["hidden_dim"] = 128
    else:
        config["backend"]["matmul_tf32"] = True
    with pytest.raises(ValueError):
        select_weights(write_probe(tmp_path, value), config)


@pytest.mark.parametrize("kind", ["nonfinite", "asymmetric", "nonpsd", "norm"])
def test_malformed_actual_gram_rejected(tmp_path, kind):
    value = probe()
    item = value["rows"][0]["gradients"]["all_trainable"]
    if kind == "nonfinite":
        item["gram"][0][0] = float("nan")
    elif kind == "asymmetric":
        item["gram"][0][1] = 1
    elif kind == "nonpsd":
        item["gram"][0][1] = item["gram"][1][0] = 100
    else:
        item["norms"]["head"] = 100
    with pytest.raises(ValueError):
        select_weights(write_probe(tmp_path, value))


def test_zero_eligible_gradient_and_missing_multipart_batches_stop(tmp_path):
    value = probe()
    for row in value["rows"]:
        row["head_relative_eligible_parts"] = 0
    with pytest.raises(ValueError, match="multipart"):
        select_weights(write_probe(tmp_path, value))
    value = probe()
    for row in value["rows"]:
        item = row["gradients"]["all_trainable"]
        item["gram"][3] = [0] * 4
        for gram_row in item["gram"]:
            gram_row[3] = 0
        item["norms"]["head_shape"] = 0
    with pytest.raises(ValueError, match="zero/tiny"):
        select_weights(write_probe(tmp_path, value))


def test_degenerate_cancellation_and_excessive_coefficients_stop(tmp_path):
    value = probe()
    # Norm policy makes geometry vectors 4, 2, 2; opposing terms cancel.
    item = measurement({
        "head": [1, 0], "head_logsigma": [0, -4],
        "head_centroid": [0, 1], "head_shape": [0, 2],
    })
    for row in value["rows"]:
        row["gradients"]["all_trainable"] = copy.deepcopy(item)
    with pytest.raises(ValueError, match="cancel"):
        select_weights(write_probe(tmp_path, value))
    with pytest.raises(ValueError, match="safety bound"):
        select_weights(write_probe(tmp_path, probe()), max_coefficient=1.5)


@pytest.mark.parametrize("change",
                         ["authors", "status", "ic", "lineage", "amp"])
def test_missing_or_unsupported_provenance_stops(tmp_path, change):
    value = probe()
    if change == "authors":
        value["authors"].pop()
    elif change == "status":
        value["status"] = "planned"
    elif change == "ic":
        value["stage"] = "ic"
    elif change == "lineage":
        value["metadata"]["initial_checkpoint_sha256"] = "old"
    else:
        value.pop("amp")
    with pytest.raises(ValueError):
        select_weights(write_probe(tmp_path, value))


def test_cli_writes_runner_selection_record_without_overwriting(tmp_path):
    path = write_probe(tmp_path, probe())
    output = tmp_path / "selection.json"
    command = [
        sys.executable, "tools/calibrate_runc_v2.py", "--probe", str(path),
        "--output", str(output),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True)
    assert json.loads(output.read_text())["schema"] == POLICY_SCHEMA
    assert json.loads(result.stdout)[
        "status"] == "selected_from_measured_gradients"
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0
