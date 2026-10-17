"""Refiner calibration contracts; these do not replace real CUDA probes."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from tools import calibrate_runc_refine as calibration
from tools import structure_first as runtime


def configuration():
    return {
        "schema": runtime.SCHEMA, "experiment": "runc", "stage": "refine",
        "checkpoint": "selected-coarse.pth",
        "j_coefficients": dict.fromkeys(calibration.J_NAMES, 1.0),
        "weights": dict.fromkeys(runtime.WEIGHT_KEYS, 1.0),
        "lowpass": {"kernel_size": 7, "sigma": 1.5, "boundary": "reflect"},
        "training": {"amp": False},
    }


def vectors(has_jt=True, scale=1.0):
    values = {
        "raw_img_smooth": [1.0, 0.0, 0.0, 0.0],
        "raw_img_gram": [0.0, 2.0, 0.0, 0.0],
        "raw_img_edge": [-0.5, 0.0, 0.1, 0.0],
        "raw_bf_smooth": [0.7, 0.2, 0.0, 0.0],
        "raw_bf_edge": [-0.1, 0.3, 0.0, 0.0],
        "j_centroid": [0.0, 0.0, 0.0, 3.0],
        "j_logsigma": [0.0, 0.0, 4.0, 0.0],
        "j_shape": [0.0, 0.5, 0.2, 0.0],
    }
    if not has_jt:
        for key in ("j_centroid", "j_logsigma", "j_shape"):
            values[key] = [0.0] * 4
    values["raw_j1"] = [sum(values[key][index] for key in
                                ("j_centroid", "j_logsigma", "j_shape"))
                            for index in range(4)]
    return {name: [scale * value for value in vector]
            for name, vector in values.items()}


def gradient_data(values):
    names = list(values)
    gram = [[sum(a * b for a, b in zip(values[left], values[right]))
             for right in names] for left in names]
    norms = {name: sum(value * value for value in vector) ** 0.5
             for name, vector in values.items()}
    cosine = {}
    for index, left in enumerate(names):
        for right in names[:index]:
            denominator = norms[left] * norms[right]
            cosine[f"{left}:{right}"] = sum(
                a * b for a, b in zip(values[left], values[right])
            ) / denominator if denominator else None
    return {"terms": names, "gram": gram, "norms": norms, "cosine": cosine}


def probe_fixture():
    rows = []
    for index in range(18):
        has_jt = index < 9
        pool = "dual" if has_jt else "bf_only"
        data = gradient_data(vectors(has_jt, 1.0 + index / 20.0))
        rows.append({
            "batch": index, "pool": pool, "indices": [index], "batch_size": 1,
            "sample_has_bf": [True], "sample_has_jt": [has_jt],
            "target_path": [f"font/new/Author{index % 9}BF/images_text/a.png"],
            "reference_path": [f"reference-{index}.png"],
            "gradients": {"refiner": data, "all_trainable": deepcopy(data)},
        })
    return {"measurements": rows, "metadata": {"parent_checkpoint_sha256": "sha"},
            "precision": {"forward_autocast": False, "forward_dtype": "float32"}}


def test_derivation_hits_initial_ratios_and_uses_cross_term_cancellation():
    probe, source = probe_fixture(), configuration()
    config = calibration.derive_config(probe, source)
    report = calibration.make_report(probe, config)
    ratios = report["gram_composed_fit_ratio_summary"]
    for name, expected in (("gram_over_smooth", 0.5), ("edge_over_smooth", 0.1),
                           ("local_edge_over_smooth", 0.1), ("bf_over_img", 0.5),
                           ("j_over_img", 0.25)):
        assert ratios[name]["median"] == pytest.approx(expected)
    jnorm = report["weighted_j_component_parameter_norm_summary"]
    assert jnorm["logsigma"]["median"] == pytest.approx(2 * jnorm["shape"]["median"])
    assert jnorm["centroid"]["median"] == pytest.approx(jnorm["shape"]["median"])
    assert report["all_batch_measurements"][0]["cancellation_ratio"]["img1"] < 1
    assert source["weights"]["gram"] == 1.0
    assert report["status"] == "derived_pending_actual_autograd_reprobe"
    assert "actual_autograd_fit_ratio_summary" not in report


@pytest.mark.parametrize("failure", ["jt_only", "no_bf", "bf_j", "stage_one",
                                     "head", "authors", "coarse"])
def test_invalid_routes_and_unrepresentative_probes_fail(failure):
    probe, config = probe_fixture(), configuration()
    row = probe["measurements"][0]
    if failure == "jt_only":
        row["pool"] = "jt_only"
    elif failure == "no_bf":
        row["sample_has_bf"] = [False]
    elif failure == "bf_j":
        row = probe["measurements"][9]
        for group in row["gradients"].values():
            group["norms"]["raw_j1"] = 1.0
    elif failure == "stage_one":
        row["gradients"]["all_trainable"]["norms"]["raw_img_smooth"] += 1
    elif failure == "head":
        row["gradients"]["head"] = {"norms": {"raw_img_smooth": 1.0}}
    elif failure == "authors":
        for item in probe["measurements"]:
            item["target_path"] = ["font/new/Author0BF/images_text/a.png"]
    else:
        config["stage"] = "coarse"
    with pytest.raises(ValueError):
        calibration.derive_config(probe, config)


def test_mixed_diagnostic_does_not_change_fit_coefficients():
    probe, config = probe_fixture(), configuration()
    expected = calibration.derive_config(probe, config)
    mixed = deepcopy(probe["measurements"][0])
    mixed.update({"batch": 18, "pool": "mixed"})
    data = gradient_data(vectors(True, 1000))
    mixed["gradients"] = {"refiner": data, "all_trainable": data}
    probe["measurements"].append(mixed)
    assert calibration.derive_config(probe, config) == expected


def test_zero_gradient_and_extreme_coefficients_do_not_get_training_weights():
    probe, config = probe_fixture(), configuration()
    for row in probe["measurements"]:
        for group in row["gradients"].values():
            group["norms"]["raw_img_gram"] = 1e-10
    with pytest.raises(ValueError, match="above 1e4"):
        calibration.derive_config(probe, config)
    for row in probe["measurements"]:
        for group in row["gradients"].values():
            group["norms"]["raw_bf_smooth"] = 0.0
    with pytest.raises(ValueError, match="denominator"):
        calibration.derive_config(probe, config)


def test_verify_requires_actual_combined_autograd_measurements():
    probe = probe_fixture()
    config = calibration.derive_config(probe, configuration())
    with pytest.raises(ValueError, match="real combined-loss"):
        calibration.make_report(probe, config, verified=True)


def test_remeasure_executes_autograd_on_only_refiner_without_update(monkeypatch):
    probe = probe_fixture()
    config = calibration.derive_config(probe, configuration())

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(4, 4).requires_grad_(False)
            self.bf_refiner = nn.Linear(4, 1, bias=False)

    model = Tiny()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    monkeypatch.setattr(runtime, "validate_config", lambda config, command: config)
    monkeypatch.setattr(runtime, "load_model", lambda config, device: (model, {}))
    monkeypatch.setattr(runtime, "training_mode", lambda model, stage: None)
    monkeypatch.setattr(runtime, "build_dataset", lambda config: object())
    monkeypatch.setattr(runtime, "metadata", lambda config: probe["metadata"])

    def collate(dataset, indices, device, seed):
        row = probe["measurements"][indices[0]]
        return {**row, "seed": seed}

    def call(model, batch, stage):
        return {"parameter": model.bf_refiner.weight.flatten()}

    def losses(model, output, batch, config):
        parameter = output["parameter"]
        raw = {key: (parameter * parameter.new_tensor(value)).sum()
               for key, value in vectors(batch["sample_has_jt"][0]).items()}
        groups = calibration.components(config)
        for name, coefficients in groups.items():
            raw[name] = sum(raw[key] * value for key, value in coefficients.items())
        raw["raw_j1"] = raw["j1"]
        raw["raw_head"] = parameter.new_zeros(())
        raw["total"] = (raw["img1"] + config["weights"]["lambda_bf"]
                        * raw["bf_local"] + config["weights"]["lambda_j1"] * raw["j1"])
        return raw

    monkeypatch.setattr(runtime, "collate_indices", collate)
    monkeypatch.setattr(runtime, "call_model", call)
    monkeypatch.setattr(runtime, "loss_for", losses)
    measured = calibration.remeasure(probe, config, "cpu")
    report = calibration.make_report(measured, config, verified=True)
    assert measured["refiner_trainable_parameter_count"] == 4
    assert measured["stage_one_requires_grad"] is False
    assert report["actual_autograd_fit_ratio_summary"]["bf_over_img"]["median"] == (
        pytest.approx(0.5, rel=1e-6)
    )
    for item in report["composite_vs_gram_relative_error_summary"].values():
        assert item["max"] < 1e-6
    assert report["weighted_total_vs_gram_relative_error_summary"]["max"] < 1e-6
    assert all(torch.equal(before[name], value)
               for name, value in model.state_dict().items())
    assert all(value.grad is None for value in model.parameters())
    # Measured AMP/backward discrepancies must be reported without re-tuning.
    for row in measured["measurements"]:
        row["gradients"]["refiner"]["norms"]["weighted_bf_local"] *= 2
    discrepant = calibration.make_report(measured, config, verified=True)
    ratio = discrepant["actual_autograd_fit_ratio_summary"]["bf_over_img"]["median"]
    assert ratio == pytest.approx(1.0, rel=1e-6)
    assert discrepant["weights"] == report["weights"]
    assert discrepant["status"] == "reprobed_requires_review_of_actual_ratios"


def test_artifact_write_never_replaces_existing_record(tmp_path):
    path = tmp_path / "record.json"
    calibration.dump_new(path, {"first": True})
    with pytest.raises(FileExistsError):
        calibration.dump_new(path, {"second": True})
