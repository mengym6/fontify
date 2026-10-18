"""Probability-warp boundaries, geometric units and response-gain numerics."""

import math

import pytest
import torch

from tools.runc_v2_condition_audit import (
    PROBABILITY_EPS,
    TRAINING_START_FIELDS,
    Variant,
    evaluation_config,
    perturb_structure,
    response_gain,
    scalar_geometry,
    spatial_moments,
    summarize_responses,
    warp_probabilities,
)


def gaussian(width=28, sigma_x=2.0, sigma_y=2.5):
    y, x = torch.meshgrid(
        torch.arange(width), torch.arange(width), indexing="ij"
    )
    center = (width - 1) / 2
    return torch.exp(
        -0.5
        * (
            ((x - center) / sigma_x).square()
            + ((y - center) / sigma_y).square()
        )
    )[None, None]


def head(probabilities):
    return {
        "occupancy_logits": torch.logit(probabilities.clamp(1e-6, 1 - 1e-6)),
        "existence_logits": torch.ones(probabilities.shape[:2]),
        "class_logits": torch.zeros(*probabilities.shape[:2], 20),
    }


def test_zero_probability_padding_never_becomes_half_occupancy():
    p = torch.zeros(1, 1, 28, 28)
    p[:, :, 12:15, 25:28] = 1
    predicted = head(p)
    variant = Variant("shift", "shift", "x", shift_x=2)
    output = perturb_structure(predicted, variant)
    occupancy = output["occupancy_logits"].sigmoid()
    torch.testing.assert_close(
        occupancy[:, :, :, :2],
        torch.full_like(occupancy[:, :, :, :2], PROBABILITY_EPS),
        rtol=1e-5,
        atol=1e-10,
    )
    assert occupancy[:, :, :, :2].max() < 1e-4
    assert output["existence_logits"] is predicted["existence_logits"]
    assert output["class_logits"] is predicted["class_logits"]


def test_identity_warp_and_baseline_preserve_predictions():
    p = gaussian()
    torch.testing.assert_close(warp_probabilities(p), p, rtol=2e-5, atol=2e-7)
    predicted = head(p)
    baseline = perturb_structure(predicted, Variant("normal"))
    torch.testing.assert_close(
        baseline["occupancy_logits"],
        predicted["occupancy_logits"],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("shift", [2.0, -2.0])
def test_translation_is_in_28_grid_pixels_with_correct_sign(axis, shift):
    p = torch.zeros(1, 1, 28, 28)
    p[:, :, 13, 13] = 1
    changed = warp_probabilities(p, **{f"shift_{axis}": shift})
    before, after = spatial_moments(p), spatial_moments(changed)
    delta = after[f"centroid_{axis}"] - before[f"centroid_{axis}"]
    assert delta.item() == pytest.approx(shift / 28, abs=1e-7)
    assert after["mass"].item() == pytest.approx(
        before["mass"].item(), abs=1e-6
    )


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("scale", [1.25, 0.8])
def test_scale_changes_soft_sigma_about_slot_centroid(axis, scale):
    p = gaussian()
    changed = warp_probabilities(p, **{f"scale_{axis}": scale})
    before, after = spatial_moments(p), spatial_moments(changed)
    ratio = (after[f"sigma_{axis}"] / before[f"sigma_{axis}"]).item()
    assert ratio == pytest.approx(scale, rel=0.04)
    assert after[f"centroid_{axis}"].item() == pytest.approx(
        before[f"centroid_{axis}"].item(),
        abs=1e-6,
    )
    other = "y" if axis == "x" else "x"
    assert after[f"sigma_{other}"].item() == pytest.approx(
        before[f"sigma_{other}"].item(),
        rel=1e-5,
    )


def test_edge_translation_loses_mass_and_empty_moments_remain_finite():
    p = torch.zeros(1, 1, 28, 28)
    p[:, :, 13, 27] = 1
    moved = warp_probabilities(p, shift_x=2)
    assert moved.sum() == 0
    moments = spatial_moments(moved)
    assert not moments["valid"].item()
    assert all(torch.isfinite(value).all() for value in moments.values())
    assert moments["mass_fraction"].item() == 0


def test_common_coordinate_units_across_28_and_112_grids():
    p = gaussian()
    large = p.repeat_interleave(4, -2).repeat_interleave(4, -1)
    small_geo, large_geo = scalar_geometry(p), scalar_geometry(large)
    assert large_geo["mass"] == pytest.approx(16 * small_geo["mass"], rel=1e-6)
    assert large_geo["mass_fraction"] == pytest.approx(
        small_geo["mass_fraction"],
        rel=1e-6,
    )
    for key in ("centroid_x", "centroid_y"):
        assert large_geo[key] == pytest.approx(small_geo[key], abs=1e-6)
    small_shift = warp_probabilities(p, shift_x=2)
    large_shift = warp_probabilities(large, shift_x=8)
    small_delta = (
        scalar_geometry(small_shift)["centroid_x"] - (small_geo["centroid_x"])
    )
    large_delta = (
        scalar_geometry(large_shift)["centroid_x"] - (large_geo["centroid_x"])
    )
    assert small_delta == pytest.approx(large_delta, abs=1e-6)


def test_density_mean_and_logsigma_have_finite_nonzero_soft_gradients():
    p = gaussian().requires_grad_(True)
    moments = spatial_moments(p)
    values = [p.mean(), moments["logsigma_x"].mean(),
              moments["logsigma_y"].mean()]
    for index, value in enumerate(values):
        gradient = torch.autograd.grad(value, p, retain_graph=index < 2)[0]
        assert torch.isfinite(gradient).all()
        assert gradient.count_nonzero() > 0


def test_gain_uses_actual_s_delta_and_excludes_zero_denominator():
    variant = Variant("scale_x", "scale", "x", scale_x=1.25)
    base = {"logsigma_x": math.log(0.2)}
    s = {"logsigma_x": math.log(0.24)}
    d = {"logsigma_x": math.log(0.22)}
    result = response_gain(base, s, base, d, variant)
    assert result["gain"] == pytest.approx(math.log(1.1) / math.log(1.2))
    assert result["direction_matches_input"]
    assert result["input_matches_requested_direction"]
    zero = response_gain(base, base, base, d, variant)
    assert zero["gain"] is None and zero["direction_matches_input"] is None
    flipped = response_gain(base, s, d, base, variant)
    assert flipped["gain"] < 0 and not flipped["direction_matches_input"]


def test_no_s_and_summary_do_not_claim_geometric_acceptance():
    predicted = head(gaussian().expand(1, 4, -1, -1))
    absent = perturb_structure(predicted, Variant("noS", "remove"))
    occupancy = absent["occupancy_logits"].sigmoid()
    assert occupancy.max() == pytest.approx(PROBABILITY_EPS, rel=1e-5)
    no_s = response_gain({}, {}, {}, {}, Variant("noS", "remove"))
    assert no_s["gain"] is None
    rows = [
        {
            "variant": "scale",
            "gain": 2.0,
            "direction_matches_input": True,
            "input_matches_requested_direction": True,
            "density_l1_delta": 0.1,
            "density_max_delta": 0.3,
        },
        {
            "variant": "scale",
            "gain": None,
            "direction_matches_input": None,
            "input_matches_requested_direction": None,
            "density_l1_delta": 0.0,
            "density_max_delta": 0.0,
        },
        {
            "variant": "scale",
            "gain": -1.0,
            "direction_matches_input": False,
            "input_matches_requested_direction": False,
            "density_l1_delta": 0.2,
            "density_max_delta": 0.4,
        },
    ]
    summary = summarize_responses(rows)["scale"]
    assert summary["samples"] == 3
    assert (
        summary["defined_gain_count"] == summary["direction_denominator"] == 2
    )
    assert summary["median_gain"] == 0.5
    assert summary["direction_matches_input"] == 1
    assert summary["exactly_equal_output_count"] == 1


@pytest.mark.parametrize("parameters", [{"scale_x": 0}, {"shift_y": math.inf}])
def test_invalid_warp_parameters_are_rejected(parameters):
    with pytest.raises(ValueError, match="finite"):
        warp_probabilities(gaussian(), **parameters)


def test_readonly_config_drops_training_bindings_and_preserves_lineage():
    config = {
        "schema": "runc-v2-runtime-v1", "experiment": "runc_v2",
        "stage": "ib", "checkpoint": "current-ib.pth",
        "initial_checkpoint_sha256": "initial-source-sha",
        "data": {"reference_policy": "distinct-original-reference-v1"},
        "model": {"unfreeze_blocks": [2, 5, 8, 11]},
        "training": {"seed": 0, "amp": True},
        "backend": {"matmul_tf32": True},
        **{name: "training-start-only" for name in TRAINING_START_FIELDS},
    }
    clean, removed = evaluation_config(config)
    assert set(removed) == set(TRAINING_START_FIELDS)
    assert not set(TRAINING_START_FIELDS) & set(clean)
    assert clean["checkpoint"] == "current-ib.pth"
    assert clean["initial_checkpoint_sha256"] == "initial-source-sha"
    assert clean["data"] == config["data"]
    assert clean["model"] == config["model"]
    assert not clean["backend"]["matmul_tf32"]
    assert not clean["backend"]["cudnn_tf32"]
    assert not clean["training"]["amp"]
    assert config["calibration_record"] == "training-start-only"
