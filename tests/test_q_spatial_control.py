"""Identity, physical response, normalization isolation and metric contracts."""

import math

import numpy as np
import pytest
import torch

from tools.evaluate_q_spatial_probe import response_metrics, sigma
from util.dual_view import DualViewFusion, DualViewHead
from util.q_layout_warp import (
    compose_g,
    intervention,
    moments,
    predicted_supports,
    scale_support,
    slot_validity,
    support_diagnostics,
    warp_features,
)


def gaussian(size=28):
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    return torch.exp(-((xx - 13.5).square() / 12 + (yy - 13.5).square() / 20))[None]


@pytest.mark.parametrize("ratio", [0.75, 0.90, 1.0, 1.10, 1.25, 1.50])
@pytest.mark.parametrize("axis", [0, 1])
def test_support_scales_in_correct_physical_direction(ratio, axis):
    support = gaussian()
    moved = scale_support(support, axis, ratio)
    _, a = moments(support)
    _, b = moments(moved)
    actual = float((b / a)[0, axis])
    assert abs(actual - ratio) < 0.04
    assert abs(float((b / a)[0, 1 - axis]) - 1) < 1e-5


def head_example():
    torch.manual_seed(3)
    head = DualViewHead(encoder_width=8, width=16, grid_size=7, heads=4)
    quadrants = [torch.randn(1, 7, 7, 8) for _ in range(4)]
    lower = torch.randn(1, 7, 7, 32)
    result = head(quadrants, lower)
    fusion = DualViewFusion(encoder_width=8, width=16)
    torch.nn.init.normal_(fusion.projection.weight, std=0.03)
    return result, lower, fusion


@pytest.mark.parametrize("group", ["A1", "A2", "A3", "A4", "A5", "AO"])
def test_identity_is_bitwise_and_intervention_keeps_upstream(group):
    head, lower, fusion = head_example()
    snapshot = {k: v.clone() for k, v in head.items()}
    support = predicted_supports(head)[:, 0]
    features, residual, _, _ = intervention(
        fusion, head, lower, support, 0, 0, 1.0, group
    )
    torch.testing.assert_close(features, lower, rtol=0, atol=0)
    torch.testing.assert_close(residual, fusion(head["layout"]), rtol=0, atol=0)
    for key in head:
        torch.testing.assert_close(snapshot[key], head[key], rtol=0, atol=0)


@pytest.mark.parametrize("group", ["A1", "A2", "A3", "A4", "A5", "AO"])
def test_changed_path_is_nonzero_and_complement_is_fixed(group):
    head, lower, fusion = head_example()
    normalized = fusion.norm(head["layout"])
    support = predicted_supports(head)[:, 0]
    features, residual, g_slice, q_slice = intervention(
        fusion, head, lower, support, 0, 0, 1.5, group
    )
    if group == "A2":
        assert (g_slice - normalized[..., :36]).norm() > 0
    else:
        torch.testing.assert_close(g_slice, normalized[..., :36], rtol=0, atol=0)
    if group in ("A2", "A3"):
        torch.testing.assert_close(q_slice, normalized[..., 36:], rtol=0, atol=0)
    else:
        assert (q_slice - normalized[..., 36:]).norm() > 0
    if group in ("A3", "A4"):
        assert (features - lower).norm() > 0
    else:
        torch.testing.assert_close(features, lower, rtol=0, atol=0)
    if group == "A3":
        torch.testing.assert_close(residual, fusion(head["layout"]), rtol=0, atol=0)


def test_slot_filter_is_prediction_only_and_composition_exact():
    head, _, _ = head_example()
    support = predicted_supports(head)
    head["existence_logits"][:] = -20
    valid, _, _ = slot_validity(head, support)
    assert not valid.any()
    torch.testing.assert_close(
        compose_g(head, head["occupancy_logits"].sigmoid())[..., 36:],
        head["query"].reshape(1, 7, 7, 16),
        rtol=0,
        atol=0,
    )


def test_warp_is_spatial_and_zero_support_is_identity():
    features = torch.randn(1, 28, 28, 4)
    torch.testing.assert_close(
        warp_features(features, torch.zeros(1, 28, 28), 0, 1.5),
        features,
        rtol=0,
        atol=0,
    )
    assert (warp_features(features, gaussian(), 0, 1.5) - features).norm() > 0
    support = gaussian()
    _, measured = support_diagnostics(support, torch.zeros_like(support), 0, 1.5)
    assert measured["support_ratio"] > 1.45


def test_metric_uses_fixed_regions_and_missing_mass_is_not_zero_gain():
    yy, xx = np.indices((224, 224))
    before = np.exp(-((xx - 112) ** 2 + (yy - 112) ** 2) / 200)
    after = np.exp(-((xx - 112) ** 2 / 1.5**2 + (yy - 112) ** 2) / 200)
    values = response_metrics(after, before, np.ones_like(before, bool), before, 0)
    assert math.isclose(values["soft_pixel_ratio"], 1.5, rel_tol=1e-5)
    assert sigma(np.zeros_like(before)) is None
