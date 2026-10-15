"""Explicit sigma gradients, initialization and missing-label contracts."""

import types

import torch
from torch import nn

from util.dual_loss import LowerJietiLoss
from util.dual_view import DualViewHead
from util.g_control import GSpatialControl, configure_stage
from util.g_scale import (
    InstanceScaleAdapter,
    affine_scale,
    enable_g_scale,
    map_geometry,
    scale_supervision,
)


def make_head():
    torch.manual_seed(71)
    head = DualViewHead(encoder_width=8, width=16, grid_size=7, heads=4)
    return head([torch.randn(2, 7, 7, 8) for _ in range(4)], torch.randn(2, 7, 7, 32))


def test_affine_identity_is_exact_without_killing_scale_gradient():
    torch.manual_seed(3)
    fields = torch.rand(2, 4, 7, 7)
    factors = torch.ones(2, 4, 2, requires_grad=True)
    warped = affine_scale(fields, factors)
    torch.testing.assert_close(fields, warped, rtol=0, atol=0)
    warped.square().sum().backward()
    assert torch.isfinite(factors.grad).all() and factors.grad.norm() > 0


def test_scale_adapter_initialization_preserves_layout_and_original_q():
    head = make_head()
    module = InstanceScaleAdapter(width=16)
    output = module(head)
    torch.testing.assert_close(output["layout"], head["layout"], rtol=0, atol=0)
    with torch.no_grad():
        module.predictor[-1].bias[:] = torch.tensor([0.4, -0.4])
    changed = module(head)
    torch.testing.assert_close(
        changed["layout"][..., 36:], head["layout"][..., 36:], rtol=0, atol=0
    )
    assert not torch.equal(changed["layout"][..., :36], head["layout"][..., :36])


def test_image_gradient_passes_through_frozen_g_control_to_scale_predictor():
    head = {k: v.detach() for k, v in make_head().items()}
    module = InstanceScaleAdapter(width=16)
    control = GSpatialControl(g_width=36, hidden=32).requires_grad_(False)
    nn.init.normal_(control.flow[-1].weight, std=0.02)
    lower = torch.randn(2, 7, 7, 32)
    prediction = lower + control(lower, module(head)["layout"])
    prediction.square().mean().backward()
    assert module.predictor[-1].weight.grad.norm() > 0
    assert all(p.grad is None for p in control.parameters())


def test_sigma_target_is_rgb_ink_and_absent_bf_is_not_zero_target():
    jmod = LowerJietiLoss(pool=8)
    model = types.SimpleNamespace(dual_jieti_loss=jmod)
    # Black foreground only in lower row; target pixel sigma is measured in
    # that GT instance's fixed Voronoi, with the metric's physical scale.
    target = torch.zeros(2, 3, 16, 8)
    targets = {
        "voro": torch.ones(2, 8, 8, dtype=torch.long),
        "valid_parts": torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]]).bool(),
        "has_jt": torch.tensor([True, False]),
    }
    matching = {
        "assignment": torch.zeros(2, 4, dtype=torch.long),
        "matched_slots": targets["valid_parts"],
    }
    prediction = torch.zeros(2, 4, 2, requires_grad=True)
    loss = scale_supervision(
        model, {"scale_log_sigma": prediction}, target, targets, matching
    )
    loss.backward()
    assert prediction.grad[0, 0].norm() > 0
    assert torch.count_nonzero(prediction.grad[0, 1:]) == 0
    assert torch.count_nonzero(prediction.grad[1]) == 0
    targets["has_jt"][:] = False
    assert (
        scale_supervision(
            model, {"scale_log_sigma": prediction}, target, targets, matching
        )
        == 0
    )


def test_only_scale_branch_and_adapter_train_in_stage_four():
    model = nn.Module()
    model.g_control = GSpatialControl()
    model.dual_head = nn.Linear(2, 2)
    model.dual_fusion = nn.Linear(2, 2)
    enable_g_scale(model)
    names = configure_stage(model, 4)
    assert names and all(n.startswith("g_scale.") for n in names)
    assert all(not p.requires_grad for p in model.g_control.parameters())


def test_scale_map_sigma_changes_along_requested_axis():
    coord = torch.arange(28) - 13.5
    base = torch.exp(-coord.square() / 18)
    field = (base[:, None] * base[None])[None, None]
    _, original = map_geometry(field)
    _, changed = map_geometry(affine_scale(field, torch.tensor([[[1.5, 1.0]]])))
    torch.testing.assert_close(
        changed[..., 0] / original[..., 0], torch.tensor([[1.5]]), rtol=0.02, atol=0.02
    )
    torch.testing.assert_close(changed[..., 1], original[..., 1])


def test_full_prediction_scale_migration_is_identity_and_has_no_query_gt_leak():
    from test_dual_model import sample_inputs, small_model

    from util.g_control import enable_g_control

    model = small_model(dual=True).eval()
    enable_g_control(model)
    nn.init.normal_(model.g_control.flow[-1].weight, std=0.02)
    configure_stage(model, 2)
    image, target, _, _ = sample_inputs(batch=1)
    expected = model.predict_full(image, target)
    enable_g_scale(model)
    configure_stage(model, 4)
    actual = model.predict_full(image, target)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    hidden_changed = target.clone()
    hidden_changed[:, :, 8:] += 100
    torch.testing.assert_close(
        actual, model.predict_full(image, hidden_changed), rtol=0, atol=0
    )
    actual.square().mean().backward()
    assert model.g_scale.predictor[-1].weight.grad.norm() > 0
    assert all(p.grad is None for p in model.dual_head.parameters())


def test_scale_checkpoint_roundtrip_requires_scale_metadata():
    import pytest
    from test_dual_eval import TinyModel

    from util.checkpoint import build_checkpoint_model
    from util.g_control import enable_g_control

    model = TinyModel(True)
    enable_g_control(model)
    enable_g_scale(model)
    checkpoint = {
        "model": model.state_dict(),
        "args": {"dual_view": True, "model": "tiny"},
        "g_control_config": model.g_control_config,
        "g_scale_config": model.g_scale_config,
        "dual_view_config": {
            "lambda_full": 0.5,
            "w_j": 0.4,
            "w_h": 0.5,
            "coef": {"centroid": 1, "logsigma": 1, "shape": 1},
            "pool": 8,
        },
    }
    restored = build_checkpoint_model(
        checkpoint, model_module=types.SimpleNamespace(tiny=TinyModel)
    )
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, checkpoint["model"][name], rtol=0, atol=0)
    del checkpoint["g_scale_config"]
    with pytest.raises(ValueError, match="metadata"):
        build_checkpoint_model(
            checkpoint, model_module=types.SimpleNamespace(tiny=TinyModel)
        )
