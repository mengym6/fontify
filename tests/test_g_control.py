"""G-only identity, gradient, routing and stage-freeze contracts."""

import pytest
import torch
from torch import nn

from util.g_control import GSpatialControl, configure_stage, enable_g_control


def test_zero_flow_preserves_features_exactly_and_has_image_gradient():
    torch.manual_seed(4)
    control = GSpatialControl(g_width=12, hidden=32)
    lower = torch.randn(2, 7, 7, 16)
    layout = torch.randn(2, 7, 7, 20)
    correction = control(lower, layout)
    torch.testing.assert_close(correction, torch.zeros_like(correction), rtol=0, atol=0)
    (correction * torch.randn_like(correction)).sum().backward()
    assert control.flow[-1].weight.grad.norm() > 0
    assert control.flow[-1].bias.grad.norm() > 0


def test_q_changes_cannot_change_g_spatial_output():
    control = GSpatialControl(g_width=12, hidden=32)
    nn.init.normal_(control.flow[-1].weight, std=0.03)
    lower = torch.randn(1, 7, 7, 16)
    first = torch.randn(1, 7, 7, 20)
    second = first.clone()
    second[..., 12:] += torch.randn_like(second[..., 12:]) * 100
    torch.testing.assert_close(
        control(lower, first), control(lower, second), rtol=0, atol=0
    )
    second.requires_grad_(True)
    control(lower, second).square().sum().backward()
    assert second.grad[..., :12].norm() > 0
    assert torch.count_nonzero(second.grad[..., 12:]) == 0


@pytest.mark.parametrize("stage,phase", [(2, 0), (3, 0), (3, 1), (3, 2)])
def test_stage_freeze_policy(stage, phase):
    model = nn.Module()
    model.dual_view = True
    model.blocks = nn.Linear(4, 4)
    model.dual_head = nn.Linear(4, 4)
    model.dual_fusion = nn.Linear(4, 4)
    model.decoder_embed = nn.Linear(4, 4)
    model.decoder_pred = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 3))
    enable_g_control(model)
    names = configure_stage(model, stage, phase)
    assert names and all(
        not n.startswith(("blocks.", "dual_head.", "dual_fusion.")) for n in names
    )
    assert any(n.startswith("decoder_pred.") for n in names) == (stage == 3)
    assert any(n.startswith("decoder_embed.") for n in names) == (
        stage == 3 and phase == 2
    )
    assert all(not p.requires_grad for p in model.dual_fusion.parameters())


def test_unknown_or_duplicate_control_is_rejected():
    model = nn.Module()
    model.dual_view = True
    with pytest.raises(ValueError):
        enable_g_control(model, {"schema": "unversioned"})
    enable_g_control(model)
    with pytest.raises(ValueError):
        enable_g_control(model)


def test_actual_fontify_zero_control_is_identity_and_frozen_q_is_stable():
    from test_dual_model import sample_inputs, small_model

    model = small_model(dual=True).eval()
    images, target, _, _ = sample_inputs(batch=1)
    with torch.no_grad():
        before = model.predict_full(images, target)
    enable_g_control(model)
    configure_stage(model, 2)
    original_q = {
        n: p.detach().clone() for n, p in model.dual_fusion.named_parameters()
    }
    optimizer = torch.optim.AdamW(model.g_control.parameters(), lr=1e-3)
    first = model.predict_full(images, target)
    torch.testing.assert_close(first, before, rtol=0, atol=0)
    first.square().mean().backward()
    assert model.g_control.flow[-1].weight.grad.norm() > 0
    optimizer.step()
    for name, parameter in model.dual_fusion.named_parameters():
        torch.testing.assert_close(parameter, original_q[name], rtol=0, atol=0)
        assert parameter.grad is None
    assert all(p.grad is None for p in model.dual_head.parameters())
    assert all(p.grad is None for p in model.decoder_pred.parameters())


def test_control_checkpoint_roundtrip_is_strict_and_requires_metadata():
    import types

    from test_dual_eval import TinyModel

    from util.checkpoint import build_checkpoint_model

    model = TinyModel(True)
    enable_g_control(model)
    checkpoint = {
        "model": model.state_dict(),
        "args": {"dual_view": True, "model": "tiny"},
        "g_control_config": model.g_control_config,
        "dual_view_config": {
            "lambda_full": 0.5,
            "w_j": 0.4,
            "w_h": 0.5,
            "coef": {"centroid": 1, "logsigma": 1, "shape": 1},
            "pool": 8,
        },
    }
    rebuilt = build_checkpoint_model(
        checkpoint, model_module=types.SimpleNamespace(tiny=TinyModel)
    )
    for key, value in rebuilt.state_dict().items():
        torch.testing.assert_close(value, checkpoint["model"][key], rtol=0, atol=0)
    del checkpoint["g_control_config"]
    with pytest.raises(ValueError, match="metadata"):
        build_checkpoint_model(
            checkpoint, model_module=types.SimpleNamespace(tiny=TinyModel)
        )
