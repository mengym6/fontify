"""Stage-5 joint tuning preserves the Q network and missing-label routes."""

import pytest
import torch
from torch import nn

from util.g_control import (
    SLOT_PREFIXES,
    configure_stage,
    enable_g_control,
    protected_prefix_exclusions,
    route_control_head,
)
from util.g_scale import enable_g_scale


def joint_model():
    from test_dual_model import small_model

    model = small_model(dual=True).eval()
    enable_g_control(model)
    nn.init.normal_(model.g_control.flow[-1].weight, std=0.02)
    enable_g_scale(model)
    configure_stage(model, 5)
    model.dual_jieti_loss.soft_fg = "sigmoid"
    model.dual_jieti_loss.pred_mass_ratio = 0
    return model


def test_joint_scope_excludes_vit_q_and_pixel_projection():
    model = joint_model()
    for name, parameter in model.named_parameters():
        allowed = name.startswith(protected_prefix_exclusions(5))
        assert parameter.requires_grad == allowed, name
    assert all(not p.requires_grad for p in model.decoder_embed.parameters())
    assert all(not p.requires_grad for p in model.dual_head.blocks.parameters())
    assert all(not p.requires_grad for p in model.dual_fusion.parameters())
    assert any(p.requires_grad for p in model.dual_head.slot_attention.parameters())


@pytest.mark.parametrize("active", [[False, False], [True, True], [True, False]])
def test_new_control_paths_obey_jt_head_gradient_route(active):
    from test_dual_model import instance_targets, sample_inputs

    model = joint_model()
    result = model(*sample_inputs(), no_gan=True, dual_targets=instance_targets(active))
    head_parameters = [p for p in model.dual_head.parameters() if p.requires_grad]
    gradients = torch.autograd.grad(
        result[8]["L_full"], head_parameters, retain_graph=True, allow_unused=True
    )
    head_norm = sum(float(g.abs().sum()) for g in gradients if g is not None)
    assert (head_norm > 0) == any(active)
    result[0].backward()
    assert all(p.grad is None for p in model.decoder_embed.parameters())
    if not any(active):
        assert all(p.grad is None or not p.grad.any() for p in head_parameters)
    assert model.g_control.flow[-1].weight.grad.norm() > 0


def test_joint_update_keeps_raw_q_bitwise_identical():
    from test_dual_model import instance_targets, sample_inputs

    model = joint_model()
    images, target, mask, valid = sample_inputs()
    original_q = (
        model._dual_full_prediction(images, target)[2]["query"].detach().clone()
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    model(
        images,
        target,
        mask,
        valid,
        no_gan=True,
        dual_targets=instance_targets([True, True]),
    )[0].backward()
    optimizer.step()
    changed_q = model._dual_full_prediction(images, target)[2]["query"]
    torch.testing.assert_close(original_q, changed_q, rtol=0, atol=0)
    assert all(
        not p.requires_grad
        for n, p in model.dual_head.named_parameters()
        if not n.startswith(SLOT_PREFIXES)
    )


def test_routing_is_numerically_identity_and_preserves_new_module_gradients():
    head = {
        "layout": torch.randn(2, 3, 3, 5, requires_grad=True),
        "slots": torch.randn(2, 4, 5, requires_grad=True),
    }
    routed = route_control_head(head, torch.tensor([True, False]))
    layer = nn.Linear(5, 1)
    for name in head:
        torch.testing.assert_close(head[name], routed[name], rtol=0, atol=0)
    layer(routed["layout"]).sum().backward()
    assert head["layout"].grad[0].norm() > 0
    assert torch.count_nonzero(head["layout"].grad[1]) == 0
    assert layer.weight.grad.norm() > 0
