"""Complete Run B tip contracts, including both image branches."""

import pytest
import torch
from test_dual_model import instance_targets, sample_inputs, small_model
from torch import nn

from util.tip_loss import EPSILON, masked_tip_loss, tip_weight_at_epoch


@pytest.mark.parametrize('active', [[False, False], [True, True], [True, False]])
def test_both_branches_replace_edge_and_preserve_full_bf_gradients(active):
    model = small_model(True)
    model.enable_tip(2.0)
    model.style_weight = 14.73
    nn.init.normal_(model.dual_fusion.projection.weight, std=0.02)
    with torch.no_grad():
        model.decoder_pred[-1].bias.fill_(0.1)
    inputs = sample_inputs()
    inputs[1][:, :, :8] += 10  # Keep the tiny upper half above legacy ignore.
    labels = instance_targets(active)
    labels['has_bf'] = torch.tensor([True, False]) if all(active) else ~labels['has_jt']
    labels['has_bf'] |= ~labels['has_jt']

    def forbidden(value):
        raise AssertionError('Tip mode must not execute the old edge detector')

    model.improved_edge_detection = forbidden
    result = model(*inputs, epoch=18, no_gan=True, dual_targets=labels)
    extra = result[-1]
    assert result[3].item() == 0
    assert extra['tip_pen'].requires_grad and extra['tip_full'].requires_grad
    assert extra['tip_full_pools'][:, 1].sum() > 0
    expected = (extra['L_pen'] + model.dual_lambda_full * extra['L_full']
                + model.jieti_loss_weight * extra['J']
                + model.dual_head_weight * extra['L_H'])
    torch.testing.assert_close(expected, result[0], rtol=0, atol=0)
    groups = [model.patch_embed.proj.weight,
              model.decoder_embed.weight, model.dual_fusion.projection.weight]
    gradients = torch.autograd.grad(extra['tip_full'], groups, retain_graph=True)
    assert all(value.abs().sum() > 0 for value in gradients)
    head_grad = torch.autograd.grad(
        extra['tip_full'], list(model.dual_head.parameters()),
        allow_unused=True, retain_graph=True,
    )
    assert any(g is not None and g.abs().sum() > 0 for g in head_grad) == any(active)
    result[0].backward()
    assert all(p.grad is not None for p in model.dual_head.parameters())


def test_zero_warmup_matches_original_runb_loss_and_gradients():
    original, tip = small_model(True), small_model(True)
    tip.enable_tip(2.0)
    inputs = sample_inputs()
    labels = instance_targets([True, False])
    labels['has_bf'] = torch.tensor([False, True])
    old = original(*inputs, epoch=0, no_gan=True, dual_targets=labels)
    new = tip(*inputs, epoch=0, no_gan=True, dual_targets=labels)
    torch.testing.assert_close(new[0], old[0], rtol=0, atol=0)
    for key in ('L_pen', 'L_full'):
        torch.testing.assert_close(new[-1][key], old[-1][key], rtol=0, atol=0)
    old[0].backward()
    new[0].backward()
    for (name, left), (_, right) in zip(original.named_parameters(),
                                       tip.named_parameters()):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0,
                                       msg=name)


def test_tip_checkpoint_restores_both_view_loss_and_schedule():
    from types import SimpleNamespace

    from util.checkpoint import build_checkpoint_model

    model = small_model(True)
    model.enable_tip(3.5)
    args = dict(model='tiny', dual_view=True, tip_weight=3.5, edge_warmup_epochs=10,
                loss_warmup_duration=8, style_weight=14.73,
                dual_lambda_full=0.5, jieti_w=1.0, dual_head_weight=1.0,
                jieti_w_centroid=1.0, jieti_w_logsigma=1.0, jieti_w_shape=1.0,
                jieti_pool=8)
    checkpoint = {'model': model.state_dict(), 'args': args, 'dual_view': True}
    loaded = build_checkpoint_model(
        checkpoint, model_module=SimpleNamespace(
            tiny=lambda dual_view: small_model(True),
        ),
    )
    assert loaded.tip_weight == 3.5
    assert loaded.get_tip_weight(10) == 0
    assert loaded.get_tip_weight(18) == 3.5
    assert loaded.edge_weight_final == 0


def normalized(gray):
    """Expand known grayscale values to normalized RGB without quantization."""
    rgb = gray[:, None].expand(-1, 3, -1, -1)
    mean = rgb.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = rgb.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    return (rgb - mean) / std


def test_hand_calculated_pixel_and_signed_gradient_formula():
    # Each 1x2 character: target rises 0->1, prediction falls 1->0.
    # Pixel error=1, signed dx error=2, no vertical pairs; no renormalization.
    target = normalized(torch.tensor([[[0., 1.], [0., 1.]]]))
    prediction = normalized(torch.tensor([[[1., 0.], [1., 0.]]]))
    loss, stats = masked_tip_loss(prediction, target, torch.ones_like(target))
    expected = 0.5 * 2 / (2 + EPSILON) + 0.25 * 2 / (1 + EPSILON)
    assert loss.item() == pytest.approx(expected, abs=2e-7)
    assert stats["dy"].eq(0).all()
    assert stats["pairs_x"].eq(1).all()


def test_equal_word_mean_keeps_empty_words_and_unequal_areas():
    target = normalized(torch.zeros(2, 2, 2))
    prediction = normalized(torch.tensor([
        [[0.2, 0.2], [0.8, 0.8]], [[0.4, 0.4], [0.6, 0.6]],
    ]))
    domain = torch.zeros_like(target)
    domain[0, :, 0, 0] = 1
    domain[0, :, 1] = 1
    domain[1, :, 0, 0] = 1
    loss, stats = masked_tip_loss(prediction, target, domain)
    expected = 0.5 * (0.2 / (1 + EPSILON) + 1.6 / (2 + EPSILON)
                      + 0.4 / (1 + EPSILON)) / 4
    assert loss.item() == pytest.approx(expected, abs=2e-7)
    assert stats["valid_words"].sum() == 3
    loss_single, _ = masked_tip_loss(prediction[:1], target[:1], domain[:1])
    loss_other, _ = masked_tip_loss(prediction[1:], target[1:], domain[1:])
    torch.testing.assert_close(loss, (loss_single + loss_other) / 2)


def test_differences_do_not_cross_word_seam_or_visible_invalid_pixels():
    target = normalized(torch.zeros(1, 4, 3))
    gray = torch.zeros(1, 4, 3)
    gray[:, 2:] = 0.8  # The inter-character step must not create a dy penalty.
    prediction = normalized(gray).requires_grad_()
    domain = torch.ones_like(target)
    domain[:, :, :, 2] = 0
    loss, stats = masked_tip_loss(prediction, target, domain)
    assert stats["dy"].eq(0).all()
    assert stats["pairs_y"].eq(2).all()
    assert stats["pairs_x"].eq(2).all()
    loss.backward()
    assert prediction.grad[:, :, :, 2].eq(0).all()
    assert prediction.grad[:, :, 2:, :2].abs().sum() > 0
    # Any invalid RGB channel excludes this grayscale pixel and both pairs.
    domain[:, 0, 0, 0] = 0
    _, stats = masked_tip_loss(prediction, target, domain)
    assert stats["pixels"][0, 0] == 3
    assert stats["pairs_x"][0, 0] == 1
    assert stats["pairs_y"][0, 0] == 1


def test_unclamped_out_of_range_prediction_has_corrective_gradient():
    target = normalized(torch.ones(1, 2, 1))
    prediction = normalized(torch.full((1, 2, 1), 2.)).requires_grad_()
    loss, _ = masked_tip_loss(prediction, target, torch.ones_like(target))
    loss.backward()
    assert loss.item() == pytest.approx(0.5, abs=1e-6)
    assert (prediction.grad > 0).all()


def test_empty_domain_stays_differentiable_and_bf16_math_is_fp32():
    target = normalized(torch.rand(2, 8, 4)).bfloat16()
    prediction = normalized(torch.rand(2, 8, 4)).bfloat16().requires_grad_()
    domain = torch.zeros_like(target)
    loss, stats = masked_tip_loss(prediction, target, domain)
    assert loss.dtype == torch.float32 and loss == 0
    loss.backward()
    assert prediction.grad.eq(0).all()
    domain.fill_(1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual, _ = masked_tip_loss(prediction, target, domain)
    expected, _ = masked_tip_loss(prediction.float(), target.float(), domain.float())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(value.dtype == torch.float32 for value in stats.values())


@pytest.mark.parametrize("epoch, expected", [(0, 0), (10, 0), (11, .25),
                                            (14, 1), (18, 2), (50, 2)])
def test_tip_schedule_matches_replaced_edge(epoch, expected):
    assert tip_weight_at_epoch(2, epoch, 10, 8) == expected
    assert tip_weight_at_epoch(2, 10, 10, 0) == 2


def test_full_resolution_patch_mask_validity_and_backward():
    target = normalized(torch.ones(1, 896, 448))
    prediction = normalized(torch.full((1, 896, 448), .8)).requires_grad_()
    domain = torch.zeros_like(target)
    domain[:, :, :16, :16] = 1
    domain[:, :, 448:464, :16] = 1
    domain[:, :, 448, 0] = 0
    loss, stats = masked_tip_loss(prediction, target, domain)
    assert stats["pixels"].tolist() == [[256, 255]]
    assert stats["pairs_x"].tolist() == [[240, 239]]
    assert stats["pairs_y"].tolist() == [[240, 239]]
    loss.backward()
    assert prediction.grad[:, :, 16:448].eq(0).all()
    assert prediction.grad[:, :, 448, 0].eq(0).all()
    assert prediction.grad[:, :, :16, :16].abs().sum() > 0
