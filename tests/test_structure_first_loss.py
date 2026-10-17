"""Run C loss arithmetic and gradients on synthetic CPU tensors, not CUDA."""

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from util.dual_loss import DualJietiLoss, InstanceHeadLoss
from util.structure_first_loss import (
    LowpassConfig,
    StructureLossWeights,
    lowpass_target,
    masked_per_sample,
    structure_losses,
)


class LowerGram:
    def __init__(self):
        self.inputs = None

    def __call__(self, pred, target, per_sample=False):
        self.inputs = pred, target
        assert per_sample
        return (pred.mean((-1, -2)) - target.mean((-1, -2))).abs().mean(-1)


def make_inputs(has_jt=(True, False), has_bf=(True, True)):
    batch_size = len(has_jt)
    torch.manual_seed(8)
    model = SimpleNamespace(
        dual_jieti_loss=DualJietiLoss(pool=8, shape_bins=4),
        dual_head_loss=InstanceHeadLoss(),
        vgg_loss=LowerGram(),
        improved_edge_detection=lambda image: image.mean(1, keepdim=True),
    )
    p0 = torch.randn(batch_size, 3, 16, 8, requires_grad=True)
    p1 = torch.randn(batch_size, 3, 16, 8, requires_grad=True)
    target = torch.randn(batch_size, 3, 16, 8)
    head = {
        "occupancy_logits": torch.zeros(
            batch_size, 4, 4, 4, requires_grad=True
        ),
        "existence_logits": torch.zeros(batch_size, 4, requires_grad=True),
        "class_logits": torch.zeros(batch_size, 4, 20, requires_grad=True),
    }
    valid_parts = torch.zeros(batch_size, 4, dtype=torch.bool)
    valid_parts[:, 0] = torch.tensor(has_jt)
    voro = torch.ones(batch_size, 8, 8, dtype=torch.long)
    local_mask = torch.zeros(batch_size, 1, 8, 8)
    local_mask[0, :, :4] = 1
    batch = {
        "target": target,
        "valid": torch.ones_like(target),
        "has_jt": torch.tensor(has_jt),
        "has_bf": torch.tensor(has_bf),
        "instance_occupancy": torch.ones(batch_size, 4, 4, 4),
        "instance_valid": valid_parts.clone(),
        "instance_classes": torch.zeros(batch_size, 4, dtype=torch.long),
        "valid_parts": valid_parts,
        "voro": voro,
        "bf_pixel_mask": local_mask,
    }
    return model, {"p0": p0, "p1": p1, "head": head}, batch


def test_lowpass_is_detached_target_only_and_reflect_constant_preserved():
    target = torch.ones(2, 3, 8, 8, requires_grad=True)
    filtered = lowpass_target(target)
    assert not filtered.requires_grad
    assert torch.allclose(filtered, torch.ones_like(filtered), atol=1e-6)
    impulse = torch.zeros(1, 3, 8, 8)
    impulse[:, :, 4, 4] = 1
    assert 0 < lowpass_target(impulse)[0, 0, 4, 4] < 1
    assert torch.equal(lowpass_target(impulse, {"kernel_size": 1}), impulse)


@pytest.mark.parametrize("kwargs", [
    {"kernel_size": 0}, {"kernel_size": 4}, {"kernel_size": 3.0},
    {"sigma": 0}, {"sigma": float("nan")}, {"boundary": "unknown"},
])
def test_lowpass_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        LowpassConfig(**kwargs)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_weights_reject_unusable_coefficients(value):
    with pytest.raises(ValueError):
        StructureLossWeights(lambda_bf=value)


def test_mask_normalization_uses_channel_pixels_and_full_batch_denominator():
    error = torch.ones(2, 3, 4, 4)
    error[0] *= 6
    mask = torch.zeros(2, 1, 4, 4)
    mask[0, :, 0, :2] = 1
    per_sample, count = masked_per_sample(error, mask)
    assert per_sample.tolist() == [6, 0]
    assert count.tolist() == [6, 0]
    assert per_sample.mean() == 3


def test_fractional_validity_does_not_clamp_nonzero_small_denominators():
    error = torch.full((1, 1, 1, 1), 9.0)
    value, count = masked_per_sample(error, torch.full_like(error, 0.1))
    assert count.item() == pytest.approx(0.1)
    assert value.item() == pytest.approx(9)


def test_coarse_arithmetic_lower_domain_jt_gate_and_unchanged_valid():
    model, outputs, batch = make_inputs()
    original_valid = batch["valid"].clone()
    weights = StructureLossWeights(lambda_low=2, lambda_j0=3, lambda_h=4)
    losses = structure_losses(model, outputs, batch, weights, "coarse")
    expected_low = F.smooth_l1_loss(
        outputs["p0"][:, :, 8:], lowpass_target(batch["target"][:, :, 8:])
    )
    assert torch.allclose(losses["low"], expected_low)
    expected_total = (
        2 * losses["low"] + 3 * losses["j0"] + 4 * losses["head"]
    )
    assert torch.allclose(losses["total"], expected_total)
    assert losses["count_jt"] == 1 and losses["count_bf"] == 2
    losses["total"].backward()
    assert outputs["p0"].grad[:, :, :8].count_nonzero() == 0
    for tensor in outputs["head"].values():
        assert tensor.grad[1].count_nonzero() == 0
    assert outputs["head"]["existence_logits"].grad[0].abs().sum() > 0
    assert torch.equal(original_valid, batch["valid"])
    assert outputs["p1"].grad is None


def test_bf_only_coarse_has_zero_j_h_but_valid_image_gradients():
    model, outputs, batch = make_inputs(has_jt=(False, False))
    losses = structure_losses(model, outputs, batch)
    assert losses["j0"] == 0 and losses["head"] == 0
    losses["total"].backward()
    assert outputs["p0"].grad.abs().sum() > 0
    for tensor in outputs["head"].values():
        assert tensor.grad.count_nonzero() == 0


def test_jt_only_is_valid_coarse_and_j_uses_original_unfiltered_gt():
    model, outputs, batch = make_inputs(has_jt=(True,), has_bf=(False,))
    original_j = model.dual_jieti_loss
    observed = []

    def jieti(prediction, target, *args):
        observed.append(target.detach().clone())
        return original_j(prediction, target, *args)

    model.dual_jieti_loss = jieti
    losses = structure_losses(model, outputs, batch, stage="coarse")
    assert losses["count_bf"] == 0 and losses["count_jt"] == 1
    assert torch.equal(observed[0], batch["target"])
    assert not torch.equal(losses["lowpass_target"], batch["target"][:, :, 8:])
    assert losses["head"] > 0


def test_refine_full_image_local_loss_exact_mean_empty_kept_and_head_monitor():
    model, outputs, batch = make_inputs()
    weights = StructureLossWeights(
        lambda_img1=2, lambda_bf=3, lambda_j1=4, gram=5, edge=6, edge_local=7
    )
    losses = structure_losses(model, outputs, batch, weights, "refine")
    expected_img = (losses["raw_img_smooth"]
                    + 5 * losses["raw_img_gram"] + 6 * losses["raw_img_edge"])
    expected_local = losses["raw_bf_smooth"] + 7 * losses["raw_bf_edge"]
    assert torch.equal(losses["img1"], expected_img)
    assert torch.equal(losses["bf_local"], expected_local)
    expected_total = (
        2 * expected_img + 3 * expected_local + 4 * losses["j1"]
    )
    assert torch.equal(losses["total"], expected_total)
    assert losses["count_local_empty"] == 1
    assert losses["count_local_nonempty"] == 1
    assert losses["count_local_channel_pixels"] == 3 * 4 * 8
    error = F.smooth_l1_loss(outputs["p1"][0, :, 8:12],
                            batch["target"][0, :, 8:12])
    assert torch.allclose(losses["raw_bf_smooth"], error / 2)
    assert not losses["head"].requires_grad
    gram_pred, gram_target = model.vgg_loss.inputs
    assert gram_pred.shape == (2, 3, 224, 224)
    assert torch.equal(gram_target, F.interpolate(
        batch["target"][:, :, 8:], (224, 224), mode="bilinear",
        align_corners=False, antialias=True,
    ))
    losses["total"].backward()
    assert outputs["p0"].grad is None
    assert outputs["p1"].grad.abs().sum() > 0
    assert outputs["p1"].grad[:, :, :8].count_nonzero() == 0
    for tensor in outputs["head"].values():
        assert tensor.grad is None


def test_empty_local_mask_zero_has_no_image_edge_boundary():
    model, outputs, batch = make_inputs()
    seen = []

    def edge(image):
        seen.append(image.detach().clone())
        return image.mean(1, keepdim=True)

    model.improved_edge_detection = edge
    batch["bf_pixel_mask"].zero_()
    losses = structure_losses(model, outputs, batch, stage="refine")
    assert losses["bf_local"] == 0
    assert losses["count_local_empty"] == 2
    assert torch.equal(seen[0], outputs["p1"][:, :, 8:])
    assert torch.equal(seen[1], batch["target"][:, :, 8:])


def test_full_local_mask_matches_full_smooth_and_edge_without_skip():
    model, outputs, batch = make_inputs()
    batch["bf_pixel_mask"].fill_(1)
    losses = structure_losses(model, outputs, batch, stage="refine")
    assert torch.equal(losses["raw_bf_smooth"], losses["raw_img_smooth"])
    assert torch.equal(losses["raw_bf_edge"], losses["raw_img_edge"])
    assert losses["count_local_empty"] == 0
    assert losses["count_local_channel_pixels"] == 2 * 3 * 8 * 8


def test_refine_bf_only_j_gate_and_reject_jt_only():
    model, outputs, batch = make_inputs(has_jt=(False, False))
    losses = structure_losses(model, outputs, batch, stage="refine")
    assert losses["j1"] == 0 and losses["head"] == 0
    batch["has_bf"][1] = False
    batch["has_jt"][1] = True
    with pytest.raises(ValueError, match="JT-only"):
        structure_losses(model, outputs, batch, stage="refine")


def test_actual_patch_support_and_valid_intersection_without_mutation():
    model, outputs, batch = make_inputs()
    del batch["bf_pixel_mask"]
    batch["bf_patch_mask"] = torch.ones(2, 2, 2)
    batch["valid"][0, :, 8:12] = 0
    original_valid = batch["valid"].clone()
    losses = structure_losses(model, outputs, batch, stage="refine")
    assert losses["local_channel_pixels_per_sample"].tolist() == [96, 192]
    assert torch.equal(original_valid, batch["valid"])


def test_smooth_local_gradients_are_zero_outside_actual_mask():
    model, outputs, batch = make_inputs()
    losses = structure_losses(model, outputs, batch, stage="refine")
    gradient = torch.autograd.grad(losses["raw_bf_smooth"], outputs["p1"])[0]
    assert gradient[0, :, 8:12].abs().sum() > 0
    assert gradient[0, :, :8].count_nonzero() == 0
    assert gradient[0, :, 12:].count_nonzero() == 0
    assert gradient[1].count_nonzero() == 0
