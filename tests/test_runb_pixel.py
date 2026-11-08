"""Pixel union precision, hidden-GT isolation and unchanged full-view inference."""

import numpy as np
import torch

from data.dual_dataset import DualDataset
from test_dual_data import RandomMask, synthetic_data
from test_dual_model import instance_targets, sample_inputs, small_model
from util.pixel_mask import expand_mask


def test_partial_patch_has_no_hidden_gt_value_or_gradient():
    model = small_model(dual=True).eval()
    image, target, _, _ = sample_inputs()
    pixels = torch.zeros(target.shape[0], *target.shape[-2:], dtype=torch.bool)
    pixels[:, 1::2, 1::2] = True
    changed = target.clone()
    changed.masked_fill_(pixels[:, None], 123)
    before = model.forward_encoder(image, target, pixels.flatten(1))
    after = model.forward_encoder(image, changed, pixels.flatten(1))
    for left, right in zip(before, after):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    target.requires_grad_(True)
    sum(tap.sum() for tap in model.forward_encoder(image, target, pixels)).backward()
    assert torch.count_nonzero(target.grad[pixels[:, None].expand_as(target)]) == 0
    assert torch.count_nonzero(target.grad[~pixels[:, None].expand_as(target)]) > 0


def test_pixel_and_patch_masks_are_identical_when_patch_aligned():
    model = small_model(dual=True).eval()
    image, target, patch, _ = sample_inputs()
    pixels = expand_mask(patch, target, model.patch_size)[:, 0]
    before = model.forward_encoder(image, target, patch.flatten(1))
    after = model.forward_encoder(image, target, pixels.flatten(1))
    for left, right in zip(before, after):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_pixel_reconstruction_preserves_fully_hidden_samples():
    model = small_model(dual=True)
    model.style_weight = 0
    model.get_dynamic_loss_weights = lambda epoch: (0, 0)
    image, target, _, valid = sample_inputs()
    prediction = target.clone().requires_grad_(True)
    prediction.data[:, :, 1, 1] += 1
    pixels = torch.ones(target.shape[0], *target.shape[-2:], dtype=torch.bool)
    output = model.forward_loss(image, prediction, target, pixels, valid, no_gan=True)
    assert output[1] > 0
    output[0].backward()
    assert prediction.grad[:, :, 1, 1].abs().sum() > 0
    assert torch.all(valid == 1)


def test_dual_pixel_backward_and_full_inference_contract():
    model = small_model(dual=True)
    image, target, patch, valid = sample_inputs()
    labels = instance_targets([True, False])
    pixels = expand_mask(patch, target, model.patch_size)[:, 0].clone()
    pixels[:, 1, 1] = 1
    output = model(image, target, pixels, valid, no_gan=True, dual_targets=labels)
    assert torch.isfinite(output[0])
    output[0].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert output[-1]['pen_mask'].numel() == pixels.numel()
    with torch.no_grad():
        baseline = model.predict_full(image, target)
        altered = target.clone()
        altered[:, :, target.shape[-2] // 2:] = 123
        torch.testing.assert_close(model.predict_full(image, altered), baseline, rtol=0, atol=0)


def test_dataset_pixel_union_and_validation_stays_patch(synthetic_data):
    root, _, _, train, val, fixed, cache = synthetic_data
    settings = dict(root=root, json_path_list=[train], target_cache_dir=cache,
                    masked_position_generator=RandomMask(), pixel_masking=True)
    dataset = DualDataset(**settings)
    dataset._bf_mask = lambda pair: np.zeros((448, 448), dtype=np.uint8)
    idx = next(i for i, row in enumerate(dataset.pairs) if row['has_bf'])
    sample = dataset.get_pair(idx, ref_index=idx, augmentation_seed=0)
    assert sample['mask'].shape == (896, 448)
    rgb = sample['target'] * torch.tensor((0.229, 0.224, 0.225))[:, None, None]
    rgb += torch.tensor((0.485, 0.456, 0.406))[:, None, None]
    expected = rgb.mean(0) >= 230 / 255
    torch.testing.assert_close(sample['mask'].bool(), expected, rtol=0, atol=0)
    settings.update(json_path_list=[val], is_train=False, fixed_pair_path=fixed)
    assert DualDataset(**settings)[0]['mask'].shape == (56, 28)
