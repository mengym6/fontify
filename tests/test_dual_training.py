"""Integration contracts for fixed evaluation and engine batch adapters."""

import json
from types import SimpleNamespace

import pytest
import torch

from util.dual_loss import LowerJietiLoss
from util.dual_training import (
    complete_accumulation_steps,
    evaluate_dual,
    unpack_batch,
)


class PredictionModel(torch.nn.Module):
    patch_size = 2

    def predict_full(self, image, target):
        return image

    def unpatchify(self, values):
        values = values.reshape(-1, 8, 4, 2, 2, 3)
        return values.permute(0, 5, 1, 3, 2, 4).reshape(-1, 3, 16, 8)


def test_batch_one_and_two_preserve_the_same_full_update_schedule():
    steps_two = complete_accumulation_steps(136, 8)
    steps_one = complete_accumulation_steps(273, 16)
    assert steps_two // 8 == steps_one // 16 == 17
    assert steps_two * 2 * 4 == steps_one * 1 * 4 == 1088
    assert [step / steps_two for step in range(0, steps_two, 8)] == [
        step / steps_one for step in range(0, steps_one, 16)
    ]
    with pytest.raises(ValueError, match='complete accumulation'):
        complete_accumulation_steps(7, 8)


def fixed_batch():
    torch.manual_seed(81)
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    target = (torch.rand(3, 3, 16, 8) - mean) / std
    prediction = target.clone()
    prediction[0, :, 8:, :3] += 0.5
    prediction[1, :, 8:, 3:] -= 0.7
    prediction[2, :, 8:] += 0.2
    mask = torch.zeros(3, 8, 4)
    mask[:, 4:] = 1
    voro = torch.ones(3, 8, 8, dtype=torch.long)
    voro[:, :, 4:] = 2
    return {
        "image": prediction, "target": target, "mask": mask,
        "valid": torch.ones_like(target),
        "has_jt": torch.tensor([True, True, False]),
        "voro": voro,
        "valid_parts": torch.tensor([[True, True, False, False]] * 3),
    }


def test_fixed_val_is_invariant_to_batch_composition_and_uses_eval_coef(tmp_path):
    coef = {"centroid": 2.0, "logsigma": 3.0, "shape": 4.0}
    path = tmp_path / "eval.json"
    path.write_text(json.dumps({"coef": coef}))
    args = SimpleNamespace(
        dual_eval_calibration_json=str(path), jieti_pool=8, jieti_k_max=4,
        jieti_soft_fg="linear", jieti_pred_mass_ratio=0.1,
        vis_every_epoch=False,
    )
    batch = fixed_batch()
    model = PredictionModel()
    grouped = evaluate_dual([batch], model, "cpu", 0, args)
    split = [{key: value[index:index + 1] for key, value in batch.items()}
             for index in range(3)]
    singles = evaluate_dual(split, model, "cpu", 0, args)
    assert grouped == pytest.approx(singles)
    assert grouped["n_jt"] == 2 and grouped["n_bf"] == 1
    jmod = LowerJietiLoss(
        pool=8, w_centroid=coef["centroid"],
        w_logsigma=coef["logsigma"], w_shape=coef["shape"],
    )
    expected = torch.stack([
        jmod(item["image"], item["target"], item["voro"],
             item["valid_parts"], item["has_jt"])[0]
        for item in split[:2]
    ]).mean().item()
    assert grouped["jieti_J"] == pytest.approx(expected)


def test_unpack_keeps_dual_labels_out_of_image_inputs():
    batch = fixed_batch()
    batch.update(
        instance_occupancy=torch.zeros(3, 4, 4, 4),
        instance_valid=torch.zeros(3, 4, dtype=torch.bool),
        instance_classes=torch.full((3, 4), -1),
        sample_key=["a", "b", "c"],
    )
    image, target, mask, valid, kwargs = unpack_batch(batch, "cpu")
    assert image is batch["image"] and target is batch["target"]
    assert mask is batch["mask"] and valid is batch["valid"]
    assert set(kwargs) == {"dual_targets"}
    assert "sample_key" not in kwargs["dual_targets"]
    legacy = (image, target, mask, valid)
    assert unpack_batch(legacy, "cpu")[-1] == {}


def test_validation_rejects_nonfinite_predictions(tmp_path):
    path = tmp_path / 'eval.json'
    path.write_text(json.dumps({'coef': dict.fromkeys(
        ('centroid', 'logsigma', 'shape'), 1.0,
    )}))
    args = SimpleNamespace(
        dual_eval_calibration_json=str(path), jieti_pool=8, jieti_k_max=4,
        jieti_soft_fg='linear', jieti_pred_mass_ratio=0.1,
        vis_every_epoch=False,
    )
    batch = fixed_batch()
    batch['image'].fill_(float('nan'))
    with pytest.raises(FloatingPointError, match='nonfinite'):
        evaluate_dual([batch], PredictionModel(), 'cpu', 0, args)
