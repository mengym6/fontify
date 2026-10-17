"""Contracts for independent runtime-check measurements and failure handling."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tools.runc_runtime_checks import (
    compare_predictions,
    gradient_check,
    gradient_measurement,
    state_digest,
    tensor_digest,
)


def test_exact_state_digest_detects_frozen_buffer_and_parameter_changes():
    model = nn.Module()
    model.stage_one = nn.Linear(2, 2)
    model.register_buffer("persistent", torch.ones(2, dtype=torch.bfloat16))
    model.bf_refiner = nn.Linear(2, 2)
    before = state_digest(model, exclude=("bf_refiner.",))
    with torch.no_grad():
        model.bf_refiner.weight.add_(1)
    assert state_digest(model, exclude=("bf_refiner.",)) == before
    model.persistent[0] = 2
    assert state_digest(model, exclude=("bf_refiner.",)) != before
    assert tensor_digest(torch.tensor([1.0])) != tensor_digest(
        torch.tensor([1.0], dtype=torch.bfloat16)
    )


def test_gradient_measurement_observes_unused_and_nonfinite_real_parameters():
    model = nn.Module()
    model.dual_head = nn.Linear(2, 1, bias=False)
    model.structure_adapter = nn.Linear(2, 1, bias=False)
    value = model.structure_adapter(torch.ones(1, 2)).sum()
    measured = gradient_measurement(value, model)
    assert measured["group_norm"]["head"] == 0
    assert measured["group_norm"]["conditioner"] == pytest.approx(2**0.5)
    assert measured["used_parameter_tensors"]["head"] == 0
    with pytest.raises(AssertionError, match="nonfinite gradient"):
        gradient_measurement(value * float("nan"), model)


def test_prediction_comparison_includes_head_and_each_automatic_mask():
    left = {
        "p0": torch.ones(1),
        "head": {"query": torch.ones(1)},
        "masks": [torch.ones(1, dtype=torch.bool)],
    }
    right = {
        "p0": left["p0"].clone(),
        "head": {"query": left["head"]["query"].clone()},
        "masks": [left["masks"][0].clone()],
    }
    compare_predictions(left, right)
    right["masks"][0].zero_()
    with pytest.raises(AssertionError, match="per-round"):
        compare_predictions(left, right)
    right["masks"][0].fill_(True)
    right["head"]["query"].zero_()
    with pytest.raises(AssertionError, match="head.query"):
        compare_predictions(left, right)


@pytest.mark.parametrize("stage", ["coarse", "refine"])
def test_independent_gradients_use_actual_fontify_loss_and_routes(
    stage,
    tmp_path,
    monkeypatch,
):
    import tools.runc_runtime_checks as checks
    import tools.structure_first as runtime

    spec = importlib.util.spec_from_file_location(
        "_runc_independent_fixture",
        Path(__file__).with_name("test_structure_first_model.py"),
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    model = fixture.new_model()
    fixture.set_structure_stage(model, stage)
    pairs = [(True, True), (False, True), (True, False)]

    class Dataset:
        def get_pair(self, index, augmentation_seed=None):
            image, target, mask, valid = fixture.sample_inputs(batch=1)
            has_jt, has_bf = pairs[index]
            patch_mask = fixture.bf_mask(batch=1)[0]
            return {
                "image": image[0],
                "target": target[0],
                "mask": mask[0],
                "valid": valid[0],
                "has_jt": torch.tensor(has_jt),
                "has_bf": torch.tensor(has_bf),
                "instance_occupancy": torch.ones(4, 4, 4),
                "instance_valid": torch.tensor([has_jt] * 2 + [False] * 2),
                "instance_classes": torch.zeros(4, dtype=torch.long),
                "voro": torch.cat(
                    (torch.zeros(8, 4), torch.ones(8, 4)), 1
                ).long(),
                "valid_parts": torch.tensor([has_jt] * 2 + [False] * 2),
                "bf_patch_mask": patch_mask,
                "bf_pixel_mask": patch_mask.repeat_interleave(
                    2,
                    0,
                )
                .repeat_interleave(2, 1)[None]
                .float(),
            }

    cases = [("dual", [0]), ("bf_only", [1])]
    if stage == "coarse":
        cases += [("jt_only", [2]), ("mixed", [1, 2, 0])]
    else:
        cases += [("mixed", [1, 0])]
    config = {
        "stage": stage,
        "weights": dict.fromkeys(runtime.WEIGHT_KEYS, 1.0),
        "lowpass": {"kernel_size": 7, "sigma": 1.5, "boundary": "reflect"},
    }
    monkeypatch.setattr(
        checks, "common_model", lambda *args: (config, model, {})
    )
    monkeypatch.setattr(
        checks, "data_and_cases", lambda *args: (Dataset(), cases, {"all": 3})
    )
    monkeypatch.setattr(runtime, "save_visual", lambda *args: None)
    result = gradient_check(
        config,
        tmp_path,
        torch.device("cpu"),
        SimpleNamespace(amp=False),
        stage,
    )
    assert len(result["checks"]) == len(cases)
    bf_only = next(row for row in result["checks"] if row["pool"] == "bf_only")
    assert bf_only["gradients"]["total"]["group_norm"]["head"] == 0


def test_runner_snapshot_roundtrip_accepts_original_args_only_metadata(
    tmp_path,
):
    from tools.structure_first import snapshot
    from util.checkpoint import build_checkpoint_model

    spec = importlib.util.spec_from_file_location(
        "_runc_snapshot_fixture",
        Path(__file__).with_name("test_structure_first_model.py"),
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    model = fixture.new_model()
    fixture.set_structure_stage(model, "refine")
    model.eval()
    nn.init.normal_(model.bf_refiner.adapter.projection.weight, std=0.01)
    optimizer = torch.optim.AdamW(model.bf_refiner.parameters(), lr=1e-4)
    # The sealed Run B Fix stores these values only in args, without the newer
    # dual_view_config key. Production snapshot must retain that compatibility.
    saved = {
        "args": {
            "model": "small",
            "dual_view": True,
            "dual_lambda_full": 0.5,
            "dual_jieti_weight": 1.0,
            "dual_head_weight": 1.0,
            "jieti_w_centroid": 0.2,
            "jieti_w_logsigma": 0.3,
            "jieti_w_shape": 0.4,
            "jieti_pool": 8,
        }
    }
    config = {
        "j_coefficients": {
            "centroid": 1.0,
            "logsigma": 1.0,
            "shape": 1.0,
        },
        "training": {"steps": 2, "batch_size": 1, "accum_iter": 1},
    }
    snapshot(model, saved, config, {}, optimizer, 2, tmp_path)
    checkpoint = torch.load(
        tmp_path / "checkpoint-step-000002.pth", weights_only=False
    )
    restored = build_checkpoint_model(
        checkpoint,
        model_module=SimpleNamespace(
            small=lambda dual_view: fixture.small_model(dual=dual_view),
        ),
    ).eval()
    images, target, _, _ = fixture.sample_inputs(batch=1)
    with torch.no_grad():
        expected = model.predict_full(images, target)
        actual = restored.predict_full(images, target)
    assert torch.equal(expected, actual)
    assert state_digest(restored) == state_digest(model)
    assert checkpoint["dual_view_config"]["coef"] == config["j_coefficients"]
