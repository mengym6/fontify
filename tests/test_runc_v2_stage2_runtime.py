"""Stage-II diagnostic runner checks using actual small Fontify CPU methods."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from tools import runc_v2_runtime as runtime

SPEC = importlib.util.spec_from_file_location(
    "_runc_v2_stage2_fixture", Path(__file__).with_name("test_runc_v2_model.py")
)
FIXTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURE)


def normalize(value):
    mean = value.new_tensor(runtime.MEAN)[None, :, None, None]
    std = value.new_tensor(runtime.STD)[None, :, None, None]
    return (value - mean) / std


@pytest.fixture
def setup_smoke(monkeypatch, tmp_path):
    import data.runc_v2_dataset as data
    import util.dual_loss as dual_loss

    torch.manual_seed(771)
    model = FIXTURE.new_model("ib")
    image, _, _, _ = FIXTURE.sample_inputs()
    rgb = torch.ones(2, 3, 16, 8)
    rgb[:, :, 1:5, 2:4] = 0.2
    rgb[:, :, 9:13, 1:3] = 0.1
    rgb[:, :, 10:15, 5:7] = 0.25
    target = normalize(rgb)
    patch = torch.zeros(2, 4, 4, dtype=torch.bool)
    patch[:, 1:3, 1:3] = True
    pixels = patch[:, None].repeat_interleave(2, 2).repeat_interleave(2, 3)
    voro = torch.ones(2, 8, 8, dtype=torch.long)
    voro[:, :, 4:] = 2
    voro[1] = 0
    rows = []
    for index in range(2):
        qsha, rsha = (
            ("a" if index == 0 else "c") * 64,
            ("b" if index == 0 else "d") * 64,
        )
        rows.append(
            {
                "image": image[index].clone(),
                "target": target[index].clone(),
                "bf_patch_mask": patch[index].clone(),
                "bf_loss_mask": pixels[index].expand(3, -1, -1).float(),
                "has_jt": torch.tensor(index == 0),
                "has_bf": torch.tensor(True),
                "voro": voro[index].clone(),
                "valid_parts": torch.tensor([index == 0, index == 0, False, False]),
                "bf_mask_stats": {
                    "requested": 11,
                    "selected": 11 if index == 0 else 7,
                    "source_layers": 12 if index == 0 else 7,
                    "nonempty_source": 11 if index == 0 else 7,
                    "nonempty_after_crop": 11 if index == 0 else 7,
                    "target_patch_coverage": 0.25,
                },
                "reference_policy": runtime.REFERENCE_POLICY,
                "query_original_identity": f"one:{qsha}",
                "reference_original_identity": f"one:{rsha}",
                "query_original_sha256": qsha,
                "reference_original_sha256": rsha,
                "query_target_sha256": ("e" if index == 0 else "f") * 64,
                "reference_target_sha256": ("1" if index == 0 else "2") * 64,
                "author": "one",
                "target_path": f"query-{index}.png",
                "reference_path": f"reference-{index}.png",
            }
        )

    class Dataset:
        reference_policy = runtime.REFERENCE_POLICY
        pairs = [{"has_jt": True}] * 466 + [{"has_jt": False}] * 352

        def get_pair(self, index, augmentation_seed=None):
            return copy.deepcopy(rows[0 if index < 466 else 1])

    monkeypatch.setattr(
        data, "build_runc_v2_refine_dataset", lambda config, validation=False: Dataset()
    )
    original_j = dual_loss.DualJietiLoss
    monkeypatch.setattr(dual_loss, "DualJietiLoss", lambda: original_j(pool=8))
    monkeypatch.setattr(
        runtime,
        "source_record",
        lambda config: {
            "reference_policy": runtime.REFERENCE_POLICY,
            "source_sha": "CPU fixture",
        },
    )
    checkpoint = tmp_path / "parent.pth"
    checkpoint.write_bytes(b"CPU fixture parent checkpoint")
    output = tmp_path / "output"
    output.mkdir()
    config = {
        "stage": "ib",
        "checkpoint": str(checkpoint),
        "training": {"seed": 0, "amp": False},
    }
    return model, config, output, rows


@pytest.mark.parametrize("amp", [False, True])
def test_actual_stage_two_checks_and_scoped_roundtrip(setup_smoke, amp):
    model, config, output, _ = setup_smoke
    config["training"]["amp"] = amp
    result = runtime.stage2_interface_smoke(config, model, output, torch.device("cpu"))
    assert result["bf_mask_stats"]["selected"] == [11, 7]
    assert result["expected_selected_layers"] == [11, 7]
    assert result["bf_patch_mask_count"] == [4, 4]
    assert result["semantic_mask_exactly_preserved"]
    assert result["stage_one_exact_unchanged"]
    assert len(result["encoder_input_checks"]) == 5
    assert all(row["encoder_calls"] == 2 for row in result["encoder_input_checks"])
    assert all(
        row["second_input_matches_generated"] for row in result["encoder_input_checks"]
    )
    gradients = result["isolated_refiner_gradients"]
    assert gradients["bf_local_dual"]["norm"] > 0
    assert gradients["bf_local_bf_only"]["norm"] > 0
    assert gradients["dual_j_per_sample"]["norm"] > 0
    assert gradients["bf_only_j_per_sample"]["norm"] == 0
    assert all(result["hidden_gt_invariance"]["automatic"].values())
    assert "same Refiner instance" in result["strict_refiner_restore_scope"]
    assert "two fixed samples" in result["sample_scope"]
    assert result["sampled_authors"] == ["one", "one"]


@pytest.mark.parametrize("fault", ["requested", "selected", "empty"])
def test_invalid_actual_bf_sampling_or_support_stops(setup_smoke, fault):
    model, config, output, rows = setup_smoke
    if fault == "requested":
        rows[0]["bf_mask_stats"]["requested"] = 10
    elif fault == "selected":
        rows[1]["bf_mask_stats"]["selected"] = 11
    else:
        rows[1]["bf_patch_mask"].zero_()
        rows[1]["bf_loss_mask"].zero_()
    with pytest.raises(RuntimeError, match="BF"):
        runtime.stage2_interface_smoke(config, model, output, torch.device("cpu"))


def test_total_gradient_cannot_hide_disconnected_j_branch(setup_smoke, monkeypatch):
    import util.dual_loss as dual_loss

    model, config, output, _ = setup_smoke

    class DisconnectedJ(torch.nn.Module):
        def forward(self, prediction, *args):
            zero = prediction.sum() * 0
            per_sample = torch.stack((zero + 0.1, zero))
            return per_sample.mean(), {"per_sample": per_sample}, zero

    monkeypatch.setattr(dual_loss, "DualJietiLoss", DisconnectedJ)
    with pytest.raises(RuntimeError, match="dual_j_per_sample"):
        runtime.stage2_interface_smoke(config, model, output, torch.device("cpu"))


def test_actual_second_encoder_input_is_checked(setup_smoke, monkeypatch):
    import util.runc_v2 as model_api

    model, config, output, _ = setup_smoke
    original = model_api.runc_v2_refine_forward

    def wrong_input_report(*args, **kwargs):
        result = original(*args, **kwargs)
        result["masked_p0"] = result["masked_p0"] + 1
        return result

    monkeypatch.setattr(model_api, "runc_v2_refine_forward", wrong_input_report)
    with pytest.raises(RuntimeError, match="actual encoder input"):
        runtime.stage2_interface_smoke(config, model, output, torch.device("cpu"))


def test_automatic_hidden_gt_invariance_is_actually_compared(setup_smoke, monkeypatch):
    import util.runc_v2 as model_api

    model, config, output, _ = setup_smoke
    original = model_api.runc_v2_refine_forward
    automatic_calls = []

    def different_automatic_result(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("inference"):
            automatic_calls.append(True)
            if len(automatic_calls) == 2:
                result["p1"] = result["p1"] + 1
        return result

    monkeypatch.setattr(model_api, "runc_v2_refine_forward", different_automatic_result)
    with pytest.raises(RuntimeError, match="automatic p1 changed"):
        runtime.stage2_interface_smoke(config, model, output, torch.device("cpu"))
    assert len(automatic_calls) == 2
