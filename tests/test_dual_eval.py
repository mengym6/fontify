"""Strict checkpoint and shared evaluation contract checks on tiny CPU tensors."""

import argparse
import ast
import copy
import json
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from tools.calibrate_dual_eval import derive_eval_coefficients, parameter_gradient_gram
from tools.compare_dual_j import compare_prediction
from tools.dual_view_protocol import hash_files
from tools.eval_s_baseline import (
    DUAL_SCHEMA,
    eval_metadata,
    evaluate,
    file_sha256,
    reconcile_legacy_l1,
    require_source_model,
    validate_eval_coefficients,
)
from util.checkpoint import (
    build_checkpoint_model,
    checkpoint_is_dual,
    load_fontify_checkpoint,
)
from util.dual_loss import LowerJietiLoss
from util.jieti_loss import JietiLoss

ROOT = Path(__file__).resolve().parents[1]


class TinyModel(nn.Module):
    def __init__(self, dual_view=False):
        super().__init__()
        self.dual_view = dual_view
        self.trunk = nn.Linear(2, 2)
        if dual_view:
            self.dual_head = nn.Linear(2, 2)
            self.dual_fusion = nn.Linear(2, 2)
            self.dual_jieti_loss = LowerJietiLoss(pool=8)

    def enable_dual_losses(self, **kwargs):
        self.loss_config = kwargs
        self.dual_jieti_loss = LowerJietiLoss(pool=kwargs.get("pool", 8))

    def enable_jieti(self, alpha_jt, weight, **kwargs):
        self.jieti_loss_mod = JietiLoss(**kwargs)


def test_checkpoint_missing_unexpected_and_shape_fail_strictly():
    model = TinyModel()
    load_fontify_checkpoint(model, {"model": model.state_dict()})
    broken = dict(model.state_dict())
    del broken["trunk.bias"]
    with pytest.raises(RuntimeError, match="Missing key"):
        load_fontify_checkpoint(model, {"model": broken})
    broken = dict(model.state_dict(), alien=torch.zeros(1))
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_fontify_checkpoint(model, {"model": broken})
    broken = dict(model.state_dict())
    broken["trunk.weight"] = torch.zeros(3, 3)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_fontify_checkpoint(model, {"model": broken})


def test_migration_is_explicit_and_only_new_modules_may_be_missing():
    old, new = TinyModel(), TinyModel(True)
    checkpoint = {"model": old.state_dict()}
    with pytest.raises(RuntimeError, match="Missing key"):
        load_fontify_checkpoint(new, checkpoint)
    load_fontify_checkpoint(new, checkpoint, allow_dual_initialization=True)
    torch.testing.assert_close(new.trunk.weight, old.trunk.weight)
    checkpoint["model"] = dict(checkpoint["model"])
    del checkpoint["model"]["trunk.bias"]
    with pytest.raises(ValueError, match="Illegal migration"):
        load_fontify_checkpoint(new, checkpoint, allow_dual_initialization=True)


def test_architecture_metadata_and_keys_must_agree():
    with pytest.raises(ValueError, match="disagree"):
        checkpoint_is_dual({"model": TinyModel(True).state_dict()})
    with pytest.raises(ValueError, match="disagree"):
        checkpoint_is_dual(
            {"model": TinyModel().state_dict(), "dual_view_config": {"pool": 8}}
        )


def test_dual_rebuild_recovers_saved_args_and_legacy_loss_buffers():
    model = TinyModel(True)
    model.enable_jieti(1.0, 1.0, pool=8)
    checkpoint = {
        "model": model.state_dict(),
        "args": argparse.Namespace(
            dual_view=True,
            model="tiny",
            dual_lambda_full=0.75,
            jieti_w=0.33,
            dual_head_weight=0.18,
            jieti_pool=8,
            jieti_w_centroid=2.0,
            jieti_w_logsigma=3.0,
            jieti_w_shape=1.0,
        ),
    }
    restored = build_checkpoint_model(
        checkpoint, model_module=types.SimpleNamespace(tiny=TinyModel)
    )
    assert restored.dual_view and restored.loss_config["lambda_full"] == 0.75
    assert restored.loss_config["jieti_weight"] == 0.33
    assert restored.loss_config["head_weight"] == 0.18
    assert restored.loss_config["w_centroid"] == 2.0
    assert set(restored.state_dict()) == set(checkpoint["model"])


def test_legacy_jieti_buffers_are_rebuilt_rather_than_dropped():
    model = TinyModel()
    model.enable_jieti(1.0, 1.0, pool=8)
    checkpoint = {
        "model": model.state_dict(),
        "args": {"model": "tiny", "jieti_pool": 8},
    }
    restored = build_checkpoint_model(
        checkpoint, model_module=types.SimpleNamespace(tiny=TinyModel)
    )
    assert restored.jieti_loss_mod.coord.numel() == 8
    assert set(restored.state_dict()) == set(checkpoint["model"])


def test_dual_reconstruction_rejects_missing_training_loss_metadata():
    checkpoint = {"model": TinyModel(True).state_dict(), "args": {"dual_view": True}}
    with pytest.raises(ValueError, match="Missing dual-view loss metadata"):
        build_checkpoint_model(
            checkpoint, architecture="tiny",
            model_module=types.SimpleNamespace(tiny=TinyModel),
        )


def coefficient_payload(metadata):
    return {
        "schema": DUAL_SCHEMA,
        "purpose": "evaluation",
        "status": "validated",
        "source_role": "ctrl_g12_best",
        "source_epoch": 46,
        "source_checkpoint_sha256": "a" * 64,
        "coef": {"centroid": 2.0, "logsigma": 3.0, "shape": 1.0},
        **metadata,
    }


def test_eval_metadata_and_stale_coefficients(tmp_path):
    fixed = ROOT / "fontdata_example/val_pairs_fixed_clean.json"
    val = tmp_path / "val.json"
    val.write_text("[]")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "index.json").write_text("{}")
    (cache / "metadata.json").write_text(json.dumps({"definition": DUAL_SCHEMA}))
    args = argparse.Namespace(
        fixed_pair_path=fixed, val_json_path=[val], dual_target_cache=cache
    )
    metadata = eval_metadata(args)
    assert metadata["fixed_pair_sha256"] == file_sha256(fixed)
    assert metadata["target_cache_sha256"] == hash_files(
        [cache / "index.json", cache / "metadata.json"]
    )
    assert (
        validate_eval_coefficients(coefficient_payload(metadata), metadata)["centroid"]
        == 2.0
    )
    for field in ("fixed_pair_sha256", "val_manifest_sha256", "target_cache_sha256"):
        payload = coefficient_payload(metadata)
        payload[field] = "0" * 64
        with pytest.raises(ValueError, match=field):
            validate_eval_coefficients(payload, metadata)
    old = {"coef": {"centroid": 0.9688, "logsigma": 0.5409, "shape": 1.0}}
    with pytest.raises(ValueError, match="evaluation coefficients"):
        validate_eval_coefficients(old, metadata)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), True])
def test_eval_coef_zero_nonfinite_or_boolean_rejected(bad):
    payload = coefficient_payload({})
    payload["coef"]["shape"] = bad
    with pytest.raises(ValueError, match="positive finite"):
        validate_eval_coefficients(payload, {})


def test_legacy_reconciliation_checks_both_l1_terms_and_counts():
    result = {"L1_JT": 0.21, "L1_BF": 0.34, "n_jt": 105, "n_bf": 143}
    reference = dict(result, J=999)
    assert reconcile_legacy_l1(result, reference)["status"] == "passed"
    with pytest.raises(ValueError, match="L1_BF"):
        reconcile_legacy_l1(dict(result, L1_BF=0.33), reference)
    with pytest.raises(ValueError, match="sample-count"):
        reconcile_legacy_l1(dict(result, n_jt=104), reference)


def test_coefficient_source_checks_saved_training_identity():
    model = types.SimpleNamespace(
        _checkpoint_args={"output_dir": "/old/server/models/finetune_jieti_ctrl_g12"},
        _checkpoint_epoch=46,
        _checkpoint_source="/new/server/checkpoint-best.pth",
        dual_view=False,
    )
    require_source_model(model, "ctrl_g12_best")
    model._checkpoint_args["output_dir"] = "models/other_epoch46"
    with pytest.raises(ValueError, match="training identity"):
        require_source_model(model, "ctrl_g12_best")


def test_calibration_uses_true_parameter_gradients_and_mean_norm_ratio():
    parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
    gram = parameter_gradient_gram(
        [parameter.sum(), 2 * parameter.sum(), parameter[0] - parameter[1]],
        [parameter],
        chunk_size=1,
    )
    assert gram == [[2, 4, 0], [4, 8, 0], [0, 0, 2]]
    coefficients, _ = derive_eval_coefficients([gram])
    assert coefficients == {"centroid": 1.0, "logsigma": 0.5, "shape": 1.0}
    with pytest.raises(ValueError, match="Zero/nonfinite"):
        derive_eval_coefficients([[[0, 0, 0], [0, 1, 0], [0, 0, 1]]])


def test_fixed_prediction_old_new_j_is_paired_and_upper_gradient_is_zero():
    torch.manual_seed(1)
    prediction = torch.rand(1, 3, 32, 16) * 2 - 1
    target = torch.rand_like(prediction) * 2 - 1
    new_partition = torch.ones(1, 8, 8, dtype=torch.long)
    new_partition[:, :, 4:] = 2
    old_partition = torch.stack((new_partition, new_partition), dim=1)
    new_valid = torch.tensor([[True, True, False, False]])
    old_valid = torch.stack((new_valid, new_valid), dim=1)
    result = compare_prediction(
        prediction,
        target,
        old_partition,
        old_valid,
        new_partition,
        new_valid,
        JietiLoss(pool=8),
        LowerJietiLoss(pool=8),
    )
    for name in ("J", "centroid", "logsigma", "shape"):
        row = result["terms"][name]
        assert row["finite"] and row["new_norm"] > 0 and row["old_norm"] > 0
        assert row["old_upper_max"] == row["new_upper_max"] == 0
    assert result["terms"]["J"]["cosine"] > 0.99


def test_eval_predict_full_has_no_annotation_inputs_and_per_item_means():
    class MetricModel:
        patch_size = 1

        def predict_full(self, images, targets, mask):
            return torch.zeros_like(targets)

        def unpatchify(self, values):
            return values.transpose(1, 2).reshape(-1, 3, 4, 2)

    class Samples:
        pairs = [{"target_path": "ref-jt"}, {"target_path": "ref-bf"}]
        _fixed_pairs = {
            "ref-jt": {"target_path": "q-jt", "type": "JT"},
            "ref-bf": {"target_path": "q-bf", "type": "BF"},
        }

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {
                "image": torch.zeros(3, 4, 2),
                "target": torch.ones(3, 4, 2) * (index + 1),
                "mask": torch.tensor([[0, 0], [0, 0], [1, 1], [1, 1]]),
                "valid": torch.ones(3, 4, 2),
                "voro": torch.zeros(8, 8),
                "valid_parts": torch.zeros(4, dtype=torch.bool),
                "has_jt": index == 0,
            }

    result, rows = evaluate(MetricModel(), Samples(), None, "cpu")
    assert result["L1_JT"] == pytest.approx(12 / 12.01)
    assert result["L1_BF"] == pytest.approx(24 / 12.01)
    assert result["n_jt"] == result["n_bf"] == 1
    assert rows[0]["target_path"] == "q-jt" and rows[1]["kind"] == "BF"


@pytest.mark.parametrize("issue", ["nan", "bf_only", "jt_only"])
def test_evaluation_rejects_nonfinite_or_missing_kind(issue):
    class Samples(list):
        pairs = [{"target_path": "ref-0"}, {"target_path": "ref-1"}]
        _fixed_pairs = {
            f"ref-{index}": {"target_path": f"q-{index}", "type": "JT"}
            for index in range(2)
        }

    samples = Samples()
    for index in range(2):
        active = issue == "jt_only" or (issue == "nan" and index == 0)
        samples.append(
            {
                "image": torch.zeros(3, 4, 2),
                "target": torch.ones(3, 4, 2),
                "mask": torch.tensor([[0, 0], [0, 0], [1, 1], [1, 1]]),
                "valid": torch.ones(3, 4, 2),
                "voro": torch.zeros(8, 8),
                "valid_parts": torch.zeros(4, dtype=torch.bool),
                "has_jt": active,
            }
        )
    model = types.SimpleNamespace(
        patch_size=1,
        predict_full=lambda images, targets, mask: torch.full_like(
            targets, float("nan") if issue == "nan" else 0
        ),
        unpatchify=lambda values: values.transpose(1, 2).reshape(-1, 3, 4, 2),
    )
    expected = FloatingPointError if issue == "nan" else ValueError
    with pytest.raises(expected):
        evaluate(model, samples, None, "cpu")


def test_standalone_and_training_evaluation_agree_in_fp32_with_valid_pixels(tmp_path):
    from util.dual_training import evaluate_dual

    class Model(nn.Module):
        patch_size = 2

        def predict_full(self, image, target, mask=None):
            return image

        def unpatchify(self, values):
            return (
                values.reshape(-1, 8, 4, 2, 2, 3)
                .permute(0, 5, 1, 3, 2, 4)
                .reshape(-1, 3, 16, 8)
            )

    class Samples(list):
        pairs = [{"target_path": f"ref-{index}"} for index in range(3)]
        _fixed_pairs = {
            f"ref-{index}": {"target_path": f"q-{index}", "type": "JT"}
            for index in range(3)
        }

    torch.manual_seed(91)
    target = torch.rand(3, 3, 16, 8) * 2 - 1
    image = target + torch.rand_like(target) * 0.3
    mask = torch.zeros(3, 8, 4)
    mask[:, 4:] = 1
    valid = torch.ones_like(target)
    valid[0, :, 8:10] = 0
    voro = torch.ones(3, 8, 8, dtype=torch.long)
    voro[:, :, 4:] = 2
    batch = {
        "image": image,
        "target": target,
        "mask": mask,
        "valid": valid,
        "has_jt": torch.tensor([True, True, False]),
        "voro": voro,
        "valid_parts": torch.tensor([[True, True, False, False]] * 3),
    }
    samples = Samples(
        {key: value[index] for key, value in batch.items()} for index in range(3)
    )
    coefficients = {"centroid": 2, "logsigma": 3, "shape": 4}
    coef_path = tmp_path / "coef.json"
    coef_path.write_text(json.dumps({"coef": coefficients}))
    args = types.SimpleNamespace(
        dual_eval_calibration_json=str(coef_path),
        jieti_pool=8,
        jieti_k_max=4,
        jieti_soft_fg="linear",
        jieti_pred_mass_ratio=0.1,
        vis_every_epoch=False,
    )
    grouped = evaluate_dual([batch], Model(), "cpu", 0, args)
    standalone, _ = evaluate(
        Model(),
        samples,
        LowerJietiLoss(pool=8, w_centroid=2, w_logsigma=3, w_shape=4),
        "cpu",
        DUAL_SCHEMA,
    )
    for key in ("L1_JT", "L1_BF", "n_jt", "n_bf"):
        assert standalone[key] == pytest.approx(grouped[key], abs=1e-7, rel=1e-7)
    assert standalone["J"] == pytest.approx(grouped["jieti_J"], abs=1e-7, rel=1e-7)


def test_models_eval_prediction_rejects_nonfull_masks_and_ignores_lower_gt():
    source = ast.parse((ROOT / "models_eval.py").read_text())
    fontify = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "Fontify"
    )
    method = next(
        node for node in fontify.body if getattr(node, "name", "") == "predict_full"
    )
    namespace = {"torch": torch}
    exec(
        compile(
            ast.Module(body=[copy.deepcopy(method)], type_ignores=[]),
            "models_eval",
            "exec",
        ),
        namespace,
    )

    class Small:
        patch_embed = types.SimpleNamespace(num_patches=8)

        def forward_encoder(self, images, targets, mask):
            return targets.flatten(2) * (~mask[:, None]).float()

        def forward_decoder(self, values):
            return values

    model = Small()
    model.predict_full = types.MethodType(namespace["predict_full"], model)
    images = torch.zeros(1, 3, 4, 2)
    targets = torch.rand_like(images)
    before = model.predict_full(images, targets)
    targets[:, :, 2:] += 10
    assert torch.equal(model.predict_full(images, targets), before)
    with pytest.raises(ValueError, match="fully masked"):
        model.predict_full(images, targets, torch.zeros(1, 8))
