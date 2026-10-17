"""Run C v2 actual Fontify-method contracts with a small CPU fixture."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from util.runc_v2 import (
    STRUCTURE_KEYS,
    enable_runc_v2,
    enable_runc_v2_refiner,
    load_runc_v2_pretrained,
    render_runc_v2,
    restore_runc_v2,
    runc_v2_checkpoint_metadata,
    runc_v2_forward,
    runc_v2_parameter_report,
    runc_v2_refine_forward,
    set_runc_v2_stage,
    state_tensor_digest,
)

SPEC = importlib.util.spec_from_file_location(
    "_runc_v2_fontify_fixture", Path(__file__).with_name("test_dual_model.py"),
)
FIXTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURE)
small_model = FIXTURE.small_model
sample_inputs = FIXTURE.sample_inputs


def new_model(stage="ia0"):
    model = small_model()
    enable_runc_v2(model, {
        "head_width": 16, "hidden_dim": 16, "stage": stage,
    })
    return model


def group_has_gradient(model, prefix):
    gradients = [parameter.grad for name, parameter in model.named_parameters()
                 if name.startswith(prefix) and parameter.grad is not None]
    return bool(gradients) and any(
        value.count_nonzero() for value in gradients
    )


def test_disabled_path_preserves_legacy_keys_and_values():
    model = small_model().eval()
    baseline = copy.deepcopy(model)
    imgs, tgts, _, _ = sample_inputs()
    assert model.state_dict().keys() == baseline.state_dict().keys()
    torch.testing.assert_close(model.predict_full(imgs, tgts),
                               baseline.predict_full(imgs, tgts),
                               rtol=0, atol=0)


def test_shapes_density_polarity_and_exactly_once_normalization():
    model = new_model("ib").eval()
    imgs, tgts, _, _ = sample_inputs()
    result = runc_v2_forward(model, imgs, tgts, return_features=True)
    assert result["density"].shape == (2, 1, 2, 2)
    assert result["occupancy_logits"].shape == (2, 1, 2, 2)
    assert result["p0_lower"].shape == (2, 3, 8, 8)
    assert result["p0"].shape == (2, 3, 16, 8)
    assert result["lower_features"].shape == (2, 4, 4, 32)
    assert result["head"]["query"].shape == (2, 16, 16)
    assert (result["density"] >= 0).all() and (result["density"] <= 1).all()
    mean = tgts.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = tgts.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    pixels = result["p0_lower"] * std + mean
    expected = 1 - torch.nn.functional.interpolate(
        result["density"], (8, 8), mode="bilinear", align_corners=False,
    )
    torch.testing.assert_close(pixels, expected.expand(-1, 3, -1, -1))
    torch.testing.assert_close(result["p0"][:, :, :8], tgts[:, :, :8])
    torch.testing.assert_close(model.predict_full(imgs, tgts), result["p0"])
    assert not result["head"]["occupancy_logits"].requires_grad


@pytest.mark.parametrize("fill", [float("nan"), float("inf"), -10000.0])
def test_deployment_ignores_arbitrary_lower_gt_before_embedding(fill):
    model = new_model("ic").eval()
    imgs, tgts, _, _ = sample_inputs()
    baseline = runc_v2_forward(model, imgs, tgts)
    changed = tgts.clone()
    changed[:, :, 8:] = fill
    actual = runc_v2_forward(model, imgs, changed)
    for key in ("density", "occupancy_logits", "p0_lower", "p0"):
        torch.testing.assert_close(actual[key], baseline[key], rtol=0, atol=0)
    for key in baseline["head"]:
        torch.testing.assert_close(actual["head"][key], baseline["head"][key],
                                   rtol=0, atol=0)
    assert baseline["full_mask"].reshape(2, 8, 4)[:, 4:].all()
    assert not baseline["full_mask"].reshape(2, 8, 4)[:, :4].any()


def test_renderer_is_slot_permutation_invariant_and_uses_s_not_slot_id():
    model = new_model("ib").eval()
    imgs, tgts, _, _ = sample_inputs()
    result = runc_v2_forward(model, imgs, tgts, return_features=True)
    permutation = torch.tensor([2, 0, 3, 1])
    structure = {key: result["head"][key][:, permutation]
                 for key in STRUCTURE_KEYS}
    actual = render_runc_v2(
        model, result["lower_features"], result["reference_features"],
        result["head"], structure,
    )
    torch.testing.assert_close(actual["density"], result["density"],
                               rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual["occupancy_logits"],
                               result["occupancy_logits"],
                               rtol=1e-5, atol=1e-6)


def test_fixed_f_q_condition_response_and_soft_occupancy_gradient():
    model = new_model("ib").eval()
    imgs, tgts, _, _ = sample_inputs()
    original = runc_v2_forward(model, imgs, tgts, return_features=True)
    structure = {key: original["head"][key].detach().clone()
                 for key in STRUCTURE_KEYS}
    occupancy = structure["occupancy_logits"].requires_grad_(True)
    changed = render_runc_v2(
        model, original["lower_features"], original["reference_features"],
        original["head"], structure,
    )
    gradient = torch.autograd.grad(changed["density"].sum(), occupancy)[0]
    assert torch.isfinite(gradient).all() and gradient.count_nonzero() > 0
    structure["occupancy_logits"] = torch.full_like(occupancy, -30)
    absent = render_runc_v2(
        model, original["lower_features"], original["reference_features"],
        original["head"], structure,
    )
    assert (absent["density"] - changed["density"]).abs().max() > 1e-7


@pytest.mark.parametrize("stage,head,vit,coarse", [
    ("ia0", True, False, False), ("ia1", True, True, False),
    ("ib", False, False, True), ("ic", True, True, True),
])
def test_stage_trainability_gradients_modes_and_frozen_values(
        stage, head, vit, coarse):
    model = new_model(stage).train()
    imgs, tgts, _, _ = sample_inputs()
    names = set_runc_v2_stage(model, stage)
    assert bool(names)
    frozen = {name: value.detach().clone()
              for name, value in model.named_parameters()
              if not value.requires_grad}
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters()
         if parameter.requires_grad],
        lr=0.1,
    )
    result = runc_v2_forward(model, imgs, tgts)
    losses = []
    if head:
        losses.extend(result["head"][key].square().mean()
                      for key in STRUCTURE_KEYS)
    if coarse:
        losses.extend([result["density"].square().mean(),
                       result["occupancy_logits"].square().mean()])
    sum(losses).backward()
    assert group_has_gradient(model, "dual_head.") == head
    assert group_has_gradient(model, "blocks.") == vit
    assert group_has_gradient(model, "runc_v2_coarse.") == coarse
    assert not group_has_gradient(model, "decoder_")
    assert not model.patch_embed.training
    assert model.dual_head.training == head
    assert model.runc_v2_coarse.training == coarse
    for index, block in enumerate(model.blocks):
        assert block.training == (vit and index in (2, 5, 8, 11))
    optimizer.step()
    for name, value in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
    report = runc_v2_parameter_report(model)
    assert report["unfreeze_blocks"] == [2, 5, 8, 11]
    assert report["groups"]["legacy_unused"]["trainable_count"] == 0


def test_checkpoint_roundtrip_and_strict_pretraining_migration():
    legacy = small_model()
    state = copy.deepcopy(legacy.state_dict())
    report = load_runc_v2_pretrained(legacy, {"model": state})
    assert report["source_model_tensor_sha256"] == state_tensor_digest(state)
    assert report["loaded_key_count"] == len(state)
    assert not report["missing"] and not report["shape_mismatch"]
    enable_runc_v2(legacy, {"head_width": 16, "hidden_dim": 16, "stage": "ib"})
    restored = small_model()
    enable_runc_v2(restored, dict(legacy.runc_v2_config))
    restored.load_state_dict(legacy.state_dict(), strict=True)
    imgs, tgts, _, _ = sample_inputs()
    torch.testing.assert_close(
        runc_v2_forward(legacy.eval(), imgs, tgts)["p0"],
        runc_v2_forward(restored.eval(), imgs, tgts)["p0"], rtol=0, atol=0,
    )
    with pytest.raises(ValueError, match="legacy model first"):
        load_runc_v2_pretrained(restored, {"model": state})
    altered = dict(state)
    altered["dual_head.fake"] = torch.ones(1)
    with pytest.raises(ValueError, match="adapted weights"):
        load_runc_v2_pretrained(small_model(), {"model": altered})
    altered = dict(state)
    altered["norm.weight"] = torch.ones(123)
    with pytest.raises(ValueError, match="Incompatible pretrained"):
        load_runc_v2_pretrained(small_model(), {"model": altered})


def test_rejects_adapted_models_and_invalid_stage_configuration():
    with pytest.raises(ValueError, match="legacy, unadapted"):
        enable_runc_v2(small_model(dual=True))
    with pytest.raises(ValueError, match="subset of taps"):
        enable_runc_v2(small_model(), {"unfreeze_blocks": [10, 11]})
    with pytest.raises(ValueError, match="model dimensions"):
        enable_runc_v2(small_model(), {"feature_width": 3072})
    with pytest.raises(ValueError, match="four unordered JT slots"):
        enable_runc_v2(small_model(), {"slots": 3})


def test_stage_two_identity_bf_mask_frozen_values_and_actual_p0_input():
    model = new_model("ib").train()
    enable_runc_v2_refiner(model, {"hidden_dim": 8})
    assert model.runc_v2_refiner_config["bf_annotation_layers"] == 11
    assert model.runc_v2_refiner_config["aggregate_bf_types"] is False
    names = set_runc_v2_stage(model, "refine")
    assert names and all(name.startswith("bf_refiner.") for name in names)
    imgs, tgts, _, _ = sample_inputs()
    layers = torch.zeros(11, 4, 4, dtype=torch.bool)
    for index in range(11):
        layers[index, index // 4, index % 4] = True
    supplied = layers.any(0).unsqueeze(0).expand(2, -1, -1)
    original_encoder = model.forward_encoder
    encoded_targets = []

    def recording_encoder(images, target, mask, **kwargs):
        encoded_targets.append(target.detach().clone())
        return original_encoder(images, target, mask, **kwargs)

    model.forward_encoder = recording_encoder
    frozen = {name: value.detach().clone()
              for name, value in model.named_parameters()
              if not name.startswith("bf_refiner.")}
    optimizer = torch.optim.SGD(model.bf_refiner.parameters(), lr=0.1)
    result = runc_v2_refine_forward(model, imgs, tgts, bf_mask=supplied)
    torch.testing.assert_close(result["p0"], result["p1"], rtol=0, atol=0)
    torch.testing.assert_close(result["mask"], supplied, rtol=0, atol=0)
    assert len(encoded_targets) == 2
    torch.testing.assert_close(encoded_targets[1], result["masked_p0"],
                               rtol=0, atol=0)
    assert torch.count_nonzero(encoded_targets[0][:, :, 8:]) == 0
    assert not torch.equal(encoded_targets[1][:, :, 8:], tgts[:, :, 8:])
    result["p1"][:, :, 8:].square().mean().backward()
    assert group_has_gradient(model, "bf_refiner.")
    assert not group_has_gradient(model, "dual_head.")
    assert not group_has_gradient(model, "runc_v2_coarse.")
    assert not group_has_gradient(model, "blocks.")
    optimizer.step()
    for name, value in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
    report = runc_v2_parameter_report(model)
    assert report["groups"]["refiner"]["trainable_count"] > 0
    assert all(report["groups"][name]["trainable_count"] == 0
               for name in ("vit", "head", "coarse", "legacy_unused"))


@pytest.mark.parametrize("inference", [False, True])
def test_stage_two_no_gt_leak_with_nonzero_refiner(inference):
    model = new_model("ib").eval()
    enable_runc_v2_refiner(model, {
        "hidden_dim": 8, "auto_candidates": 3, "auto_rounds": 2,
    })
    torch.nn.init.normal_(model.bf_refiner.output.weight, std=0.01)
    imgs, tgts, _, _ = sample_inputs()
    supplied = torch.zeros(2, 4, 4, dtype=torch.bool)
    supplied[:, 1:3, 2] = True
    expected = runc_v2_refine_forward(
        model, imgs, tgts, bf_mask=supplied, inference=inference,
    )
    altered = tgts.clone()
    altered[:, :, 8:] = float("nan")
    actual = runc_v2_refine_forward(
        model, imgs, altered, bf_mask=supplied, inference=inference,
    )
    assert len(actual["masks"]) == (2 if inference else 1)
    for key in ("p0", "p1", "mask", "masked_p0"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    if inference:
        torch.testing.assert_close(model.predict_full(imgs, altered),
                                   actual["p1"], rtol=0, atol=0)


def test_stage_two_rejects_missing_mask_and_aggregated_bf_contract():
    model = new_model()
    with pytest.raises(ValueError, match="unaggregated BF"):
        enable_runc_v2_refiner(model, {"aggregate_bf_types": True})
    enable_runc_v2_refiner(model)
    imgs, tgts, _, _ = sample_inputs()
    with pytest.raises(ValueError, match="requires an existing BF"):
        runc_v2_refine_forward(model, imgs, tgts)


def test_stage_two_checkpoint_restoration_is_strict_and_preserves_metadata():
    model = new_model("ib").eval()
    enable_runc_v2_refiner(model, {"hidden_dim": 8, "auto_rounds": 2})
    torch.nn.init.normal_(model.bf_refiner.output.weight, std=0.01)
    metadata = runc_v2_checkpoint_metadata(model)
    checkpoint = {**metadata, "model": copy.deepcopy(model.state_dict())}
    restored = restore_runc_v2(small_model(), checkpoint).eval()
    assert restored.runc_v2_stage == "refine"
    assert restored.runc_v2_refiner_config == model.runc_v2_refiner_config
    imgs, tgts, _, _ = sample_inputs()
    torch.testing.assert_close(restored.predict_full(imgs, tgts),
                               model.predict_full(imgs, tgts), rtol=0, atol=0)
    set_runc_v2_stage(model, "ib")
    assert metadata["runc_v2_config"]["stage"] == "refine"
    checkpoint = {
        **runc_v2_checkpoint_metadata(model),
        "model": copy.deepcopy(model.state_dict()),
    }
    assert restore_runc_v2(small_model(), checkpoint).runc_v2_stage == "ib"
    broken = copy.deepcopy(checkpoint)
    del broken["runc_v2_refiner_config"]
    with pytest.raises(ValueError, match="metadata and tensor keys disagree"):
        restore_runc_v2(small_model(), broken)
    broken = copy.deepcopy(checkpoint)
    del broken["model"]["bf_refiner.output.weight"]
    with pytest.raises(RuntimeError, match="Missing key"):
        restore_runc_v2(small_model(), broken)
    broken = copy.deepcopy(checkpoint)
    broken["model"]["bf_refiner.output.weight"] = (
        broken["model"]["bf_refiner.output.weight"].half()
    )
    with pytest.raises(ValueError, match="tensor dtype mismatch"):
        restore_runc_v2(small_model(), broken)
    broken = copy.deepcopy(checkpoint)
    del broken["runc_v2_config"]["tap_indices"]
    with pytest.raises(ValueError, match="Incomplete Run C v2"):
        restore_runc_v2(small_model(), broken)


def test_stage_one_checkpoint_requires_real_keys_and_fresh_model():
    model = new_model("ib")
    checkpoint = {**runc_v2_checkpoint_metadata(model),
                  "model": copy.deepcopy(model.state_dict())}
    restored = restore_runc_v2(small_model(), checkpoint)
    assert restored.runc_v2_stage == "ib"
    with pytest.raises(ValueError, match="legacy, unadapted"):
        restore_runc_v2(restored, checkpoint)
    broken = copy.deepcopy(checkpoint)
    broken["model"] = {name: value for name, value in broken["model"].items()
                       if not name.startswith("runc_v2_coarse.")}
    with pytest.raises(ValueError, match="missing module"):
        restore_runc_v2(small_model(), broken)
