"""Run C actual Fontify method contracts with a small CPU fixture."""

import copy
import importlib.util
import io
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from util.checkpoint import build_checkpoint_model, load_fontify_checkpoint
from util.structure_first import (
    SCHEMA,
    auto_structure_mask,
    enable_structure_first,
    normalize_bf_mask,
    set_structure_stage,
    structure_forward,
)

SPEC = importlib.util.spec_from_file_location(
    "_runc_dual_fixture", Path(__file__).with_name("test_dual_model.py"),
)
FIXTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURE)
small_model = FIXTURE.small_model
sample_inputs = FIXTURE.sample_inputs


def new_model(**config):
    model = small_model(dual=True)
    enable_structure_first(
        model, {"hidden_dim": 16, "rgb_hidden": 8, **config},
    )
    return model


def bf_mask(batch=2):
    mask = torch.zeros(batch, 4, 4, dtype=torch.bool)
    mask[:, 1:3, 2] = True
    return mask


def assert_no_grad(parameters):
    assert all(value.grad is None or value.grad.count_nonzero() == 0
               for value in parameters)


def assert_nonzero_grad(parameters):
    values = [value.grad for value in parameters if value.grad is not None]
    assert values and all(torch.isfinite(value).all() for value in values)
    assert any(value.count_nonzero() > 0 for value in values)


def test_disabled_keys_and_output_remain_exactly_legacy():
    model = small_model(dual=True).eval()
    baseline = copy.deepcopy(model)
    images, target, _, _ = sample_inputs()
    expected = baseline.predict_full(images, target)
    actual = model.predict_full(images, target)
    assert model.state_dict().keys() == baseline.state_dict().keys()
    assert not any("structure" in key for key in model.state_dict())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_zero_conditioner_coarse_equals_legacy_and_no_gt_nan_leak():
    model = small_model(dual=True).eval()
    nn.init.normal_(model.dual_fusion.projection.weight, std=0.01)
    images, target, _, _ = sample_inputs()
    expected = model.predict_full(images, target)
    enable_structure_first(model, {"hidden_dim": 16, "rgb_hidden": 8})
    output = model.predict_full(images, target)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    changed = target.clone()
    changed[:, :, 8:] = float("nan")
    torch.testing.assert_close(model.predict_full(images, changed), output,
                               rtol=0, atol=0)


@pytest.mark.parametrize(
    "active", [[False, False], [True, True], [True, False]],
)
def test_all_head_sources_route_bf_only_stop_gradient(active):
    model = new_model().train()
    nn.init.normal_(model.structure_adapter.projection.weight, std=0.02)
    nn.init.normal_(model.dual_fusion.projection.weight, std=0.02)
    images, target, _, _ = sample_inputs()
    result = structure_forward(
        model, images, target, "coarse", has_jt=torch.tensor(active),
    )
    for value in result["head"].values():
        value.retain_grad()
    result["p0"][:, :, 8:].square().mean().backward()
    for key, value in result["head"].items():
        if value.grad is None:
            continue
        for index, enabled in enumerate(active):
            assert bool(value.grad[index].count_nonzero() > 0) == enabled, key
    if any(active):
        assert_nonzero_grad(model.dual_head.parameters())
    else:
        assert_no_grad(model.dual_head.parameters())
    assert_nonzero_grad(model.structure_adapter.parameters())
    assert_nonzero_grad(model.dual_fusion.parameters())
    assert_nonzero_grad(model.decoder_pred.parameters())
    assert_no_grad(model.blocks.parameters())


def test_stage_two_identity_frozen_stage_one_and_valid_gradient():
    model = new_model().train()
    names = set_structure_stage(model, "refine")
    assert names and all(name.startswith("bf_refiner.") for name in names)
    images, target, _, _ = sample_inputs()
    original = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
        if not name.startswith("bf_refiner.")
    }
    optimizer = torch.optim.SGD(model.bf_refiner.parameters(), lr=0.1)
    result = structure_forward(
        model, images, target, "refine", bf_mask=bf_mask(),
    )
    torch.testing.assert_close(result["p0"], result["p1"], rtol=0, atol=0)
    assert result["mask"].shape == (2, 4, 4)
    assert torch.count_nonzero(result["masked_p0"][:, :, 10:14, 4:6]) == 0
    result["p1"][:, :, 8:].square().mean().backward()
    assert_no_grad(value for name, value in model.named_parameters()
                   if not name.startswith("bf_refiner."))
    assert_nonzero_grad(model.bf_refiner.adapter.parameters())
    assert_nonzero_grad(model.bf_refiner.rgb.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    result = structure_forward(
        model, images, target, "refine", bf_mask=bf_mask(),
    )
    result["p1"][:, :, 8:].square().mean().backward()
    assert_nonzero_grad(model.bf_refiner.adapter.feature.parameters())
    for name, value in model.named_parameters():
        if name in original:
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)


@pytest.mark.parametrize("fill", [float("nan"), -10000.0, 10000.0])
def test_refine_and_multi_round_inference_ignore_target_lower(fill):
    model = new_model(auto_rounds=2).eval()
    set_structure_stage(model, "refine")
    nn.init.normal_(model.bf_refiner.adapter.projection.weight, std=0.02)
    nn.init.normal_(model.bf_refiner.rgb[-1].weight, std=0.02)
    images, target, _, _ = sample_inputs()
    changed = target.clone()
    changed[:, :, 8:] = fill
    for stage in ("refine", "infer"):
        expected = structure_forward(
            model, images, target, stage, bf_mask=bf_mask(),
        )
        actual = structure_forward(
            model, images, changed, stage, bf_mask=bf_mask(),
        )
        repeated = structure_forward(
            model, images, target, stage, bf_mask=bf_mask(),
        )
        for key in ("p0", "p1", "masked_p0", "mask"):
            torch.testing.assert_close(
                actual[key], expected[key], rtol=0, atol=0,
            )
            torch.testing.assert_close(
                repeated[key], expected[key], rtol=0, atol=0,
            )
        assert len(actual["masks"]) == (2 if stage == "infer" else 1)
        if stage == "infer":
            torch.testing.assert_close(
                actual["intermediates"][1], expected["intermediates"][1],
                rtol=0, atol=0,
            )


def test_empty_background_and_stable_score_ties_have_nonempty_mask():
    model = new_model(auto_candidates=3, auto_threshold=0.0).eval()
    images, target, _, _ = sample_inputs()
    head = structure_forward(model, images, target, "coarse")["head"]
    head = {key: torch.zeros_like(value) for key, value in head.items()}
    # Standard normalized RGB white and black give constant patch scores.
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    for pixels in (torch.ones_like(target), torch.zeros_like(target)):
        prediction = (pixels - mean) / std
        mask = auto_structure_mask(model, prediction, head)
        assert mask.flatten(1)[:, :3].all()
        assert mask.sum().item() == 6
    model.structure_first_config["auto_threshold"] = 1.0
    fallback = auto_structure_mask(
        model, (torch.ones_like(target) - mean) / std, head,
    )
    assert fallback.flatten(1)[:, 0].all()
    assert fallback.sum().item() == 2


@pytest.mark.parametrize("mode", ["empty", "all", "overlap"])
def test_actual_mask_is_not_resampled(mode):
    model = new_model().eval()
    set_structure_stage(model, "refine")
    images, target, _, _ = sample_inputs()
    supplied = bf_mask()
    if mode == "empty":
        supplied.zero_()
    elif mode == "all":
        supplied.fill_(True)
    result = structure_forward(
        model, images, target, "refine", bf_mask=supplied,
    )
    torch.testing.assert_close(result["mask"], supplied, rtol=0, atol=0)
    full = torch.cat((torch.zeros_like(supplied), supplied), 1)
    torch.testing.assert_close(normalize_bf_mask(full, images, 2), supplied)


def test_forward_api_and_compact_decoder_parameter_selection():
    model = new_model(coarse_decoder="pred")
    names = set_structure_stage(model, "coarse")
    assert not any(name.startswith("decoder_embed.") for name in names)
    images, target, mask, valid = sample_inputs()
    output = model(images, target, mask, valid, no_gan=True,
                   dual_targets={"has_jt": torch.tensor([True, False])})
    assert "p0" in output and "p1" not in output
    set_structure_stage(model, "refine")
    output = model(images, target, mask, valid, no_gan=True,
                   dual_targets={"bf_patch_mask": bf_mask()})
    assert "p1" in output


def checkpoint_for(model):
    return {
        "model": model.state_dict(), "args": {"dual_view": True},
        "dual_view_config": {
            "lambda_full": 0.5, "w_j": 1.0, "w_h": 1.0,
            "coef": {"centroid": 1.0, "logsigma": 1.0, "shape": 1.0},
            "pool": 8,
        },
        "structure_first_config": dict(model.structure_first_config),
    }


def test_checkpoint_roundtrip_reconstructs_exact_architecture_and_stages():
    model = new_model(auto_rounds=2).eval()
    set_structure_stage(model, "refine")
    nn.init.normal_(model.bf_refiner.adapter.projection.weight, std=0.01)
    saved = io.BytesIO()
    torch.save(checkpoint_for(model), saved)
    saved.seek(0)
    checkpoint = torch.load(saved, weights_only=False)
    module = SimpleNamespace(
        small=lambda dual_view: small_model(dual=dual_view),
    )
    restored = build_checkpoint_model(
        checkpoint, architecture="small", model_module=module,
    )
    restored.eval()
    assert restored.structure_first_stage == "refine"
    assert restored.structure_first_config == model.structure_first_config
    images, target, _, _ = sample_inputs()
    torch.testing.assert_close(
        restored.predict_full(images, target), model.predict_full(images, target),
        rtol=0, atol=0,
    )
    assert all(
        not value.requires_grad for name, value in restored.named_parameters()
        if not name.startswith("bf_refiner.")
    )


@pytest.mark.parametrize(
    "change", ["metadata", "missing_key", "dimension", "schema"],
)
def test_checkpoint_rejects_metadata_key_or_shape_disagreement(change):
    model = new_model()
    checkpoint = checkpoint_for(model)
    if change == "metadata":
        checkpoint.pop("structure_first_config")
    elif change == "missing_key":
        checkpoint["model"].pop("structure_adapter.projection.weight")
    elif change == "dimension":
        checkpoint["structure_first_config"]["feature_width"] = 31
    else:
        checkpoint["structure_first_config"]["schema"] = SCHEMA + "-unknown"
    with pytest.raises((ValueError, RuntimeError)):
        load_fontify_checkpoint(small_model(dual=True), checkpoint)


@pytest.mark.parametrize("stage", ["coarse", "refine"])
def test_cpu_ddp_static_graph_across_pool_mixtures(stage):
    """One gloo rank tests graph reuse; GPU/multiple ranks need own smoke."""
    if not torch.distributed.is_available():
        pytest.skip("distributed unavailable")
    if torch.distributed.is_initialized():
        pytest.skip("caller already owns a distributed process group")
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = Path(directory) / "gloo-store"
        torch.distributed.init_process_group(
            "gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1,
        )
        try:
            model = new_model()
            set_structure_stage(model, stage)
            wrapped = nn.parallel.DistributedDataParallel(
                model, static_graph=True,
            )
            optimizer = torch.optim.SGD(
                (value for value in wrapped.parameters()
                 if value.requires_grad),
                lr=0.01,
            )
            for active in ([False, False], [True, True], [True, False]):
                optimizer.zero_grad(set_to_none=True)
                output = wrapped(
                    *sample_inputs(), no_gan=True,
                    dual_targets={"has_jt": torch.tensor(active),
                                  "bf_patch_mask": bf_mask()},
                )
                loss = output["p0" if stage == "coarse" else "p1"]
                loss.square().mean().backward()
                optimizer.step()
        finally:
            torch.distributed.destroy_process_group()


def test_real_vit_attention_small_grid_two_stage_backward():
    """Actual attention/ViT code, tiny dimensions and a VGG substitute."""
    model = FIXTURE.fontify_class(real_blocks=True)(
        img_size=(16, 8), patch_size=2, embed_dim=8, depth=12, num_heads=2,
        use_abs_pos=False, use_rel_pos=True, window_size=2,
        window_block_indexes=(0, 1, 3, 4, 6, 7, 9, 10),
        decoder_embed_dim=3, dual_view=True,
    )
    enable_structure_first(model, {"hidden_dim": 16, "rgb_hidden": 8})
    images, target, _, _ = sample_inputs()
    output = structure_forward(
        model, images, target, "coarse", has_jt=torch.tensor([True, False]),
    )
    output["p0"].square().mean().backward()
    assert_nonzero_grad(model.structure_adapter.parameters())
    assert_no_grad(model.blocks.parameters())
    model.zero_grad(set_to_none=True)
    set_structure_stage(model, "refine")
    output = structure_forward(
        model, images, target, "refine", bf_mask=bf_mask(),
    )
    output["p1"].square().mean().backward()
    assert_nonzero_grad(model.bf_refiner.adapter.parameters())
    assert_no_grad(value for name, value in model.named_parameters()
                   if not name.startswith("bf_refiner."))


def test_predicted_s_is_effective_after_conditioner_learning_steps():
    """Hold Q fixed and verify learned responses depend on predicted S."""
    model = new_model().eval()
    images, target, _, _ = sample_inputs()
    head = structure_forward(model, images, target, "coarse")["head"]
    head = {key: value.detach() for key, value in head.items()}
    lower = torch.randn(2, 4, 4, 32)
    reference = torch.randn_like(lower)
    desired = torch.randn_like(lower)
    conditioner = model.structure_adapter
    original = conditioner(lower, reference, head)
    assert original.count_nonzero() == 0
    changed = {key: value.clone() for key, value in head.items()}
    changed["occupancy_logits"][:, 0] += 4
    changed["existence_logits"][:, 0] += 4
    torch.testing.assert_close(
        conditioner(lower, reference, changed), original, rtol=0, atol=0,
    )
    optimizer = torch.optim.AdamW(conditioner.parameters(), lr=0.01)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = (conditioner(lower, reference, head) - desired).square().mean()
        loss.backward()
        optimizer.step()
    actual = conditioner(lower, reference, head)
    perturbed = conditioner(lower, reference, changed)
    assert (actual - perturbed).abs().max().item() > 1e-7
    # Q is bitwise identical: this difference must come from predicted S.
    torch.testing.assert_close(changed["query"], head["query"], rtol=0, atol=0)
    probe = {
        key: value.clone().requires_grad_(True) for key, value in head.items()
    }
    (conditioner(lower, reference, probe) - desired).square().mean().backward()
    for name in ("occupancy_logits", "existence_logits", "class_logits",
                 "slots"):
        assert probe[name].grad is not None
        assert torch.isfinite(probe[name].grad).all()
        assert probe[name].grad.count_nonzero() > 0, name


def test_stage_two_j_updates_only_refiner_and_only_jt_bearing_sample():
    model = new_model().eval()
    set_structure_stage(model, "refine")
    images, target, _, _ = sample_inputs()
    output = structure_forward(
        model, images, target, "refine", bf_mask=bf_mask(),
    )
    output["p1"].retain_grad()
    targets = FIXTURE.instance_targets([True, False])
    jieti, _, shield = model.dual_jieti_loss(
        output["p1"], target, targets["voro"], targets["valid_parts"],
        targets["has_jt"],
    )
    assert torch.isfinite(jieti) and jieti.item() > 0
    assert shield.item() == 0
    jieti.backward()
    assert output["p1"].grad[0].count_nonzero() > 0
    assert output["p1"].grad[1].count_nonzero() == 0
    assert_nonzero_grad(model.bf_refiner.adapter.parameters())
    assert_nonzero_grad(model.bf_refiner.rgb.parameters())
    assert_no_grad(value for name, value in model.named_parameters()
                   if not name.startswith("bf_refiner."))
