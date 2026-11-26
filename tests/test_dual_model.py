"""Actual Fontify AST contracts without optional detectron2/VGG downloads."""

import ast
import subprocess
import tempfile
import types
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from util.dual_view import DualViewFusion, DualViewHead

ROOT = Path(__file__).resolve().parents[1]


class PatchEmbed(nn.Module):
    def __init__(self, kernel_size, stride, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size, stride)

    def forward(self, image):
        return self.proj(image).permute(0, 2, 3, 1)


class SmallBlock(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.projection = nn.Linear(dim, dim)

    def forward(self, value):
        return value + self.projection(value)


class SmallVgg(nn.Module):
    def forward(self, prediction, target, per_sample=False):
        result = (prediction - target).abs().mean((1, 2, 3))
        return result if per_sample else result.mean()


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, value):
        return F.interpolate(value, self.size, mode="bilinear",
                             align_corners=False)


def fontify_class(source=None, real_blocks=False):
    if source is None:
        source = (ROOT / "models_train.py").read_text()
    tree = ast.parse(source)
    node = next(item for item in tree.body
                if isinstance(item, ast.ClassDef) and item.name == "Fontify")
    namespace = {
        "torch": torch, "nn": nn, "F": F, "Block": SmallBlock,
        "PatchEmbed": PatchEmbed, "Discriminator": lambda: nn.Linear(1, 1),
        "LayerNorm2D": lambda channels: nn.GroupNorm(1, channels),
        "VGGPerceptualLoss": SmallVgg,
        "trunc_normal_": nn.init.trunc_normal_,
        "transforms": types.SimpleNamespace(
            Compose=lambda functions: functions[0], Resize=Resize,
        ),
    }
    nodes = [node]
    if real_blocks:
        from timm.models.layers import DropPath
        from timm.models.vision_transformer import Mlp

        from util.vitdet_utils import (
            add_decomposed_rel_pos,
            window_partition,
            window_unpartition,
        )

        nodes = [item for item in tree.body if isinstance(item, ast.ClassDef)
                 and item.name in {"Attention", "Block", "Fontify"}]
        namespace.update(
            DropPath=DropPath, Mlp=Mlp,
            add_decomposed_rel_pos=add_decomposed_rel_pos,
            window_partition=window_partition,
            window_unpartition=window_unpartition,
        )
    exec(compile(ast.Module(body=nodes, type_ignores=[]),  # noqa: S102
                 str(ROOT / "models_train.py"), "exec"), namespace)
    return namespace["Fontify"]


def small_model(dual=False, source=None, ablation="none", fusion_mode="joint"):
    torch.manual_seed(13)
    model = fontify_class(source)(
        img_size=(16, 8), patch_size=2, embed_dim=8, depth=12,
        num_heads=1, use_abs_pos=False, decoder_embed_dim=3, dual_view=dual,
        dual_ablation=ablation,
        dual_fusion_mode=fusion_mode,
    ) if source is None else fontify_class(source)(
        img_size=(16, 8), patch_size=2, embed_dim=8, depth=12,
        num_heads=1, use_abs_pos=False, decoder_embed_dim=3,
    )
    model.improved_edge_detection = lambda value: value
    model._dbg_step = 1
    if dual:
        model.enable_dual_losses(pool=8)
    return model


def sample_inputs(batch=2):
    torch.manual_seed(31)
    images = torch.rand(batch, 3, 16, 8)
    targets = torch.rand_like(images)
    mask = torch.zeros(batch, 8, 4, dtype=torch.bool)
    mask[:, 2:4] = True
    return images, targets, mask, torch.ones_like(images)


@pytest.mark.parametrize("ablation", ["no_j", "no_head", "no_injection"])
@pytest.mark.parametrize("has_jt", [False, True])
def test_ablation_loss_routes_and_checkpoint_round_trip(ablation, has_jt):
    from util.checkpoint import build_checkpoint_model

    model = small_model(dual=True, ablation=ablation)
    model.enable_dual_losses(jieti_weight=0.7, head_weight=0.8, pool=8)
    images, target, mask, valid = sample_inputs()
    labels = {
        "has_jt": torch.full((2,), has_jt),
        "instance_occupancy": torch.ones(2, 4, 4, 4),
        "instance_valid": torch.full((2, 4), has_jt),
        "instance_classes": torch.zeros(2, 4, dtype=torch.long),
        "voro": torch.zeros(2, 8, 8, dtype=torch.long),
        "valid_parts": torch.zeros(2, 4, dtype=torch.bool),
    }
    output = model(images, target, mask, valid, epoch=18, no_gan=True,
                   dual_targets=labels)
    extra = output[8]
    expected = extra["L_pen"] + 0.5 * extra["L_full"]
    if ablation != "no_j":
        expected = expected + 0.7 * extra["J"]
    if ablation != "no_head":
        expected = expected + 0.8 * extra["L_H"]
    torch.testing.assert_close(output[0], expected)
    assert model.jieti_loss_weight == (0.0 if ablation == "no_j" else 0.7)
    output[0].backward()
    assert model.decoder_embed.weight.grad.abs().sum() > 0
    if ablation == "no_head":
        assert model.dual_head is None and model.dual_fusion is None
        assert extra["L_H"].item() == 0
        assert "head_occupancy_logits" not in extra
    elif ablation == "no_injection":
        assert model.dual_fusion is None
        gradients = [p.grad for p in model.dual_head.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert (sum(g.abs().sum() for g in gradients) > 0).item() == has_jt

    checkpoint = {
        "model": model.state_dict(),
        "args": {
            "model": "tiny", "dual_view": True, "dual_ablation": ablation,
            "dual_lambda_full": 0.5, "jieti_w": model.jieti_loss_weight,
            "dual_head_weight": model.dual_head_weight, "jieti_pool": 8,
            "jieti_w_centroid": 1.0, "jieti_w_logsigma": 1.0,
            "jieti_w_shape": 1.0,
        },
    }
    module = types.SimpleNamespace(tiny=lambda dual_view, dual_ablation:
                                   small_model(dual_view, ablation=dual_ablation))
    restored = build_checkpoint_model(checkpoint, model_module=module)
    model.eval()
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(model.predict_full(images, target),
                                   restored.predict_full(images, target),
                                   rtol=0, atol=0)


def test_no_injection_predictions_are_independent_of_head_parameters():
    model = small_model(dual=True, ablation="no_injection").eval()
    images, target, _, _ = sample_inputs()
    with torch.no_grad():
        before = model.predict_full(images, target)
        for parameter in model.dual_head.parameters():
            parameter.add_(torch.randn_like(parameter))
        after = model.predict_full(images, target)
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_ablation_worker_passes_saved_args_to_native_entry(tmp_path, monkeypatch):
    import argparse
    import json
    import sys
    from tools.run_runb_ablations import train

    captured = []
    # Native get_args_parser parses process argv and returns (args, ds_init).
    # The worker must pass its saved configuration directly to main.
    entry = types.SimpleNamespace(main=lambda args, ds_init: captured.append(args))
    monkeypatch.setitem(sys.modules, "main_train", entry)
    hparams = tmp_path / "hparams.json"
    hparams.write_text(json.dumps({
        "dual_view": True, "dual_pixel_mask": True,
        "epochs": 51, "seed": 0, "lr": 0.00195699,
    }))
    train(argparse.Namespace(hparams=hparams, variant="no_j", output_dir=tmp_path / "run"))
    assert len(captured) == 1
    args = captured[0]
    assert args.dual_ablation == "no_j" and args.epochs == 50
    assert args.batch_size * args.accum_iter * args.world_size == 64
    assert args.save_freq == 10 and args.save_best
    assert args.lr == 0.00195699 and args.seed == 0


def test_off_state_and_values_exactly_match_original():
    source = subprocess.check_output(
        ["git", "show", "d93ba30:models_train.py"], cwd=ROOT, text=True,
    )
    original = small_model(source=source)
    current = small_model()
    assert original.state_dict().keys() == current.state_dict().keys()
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, current.state_dict()[key],
                                   rtol=0, atol=0)
    inputs = sample_inputs()
    old = original(*inputs, no_gan=True)
    new = current(*inputs, no_gan=True)
    for expected, actual in zip(old[:8], new[:8]):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert old[8] is None and new[8] is None


@pytest.mark.parametrize("fusion_mode", ["joint", "split"])
def test_zero_fusion_matches_both_original_predictions(fusion_mode):
    from util.checkpoint import load_fontify_checkpoint

    original = small_model()
    dual = small_model(dual=True, fusion_mode=fusion_mode)
    load_fontify_checkpoint(
        dual, {"model": original.state_dict()}, allow_dual_initialization=True,
    )
    images, target, mask, _ = sample_inputs()
    expected_pen = original.forward_decoder(original.forward_encoder(
        images, target, mask.flatten(1),
    ))
    actual_pen = dual.forward_decoder(dual.forward_encoder(
        images, target, mask.flatten(1),
    ))
    torch.testing.assert_close(expected_pen, actual_pen, rtol=0, atol=0)
    full_mask = dual.full_target_mask(images)
    expected_full = original.forward_decoder(original.forward_encoder(
        images, target, full_mask,
    ))
    actual_full = dual.predict_full(images, target)
    torch.testing.assert_close(expected_full, actual_full, rtol=0, atol=0)
    changed_gt = target.clone()
    changed_gt[:, :, 8:] += 100
    torch.testing.assert_close(actual_full, dual.predict_full(images, changed_gt),
                               rtol=0, atol=0)
    if fusion_mode == "joint":
        assert dual.dual_fusion.projection.in_features == 532
        assert dual.dual_fusion.projection.out_features == 32
    else:
        assert dual.dual_fusion.instance_projection.in_features == 276
        assert dual.dual_fusion.query_projection.in_features == 256
        assert dual.dual_fusion.query_projection.out_features == 32


def test_legacy_full_prediction_uses_the_exact_existing_encoder_decoder():
    model = small_model()
    images, target, _, _ = sample_inputs()
    mask = model.full_target_mask(images)
    expected = model.forward_decoder(model.forward_encoder(images, target, mask))
    actual = model.predict_full(images, target)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    changed_gt = target.clone()
    changed_gt[:, :, 8:] += 100
    torch.testing.assert_close(actual, model.predict_full(images, changed_gt),
                               rtol=0, atol=0)


@pytest.mark.parametrize("active", [[False, False], [True, True], [True, False]])
def test_fusion_routes_layout_but_keeps_fusion_gradients(active):
    torch.manual_seed(7)
    head = DualViewHead(encoder_width=8, width=16, grid_size=2, heads=4)
    fusion = DualViewFusion(encoder_width=8, width=16)
    nn.init.normal_(fusion.projection.weight, std=0.02)
    quadrants = tuple(torch.randn(2, 2, 2, 8) for _ in range(4))
    full = torch.randn(2, 2, 2, 32)
    output = head(quadrants, full)
    layout = output["layout"]
    layout.retain_grad()
    loss = fusion(layout, torch.tensor(active)).square().mean()
    loss.backward()
    assert fusion.projection.weight.grad.abs().sum() > 0
    for index, enabled in enumerate(active):
        assert bool(layout.grad[index].abs().sum() > 0) == enabled


def test_no_jt_dual_forward_touches_all_head_parameters_with_zero_gradients():
    model = small_model(dual=True)
    nn.init.normal_(model.dual_fusion.projection.weight, std=0.01)
    images, target, mask, valid = sample_inputs()
    original_valid = valid.clone()
    targets = {
        "has_jt": torch.zeros(2, dtype=torch.bool),
        "instance_occupancy": torch.zeros(2, 4, 4, 4),
        "instance_valid": torch.zeros(2, 4, dtype=torch.bool),
        "instance_classes": torch.full((2, 4), -1),
        "voro": torch.zeros(2, 8, 8, dtype=torch.long),
        "valid_parts": torch.zeros(2, 4, dtype=torch.bool),
    }
    output = model(images, target, mask, valid, no_gan=True,
                   dual_targets=targets)
    assert len(output) == 9
    assert output[8]["J"].item() == 0
    assert output[8]["L_H"].item() == 0
    torch.testing.assert_close(valid, original_valid, rtol=0, atol=0)
    output[0].backward()
    for name, parameter in model.dual_head.named_parameters():
        assert parameter.grad is not None, name
        assert torch.count_nonzero(parameter.grad) == 0, name
    assert model.dual_fusion.projection.weight.grad.abs().sum() > 0
    assert model.patch_embed.proj.weight.grad.abs().sum() > 0
    assert model.decoder_embed.weight.grad.abs().sum() > 0


def test_production_head_shapes_and_zero_projection():
    head = DualViewHead(encoder_width=8, width=256, grid_size=4)
    quadrants = tuple(torch.randn(1, 4, 4, 8) for _ in range(4))
    output = head(quadrants, torch.randn(1, 4, 4, 32))
    assert output["layout"].shape == (1, 4, 4, 532)
    assert output["occupancy_logits"].shape == (1, 4, 4, 4)
    assert output["existence_logits"].shape == (1, 4)
    assert output["class_logits"].shape == (1, 4, 20)
    fusion = DualViewFusion()
    residual = fusion(output["layout"])
    assert residual.shape == (1, 4, 4, 3072)
    assert torch.count_nonzero(residual) == 0


def instance_targets(active):
    batch = len(active)
    voro = torch.ones(batch, 8, 8, dtype=torch.long)
    voro[:, :, 4:] = 2
    occupancy = torch.zeros(batch, 4, 4, 4)
    occupancy[:, 0, :, :2] = 1
    occupancy[:, 1, :, 2:] = 1
    valid = torch.tensor([[True, True, False, False]]).expand(batch, -1)
    return {
        "has_jt": torch.tensor(active), "voro": voro,
        "instance_occupancy": occupancy, "instance_valid": valid,
        "instance_classes": torch.tensor([[1, 2, -1, -1]]).expand(batch, -1),
        "valid_parts": valid,
    }


@pytest.mark.parametrize("active", [[False, False], [True, True], [True, False]])
@pytest.mark.parametrize("detach_backbone", [False, True])
@pytest.mark.parametrize("fusion_mode", ["joint", "split"])
def test_individual_loss_gradient_routing(active, detach_backbone, fusion_mode):
    model = small_model(dual=True, fusion_mode=fusion_mode)
    model.dual_head_detach_backbone = detach_backbone
    for name, parameter in model.dual_fusion.named_parameters():
        if name.endswith("projection.weight"):
            nn.init.normal_(parameter, std=0.03)
    # Keep the tiny substitute decoder in the soft foreground's active range.
    model.dual_jieti_loss.soft_fg = "sigmoid"
    model.dual_jieti_loss.pred_mass_ratio = 0
    inputs = sample_inputs()
    extra = model(*inputs, no_gan=True, dual_targets=instance_targets(active))[8]
    groups = {
        "encoder": list(model.patch_embed.parameters()),
        "decoder": list(model.decoder_embed.parameters()),
        "fusion": list(model.dual_fusion.parameters()),
        "head": list(model.dual_head.parameters()),
    }
    expectation = {
        "L_pen": {"encoder": True, "decoder": True,
                  "fusion": False, "head": False},
        "L_full": {"encoder": True, "decoder": True,
                   "fusion": True, "head": any(active)},
        "J": {"encoder": any(active), "decoder": any(active),
              "fusion": any(active), "head": any(active)},
        "L_H": {"encoder": any(active) and not detach_backbone, "decoder": False,
                "fusion": False, "head": any(active)},
    }
    for loss_name, expected in expectation.items():
        for group_name, parameters in groups.items():
            gradients = torch.autograd.grad(
                extra[loss_name], parameters, retain_graph=True, allow_unused=True,
            )
            actual = any(gradient is not None and gradient.abs().sum() > 0
                         for gradient in gradients)
            assert bool(actual) == expected[group_name], (loss_name, group_name)


@pytest.mark.parametrize("detach_backbone", [False, True])
@pytest.mark.parametrize("fusion_mode", ["joint", "split"])
def test_cpu_ddp_static_graph_across_pool_mixtures(detach_backbone, fusion_mode):
    """Small actual-method contract; this does not replace full ViT GPU smoke."""
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
            model = small_model(dual=True, fusion_mode=fusion_mode)
            model.dual_head_detach_backbone = detach_backbone
            wrapped = nn.parallel.DistributedDataParallel(model, static_graph=True)
            optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.01)
            for active in ([False, False], [True, True], [True, False]):
                optimizer.zero_grad(set_to_none=True)
                output = wrapped(*sample_inputs(), no_gan=True,
                                 dual_targets=instance_targets(active))
                output[0].backward()
                optimizer.step()
        finally:
            torch.distributed.destroy_process_group()


def test_actual_attention_and_blocks_small_grid_forward_backward():
    """Real transformer code; tiny dimensions/VGG substitute, CPU only."""
    torch.manual_seed(8)
    model = fontify_class(real_blocks=True)(
        img_size=(16, 8), patch_size=2, embed_dim=8, depth=12, num_heads=2,
        use_abs_pos=False, use_rel_pos=True, window_size=2,
        window_block_indexes=(0, 1, 3, 4, 6, 7, 9, 10),
        decoder_embed_dim=3, dual_view=True,
    )
    model.enable_dual_losses(pool=8)
    model._dbg_step = 1
    model.improved_edge_detection = lambda value: value
    output = model(*sample_inputs(), no_gan=True,
                   dual_targets=instance_targets([True, False]))
    output[0].backward()
    assert torch.isfinite(output[0])
    assert model.blocks[0].attn.qkv.weight.grad.abs().sum() > 0
    assert model.blocks[2].attn.rel_pos_h.grad.abs().sum() > 0
    assert model.dual_head.existence.weight.grad.abs().sum() > 0


def test_split_layout_keeps_existence_probability_amplitude():
    torch.manual_seed(7)
    head = DualViewHead(encoder_width=8, width=8, grid_size=4, classes=3,
                        normalize_instance_features=True)
    quadrants = tuple(torch.randn(1, 4, 4, 8) for _ in range(4))
    features = torch.randn(1, 4, 4, 32)
    with torch.no_grad():
        head.existence.weight.zero_()
        head.existence.bias.fill_(torch.logit(torch.tensor(0.8)))
        high = head(quadrants, features)
        head.existence.bias.fill_(torch.logit(torch.tensor(0.2)))
        low = head(quadrants, features)
    torch.testing.assert_close(low["layout"][..., :11],
                               high["layout"][..., :11] * 0.25)
    torch.testing.assert_close(low["query"], high["query"], rtol=0, atol=0)


def test_split_fusion_checkpoint_round_trip_and_rng_alignment():
    from util.checkpoint import build_checkpoint_model

    model = small_model(dual=True, fusion_mode="joint")
    joint_rng = torch.get_rng_state().clone()
    split = small_model(dual=True, fusion_mode="split")
    torch.testing.assert_close(torch.get_rng_state(), joint_rng, rtol=0, atol=0)
    for key, value in model.dual_head.state_dict().items():
        torch.testing.assert_close(value, split.dual_head.state_dict()[key],
                                   rtol=0, atol=0)
    nn.init.normal_(split.dual_fusion.instance_projection.weight, std=0.02)
    nn.init.normal_(split.dual_fusion.query_projection.weight, std=0.02)
    checkpoint = {"model": split.state_dict(), "args": {
        "model": "tiny", "dual_view": True, "dual_fusion_mode": "split",
        "dual_lambda_full": 0.5, "jieti_w": 1.0, "dual_head_weight": 1.0,
        "jieti_pool": 8, "jieti_w_centroid": 1.0,
        "jieti_w_logsigma": 1.0, "jieti_w_shape": 1.0,
    }}
    module = types.SimpleNamespace(
        tiny=lambda dual_view, dual_fusion_mode: small_model(
            dual=dual_view, fusion_mode=dual_fusion_mode,
        )
    )
    restored = build_checkpoint_model(checkpoint, model_module=module)
    split.eval()
    restored.eval()
    image, target, _, _ = sample_inputs()
    with torch.no_grad():
        torch.testing.assert_close(split.predict_full(image, target),
                                   restored.predict_full(image, target),
                                   rtol=0, atol=0)
