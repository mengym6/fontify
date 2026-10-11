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


def small_model(dual=False, source=None):
    torch.manual_seed(13)
    model = fontify_class(source)(
        img_size=(16, 8), patch_size=2, embed_dim=8, depth=12,
        num_heads=1, use_abs_pos=False, decoder_embed_dim=3, dual_view=dual,
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


def test_zero_fusion_matches_both_original_predictions():
    original = small_model()
    dual = small_model(dual=True)
    # Only old keys are copied, mimicking migration from checkpoint-14.
    dual.load_state_dict(original.state_dict(), strict=False)
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
    assert dual.dual_fusion.projection.in_features == 532
    assert dual.dual_fusion.projection.out_features == 32


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
def test_individual_loss_gradient_routing(active):
    model = small_model(dual=True)
    nn.init.normal_(model.dual_fusion.projection.weight, std=0.03)
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
        "L_H": {"encoder": any(active), "decoder": False,
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


def test_cpu_ddp_static_graph_across_pool_mixtures():
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
            model = small_model(dual=True)
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
