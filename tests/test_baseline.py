"""Baseline data, inference and model contracts on CPU."""

import ast
import copy
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from util.stage3_data import audit, fixed_sample, read_manifest
from util.font_metrics import metrics
from util.checkpoint import load_baseline_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def test_scaler_clips_a_parameter_generator():
    tree = ast.parse((ROOT / "util/misc.py").read_text())
    names = {"NativeScalerWithGradNormCount", "get_grad_norm_"}
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    namespace = {"torch": torch, "inf": float("inf")}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "scaler", "exec"), namespace)
    scaler = namespace["NativeScalerWithGradNormCount"]()
    model = nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()
    loss = model(torch.ones(1, 2)).sum() * 100
    scaler(loss, optimizer, clip_grad=3, parameters=model.parameters())
    assert scaler.last_unclipped_norm > 3
    assert scaler.last_clipped_norm == pytest.approx(3, abs=1e-5)
    assert (model.weight.detach() - before).norm() == pytest.approx(0.3, abs=1e-5)


def source_method(name, namespace):
    """Execute the real method without importing optional detectron2."""
    tree = ast.parse((ROOT / "models_train.py").read_text())
    fontify = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Fontify"
    )
    node = next(
        n for n in fontify.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    module = ast.Module(body=[copy.deepcopy(node)], type_ignores=[])
    # Only repository AST nodes are executed, never user-supplied code.
    exec(compile(module, str(ROOT / "models_train.py"), "exec"), namespace)  # noqa: S102
    return namespace[name]


class Patch(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, 4).permute(0, 2, 3, 1)


def small_encoder():
    model = nn.Module()
    model.patch_embed = Patch()
    model.patch_size = 4
    model.depth = 12
    model.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])
    model.norm = nn.Identity()
    model.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 3))
    model.segment_token_x = nn.Parameter(torch.zeros(1, 1, 1, 3))
    model.segment_token_y = nn.Parameter(torch.zeros(1, 1, 1, 3))
    model.pos_embed = None
    model.forward_encoder = types.MethodType(
        source_method("forward_encoder", {"torch": torch}), model
    )
    return model


def test_metrics_identity():
    image = torch.randn(1, 3, 32, 32)
    result = metrics(image, image)
    assert result["edge_f1"] == pytest.approx(1)
    assert all(v == 0 for k, v in result.items() if k != "edge_f1")


def test_encoder_masks_query_target():
    model = small_encoder()
    source = torch.randn(2, 3, 64, 32)
    target = torch.randn_like(source)
    mask = torch.zeros(2, 16, 8, dtype=torch.bool)
    mask[:, 8:] = True
    expected = model.forward_encoder(source, target, mask.flatten(1))
    target[:, :, 32:] += 100
    actual = model.forward_encoder(source, target, mask.flatten(1))
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_checkpoint_rejects_removed_film_and_missing_weights():
    model = nn.Linear(3, 2)
    checkpoint = {"model": model.state_dict()}
    load_baseline_checkpoint(model, checkpoint)
    checkpoint["stage3_config"] = {"style_mode": "reference"}
    with pytest.raises(ValueError, match="FiLM"):
        load_baseline_checkpoint(model, checkpoint)
    with pytest.raises(RuntimeError):
        load_baseline_checkpoint(model, {"model": {}})


@pytest.mark.parametrize("no_gan", [True, False])
def test_baseline_loss_and_backward(no_gan):
    """Run the actual loss with deterministic VGG/edge substitutes on CPU."""
    class SmallLoss:
        patch_size = 1
        loss_func = "smoothl1"
        _dbg_step = 1

        def unpatchify(self, value):
            return value.transpose(1, 2).reshape(-1, 3, 16, 8)

        def get_dynamic_loss_weights(self, epoch):
            return 0.4, 0.3

        def improved_edge_detection(self, value):
            return value

        def vgg_loss(self, prediction, target):
            return (prediction - target).abs().mean()

        def resize(self, value):
            return value

        def discriminator(self, value):
            return value.mean((1, 2, 3), keepdim=False).unsqueeze(1)

    model = SmallLoss()
    forward = types.MethodType(source_method(
        "forward_loss", {"torch": torch, "F": F, "transforms": transforms}
    ), model)
    torch.manual_seed(3)
    pred = torch.rand(2, 3, 16, 8, requires_grad=True)
    target = torch.rand_like(pred) + 1
    mask = torch.zeros(2, 128)
    mask[:, -16:] = 1
    loss, recon, style, edge, adv = forward(
        pred, pred, target, mask, torch.ones_like(pred), no_gan=no_gan
    )
    torch.testing.assert_close(edge, F.l1_loss(pred, target))
    expected = recon + style + 0.3 * edge
    if not no_gan:
        logits = model.discriminator(pred).squeeze(1)
        torch.testing.assert_close(adv, F.binary_cross_entropy_with_logits(
            logits, torch.ones_like(logits)
        ))
        expected += 0.4 * adv
    else:
        assert adv.item() == 0
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0


def make_rows(tmp_path):
    rows = {}
    for i, split in enumerate(("train", "val_seen")):
        rows[split] = []
        for j in range(2):
            name = f"{split}-{j}.png"
            Image.new("RGB", (8, 8), (i * 40 + j, 0, 0)).save(tmp_path / name)
            rows[split].append(
                {
                    "image_path": name,
                    "target_path": name,
                    "type": "JT",
                    "style_id": "seen",
                    "character": f"char-{i}-{j}",
                    "glyph_id": name,
                }
            )
    return rows


def test_audit_leakage_and_identity(tmp_path):
    rows = make_rows(tmp_path)
    assert audit(rows, tmp_path)["train"]["records"] == 2
    original = copy.deepcopy(rows)
    rows["val_seen"][0]["glyph_id"] = rows["train"][0]["glyph_id"]
    with pytest.raises(ValueError, match="leakage"):
        audit(rows, tmp_path)
    rows = original
    del rows["train"][0]["style_id"]
    with pytest.raises(ValueError):
        audit(rows, tmp_path)


def test_fixed_sample_hides_query_gt(tmp_path):
    rows = make_rows(tmp_path)["train"]
    _, context, truth, mask = fixed_sample(tmp_path, rows[0], rows[1])
    assert mask[28:].all() and not mask[:28].any()
    assert not torch.equal(context[:, 448:], truth[:, 448:])


def test_manifest_splits(tmp_path):
    rows = make_rows(tmp_path)
    for split, records in rows.items():
        (tmp_path / f"{split}.json").write_text(json.dumps(records))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"train": ["train.json"], "val_seen": ["val_seen.json"]})
    )
    rows, paths = read_manifest(manifest_path)
    assert set(rows) == set(paths) == {"train", "val_seen"}
    rows["val_seen"][0]["style_id"] = "other"
    with pytest.raises(ValueError, match="val_seen styles"):
        audit(rows, tmp_path)
    manifest_path.write_text(json.dumps({"train": ["train.json"], "val_seen": []}))
    with pytest.raises(ValueError, match="Empty split: val_seen"):
        read_manifest(manifest_path)
    manifest_path.write_text(
        json.dumps(
            {
                "train": ["train.json"],
                "val_seen": ["val_seen.json"],
                "val_unseen": ["val_seen.json"],
            }
        )
    )
    with pytest.raises(ValueError, match="Unknown manifest splits"):
        read_manifest(manifest_path)


def test_build_stage3_manifest_passes_audit(tmp_path):
    import numpy as np

    characters = "一二三四五六七八九十"
    source = tmp_path / "ttf/SourceHanSansSC-Regular"
    source.mkdir(parents=True)
    for character in characters + "零":
        Image.new("RGB", (8, 8), (0, 0, 0)).save(source / f"{character}.png")
    unique = 0
    for writer, offset in (("A", 0), ("B", 2)):
        for role in ("BF", "JT"):
            folder = tmp_path / f"font/train/new/{writer}{role}"
            (folder / "images_text_denoised").mkdir(parents=True)
            (folder / "semantic_masks").mkdir()
            names = list(characters[offset : offset + 8]) + ["一1", "零"]
            for name in names:
                unique += 1
                Image.new("RGB", (8, 8), (unique, 0, 0)).save(
                    folder / f"images_text_denoised/{name}.png"
                )
                if name != "零":
                    np.save(folder / f"semantic_masks/{name}.npy", np.zeros((1, 8, 8)))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/build_stage3_manifest.py"),
            "--data-root",
            str(tmp_path),
            "--val-ratio",
            "0.4",
            "--min-val-characters",
            "2",
            "--output",
            str(tmp_path / "stage3_json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    rows, _ = read_manifest(tmp_path / "stage3_json/manifest.json")
    report = audit(rows, tmp_path)
    assert set(rows) == {"train", "val_seen"}
    assert report["train"]["styles"] == report["val_seen"]["styles"] == 2
    train_chars = {r["character"] for r in rows["train"]}
    val_chars = {r["character"] for r in rows["val_seen"]}
    assert not train_chars & val_chars and "零" not in train_chars | val_chars
    # 同字的 BF/JT、重复书写版本以及其他书家的同字必须落在同一划分。
    for split in rows.values():
        one = [r["glyph_id"] for r in split if r["character"] == "一"]
        assert len(one) in (0, 6)
    summary = json.loads((tmp_path / "stage3_json/split_summary.json").read_text())
    assert len(summary["dropped"]) == 4
    assert all(
        row["val_seen"]["characters"] >= 2 for row in summary["writers"].values()
    )


def test_pairdataset_enforces_style_and_character(tmp_path):
    from data import pair_transforms
    from data.pairdataset import PairDataset
    from util.masking_generator import MaskingGenerator

    records = []
    for style in range(2):
        for character in range(2):
            name = f"s{style}-c{character}.png"
            Image.new("RGB", (8, 8), (40 + 120 * style + character, 0, 0)).save(
                tmp_path / name
            )
            records.append(
                {
                    "image_path": name,
                    "target_path": name,
                    "style_id": str(style),
                    "character": str(character),
                    "type": "BF",
                }
            )
    path = tmp_path / "train.json"
    path.write_text(json.dumps(records))
    transform = pair_transforms.Compose([pair_transforms.ToTensor()])
    dataset = PairDataset(
        str(tmp_path),
        [str(path)],
        transform=transform,
        transform2=transform,
        masked_position_generator=MaskingGenerator((2, 1), num_masking_patches=1),
        half_mask_ratio=1.0,
        strict_style_pairing=True,
    )
    for index in range(4):
        _, target, _, _ = dataset[index]
        top, bottom = target[0, :8].mean(), target[0, 8:].mean()
        assert abs((top - bottom).item()) == pytest.approx(1 / 255, abs=1e-6)


@pytest.mark.parametrize("source_dataset", [None, "calliphase"])
def test_no_jt_uses_random_mask_for_jt_and_keeps_bf_semantic(
    tmp_path, source_dataset
):
    from data import pair_transforms
    from data.pairdataset import PairDataset
    from util.masking_generator import MaskingGenerator

    records = []
    for pair_type in ("JT", "BF"):
        name = f"{pair_type}.png"
        Image.new("RGB", (16, 32), (255, 255, 255)).save(tmp_path / name)
        np.save(tmp_path / f"{pair_type}.npy", np.ones((1, 32, 16)))
        pair = {
            "image_path": name,
            "target_path": name,
            "semantic_mask_path": f"{pair_type}.npy",
            "type": pair_type,
        }
        if source_dataset is not None:
            pair["source_dataset"] = source_dataset
        records.append(pair)
    json_path = tmp_path / "train.json"
    json_path.write_text(json.dumps(records))
    transform = pair_transforms.Compose([pair_transforms.ToTensor()])
    dataset = PairDataset(
        str(tmp_path),
        [str(json_path)],
        transform=transform,
        masked_position_generator=MaskingGenerator(
            (2, 1), num_masking_patches=1
        ),
        use_two_pairs=False,
        half_mask_ratio=0.0,
        no_jt=True,
    )
    _, _, jt_mask, _ = dataset[0]
    _, _, bf_mask, _ = dataset[1]
    assert jt_mask.shape == (2, 1) and jt_mask.sum() == 1
    assert bf_mask.shape == (2, 1) and bf_mask.sum() == 2

    dataset.half_mask_ratio = 1.0
    _, _, jt_mask, _ = dataset[0]
    assert jt_mask.sum() == 1

    dataset.no_jt = False
    dataset.half_mask_ratio = 0.0
    _, _, jt_semantic_mask, _ = dataset[0]
    assert jt_semantic_mask.sum() == 2


def test_no_jt_mask_mix_keeps_jt_sample_without_annotation(tmp_path):
    from data import pair_transforms
    from data.pairdataset import PairDataset
    from util.masking_generator import MaskingGenerator

    records = []
    for pair_type, color in (("JT", (255, 0, 0)), ("BF", (0, 255, 0))):
        name = f"{pair_type}.png"
        Image.new("RGB", (16, 32), color).save(tmp_path / name)
        records.append({
            "image_path": name,
            "target_path": name,
            "type": pair_type,
        })
    json_path = tmp_path / "train.json"
    json_path.write_text(json.dumps(records))
    dataset = PairDataset(
        str(tmp_path),
        [str(json_path)],
        transform=pair_transforms.Compose([pair_transforms.ToTensor()]),
        masked_position_generator=MaskingGenerator(
            (2, 1), num_masking_patches=1
        ),
        use_two_pairs=False,
        mask_mix_probs=[0.0, 1.0, 0.0],
        no_jt=True,
    )
    _, target, mask, _ = dataset[1]
    assert target[0].mean().item() == pytest.approx(1.0)
    assert target[1].mean().item() == pytest.approx(0.0)
    assert mask.sum() == 1
