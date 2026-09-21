"""CPU contracts for conditioning, calibration, auditing and legacy losses."""

import ast
import copy
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from util.stage3_calibration import calibrate_measurements
from util.stage3_data import audit, fixed_sample, read_manifest
from util.stage3_runtime import configure, metrics, optimizer_groups
from util.style_conditioning import ReferenceConditioner, enable_conditioning

ROOT = Path(__file__).resolve().parents[1]


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


def test_identity_and_no_gt_leakage():
    torch.manual_seed(3)
    model = small_encoder()
    source, target = torch.randn(2, 3, 64, 32), torch.randn(2, 3, 64, 32)
    mask = torch.zeros(2, 16, 8, dtype=torch.bool)
    mask[:, 8:] = True
    original = model.forward_encoder(source, target, mask.flatten(1))
    enable_conditioning(model, "reference")
    current = model.forward_encoder(source, target, mask.flatten(1))
    for a, b in zip(original, current):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for head in model.style_conditioner.heads:
        nn.init.normal_(head.weight)
    target2 = target.clone()
    target2[:, :, 32:] = torch.randn_like(target2[:, :, 32:]) * 100
    a = model.forward_encoder(source, target, mask.flatten(1))
    b = model.forward_encoder(source, target2, mask.flatten(1))
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    mask[:, :4] = True
    target2 = target.clone()
    target2[:, :, :16] += 100
    a = model.style_conditioner(target, mask, 4)
    b = model.style_conditioner(target2, mask, 4)
    for left, right in zip(a, b):
        for x, y in zip(left, right):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_conditioning_gradients_and_roundtrip(tmp_path):
    module = ReferenceConditioner(6)
    image = torch.randn(2, 3, 64, 32)
    mask = torch.zeros(2, 16, 8)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        loss = sum(
            (scale - 1).square().mean() + shift.square().mean()
            for scale, shift in module(image, mask, 4)
        )
        loss.backward()
        optimizer.step()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in module.encoder.parameters()
    )
    path = tmp_path / "module.pth"
    torch.save(module.state_dict(), path)
    restored = ReferenceConditioner(6)
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    for left, right in zip(module(image, mask, 4), restored(image, mask, 4)):
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b)


def test_constant_control():
    module = ReferenceConditioner(6, "constant")
    for head in module.heads:
        nn.init.normal_(head.weight)
    a = module(torch.randn(2, 3, 64, 32), torch.zeros(2, 16, 8), 4)
    b = module(torch.randn(2, 3, 64, 32), torch.ones(2, 16, 8), 4)
    for left, right in zip(a, b):
        for x, y in zip(left, right):
            torch.testing.assert_close(x, y)


def test_checkpoint_migration_rejects_unrelated_missing_keys(tmp_path, monkeypatch):
    from util.stage3_runtime import load_model

    factory = types.SimpleNamespace(
        vit_base_patch16_input896x448_win_dec64_8glb_sl1=small_encoder
    )
    monkeypatch.setitem(sys.modules, "models_train", factory)
    path = tmp_path / "legacy.pth"
    state = small_encoder().state_dict()
    torch.save({"model": state}, path)
    model, config = load_model(path, {"style_mode": "reference"})
    assert model.style_conditioner.mode == "reference"
    migrated = tmp_path / "stage3.pth"
    torch.save({"model": model.state_dict(), "stage3_config": config}, migrated)
    restored, _ = load_model(migrated)
    assert restored.style_conditioner.mode == "reference"
    del state["segment_token_y"]
    torch.save({"model": state}, path)
    with pytest.raises(ValueError, match="Incompatible checkpoint"):
        load_model(path, {"style_mode": "reference"})


def test_calibration_and_stop_conditions():
    row = {
        "pixel_norms": [1, 2, 4, 8],
        "parameter_gram": [[float(i == j) for j in range(4)] for i in range(4)],
    }
    result = calibrate_measurements([row] * 32)
    assert result["status"] == "ok"
    coefficients = result["structure_coefficients"]
    assert max(c * g for c, g in zip(coefficients, row["pixel_norms"])) == (
        min(c * g for c, g in zip(coefficients, row["pixel_norms"]))
    )
    norm = sum(c * c for c in coefficients) ** 0.5
    assert norm * result["structure_common_scale"] == pytest.approx(2)
    for invalid in (0, float("nan"), 1e-12):
        row["pixel_norms"][0] = invalid
        assert calibrate_measurements([row])["status"] == "blocked"


def test_config_and_optimizer_invariants():
    model = small_encoder()
    enable_conditioning(model, "reference")
    configure(model, {})
    assert model.detail_gradient_ratio == 0.1
    assert model.detail_per_sample_normalize is False
    assert model.structure_coefficients == [1.0] * 4
    with pytest.raises(ValueError):
        configure(model, {"detail_per_sample_normalize": True})
    groups = optimizer_groups(model)
    ids = [id(p) for group in groups for p in group["params"]]
    assert len(ids) == len(set(ids))
    for group in groups:
        for param in group["params"]:
            if any(param is q for q in model.style_conditioner.parameters()):
                assert group["lr"] == 3e-4


def test_metrics_identity():
    image = torch.randn(1, 3, 32, 32)
    result = metrics(image, image)
    assert result["edge_f1"] == pytest.approx(1)
    assert all(v == 0 for k, v in result.items() if k != "edge_f1")


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


def test_audit_cli_and_experiment_generator(tmp_path):
    rows = make_rows(tmp_path)
    manifest = {}
    for split, records in rows.items():
        path = tmp_path / f"{split}.json"
        path.write_text(json.dumps(records))
        manifest[split] = [path.name]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/stage3.py"),
            "audit",
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "report"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "report/audit.json").exists()
    from tools.stage3_experiments import experiments

    candidates = experiments("local", 0, 0.05, None)
    assert len(candidates) == 3
    assert all(candidate[1] == 0 for candidate in candidates)
    assert len(experiments("style", 0.05, 0.05, None)) == 3


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
        semantic_only_epochs=0,
    )
    for index in range(4):
        _, target, _, _ = dataset[index]
        top, bottom = target[0, :8].mean(), target[0, 8:].mean()
        assert abs((top - bottom).item()) == pytest.approx(1 / 255, abs=1e-6)


def test_real_loss_coefficients_and_experiment2_formula():
    """Exercise actual forward_loss with a small deterministic VGG substitute."""
    tree = ast.parse((ROOT / "models_train.py").read_text())
    namespace = {"torch": torch, "F": F, "transforms": transforms}
    helpers = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("rgb_to_gray", "gaussian_blur", "highpass", "sobel_gradients")
    ]
    exec(  # noqa: S102
        compile(
            ast.Module(body=helpers, type_ignores=[]),
            "loss_helpers",
            "exec",
        ),
        namespace,
    )

    class SmallLoss:
        patch_size = 1
        loss_func = "smoothl1"
        _dbg_step = 1

        def unpatchify(self, value):
            return value.transpose(1, 2).reshape(-1, 3, 16, 8)

        def get_dynamic_loss_weights(self, epoch):
            return "jt_bf_sync", 0.0, 0.3

        def get_loss_phase(self, epoch):
            return "jt_bf_sync", epoch

        def improved_edge_detection(self, value):
            return value

        def vgg_loss(self, pred, target):
            self.vgg_input = pred.detach()
            return (pred - target).abs().mean()

    model = SmallLoss()
    configure(model, {"vgg_input_mode": "rgb"})
    forward = types.MethodType(source_method("forward_loss", namespace), model)
    pred = (torch.rand(2, 3, 16, 8) - 0.5).requires_grad_()
    target = torch.rand_like(pred)
    mask = torch.ones(2, 128)
    # Avoid the legacy minimum-visible-pixel gate on this tiny fixture.
    mask[:, :112] = 0
    target = target + 1
    forward(
        pred,
        pred,
        target,
        mask,
        torch.ones_like(pred),
        epoch=35,
        no_gan=True,
        keep_loss_graph=True,
    )
    graph = model.last_loss_graph
    expected = transforms.Resize((224, 224))(pred).float()
    mean = pred.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = pred.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    torch.testing.assert_close(model.vgg_input, expected * std + mean)
    torch.testing.assert_close(
        graph["detail"], graph["highpass"] + 0.1 * graph["gradient"]
    )
    original = graph["structure"].detach()
    configure(model, {"structure_coefficients": [2, 2, 2, 2]})
    forward(
        pred,
        pred,
        target,
        mask,
        torch.ones_like(pred),
        epoch=35,
        no_gan=True,
        keep_loss_graph=True,
    )
    torch.testing.assert_close(model.last_loss_graph["structure"], 2 * original)
    model.last_loss_graph["total_weighted"].backward()
    assert torch.isfinite(pred.grad).all()
