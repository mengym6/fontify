"""Regression for padded BF masks silently paired with stretched image GT."""

import json

import numpy as np
import pytest
from PIL import Image
from test_dual_data import RandomMask, write_records

from data.dual_dataset import DualDataset
from tools.build_dual_manifest import merge_records
from tools.build_dual_targets import build_cache
from tools.repair_calliphase_geometry import SCHEMA, install, sha256, verify
from tools.verify_calliphase_training_data import verify as verify_training
from util.bf_geometry import AUDIT_SCHEMA
from util.dual_targets import resize_native_layer


def native_bf(root, gt_geometry, mask_geometry):
    row = write_records(root, "Native", "BF", ["A"])[0]
    ink = np.zeros((120, 240), dtype=np.uint8)
    ink[10:40, 60:180] = 1
    original = root / "font/train/new/NativeBF/images/A.png"
    Image.fromarray(255 * (1 - ink)).convert("RGB").save(original)
    image = Image.fromarray(ink * 255)
    if gt_geometry == "pad_square":
        padded = Image.new("L", (240, 240), 0)
        padded.paste(image, (0, 60))
        image = padded
    gt = 255 - np.asarray(image.resize((448, 448), Image.Resampling.LANCZOS))
    Image.fromarray(gt).convert("RGB").save(root / row["target_path"])
    np.save(root / row["semantic_mask_path"],
            resize_native_layer(ink, mask_geometry)[None])
    path = root / row["annotation_path"]
    annotation = json.loads(path.read_text())
    annotation["images"][0].update(height=120, width=240)
    for item in annotation["annotations"]:
        item["segmentation"] = [[60, 10, 179, 10, 179, 39, 60, 39]]
    path.write_text(json.dumps(annotation))
    pairs, _ = merge_records([row], root)
    manifest = root / "train.json"
    manifest.write_text(json.dumps(pairs))
    return pairs, manifest


@pytest.mark.parametrize("cache_geometry", ["direct_resize", "pad_square"])
def test_wrong_bf_geometry_is_rejected_without_any_jt_records(tmp_path, cache_geometry):
    pairs, manifest = native_bf(tmp_path, "direct_resize", "pad_square")
    with pytest.raises(ValueError, match="cache failures"):
        build_cache(tmp_path, pairs, tmp_path / "cache", [manifest],
                    native_geometry=cache_geometry)
    metadata = json.loads((tmp_path / "cache/metadata.json").read_text())
    assert metadata["cached_targets"] == 0  # This is precisely the BF-only gap.
    assert metadata["status"] == "failed"
    assert "coordinate audit failed" in metadata["failures"][0]["error"]


def test_aligned_bf_cache_seals_annotations_and_checks_crop_alignment(tmp_path):
    pairs, manifest = native_bf(tmp_path, "pad_square", "pad_square")
    cache = tmp_path / "cache"
    metadata = build_cache(tmp_path, pairs, cache, [manifest])
    audit = metadata["bf_coordinate_audit"]
    assert audit["schema"] == AUDIT_SCHEMA and audit["status"] == "passed"
    assert len(audit["records"]) == 1
    data = DualDataset(tmp_path, [manifest], cache, RandomMask())
    first = data.get_pair(0, 0, 7)
    second = data.get_pair(0, 0, 7)
    np.testing.assert_array_equal(first["mask"].numpy(), second["mask"].numpy())
    annotation = tmp_path / next(iter(audit["records"].values()))["annotation_path"]
    annotation.write_text(annotation.read_text() + " ")
    with pytest.raises(ValueError, match="source changed"):
        DualDataset(tmp_path, [manifest], cache, RandomMask())


@pytest.mark.parametrize("mutation", ["missing", "failed", "wrong_geometry"])
def test_old_or_unverified_bf_cache_cannot_be_used(tmp_path, mutation):
    pairs, manifest = native_bf(tmp_path, "pad_square", "pad_square")
    cache = tmp_path / "cache"
    metadata = build_cache(tmp_path, pairs, cache, [manifest])
    if mutation == "missing":
        del metadata["bf_coordinate_audit"]
    elif mutation == "failed":
        metadata["bf_coordinate_audit"]["status"] = "failed"
    else:
        metadata["bf_coordinate_audit"]["native_geometry"] = "direct_resize"
    (cache / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="BF native geometry audit"):
        DualDataset(tmp_path, [manifest], cache, RandomMask())


def test_installation_preserves_old_bytes_and_checks_all_payloads_first(tmp_path):
    data, staged = tmp_path / "data", tmp_path / "staged"
    data.mkdir()
    staged.mkdir()
    (data / "image.png").write_bytes(b"old")
    (staged / "image.png").write_bytes(b"correct")
    manifest = {"schema": SCHEMA, "native_geometry": "pad_square",
                "files": {"image.png": sha256(staged / "image.png"),
                          "missing.npy": "0" * 64}}
    with pytest.raises(ValueError, match="Staged asset"):
        install(data, manifest, staged, tmp_path / "bad-backup")
    assert (data / "image.png").read_bytes() == b"old"
    del manifest["files"]["missing.npy"]
    result = install(data, manifest, staged, tmp_path / "backup")
    assert result["status"] == "passed"
    assert (tmp_path / "backup/image.png").read_bytes() == b"old"
    assert verify(data, manifest)["status"] == "passed"


def test_installation_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="Invalid relative"):
        verify(tmp_path, {"files": {"../escape": "0" * 64}})


def test_actual_training_verifier_runs_the_real_dataset(tmp_path):
    pairs, manifest = native_bf(tmp_path, "pad_square", "pad_square")
    cache = tmp_path / "cache"
    build_cache(tmp_path, pairs, cache, [manifest])
    result = verify_training(tmp_path, manifest, cache, tmp_path / "verification")
    assert result["status"] == "passed"
    assert result["samples"] == 1
    assert result["pool_counts"] == {"bf_only": 1}
    assert len(result["visuals"]) == 1
