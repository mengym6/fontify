"""Run B data contracts using small synthetic COCO files and actual CPU torch."""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from data.dual_dataset import DualDataset
from tools.build_dual_manifest import merge_records
from tools.build_dual_targets import build_cache
from util.dual_targets import (
    CACHE_SCHEMA,
    SOURCE_SEAL,
    build_instance_targets,
    coordinate_audit,
    deduplicate_instances,
    load_cached_targets,
    pool_weights,
    record_source_paths,
    transform_layers,
)
from util.jieti_partition import build_partition, ink_label_map


class RandomMask:
    def __call__(self):
        mask = np.zeros((56, 28), dtype=np.int64)
        mask[:, ::2] = 1
        return mask


def write_records(root, author, kind, letters):
    """Build native/GT images, individual BF masks and polygon COCO labels."""
    folder = Path("font/train/new") / f"{author}{kind}"
    for name in ("images", "images_text_denoised", "semantic_masks", "annotations"):
        (root / folder / name).mkdir(parents=True, exist_ok=True)
    images, annotations, records = [], [], []
    for index, letter in enumerate(letters):
        left = 80 + index * 5
        text = np.zeros((448, 448), dtype=np.uint8)
        text[80:180, left : left + 100] = 1
        canvas = (255 * (1 - text)).astype(np.uint8)
        filename = f"{letter}.png"
        Image.fromarray(canvas).convert("RGB").save(root / folder / "images" / filename)
        Image.fromarray(canvas).convert("RGB").save(
            root / folder / "images_text_denoised" / filename
        )
        source = Path("ttf/source") / filename
        (root / source).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(canvas).convert("RGB").save(root / source)
        np.save(root / folder / "semantic_masks" / f"{letter}.npy", text[None])
        images.append(
            {"id": index + 1, "file_name": filename, "width": 448, "height": 448}
        )
        polygon = [[left, 80, left + 99, 80, left + 99, 179, left, 179]]
        annotations.extend(
            [
                {
                    "id": index * 2,
                    "image_id": index + 1,
                    "category_id": 1,
                    "segmentation": polygon,
                },
                {
                    "id": index * 2 + 1,
                    "image_id": index + 1,
                    "category_id": 2,
                    "segmentation": polygon,
                },
            ]
        )
        records.append(
            {
                "image_path": str(source),
                "target_path": str(folder / "images_text_denoised" / filename),
                "type": f"font_{author}{kind}",
                "font_folder": f"{author}{kind}",
                "annotation_path": str(folder / "annotations/instances_default.json"),
                "semantic_mask_path": str(folder / "semantic_masks" / f"{letter}.npy"),
            }
        )
    annotation_path = root / folder / "annotations/instances_default.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "text"}, {"id": 2, "name": "左"}],
            }
        ),
        encoding="utf-8",
    )
    return records


@pytest.fixture
def synthetic_data(tmp_path):
    bf = write_records(tmp_path, "Writer", "BF", ["A", "B", "C", "D"])
    jt = write_records(tmp_path, "Writer", "JT", ["A", "B", "E", "F"])
    # Make E/F originals distinct from C/D even though polygon positions coincide.
    for letter in ("E", "F"):
        raw = tmp_path / "font/train/new/WriterJT/images" / f"{letter}.png"
        image = np.array(Image.open(raw))
        image[0, 0] = 123
        Image.fromarray(image).save(raw)
    merged, summary = merge_records(bf + jt, tmp_path)
    manifest = tmp_path / "train_dual.json"
    manifest.write_text(json.dumps(merged), encoding="utf-8")
    val = tmp_path / "val.json"
    val.write_text(json.dumps(bf + jt), encoding="utf-8")
    fixed = tmp_path / "fixed.json"
    fixed.write_text(
        json.dumps({r["target_path"]: r for r in bf + jt}), encoding="utf-8"
    )
    cache = tmp_path / "cache"
    build_cache(tmp_path, merged + bf + jt, cache, [manifest, val], fixed)
    return tmp_path, merged, summary, manifest, val, fixed, cache


def dataset(data, **kwargs):
    root, _, _, manifest, _, _, cache = data
    return DualDataset(root, [manifest], cache, RandomMask(), **kwargs)


def test_manifest_merges_bf_target_and_preserves_originals(synthetic_data):
    root, merged, summary, _, _, _, _ = synthetic_data
    assert summary["merged_records"] == 6
    assert summary["pools"] == {"dual": 2, "bf_only": 2, "jt_only": 2}
    dual = [p for p in merged if p["pool_type"] == "dual"]
    assert all("WriterBF/images_text_denoised" in p["target_path"] for p in dual)
    assert all("WriterJT/annotations" in p["jt_annotation_path"] for p in dual)
    assert (root / dual[0]["original_path"]).is_file()


def test_scheme_d_pool_mass():
    rows = [{"calligrapher": "A", "pool_type": "dual"}] * 9
    rows += [{"calligrapher": "A", "pool_type": "jt_only"}] * 4
    weights = pool_weights(rows)
    assert sum(weights[:9]) == pytest.approx(3)
    assert sum(weights[9:]) == pytest.approx(2)


def test_deduplicate_by_instance_before_text_intersection():
    region = np.zeros((20, 20), dtype=bool)
    region[3:17, 3:17] = 1
    regions, classes, ids, removed = deduplicate_instances(
        [region, region.copy(), ~region], [0, 1, 2], [5, 6, 7]
    )
    assert len(regions) == 2 and classes == [0, 2] and ids == [5, 7]
    assert removed == [{"annotation_id": 6, "kept_annotation_id": 5}]


def test_occupancy_ink_intersection_and_exact_partition():
    regions = np.zeros((4, 448, 448), dtype=np.uint8)
    regions[0, 0:64, 0:64] = 1
    regions[1, 384:, 384:] = 1
    text = regions.any(0).astype(np.uint8)
    text[:16, :16] = 0
    targets = build_instance_targets(regions, text, [0, 1, -1, -1])
    assert not targets["instance_occupancy"][0, 0, 0]
    assert targets["instance_occupancy"][0, 1, 1]
    assert targets["instance_valid"].tolist() == [True, True, False, False]
    assert targets["voro"][0, 0] == 1
    assert targets["voro"][-1, -1] == 2
    assert targets["voro"].shape == (224, 224)


def test_overlapping_instance_ink_remains_separate():
    regions = np.zeros((4, 448, 448), dtype=np.uint8)
    regions[0, 100:200, 100:200] = 1
    regions[1, 150:250, 150:250] = 1
    targets = build_instance_targets(regions, regions.any(0), [0, 1, -1, -1])
    assert targets["instance_occupancy"][0, 10, 10]
    assert targets["instance_occupancy"][1, 10, 10]
    assert set(np.unique(targets["voro"])) == {1, 2}


def test_nested_inner_outer_uses_existing_later_layer_ownership():
    regions = np.zeros((4, 448, 448), dtype=np.uint8)
    regions[0, 60:390, 60:390] = 1
    regions[1, 140:300, 140:300] = 1
    text = regions.any(0)
    targets = build_instance_targets(regions, text, [19, 18, -1, -1], pool=28)
    labels = ink_label_map(regions & text[None], 4)
    expected = build_partition(labels, targets["valid_parts"], 4, 28)
    assert np.array_equal(targets["voro"], expected)
    assert set(np.unique(targets["voro"])) == {1, 2}
    assert targets["voro"][14, 14] == 2
    # H keeps overlapping occupancy for both inner and outer instances.
    assert targets["instance_occupancy"][0, 14, 14] == 1
    assert targets["instance_occupancy"][1, 14, 14] == 1
    assert not targets["j_downsample_lost"].any()
    assert not targets["j_partition_lost"].any()


def test_nearest_grid_ink_loss_is_audited_without_changing_head_validity():
    from util.dual_loss import DualJietiLoss, InstanceHeadLoss

    regions = np.zeros((4, 448, 448), dtype=np.uint8)
    regions[0, 1, :] = 1
    targets = build_instance_targets(regions, regions.any(0), [0, -1, -1, -1])
    assert targets["instance_valid"][0]
    assert targets["instance_occupancy"][0].any()
    assert targets["j_downsample_lost"].tolist() == [True, False, False, False]
    assert not targets["j_partition_lost"].any()
    assert not targets["voro"].any()
    tensors = {k: torch.from_numpy(v)[None] for k, v in targets.items()}
    active = torch.tensor([True])
    image = torch.ones(1, 3, 896, 448, requires_grad=True)
    j, _, _ = DualJietiLoss()(
        image, image.detach(), tensors["voro"], tensors["valid_parts"], active
    )
    assert torch.isfinite(j) and j.item() == 0
    head = {
        "occupancy_logits": torch.zeros(1, 4, 28, 28, requires_grad=True),
        "existence_logits": torch.zeros(1, 4, requires_grad=True),
        "class_logits": torch.zeros(1, 4, 20, requires_grad=True),
    }
    loss, _ = InstanceHeadLoss()(
        head,
        tensors["instance_occupancy"],
        tensors["instance_valid"],
        tensors["instance_classes"],
        active,
    )
    assert torch.isfinite(loss) and loss.item() > 0
    (loss + j).backward()
    assert head["occupancy_logits"].grad.abs().sum() > 0
    assert torch.isfinite(image.grad).all()


def test_layers_use_identical_crop_and_nearest_interpolation():
    layers = np.zeros((3, 448, 448), dtype=np.uint8)
    layers[:, 80:180, 100:200] = 1
    transformed = transform_layers(layers, (40, 30, 300, 320))
    assert np.array_equal(transformed[0], transformed[1])
    assert np.array_equal(transformed[1], transformed[2])
    assert set(np.unique(transformed)) == {0, 1}


def test_pairing_pool_mask_and_no_jt_zero_supervision(synthetic_data):
    data = dataset(synthetic_data)
    for pool_type in ("dual", "bf_only", "jt_only"):
        index = next(i for i, p in enumerate(data.pairs) if p["pool_type"] == pool_type)
        sample = data.get_pair(index, augmentation_seed=31)
        top = next(
            p for p in data.pairs if p["target_path"] == sample["reference_path"]
        )
        assert top["pool_type"] == pool_type
        assert sample["image"].shape == sample["target"].shape == (3, 896, 448)
        assert sample["mask"].shape == (56, 28)
        assert sample["instance_occupancy"].shape == (4, 28, 28)
        assert sample["valid_parts"].shape == (4,)
        if pool_type == "bf_only":
            assert not sample["has_jt"]
            assert not sample["instance_occupancy"].any()
            assert not sample["instance_valid"].any()
            assert not sample["voro"].any()
        if pool_type == "jt_only":
            assert torch.equal(sample["mask"], torch.from_numpy(RandomMask()()))
        else:
            assert sample["mask"].sum() < 784


def test_ref_cannot_cross_pool(synthetic_data):
    data = dataset(synthetic_data)
    indices = {
        kind: next(i for i, p in enumerate(data.pairs) if p["pool_type"] == kind)
        for kind in ("dual", "bf_only")
    }
    with pytest.raises(ValueError, match="may not cross"):
        data.get_pair(indices["dual"], ref_index=indices["bf_only"])


def test_synchronous_crop_aligns_source_gt_and_occupancy(synthetic_data):
    data = dataset(synthetic_data, crop_scale=(0.6, 0.6), crop_ratio=(1, 1))
    index = next(i for i, p in enumerate(data.pairs) if p["pool_type"] == "dual")
    sample = data.get_pair(index, ref_index=index, augmentation_seed=17)
    assert torch.equal(sample["image"], sample["target"]) is False  # bicubic vs nearest
    std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
    mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
    gray = (sample["target"][:, 448:] * std + mean).mean(0)
    ink = gray < 0.5
    expected = ink.reshape(28, 16, 28, 16).any((1, 3)).float()
    assert torch.equal(sample["instance_occupancy"][0], expected)
    assert torch.equal(sample["target"][:, :448], sample["target"][:, 448:])


def test_seeded_probe_does_not_advance_rng(synthetic_data):
    data = dataset(synthetic_data)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    a = data.get_pair(0, augmentation_seed=99)
    b = data.get_pair(0, augmentation_seed=99)
    assert torch.equal(a["target"], b["target"])
    assert torch.equal(a["mask"], b["mask"])
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_validation_keeps_unmerged_fixed_pairs_and_full_lower_mask(synthetic_data):
    root, _, _, _, val, fixed, cache = synthetic_data
    data = DualDataset(
        root, [val], cache, RandomMask(), is_train=False, fixed_pair_path=fixed
    )
    assert len(data) == 8
    kinds = Counter(p["pool_type"] for p in data.pairs)
    assert kinds == {"bf_only": 4, "jt_only": 4}
    sample = data[0]
    assert not sample["mask"][:28].any()
    assert sample["mask"][28:].all()
    assert sample["geometry_crop"].tolist() == [0, 0, 448, 448]


def test_cache_hash_and_source_drift_are_rejected(synthetic_data):
    root, merged, _, _, _, _, cache = synthetic_data
    pair = next(p for p in merged if p["has_jt"])
    raw = load_cached_targets(cache, pair)
    assert raw["regions"].shape == (4, 448, 448)
    annotation = root / pair["jt_annotation_path"]
    annotation.write_text(annotation.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        dataset(synthetic_data)


def test_full_source_seal_covers_all_records_and_hashes_once(
    synthetic_data, monkeypatch
):
    import tools.build_dual_targets as builder

    root, merged, _, manifest, val, fixed, _ = synthetic_data
    rows = merged + json.loads(val.read_text(encoding="utf-8"))
    normalized = [p if "has_jt" in p else builder.normalize_record(p) for p in rows]
    expected = {
        path for pair in normalized for path in record_source_paths(pair).values()
    }
    real_hash = builder.sha256_file
    calls = Counter()

    def count_hash(path):
        calls[Path(path)] += 1
        return real_hash(path)

    monkeypatch.setattr(builder, "sha256_file", count_hash)
    metadata = build_cache(root, rows, root / "rebuild", [manifest, val], fixed)
    assert metadata["schema"] == CACHE_SCHEMA
    assert metadata["source_seal"] == SOURCE_SEAL
    assert set(metadata["source_sha256"]) == expected
    assert all(calls[root / path] == 1 for path in expected)
    assert dataset(synthetic_data).source_verification_scope == SOURCE_SEAL


@pytest.mark.parametrize(
    "field", ["image_path", "target_path", "original_path", "bf_semantic_mask_path"]
)
def test_bf_only_actual_input_mutation_is_rejected(synthetic_data, field):
    root, merged, _, _, _, _, _ = synthetic_data
    pair = next(p for p in merged if p["pool_type"] == "bf_only")
    path = root / pair[field]
    if field == "bf_semantic_mask_path":
        layers = np.load(path, allow_pickle=False)
        layers[0, 0, 0] ^= 1
        np.save(path, layers)
    else:
        with Image.open(path) as image:
            pixels = np.array(image)
        pixels[0, 0] = 77
        Image.fromarray(pixels).save(path)
    with pytest.raises(ValueError, match="source changed"):
        dataset(synthetic_data)


def test_fixed_reference_inputs_are_sealed_without_matching_query(synthetic_data):
    root, merged, _, manifest, val, fixed, _ = synthetic_data
    reference = write_records(root, "Partner", "BF", ["Z"])[0]
    raw = json.loads(fixed.read_text(encoding="utf-8"))
    raw[next(iter(raw))] = reference
    fixed.write_text(json.dumps(raw), encoding="utf-8")
    cache = root / "fixed-reference-cache"
    metadata = build_cache(
        root,
        merged + json.loads(val.read_text(encoding="utf-8")),
        cache,
        [manifest, val],
        fixed,
    )
    assert reference["image_path"] in metadata["source_sha256"]
    data = DualDataset(
        root, [val], cache, RandomMask(), is_train=False, fixed_pair_path=fixed
    )
    assert data[0]["query_target_path"] == reference["target_path"]
    path = root / reference["target_path"]
    with Image.open(path) as image:
        pixels = np.array(image)
    pixels[0, 0] = 77
    Image.fromarray(pixels).save(path)
    with pytest.raises(ValueError, match="source changed"):
        DualDataset(
            root, [val], cache, RandomMask(), is_train=False, fixed_pair_path=fixed
        )


def test_fixed_queries_reject_equal_count_duplicate_and_missing(synthetic_data):
    root, _, _, _, val, fixed, cache = synthetic_data
    rows = json.loads(val.read_text(encoding="utf-8"))
    rows[1] = rows[0]
    val.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="queries missing or duplicated"):
        DualDataset(
            root, [val], cache, RandomMask(), is_train=False, fixed_pair_path=fixed
        )


@pytest.mark.parametrize("field", ["source_seal", "source_inputs", "source_sha256"])
def test_new_cache_missing_seal_fields_fails(synthetic_data, field):
    cache = synthetic_data[-1]
    path = cache / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    del metadata[field]
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="full source seal is incomplete"):
        dataset(synthetic_data)


def test_new_cache_missing_bf_source_hash_fails(synthetic_data):
    _, merged, _, _, _, _, cache = synthetic_data
    pair = next(p for p in merged if p["pool_type"] == "bf_only")
    path = cache / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    del metadata["source_sha256"][pair["bf_semantic_mask_path"]]
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="source hash missing"):
        dataset(synthetic_data)


def test_legacy_cache_reports_only_legacy_jt_verification(synthetic_data):
    cache = synthetic_data[-1]
    path = cache / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["schema"] = "dual-target-cache-v1"
    del metadata["source_seal"]
    del metadata["source_inputs"]
    index = json.loads((cache / "index.json").read_text(encoding="utf-8"))
    paths = {
        path
        for entry in index.values()
        for path in (
            entry["annotation_path"],
            entry["target_path"],
            entry["coordinate_audit"]["original_path"],
        )
    }
    metadata["source_sha256"] = {
        key: value
        for key, value in metadata["source_sha256"].items()
        if key in paths
    }
    path.write_text(json.dumps(metadata), encoding="utf-8")
    assert dataset(synthetic_data).source_verification_scope == "jt-cache-only-legacy"


def test_coordinate_audit_rejects_displaced_gt(synthetic_data):
    root, merged, _, _, _, _, cache = synthetic_data
    pair = next(p for p in merged if p["has_jt"])
    raw = load_cached_targets(cache, pair)
    image = np.full((448, 448), 255, dtype=np.uint8)
    image[300:400, 300:400] = 0
    Image.fromarray(image).save(root / pair["target_path"])
    index = json.loads((cache / "index.json").read_text())
    with pytest.raises(ValueError, match="coordinate audit failed"):
        coordinate_audit(root, pair, raw["text"], index[pair["target_path"]])
