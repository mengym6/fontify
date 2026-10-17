"""Run C BF masks reuse actual legacy sampling, crops and query pairing."""

import random

import numpy as np
import pytest
import test_dual_data as legacy_data
import torch

from data.dual_dataset import DualDataset
from data.structure_first_dataset import StructureFirstDataset

RandomMask = legacy_data.RandomMask
synthetic_data = legacy_data.synthetic_data


def make_base(data, **kwargs):
    root, _, _, manifest, _, _, cache = data
    return DualDataset(root, [manifest], cache, RandomMask(), **kwargs)


def assert_equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    else:
        assert left == right


def test_instrumentation_preserves_values_and_rng(synthetic_data):
    plain = make_base(synthetic_data)
    observed = make_base(synthetic_data, collect_bf_mask_stats=True)
    index = next(i for i, pair in enumerate(plain.pairs) if pair["has_bf"])
    random.seed(98)
    np.random.seed(98)
    torch.manual_seed(98)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state()
    legacy = plain.get_pair(index)
    after_plain = (
        random.getstate(), np.random.get_state(), torch.get_rng_state()
    )
    random.setstate(before[0])
    np.random.set_state(before[1])
    torch.set_rng_state(before[2])
    extra = observed.get_pair(index)
    after_stats = (
        random.getstate(), np.random.get_state(), torch.get_rng_state()
    )
    for key, value in legacy.items():
        assert_equal(value, extra[key])
    assert after_plain[0] == after_stats[0]
    assert np.array_equal(after_plain[1][1], after_stats[1][1])
    assert torch.equal(after_plain[2], after_stats[2])
    assert extra["bf_mask_stats"]["selected"] == 1
    assert extra["bf_mask_stats"]["requested"] == 11


def test_stage_view_preserves_reference_pool(synthetic_data):
    base = make_base(synthetic_data)
    original_pairs = list(base.pairs)
    coarse = StructureFirstDataset(base_dataset=base)
    refine = StructureFirstDataset(base_dataset=base, stage="refine")
    assert len(coarse) == 6 and len(refine) == 4
    assert base.pairs == original_pairs
    assert {pair["pool_type"] for pair in refine.pairs} == {"dual", "bf_only"}
    for index, pair in enumerate(refine.pairs):
        sample = refine.get_pair(index, augmentation_seed=31)
        reference = next(
            row for row in base.pairs
            if row["target_path"] == sample["reference_path"]
        )
        assert reference["calligrapher"] == pair["calligrapher"]
        assert reference["pool_type"] == pair["pool_type"]
        assert torch.equal(sample["bf_patch_mask"], sample["legacy_mask"][28:])
        assert not sample["mask"][:28].any() and sample["mask"][28:].all()
        assert not sample["input_target"][:, 448:].any()
        assert torch.equal(
            sample["input_target"][:, :448], sample["target"][:, :448]
        )
        support = sample["bf_patch_mask"].repeat_interleave(16, 0)
        support = support.repeat_interleave(16, 1)[None]
        assert torch.equal(sample["bf_pixel_mask"], support)
        assert torch.equal(
            sample["bf_loss_mask"], support * sample["valid"][:, 448:]
        )
    jt_index = next(
        i for i, pair in enumerate(coarse.pairs) if not pair["has_bf"]
    )
    jt = coarse.get_pair(jt_index, augmentation_seed=23)
    assert not jt["bf_patch_mask"].any()
    assert jt["bf_mask_stats"]["selected"] == 0


@pytest.mark.parametrize("layers_count", [1, 3, 11, 17])
@pytest.mark.parametrize("kind", ["empty", "overlap", "full"])
def test_layer_sampling_counts_no_padding_and_no_resampling(
        tmp_path, monkeypatch, layers_count, kind):
    dataset = DualDataset.__new__(DualDataset)
    dataset.root = tmp_path
    dataset.image_size = 448
    dataset.num_mask_annotations_bf = 11
    layers = np.zeros((layers_count, 448, 448), dtype=np.uint8)
    if kind == "overlap":
        layers[:, 10:30, 20:40] = 1
    if kind == "full":
        layers[:] = 1
    np.save(tmp_path / "layers.npy", layers)
    calls = []
    original_sample = random.sample

    def observed_sample(population, count):
        selected = original_sample(population, count)
        calls.append(selected)
        return selected

    monkeypatch.setattr(random, "sample", observed_sample)
    union, selected_layers, stats = dataset._bf_mask(
        {"bf_semantic_mask_path": "layers.npy"}, return_stats=True
    )
    assert len(calls) == 1
    assert len(calls[0]) == len(set(calls[0])) == min(11, layers_count)
    assert np.array_equal(union, np.any(layers[calls[0]] > 0, axis=0))
    assert selected_layers.shape[0] == min(11, layers_count)
    assert stats["requested"] == 11
    assert stats["selected"] == min(11, layers_count)
    nonempty = 0 if kind == "empty" else min(11, layers_count)
    assert stats["nonempty_source"] == nonempty


def test_zero_layer_source_keeps_legacy_error(tmp_path):
    dataset = DualDataset.__new__(DualDataset)
    dataset.root = tmp_path
    dataset.image_size = 448
    dataset.num_mask_annotations_bf = 11
    np.save(tmp_path / "empty.npy", np.zeros((0, 448, 448), dtype=np.uint8))
    with pytest.raises(ValueError, match="no layers"):
        dataset._bf_mask({"bf_semantic_mask_path": "empty.npy"})


def test_patch_coverage_strict_threshold_and_same_crop(synthetic_data):
    root, rows, _, _, _, _, _ = synthetic_data
    # 25/256 fails strict >0.1, while 26/256 passes. These are individual
    # annotation layers whose overlapping union is not merged into new labels.
    layers = np.zeros((3, 448, 448), dtype=np.uint8)
    layers[0, :2, :13] = 1
    layers[1, :1, 16:32] = 1
    layers[1, 1, 16:25] = 1
    layers[2] = layers[0]
    for row in rows:
        if row["has_bf"]:
            np.save(root / row["bf_semantic_mask_path"], layers)
    base = make_base(synthetic_data, verify_sources=False,
                     crop_scale=(1, 1), crop_ratio=(1, 1))
    dataset = StructureFirstDataset(base_dataset=base, stage="refine")
    sample = dataset.get_pair(0, augmentation_seed=5)
    assert sample["geometry_crop"].tolist() == [0, 0, 448, 448]
    assert sample["bf_patch_mask"][0, 0] == 1
    assert sample["bf_patch_mask"][0, 1] == 0
    stats = sample["bf_mask_stats"]
    assert stats["selected"] == stats["nonempty_after_crop"] == 3
    assert stats["target_patch_coverage"] == pytest.approx(1 / 784)


def test_crop_reports_lost_selected_layers_without_fallback(synthetic_data):
    base = make_base(synthetic_data, collect_bf_mask_stats=True)
    pair = next(pair for pair in base.pairs if pair["has_bf"])
    # All synthetic BF ink is at 80:180; this exact crop excludes every layer.
    transformed = base._transform_row(pair, (300, 300, 100, 100), False)
    assert not transformed[2].any()
    assert transformed[4]["selected"] == 1
    assert transformed[4]["nonempty_source"] == 1
    assert transformed[4]["nonempty_after_crop"] == 0
