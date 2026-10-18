"""Run C v2 data contracts with real pooling and instance augmentation."""

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image

from data.runc_v2_dataset import (
    RuncV2Dataset,
    build_runc_v2_dataset,
    build_runc_v2_refine_dataset,
    coarse_targets_from_normalized,
)


def normalized(rgb):
    mean = rgb.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = rgb.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    return (rgb - mean) / std


def test_density_polarity_exact_mass_and_occupancy():
    rgb = torch.ones(3, 448, 448)
    rgb[:, 0, 0] = 0
    result = coarse_targets_from_normalized(normalized(rgb))
    assert result["density_target"].shape == (1, 112, 112)
    assert result["density_target"].dtype == torch.float32
    torch.testing.assert_close(result["density_target"].sum(), torch.tensor(1 / 16))
    assert result["occupancy_target"].sum() == 1
    assert result["valid_coarse"].min() == 1


def test_grayscale_threshold_127_128_and_continuous_gray():
    rgb = torch.ones(3, 448, 448)
    rgb[:, :4, :4] = 127 / 255
    rgb[:, :4, 4:8] = 128 / 255
    result = coarse_targets_from_normalized(normalized(rgb))
    assert result["occupancy_target"][0, 0, 0] == 1
    assert result["occupancy_target"][0, 0, 1] == 0
    torch.testing.assert_close(
        result["density_target"][0, 0, 1], torch.tensor(127 / 255)
    )


def test_rgb_luminance_once_normalized_and_valid_weights():
    rgb = torch.ones(3, 448, 448)
    rgb[:, :4, :4] = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
    valid = torch.ones_like(rgb)
    valid[:, :2, :4] = 0
    result = coarse_targets_from_normalized(normalized(rgb), valid=valid)
    torch.testing.assert_close(result["density_target"][0, 0, 0], torch.tensor(0.701))
    assert result["valid_coarse"][0, 0, 0] == 0.5
    assert result["occupancy_target"][0, 0, 0] == 1


def test_background_noise_does_not_become_hard_occupancy():
    rgb = torch.ones(3, 448, 448) * 0.98
    result = coarse_targets_from_normalized(normalized(rgb).unsqueeze(0))
    assert result["occupancy_target"].sum() == 0
    torch.testing.assert_close(result["density_target"].mean(), torch.tensor(0.02))
    with pytest.raises(ValueError, match="448"):
        coarse_targets_from_normalized(torch.ones(3, 112, 112))
    with pytest.raises(ValueError, match="threshold"):
        coarse_targets_from_normalized(normalized(rgb), ink_threshold=1)


class FakeBase:
    """Supply synthetic images; a BF sampler call is always a test failure."""

    is_train = True
    image_size = 448
    patch_size = 16
    root = Path(".")
    target_cache_dir = Path(".")
    _cache_index = {}
    _cache_metadata = {}
    crop_scale = (1.0, 1.0)
    crop_ratio = (1.0, 1.0)
    jieti_pool = 224
    jieti_valid_ink = 200
    _fixed_pairs = None

    def __init__(self):
        self.pairs = []
        for index, (has_jt, pool) in enumerate(
            [
                (True, "dual"),
                (True, "dual"),
                (False, "bf_only"),
            ]
        ):
            self.pairs.append(
                {
                    "image_path": f"source{index}",
                    "target_path": f"gt{index}",
                    "has_bf": True,
                    "has_jt": has_jt,
                    "pool_type": pool,
                    "calligrapher": "author",
                    "character": f"字{index}",
                    "record_id": f"author:{index}",
                    "original_md5": str(index),
                    "original_path": f"original{index}",
                }
            )
        self.weights = [1.0, 1.0, 1.0]
        self.query_black = True
        self._cache_metadata = {
            "source_sha256": {
                **{f"original{index}": f"{index:064x}" for index in range(3)},
                **{f"gt{index}": f"{index + 100:064x}" for index in range(3)},
            }
        }

    def _load_image(self, path, is_target=False):
        image = np.full((448, 448, 3), 255, dtype=np.uint8)
        if is_target and (path != "gt0" or self.query_black):
            image[100:200, 120:200] = 0
        return Image.fromarray(image)

    def _bf_mask(self, *args, **kwargs):
        raise AssertionError("BF sampling must not run in stage I")


def raw_layers(*args, **kwargs):
    regions = np.zeros((4, 448, 448), dtype=np.uint8)
    regions[0, 100:200, 120:200] = 1
    return {
        "regions": regions,
        "text": regions[0],
        "classes": np.array([0, -1, -1, -1]),
    }


@mock.patch("data.runc_v2_dataset.load_cached_targets", side_effect=raw_layers)
def test_jt_only_full_mask_no_bf_sampling_and_shared_geometry(_loader):
    dataset = RuncV2Dataset(FakeBase())
    assert len(dataset) == 2
    sample = dataset.get_pair(0, ref_index=1, augmentation_seed=42)
    assert sample["mask"][:28].sum() == 0
    assert sample["mask"][28:].sum() == 784
    assert sample["input_target"][:, 448:].count_nonzero() == 0
    assert not sample["bf_sampling_enabled"]
    assert sample["instance_occupancy"][0, 6:13, 7:13].sum() > 0
    assert sample["coarse_voro"].shape == (112, 112)
    assert sample["voro"].shape == (224, 224)
    assert sample["density_target"].sum() > 0


@mock.patch("data.runc_v2_dataset.load_cached_targets", side_effect=raw_layers)
def test_gt_replacement_does_not_change_deployment_inputs(_loader):
    base = FakeBase()
    dataset = RuncV2Dataset(base)
    first = dataset.get_pair(0, ref_index=1, augmentation_seed=7)
    base.query_black = False
    second = dataset.get_pair(0, ref_index=1, augmentation_seed=7)
    for field in ("image", "input_target", "mask"):
        assert torch.equal(first[field], second[field])
    assert not torch.equal(first["target"], second["target"])


def test_duplicate_original_identity_rejected():
    base = FakeBase()
    base.pairs[1]["original_md5"] = base.pairs[0]["original_md5"]
    with pytest.raises(ValueError, match="Duplicate"):
        RuncV2Dataset(base)


@mock.patch("data.runc_v2_dataset.load_cached_targets", side_effect=raw_layers)
def test_global_query_loader_replacement_cannot_reach_visible_reference(_loader):
    base = FakeBase()
    dataset = RuncV2Dataset(base)
    first = dataset.get_pair(0, augmentation_seed=14)
    original_loader = base._load_image

    def substituted_loader(path, is_target=False):
        if path == dataset.pairs[0]["target_path"]:
            return Image.new("RGB", (448, 448), "white")
        return original_loader(path, is_target)

    # Replace every query-GT load, including a potential upper-reference load.
    base._load_image = substituted_loader
    second = dataset.get_pair(0, augmentation_seed=14)
    assert first["reference_path"] != first["target_path"]
    for field in ("image", "input_target", "mask"):
        assert torch.equal(first[field], second[field])
    assert not torch.equal(first["target"], second["target"])
    assert first["query_original_identity"] != first["reference_original_identity"]


def test_reference_candidates_exclude_original_and_target_aliases():
    base = FakeBase()
    base.pairs[2].update(has_jt=True, pool_type="dual")
    hashes = base._cache_metadata["source_sha256"]
    hashes["original1"] = hashes["original0"]
    dataset = RuncV2Dataset(base)
    assert dataset.reference_candidates[0] == (2,)
    assert dataset.reference_candidates[1] == (2,)
    with pytest.raises(ValueError, match="Explicit reference"):
        dataset.get_pair(0, ref_index=1)
    with pytest.raises(ValueError, match="Explicit reference"):
        dataset.get_pair(0, ref_index=0)
    hashes["original1"] = f"{1:064x}"
    hashes["gt1"] = hashes["gt0"]
    dataset = RuncV2Dataset(base)
    assert dataset.reference_candidates[0] == (2,)


def test_empty_reference_pool_fails_without_fallback():
    base = FakeBase()
    base.pairs[1]["pool_type"] = "jt_only"
    with pytest.raises(ValueError, match="No distinct-original reference"):
        RuncV2Dataset(base)


@mock.patch("data.runc_v2_dataset.load_cached_targets", side_effect=raw_layers)
def test_multiple_seeds_never_select_self_and_same_character_is_allowed(_loader):
    base = FakeBase()
    base.pairs[1]["character"] = base.pairs[0]["character"]
    dataset = RuncV2Dataset(base)
    assert dataset.reference_candidates == {0: (1,), 1: (0,)}
    for seed in range(32):
        sample = dataset.get_pair(seed % 2, augmentation_seed=seed)
        assert sample["query_original_sha256"] != sample["reference_original_sha256"]
        assert sample["query_target_sha256"] != sample["reference_target_sha256"]


def test_fixed_validation_identity_guard_rejects_without_repairing_pairs():
    base = FakeBase()
    base.is_train = False
    base.pairs = base.pairs[:1]
    base._fixed_pairs = {"gt0": dict(base.pairs[0])}
    with pytest.raises(ValueError, match="Reference must have a distinct"):
        RuncV2Dataset(base)


def real_data_config():
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/runc_v2/smoke.json"
    if not path.exists():
        pytest.skip("Real Run C v2 config is unavailable")
    config = json.loads(path.read_text(encoding="utf-8"))
    data = config["data"]
    for field in ("root", "cache", "fixed_pairs"):
        data[field] = str(root / data[field])
    for field in ("train_json", "val_json"):
        data[field] = [str(root / value) for value in data[field]]
    if not Path(
        data["root"], "font/train/new/CaoqbBF/images_text_denoised/述.png"
    ).exists():
        pytest.skip("Real CalliPhase image assets are unavailable")
    return config


def test_real_query_loader_substitution_and_fixed_validation_counts(monkeypatch):
    config = real_data_config()
    dataset = build_runc_v2_dataset(config)
    assert len(dataset) == 740
    assert dataset.reference_candidate_summary["minimum_candidates"] == 5
    for query, candidates in dataset.reference_candidates.items():
        assert query not in candidates
        assert all(
            dataset._references_distinct(dataset.pairs[query], dataset.pairs[reference])
            for reference in candidates
        )
    first = dataset.get_pair(285, augmentation_seed=14)
    loader = dataset.base_dataset._load_image

    def replace_every_query_load(path, is_target=False):
        if path == first["target_path"]:
            return Image.new("RGB", (448, 448), (17, 91, 203))
        return loader(path, is_target)

    monkeypatch.setattr(dataset.base_dataset, "_load_image", replace_every_query_load)
    second = dataset.get_pair(285, augmentation_seed=14)
    for field in ("image", "input_target", "mask"):
        assert torch.equal(first[field], second[field])
    assert torch.equal(first["target"][:, :448], second["target"][:, :448])
    assert not torch.equal(first["target"][:, 448:], second["target"][:, 448:])
    validation = build_runc_v2_dataset(config, validation=True)
    assert len(validation) == 101
    samples = [validation.get_pair(index) for index in range(len(validation))]
    assert sum(int(sample["instance_valid"].sum()) for sample in samples) == 216
    assert len({sample["query_original_identity"] for sample in samples}) == 77
    assert all(
        sample["query_original_sha256"] != sample["reference_original_sha256"]
        for sample in samples
    )


def test_real_refine_safe_view_preserves_legacy_bf11_sampling(monkeypatch):
    from data.structure_first_dataset import StructureFirstDataset

    dataset = build_runc_v2_refine_dataset(real_data_config())
    assert len(dataset) == 818
    assert dataset.base_dataset.num_mask_annotations_bf == 11
    reference = dataset.reference_candidates[0][0]
    safe = dataset.get_pair(0, ref_index=reference, augmentation_seed=123)
    legacy = StructureFirstDataset(base_dataset=dataset.base_dataset, stage="refine")
    unchanged = legacy.get_pair(0, ref_index=reference, augmentation_seed=123)
    for field in (
        "image",
        "target",
        "input_target",
        "legacy_mask",
        "bf_patch_mask",
        "bf_pixel_mask",
        "bf_loss_mask",
    ):
        assert torch.equal(safe[field], unchanged[field])
    assert safe["bf_mask_stats"] == unchanged["bf_mask_stats"]
    assert safe["bf_mask_stats"]["requested"] == 11
    assert safe["query_original_identity"] != safe["reference_original_identity"]
    with pytest.raises(ValueError, match="Explicit reference"):
        dataset.get_pair(0, ref_index=0)
    loader = dataset.base_dataset._load_image

    def replace_query(path, is_target=False):
        if path == safe["target_path"]:
            return Image.new("RGB", (448, 448), (17, 91, 203))
        return loader(path, is_target)

    monkeypatch.setattr(dataset.base_dataset, "_load_image", replace_query)
    substituted = dataset.get_pair(0, ref_index=reference, augmentation_seed=123)
    for field in ("image", "input_target", "bf_patch_mask", "bf_loss_mask"):
        assert torch.equal(safe[field], substituted[field])
    assert not torch.equal(safe["target"], substituted["target"])
