"""Run C v2 data contracts with real pooling and instance augmentation."""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image

from data.runc_v2_dataset import (
    RuncV2Dataset,
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
                }
            )
        self.weights = [1.0, 1.0, 1.0]
        self.query_black = True

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
