"""Representation checks independent of trainable model parameters."""

import numpy as np
import torch
from torch.nn import functional as F

from data.runc_v2_dataset import coarse_targets_from_normalized
from tools.runc_v2_audit import component_details, merge_details


def test_four_separated_small_dots_survive_density_and_occupancy():
    rgb = torch.ones(1, 3, 448, 448)
    for x in (80, 160, 240, 320):
        end = x + 8
        rgb[:, :, 380:388, x:end] = 0
    mean = rgb.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = rgb.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    targets = coarse_targets_from_normalized((rgb - mean) / std)
    oracle = F.interpolate(
        targets["density_target"], (448, 448), mode="bilinear", align_corners=False
    )
    for x in (80, 160, 240, 320):
        end, cell_start, cell_end = x + 8, x // 4, x // 4 + 2
        assert oracle[0, 0, 380:388, x:end].sum() > 0
        assert targets["occupancy_target"][0, 0, 95:97, cell_start:cell_end].all()
    torch.testing.assert_close(
        targets["density_target"].sum() * 16, torch.tensor(4 * 64.0)
    )


def test_bilinear_density_oracle_remains_differentiable():
    density = torch.rand(2, 1, 112, 112, requires_grad=True)
    p0 = 1 - F.interpolate(density, (448, 448), mode="bilinear", align_corners=False)
    p0.square().mean().backward()
    assert density.grad.isfinite().all()
    assert density.grad.abs().sum() > 0


def test_topology_detail_coordinates_are_in_original_target_pixels():
    source = np.zeros((448, 448), dtype=np.float32)
    source[100:110, 100:110] = 1
    source[100:110, 112:122] = 1
    source[300, 330] = 1
    oracle = source.copy()
    oracle[100:110, 110:112] = 1
    oracle[300, 330] = 0
    lost = component_details(source, oracle)
    merged = merge_details(source, oracle)
    assert len(lost) == 1
    assert lost[0]["bbox_xyxy"] == [330, 300, 331, 301]
    assert lost[0]["area"] == 1
    assert len(merged) == 1
    assert merged[0]["closest_distance_pixels"] == 3
    assert merged[0]["source_component_areas"] == [100, 100]
