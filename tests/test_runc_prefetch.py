"""Prefetch must preserve actual BF sampling, geometry, seeds and draw order."""

import test_dual_data as legacy_data
import torch

from data.dual_dataset import DualDataset
from data.structure_first_dataset import StructureFirstDataset
from tools.structure_first import FixedDrawSamples, collate_indices

synthetic_data = legacy_data.synthetic_data


def assert_identical(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_identical(left[key], right[key])
    else:
        assert left == right


def test_spawn_prefetch_is_identical_to_serial_bf_and_mixed_draws(synthetic_data):
    root, _, _, manifest, _, _, cache = synthetic_data
    base = DualDataset(root, [manifest], cache, legacy_data.RandomMask())
    dataset = StructureFirstDataset(base_dataset=base)
    draws = list(range(len(dataset))) * 2
    draws = draws[:len(draws) // 2 * 2]
    stream = FixedDrawSamples(dataset, draws, micro=2, world=2, rank=1, seed=31)
    loader = torch.utils.data.DataLoader(
        stream, batch_size=2, num_workers=2, multiprocessing_context="spawn",
        generator=torch.Generator().manual_seed(31),
    )
    for number, actual in enumerate(loader):
        offset = number * 2
        expected = collate_indices(
            dataset, draws[offset:offset + 2], torch.device("cpu"),
            seed=31 + offset * 2 + 1,
        )
        assert_identical(actual, expected)
