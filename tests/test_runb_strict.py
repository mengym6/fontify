"""Regression coverage for strict Run B reference sampling and GT isolation."""

import json
import random

import pytest
import torch
from PIL import Image
from test_dual_data import dataset

from tools.dual_view_protocol import choose_fixed_rows
from util.strict_reference import STRICT_REFERENCE_POLICY, references_distinct


@pytest.fixture
def base_data(tmp_path):
    from test_dual_data import synthetic_data

    return synthetic_data.__wrapped__(tmp_path)


@pytest.fixture
def strict_data(base_data):
    metadata = base_data[-1] / "metadata.json"
    value = json.loads(metadata.read_text())
    value["reference_policy"] = STRICT_REFERENCE_POLICY
    metadata.write_text(json.dumps(value))
    return base_data


def test_identity_rejects_original_and_target_aliases():
    first = {"original_path": "a", "target_path": "x"}
    second = {"original_path": "b", "target_path": "y"}
    hashes = {"a": "a" * 64, "b": "b" * 64, "x": "c" * 64, "y": "d" * 64}
    assert references_distinct(first, second, hashes)
    assert not references_distinct(first, first, hashes)
    for source, alias in (("a", "b"), ("x", "y")):
        changed = dict(hashes)
        changed[alias] = changed[source]
        assert not references_distinct(first, second, changed)
    with pytest.raises(ValueError, match="sealed"):
        references_distinct(first, second, {})


def test_default_legacy_pairing_is_unchanged(base_data):
    value = dataset(base_data)
    assert value.reference_candidates is None
    sample = value.get_pair(0, ref_index=0, augmentation_seed=0)
    assert sample["target_path"] == sample["reference_path"]


def test_strict_candidates_and_explicit_self_rejection(strict_data):
    value = dataset(strict_data)
    for index, candidates in value.reference_candidates.items():
        assert index not in candidates and candidates
        with pytest.raises(ValueError, match="Self-reference"):
            value.get_pair(index, ref_index=index, augmentation_seed=0)
        for seed in range(8):
            sample = value.get_pair(index, augmentation_seed=seed)
            assert sample["target_path"] != sample["reference_path"]


def test_query_gt_loader_replacement_cannot_reach_visible_reference(strict_data):
    value = dataset(strict_data)
    query_path = value.pairs[0]["target_path"]
    before = value.get_pair(0, augmentation_seed=19)
    loader = value._load_image

    def replace_query(path, is_target=False):
        if path == query_path:
            return Image.new("RGB", (448, 448), "white")
        return loader(path, is_target=is_target)

    value._load_image = replace_query
    after = value.get_pair(0, augmentation_seed=19)
    assert torch.equal(before["image"], after["image"])
    assert torch.equal(before["mask"], after["mask"])
    assert torch.equal(before["target"][:, :448], after["target"][:, :448])
    assert not torch.equal(before["target"][:, 448:], after["target"][:, 448:])


def test_fixed_probes_use_strict_candidates(strict_data):
    value = dataset(strict_data)
    rows = choose_fixed_rows(
        value.pairs,
        value.weights,
        range(len(value)),
        len(value),
        random.Random(0),
        value.reference_candidates,
    )
    assert all(row["target_index"] != row["reference_index"] for row in rows)


def test_strict_validation_rejects_old_self_pairs(strict_data):
    from test_dual_data import RandomMask

    from data.dual_dataset import DualDataset

    root, _, _, _, val, fixed, cache = strict_data
    with pytest.raises(ValueError, match="Self-reference"):
        DualDataset(
            root, [val], cache, RandomMask(), is_train=False, fixed_pair_path=fixed
        )


def test_sealed_bf_audit_reuse_rejects_changed_assets(base_data):
    from tools.build_dual_targets import reuse_sealed_bf_audit

    root, merged, _, _, _, _, cache = base_data
    prior = json.loads((cache / "metadata.json").read_text())
    pair = next(row for row in merged if row["has_bf"])
    audit = reuse_sealed_bf_audit(root, pair, "pad_square", prior)
    assert audit["different_mask_pixels"] == 0
    assert audit["verification"] == "reused-certificate-with-identical-input-sha256"
    changed = root / pair["bf_semantic_mask_path"]
    changed.write_bytes(changed.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="input changed"):
        reuse_sealed_bf_audit(root, pair, "pad_square", prior)


def test_sealed_bf_audit_reuse_rejects_wrong_geometry(base_data):
    from tools.build_dual_targets import reuse_sealed_bf_audit

    root, merged, _, _, _, _, cache = base_data
    prior = json.loads((cache / "metadata.json").read_text())
    pair = next(row for row in merged if row["has_bf"])
    with pytest.raises(ValueError, match="certificate"):
        reuse_sealed_bf_audit(root, pair, "direct_resize", prior)
