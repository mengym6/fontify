"""CPU checks for Run B target accumulation and absolute evaluation scores."""

import copy
import json
import sys
from collections import Counter

import pytest
import torch

from tools.dual_report import paired_comparison
from tools.eval_s_baseline import (
    DUAL_SCHEMA,
    EVAL_HASH_KEYS,
    absolute_s_score,
    file_sha256,
    validate_s_baseline,
)
from tools.eval_s_baseline import (
    main as evaluate_main,
)
from tools.smoke_dual_view import (
    EFFECTIVE_BATCH_SIZE,
    POOL_TYPES,
    microbatch_fractions,
    pool_sequence,
    update_batch_layout,
)


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("physical", [1, 2, 3])
def test_every_smoke_update_has_exactly_64_global_targets(world, physical):
    counts = update_batch_layout(world, physical)
    assert sum(counts) * world == EFFECTIVE_BATCH_SIZE
    assert all(1 <= count <= physical for count in counts)
    assert sum(microbatch_fractions(counts)) == pytest.approx(1)
    for category in (*POOL_TYPES, "mixed"):
        for rank in range(world):
            kinds = pool_sequence(category, sum(counts), 0, rank)
            assert len(kinds) == 64 // world
            observed = Counter(kinds)
            if category == "mixed":
                assert set(observed) == set(POOL_TYPES)
                assert max(observed.values()) - min(observed.values()) <= 1
            else:
                assert observed == {category: 64 // world}


def test_mixed_pool_imbalance_rotates_between_updates_and_ranks():
    for rank in range(4):
        totals = Counter()
        for iteration in range(3):
            totals.update(pool_sequence("mixed", 16, iteration, rank))
        assert totals == {name: 16 for name in POOL_TYPES}


@pytest.mark.parametrize("world", [0, 3, 8, True, 4.0, "4"])
def test_smoke_rejects_unsupported_or_noninteger_world(world):
    with pytest.raises(ValueError, match="world-size"):
        update_batch_layout(world, 2)


@pytest.mark.parametrize("physical", [0, -1, True, 2.0, "2"])
def test_smoke_rejects_noninteger_physical_batch(physical):
    with pytest.raises(ValueError, match="batch-size"):
        update_batch_layout(4, physical)


@pytest.mark.parametrize("counts", [[], [0], [1, -1], [True], [1.5]])
def test_smoke_rejects_invalid_weighted_accumulation_counts(counts):
    with pytest.raises(ValueError, match="positive integers"):
        microbatch_fractions(counts)


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("physical", [1, 2, 3])
def test_weighted_accumulated_rank_gradients_match_one_global_batch(world, physical):
    x = torch.linspace(-2, 2, 64, dtype=torch.float64)
    y = x.square() + 0.2
    weight = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    (weight * x - y).square().mean().backward()
    expected = weight.grad.clone()
    rank_gradients = []
    counts = update_batch_layout(world, physical)
    samples_per_rank = sum(counts)
    for rank in range(world):
        parameter = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        start = rank * samples_per_rank
        for count, fraction in zip(counts, microbatch_fractions(counts)):
            loss = (parameter * x[start:start + count] - y[start:start + count])
            (loss.square().mean() * fraction).backward()
            start += count
        rank_gradients.append(parameter.grad)
    # DDP averages synchronized rank gradients; local weighted means must
    # therefore reproduce the independent 64-target mean above.
    actual = torch.stack(rank_gradients).mean()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def denominator_payload(metadata=None):
    return {
        "schema": DUAL_SCHEMA,
        "purpose": "s_baseline",
        "source_role": "baseline_vggfix_ep50",
        "source_epoch": 50,
        "source_checkpoint_sha256": "b" * 64,
        "legacy_l1_reconciliation": {"status": "passed"},
        "L1_JT": 2.0,
        "L1_BF": 4.0,
        "J": 8.0,
        **({key: "a" * 64 for key in EVAL_HASH_KEYS} if metadata is None else metadata),
    }


def test_absolute_s_is_sum_of_three_ratios_and_initial_denominator_scores_three():
    baseline = denominator_payload()
    score, components = absolute_s_score(baseline, baseline)
    assert score == 3
    assert components == {"L1_JT": 1, "L1_BF": 1, "J": 1}
    score, components = absolute_s_score({"L1_JT": 1, "L1_BF": 4, "J": 2}, baseline)
    assert score == 1.75
    assert components == {"L1_JT": 0.5, "L1_BF": 1, "J": 0.25}


@pytest.mark.parametrize("key", EVAL_HASH_KEYS)
def test_stale_s_denominator_hash_rejected(key):
    metadata = {name: "a" * 64 for name in EVAL_HASH_KEYS}
    baseline = denominator_payload(metadata)
    baseline[key] = "c" * 64
    with pytest.raises(ValueError, match=key):
        validate_s_baseline(baseline, metadata)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True])
def test_invalid_s_denominator_is_rejected(bad):
    baseline = denominator_payload()
    baseline["J"] = bad
    with pytest.raises(ValueError, match="positive and finite"):
        validate_s_baseline(baseline, {key: "a" * 64 for key in EVAL_HASH_KEYS})


@pytest.mark.parametrize("field,value", [
    ("schema", "legacy"), ("purpose", "evaluation"),
    ("source_epoch", 49), ("source_role", "ctrl_g12_best"),
    ("source_checkpoint_sha256", "missing"),
    ("legacy_l1_reconciliation", {"status": "failed"}),
])
def test_s_denominator_requires_new_definition_and_verified_baseline_source(
    field, value
):
    baseline = denominator_payload()
    baseline[field] = value
    with pytest.raises(ValueError):
        validate_s_baseline(baseline, {key: "a" * 64 for key in EVAL_HASH_KEYS})


def paired_record(jt_l1, bf_l1, jt_j):
    record = {"schema": DUAL_SCHEMA, **{key: "a" * 64 for key in EVAL_HASH_KEYS}}
    record["per_item"] = [
        {"index": i, "kind": kind, "target_path": f"{i}.png", "ref_path": "ref.png",
         "L1": l1, "J": j, "centroid": 1, "logsigma": 2, "shape": 3}
        for i, (kind, l1, j) in enumerate(
            [("JT", value, jt_j) for value in jt_l1]
            + [("BF", value, None) for value in bf_l1]
        )
    ]
    return record


def test_paired_absolute_scores_use_separate_jt_bf_means_and_match_existing_delta():
    reference = paired_record([2, 4], [4, 8, 12], 8)
    run = paired_record([1, 3], [2, 4, 6], 4)
    result = paired_comparison(run, reference, denominator_payload(), 100, 0)
    score = result["S"]
    assert score["run_components"] == {"L1_JT": 1, "L1_BF": 1, "J": 0.5}
    assert score["reference_components"] == {"L1_JT": 1.5, "L1_BF": 2, "J": 1}
    assert score["run_S"] == 2.5
    assert score["reference_S"] == 4.5
    assert score["mean_difference"] == score["run_S"] - score["reference_S"]
    assert score["mean_difference"] == -2


def test_paired_report_rejects_changed_cohort_identity():
    reference = paired_record([2, 4], [4, 8, 12], 8)
    run = copy.deepcopy(reference)
    run["per_item"][0]["kind"] = "BF"
    with pytest.raises(ValueError, match="cohort membership"):
        paired_comparison(run, reference, denominator_payload(), 100, 0)


@pytest.mark.parametrize("with_denominator", [True, False])
def test_standalone_eval_writes_absolute_s_and_hash_only_when_requested(
    tmp_path, monkeypatch, with_denominator
):
    import tools.eval_s_baseline as evaluator

    metadata = {key: "a" * 64 for key in EVAL_HASH_KEYS if key != "eval_coef_sha256"}
    coefficient_path = tmp_path / "eval_coef.json"
    coefficient_path.write_text(json.dumps({
        "schema": DUAL_SCHEMA, "purpose": "evaluation", "status": "validated",
        "source_role": "ctrl_g12_best", "source_epoch": 46,
        "source_checkpoint_sha256": "c" * 64,
        "coef": {"centroid": 1, "logsigma": 1, "shape": 1}, **metadata,
    }))
    denominator_path = tmp_path / "denominator.json"
    denominator_path.write_text(json.dumps(denominator_payload({
        **metadata, "eval_coef_sha256": file_sha256(coefficient_path),
    })))
    output = tmp_path / "evaluation.json"

    class Model:
        _checkpoint_sha256 = "d" * 64
        _checkpoint_source = "test-model.pth"
        _checkpoint_epoch = 50

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(evaluator, "eval_metadata", lambda args: metadata)
    monkeypatch.setattr(evaluator, "load_model", lambda path: Model())
    monkeypatch.setattr(evaluator, "build_dataset", lambda args: object())
    monkeypatch.setattr(evaluator, "make_jieti_loss", lambda *args: Model())
    monkeypatch.setattr(evaluator, "evaluate", lambda *args: (
        {"L1_JT": 1, "L1_BF": 4, "J": 2, "n_jt": 100, "n_bf": 140},
        [{}] * 240,
    ))
    argv = [
        "eval_s_baseline.py", "--checkpoint", "test-model.pth",
        "--data_path", "unused", "--val_json_path", "unused",
        "--fixed_pair_path", "unused", "--j_definition", DUAL_SCHEMA,
        "--calibration_json", str(coefficient_path), "--device", "cpu",
        "--output_json", str(output),
    ]
    if with_denominator:
        argv.extend(["--s_baseline_path", str(denominator_path)])
    monkeypatch.setattr(sys, "argv", argv)
    evaluate_main()
    result = json.loads(output.read_text())
    if with_denominator:
        assert result["S"] == 1.75
        assert result["S_components"] == {"L1_JT": 0.5, "L1_BF": 1, "J": 0.25}
        assert result["s_baseline_sha256"] == file_sha256(denominator_path)
    else:
        assert "S" not in result and "s_baseline_sha256" not in result
