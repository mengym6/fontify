"""CPU health contracts for freezing and sample-weighted tiny updates."""

import copy
from types import SimpleNamespace

import pytest
import torch

from tools import runc_v2_health as health


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dual_head = torch.nn.Linear(2, 1)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.blocks.requires_grad_(False)
        self.register_buffer("fixed_buffer", torch.tensor([1.0]))


def test_frozen_parameters_and_buffers_are_both_checked():
    model = ToyModel()
    before = health.state_fingerprints(model)
    with torch.no_grad():
        model.dual_head.weight.add_(1)
    changed = health.compare_fingerprints(before, health.state_fingerprints(model))
    assert changed["all_frozen_parameters_and_buffers_unchanged"]
    assert changed["changed_tensor_counts_by_group"] == {"head": 1}
    with torch.no_grad():
        model.fixed_buffer.add_(1)
    changed = health.compare_fingerprints(before, health.state_fingerprints(model))
    assert not changed["all_frozen_parameters_and_buffers_unchanged"]
    assert "buffer:fixed_buffer" in changed["frozen_changed"]


def test_trainability_changes_cannot_escape_hash_checks():
    model = ToyModel()
    before = health.state_fingerprints(model)
    model.blocks.requires_grad_(True)
    with pytest.raises(RuntimeError, match="Trainability"):
        health.compare_fingerprints(before, health.state_fingerprints(model))


def test_fixed_targets_are_from_distinct_authors_and_jt_only():
    dataset = SimpleNamespace(
        pairs=[
            {"calligrapher": "A", "has_jt": False},
            {"calligrapher": "B", "has_jt": True},
            {"calligrapher": "A", "has_jt": True},
            {"calligrapher": "B", "has_jt": True},
        ]
    )
    assert health.fixed_author_indices(dataset) == [2, 1]
    with pytest.raises(ValueError, match="two different"):
        health.fixed_author_indices(dataset, 3)


def test_unequal_microbatches_match_one_full_batch_update(monkeypatch):
    torch.manual_seed(7)
    full_model = ToyModel()
    micro_model = copy.deepcopy(full_model)
    monkeypatch.setattr(health.runtime, "training_mode", lambda *args: None)
    monkeypatch.setattr(
        health.runtime,
        "call_model",
        lambda model, batch, render: model.dual_head(batch["image"]),
    )

    def loss_for(output, batch, config):
        value = (output - batch["target"]).square().mean()
        return {"total": value, "raw_head": value, "raw_head_geo": value / 2}

    monkeypatch.setattr(health.runtime, "loss_for", loss_for)
    config = {
        "stage": "ia0",
        "training": {
            "lr_head": 0.01,
            "lr_vit": 0.001,
            "lr_coarse": 0.01,
            "weight_decay": 0,
            "amp": False,
            "clip_grad": 10,
        },
    }
    batch = {
        "image": torch.tensor([[1.0, 2.0], [2.0, 4.0], [-1.0, 3.0]]),
        "target": torch.tensor([[1.0], [-1.0], [2.0]]),
        "author": ["A", "B", "C"],
    }
    full, _ = health.run_updates(
        full_model, full_model, batch, config, torch.device("cpu"), 1, 3
    )
    micro, _ = health.run_updates(
        micro_model, micro_model, batch, config, torch.device("cpu"), 1, 2
    )
    torch.testing.assert_close(
        full_model.dual_head.weight, micro_model.dual_head.weight
    )
    assert full[0]["losses"]["total"] == pytest.approx(
        micro[0]["losses"]["total"], rel=1e-6
    )
    assert full[0]["gradient_pre_clip"] == pytest.approx(
        micro[0]["gradient_pre_clip"], rel=1e-6
    )
    assert micro_model.blocks[0].weight.grad is None


def test_full_forward_profiler_explicitly_reports_incomplete_flops(monkeypatch):
    model = ToyModel()
    monkeypatch.setattr(
        health.runtime,
        "call_model",
        lambda wrapped, batch, render: {"density": wrapped.dual_head(batch["image"])},
    )
    batch = {"image": torch.ones(2, 2)}
    result = health.profile_full_forward(
        model, model, batch, {"training": {"amp": False}}, torch.device("cpu")
    )
    assert result["profiler_counted_operator_flops"] > 0
    assert result["output_shapes"]["density"]["shape"] == [1, 1]
    assert any("SDPA" in message for message in result["limitations"])
    assert "peak_allocated_bytes" not in result
