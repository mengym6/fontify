"""CPU contracts for runc orchestration, separate from production GPU checks."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import tools.structure_first as runtime
from tools.structure_first import (
    SCHEMA,
    WEIGHT_KEYS,
    assert_optimizer,
    clear_unlabelled_head_grads,
    head_metrics,
    parameter_grams,
    prepare_output,
    require_coarse_acceptance,
    require_lowpass_acceptance,
    require_training_authorization,
    sha256,
    training_authorization_digest,
    validate_config,
)


def configuration():
    return {
        "schema": SCHEMA,
        "experiment": "runc",
        "stage": "coarse",
        "checkpoint": "sealed-run-b.pth",
        "weights": dict.fromkeys(WEIGHT_KEYS, 1.0),
        "j_coefficients": {"centroid": 1.0, "logsigma": 1.0, "shape": 1.0},
        "lowpass": {"kernel_size": 7, "sigma": 1.5, "boundary": "reflect"},
        "training": {
            "steps": 8, "batch_size": 1, "accum_iter": 2,
            "lr": 1e-4, "weight_decay": 0.05, "seed": 0,
            "save_every": 8, "amp": False, "freeze_vit": True,
        },
    }


def test_formal_train_waits_for_lowpass_review(tmp_path):
    config = configuration()
    assert validate_config(config, "smoke") == config
    with pytest.raises(ValueError, match="lowpass_acceptance_json"):
        validate_config(config, "train")
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps({
        "schema": SCHEMA, "user_confirmed": True,
        "confirmation_reference": "user message accepting the displayed panels",
        "lowpass": config["lowpass"],
    }))
    config["lowpass_acceptance_json"] = str(approval)
    with pytest.raises(ValueError, match="formal_training_authorization_json"):
        validate_config(config, "train")
    checkpoint = tmp_path / "coarse.pth"
    checkpoint.write_bytes(b"sealed test checkpoint")
    config["checkpoint"] = str(checkpoint)
    authorization = tmp_path / "formal-authorization.json"
    authorization.write_text(json.dumps({
        "schema": SCHEMA, "user_confirmed": True, "training_authorized": True,
        "confirmation_reference": "a separate user message authorizing training",
        "specification_sha256": training_authorization_digest(config),
    }))
    config["formal_training_authorization_json"] = str(authorization)
    assert validate_config(config, "train") == config
    config["lowpass"]["sigma"] = 2.0
    with pytest.raises(ValueError, match="exact settings"):
        require_lowpass_acceptance(config)


def test_stage_two_gate_binds_exact_coarse_checkpoint(tmp_path):
    config = configuration()
    checkpoint = tmp_path / "coarse.pth"
    checkpoint.write_bytes(b"candidate coarse checkpoint")
    config["checkpoint"] = str(checkpoint)
    approval = tmp_path / "coarse.json"
    record = {
        "schema": SCHEMA, "user_confirmed": True,
        "confirmation_reference": "user reviewed P0 and scale diagnostics",
        "checkpoint_sha256": sha256(checkpoint),
    }
    approval.write_text(json.dumps(record))
    config["coarse_acceptance_json"] = str(approval)
    assert require_coarse_acceptance(config) == record
    checkpoint.write_bytes(b"different coarse checkpoint")
    with pytest.raises(ValueError, match="user confirmation and SHA"):
        require_coarse_acceptance(config)


def test_delegated_acceptance_records_agent_and_real_evidence(tmp_path):
    config = configuration()
    evidence = tmp_path / "reviewed-panels.png"
    evidence.write_bytes(b"evidence exists")
    record = {
        "schema": SCHEMA, "actor": "codex", "status": "accepted",
        "delegated_authority_reference": "user: proceed without waiting for me",
        "evidence_paths": [str(evidence)], "lowpass": config["lowpass"],
    }
    path = tmp_path / "delegated-acceptance.json"
    path.write_text(json.dumps(record))
    config["lowpass_acceptance_json"] = str(path)
    assert require_lowpass_acceptance(config)["actor"] == "codex"
    config["training_authorization"] = {
        "authorized": True,
        "reference": "user: proceed without waiting for me",
    }
    assert require_training_authorization(config) == config["training_authorization"]
    evidence.unlink()
    with pytest.raises(ValueError):
        require_lowpass_acceptance(config)


@pytest.mark.parametrize("path", ["existing", "empty"])
def test_output_cannot_overwrite_any_existing_directory(tmp_path, path):
    target = prepare_output(tmp_path / path)
    with pytest.raises(FileExistsError):
        prepare_output(target)


@pytest.mark.parametrize("mutation", [
    {"experiment": "runb"},
    {"stage": "unknown"},
    {"weights": {"lambda_full": 0.5}},
    {"lowpass": {"kernel_size": 6, "sigma": 1.5, "boundary": "reflect"}},
])
def test_invalid_protocol_rejected(mutation):
    config = configuration()
    config.update(mutation)
    with pytest.raises(ValueError):
        validate_config(config, "smoke")


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dual_head = nn.Linear(2, 1)
        self.bf_refiner = nn.Linear(2, 1)


def test_balanced_probe_covers_authors_before_revisiting():
    pairs = [{"calligrapher": name} for name in ("a", "b", "c")
             for _ in range(4)]
    order = runtime.balanced_probe_order(pairs, list(range(12)), seed=13)
    assert len(order) == len(set(order)) == 12
    for start in range(0, 12, 3):
        assert {pairs[index]["calligrapher"]
                for index in order[start:start + 3]} == {"a", "b", "c"}
    assert order == runtime.balanced_probe_order(pairs, list(range(12)), seed=13)


def test_bf_only_head_adamw_noop_after_momentum_exists():
    model = SmallModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
    model.dual_head(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before = {key: value.detach().clone()
              for key, value in model.dual_head.named_parameters()}
    # A zero gradient tensor is insufficient: decay and Adam momentum persist.
    (model.dual_head(torch.ones(1, 2)).sum() * 0).backward()
    assert all(p.grad is not None for p in model.dual_head.parameters())
    assert clear_unlabelled_head_grads(model, 0) == 0
    optimizer.step()
    assert all(torch.equal(before[key], value)
               for key, value in model.dual_head.named_parameters())


def test_mixed_window_keeps_previous_jt_gradient():
    model = SmallModel()
    model.dual_head(torch.ones(1, 2)).sum().backward()
    before = [p.grad.clone() for p in model.dual_head.parameters()]
    (model.dual_head(torch.ones(1, 2)).sum() * 0).backward()
    assert clear_unlabelled_head_grads(model, 1) == 1
    assert all(torch.equal(left, right.grad)
               for left, right in zip(before, model.dual_head.parameters()))


def test_optimizer_rejects_unfrozen_stage_one():
    model = SmallModel()
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(RuntimeError, match="stage II"):
        assert_optimizer(model, optimizer, "refine")
    model.dual_head.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.bf_refiner.parameters())
    assert_optimizer(model, optimizer, "refine")


def test_raw_gradient_gram_detects_cancellation_and_unused_terms():
    parameter = nn.Parameter(torch.tensor([2.0, 1.0]))
    unused = nn.Parameter(torch.tensor([3.0]))
    terms = {
        "positive": parameter.sum(),
        "negative": -parameter.sum(),
        "constant": torch.tensor(0.0),
    }
    result = parameter_grams(terms, [("weight", parameter), ("unused", unused)])
    assert result["norms"]["positive"] == pytest.approx(2 ** 0.5)
    assert result["norms"]["constant"] == 0
    assert result["cosine"]["negative:positive"] == pytest.approx(-1)
    assert result["cosine"]["constant:positive"] is None


def test_training_log_counts_sum_and_losses_mean_across_ranks(monkeypatch):
    def add_other_rank(values, op):
        values.add_(torch.tensor([6.0, 8.0], dtype=values.dtype))

    monkeypatch.setattr(torch.distributed, "all_reduce", add_other_rank)
    logs = [{"count_batch": 2, "raw_low": 2.0},
            {"count_batch": 2, "raw_low": 6.0}]
    result = runtime.reduce_training_logs(logs, torch.device("cpu"), world=2)
    assert result == {"count_batch": 10.0, "raw_low": 6.0}


def test_snapshot_keeps_parent_epoch_separate_from_new_updates(tmp_path):
    model = SmallModel()
    model.structure_first_config = {"experiment": "runc"}
    config = configuration()
    optimizer = torch.optim.AdamW(model.parameters())
    runtime.snapshot(model, {"epoch": 50, "args": {"model": "parent"}}, config,
                     {}, optimizer, step=3, output=tmp_path, world=2)
    saved = torch.load(tmp_path / "checkpoint-step-000003.pth", weights_only=False)
    assert saved["epoch"] is None and saved["parent_epoch"] == 50
    assert saved["optimizer_step"] == 3
    assert saved["effective_batch"] == 4
    assert saved["processed_target_count"] == 12
    assert saved["planned_target_count"] == 32
    assert "architecture reconstruction" in saved["args_role"]


def test_head_metrics_use_matching_and_valid_instance_count():
    target = torch.zeros(1, 2, 2, 2)
    target[0, 0, 0, 0] = 1
    logits = torch.full((1, 2, 2, 2), -20.0)
    logits[0, 1, 0, 0] = 20
    head = {"occupancy_logits": logits,
            "class_logits": torch.tensor([[[0., 1.], [1., 0.]]])}
    batch = {"instance_occupancy": target, "has_jt": torch.tensor([True]),
             "instance_classes": torch.tensor([[0, -1]])}
    details = {"assignment": torch.tensor([[1, 0]]),
               "matched_slots": torch.tensor([[False, True]])}
    result = head_metrics(head, batch, details)
    assert result["head_instance_count"] == 1
    assert result["head_role_correct_sum"] == 1
    assert result["head_soft_iou_sum"] == pytest.approx(1)


def test_visualization_accepts_bfloat16_images_masks_and_occupancy(tmp_path):
    from PIL import Image

    image = torch.randn(1, 3, 16, 8, dtype=torch.bfloat16)
    outputs = {
        "p0": image.clone(), "p1": image.clone(),
        "mask": torch.ones(1, 4, 4, dtype=torch.bfloat16),
        "head": {"occupancy_logits": torch.randn(
            1, 4, 4, 4, dtype=torch.bfloat16,
        )},
    }
    destination = tmp_path / "bfloat16.png"
    runtime.save_visual(destination, outputs, {"image": image, "target": image})
    with Image.open(destination) as opened:
        assert opened.mode == "RGB"
        assert opened.size == (48, 8)
    metadata = json.loads(Path(str(destination) + ".json").read_text())
    assert metadata["panels"][-1] == "predicted-S-occupancy"


@pytest.mark.parametrize("stage", ["coarse", "refine"])
def test_runner_smoke_integrates_actual_model_loss_and_pool_windows(
    stage, tmp_path, monkeypatch,
):
    spec = importlib.util.spec_from_file_location(
        "_runtime_structure_fixture",
        Path(__file__).with_name("test_structure_first_model.py"),
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    model = fixture.new_model()
    fixture.set_structure_stage(model, stage)

    class Dataset:
        pairs = [
            {"has_jt": True, "has_bf": True},
            {"has_jt": False, "has_bf": True},
            *([{"has_jt": True, "has_bf": False}] if stage == "coarse" else []),
        ]

        def get_pair(self, index, augmentation_seed=None):
            image, target, mask, valid = fixture.sample_inputs(batch=1)
            pair = self.pairs[index]
            sample = {
                "image": image[0], "target": target[0], "mask": mask[0],
                "valid": valid[0], "has_jt": torch.tensor(pair["has_jt"]),
                "has_bf": torch.tensor(pair["has_bf"]),
                "instance_occupancy": torch.ones(4, 4, 4),
                "instance_valid": torch.tensor([pair["has_jt"], False, False, False]),
                "instance_classes": torch.zeros(4, dtype=torch.long),
                "voro": torch.ones(8, 8, dtype=torch.long),
                "valid_parts": torch.tensor([pair["has_jt"], False, False, False]),
                "bf_patch_mask": fixture.bf_mask(batch=1)[0],
            }
            sample["bf_pixel_mask"] = sample["bf_patch_mask"].repeat_interleave(
                2, 0,
            ).repeat_interleave(2, 1)[None].float()
            return sample

    config = configuration()
    config["stage"] = stage
    config["training"]["amp"] = True
    monkeypatch.setattr(runtime, "metadata", lambda config: {})
    monkeypatch.setattr(runtime, "save_visual", lambda *args: None)
    runtime.smoke(config, model, Dataset(), tmp_path, torch.device("cpu"), 0, 1)
    report = json.loads((tmp_path / "smoke-rank-0.json").read_text())
    assert report["gt_perturbation_equal"]
    assert report["deterministic_equal"]
    assert report["precision"] == {
        "forward_autocast": True, "forward_dtype": "bfloat16",
        "determinism_inference_dtype": "float32", "device_type": "cpu",
    }
    assert report["ddp"]["static_graph"] is False
    assert report["ddp"]["find_unused_parameters"] is True
    assert any(row["pool"] == "mixed_accumulation" for row in report["checks"])
    if stage == "refine":
        assert all(row["refiner_grad_norm"] > 0 for row in report["checks"])
    config["probe_batch_size"] = 2
    config["probe_include_mixed"] = True
    config["training"]["amp"] = True
    runtime.probe(config, model, Dataset(), tmp_path, torch.device("cpu"), batches=3)
    probe_report = json.loads((tmp_path / "gradient-probe.json").read_text())
    assert probe_report["precision"] == {
        "forward_autocast": True, "forward_dtype": "bfloat16",
        "gradient_inner_product_dtype": "float32", "device_type": "cpu",
    }
    assert all(row["batch_size"] == 2
               for row in probe_report["measurements"] if row["pool"] != "mixed")
    mixed = probe_report["measurements"][-1]
    assert mixed["pool"] == "mixed"
    assert mixed["batch_size"] == (3 if stage == "coarse" else 2)
    assert any(mixed["sample_has_jt"]) and not all(mixed["sample_has_jt"])
