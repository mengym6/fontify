"""Runtime policy and aggregation tests independent of the full CUDA model."""

import json
import tempfile
import unittest
from pathlib import Path

from tools.runc_v2_runtime import (
    SCHEMA,
    _reduce_logs,
    author_probe_indices,
    build_optimizer,
    parameter_grams,
    quantiles,
    safe_target,
    sha256_file,
    summarize_rows,
    validate_config,
    validate_stage_gate,
)

try:
    import torch
except ImportError:
    torch = None


def configuration():
    return {
        "schema": SCHEMA,
        "experiment": "runc_v2",
        "stage": "ia1",
        "checkpoint": "checkpoint-14.pth",
        "initial_checkpoint_sha256": "a" * 64,
        "data": {},
        "calibration_record": "measured-probe.json",
        "training": {
            "seed": 0,
            "steps": 3,
            "batch_size": 1,
            "accum_iter": 2,
            "save_every": 1,
            "lr_head": 0.001,
            "lr_vit": 0.00001,
            "lr_coarse": 0.001,
            "clip_grad": 3.0,
            "weight_decay": 0.01,
            "amp": True,
        },
    }


class RuntimePolicyTest(unittest.TestCase):
    def test_stage_two_cannot_be_trained(self):
        config = configuration()
        config["stage"] = "refine"
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_pretrained_learning_rate_is_separate_and_lower(self):
        config = configuration()
        config["training"]["lr_vit"] = config["training"]["lr_head"]
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_formal_training_needs_measured_probe_reference(self):
        config = configuration()
        del config["calibration_record"]
        with self.assertRaises(ValueError):
            validate_config(config)
        self.assertEqual(validate_config(config, "smoke")["stage"], "ia1")

    def test_old_v1_objective_rejected(self):
        config = configuration()
        config["lowpass"] = {"sigma": 1.5}
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_pair_and_unique_denominators_differ(self):
        rows = [
            {"identity": "A", "author": "one", "character": "x", "error": 0},
            {"identity": "A", "author": "one", "character": "x", "error": 2},
            {"identity": "B", "author": "two", "character": "y", "error": 4},
        ]
        summary = summarize_rows(rows, ["error"])
        self.assertEqual(summary["pair_count"], 3)
        self.assertEqual(summary["unique_target_count"], 2)
        self.assertEqual(summary["pairs"]["error"]["mean"], 2)
        self.assertEqual(summary["unique_targets"]["error"]["mean"], 2.5)

    def test_tail_quantiles_are_linear_and_finite(self):
        self.assertAlmostEqual(quantiles([0, 10])["p95"], 9.5)
        with self.assertRaises(ValueError):
            quantiles([float("nan")])

    def test_probe_covers_each_source(self):
        class Dataset:
            pairs = [{"calligrapher": str(i)} for i in range(9)]

        rows = author_probe_indices(Dataset(), 2, repeats=2)
        self.assertEqual(len(rows), 18)
        self.assertEqual(len({author for author, _ in rows}), 9)
        self.assertTrue(all(len(indices) == 2 for _, indices in rows))

    def test_measured_calibration_is_sha_and_weight_bound(self):
        config = configuration()
        config["weights"] = {"head": 1.0, "head_geo": 0.2}
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "checkpoint.pth"
            checkpoint.write_bytes(b"test checkpoint bytes")
            config["checkpoint"] = str(checkpoint)
            probe = Path(folder) / "probe.json"
            probe.write_text(
                json.dumps(
                    {
                        "status": "measured",
                        "stage": "ia1",
                        "rows": [
                            {
                                "batch": 0,
                                "gradients": {
                                    "all_trainable": {
                                        "norms": {
                                            "head": 0.3,
                                            "head_logsigma": 0.2,
                                            "head_shape": 0.1,
                                        }
                                    }
                                },
                            }
                        ],
                        "authors": list("123456789"),
                        "median_raw_norm": {"head": 0.3},
                        "metadata": {
                            "initial_checkpoint_sha256": "a" * 64,
                            "checkpoint_sha256": sha256_file(checkpoint),
                        },
                        "amp": True,
                    }
                )
            )
            selection = Path(folder) / "selection.json"
            selection.write_text(
                json.dumps(
                    {
                        "status": "selected_from_measured_gradients",
                        "stage": "ia1",
                        "initial_checkpoint_sha256": "a" * 64,
                        "selected_weights": config["weights"],
                        "rationale": "matched measured geometry to head budget",
                        "source_probe": str(probe),
                        "source_probe_sha256": sha256_file(probe),
                    }
                )
            )
            config["calibration_record"] = str(selection)
            self.assertEqual(validate_config(config)["stage"], "ia1")
            config["weights"]["head_geo"] = 0.4
            with self.assertRaises(ValueError):
                validate_config(config)
            config["weights"]["head_geo"] = 0.2
            broken = json.loads(probe.read_text())
            broken["rows"][0]["gradients"]["all_trainable"]["norms"]["head_shape"] = 0
            probe.write_text(json.dumps(broken))
            record = json.loads(selection.read_text())
            record["source_probe_sha256"] = sha256_file(probe)
            selection.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "head_shape"):
                validate_config(config)

    def test_ib_gate_binds_prior_snapshot_and_evaluation(self):
        config = configuration()
        config["stage"] = "ib"
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "ia1.pth"
            checkpoint.write_bytes(b"actual IA1 snapshot")
            evaluation = Path(folder) / "evaluation.json"
            evaluation.write_text("{}")
            config["checkpoint"] = str(checkpoint)
            record = Path(folder) / "decision.json"
            record.write_text(
                json.dumps(
                    {
                        "status": "technical_gate_passed",
                        "source_stage": "ia1",
                        "stage": "ib",
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "supporting_evaluations": [
                            {
                                "path": str(evaluation),
                                "sha256": sha256_file(evaluation),
                            }
                        ],
                        "rationale": "Head occupancy and scale improved in train/val",
                    }
                )
            )
            config["stage_gate_record"] = str(record)
            self.assertIsNotNone(validate_stage_gate(config))
            evaluation.write_text('{"changed": true}')
            with self.assertRaises(ValueError):
                validate_stage_gate(config)


@unittest.skipIf(torch is None, "PyTorch is unavailable in the default Python")
class RuntimeTensorTest(unittest.TestCase):
    def test_safe_target_ignores_lower_gt_without_mutating_batch(self):
        target = torch.rand(1, 3, 8, 4)
        original = target.clone()
        safe = safe_target({"target": target})
        self.assertTrue(torch.equal(safe[:, :, :4], target[:, :, :4]))
        self.assertEqual(int(torch.count_nonzero(safe[:, :, 4:])), 0)
        self.assertTrue(torch.equal(target, original))

    def test_parameter_grams_measure_actual_orthogonal_gradients(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        terms = {
            "left": (parameter * torch.tensor([3.0, 4.0])).sum(),
            "right": (parameter * torch.tensor([-4.0, 3.0])).sum(),
        }
        measured = parameter_grams(terms, [("parameter", parameter)])
        self.assertEqual(measured["gram"], [[25.0, 0.0], [0.0, 25.0]])
        self.assertEqual(measured["norms"], {"left": 5.0, "right": 5.0})
        self.assertEqual(measured["cosine"]["right:left"], 0.0)
        (terms["left"] + terms["right"]).backward()
        self.assertTrue(torch.equal(parameter.grad, torch.tensor([-1.0, 7.0])))

    def test_training_counts_sum_while_losses_average(self):
        logs = [{"count_jt": 2.0, "total": 1.0}, {"count_jt": 2.0, "total": 3.0}]
        reduced = _reduce_logs(logs, torch.device("cpu"), world=1)
        self.assertEqual(reduced["count_jt"], 4)
        self.assertEqual(reduced["total"], 2)

    def test_optimizer_separates_lr_and_excludes_frozen_parameters(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dual_head = torch.nn.Linear(2, 2)
                self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
                self.runc_v2_renderer = torch.nn.Linear(2, 2)
                self.decoder_embed = torch.nn.Linear(2, 2)
                self.decoder_embed.requires_grad_(False)

        model = Tiny()
        settings = configuration()["training"]
        optimizer, names = build_optimizer(model, settings)
        self.assertEqual(set(names), {"head", "vit", "coarse"})
        rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
        self.assertLess(rates["vit"], rates["head"])
        membership = {
            id(p) for group in optimizer.param_groups for p in group["params"]
        }
        self.assertTrue(
            all(id(p) not in membership for p in model.decoder_embed.parameters())
        )

    def test_calibration_rejects_nonfinite_probe(self):
        config = configuration()
        config["weights"] = {}
        with tempfile.TemporaryDirectory() as folder:
            probe = Path(folder) / "probe.json"
            probe.write_text(
                json.dumps(
                    {
                        "status": "measured",
                        "stage": "ia1",
                        "rows": [{"batch": 0}],
                        "authors": list("123456789"),
                        "median_raw_norm": {"head": float("nan")},
                        "metadata": {"initial_checkpoint_sha256": "a" * 64},
                    }
                )
            )
            selection = Path(folder) / "selection.json"
            selection.write_text(
                json.dumps(
                    {
                        "status": "selected_from_measured_gradients",
                        "stage": "ia1",
                        "initial_checkpoint_sha256": "a" * 64,
                        "selected_weights": {},
                        "rationale": "unit test",
                        "source_probe": str(probe),
                        "source_probe_sha256": sha256_file(probe),
                    }
                )
            )
            config["calibration_record"] = str(selection)
            with self.assertRaises(ValueError):
                validate_config(config)


if __name__ == "__main__":
    unittest.main()
