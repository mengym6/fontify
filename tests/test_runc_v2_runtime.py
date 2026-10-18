"""Runtime policy and aggregation tests independent of the full CUDA model."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.runc_v2_runtime import (
    REFERENCE_POLICY,
    SCHEMA,
    _reduce_logs,
    assert_clean_checkpoint_lineage,
    assert_dataset_reference_policy,
    assert_distinct_references,
    author_probe_indices,
    build_optimizer,
    call_model,
    collate_indices,
    parameter_grams,
    quantiles,
    safe_target,
    sha256_file,
    stage_start_binding,
    summarize_rows,
    validate_calibration,
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
        "data": {"reference_policy": REFERENCE_POLICY},
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


def safe_reference_metadata(count=1, scalar=False):
    """Explicit sealed identities for synthetic samples, never path heuristics."""
    values = {
        "query_original_identity": ["one:" + "a" * 64] * count,
        "reference_original_identity": ["one:" + "b" * 64] * count,
        "query_original_sha256": ["a" * 64] * count,
        "reference_original_sha256": ["b" * 64] * count,
        "query_target_sha256": ["c" * 64] * count,
        "reference_target_sha256": ["d" * 64] * count,
        "reference_policy": REFERENCE_POLICY,
    }
    if scalar:
        return {
            key: value[0] if isinstance(value, list) else value
            for key, value in values.items()
        }
    return values


class RuntimePolicyTest(unittest.TestCase):
    def test_safe_reference_identity_is_required_for_every_runtime_command(self):
        for command in ("probe", "smoke", "evaluate", "stage2-smoke", "audit"):
            config = configuration()
            config["data"].clear()
            with self.subTest(command=command), self.assertRaises(ValueError):
                validate_config(config, command)

    def test_reference_guard_accepts_explicit_safe_single_and_batched_rows(self):
        self.assertEqual(
            assert_distinct_references(safe_reference_metadata())["reference_policy"],
            REFERENCE_POLICY,
        )
        assert_distinct_references(safe_reference_metadata(count=2))
        assert_distinct_references(safe_reference_metadata(scalar=True))

    def test_reference_guard_rejects_missing_metadata_and_self_reference(self):
        for key in safe_reference_metadata():
            bad = safe_reference_metadata()
            del bad[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                assert_distinct_references(bad)
        for query, reference in (
            ("query_original_identity", "reference_original_identity"),
            ("query_original_sha256", "reference_original_sha256"),
            ("query_target_sha256", "reference_target_sha256"),
        ):
            bad = safe_reference_metadata()
            bad[reference] = bad[query]
            with (
                self.subTest(equal=query),
                self.assertRaisesRegex(ValueError, "self-reference"),
            ):
                assert_distinct_references(bad)

    def test_reference_guard_rejects_cardinality_and_hash_identity_disagreement(self):
        bad = safe_reference_metadata(count=2)
        bad["reference_target_sha256"] = ["d" * 64]
        with self.assertRaisesRegex(ValueError, "cardinality"):
            assert_distinct_references(bad)
        bad = safe_reference_metadata()
        bad["query_original_identity"] = ["one:" + "e" * 64]
        with self.assertRaisesRegex(ValueError, "disagree"):
            assert_distinct_references(bad)

    def test_dataset_guard_rejects_legacy_view(self):
        with self.assertRaises(ValueError):
            assert_dataset_reference_policy(object())

    def test_old_adapted_checkpoint_cannot_reenter_clean_trajectory(self):
        saved = {"runc_v2_metadata": {"initial_checkpoint_sha256": "a" * 64}}
        with self.assertRaisesRegex(ValueError, "lineage"):
            assert_clean_checkpoint_lineage(saved)
        saved["runc_v2_metadata"]["reference_policy"] = REFERENCE_POLICY
        with self.assertRaisesRegex(ValueError, "lineage"):
            assert_clean_checkpoint_lineage(saved)
        saved["runc_v2_runtime"] = {"data": {"reference_policy": REFERENCE_POLICY}}
        assert_clean_checkpoint_lineage(saved)

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
                                **safe_reference_metadata(),
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
                            "reference_policy": REFERENCE_POLICY,
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
                        "reference_policy": REFERENCE_POLICY,
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
            evaluation.write_text(
                json.dumps(
                    {
                        "metadata": {"reference_policy": REFERENCE_POLICY},
                    }
                )
            )
            config["checkpoint"] = str(checkpoint)
            record = Path(folder) / "decision.json"
            record.write_text(
                json.dumps(
                    {
                        "status": "technical_gate_passed",
                        "reference_policy": REFERENCE_POLICY,
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
    def test_call_boundary_rejects_self_reference_before_forward(self):
        calls = []

        def wrapped(*args, **kwargs):
            calls.append(True)

        batch = {
            **safe_reference_metadata(),
            "image": torch.rand(1, 3, 8, 4),
            "target": torch.rand(1, 3, 8, 4),
        }
        batch["reference_target_sha256"] = batch["query_target_sha256"]
        with self.assertRaisesRegex(ValueError, "self-reference"):
            call_model(wrapped, batch)
        self.assertEqual(calls, [])

    def test_collation_checks_dataset_and_each_sample_identity(self):
        class Dataset:
            reference_policy = REFERENCE_POLICY

            def get_pair(self, index, augmentation_seed=None):
                return {
                    **safe_reference_metadata(scalar=True),
                    "image": torch.rand(3, 8, 4),
                    "target": torch.rand(3, 8, 4),
                    "target_path": "query.png",
                    "reference_path": "reference.png",
                }

        dataset = Dataset()
        batch = collate_indices(dataset, [0, 1], torch.device("cpu"))
        self.assertEqual(len(batch["query_original_identity"]), 2)
        self.assertEqual(
            assert_distinct_references(batch)["reference_policy"], REFERENCE_POLICY
        )
        dataset.reference_policy = "old-unrestricted-pairing"
        with self.assertRaisesRegex(ValueError, "dataset"):
            collate_indices(dataset, [0], torch.device("cpu"))

        class LeakingDataset(Dataset):
            def get_pair(self, index, augmentation_seed=None):
                row = super().get_pair(index, augmentation_seed)
                row["reference_original_identity"] = row["query_original_identity"]
                return row

        with self.assertRaisesRegex(ValueError, "self-reference"):
            collate_indices(LeakingDataset(), [0], torch.device("cpu"))

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


@unittest.skipIf(torch is None, "PyTorch is unavailable in the default Python")
class ResumeBindingTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.start = self.root / "stage-start.pth"
        self.start.write_bytes(b"verified original stage-start checkpoint")
        self.fresh = configuration()
        self.fresh.update(
            {
                "checkpoint": str(self.start),
                "stage_start_checkpoint": str(self.start),
                "weights": {"head": 1.0, "head_geo": 0.2},
            }
        )
        probe = self.root / "probe.json"
        probe.write_text(
            json.dumps(
                {
                    "status": "measured",
                    "stage": "ia1",
                    "authors": list("123456789"),
                    "rows": [
                        {
                            **safe_reference_metadata(),
                            "gradients": {
                                "all_trainable": {
                                    "norms": {
                                        "head": 0.3,
                                        "head_logsigma": 0.2,
                                        "head_shape": 0.1,
                                    }
                                },
                            },
                        }
                        for _ in range(9)
                    ],
                    "median_raw_norm": {"head": 0.3},
                    "amp": True,
                    "metadata": {
                        "reference_policy": REFERENCE_POLICY,
                        "initial_checkpoint_sha256": "a" * 64,
                        "checkpoint_sha256": sha256_file(self.start),
                    },
                }
            )
        )
        selection = self.root / "calibration.json"
        selection.write_text(
            json.dumps(
                {
                    "status": "selected_from_measured_gradients",
                    "stage": "ia1",
                    "reference_policy": REFERENCE_POLICY,
                    "initial_checkpoint_sha256": "a" * 64,
                    "selected_weights": self.fresh["weights"],
                    "rationale": "synthetic measured calibration for CPU resume test",
                    "source_probe": str(probe),
                    "source_probe_sha256": sha256_file(probe),
                }
            )
        )
        self.fresh["calibration_record"] = str(selection)
        marker = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([marker], lr=0.001)
        marker.grad = torch.tensor([0.1])
        optimizer.step()
        self.payload = {
            "experiment": "runc_v2",
            "stage": "ia1",
            "optimizer_step": 1,
            "model": {"marker": marker.detach()},
            "optimizer": optimizer.state_dict(),
            "runc_v2_runtime": copy.deepcopy(self.fresh),
            "runc_v2_metadata": {
                "reference_policy": REFERENCE_POLICY,
                "initial_checkpoint_sha256": "a" * 64,
                "checkpoint_sha256": sha256_file(self.start),
                "stage_start_checkpoint_sha256": sha256_file(self.start),
                "calibration": {"calibration_sha256": sha256_file(selection)},
            },
        }
        self.snapshot = self.root / "checkpoint-step-000001.pth"
        torch.save(self.payload, self.snapshot)
        self.resume = {
            **copy.deepcopy(self.fresh),
            "checkpoint": str(self.snapshot),
            "resume_optimizer": True,
        }

    def test_resume_uses_saved_stage_start_instead_of_midstage_checkpoint(self):
        validated = validate_config(self.resume)
        binding = validate_calibration(validated)
        self.assertEqual(
            binding["stage_start_checkpoint_sha256"], sha256_file(self.start)
        )
        self.assertEqual(
            binding["resumed_from_checkpoint_sha256"], sha256_file(self.snapshot)
        )
        self.assertNotEqual(sha256_file(self.start), sha256_file(self.snapshot))

    def test_resume_rejects_forged_start_and_changed_weights(self):
        other = self.root / "other-start.pth"
        other.write_bytes(b"unrelated initial checkpoint")
        config = {**self.resume, "stage_start_checkpoint": str(other)}
        with self.assertRaisesRegex(ValueError, "stage-start"):
            validate_config(config)
        config = copy.deepcopy(self.resume)
        config["weights"]["head_geo"] = 0.4
        with self.assertRaisesRegex(ValueError, "selected loss weights"):
            stage_start_binding(config)

    def test_resume_rejects_changed_calibration_file_even_with_equal_weights(self):
        substitute = self.root / "substitute-calibration.json"
        substitute.write_text(Path(self.fresh["calibration_record"]).read_text() + "\n")
        config = {**self.resume, "calibration_record": str(substitute)}
        with self.assertRaisesRegex(ValueError, "calibration file SHA"):
            validate_config(config)

    def test_resume_rejects_other_stage_and_old_run1_policy(self):
        with self.assertRaisesRegex(ValueError, "same-stage"):
            stage_start_binding({**self.resume, "stage": "ib"})
        del self.payload["runc_v2_metadata"]["reference_policy"]
        torch.save(self.payload, self.snapshot)
        with self.assertRaisesRegex(ValueError, "clean reference-policy"):
            validate_config(self.resume)

    def test_repeated_resume_keeps_the_original_stage_start_binding(self):
        self.payload["runc_v2_runtime"] = copy.deepcopy(self.resume)
        self.payload["runc_v2_metadata"]["checkpoint_sha256"] = sha256_file(
            self.snapshot
        )
        second = self.root / "checkpoint-step-000002.pth"
        torch.save(self.payload, second)
        validated = validate_config({**self.resume, "checkpoint": str(second)})
        self.assertEqual(
            validate_calibration(validated)["stage_start_checkpoint_sha256"],
            sha256_file(self.start),
        )

    def test_fresh_binding_remains_strict(self):
        other = self.root / "other-fresh-start.pth"
        other.write_bytes(b"different fresh origin")
        with self.assertRaisesRegex(ValueError, "fresh stage_start_checkpoint"):
            validate_config({**self.fresh, "stage_start_checkpoint": str(other)})


if __name__ == "__main__":
    unittest.main()
