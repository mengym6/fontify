"""Hyperparameter forwarding and resume contracts without loading torch/models."""

import argparse
from contextlib import redirect_stdout, nullcontext
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


MODULE = Path(__file__).resolve().parents[1] / "train.py"
sys.path.insert(0, str(MODULE.parent))
SPEC = importlib.util.spec_from_file_location("calli_training_args", MODULE)
train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train)


DEFAULTS = {
    "optimizer": "auto", "lr0": .01, "lrf": .01, "weight_decay": .0005,
    "warmup_epochs": 3.0, "save_period": -1, "cos_lr": False, "amp": True,
    "box": 7.5, "cls": .5, "dfl": 1.5, "translate": .1, "scale": .5,
    "shear": 0.0, "degrees": 0.0, "freeze": 0, "mask_ratio": 4,
    "warmup_bias_lr": .1, "momentum": .937, "hsv_h": .015, "hsv_s": .7, "hsv_v": .4,
    "mosaic": 1.0, "fliplr": 0.0, "flipud": 0.0, "perspective": 0.0,
    "epochs": 60, "batch": 24, "imgsz": 640, "seed": 42, "patience": 0, "plots": False,
}
MANUAL_ARGUMENTS = [
    "--optimizer", "AdamW", "--lr0", ".0003", "--lrf", ".1",
    "--weight-decay", ".002", "--warmup-epochs", "1.5", "--save-period", "5",
    "--cos-lr", "--no-amp",
    "--box-weight", "5", "--cls-weight", ".7", "--dfl-weight", "1.2",
    "--translate", ".05", "--scale", ".15", "--shear", "1", "--degrees", "0",
    "--freeze", "4", "--mask-ratio", "2",
    "--warmup-bias-lr", "0", "--momentum", ".9", "--hsv-h", ".01",
    "--hsv-s", ".2", "--hsv-v", ".15", "--mosaic", ".5",
]
MANUAL_SETTINGS = {
    "optimizer": "AdamW", "lr0": .0003, "lrf": .1, "weight_decay": .002,
    "warmup_epochs": 1.5, "save_period": 5, "cos_lr": True, "amp": False,
    "box": 5.0, "cls": .7, "dfl": 1.2, "translate": .05, "scale": .15,
    "shear": 1.0, "degrees": 0.0, "freeze": 4, "mask_ratio": 2,
    "warmup_bias_lr": 0.0, "momentum": .9, "hsv_h": .01, "hsv_s": .2, "hsv_v": .15, "mosaic": .5,
}


class TrainingArgumentTests(unittest.TestCase):
    def make_input(self, root):
        for split in ("train", "val"):
            image = root / "images" / split / "glyph.png"
            label = root / "labels" / split / "glyph.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.touch()
            label.write_text("0 .1 .1 .8 .1 .5 .8\n")
        data = root / "data.yaml"
        data.write_text("path: .\ntrain: images/train\nval: images/val\nnames: [phase]\n")
        model = root / "initial.pt"
        model.touch()
        return ["train.py", "--data", str(data), "--model", str(model),
                "--project", str(root / "runs")]

    def test_dry_run_uses_official_augmentation_with_flips_disabled(self):
        with TemporaryDirectory() as temp:
            arguments = self.make_input(Path(temp)) + ["--dry-run"]
            output = io.StringIO()
            with patch("sys.argv", arguments), redirect_stdout(output):
                train.main()
            plan = json.loads(output.getvalue())
            self.assertEqual({key: plan["train_settings"][key] for key in DEFAULTS}, DEFAULTS)
            self.assertIn("ignoring the supplied lr0", plan["optimizer_note"])
            self.assertEqual(plan["visualization"]["tensorboard_root"], str((Path(temp) / "runs/dashboard_full").resolve()))
            self.assertFalse((Path(temp) / "runs").exists())

    def test_dry_run_reports_requested_manual_settings(self):
        with TemporaryDirectory() as temp:
            arguments = self.make_input(Path(temp)) + MANUAL_ARGUMENTS + ["--dry-run"]
            output = io.StringIO()
            with patch("sys.argv", arguments), redirect_stdout(output):
                train.main()
            plan = json.loads(output.getvalue())
            self.assertEqual({key: plan["train_settings"][key] for key in MANUAL_SETTINGS}, MANUAL_SETTINGS)
            self.assertIn("new run only", plan["optimizer_note"])

    def test_launcher_defaults_and_augmentation_overrides_match_cli(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_input(root)
            environment = os.environ.copy()
            for key in ("EPOCHS", "BATCH", "TRANSLATE", "SCALE", "HSV_H", "HSV_S", "HSV_V", "MOSAIC", "SAVE_PERIOD"):
                environment.pop(key, None)
            environment.update(PYTHON_BIN=sys.executable, DATA_YAML=str(root / "data.yaml"),
                               MODEL=str(root / "initial.pt"), PROJECT_DIR=str(root / "runs"),
                               RUN_NAME="argument_check")
            command = ["bash", str(MODULE.parent / "start_train.sh"), "--dry-run"]
            result = subprocess.run(command, env=environment, text=True, capture_output=True, check=True)
            settings = json.loads(result.stdout)["train_settings"]
            for key in ("epochs", "batch", "translate", "scale", "hsv_h", "hsv_s", "hsv_v", "mosaic", "save_period", "plots"):
                self.assertEqual(settings[key], DEFAULTS[key])
            environment.update(HSV_H=".02", HSV_S=".3", HSV_V=".6", MOSAIC=".4")
            result = subprocess.run(command, env=environment, text=True, capture_output=True, check=True)
            settings = json.loads(result.stdout)["train_settings"]
            self.assertEqual({key: settings[key] for key in ("hsv_h", "hsv_s", "hsv_v", "mosaic")},
                             {"hsv_h": .02, "hsv_s": .3, "hsv_v": .6, "mosaic": .4})
            self.assertFalse((root / "runs").exists())

    def test_auxiliary_weights_are_separate_from_native_ultralytics_kwargs(self):
        with TemporaryDirectory() as temp:
            arguments = self.make_input(Path(temp)) + ["--sem-weight", ".25", "--dry-run"]
            output = io.StringIO()
            with patch("sys.argv", arguments), redirect_stdout(output):
                train.main()
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["auxiliary_loss"], {"sem_weight": .25})
            self.assertNotIn("sem_weight", plan["train_settings"])

    def test_resume_inherits_saved_semantic_weight_and_rejects_a_change(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            arguments = self.make_input(root)
            checkpoint = root / "previous_run/weights/last.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            (checkpoint.parent.parent / "loss_config.json").write_text(json.dumps({
                "schema": "calli-yolo26-semantic-loss-v1", "semantic_weight": .25,
            }))
            arguments += ["--resume", str(checkpoint), "--dry-run"]
            output = io.StringIO()
            with patch("sys.argv", arguments), redirect_stdout(output):
                train.main()
            self.assertEqual(json.loads(output.getvalue())["auxiliary_loss"], {"sem_weight": .25})
            with patch("sys.argv", arguments + ["--sem-weight", "1"]), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "changing the loss requires a new run"):
                    train.main()

    def test_invalid_floats_and_ranges_are_rejected(self):
        defaults = train.parser().parse_args([])
        invalid = [
            ("lr0", 0), ("lr0", -1), ("lr0", 1.1), ("lr0", float("nan")),
            ("lrf", -0.1), ("lrf", 1.1), ("lrf", float("inf")),
            ("weight_decay", -0.1), ("weight_decay", 1.1), ("weight_decay", float("nan")),
            ("warmup_epochs", -1), ("warmup_epochs", float("inf")),
            ("save_period", -2),
            ("box_weight", -1), ("cls_weight", float("nan")), ("dfl_weight", -1),
            ("sem_weight", -1), ("warmup_bias_lr", float("inf")),
            ("momentum", 1), ("hsv_h", float("nan")), ("hsv_s", -.1),
            ("hsv_v", 1.1), ("mosaic", 1.01),
            ("translate", 1.1), ("scale", -.1), ("shear", 90), ("degrees", 181),
            ("freeze", -1), ("mask_ratio", 0),
        ]
        for name, value in invalid:
            with self.subTest(name=name, value=value):
                args = argparse.Namespace(**vars(defaults))
                setattr(args, name, value)
                with self.assertRaises(ValueError):
                    train.validate_hyperparameters(args)

    def test_valid_boundary_values_and_short_smoke_are_accepted(self):
        args = train.parser().parse_args([
            "--epochs", "2", "--lr0", "1", "--lrf", "0",
            "--weight-decay", "0", "--warmup-epochs", "3", "--save-period", "-1",
        ])
        train.validate_hyperparameters(args)
        args.save_period = 0
        train.validate_hyperparameters(args)

    def run_with_mock_model(self, root, extra, resume=False):
        calls = []
        class Model:
            def __init__(self, path):
                self.task = "segment"
                self.ckpt = {"epoch": 2, "optimizer": {}}
                self.trainer = SimpleNamespace(save_dir=root / "runs/mock")
            def train(self, **kwargs):
                calls.append(kwargs)
            def add_callback(self, name, callback):
                self.callback = (name, callback)
        class Preview:
            def __init__(self, *args):
                pass
            def attach(self, model):
                pass
            def close(self):
                pass
        arguments = self.make_input(root) + extra
        if resume:
            checkpoint = root / "existing/weights/last.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            arguments += ["--resume", str(checkpoint)]
        modules = {
            "ultralytics": SimpleNamespace(YOLO=Model),
            "live_visualization": SimpleNamespace(
                LiveVisualization=Preview, isolated_logging_integrations=nullcontext),
        }
        output = io.StringIO()
        with patch.dict("sys.modules", modules), patch("sys.argv", arguments), redirect_stdout(output):
            train.main()
        return calls, output.getvalue()

    def test_manual_settings_are_forwarded_to_new_training(self):
        with TemporaryDirectory() as temp:
            calls, _ = self.run_with_mock_model(Path(temp), MANUAL_ARGUMENTS)
            self.assertEqual(len(calls), 1)
            self.assertEqual({key: calls[0][key] for key in MANUAL_SETTINGS}, MANUAL_SETTINGS)
            self.assertTrue(calls[0]["save"])
            self.assertFalse(calls[0]["overlap_mask"])
            self.assertFalse(calls[0]["plots"])
            # Only the YAML actually consumed by Ultralytics/resume remains.
            configs = Path(temp) / "runs/_configs"
            self.assertEqual(len(list(configs.glob("dataset_*.yaml"))), 1)
            self.assertFalse(list(configs.glob("launch_*.json")))

    def test_semantic_weight_is_connected_to_callback_not_native_kwargs(self):
        with TemporaryDirectory() as temp:
            with patch("calli_loss.make_semantic_loss_callback", return_value=object()) as factory:
                calls, _ = self.run_with_mock_model(Path(temp), ["--sem-weight", ".25"])
            factory.assert_called_once_with(.25)
            self.assertNotIn("sem_weight", calls[0])

    def test_resume_does_not_apply_new_hyperparameters(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            calls, output = self.run_with_mock_model(root, MANUAL_ARGUMENTS, resume=True)
            self.assertEqual(len(calls), 1)
            self.assertTrue(calls[0]["resume"])
            self.assertTrue(set(MANUAL_SETTINGS).isdisjoint(calls[0]))
            self.assertNotIn("epochs", calls[0])
            self.assertEqual(calls[0]["save_dir"], str((root / "existing").resolve()))
            self.assertIn("not applied to resume", output)


if __name__ == "__main__":
    unittest.main()
