"""CPU-only callback contract tests; real GPU inference is validated separately."""

import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


MODULE = Path(__file__).resolve().parents[1] / "live_visualization.py"
sys.path.insert(0, str(MODULE.parent))
SPEC = importlib.util.spec_from_file_location("live_visualization", MODULE)
live = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(live)


class FakeTorch:
    def __init__(self):
        self.cpu = [11]
        self.gpu = [[22], [33]]
        self.random = SimpleNamespace(
            get_rng_state=lambda: self.cpu.copy(),
            set_rng_state=lambda state: setattr(self, "cpu", state.copy()),
        )
        self.cuda = SimpleNamespace(
            is_initialized=lambda: True,
            get_rng_state_all=lambda: [state.copy() for state in self.gpu],
            set_rng_state_all=lambda states: setattr(self, "gpu", [state.copy() for state in states]),
        )


class FakeWriter:
    def __init__(self):
        self.scalars = []
        self.images = []

    def add_scalar(self, key, value, step):
        self.scalars.append((key, value, step))

    def add_image(self, key, value, step, **kwargs):
        self.images.append((key, value.copy(), step))

    def flush(self):
        pass

    def close(self):
        pass


class ArrayTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def tolist(self):
        return self.value.tolist()


class Boxes:
    cls = ArrayTensor([0, 50])
    conf = ArrayTensor([.8, .9])

    def __len__(self):
        return 2


class Masks:
    def __init__(self, height, width):
        mask = np.zeros((2, height, width), dtype=bool)
        mask[0, 3:10, 3:20] = True
        mask[1, 2:25, 2:25] = True
        self.data = ArrayTensor(mask)

    def __len__(self):
        return 2


class LiveVisualizationTests(unittest.TestCase):
    def make_data(self, root):
        images = []
        for index in range(6):
            image = root / "images/val" / f"glyph_{index}.png"
            label = root / "labels/val" / f"glyph_{index}.txt"
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (48, 40), (index, index, index)).save(image)
            label.write_text("0 .1 .1 .4 .1 .4 .2\n50 .1 .1 .7 .1 .7 .8\n")
            images.append(image)
        return {"path": str(root), "names": dict(enumerate(f"class_{i}" for i in range(70)))}, images

    def test_schedule_first_interval_last_and_early_stop(self):
        self.assertEqual([n for n in range(1, 13) if live.should_preview(n, 12, 5)], [1, 5, 10, 12])
        self.assertEqual([n for n in range(8, 13) if live.should_preview(n, 12, 5, first=8)], [8, 10, 12])
        self.assertTrue(live.should_preview(3, 100, 5, stopping=True))

    def test_gt_odd_pixel_boundaries_use_truncation_not_rounding(self):
        with tempfile.TemporaryDirectory() as temp:
            label = Path(temp) / "odd.txt"
            points = np.array([[1, 1], [3, 1], [3, 3], [1, 3]], dtype=float)
            coordinates = (points + .5) / 8
            label.write_text("0 " + " ".join(f"{value:.8f}" for value in coordinates.ravel()))
            actual = live.gt_instances(label, 8, 8)[0]["mask"]
            expected = np.zeros((8, 8), dtype=bool)
            expected[1:4, 1:4] = True
            np.testing.assert_array_equal(actual, expected)

    def test_real_validation_txt_matches_training_rasterizer(self):
        import cv2
        dataset = MODULE.parent / "dataset"
        labels = sorted((dataset / "labels/val").glob("*.txt"))
        if not labels:
            self.skipTest("Packaged dataset not available")
        label = labels[0]
        image_path = dataset / "images/val" / (label.stem + ".png")
        with Image.open(image_path) as image:
            width, height = image.size
        instances = live.gt_instances(label, height, width)
        rows = [line.split() for line in label.read_text().splitlines() if line.strip()]
        self.assertEqual(len(instances), len(rows))
        for instance, row in zip(instances, rows):
            coordinates = np.asarray(row[1:], dtype=np.float32).reshape(-1, 2)
            coordinates = (coordinates * np.asarray([width, height], dtype=np.float32)).astype(np.int32)
            expected = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(expected, [coordinates], 1)
            np.testing.assert_array_equal(instance["mask"], expected.astype(bool))

    def test_rng_restored_on_success_and_exception(self):
        torch = FakeTorch()
        state_py, state_np = random.getstate(), np.random.get_state()
        with patch.dict("sys.modules", {"torch": torch}):
            with self.assertRaisesRegex(ValueError, "intentional"):
                with live.preserve_rng():
                    random.random()
                    np.random.rand(10)
                    torch.cpu[0] = 100
                    torch.gpu[0][0] = 200
                    raise ValueError("intentional")
        self.assertEqual(random.getstate(), state_py)
        self.assertEqual(np.random.get_state()[0], state_np[0])
        np.testing.assert_array_equal(np.random.get_state()[1], state_np[1])
        self.assertEqual(torch.cpu, [11])
        self.assertEqual(torch.gpu, [[22], [33]])

    def test_manifest_reused_and_label_change_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, images = self.make_data(root)
            path = root / "run/previews/fixed_samples.json"
            state = random.getstate()
            first = live.fixed_samples(root, images, 4, 42, path)
            second = live.fixed_samples(root, images[::-1], 4, 42, path)
            self.assertEqual(first, second)
            self.assertEqual(random.getstate(), state)
            (root / first["samples"][0]["label"]).write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                live.fixed_samples(root, images, 4, 42, path)

    def test_old_four_image_manifest_is_reused_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, images = self.make_data(root)
            path = root / "run/previews/fixed_samples.json"
            created = live.fixed_samples(root, images, 4, 42, path)
            legacy = {"schema": "calli-fixed-validation-preview-v1", "count": 4, "seed": 42,
                      "samples": [{k: v for k, v in row.items() if k != "author"}
                                  for row in created["samples"]]}
            path.write_text(json.dumps(legacy, indent=2))
            original_bytes = path.read_bytes()
            recovered = live.fixed_samples(root, images, 45, 123, path)
            self.assertEqual(recovered, legacy)
            self.assertEqual(path.read_bytes(), original_bytes)

    def test_author_stratified_45_and_shortage_redistribution(self):
        for short_author in (False, True):
            with self.subTest(short_author=short_author), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                images, rows = [], []
                for author in range(9):
                    for index in range(2 if short_author and author == 0 else 6):
                        # Deliberately use a common filename prefix: grouping
                        # must use manifest authors, not the filename fallback.
                        image = root / "images/val" / f"glyph_{author}_{index}.png"
                        label = root / "labels/val" / (image.stem + ".txt")
                        image.parent.mkdir(parents=True, exist_ok=True)
                        label.parent.mkdir(parents=True, exist_ok=True)
                        Image.new("RGB", (8, 8), "white").save(image)
                        label.write_text("0 .1 .1 .8 .1 .5 .8\n")
                        images.append(image)
                        rows.append({"image": str(image.relative_to(root)), "author": f"author{author}"})
                (root / "manifest.json").write_text(json.dumps({"images": rows}))
                state = random.getstate()
                result = live.fixed_samples(root, images, 45, 42, root / "fixed.json")
                self.assertEqual(random.getstate(), state)
                self.assertEqual(result["count"], 45)
                self.assertEqual(len({row["image"] for row in result["samples"]}), 45)
                self.assertEqual(len(result["selected_per_author"]), 9)
                self.assertEqual(result["target_per_author"], 5)
                if short_author:
                    self.assertEqual(result["selected_per_author"]["author0"], 2)
                    self.assertTrue(all(result["selected_per_author"][f"author{i}"] >= 5
                                        for i in range(1, 9)))
                else:
                    self.assertEqual(set(result["selected_per_author"].values()), {5})

    def test_integrations_are_process_local_and_restored(self):
        class Settings(dict):
            def __setitem__(self, key, value):
                raise AssertionError("Persistence-aware mutation must not be called")
        settings = Settings(tensorboard=True, wandb=True, mlflow=True)
        module = SimpleNamespace(SETTINGS=settings)
        with patch.dict("sys.modules", {"ultralytics.utils": module}):
            with live.isolated_logging_integrations():
                self.assertFalse(settings["tensorboard"])
                self.assertFalse(settings["wandb"])
                self.assertTrue(settings["mlflow"])
        self.assertEqual(settings, {"tensorboard": True, "wandb": True, "mlflow": True})

    def test_nonprimary_rank_does_not_initialize_or_log(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"RANK": "1"}):
            root = Path(temp)
            data, images = self.make_data(root)
            callback = live.LiveVisualization(data, images, tensorboard=True)
            callback.on_start(SimpleNamespace(save_dir=root / "run"))
            callback.on_fit_end(SimpleNamespace())
            self.assertFalse((root / "run").exists())

    def test_final_best_hook_does_not_overwrite_epoch_preview(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"RANK": "-1"}):
            root = Path(temp)
            data, images = self.make_data(root)
            callback = live.LiveVisualization(data, images, tensorboard=False)
            writer = callback.writer = FakeWriter()
            previews = []
            callback._preview = lambda trainer, step: previews.append(step)
            trainer = SimpleNamespace(epoch=0, epochs=12, start_epoch=0, stop=False,
                                      tloss=None, lr={"lr/pg0": .1}, metrics={"metrics/mAP50(M)": .2},
                                      label_loss_items=lambda values, prefix: {prefix + "/seg_loss": 1.0})
            callback.on_epoch_start(trainer)
            callback.on_fit_end(trainer)
            callback.on_fit_end(trainer)
            trainer.epoch += 1  # native final_eval best.pt reporting hook
            callback.on_fit_end(trainer)
            self.assertEqual(previews, [1])
            self.assertEqual(sum(key == "train/seg_loss" for key, _, _ in writer.scalars), 1)
            self.assertTrue(any(key.startswith("final_best/") for key, _, _ in writer.scalars))

    def test_current_checkpoint_per_image_prediction_grouping_and_rng(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"RANK": "-1"}):
            root = Path(temp)
            data, images = self.make_data(root)
            run = root / "run"
            last = run / "weights/last.pt"
            last.parent.mkdir(parents=True)
            last.write_bytes(b"fake checkpoint for callback contract")
            callback = live.LiveVisualization(data, images, count=4, tensorboard=False)
            trainer = SimpleNamespace(save_dir=run, last=last, epoch=4, device="cpu", args=SimpleNamespace(imgsz=640))
            callback.on_start(trainer)
            writer = callback.writer = FakeWriter()
            torch, calls = FakeTorch(), []
            expected_checkpoint = str(last)
            class IndependentYOLO:
                def __init__(self, path):
                    self.ckpt = {"epoch": 4}
                    assert path == expected_checkpoint
                def predict(self, **kwargs):
                    calls.append(kwargs)
                    assert isinstance(kwargs["source"], str)
                    random.random()
                    np.random.rand()
                    torch.cpu[0] = 999
                    torch.gpu[0][0] = 888
                    return [SimpleNamespace(path=kwargs["source"], orig_shape=(40, 48),
                                            boxes=Boxes(), masks=Masks(40, 48))]
            py_state = random.getstate()
            with patch.dict("sys.modules", {"torch": torch, "ultralytics": SimpleNamespace(YOLO=IndependentYOLO)}):
                callback._preview(trainer, 5)
            self.assertEqual(len(calls), len(images))
            self.assertEqual(len({call["source"] for call in calls}), len(images))
            for call in calls:
                self.assertIsInstance(call["source"], str)
                self.assertEqual(call["batch"], 1)
                self.assertNotIn("classes", call)
            self.assertEqual(random.getstate(), py_state)
            self.assertEqual(torch.cpu, [11])
            self.assertEqual(torch.gpu, [[22], [33]])
            self.assertEqual(len(writer.images), 4)
            metadata = json.loads((run / "previews/epoch_005/preview.json").read_text())
            self.assertEqual(metadata["checkpoint_epoch_zero_based"], 4)
            self.assertEqual(metadata["evaluated_images"], len(images))
            self.assertEqual(metadata["preview_images"], 4)
            self.assertEqual(metadata["samples"][0]["pred_instances"], 2)
            report = json.loads((run / "region_metrics/epoch_005.json").read_text())
            self.assertTrue(report["full_validation"])
            self.assertEqual(len(report["per_image"]), len(images))
            self.assertEqual(report["summary"]["BF"]["gt_count"], len(images))
            self.assertEqual(report["summary"]["JT"]["pred_count"], len(images))
            scalar_names = {key for key, _, _ in writer.scalars}
            self.assertIn("overview/BF/miss_rate_pct", scalar_names)
            self.assertIn("diagnostics/BF/gt_count", scalar_names)
            if report["summary"]["BF"]["label_accuracy"] is None:
                self.assertNotIn("overview/BF/label_accuracy_pct", scalar_names)
            for file in (run / "previews/epoch_005").glob("*.png"):
                with Image.open(file) as image:
                    self.assertEqual(image.size, (48 * 5, 40 + 88))

    def test_disabling_images_keeps_periodic_full_val_metrics_hook(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, images = self.make_data(root)
            callback = live.LiveVisualization(data, images, count=0, tensorboard=False)
            calls = []
            callback._preview = lambda trainer, step: calls.append(step)
            trainer = SimpleNamespace(epoch=0, epochs=100, start_epoch=0, stop=False,
                                      tloss=None, lr={}, metrics={},
                                      label_loss_items=lambda values, prefix: {})
            callback.on_epoch_start(trainer)
            callback.on_fit_end(trainer)
            trainer.epoch = 1
            callback.on_epoch_start(trainer)
            callback.on_fit_end(trainer)
            self.assertEqual(calls, [1])

    def test_stale_or_stripped_checkpoint_rejected(self):
        for checkpoint in (None, {"epoch": -1}, {"epoch": 3}):
            with self.assertRaisesRegex(RuntimeError, "Preview requires the current"):
                live.checkpoint_matches_epoch(checkpoint, 4)
        live.checkpoint_matches_epoch({"epoch": 4}, 4)


if __name__ == "__main__":
    unittest.main()
