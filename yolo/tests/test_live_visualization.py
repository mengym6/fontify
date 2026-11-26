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
import cv2
from PIL import Image
from tensorboard.compat.proto import summary_pb2, tensor_pb2, tensor_shape_pb2, types_pb2
from tensorboard.plugins.image import metadata as image_metadata


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
        self.summaries = []
        self.file_writer = self

    def add_scalar(self, key, value, step):
        self.scalars.append((key, value, step))

    def add_image(self, key, value, step, **kwargs):
        self.images.append((key, value.copy(), step))

    def add_summary(self, summary, global_step=None):
        self.summaries.append((summary, global_step))

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
    def make_data(self, root, count=6):
        images = []
        for index in range(count):
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

    def test_author_stratified_60_and_shortage_redistribution(self):
        for short_author in (False, True):
            with self.subTest(short_author=short_author), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                images, rows = [], []
                for author in range(9):
                    for index in range(2 if short_author and author == 0 else 9):
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
                result = live.fixed_samples(root, images, 60, 42, root / "fixed.json", split_id="new-split")
                self.assertEqual(random.getstate(), state)
                self.assertEqual(result["count"], 60)
                self.assertEqual(len({row["image"] for row in result["samples"]}), 60)
                self.assertEqual(len(result["selected_per_author"]), 9)
                self.assertEqual(result["target_per_author"], 6)
                if short_author:
                    self.assertEqual(result["selected_per_author"]["author0"], 2)
                    self.assertTrue(all(result["selected_per_author"][f"author{i}"] >= 7
                                        for i in range(1, 9)))
                else:
                    self.assertEqual(set(result["selected_per_author"].values()), {6, 7})

    def test_new_run_rejects_short_validation_and_foreign_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, images = self.make_data(root)
            path = root / "fixed.json"
            with self.assertRaisesRegex(ValueError, "Need 60"):
                live.fixed_samples(root, images, 60, 42, path, split_id="full")
            self.assertFalse(path.exists())
            live.fixed_samples(root, images, 4, 42, path)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "another dataset split"):
                live.fixed_samples(root, images, 4, 42, path, split_id="full")
            self.assertEqual(path.read_bytes(), before)

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
                                      save_dir=root / "run",
                                      tloss=None, lr={"lr/pg0": .1},
                                      metrics={"metrics/mAP50(M)": .2,
                                               **{f"val/{name}": 2.0 for name in live.COMMON_LOSS_NAMES}},
                                      label_loss_items=lambda values, prefix: {
                                          f"{prefix}/{name}": 1.0 for name in live.COMMON_LOSS_NAMES})
            callback.on_epoch_start(trainer)
            callback.on_fit_end(trainer)
            callback.on_fit_end(trainer)
            trainer.epoch += 1  # native final_eval best.pt reporting hook
            callback.on_fit_end(trainer)
            self.assertEqual(previews, [1])
            self.assertEqual(writer.scalars, [])  # no criterion sidechannel, no invented grouped losses

    def test_group_loss_totals_use_sidechannel_and_never_split_total(self):
        values = live.group_loss_totals({"BF": 2.0, "JT": 7.0, "common_total": 9.0},
                                        {"BF": 3.0, "JT": None, "common_total": 3.0})
        self.assertEqual(values, {"BF/loss/train_combined": 2.0, "JT/loss/train_combined": 7.0,
                                  "BF/loss/val_combined": 3.0, "JT/loss/val_combined": None})
        self.assertIsNone(live.group_loss_totals({"BF": float("nan")}, {})["BF/loss/train_combined"])

    def test_train_and_validation_sidechannels_are_reset_and_read_independently(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        train_model, ema_model = object(), object()
        trainer = SimpleNamespace(model=train_model, ema=SimpleNamespace(ema=ema_model), epoch=0,
                                  epochs=10, start_epoch=0, stop=False, save_dir=Path(temporary.name))
        callback = live.LiveVisualization({"path": ".", "names": {}}, [], count=0, tensorboard=False)
        callback.writer = FakeWriter()
        callback._preview = lambda trainer, step: None
        resets = []
        api = SimpleNamespace(reset_group_loss_accumulators=lambda model: resets.append(model),
                              read_group_loss_accumulators=lambda model: (
                                  {"BF": 2.0, "JT": 5.0} if model is train_model else {"BF": 3.0, "JT": 7.0}))
        with patch.dict("sys.modules", {"partial_supervision": api}):
            callback.on_epoch_start(trainer)
            callback.on_val_start(SimpleNamespace())
            callback.on_fit_end(trainer)
        self.assertEqual(resets, [train_model, ema_model])
        self.assertEqual(dict((key, value) for key, value, _ in callback.writer.scalars),
                         {"BF/loss/train_combined": 2.0, "JT/loss/train_combined": 5.0,
                          "BF/loss/val_combined": 3.0, "JT/loss/val_combined": 7.0})

    def test_dashboard_writer_isolates_history_and_filters_scalar_tags(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "old_run"
            old_event = run / "tensorboard/events.old"
            old_event.parent.mkdir(parents=True)
            old_event.write_bytes(b"historical events")
            callback = live.LiveVisualization({"path": str(root), "names": {}}, [], count=0, tensorboard=False)
            writer, directories = FakeWriter(), []
            def factory(directory):
                directories.append(directory)
                return writer
            callback.writer_type = factory
            callback.on_start(SimpleNamespace(save_dir=run))
            self.assertEqual(directories, [str(root / "dashboard_full/old_run")])
            self.assertEqual(old_event.read_bytes(), b"historical events")
            protocol = json.loads((run / "dashboard_protocol.json").read_text())
            self.assertIn("not the complete E2E", protocol["loss_definition"])
            self.assertEqual(protocol["schema"], "calli-grouped-dashboard-v3")
            self.assertEqual(protocol["image_tag_template"], "previews/epoch_{epoch:03d}")
            values = {key: .5 for key in live.DASHBOARD_SCALAR_TAGS}
            values.update({"train/seg_loss": 99, "val/box_loss": 99, "metrics/mAP50(M)": 99,
                           "lr/pg0": 99, "diagnostics/BF/gt_count": 99,
                           "overview/BF/region_recall_pct": 99, "final_best/metrics/mAP50(M)": 99})
            callback._scalars(values, 1)
            self.assertEqual({key for key, _, _ in writer.scalars}, live.DASHBOARD_SCALAR_TAGS)

    def test_current_checkpoint_per_image_prediction_grouping_and_rng(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"RANK": "-1"}):
            root = Path(temp)
            data, images = self.make_data(root, count=66)
            data["split_id"] = "full-new-split"
            availability = {str(path.relative_to(root)): {"has_bf": i % 2 == 0, "has_jt": i % 2 == 1}
                            for i, path in enumerate(images)}
            (root / "annotation_availability.json").write_text(json.dumps(availability))
            run = root / "run"
            last = run / "weights/last.pt"
            last.parent.mkdir(parents=True)
            last.write_bytes(b"fake checkpoint for callback contract")
            callback = live.LiveVisualization(data, images, count=60, tensorboard=False)
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
            self.assertEqual(len(writer.images), 0)
            self.assertEqual(len(writer.summaries), 1)
            self.assertEqual(len(writer.summaries[0][0].value[0].tensor.string_val), 62)
            self.assertEqual(writer.summaries[0][0].value[0].tag, "previews/epoch_005")
            metadata = json.loads((run / "previews/epoch_005/preview.json").read_text())
            self.assertEqual(metadata["checkpoint_epoch_zero_based"], 4)
            self.assertEqual(metadata["tensorboard_image_tag"], "previews/epoch_005")
            self.assertEqual(metadata["evaluated_images"], len(images))
            self.assertEqual(metadata["preview_images"], 60)
            self.assertEqual(len(list((run / "previews/epoch_005").glob("*.png"))), 60)
            self.assertEqual(metadata["samples"][0]["pred_instances"], 2)
            report = json.loads((run / "region_metrics/epoch_005.json").read_text())
            self.assertTrue(report["full_validation"])
            self.assertEqual(len(report["per_image"]), len(images))
            self.assertEqual(report["summary"]["BF"]["gt_count"], 33)
            self.assertEqual(report["summary"]["JT"]["pred_count"], 33)
            self.assertEqual(report["summary"]["BF"]["labeled_images"], 33)
            self.assertEqual(report["summary"]["JT"]["labeled_images"], 33)
            self.assertEqual(report["summary"]["all"]["labeled_images"], 66)
            scalar_names = {key for key, _, _ in writer.scalars}
            self.assertIn("BF/metrics/recall_pct", scalar_names)
            self.assertIn("BF/metrics/precision_pct", scalar_names)
            self.assertFalse(any(key.startswith(("overview/", "diagnostics/")) for key in scalar_names))
            if report["summary"]["BF"]["mean_matched_iou"] is None:
                self.assertNotIn("BF/metrics/iou_pct", scalar_names)
            for file in (run / "previews/epoch_005").glob("*.png"):
                with Image.open(file) as image:
                    self.assertEqual(image.size, (48 * 5, 40 + 88))

    def test_disabling_images_keeps_every_epoch_full_val_metrics_hook(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, images = self.make_data(root)
            callback = live.LiveVisualization(data, images, count=0, tensorboard=False)
            calls = []
            callback._preview = lambda trainer, step: calls.append(step)
            trainer = SimpleNamespace(epoch=0, epochs=100, start_epoch=0, stop=False,
                                      save_dir=root / "run",
                                      tloss=None, lr={}, metrics={},
                                      label_loss_items=lambda values, prefix: {})
            callback.on_epoch_start(trainer)
            callback.on_fit_end(trainer)
            trainer.epoch = 1
            callback.on_epoch_start(trainer)
            callback.on_fit_end(trainer)
            self.assertEqual(calls, [1, 2])

    def test_manual_stop_is_handled_after_current_epoch_preview(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"RANK": "-1"}):
            run = Path(temp)
            request = run / "STOP"
            request.write_text("finish this epoch")
            trainer = SimpleNamespace(epoch=2, save_dir=run, stop=False)
            callback = live.LiveVisualization({"path": str(run), "names": {}}, [], count=0, tensorboard=False)
            observations = []
            callback._preview = lambda trainer, step: observations.append((step, trainer.stop, request.exists()))
            callback.on_epoch_start(trainer)
            callback.on_fit_end(trainer)
            callback.on_fit_end(trainer)
            self.assertEqual(observations, [(3, False, True)])
            self.assertTrue(trainer.stop)
            self.assertFalse(request.exists())
            self.assertEqual((run / "STOP.handled_epoch_003").read_text(), "finish this epoch")

    def test_posthoc_rejects_old_calli_weights_before_inference_on_new_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, images = self.make_data(root)
            data.update(train="images/val", val="images/val", split_id="new-full-split",
                        annotation_availability="annotation_availability.json")
            (root / "data.yaml").write_text(json.dumps(data))
            availability = {str(image.relative_to(root)): {"has_bf": True, "has_jt": True}
                            for image in images}
            (root / "annotation_availability.json").write_text(json.dumps(availability))
            weights = root / "old_run/weights/best.pt"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"mock old Calli checkpoint, no split provenance")
            class OldYOLO:
                task = "segment"
                names = data["names"]
                def __init__(self, path):
                    pass
                def predict(self, **kwargs):
                    raise AssertionError("Split guard must reject before inference")
            spec = importlib.util.spec_from_file_location("posthoc_eval", MODULE.parent / "evaluate_regions.py")
            evaluator = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(evaluator)
            argv = ["evaluate_regions.py", "--weights", str(weights), "--data", str(root / "data.yaml"),
                    "--output", str(root / "new_eval"), "--no-tensorboard"]
            with patch.object(sys, "argv", argv), patch.dict("sys.modules", {"ultralytics": SimpleNamespace(YOLO=OldYOLO)}):
                with self.assertRaisesRegex(ValueError, "no new-split provenance"):
                    evaluator.main()
            self.assertFalse((root / "new_eval").exists())

    def test_two_epoch_cards_each_preserve_60_samples_and_one_step(self):
        import io
        from tensorboard.compat.proto import types_pb2, summary_pb2, event_pb2
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        with tempfile.TemporaryDirectory() as temp:
            paths = []
            for index in range(60):
                path = Path(temp) / f"{index:02d}.png"
                Image.new("RGB", (10, 8), (index, 0, 0)).save(path)
                paths.append(path)
            writer = FakeWriter()
            for epoch in (1, 2):
                live.write_image_card(writer, paths, epoch)
            self.assertEqual(len(writer.summaries), 2)
            events = EventFileWriter(str(Path(temp) / "events"))
            for epoch, (summary, step) in enumerate(writer.summaries, start=1):
                self.assertEqual(step, epoch)
                self.assertEqual(len(summary.value), 1)
                value = summary.value[0]
                self.assertEqual(value.tag, f"previews/epoch_{epoch:03d}")
                self.assertEqual(value.metadata.display_name, f"Epoch {epoch:03d} · fixed validation samples")
                self.assertIn(f"Epoch {epoch:03d}.", value.metadata.summary_description)
                self.assertEqual(value.tensor.dtype, types_pb2.DT_STRING)
                self.assertEqual(value.tensor.tensor_shape.dim[0].size, 62)
                self.assertEqual(value.metadata.plugin_data.plugin_name, "images")
                self.assertEqual(value.metadata.data_class, summary_pb2.DATA_CLASS_BLOB_SEQUENCE)
                self.assertEqual(list(value.tensor.string_val[:2]), [b"10", b"8"])
                events.add_event(event_pb2.Event(step=step, summary=summary))
            events.flush()
            events.close()
            reader = EventAccumulator(str(Path(temp) / "events"), size_guidance={"tensors": 0}).Reload()
            self.assertEqual(sorted(reader.Tags()["tensors"]), ["previews/epoch_001", "previews/epoch_002"])
            for epoch in (1, 2):
                entries = reader.Tensors(f"previews/epoch_{epoch:03d}")
                self.assertEqual([entry.step for entry in entries], [epoch])
                self.assertEqual(len(entries[0].tensor_proto.string_val) - 2, 60)
                with Image.open(io.BytesIO(entries[0].tensor_proto.string_val[61])) as sample59:
                    self.assertEqual(sample59.getpixel((0, 0)), (59, 0, 0))

    def test_annotation_tasks_is_explicit_and_missing_rows_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, images = self.make_data(root)
            self.assertTrue(all(v == {"BF": True, "JT": True}
                                for v in live.annotation_tasks(data, images).values()))
            availability = {str(path.relative_to(root)): {"has_bf": i % 2 == 0, "has_jt": i % 2 == 1}
                            for i, path in enumerate(images)}
            (root / "annotation_availability.json").write_text(json.dumps(availability))
            result = live.annotation_tasks(data, images)
            self.assertEqual(result[images[0].resolve()], {"BF": True, "JT": False})
            self.assertEqual(result[images[1].resolve()], {"BF": False, "JT": True})
            del availability[str(images[0].relative_to(root))]
            (root / "annotation_availability.json").write_text(json.dumps(availability))
            with self.assertRaisesRegex(ValueError, "Missing or invalid"):
                live.annotation_tasks(data, images)

    def test_unannotated_truth_column_keeps_both_prediction_columns(self):
        image = Image.new("RGB", (80, 80), "white")
        mask = np.zeros((80, 80), bool)
        mask[40:70, 40:70] = True
        predicted = [{"class_id": 0, "mask": mask}, {"class_id": 50, "mask": mask}]
        panel = live.panel_image(image, [], predicted, {}, 1, "sample", known_tasks={"BF"})
        pixels = np.asarray(panel)
        self.assertTrue(np.all(pixels[120, 3 * 80 + 60] == 255))  # JT truth has no overlay
        self.assertFalse(np.all(pixels[120, 2 * 80 + 60] == 255))
        self.assertFalse(np.all(pixels[120, 4 * 80 + 60] == 255))

    def test_stale_or_stripped_checkpoint_rejected(self):
        for checkpoint in (None, {"epoch": -1}, {"epoch": 3}):
            with self.assertRaisesRegex(RuntimeError, "Preview requires the current"):
                live.checkpoint_matches_epoch(checkpoint, 4)
        live.checkpoint_matches_epoch({"epoch": 4}, 4)


if __name__ == "__main__":
    unittest.main()
