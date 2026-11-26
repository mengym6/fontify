"""Partial-label contracts. Torch tests use forward/backward only, no optimizer."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import partial_supervision as partial

try:
    import torch
    from ultralytics.utils.loss import SegmentLoss26
except ImportError:
    torch = None
    SegmentLoss26 = None


class MetadataTests(unittest.TestCase):
    def test_partial_flags_unknown_are_not_negative_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = {"images/a.png": {"has_bf": True, "has_jt": False},
                    "images/b.png": {"has_bf": False, "has_jt": True}}
            path = root / "annotation_availability.json"
            path.write_text(json.dumps(rows))
            data = {"path": str(root), "annotation_availability": path.name}
            loaded = partial.load_annotation_availability(data)
            self.assertEqual(partial.image_task_valid(root / "images/a.png", root, loaded), (True, False))
            self.assertEqual(partial.image_task_valid(root / "images/b.png", root, loaded), (False, True))
            self.assertEqual(partial.image_task_valid(root / "missing.png", root, None), (True, True))
            with self.assertRaises(KeyError):
                partial.image_task_valid(root / "images/missing.png", root, loaded)

    def test_mixed_image_augmentation_is_rejected(self):
        for name in ("mixup", "copy_paste", "cutmix"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                partial.reject_mixed_image_augmentation(SimpleNamespace(**{name: .1}))
        partial.reject_mixed_image_augmentation(SimpleNamespace(mosaic=1))
        partial.reject_mixed_image_augmentation(SimpleNamespace(mosaic=1), augment=False)
        with self.assertRaisesRegex(ValueError, "overlap_mask"):
            partial.reject_mixed_image_augmentation(SimpleNamespace(overlap_mask=True))

    def test_mosaic_donors_use_the_primary_task_pool(self):
        class NativeMosaicInterface:
            def get_params(self, labels):
                return {"indexes": self.get_indexes()}
        mosaic_class = partial._same_task_mosaic_class(NativeMosaicInterface)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            flags = [(True, False), (True, False), (False, True),
                     (False, True), (True, True), (True, True)]
            availability = {f"images/{i}.png": {"has_bf": valid[0], "has_jt": valid[1]}
                            for i, valid in enumerate(flags)}
            pools = {valid: [i for i, item in enumerate(flags) if item == valid] for valid in flags}
            dataset = SimpleNamespace(availability_root=root, annotation_availability=availability,
                                      mosaic_task_flags=flags, mosaic_task_pools=pools, buffer=[1, 3, 5])
            mosaic = mosaic_class()
            mosaic.dataset, mosaic.n, mosaic.buffer_enabled = dataset, 4, True
            for primary in (0, 2, 4):
                # Empty surviving labels must not erase known task flags.
                labels = {"im_file": str(root / f"images/{primary}.png"), "cls": []}
                selected = mosaic.get_params(labels)["indexes"]
                self.assertEqual(selected, [primary + 1] * 3)
                self.assertTrue(all(flags[index] == flags[primary] for index in selected))
                self.assertIsNone(mosaic._active_task_valid)
            dataset.buffer = [2, 4]  # no BF donor in native buffer: fall back to the BF pool
            selected = mosaic.get_params({"im_file": str(root / "images/0.png")})["indexes"]
            self.assertTrue(set(selected).issubset({0, 1}))
            mosaic.buffer_enabled = False
            selected = mosaic.get_params({"im_file": str(root / "images/2.png")})["indexes"]
            self.assertTrue(set(selected).issubset({2, 3}))

    def test_group_state_selects_one2one_and_preserves_unknown_none(self):
        empty = SimpleNamespace()
        self.assertEqual(partial.read_group_loss_accumulators(empty)["batches"], 0)
        partial.reset_group_loss_accumulators(empty)
        self.assertFalse(hasattr(empty, "criterion"))
        state = {"sums": [[2, 4, 6, 8], [0, 0, 0, 0]], "batches": 2, "known_samples": [4, 0]}
        one = SimpleNamespace(_calli_group_accumulator=state)
        model = SimpleNamespace(criterion=SimpleNamespace(one2many=SimpleNamespace(), one2one=one))
        actual = partial.read_group_loss_accumulators(model)
        self.assertEqual(actual["BF"], 10)
        self.assertIsNone(actual["JT"])
        self.assertEqual(actual["common_total"], 10)
        partial.reset_group_loss_accumulators(model)
        self.assertEqual(partial.read_group_loss_accumulators(model)["batches"], 0)

    def test_protocol_records_and_guards_split_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "availability.json").write_text('{}')
            trainer = SimpleNamespace(save_dir=root / "run", data={
                "path": str(root), "annotation_availability": "availability.json", "split_id": "split-a"})
            result = partial.save_partial_protocol(trainer)
            self.assertEqual(result["mode"], "availability-masked")
            self.assertTrue(result["availability_sha256"])
            trainer.data["split_id"] = "split-b"
            with self.assertRaisesRegex(ValueError, "split_id"):
                partial.save_partial_protocol(trainer)


@unittest.skipIf(torch is None, "Requires torch + pinned Ultralytics; no optimizer updates are performed")
class TensorContracts(unittest.TestCase):
    def test_native_mosaic_and_close_mosaic_keep_task_flags(self):
        from PIL import Image
        from ultralytics.cfg import get_cfg
        from ultralytics.data.augment import Mosaic
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "images/train").mkdir(parents=True)
            (root / "labels/train").mkdir(parents=True)
            flags = [(True, False), (True, False), (False, True),
                     (False, True), (True, True), (True, True)]
            availability = {}
            for index, valid in enumerate(flags):
                image = root / f"images/train/image_{index}.png"
                Image.new("RGB", (64, 64), "white").save(image)
                classes = ([0] if valid[0] else []) + ([50] if valid[1] else [])
                (root / f"labels/train/image_{index}.txt").write_text(
                    "\n".join(f"{class_id} .2 .2 .8 .2 .8 .8 .2 .8" for class_id in classes))
                availability[image.relative_to(root).as_posix()] = {"has_bf": valid[0], "has_jt": valid[1]}
            (root / "availability.json").write_text(json.dumps(availability))
            data = {"path": str(root), "nc": 70, "names": {i: str(i) for i in range(70)},
                    "annotation_availability": "availability.json"}
            hyp = get_cfg(overrides={"mosaic": 1., "mixup": 0., "copy_paste": 0., "cutmix": 0.,
                                     "overlap_mask": False, "fliplr": 0., "flipud": 0., "task": "segment"})
            dataset = partial.PartialYOLODataset(
                img_path=str(root / "images/train"), data=data, imgsz=64, batch_size=3,
                augment=True, hyp=hyp, cache=False, rect=False, pad=0., task="segment",
            )
            native_mosaic = dataset.transforms.transforms[0].transforms[0]
            self.assertIsInstance(native_mosaic, Mosaic)
            self.assertEqual(type(native_mosaic).__name__, "SameTaskMosaic")
            original = dataset.get_image_and_label
            for primary in (0, 2, 4):
                seen = []
                def record(index, *args, **kwargs):
                    seen.append(index)
                    return original(index, *args, **kwargs)
                with patch.object(dataset, "get_image_and_label", side_effect=record):
                    item = dataset[primary]
                self.assertEqual(len(seen), 4)
                self.assertTrue(all(dataset.mosaic_task_flags[index] == flags[primary] for index in seen))
                self.assertEqual(item["task_valid"].tolist(), list(flags[primary]))
            dataset.close_mosaic(hyp)
            self.assertEqual(dataset.transforms.transforms[0].transforms[0].p, 0.)
            seen = []
            with patch.object(dataset, "get_image_and_label", side_effect=record):
                item = dataset[0]
            self.assertEqual(seen, [0])
            self.assertEqual(item["task_valid"].tolist(), [True, False])

    def test_fp16_zero_graph_connection_does_not_overflow(self):
        values = torch.full((256, 256), 16., dtype=torch.float16, requires_grad=True)
        self.assertTrue(torch.isinf(values.sum()))
        zero = partial.connected_zero(values)
        self.assertTrue(torch.isfinite(zero))
        self.assertEqual(zero.item(), 0)
        zero.backward()
        self.assertIsNotNone(values.grad)
        self.assertEqual(torch.count_nonzero(values.grad).item(), 0)

    def fixture(self):
        class MinimalModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))
                self.args = SimpleNamespace(overlap_mask=False, box=7.5, cls=.5, dfl=1.5, epochs=10)
                self.model = [SimpleNamespace(nc=70, reg_max=1, stride=torch.tensor([8.]))]
        model = MinimalModel()
        generator = torch.Generator().manual_seed(42)
        predictions = {
            "feats": [torch.zeros(2, 8, 4, 4)],
            "boxes": (torch.rand(2, 4, 16, generator=generator) * 2 + .5).requires_grad_(),
            "scores": torch.randn(2, 70, 16, generator=generator, requires_grad=True),
            "mask_coefficient": torch.randn(2, 4, 16, generator=generator, requires_grad=True),
            "proto": (torch.randn(2, 4, 8, 8, generator=generator, requires_grad=True),
                      torch.randn(2, 70, 8, 8, generator=generator, requires_grad=True)),
        }
        masks = torch.zeros(2, 8, 8)
        masks[:, 2:6, 2:6] = 1
        semantic = torch.zeros(2, 8, 8)
        semantic[1, 2:6, 2:6] = 50
        batch = {"batch_idx": torch.tensor([0., 1.]), "cls": torch.tensor([[0.], [50.]]),
                 "bboxes": torch.tensor([[.5, .5, .5, .5], [.5, .5, .5, .5]]),
                 "masks": masks, "sem_masks": semantic,
                 "task_valid": torch.tensor([[True, True], [True, True]])}
        native = SegmentLoss26(model)
        wrapped = partial._partial_segment_class(SegmentLoss26)(model)
        return model, predictions, batch, native, wrapped

    def test_full_known_nonoverlap_matches_native_and_group_sum(self):
        _, prediction, batch, native, wrapped = self.fixture()
        expected, _ = native.loss(prediction, batch)
        actual, items = wrapped.loss(prediction, batch)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(wrapped._last_group_components.sum(0), actual[:4] / 2, rtol=1e-6, atol=1e-6)
        read = partial.read_group_loss_accumulators(SimpleNamespace(criterion=wrapped))
        self.assertAlmostEqual(read["BF"] + read["JT"], sum(float(items[key]) for key in partial.PRIMARY_NAMES), places=4)

    def test_unknown_class_and_semantic_channels_have_zero_gradient(self):
        _, prediction, batch, _, wrapped = self.fixture()
        batch["task_valid"] = torch.tensor([[True, False], [False, True]])
        loss, _ = wrapped.loss(prediction, batch)
        loss.sum().backward()
        self.assertEqual(torch.count_nonzero(prediction["scores"].grad[0, 50:]).item(), 0)
        self.assertEqual(torch.count_nonzero(prediction["scores"].grad[1, :50]).item(), 0)
        self.assertGreater(prediction["scores"].grad[0, :50].abs().sum().item(), 0)
        self.assertGreater(prediction["scores"].grad[1, 50:].abs().sum().item(), 0)
        semantic = prediction["proto"][1]
        self.assertEqual(torch.count_nonzero(semantic.grad[0, 50:]).item(), 0)
        self.assertEqual(torch.count_nonzero(semantic.grad[1, :50]).item(), 0)
        torch.testing.assert_close(wrapped._last_group_components.sum(0), loss[:4].detach() / 2)

    def test_bf_only_loss_filters_jt_gt_and_has_zero_jt_gradients(self):
        from staged_training import bf_only_targets
        _, predictions, batch, _, criterion = self.fixture()
        original_classes = batch["cls"].clone()
        criterion._calli_bf_only = True
        losses, _ = criterion.loss(predictions, batch)
        losses.sum().backward()
        self.assertGreater(predictions["scores"].grad[:, :50].abs().sum().item(), 0)
        self.assertEqual(torch.count_nonzero(predictions["scores"].grad[:, 50:]).item(), 0)
        self.assertEqual(torch.count_nonzero(predictions["proto"][1].grad[:, 50:]).item(), 0)
        self.assertIsNone(partial.read_group_loss_accumulators(SimpleNamespace(criterion=criterion))["JT"])
        filtered = bf_only_targets(batch, 2)
        self.assertEqual(filtered["cls"].reshape(-1).tolist(), [0.])
        self.assertEqual(filtered["task_valid"].tolist(), [[True, False], [True, False]])
        torch.testing.assert_close(batch["cls"], original_classes)
        self.assertTrue(batch["task_valid"].all())
        # The untouched validation criterion still evaluates the JT annotations.
        _, predictions, batch, _, validation = self.fixture()
        validation.loss(predictions, batch)
        self.assertIsNotNone(partial.read_group_loss_accumulators(SimpleNamespace(criterion=validation))["JT"])
        # A JT-only image must not become a BF-negative training example.
        batch['task_valid'][:, 0] = False
        batch['cls'][:] = 50
        criterion._calli_group_accumulator = None
        loss, _ = criterion.loss(predictions, batch)
        self.assertEqual(loss.detach().sum().item(), 0.)

    def test_amp_outputs_keep_native_float32_loss_accumulation(self):
        _, prediction, batch, native, wrapped = self.fixture()
        for key in ("scores", "boxes", "mask_coefficient"):
            prediction[key] = prediction[key].detach().to(torch.bfloat16).requires_grad_()
        prediction["proto"] = tuple(value.detach().to(torch.bfloat16).requires_grad_()
                                    for value in prediction["proto"])
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            expected, _ = native.loss(prediction, batch)
            actual, _ = wrapped.loss(prediction, batch)
        self.assertEqual(wrapped._last_group_components.dtype, torch.float32)
        torch.testing.assert_close(actual[:4], expected[:4], rtol=3e-5, atol=3e-5)

    def test_semantic_targets_preserve_cross_group_overlap(self):
        masks = torch.ones(2, 4, 4)
        target = partial.multihot_semantic_targets(masks, torch.tensor([0, 0]), torch.tensor([0, 50]), 1, (4, 4))
        self.assertEqual(target[0, 0].sum().item(), 16)
        self.assertEqual(target[0, 50].sum().item(), 16)
        logits = torch.randn_like(target, requires_grad=True)
        loss = partial.masked_semantic_loss(logits, target, torch.tensor([[True, False]]))
        loss.backward()
        self.assertEqual(torch.count_nonzero(logits.grad[:, 50:]).item(), 0)
        self.assertGreater(logits.grad[:, :50].abs().sum().item(), 0)

    def test_known_task_with_zero_remaining_gt_still_has_class_negatives(self):
        _, prediction, batch, _, wrapped = self.fixture()
        batch.update(batch_idx=torch.empty(0), cls=torch.empty(0, 1), bboxes=torch.empty(0, 4),
                     masks=torch.empty(0, 8, 8), task_valid=torch.tensor([[True, False], [True, False]]))
        loss, _ = wrapped.loss(prediction, batch)
        loss.sum().backward()
        self.assertGreater(prediction["scores"].grad[:, :50].abs().sum().item(), 0)
        self.assertEqual(torch.count_nonzero(prediction["scores"].grad[:, 50:]).item(), 0)
        state = partial.read_group_loss_accumulators(SimpleNamespace(criterion=wrapped))
        self.assertIsNone(state["JT"])

    def test_metric_filter_does_not_modify_predictions(self):
        source = [{"cls": torch.tensor([0., 50.]), "conf": torch.tensor([.8, .9]),
                   "bboxes": torch.zeros(2, 4), "masks": torch.zeros(2, 8, 8)}]
        filtered = partial.filter_unknown_task_predictions(source, torch.tensor([[True, False]]))
        self.assertEqual(filtered[0]["cls"].tolist(), [0.])
        self.assertEqual(source[0]["cls"].tolist(), [0., 50.])

    def test_collate_stacks_boolean_task_valid(self):
        dataset_class = partial.PartialYOLODataset
        batch = [{"img": torch.zeros(3, 8, 8), "im_file": "a", "task_valid": torch.tensor([True, False])},
                 {"img": torch.zeros(3, 8, 8), "im_file": "b", "task_valid": torch.tensor([False, True])}]
        output = dataset_class.collate_fn(batch)
        self.assertEqual(tuple(output["task_valid"].shape), (2, 2))
        self.assertEqual(output["task_valid"].dtype, torch.bool)
        self.assertEqual(output["task_valid"].tolist(), [[True, False], [False, True]])


if __name__ == "__main__":
    unittest.main()
