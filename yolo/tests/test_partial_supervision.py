"""Partial-label contracts. Torch tests use forward/backward only, no optimizer."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


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
        for name in ("mosaic", "mixup", "copy_paste", "cutmix"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                partial.reject_mixed_image_augmentation(SimpleNamespace(**{name: .1}))
        partial.reject_mixed_image_augmentation(SimpleNamespace(mosaic=1), augment=False)
        with self.assertRaisesRegex(ValueError, "overlap_mask"):
            partial.reject_mixed_image_augmentation(SimpleNamespace(overlap_mask=True))

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
