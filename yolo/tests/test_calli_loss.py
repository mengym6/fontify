"""Semantic-only weighting, resume configuration, and criterion installation."""

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / 'calli_loss.py'
SPEC = importlib.util.spec_from_file_location('calli_loss', MODULE)
loss_options = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loss_options)

try:
    import torch
except ImportError:
    torch = None


class ArrayTensor:
    """Only the tensor operations used by the adapter, without fake autograd."""
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.shape = self.values.shape

    def new_tensor(self, values):
        return ArrayTensor(values)

    def __mul__(self, other):
        return ArrayTensor(self.values * other.values)


class NativeSegment:
    def __init__(self):
        self.assigner = object()
        self.hyp = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        self.output = ArrayTensor([2, 4, 6, 8, 10])
        self.items = dict(zip(('box_loss', 'seg_loss', 'cls_loss', 'dfl_loss', 'sem_loss'),
                              [1.0, 2.0, 3.0, 4.0, 5.0]))

    def loss(self, preds, batch):
        return self.output, self.items


class NativeE2E:
    def __init__(self):
        self.one2many, self.one2one = NativeSegment(), NativeSegment()
        self.updates, self.total, self.o2m, self.o2o = 42, 1.0, 0.4, 0.6
        self.o2m_copy, self.final_o2m = 0.8, 0.1


@contextmanager
def native_modules():
    package = ModuleType('ultralytics')
    package.__version__ = '8.4.218'
    loss_module = ModuleType('ultralytics.utils.loss')
    loss_module.SegmentLoss26, loss_module.E2ELoss = NativeSegment, NativeE2E
    torch_utils = ModuleType('ultralytics.utils.torch_utils')
    torch_utils.unwrap_model = lambda model: model
    with patch.dict(sys.modules, {'ultralytics': package,
                                  'ultralytics.utils.loss': loss_module,
                                  'ultralytics.utils.torch_utils': torch_utils}):
        yield


class ConfigurationTests(unittest.TestCase):
    def test_default_and_invalid_weights(self):
        self.assertEqual(loss_options.resolve_semantic_weight(None), 1.0)
        self.assertEqual(loss_options.resolve_semantic_weight(.25), .25)
        for value in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                loss_options.resolve_semantic_weight(value)

    def test_resume_inherits_sidecar_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            checkpoint = run / 'weights/last.pt'
            self.assertEqual(loss_options.resolve_semantic_weight(None, checkpoint), 1)
            with self.assertRaises(ValueError):
                loss_options.resolve_semantic_weight(.25, checkpoint)
            (run / loss_options.CONFIG_NAME).write_text(json.dumps({
                'schema': loss_options.SCHEMA, 'semantic_weight': .25,
            }))
            self.assertEqual(loss_options.resolve_semantic_weight(None, checkpoint), .25)
            self.assertEqual(loss_options.resolve_semantic_weight(.25, checkpoint), .25)
            with self.assertRaisesRegex(ValueError, 'requires a new run'):
                loss_options.resolve_semantic_weight(1, checkpoint)


class SemanticWeightTests(unittest.TestCase):
    def test_one_is_identical_and_quarter_only_changes_semantic(self):
        native = NativeSegment()
        original = native.output.values.copy()
        losses, items = loss_options.scale_semantic_outputs(native.output, native.items, 1)
        self.assertIs(losses, native.output)
        self.assertIs(items, native.items)
        losses, items = loss_options.scale_semantic_outputs(native.output, native.items, .25)
        np.testing.assert_array_equal(losses.values, [2, 4, 6, 8, 2.5])
        np.testing.assert_array_equal(native.output.values, original)
        self.assertEqual(items['sem_loss'], 1.25)
        self.assertEqual(native.items['sem_loss'], 5)
        for key in ('box_loss', 'seg_loss', 'cls_loss', 'dfl_loss'):
            self.assertEqual(items[key], native.items[key])

    def test_both_e2e_branches_keep_assigners_and_schedule(self):
        criterion = NativeE2E()
        state = {key: value for key, value in vars(criterion).items()
                 if key not in ('one2many', 'one2one')}
        assigners = (criterion.one2many.assigner, criterion.one2one.assigner)
        model = SimpleNamespace(criterion=criterion)
        with native_modules():
            result = loss_options.install_semantic_weight(model, .25)
            self.assertIs(model.criterion, criterion)
            self.assertEqual(result['branches'], 2)
            self.assertEqual(state, {key: value for key, value in vars(criterion).items()
                                     if key not in ('one2many', 'one2one')})
            for branch, assigner in zip((criterion.one2many, criterion.one2one), assigners):
                self.assertIs(branch.assigner, assigner)
                self.assertEqual(branch.loss(None, None)[0].values[4], 2.5)
            loss_options.install_semantic_weight(model, .25)
            self.assertEqual(criterion.one2one.loss(None, None)[0].values[4], 2.5)

    def test_callback_installs_model_and_ema_and_saves_config(self):
        with tempfile.TemporaryDirectory() as directory, native_modules():
            model = SimpleNamespace(criterion=NativeE2E())
            ema = SimpleNamespace(criterion=None, init_criterion=NativeE2E)
            trainer = SimpleNamespace(model=model, ema=SimpleNamespace(ema=ema), save_dir=directory)
            with patch.dict('os.environ', {'RANK': '0'}):
                loss_options.make_semantic_loss_callback(.25)(trainer)
            config = json.loads((Path(directory) / loss_options.CONFIG_NAME).read_text())
            self.assertEqual(config['semantic_weight'], .25)
            self.assertEqual(config['installed']['model']['branches'], 2)
            self.assertEqual(config['installed']['ema']['branches'], 2)
            self.assertEqual(ema.criterion.one2many.loss(None, None)[1]['sem_loss'], 1.25)
            with self.assertRaises(ValueError):
                loss_options.make_semantic_loss_callback(1)(trainer)

    @unittest.skipIf(torch is None, 'Real autograd requires torch; run on training host')
    def test_real_autograd_scales_only_semantic_and_keeps_zero_connected(self):
        names = ('box_loss', 'seg_loss', 'cls_loss', 'dfl_loss', 'sem_loss')
        for weight in (1.0, .25, 0.0):
            leaf = torch.arange(1., 6., requires_grad=True)
            native = leaf.square()
            items = dict(zip(names, native.detach()))
            scaled, logged = loss_options.scale_semantic_outputs(native, items, weight)
            gradient, = torch.autograd.grad(scaled.sum(), leaf)
            expected = 2 * leaf.detach() * torch.tensor([1, 1, 1, 1, weight])
            torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
            torch.testing.assert_close(scaled[:4], native[:4], rtol=0, atol=0)
            self.assertEqual(logged['sem_loss'].item(), 25 * weight)
            self.assertTrue(scaled.requires_grad)
            if weight == 1:
                self.assertIs(scaled, native)


if __name__ == '__main__':
    unittest.main()
