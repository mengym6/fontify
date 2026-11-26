"""Stage boundary, resume settings, and true frozen-weight behavior."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staged_training import resolve_stages, apply_training_stage, save_joint_checkpoint


class StageConfigTests(unittest.TestCase):
    def test_new_defaults_and_legacy_resume(self):
        self.assertEqual(resolve_stages(), {'bf_only_after': 40, 'freeze_layers': 4})
        self.assertIsNone(resolve_stages(0))
        with tempfile.TemporaryDirectory() as temp:
            ckpt = Path(temp) / 'weights/last.pt'
            self.assertIsNone(resolve_stages(resume_checkpoint=ckpt))
            with self.assertRaises(ValueError):
                resolve_stages(30, 4, ckpt)
            config = {'bf_only_after': 30, 'freeze_layers': 4}
            (Path(temp) / 'partial_supervision.json').write_text(json.dumps({'staged_training': config}))
            self.assertEqual(resolve_stages(resume_checkpoint=ckpt), config)
            with self.assertRaises(ValueError):
                resolve_stages(20, 4, ckpt)
        for args in ((-1, 4), (30, -1)):
            with self.assertRaises(ValueError):
                resolve_stages(*args)

    def test_joint_boundary_checkpoint_is_saved_once_per_boundary_epoch(self):
        with tempfile.TemporaryDirectory() as temp:
            last = Path(temp) / 'last.pt'
            last.write_bytes(b'joint checkpoint')
            trainer = SimpleNamespace(staged_training=resolve_stages(30, 4), epoch=28, last=last)
            save_joint_checkpoint(trainer)
            self.assertFalse(last.with_name('joint_end.pt').exists())
            trainer.epoch = 29
            save_joint_checkpoint(trainer)
            trainer.epoch = 30
            last.write_bytes(b'bf checkpoint')
            save_joint_checkpoint(trainer)
            self.assertEqual(last.with_name('joint_end.pt').read_bytes(), b'joint checkpoint')


try:
    import torch
    from ultralytics.engine.trainer import BaseTrainer
except ImportError:
    torch = BaseTrainer = None


@unittest.skipIf(torch is None or BaseTrainer is None, 'Requires pinned Torch/Ultralytics')
class FrozenStageTests(unittest.TestCase):
    def test_switch_freezes_parameters_bn_and_clears_pending_joint_gradients(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.ModuleList([
                    torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.BatchNorm2d(3))
                    for _ in range(5)])
                self.yaml = {'backbone': [0] * 5}
                self.criterion = SimpleNamespace(one2many=SimpleNamespace(), one2one=SimpleNamespace())
            def forward(self, x):
                for layer in self.model:
                    x = layer(x)
                return x
        model, ema_model = Model(), Model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
        trainer = SimpleNamespace(model=model, ema=SimpleNamespace(ema=ema_model), optimizer=optimizer,
                                  freeze_layer_names=[], staged_training=resolve_stages(30, 4), epoch=29)
        apply_training_stage(trainer)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertFalse(model.criterion.one2one._calli_bf_only)
        model(torch.randn(2, 3, 8, 8)).square().mean().backward()
        optimizer.step()  # Populate Adam state, leave stale joint gradients in place.
        frozen = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if any(k.startswith(f'model.{i}.') for i in range(4))}
        unfrozen = model.model[4][0].weight.detach().clone()
        trainer.epoch = 30
        apply_training_stage(trainer)
        BaseTrainer._model_train(trainer)
        self.assertTrue(model.criterion.one2one._calli_bf_only)
        self.assertTrue(model.criterion.one2many._calli_bf_only)
        self.assertFalse(ema_model.criterion.one2one._calli_bf_only)
        self.assertTrue(all(p.grad is None and not p.requires_grad for layer in model.model[:4]
                            for p in layer.parameters()))
        model(torch.randn(2, 3, 8, 8)).square().mean().backward()
        optimizer.step()
        for name, before in frozen.items():
            torch.testing.assert_close(model.state_dict()[name], before, rtol=0, atol=0)
        self.assertFalse(torch.equal(model.model[4][0].weight, unfrozen))
        # Reapply the stage when resuming in the BF-only half.
        del trainer._bf_stage_applied
        trainer.epoch = 45
        apply_training_stage(trainer)
        BaseTrainer._model_train(trainer)
        self.assertFalse(model.model[0][1].training)


if __name__ == '__main__':
    unittest.main()
