"""Numerical checks for the minimal image perception experiment."""

import unittest

import torch

from tools.calli_perception_minimal.probe import (
    PerceptionHead,
    scores,
    shift,
)
from util.calli_labels import label_loss


class PerceptionContractTest(unittest.TestCase):
    """Check missing-task supervision, metric identity, and translation."""

    def test_missing_task_has_zero_output_gradient(self):
        torch.manual_seed(7)
        model = PerceptionHead()
        logits = model(torch.randn(2, 768, 28, 28))
        logits.retain_grad()
        target = torch.zeros_like(logits)
        target[0, 0, 5:12, 5:12] = 1
        target[1, 50, 4:16, 2:14] = 1
        loss = label_loss(logits, target, torch.tensor([0, 1]))
        loss.backward()
        self.assertEqual(logits.shape, (2, 70, 28, 28))
        self.assertEqual(logits.grad[0, 50:].abs().sum().item(), 0)
        self.assertEqual(logits.grad[1, :50].abs().sum().item(), 0)
        self.assertGreater(logits.grad[0, :50].abs().sum().item(), 0)
        self.assertGreater(logits.grad[1, 50:].abs().sum().item(), 0)

    def test_metric_identity_and_false_positive_penalty(self):
        target = torch.zeros(2, 70, 28, 28)
        target[0, 0, 5:12, 5:12] = 1
        target[1, 50, 4:16, 2:14] = 1
        kinds = torch.tensor([0, 1])
        perfect = scores(target, target, kinds)
        wrong = target.clone()
        wrong[0, 1] = 1
        wrong[1, 51] = 1
        bad = scores(wrong, target, kinds)
        for kind in ("BF", "JT"):
            self.assertEqual(perfect[kind]["soft_dice"], 1)
            self.assertLess(bad[kind]["all_channel_soft_iou"], 1)

    def test_translation_does_not_wrap(self):
        x = torch.zeros(1, 1, 8, 8)
        x[..., 0, 0] = 1
        x[..., 7, 7] = 1
        moved = shift(x, 2, 1)
        self.assertEqual(moved[..., 1, 2].item(), 1)
        self.assertEqual(moved.sum().item(), 1)


if __name__ == "__main__":
    unittest.main()
