import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from PIL import Image

from yolo.pairing.data import PairDataset, validate_pairs
from yolo.pairing.loss import PairLoss
from yolo.pairing.model import PairRegions


class TinyEncoder(nn.Module):
    channels = 8
    mask_channels = 4

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, stride=4), nn.BatchNorm2d(8), nn.SiLU()
        )
        self.proto = nn.Conv2d(8, 4, 1)

    def forward(self, x):
        y = self.feature(x)
        return y, self.proto(y)


def target(unknown=False, absent=False):
    masks = torch.zeros(2, 2, 8, 8)
    masks[0, :, 1:4, 1:4] = 1
    masks[1, :, 4:7, 4:7] = 1
    present = torch.ones(2, 2)
    if unknown:
        present[:, 1] = -1
    if absent:
        present[:, 1] = 0
        masks[:, 1] = 0
    return {
        "masks": masks,
        "present": present,
        "groups": torch.tensor([0, 1]),
        "complete": torch.tensor([[True, True], [not unknown, not unknown]]),
    }


class PairTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.output = {
            "masks": torch.randn(1, 4, 2, 8, 8, requires_grad=True),
            "presence": torch.randn(1, 4, 2, requires_grad=True),
            "groups": torch.tensor([0, 0, 1, 1]),
        }

    def test_unknown_side_has_no_direct_loss_gradient(self):
        loss, _ = PairLoss()(self.output, [target(unknown=True)])
        loss.backward()
        self.assertEqual(float(self.output["masks"].grad[:, :, 1].abs().sum()), 0)
        self.assertEqual(float(self.output["presence"].grad[:, :, 1].abs().sum()), 0)
        self.assertGreater(float(self.output["masks"].grad[:, :, 0].abs().sum()), 0)

    def test_absent_side_supervises_presence_not_mask(self):
        loss, _ = PairLoss()(self.output, [target(absent=True)])
        loss.backward()
        self.assertEqual(float(self.output["masks"].grad[:, :, 1].abs().sum()), 0)
        self.assertGreater(float(self.output["presence"].grad[:, :, 1].abs().sum()), 0)

    def test_missing_region_keeps_location_supervision(self):
        t = target()
        t["present"][:, 1] = 0
        t["mask_supervised"] = torch.ones(2, 2, dtype=torch.bool)
        loss, _ = PairLoss()(self.output, [t])
        loss.backward()
        self.assertGreater(float(self.output["masks"].grad[:, :, 1].abs().sum()), 0)
        self.assertTrue((self.output["presence"].grad[:, :, 1] > 0).all())

    def test_target_order_does_not_change_pair_loss(self):
        a = target()
        b = {
            k: (v.flip(0) if k in ["masks", "present", "groups"] else v)
            for k, v in a.items()
        }
        x, _ = PairLoss()(self.output, [a])
        y, _ = PairLoss()(self.output, [b])
        torch.testing.assert_close(x, y)

    def test_joint_query_shapes_and_two_image_gradients(self):
        model = PairRegions(
            TinyEncoder(),
            width=16,
            bf_queries=2,
            jt_queries=2,
            layers=1,
            heads=4,
            token_grid=2,
        )
        images = torch.randn(1, 2, 3, 32, 32, requires_grad=True)
        result = model(images)
        self.assertEqual(result["masks"].shape, (1, 4, 2, 8, 8))
        loss, _ = PairLoss()(result, [target()])
        loss.backward()
        for side in [0, 1]:
            self.assertGreater(float(images.grad[:, side].abs().sum()), 0)

    def test_other_image_changes_joint_output(self):
        model = PairRegions(
            TinyEncoder(),
            width=16,
            bf_queries=2,
            jt_queries=2,
            layers=1,
            heads=4,
            token_grid=2,
        ).eval()
        x = torch.randn(1, 2, 3, 32, 32)
        y = x.clone()
        y[:, 1] = 0
        with torch.no_grad():
            a = model(x)["masks"][:, :, 0]
            b = model(y)["masks"][:, :, 0]
        self.assertFalse(torch.allclose(a, b))

    def test_encoder_freeze_preserves_batchnorm(self):
        model = PairRegions(
            TinyEncoder(),
            width=16,
            bf_queries=2,
            jt_queries=2,
            layers=1,
            heads=4,
            token_grid=2,
        )
        model.freeze_encoder()
        model.train()
        self.assertFalse(model.encoder.training)
        self.assertTrue(all(not p.requires_grad for p in model.encoder.parameters()))

    def test_image_pairs_are_not_formal_training_labels(self):
        d = {
            "pairs": [
                {
                    "id": "x",
                    "split": "train",
                    "complete": [[True, False], [False, False]],
                    "regions": [
                        {
                            "pair_id": "one",
                            "group": "BF",
                            "a": {
                                "present": True,
                                "source": "human",
                                "polygons": [[[0, 0], [1, 0], [1, 1]]],
                            },
                            "b": {
                                "present": None,
                                "source": "unannotated",
                                "polygons": [],
                            },
                        }
                    ],
                }
            ]
        }
        self.assertEqual(validate_pairs(d), 0)
        with self.assertRaisesRegex(ValueError, "lack generated-side"):
            validate_pairs(d, require_paired=True)
        d["pairs"][0]["regions"][0]["b"] = {
            "present": True,
            "source": "pseudo",
            "polygons": [[[0, 0], [1, 0], [1, 1]]],
        }
        with self.assertRaisesRegex(ValueError, "Pseudo"):
            validate_pairs(d)
        self.assertEqual(validate_pairs(d, require_paired=True, allow_pseudo=True), 1)

    def test_query_capacity_error(self):
        t = target()
        t["groups"] = torch.zeros(2, dtype=torch.long)
        out = {
            k: (v[:, :1] if k != "groups" else v[:1]) for k, v in self.output.items()
        }
        t = {
            k: (torch.cat([v, v[:1]]) if k in ["masks", "present", "groups"] else v)
            for k, v in t.items()
        }
        with self.assertRaisesRegex(ValueError, "More GT"):
            PairLoss()(out, [t])

    def test_loader_letterbox_and_unknown_region(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (64, 32), "black").save(root / "a.png")
            Image.new("RGB", (32, 64), "black").save(root / "b.png")
            document = {
                "schema": "calli-paired-regions-v1",
                "pairs": [
                    {
                        "id": "x",
                        "split": "train",
                        "image_a": "a.png",
                        "image_b": "b.png",
                        "complete": [[True, False], [False, False]],
                        "regions": [
                            {
                                "pair_id": "one",
                                "group": "BF",
                                "a": {
                                    "present": True,
                                    "source": "human",
                                    "polygons": [[[0, 0], [1, 0], [1, 1], [0, 1]]],
                                },
                                "b": {
                                    "present": None,
                                    "source": "unannotated",
                                    "polygons": [],
                                },
                            }
                        ],
                    }
                ],
            }
            path = root / "pairs.json"
            path.write_text(json.dumps(document))
            images, t = PairDataset(path, size=64)[0]
            self.assertEqual(images.shape, (2, 3, 64, 64))
            self.assertEqual(float(t["masks"][0, 0].sum()), 64 * 32)
            self.assertEqual(float(t["masks"][0, 1].sum()), 0)
            self.assertEqual(float(t["present"][0, 1]), -1)
            self.assertTrue((images[0, :, 0, :] == 1).all())
            self.assertTrue((images[1, :, :, 0] == 1).all())


if __name__ == "__main__":
    unittest.main()
