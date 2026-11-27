import json, tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from yolo.correspondence_v9.model import GTCorrespondence
from yolo.correspondence_v9.loss import CorrespondenceLoss
from yolo.correspondence_v9.metrics import evaluate_sample, summarize
from yolo.correspondence_v9.prepare import convert
from yolo.correspondence_v9.data import read_data
from yolo.correspondence_v9.predict import encode_prediction
from yolo.correspondence_v9.visualization import preview


class TinyEncoder(nn.Module):
    channels = 8
    mask_channels = 4

    def __init__(self):
        super().__init__()
        self.feature = nn.Conv2d(3, 8, 3, padding=1, stride=4)
        self.proto = nn.Conv2d(8, 4, 1)

    def forward(self, x):
        f = self.feature(x)
        return f, self.proto(f)


def target():
    masks = torch.zeros(3, 8, 8)
    masks[0, :3, :3] = 1
    masks[1, 5:, 5:] = 1
    masks[2, 3:5, 3:5] = 1
    return {
        "gt_masks": masks,
        "gen_masks": masks.clone(),
        "matchable": torch.tensor([1, 1, 0]),
        "groups": torch.tensor([0, 0, 1]),
        "gt_ids": ["a", "b", "c"],
        "original_sizes": [(32, 32), (32, 32)],
    }


class V9Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)

    def test_preview_accepts_region_that_disappears_when_downsampled(self):
        mask = torch.zeros(1, 320, 320)
        mask[0, 2, 2] = 1
        t = {"gt_masks": mask, "gen_masks": mask, "groups": torch.tensor([0]), "matchable": torch.tensor([1])}
        out = {"mask_logits": mask * 20 - 10, "match_logits": torch.tensor([5.])}
        image = preview(torch.ones(2, 3, 320, 320), t, out)
        self.assertEqual(image.shape, (596, 768, 3))

    def test_one_output_per_anchor_and_both_images_receive_gradients(self):
        m = GTCorrespondence(TinyEncoder(), width=16, layers=1, heads=4, token_grid=2)
        t = target()
        x = torch.rand(1, 2, 3, 32, 32, requires_grad=True)
        o = m(x, [t["gt_masks"]], [t["groups"]])
        self.assertEqual(o[0]["mask_logits"].shape, (3, 8, 8))
        loss, _ = CorrespondenceLoss()(o, [t])
        loss.backward()
        for side in (0, 1):
            self.assertGreater(float(x.grad[:, side].abs().sum()), 0)

    def test_anchor_permutation_preserves_identity(self):
        m = GTCorrespondence(
            TinyEncoder(), width=16, layers=1, heads=4, token_grid=2
        ).eval()
        t = target()
        x = torch.rand(1, 2, 3, 32, 32)
        order = torch.tensor([2, 0, 1])
        with torch.no_grad():
            a = m(x, [t["gt_masks"]], [t["groups"]])[0]
            b = m(x, [t["gt_masks"][order]], [t["groups"][order]])[0]
        torch.testing.assert_close(
            a["mask_logits"][order], b["mask_logits"], atol=1e-5, rtol=1e-5
        )

    def test_unmatchable_has_rejection_but_no_mask_gradient(self):
        t = target()
        o = {
            "mask_logits": torch.randn(3, 8, 8, requires_grad=True),
            "match_logits": torch.zeros(3, requires_grad=True),
        }
        loss, _ = CorrespondenceLoss()([o], [t])
        loss.backward()
        self.assertEqual(float(o["mask_logits"].grad[2].abs().sum()), 0)
        self.assertGreater(float(o["match_logits"].grad[2]), 0)

    def test_unknown_anchor_does_not_become_negative(self):
        t = target()
        t["matchable"][2] = -1
        o = {
            "mask_logits": torch.randn(3, 8, 8, requires_grad=True),
            "match_logits": torch.zeros(3, requires_grad=True),
        }
        loss, _ = CorrespondenceLoss()([o], [t])
        loss.backward()
        self.assertEqual(float(o["match_logits"].grad[2]), 0)

    def test_perfect_matching_and_rejection_metrics(self):
        t = target()
        g = t["gen_masks"].numpy() > 0
        rows = evaluate_sample([0.9, 0.9, 0.1], g, g, t["matchable"], t["groups"])
        s = summarize(rows)["all"]
        self.assertEqual(s["correct_match"], 2)
        self.assertEqual(s["correct_reject"], 1)
        self.assertEqual(s["match_precision"], 1)
        self.assertEqual(s["unmatchable_recall"], 1)

    def test_swapped_regions_are_wrong_even_if_both_have_boxes(self):
        t = target()
        g = t["gen_masks"].numpy() > 0
        pred = g[[1, 0, 2]]
        s = summarize(
            evaluate_sample([0.9, 0.9, 0.1], pred, g, t["matchable"], t["groups"])
        )["all"]
        self.assertEqual(s["wrong_match"], 2)

    def test_always_reject_and_always_accept(self):
        t = target()
        g = t["gen_masks"].numpy() > 0
        a = summarize(evaluate_sample([0.1] * 3, g, g, t["matchable"], t["groups"]))[
            "all"
        ]
        self.assertEqual(a["false_reject"], 2)
        b = summarize(evaluate_sample([0.9] * 3, g, g, t["matchable"], t["groups"]))[
            "all"
        ]
        self.assertEqual(b["false_accept"], 1)

    def test_no_negative_support_is_not_success(self):
        t = target()
        g = t["gen_masks"][:2].numpy() > 0
        s = summarize(evaluate_sample([0.9, 0.9], g, g, [1, 1], [0, 0]))["all"]
        self.assertIsNone(s["unmatchable_recall"])
        self.assertEqual(s["unmatchable_support"], 0)

    def test_duplicate_union_predictions_are_not_two_correct_matches(self):
        t = target()
        g = t["gen_masks"][:2].numpy() > 0
        union = g[0] | g[1]
        pred = np.stack([union, union])
        s = summarize(evaluate_sample([0.9, 0.9], pred, g, [1, 1], [0, 0]))["all"]
        self.assertEqual(s["duplicate_conflict_queries"], 2)
        self.assertEqual(s["correct_match"], 0)

    def test_rejected_prediction_has_no_invented_gen_mask(self):
        t = target()
        o = {
            "match_logits": torch.tensor([5.0, 5.0, -5.0]),
            "mask_logits": torch.ones(3, 8, 8),
        }
        rows = encode_prediction(o, t, 32)
        self.assertEqual(rows[2]["status"], "unmatched")
        self.assertIsNone(rows[2]["gen_mask"])

    def test_conversion_drops_gen_only_and_missing_placeholder(self):
        poly = [[[0, 0], [1, 0], [1, 1]]]
        row = {
            "id": "s",
            "split": "train",
            "leak_group": "g",
            "image_a": "a.png",
            "image_b": "b.png",
            "regions": [
                {
                    "pair_id": "normal",
                    "group": "BF",
                    "a": {"present": True, "polygons": poly},
                    "b": {"present": True, "polygons": poly},
                },
                {
                    "pair_id": "missing",
                    "group": "BF",
                    "a": {"present": True, "polygons": poly},
                    "b": {
                        "present": False,
                        "polygons": poly,
                        "mask_role": "expected_missing_region",
                    },
                },
                {
                    "pair_id": "extra",
                    "group": "BF",
                    "a": {"present": False, "polygons": []},
                    "b": {"present": True, "polygons": poly},
                },
            ],
        }
        with tempfile.TemporaryDirectory() as d:
            a = Path(d) / "old.json"
            b = Path(d) / "new.json"
            a.write_text(
                json.dumps({"schema": "calli-paired-regions-v1", "pairs": [row]})
            )
            report = convert(a, b)
            new = read_data(b)
            self.assertEqual(report["generated_only_dropped"], 1)
            self.assertEqual(len(new["pairs"][0]["regions"]), 2)
            self.assertEqual(new["pairs"][0]["regions"][1]["gen_polygons"], [])


if __name__ == "__main__":
    unittest.main()
