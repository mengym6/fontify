"""Region-count semantics and optimal assignment, using only NumPy/SciPy."""

import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "region_metrics.py"
SPEC = importlib.util.spec_from_file_location("region_metrics", MODULE)
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


def region(class_id=0, positions=range(10), confidence=None):
    mask = np.zeros((4, 20), dtype=bool)
    mask.flat[list(positions)] = True
    result = {"class_id": class_id, "mask": mask}
    if confidence is not None:
        result["confidence"] = confidence
    return result


class RegionMetricsTests(unittest.TestCase):
    def test_wrong_fine_label_preserves_geometric_recall(self):
        report = metrics.evaluate_regions([region(0)], [region(1)])
        bf = report["BF"]
        self.assertEqual(bf["matched_count"], 1)
        self.assertEqual(bf["region_recall"], 1.0)
        self.assertEqual(bf["region_precision"], 1.0)
        self.assertEqual(bf["label_aware_precision"], 0.0)
        self.assertEqual(bf["label_accuracy"], 0.0)
        self.assertEqual(bf["miss_rate"], 0.0)
        self.assertEqual(bf["mean_matched_iou"], 1.0)

    def test_overreporting_is_counted_once_per_extra_region(self):
        report = metrics.evaluate_regions([region()], [region(), region()])["BF"]
        self.assertEqual(report["matched_count"], 1)
        self.assertEqual(report["false_positive_count"], 1)
        self.assertEqual(report["region_precision"], 0.5)
        self.assertEqual(report["label_aware_precision"], 0.5)
        self.assertEqual(report["region_recall"], 1.0)
        self.assertEqual(report["mean_matched_iou"], 1.0)

    def test_one_left_component_split_into_wrong_left_top_and_left_bottom(self):
        # GT left = class 50; left-top/bottom are different JT fine labels.
        gt = [region(50, range(20))]
        pred = [region(55, range(10)), region(56, range(10, 20))]
        jt = metrics.evaluate_regions(gt, pred)["JT"]
        self.assertEqual(jt["matched_count"], 1)
        self.assertEqual(jt["pred_count"], 2)
        self.assertEqual(jt["region_recall"], 1.0)
        self.assertEqual(jt["region_precision"], 0.5)  # legacy geometric field
        self.assertEqual(jt["label_aware_precision"], 0.0)
        self.assertEqual(jt["mean_matched_iou"], 0.5)

    def test_split_correctly_labelled_halves_penalize_extra_prediction_and_iou(self):
        gt = [region(50, range(20))]
        pred = [region(50, range(10)), region(50, range(10, 20))]
        jt = metrics.evaluate_regions(gt, pred)["JT"]
        self.assertEqual(jt["region_recall"], 1.0)
        self.assertEqual(jt["label_aware_precision"], 0.5)
        self.assertEqual(jt["mean_matched_iou"], 0.5)

    def test_split_pieces_below_iou_threshold_do_not_count_as_recalled(self):
        gt = [region(50, range(20))]
        pred = [region(50, range(8)), region(50, range(8, 16))]
        jt = metrics.evaluate_regions(gt, pred)["JT"]
        self.assertEqual(jt["region_recall"], 0.0)
        self.assertEqual(jt["label_aware_precision"], 0.0)
        self.assertEqual(jt["false_positive_count"], 2)
        self.assertIsNone(jt["mean_matched_iou"])

    def test_miss_uses_region_count_not_area(self):
        gt = [region(0, range(40)), region(1, [60])]
        bf = metrics.evaluate_regions(gt, [region(0, range(40))])["BF"]
        self.assertEqual(bf["missed_count"], 1)
        self.assertEqual(bf["region_recall"], 0.5)
        self.assertEqual(bf["miss_rate"], 0.5)

    def test_empty_ground_truth(self):
        bf = metrics.evaluate_regions([], [region()])["BF"]
        self.assertIsNone(bf["region_recall"])
        self.assertIsNone(bf["miss_rate"])
        self.assertIsNone(bf["label_accuracy"])
        self.assertIsNone(bf["mean_matched_iou"])
        self.assertEqual(bf["region_precision"], 0.0)
        self.assertEqual(bf["label_aware_precision"], 0.0)
        self.assertEqual(bf["false_positive_count"], 1)

    def test_empty_predictions(self):
        jt = metrics.evaluate_regions([region(50)], [])["JT"]
        self.assertEqual(jt["region_recall"], 0.0)
        self.assertEqual(jt["miss_rate"], 1.0)
        self.assertIsNone(jt["region_precision"])
        self.assertIsNone(jt["label_aware_precision"])
        self.assertIsNone(jt["label_accuracy"])

    def test_both_empty_and_empty_aggregation_are_undefined(self):
        report = metrics.evaluate_regions([], [])
        self.assertEqual(report["BF"]["labeled_images"], 1)
        self.assertEqual(metrics.aggregate_region_metrics([])["BF"]["labeled_images"], 0)
        for group in report.values():
            for key in ("region_recall", "region_precision", "label_accuracy",
                        "label_aware_precision", "miss_rate"):
                self.assertIsNone(group[key])

    def test_unknown_task_predictions_are_not_false_positives(self):
        report = metrics.evaluate_regions([region(0)], [region(0), region(50)], known_tasks={"BF"})
        self.assertEqual(report["BF"]["labeled_images"], 1)
        self.assertEqual(report["JT"]["labeled_images"], 0)
        self.assertEqual(report["JT"]["pred_count"], 0)
        self.assertEqual(report["JT"]["false_positive_count"], 0)
        self.assertIsNone(report["JT"]["label_aware_precision"])
        self.assertEqual(report["all"]["pred_count"], 1)
        self.assertEqual(report["all"]["labeled_images"], 1)

    def test_known_empty_task_predictions_are_false_positives(self):
        report = metrics.evaluate_regions([], [region(50)], known_tasks={"JT": True})
        self.assertEqual(report["JT"]["labeled_images"], 1)
        self.assertEqual(report["JT"]["false_positive_count"], 1)
        self.assertEqual(report["JT"]["label_aware_precision"], 0.0)

    def test_partial_task_aggregation_counts_labeled_images_not_task_sum(self):
        acc = metrics.RegionMetricsAccumulator()
        acc.update([region(0)], [region(0), region(50)], known_tasks={"BF"})
        acc.update([region(50)], [region(0), region(50)], known_tasks={"JT"})
        acc.update([region(0), region(50)], [region(0), region(50)])
        acc.update([], [region(0), region(50)], known_tasks=set())
        result = acc.summary()
        self.assertEqual(result["BF"]["labeled_images"], 2)
        self.assertEqual(result["JT"]["labeled_images"], 2)
        self.assertEqual(result["all"]["labeled_images"], 3)
        self.assertEqual(result["all"]["pred_count"], 4)
        self.assertEqual(result["all"]["label_aware_precision"], 1.0)

    def test_invalid_known_tasks_rejected(self):
        for tasks in ({"has_bf": True}, "BF", {"BF": "false"}, {"unknown"}):
            with self.assertRaises(ValueError):
                metrics.evaluate_regions([], [], known_tasks=tasks)

    def test_overlapping_instances_remain_separate(self):
        gt = [region(50, range(20)), region(51, range(5, 15))]
        jt = metrics.evaluate_regions(gt, gt, deduplicate_gt=True)["JT"]
        self.assertEqual(jt["gt_count"], 2)
        self.assertEqual(jt["matched_count"], 2)
        self.assertEqual(jt["label_accuracy"], 1.0)
        self.assertEqual(jt["mean_matched_iou"], 1.0)

    def test_cross_group_error_is_miss_and_false_positive(self):
        report = metrics.evaluate_regions([region(0)], [region(50)])
        self.assertEqual(report["BF"]["missed_count"], 1)
        self.assertEqual(report["JT"]["false_positive_count"], 1)
        self.assertEqual(report["all"]["matched_count"], 0)
        self.assertEqual(report["all"]["region_recall"], 0.0)
        self.assertIsNone(report["all"]["label_accuracy"])

    def test_confidence_threshold_inclusive_and_missing_confidence(self):
        pred = [region(0, confidence=.24999), region(0, confidence=.25), region(0)]
        bf = metrics.evaluate_regions([region()], pred)["BF"]
        self.assertEqual(bf["pred_count"], 2)
        self.assertEqual(bf["false_positive_count"], 1)
        self.assertEqual(metrics.evaluate_regions([region()], pred, conf_threshold=.5)["BF"]["pred_count"], 1)

    def test_iou_threshold_inclusive(self):
        gt, pred = [region(0, range(10))], [region(0, range(5))]
        self.assertEqual(metrics.evaluate_regions(gt, pred)["BF"]["matched_count"], 1)
        self.assertEqual(metrics.evaluate_regions(gt, pred, iou_threshold=.50001)["BF"]["matched_count"], 0)

    def test_hungarian_maximizes_cardinality_before_iou(self):
        # A greedy first match G0-P0 (IoU 2/3) strands G1. The optimal
        # assignment is G0-P1 (1/2), G1-P0 (2/3), preserving two regions.
        gt = [region(0, range(10)), region(1, range(5, 15))]
        pred = [region(1, range(15)), region(0, range(5))]
        bf = metrics.evaluate_regions(gt, pred)["BF"]
        self.assertEqual(bf["matched_count"], 2)
        self.assertEqual(bf["label_correct_count"], 2)
        self.assertAlmostEqual(bf["mean_matched_iou"], (0.5 + 2 / 3) / 2)

    def test_total_iou_breaks_equal_cardinality(self):
        overlaps = np.array([[.9, .6], [.6, .8]])
        self.assertEqual(metrics._match_iou(overlaps, .5), [(0, 0), (1, 1)])
        self.assertEqual(metrics._match_iou(np.full((2, 3), .49), .5), [])

    def test_deduplication_is_explicit_and_counts_duplicates(self):
        gt = [region(), region()]
        original = gt[0]["mask"].copy()
        kept = metrics.evaluate_regions(gt, [region()])["BF"]
        deduped = metrics.evaluate_regions(gt, [region()], deduplicate_gt=True)["BF"]
        self.assertEqual(kept["gt_count"], 2)
        self.assertEqual(kept["region_recall"], .5)
        self.assertEqual(kept["duplicate_gt_count"], 1)
        self.assertEqual(kept["removed_duplicate_gt_count"], 0)
        self.assertEqual(deduped["raw_gt_count"], 2)
        self.assertEqual(deduped["gt_count"], 1)
        self.assertEqual(deduped["removed_duplicate_gt_count"], 1)
        self.assertEqual(deduped["region_recall"], 1.0)
        unique, removed = metrics.deduplicate_gt_instances(gt)
        self.assertEqual(removed, 1)
        self.assertIs(unique[0], gt[0])
        self.assertEqual(len(gt), 2)
        np.testing.assert_array_equal(gt[0]["mask"], original)

    def test_different_labels_with_identical_masks_are_not_duplicates(self):
        unique, removed = metrics.deduplicate_gt_instances([region(0), region(1)])
        self.assertEqual(len(unique), 2)
        self.assertEqual(removed, 0)

    def test_aggregation_uses_micro_denominators(self):
        first = metrics.evaluate_regions([region()], [region()])
        second = metrics.evaluate_regions([region(0, [i]) for i in range(9)], [])
        total = metrics.aggregate_region_metrics([first, second])["BF"]
        self.assertEqual(total["gt_count"], 10)
        self.assertEqual(total["region_recall"], .1)
        self.assertEqual(total["miss_rate"], .9)
        self.assertEqual(total["region_precision"], 1.0)
        self.assertEqual(total["label_accuracy"], 1.0)
        self.assertEqual(total["label_aware_precision"], 1.0)

    def test_aggregation_weights_label_accuracy_and_iou_by_matches(self):
        first = metrics.evaluate_regions([region()], [region()])
        second = metrics.evaluate_regions(
            [region(50, range(10)), region(51, range(20, 30))],
            [region(52, range(5)), region(53, range(20, 25))],
        )
        all_metrics = metrics.aggregate_region_metrics([first, second])["all"]
        self.assertEqual(all_metrics["label_accuracy"], 1 / 3)
        self.assertEqual(all_metrics["label_aware_precision"], 1 / 3)
        self.assertAlmostEqual(all_metrics["mean_matched_iou"], 2 / 3)

    def test_label_aware_precision_micro_denominator_includes_wrong_and_extra_predictions(self):
        correct = metrics.evaluate_regions([region(0)], [region(0)])
        wrong_and_extra = metrics.evaluate_regions(
            [region(50)], [region(51), region(52, range(30, 40))],
        )
        total = metrics.aggregate_region_metrics([correct, wrong_and_extra])["all"]
        self.assertEqual(total["matched_count"], 2)
        self.assertEqual(total["label_correct_count"], 1)
        self.assertEqual(total["pred_count"], 3)
        self.assertEqual(total["region_recall"], 1.0)
        self.assertEqual(total["region_precision"], 2 / 3)
        self.assertEqual(total["label_aware_precision"], 1 / 3)

    def test_accumulator_is_serializable_and_does_not_retain_masks(self):
        accumulator = metrics.RegionMetricsAccumulator(deduplicate_gt=True)
        report = accumulator.update([region(), region()], [region()], image_id=Path("val/a.png"))
        self.assertEqual(report["image_id"], "val/a.png")
        self.assertEqual(accumulator.summary()["BF"]["removed_duplicate_gt_count"], 1)
        encoded = json.dumps({"per_image": accumulator.per_image, "summary": accumulator.summary()}, allow_nan=False)
        self.assertNotIn('"mask"', encoded)

    def test_tensorboard_displays_only_three_percentage_metrics_per_group(self):
        report = metrics.evaluate_regions(
            [region(), region(), region(50)], [region(), region(51)],
        )
        scalars = metrics.tensorboard_scalars(report)
        self.assertEqual(scalars, {
            "BF/metrics/recall_pct": 50.0,
            "BF/metrics/precision_pct": 100.0,
            "BF/metrics/iou_pct": 100.0,
            "JT/metrics/recall_pct": 100.0,
            "JT/metrics/precision_pct": 0.0,
            "JT/metrics/iou_pct": 100.0,
        })
        self.assertTrue(all(isinstance(value, float) for value in scalars.values()))

    def test_tensorboard_empty_groups_omit_all_main_rates(self):
        scalars = metrics.tensorboard_scalars(metrics.evaluate_regions([], []))
        self.assertEqual(scalars, {})

    def test_invalid_classes_shapes_and_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            metrics.evaluate_regions([region(70)], [])
        invalid_shape = {"class_id": 0, "mask": np.ones((2, 2), bool)}
        with self.assertRaisesRegex(ValueError, "same HxW"):
            metrics.evaluate_regions([region()], [invalid_shape])
        for value in (0, -.1, 1.01, float("nan")):
            with self.assertRaises(ValueError):
                metrics.evaluate_regions([], [], iou_threshold=value)

    def test_empty_masks_are_unmatched_regions(self):
        report = metrics.evaluate_regions([region(0, [])], [region(0, [])])["BF"]
        self.assertEqual(report["matched_count"], 0)
        self.assertEqual(report["missed_count"], 1)
        self.assertEqual(report["false_positive_count"], 1)


if __name__ == "__main__":
    unittest.main()
