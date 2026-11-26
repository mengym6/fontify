"""Grouped official AP extraction, including absent/undetected GT classes."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from official_metrics import grouped_mask_map_scalars
import partial_supervision


class GroupedMaskAPTests(unittest.TestCase):
    def test_group_means_use_real_class_indices_and_all_ten_thresholds(self):
        rows = np.array([[.8] + [.4] * 9, [.2] + [.1] * 9, [.9] + [.7] * 9])
        metric = SimpleNamespace(ap_class_index=np.array([50, 2, 49]), all_ap=rows)
        actual = grouped_mask_map_scalars(metric)
        self.assertAlmostEqual(actual['metrics/BF/mAP50(M)'], .55)
        self.assertAlmostEqual(actual['metrics/BF/mAP50-95(M)'], rows[[1, 2]].mean())
        self.assertAlmostEqual(actual['metrics/JT/mAP50(M)'], .8)
        self.assertAlmostEqual(actual['metrics/JT/mAP50-95(M)'], .44)

    def test_gt_class_without_predictions_contributes_zero(self):
        metric = SimpleNamespace(ap_class_index=[0, 1, 50],
                                 all_ap=np.array([[.8] * 10, [0.] * 10, [0.] * 10]))
        actual = grouped_mask_map_scalars(metric)
        self.assertEqual(actual['metrics/BF/mAP50(M)'], .4)
        self.assertEqual(actual['metrics/JT/mAP50(M)'], 0.0)
        self.assertEqual(actual['metrics/JT/mAP50-95(M)'], 0.0)

    def test_absent_group_is_omitted_not_filled_with_zero_or_global_mean(self):
        metric = SimpleNamespace(ap_class_index=[69], all_ap=np.full((1, 10), .7))
        actual = grouped_mask_map_scalars(metric)
        self.assertEqual(set(actual), {'metrics/JT/mAP50(M)', 'metrics/JT/mAP50-95(M)'})
        self.assertEqual(grouped_mask_map_scalars(SimpleNamespace(ap_class_index=[], all_ap=[])), {})

    def test_maps_filled_values_are_never_accessed_and_inputs_are_unchanged(self):
        class Metric:
            ap_class_index = np.array([0, 50])
            all_ap = np.array([[.2] * 10, [.8] * 10])
            @property
            def maps(self):
                raise AssertionError('maps contains filled absent-class values and must not be used')
        metric = Metric()
        before = metric.all_ap.copy()
        actual = grouped_mask_map_scalars(metric)
        self.assertAlmostEqual(actual['metrics/BF/mAP50-95(M)'], .2)
        self.assertAlmostEqual(actual['metrics/JT/mAP50-95(M)'], .8)
        np.testing.assert_array_equal(metric.all_ap, before)

    def test_misaligned_ap_rows_are_rejected(self):
        with self.assertRaises(ValueError):
            grouped_mask_map_scalars(SimpleNamespace(ap_class_index=[0, 50], all_ap=np.zeros((1, 10))))

    def test_validator_get_stats_is_unchanged_and_replaces_grouped_sidechannel(self):
        class NativeValidator:
            def get_stats(self):
                self.native_calls += 1
                return dict(self.native_stats)
        segment = ModuleType('ultralytics.models.yolo.segment')
        segment.SegmentationTrainer = type('NativeTrainer', (), {})
        segment.SegmentationValidator = NativeValidator
        dataset = ModuleType('ultralytics.data.dataset')
        dataset.YOLODataset = type('NativeDataset', (), {})
        build = ModuleType('ultralytics.data.build')
        build.get_split_fraction = lambda *args: 1.0
        utils = ModuleType('ultralytics.utils')
        utils.colorstr = lambda value: value
        torch_utils = ModuleType('ultralytics.utils.torch_utils')
        torch_utils.unwrap_model = lambda value: value
        modules = {'torch': ModuleType('torch'), 'ultralytics.data.dataset': dataset,
                   'ultralytics.data.build': build, 'ultralytics.models.yolo.segment': segment,
                   'ultralytics.utils': utils, 'ultralytics.utils.torch_utils': torch_utils}
        with patch.dict(sys.modules, modules):
            cls = partial_supervision._training_classes()['PartialSegmentationValidator']
            validator = cls()
            validator.native_calls = 0
            validator.native_stats = {'fitness': .975432198765, 'metrics/mAP50-95(B)': .4,
                                      'metrics/mAP50-95(M)': .575432198765, 'metrics/precision(M)': .6}
            validator.metrics = SimpleNamespace(seg=SimpleNamespace(
                ap_class_index=[0, 50], all_ap=np.array([[.1] * 10, [.9] * 10])))
            before = dict(validator.native_stats)
            actual = validator.get_stats()
            self.assertEqual(len(validator.grouped_mask_map), 4)
            self.assertAlmostEqual(validator.grouped_mask_map['metrics/BF/mAP50(M)'], .1)
            validator.metrics.seg = SimpleNamespace(ap_class_index=[50], all_ap=np.full((1, 10), .6))
            second = validator.get_stats()
        self.assertEqual(validator.native_calls, 2)
        self.assertEqual(actual, before)
        self.assertEqual(second, before)
        self.assertEqual(validator.native_stats, before)
        self.assertEqual(set(validator.grouped_mask_map),
                         {'metrics/JT/mAP50(M)', 'metrics/JT/mAP50-95(M)'})


try:
    from ultralytics.utils.metrics import SegmentMetrics
    from ultralytics.models.yolo.segment import SegmentationValidator
except ImportError:
    SegmentMetrics = SegmentationValidator = None


@unittest.skipIf(SegmentMetrics is None, 'Requires pinned native Ultralytics; pure helper tests run without torch')
class NativeMetricIntegrationTests(unittest.TestCase):
    def metric(self, empty_predictions=False):
        metric = SegmentMetrics(names={index: str(index) for index in range(70)})
        predicted = np.empty(0) if empty_predictions else np.array([0, 50])
        confidence = np.empty(0) if empty_predictions else np.array([.9, .8])
        tp = np.zeros((len(predicted), 10), dtype=bool)
        if len(tp):
            tp[0] = True
        metric.update_stats({'tp': tp.copy(), 'tp_m': tp.copy(), 'conf': confidence,
                             'pred_cls': predicted, 'target_cls': np.array([0, 1, 50]),
                             'target_img': np.array([0, 1, 50]), 'im_name': 'synthetic.png'})
        return metric

    def test_native_no_predictions_retains_zero_ap_for_gt_classes(self):
        metric = self.metric(empty_predictions=True)
        metric.process(plot=False)
        self.assertEqual(metric.seg.ap_class_index.tolist(), [0, 1, 50])
        np.testing.assert_array_equal(metric.seg.all_ap, np.zeros((3, 10)))
        self.assertEqual(set(grouped_mask_map_scalars(metric.seg).values()), {0.0})

    def test_actual_native_and_partial_get_stats_have_identical_fitness(self):
        cls = partial_supervision.PartialSegmentationValidator
        base = object.__new__(SegmentationValidator)
        extended = object.__new__(cls)
        for validator in (base, extended):
            validator.metrics = self.metric()
            validator.args = SimpleNamespace(plots=False, save_json=False, task='segment')
            validator.save_dir = Path('.')
        expected, actual = base.get_stats(), extended.get_stats()
        self.assertEqual(actual, expected)
        self.assertEqual(extended.grouped_mask_map, grouped_mask_map_scalars(extended.metrics.seg))
        self.assertEqual(actual['fitness'], extended.metrics.box.map + extended.metrics.seg.map)
        no_prediction_row = list(extended.metrics.seg.ap_class_index).index(1)
        np.testing.assert_array_equal(extended.metrics.seg.all_ap[no_prediction_row], np.zeros(10))


if __name__ == '__main__':
    unittest.main()
