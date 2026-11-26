"""Geometry checks for inferred stroke groups and YOLO polygon export."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / 'build_dataset.py'
spec = importlib.util.spec_from_file_location('build_dataset', MODULE)
build_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_dataset)


class OriginalLabelTests(unittest.TestCase):
    def test_original_50_phase_and_20_component_mapping(self):
        self.assertEqual(len(build_dataset.BF_CLASSES), 50)
        self.assertEqual(len(build_dataset.CLASSES), 70)
        self.assertEqual(build_dataset.class_id_for('1-横-起笔', 'BF'), 0)
        self.assertEqual(build_dataset.class_id_for('2-横-中笔', 'BF'), 1)
        self.assertEqual(build_dataset.class_id_for('左', 'JT'), 50)
        self.assertEqual(build_dataset.class_id_for('外', 'JT'), 69)
        self.assertNotIn('点-中笔', build_dataset.BF_CLASSES)

    def test_identical_annotations_remain_individual_instances(self):
        import argparse
        import contextlib
        import io
        import json
        import tempfile
        from unittest.mock import patch
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            info = {'id': 1, 'file_name': 'glyph.png', 'width': 24, 'height': 24}
            shape = [[4, 4, 18, 4, 18, 18, 4, 18]]
            for kind in ['BF', 'JT']:
                folder = source / ('Writer' + kind)
                (folder / 'images').mkdir(parents=True)
                (folder / 'annotations').mkdir()
                Image.new('RGB', (24, 24), 'white').save(folder / 'images/glyph.png')
                names = ['1-横-起笔', 'text'] if kind == 'BF' else ['左', 'text']
                anns = [{'id': 1, 'image_id': 1, 'category_id': 1, 'segmentation': shape},
                        {'id': 2, 'image_id': 1, 'category_id': 2, 'segmentation': shape}]
                if kind == 'BF':
                    anns.append({'id': 3, 'image_id': 1, 'category_id': 1, 'segmentation': shape})
                doc = {'images': [info], 'categories': [{'id': i + 1, 'name': n} for i, n in enumerate(names)],
                       'annotations': anns}
                (folder / 'annotations/instances_default.json').write_text(json.dumps(doc))
            val = root / 'val.json'; val.write_text('[]')
            args = argparse.Namespace(source_root=source, output=root / 'dataset',
                                      preview_dir=root / 'preview', val_records=val,
                                      size=24, min_contour_iou=0.98)
            with patch.object(build_dataset, 'draw_preview'), contextlib.redirect_stdout(io.StringIO()):
                build_dataset.build(args)
            manifest = json.loads((args.output / 'manifest.json').read_text())
            instances = manifest['images'][0]['instances']
            self.assertEqual(len(instances), 3)
            self.assertEqual([i['source_annotation_id'] for i in instances if i['source_task'] == 'BF'], [1, 3])
            self.assertEqual([i['class_id'] for i in instances], [0, 0, 50])
            self.assertEqual(len(list(args.output.glob('images/train/*.png'))), 1)
            # An alias target name still identifies holdout by original bytes.
            val.write_text(json.dumps([{'target_path': 'font/train/new/WriterJT/images_text_denoised/glyph.png',
                                        'font_folder': 'WriterJT'}]))
            hashes = build_dataset.holdout_hashes(val, source)
            self.assertIn(manifest['images'][0]['raw_sha256'], hashes)


class PolygonExportTests(unittest.TestCase):
    def test_disjoint_regions_are_preserved(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[5:25, 5:25] = True
        mask[30:50, 30:50] = True
        polygon, iou = build_dataset.mask_to_polygon(mask)
        self.assertGreater(iou, 0.98)
        self.assertTrue((polygon[:, 0] >= 30).any())
        self.assertTrue((polygon[:, 0] <= 5).any())

    def test_hole_is_not_filled_by_a_convex_hull(self):
        mask = np.zeros((80, 80), dtype=bool)
        mask[5:70, 5:70] = True
        mask[25:50, 25:50] = False
        _, iou = build_dataset.mask_to_polygon(mask)
        self.assertGreater(iou, 0.98)

    def test_uncompressed_and_compressed_rle(self):
        from pycocotools import mask as mask_util
        original = np.zeros((5, 4), dtype=np.uint8)
        original[1:4, 1:3] = 1
        compressed = mask_util.encode(np.asfortranarray(original))
        compressed['counts'] = compressed['counts'].decode('ascii')
        decoded = build_dataset.polygon_mask(compressed, 5, 4)
        self.assertTrue(np.array_equal(decoded, original))
        flat = original.flatten(order='F')
        runs, value, count = [], 0, 0
        for item in flat:
            if item == value:
                count += 1
            else:
                runs.append(count); count = 1; value = int(item)
        runs.append(count)
        decoded = build_dataset.polygon_mask({'size': [5, 4], 'counts': runs}, 5, 4)
        self.assertTrue(np.array_equal(decoded, original))

    def test_serialized_float32_coordinates_preserve_integer_pixels(self):
        import cv2
        mask = np.zeros((448, 448), dtype=bool)
        mask[1:58, 4:118] = True
        polygon, _ = build_dataset.mask_to_polygon(mask)
        line = build_dataset.yolo_label_line(0, polygon, 448)
        self.assertEqual(build_dataset.serialized_label_iou(line, mask), 1.0)
        for position in range(448):
            encoded = np.float32(f'{(position + 0.5) / 448:.8f}')
            self.assertEqual(int(np.float32(encoded * np.float32(448))), position)

    def test_padding_preserves_coordinates_and_aspect(self):
        mask = np.zeros((10, 20), dtype=bool)
        mask[0:2, 0:2] = True
        result = build_dataset.pad_resize(mask, size=20)
        self.assertTrue(result[5:7, 0:2].all())
        self.assertEqual(result.sum(), 4)


if __name__ == '__main__':
    unittest.main()
