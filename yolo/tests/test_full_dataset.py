"""Full-corpus coordinate, availability, and leakage-group split contracts."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location('build_full_dataset', ROOT / 'build_full_dataset.py')
full = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(full)


def sample(identifier, author, raw, aliases, pixels):
    return {'id': identifier, 'author': author, 'raw_sha256': raw,
            'character_aliases': aliases, 'all_denoised_pixel_sha256': pixels,
            'has_bf': True, 'has_jt': True, 'instances': []}


class FullDatasetTests(unittest.TestCase):
    def test_canonical_copy_suffixes_do_not_guess_variants(self):
        self.assertEqual(full.canonical_character('夫2.png'), '夫')
        self.assertEqual(full.canonical_character('𭤔.png'), '𭤔')
        self.assertNotEqual(full.canonical_character('岡.png'), full.canonical_character('崗.png'))
        with self.assertRaises(ValueError):
            full.canonical_character('unknown.png')

    def test_float_polygon_survives_subpixel_size(self):
        original = [[100.1, 101.1, 100.4, 101.1, 100.4, 101.4, 100.1, 101.4]]
        coords, origin = full.normalized_polygon(original, 1000, 2000, 'pad_square')
        self.assertEqual(origin['kind'], 'original_float_polygon')
        np.testing.assert_allclose(coords[0], [100.1 / 2000, 601.1 / 2000])
        self.assertLess(np.ptp(coords[:, 0]) * 448, 1)
        line = full.export_label(0, coords)
        values = np.asarray(line.split()[1:], float).reshape(-1, 2)
        self.assertEqual(len(np.unique(values, axis=0)), 4)
        self.assertGreater(np.ptp(values[:, 0]), 0)
        self.assertGreater(np.ptp(values[:, 1]), 0)

    def test_geometry_is_identified_from_real_pixels(self):
        info = {'height': 10, 'width': 20}
        polygon = [[2, 1, 5, 1, 5, 3, 2, 3]]
        record = {'image': info, 'annotations': [{'category_name': 'text', 'segmentation': polygon}]}
        mask = full.polygon_mask(polygon, 10, 20)
        rgb = np.repeat(np.where(mask, 0, 255).astype(np.uint8)[..., None], 3, 2)
        native = Image.fromarray(rgb)
        pad = Image.new('RGB', (20, 20), 'white')
        pad.paste(native, (0, 5))
        self.assertEqual(full.source_geometry(record, pad.resize((448, 448), Image.Resampling.LANCZOS)), 'pad_square')
        self.assertEqual(full.source_geometry(record, native.resize((448, 448), Image.Resampling.LANCZOS)), 'direct_resize')
        with self.assertRaises(ValueError):
            full.source_geometry(record, Image.new('RGB', (448, 448), (127, 127, 127)))

    def test_conflicting_task_is_unknown_without_dropping_other_task(self):
        def record(points, name='1-横-起笔'):
            return {'image': {'height': 20, 'width': 20, 'file_name': '字.png', 'id': 1},
                    'annotation_path': Path('source.json'),
                    'annotations': [{'category_name': name, 'segmentation': [points]}]}
        a = record([1, 1, 5, 1, 5, 5, 1, 5])
        b = record([10, 10, 15, 10, 15, 15, 10, 15])
        jt = record([1, 1, 18, 1, 18, 18, 1, 18], '左')
        reliable, issues = full.reliable_tasks({'BF': [a, b], 'JT': [jt]})
        self.assertEqual(set(reliable), {'JT'})
        self.assertEqual(issues['BF']['reason'], 'conflicting_annotation_versions')
        reliable, _ = full.reliable_tasks({'BF': [a, a], 'JT': [jt]})
        self.assertEqual(set(reliable), {'BF', 'JT'})

    def test_leakage_unions_are_global_and_transitive(self):
        rows = [sample('a', 'author1', 'raw_a', ['岡', '崗'], ['pix_a']),
                sample('b', 'author2', 'raw_b', ['崗'], ['pix_b']),
                sample('c', 'author3', 'raw_c', ['新'], ['pix_b']),
                sample('d', 'author4', 'raw_c', ['別'], ['pix_d']),
                sample('e', 'author5', 'raw_e', ['獨'], ['pix_e'])]
        groups = full.leakage_groups(rows)
        self.assertEqual(sorted(len(g['samples']) for g in groups), [1, 4])
        self.assertEqual(len({r['leak_group'] for r in rows[:4]}), 1)
        self.assertEqual(len(rows), 5)  # grouping never merges source identities

    def test_ratio_counts_images_and_keeps_all_authors_in_val(self):
        rows = []
        for author in range(9):
            for index in range(10):
                alias = 'shared' if index == 0 else f'character_{author}_{index}'
                identifier = f'{author}_{index}'
                rows.append(sample(identifier, str(author), identifier, [alias], [identifier]))
        groups = full.leakage_groups(rows)
        report = full.assign_split(groups, seed=42)
        self.assertEqual(report['actual_val_images'], 9)
        self.assertEqual(report['train_images'], 81)
        self.assertEqual(report['primary_absolute_deviation'], 0)
        self.assertEqual(report['author_val_targets'], report['author_val_actual'])
        self.assertEqual(report['pool_val_targets'], report['pool_val_actual'])
        self.assertEqual({r['author'] for r in rows if r['split'] == 'val'}, {str(i) for i in range(9)})
        for group in groups:
            self.assertEqual(len({r['split'] for r in group['samples']}), 1)
        first = {r['id']: r['split'] for r in rows}
        full.assign_split(groups, seed=42)
        self.assertEqual(first, {r['id']: r['split'] for r in rows})


    def test_only_example_of_an_observed_class_stays_in_train(self):
        rows = []
        for author in range(9):
            for index in range(10):
                identifier = f'{author}_{index}'
                alias = 'shared' if index == 0 else f'unique_{identifier}'
                row = sample(identifier, str(author), identifier, [alias], [identifier])
                rows.append(row)
        rows[0]['instances'] = [{'class_id': 69}]
        groups = full.leakage_groups(rows)
        report = full.assign_split(groups, seed=42)
        self.assertEqual(rows[0]['split'], 'train')
        self.assertEqual(report['actual_val_images'], 9)
        self.assertEqual(report['primary_absolute_deviation'], 0)
        self.assertIn(69, report['observed_classes_retained_in_train'])


if __name__ == '__main__':
    unittest.main()
