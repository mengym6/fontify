"""Build the full Calli phase/JT dataset with task availability and grouped 9:1 splits.

Source images and annotations are read only. Character aliases, original-image
hashes and denoised pixel hashes form transitive leakage groups before splitting.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import unicodedata

import numpy as np
from PIL import Image

from build_dataset import (
    CLASSES, class_id_for, file_sha, mask_to_polygon,
    polygon_mask, source_index,
)

SCHEMA = 'calli-full-partial-annotations-v1'


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def canonical_character(filename):
    stem = unicodedata.normalize('NFKC', Path(filename).stem)
    character = re.sub(r'[\s_-]*\d+$', '', stem)
    if len(character) != 1 or 'CJK' not in unicodedata.name(character, ''):
        raise ValueError(f'Character alias requires review: {filename}')
    return character


def pixel_sha(image):
    rgb = image.convert('RGB')
    digest = hashlib.sha256(f'RGB:{rgb.width}:{rgb.height}:'.encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def annotation_signature(record):
    """Compare original class + native raster mask, including multiplicity."""
    info = record['image']
    result = Counter()
    for ann in record['annotations']:
        if ann['category_name'] == 'text':
            continue
        mask = polygon_mask(ann['segmentation'], info['height'], info['width'])
        name = re.sub(r'^\d+-', '', ann['category_name'])
        result[(name, hashlib.sha256(mask.tobytes()).hexdigest())] += 1
    return result


def reliable_tasks(tasks):
    chosen, issues = {}, {}
    for kind in ('BF', 'JT'):
        records = tasks.get(kind, [])
        if not records:
            issues[kind] = {'reason': 'annotation_task_missing'}
            continue
        regions = [a for a in records[0]['annotations'] if a['category_name'] != 'text']
        if len(records) > 1:
            signatures = [annotation_signature(record) for record in records]
            if any(value != signatures[0] for value in signatures[1:]):
                issues[kind] = {
                    'reason': 'conflicting_annotation_versions',
                    'sources': [{'file_name': r['image']['file_name'],
                                 'image_id': r['image']['id'],
                                 'annotation_path': str(r['annotation_path'])} for r in records],
                }
                continue
        if not regions:
            issues[kind] = {'reason': 'no_nontext_annotations',
                            'file_name': records[0]['image']['file_name']}
            continue
        chosen[kind] = records[0]
    return chosen, issues


def source_geometry(record, actual):
    """Identify the real PNG's native-to-target transform by exact reproduction.

    Mirrors Fontify's mask mode: native COCO text union -> black on white ->
    optional centered padding -> LANCZOS resize. Does not rewrite source PNGs.
    """
    info = record['image']
    text = np.zeros((info['height'], info['width']), dtype=bool)
    for ann in record['annotations']:
        if ann['category_name'] == 'text':
            text |= polygon_mask(ann['segmentation'], info['height'], info['width'])
    if not text.any():
        raise ValueError('No text annotation to identify existing image geometry')
    native = Image.fromarray(np.repeat(np.where(text, 0, 255).astype(np.uint8)[..., None], 3, 2))
    side = max(native.size)
    padded = Image.new('RGB', (side, side), 'white')
    padded.paste(native, ((side - native.width) // 2, (side - native.height) // 2))
    expected = np.asarray(padded.resize(actual.size, Image.Resampling.LANCZOS))
    current = np.asarray(actual)
    if np.array_equal(current, expected):
        return 'pad_square'
    expected = np.asarray(native.resize(actual.size, Image.Resampling.LANCZOS))
    if np.array_equal(current, expected):
        return 'direct_resize'
    raise ValueError('Existing denoised PNG does not exactly match pad/direct production geometry')


def choose_image(chosen, source_root):
    absent = []
    for kind in ('BF', 'JT'):
        record = chosen.get(kind)
        if record is None:
            continue
        path = record['image_path'].parent.parent / 'images_text_denoised' / record['image']['file_name']
        if not path.is_file():
            absent.append(str(path.relative_to(source_root)))
            continue
        with Image.open(path) as loaded:
            if loaded.mode != 'RGB' or loaded.size != (448, 448):
                raise ValueError(f'Expected existing RGB 448x448 target: {path}')
            image = loaded.copy()
        geometry = source_geometry(record, image)
        return path, image, {
            'source_task': kind, 'source_path': str(path.relative_to(source_root)),
            'source_sha256': file_sha(path), 'pixel_sha256': pixel_sha(image),
            'geometry': geometry, 'geometry_verification': 'exact production pixel equality',
            'missing_preferred_sources': absent,
        }
    raise FileNotFoundError('No denoised image for reliable annotated tasks: ' + ', '.join(absent))


def normalized_polygon(segmentation, height, width, geometry):
    """Map original floating vertices directly, avoiding 448-pixel quantization."""
    if isinstance(segmentation, dict):
        mask = polygon_mask(segmentation, height, width)
        # RLE starts as pixels: retain its disjoint contours/holes with retraced
        # zero-area bridges, then transform the native pixel centers.
        polygon, native_iou = mask_to_polygon(mask)
        parts = [polygon.astype(float) + 0.5]
        origin = {'kind': 'rle_native_contours', 'native_contour_iou': native_iou,
                  'native_rle': segmentation}
    else:
        polygons = segmentation or []
        if polygons and isinstance(polygons[0], (int, float)):
            polygons = [polygons]
        parts = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in polygons]
        origin = {'kind': 'original_float_polygon'}
    if not parts:
        raise ValueError('Empty source segmentation')
    for part in parts:
        if len(part) < 3 or len(np.unique(part, axis=0)) < 3 or not np.isfinite(part).all():
            raise ValueError('Invalid source polygon vertices')
        signed_area = np.dot(part[:, 0], np.roll(part[:, 1], 1)) - np.dot(part[:, 1], np.roll(part[:, 0], 1))
        if abs(signed_area) <= 1e-10:
            raise ValueError('Zero-area source polygon')
        if (part < 0).any() or (part[:, 0] > width).any() or (part[:, 1] > height).any():
            raise ValueError('Source polygon outside original canvas; needs review')
    polygon = parts[0]
    for part in parts[1:]:
        d2 = ((polygon[:, None] - part[None]) ** 2).sum(2)
        i, j = np.unravel_index(d2.argmin(), d2.shape)
        loop = np.concatenate((part[j:], part[:j + 1]))
        polygon = np.concatenate((polygon[:i + 1], loop, polygon[i:]))
    if geometry == 'pad_square':
        side = max(height, width)
        offset = np.array([(side - width) // 2, (side - height) // 2])
        result = (polygon + offset) / side
    elif geometry == 'direct_resize':
        result = polygon / np.array([width, height])
    else:
        raise ValueError(f'Unknown geometry {geometry}')
    if not np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
        raise ValueError('Invalid normalized polygon')
    return result, origin


def export_label(class_id, normalized):
    return str(class_id) + ' ' + ' '.join(f'{coordinate:.10f}' for coordinate in normalized.reshape(-1))


def leakage_groups(samples):
    """Union all aliases and all available image hashes, not only the primary name."""
    parents = list(range(len(samples)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, sample in enumerate(samples):
        keys = [('raw', sample['raw_sha256'])]
        keys += [('character', char) for char in sample['character_aliases']]
        keys += [('pixels', digest) for digest in sample['all_denoised_pixel_sha256']]
        for key in keys:
            if key in seen:
                parents[find(index)] = find(seen[key])
            else:
                seen[key] = index
    grouped = defaultdict(list)
    for index, sample in enumerate(samples):
        grouped[find(index)].append(sample)
    groups = []
    for rows in grouped.values():
        identifier = stable_hash(sorted(row['id'] for row in rows))
        for row in rows:
            row['leak_group'] = identifier
        groups.append({'id': identifier, 'samples': rows})
    return sorted(groups, key=lambda row: row['id'])


def assign_split(groups, seed=42, val_fraction=0.1):
    """Choose leakage groups with exact image count and balanced source/task pools.

    Primary objective: minimum summed absolute deviations from proportionally
    apportioned author/pool counts. A seed-fixed tie-break has total magnitude
    below 0.001, so it cannot outweigh one image of primary integer deviation.
    Every observed class retains at least one annotated training image.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csc_matrix

    groups = sorted(groups, key=lambda group: group['id'])
    samples = [row for group in groups for row in group['samples']]
    total = len(samples)
    target = int(total * val_fraction + 0.5)
    authors = sorted({row['author'] for row in samples})
    pools = ['bf_only', 'dual', 'jt_only']

    def pool(row):
        return 'dual' if row['has_bf'] and row['has_jt'] else 'bf_only' if row['has_bf'] else 'jt_only'

    categories = [('author', author) for author in authors] + [('pool', name) for name in pools]
    category_counts = np.asarray([
        [sum((row['author'] if kind == 'author' else pool(row)) == name for row in group['samples'])
         for group in groups] for kind, name in categories
    ], dtype=float)

    def apportion(counts):
        quotas = counts * target / total
        result = np.floor(quotas).astype(int)
        ranking = sorted(range(len(quotas)), key=lambda i: (-(quotas[i] - result[i]), i))
        for index in ranking[:target - int(result.sum())]:
            result[index] += 1
        return result

    author_targets = apportion(category_counts[:len(authors)].sum(1))
    pool_targets = apportion(category_counts[len(authors):].sum(1))
    targets = np.concatenate((author_targets, pool_targets))
    group_count, category_count = len(groups), len(categories)
    matrix, lower, upper = [], [], []

    def constraint(coefficients, low, high, deviation=None):
        extra = np.zeros(category_count)
        if deviation is not None:
            extra[deviation] = -1.0
        matrix.append(np.concatenate((coefficients, extra)))
        lower.append(low)
        upper.append(high)

    constraint(np.asarray([len(group['samples']) for group in groups]), target, target)
    for index, counts in enumerate(category_counts):
        constraint(counts, -np.inf, targets[index], index)
        constraint(-counts, -np.inf, -targets[index], index)
    for counts in category_counts[:len(authors)]:
        constraint(counts, 1, counts.sum() - 1)
    observed_classes = []
    for class_id in range(len(CLASSES)):
        counts = np.asarray([
            sum(any(item['class_id'] == class_id for item in row.get('instances', []))
                for row in group['samples']) for group in groups
        ], dtype=float)
        if counts.sum():
            constraint(counts, -np.inf, counts.sum() - 1)
            observed_classes.append(class_id)
    rng = np.random.default_rng(seed)
    objective = np.concatenate((rng.uniform(0, 1, group_count) * .001 / group_count,
                                np.ones(category_count)))
    result = milp(
        objective, integrality=np.concatenate((np.ones(group_count), np.zeros(category_count))),
        bounds=Bounds(np.zeros(group_count + category_count),
                      np.concatenate((np.ones(group_count), np.full(category_count, np.inf)))),
        constraints=LinearConstraint(csc_matrix(matrix), lower, upper),
        options={'mip_rel_gap': 0.0, 'time_limit': 45.0},
    )
    if result.status != 0 or result.x is None:
        raise ValueError(f'Balanced grouped split did not reach a certified optimum: {result.message}')
    selected = np.rint(result.x[:group_count]).astype(int)
    for choose, group in zip(selected, groups):
        for row in group['samples']:
            row['split'] = 'val' if choose else 'train'
    actual = np.rint(category_counts @ selected).astype(int)
    actual_val = sum(len(group['samples']) * choose for choose, group in zip(selected, groups))
    if actual_val != target:
        raise RuntimeError('Solver returned a split with the wrong image count')
    return {
        'target_val_images': target, 'actual_val_images': int(actual_val),
        'train_images': total - int(actual_val), 'val_fraction': float(actual_val / total),
        'val_authors': authors, 'seed': seed,
        'method': 'binary leakage-group MILP; exact image total; author/pool absolute deviations',
        'author_val_targets': dict(zip(authors, map(int, author_targets))),
        'author_val_actual': dict(zip(authors, map(int, actual[:len(authors)]))),
        'pool_val_targets': dict(zip(pools, map(int, pool_targets))),
        'pool_val_actual': dict(zip(pools, map(int, actual[len(authors):]))),
        'primary_absolute_deviation': int(np.abs(actual - targets).sum()),
        'solver_status': 'optimal', 'solver_mip_gap': float(result.mip_gap),
        'tie_break_total_bound': .001,
        'observed_classes_retained_in_train': observed_classes,
        'selection_inputs': 'image/author/annotation availability/class presence and leakage groups only; no model outputs',
    }


def collect(source_root):
    indexed, ann_paths = source_index(source_root)
    samples, exclusions, task_issues, invalid_annotations = [], [], [], []
    for (author, raw_sha), tasks in sorted(indexed.items()):
        identifier = author + '_' + raw_sha[:16]
        aliases = sorted({canonical_character(r['image']['file_name']) for records in tasks.values() for r in records})
        all_pixel_hashes = set()
        source_records = []
        for kind, records in tasks.items():
            for record in records:
                path = record['image_path'].parent.parent / 'images_text_denoised' / record['image']['file_name']
                if path.is_file():
                    with Image.open(path) as image:
                        all_pixel_hashes.add(pixel_sha(image))
                source_records.append({'task': kind, 'image_file_name': record['image']['file_name'],
                                       'image_id': record['image']['id'],
                                       'raw_path': str(record['image_path'].relative_to(source_root)),
                                       'annotation_path': str(record['annotation_path'].relative_to(source_root))})
        chosen, issues = reliable_tasks(tasks)
        for task, issue in issues.items():
            if issue['reason'] != 'annotation_task_missing':
                task_issues.append({'id': identifier, 'task': task, **issue})
        if not chosen:
            exclusions.append({'id': identifier, 'reason': 'no_reliable_annotation_task',
                               'sources': source_records, 'issues': issues})
            continue
        try:
            image_path, _, image_source = choose_image(chosen, source_root)
        except (FileNotFoundError, ValueError) as exc:
            exclusions.append({'id': identifier, 'reason': 'denoised_source_unavailable_or_geometry_unverified',
                               'detail': str(exc), 'sources': source_records})
            continue
        instances = []
        for kind, record in chosen.items():
            info = record['image']
            for ann in record['annotations']:
                if ann['category_name'] == 'text':
                    continue
                try:
                    class_id = class_id_for(ann['category_name'], kind)
                    coords, origin = normalized_polygon(ann['segmentation'], info['height'], info['width'], image_source['geometry'])
                    line = export_label(class_id, coords)
                    # Check the actual serialized geometry, including subpixel polygons.
                    readback = np.asarray(line.split()[1:], dtype=np.float64).reshape(-1, 2)
                    if len(np.unique(readback, axis=0)) < 3:
                        raise ValueError('Serialized vertices collapse; annotation needs review')
                except ValueError as exc:
                    invalid_annotations.append({'id': identifier, 'task': kind, 'annotation_id': ann['id'],
                                                'reason': str(exc), 'annotation_path': str(record['annotation_path'].relative_to(source_root))})
                    continue
                instances.append({'class_id': class_id, 'source_task': kind, 'source_annotation_id': ann['id'],
                                  'source_category_name': ann['category_name'], 'source_image_id': info['id'],
                                  'annotation_path': str(record['annotation_path'].relative_to(source_root)),
                                  'coordinates': readback.tolist(), 'export': origin})
        has_bf = any(i['source_task'] == 'BF' for i in instances)
        has_jt = any(i['source_task'] == 'JT' for i in instances)
        if not has_bf and not has_jt:
            exclusions.append({'id': identifier, 'reason': 'no_valid_annotations', 'sources': source_records})
            continue
        samples.append({'id': identifier, 'author': author, 'raw_sha256': raw_sha,
                        'character': '|'.join(aliases), 'character_aliases': aliases,
                        'character_name_ambiguous': len(aliases) > 1,
                        'all_denoised_pixel_sha256': sorted(all_pixel_hashes),
                        'has_bf': has_bf, 'has_jt': has_jt, 'annotation_task_issues': issues,
                        'source_records': source_records, 'denoised_source': image_source,
                        'instances': instances, '_source_image': str(image_path)})
        if len(samples) % 200 == 0:
            print(f'Prepared {len(samples)} full-corpus identities', flush=True)
    return samples, {'raw_identity_count': len(indexed), 'source_record_count': sum(len(r) for tasks in indexed.values() for r in tasks.values()),
                     'source_annotations': {str(p.relative_to(source_root)): file_sha(p) for p in ann_paths},
                     'excluded_images': exclusions, 'unknown_task_issues': task_issues,
                     'invalid_annotations': invalid_annotations}


def build(args):
    source = args.source_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'New output must not already exist: {output}')
    samples, audit = collect(source)
    groups = leakage_groups(samples)
    split = assign_split(groups, seed=args.seed)
    split_id = stable_hash({'schema': SCHEMA, 'seed': args.seed,
                           'groups': sorted((s['id'], s['split'], s['leak_group']) for s in samples),
                           'source_annotations': audit['source_annotations']})
    output.mkdir(parents=True)
    for part in ('train', 'val'):
        for kind in ('images', 'labels'):
            (output / kind / part).mkdir(parents=True)
    availability = {}
    for row in samples:
        source_image = Path(row.pop('_source_image'))
        row['image'] = f"images/{row['split']}/{row['id']}.png"
        row['label'] = f"labels/{row['split']}/{row['id']}.txt"
        shutil.copyfile(source_image, output / row['image'])
        if file_sha(output / row['image']) != row['denoised_source']['source_sha256']:
            raise RuntimeError('Exported PNG differs from Fontify denoised source')
        lines = [export_label(item['class_id'], np.asarray(item['coordinates'])) for item in row['instances']]
        (output / row['label']).write_text('\n'.join(lines) + '\n', encoding='utf-8')
        availability[row['image']] = {key: row[key] for key in ('has_bf', 'has_jt', 'author', 'character', 'leak_group')}
    lines = ['path: .', 'train: images/train', 'val: images/val',
             'annotation_availability: annotation_availability.json', f'split_id: {split_id}', 'names:']
    lines.extend(f"  {item['id']}: {item['name']}" for item in CLASSES)
    (output / 'data.yaml').write_text('\n'.join(lines) + '\n')
    (output / 'classes.json').write_text(json.dumps(CLASSES, ensure_ascii=False, indent=2) + '\n')
    (output / 'annotation_availability.json').write_text(json.dumps(availability, ensure_ascii=False, indent=2) + '\n')
    (output / 'manifest.json').write_text(json.dumps({'schema': SCHEMA, 'split_id': split_id, 'images': samples, **audit}, ensure_ascii=False, indent=2) + '\n')
    class_counts = {part: Counter(item['class_id'] for row in samples if row['split'] == part for item in row['instances'])
                    for part in ('train', 'val')}
    summary = {'schema': SCHEMA, 'split_id': split_id, 'image_count': len(samples), 'split': split,
               'leak_groups': len(groups), 'image_sources': dict(Counter(row['denoised_source']['source_task'] for row in samples)),
               'image_geometry': dict(Counter(row['denoised_source']['geometry'] for row in samples)),
               'pools': dict(Counter('dual' if r['has_bf'] and r['has_jt'] else 'bf_only' if r['has_bf'] else 'jt_only' for r in samples)),
               'by_author': {author: dict(Counter(r['split'] for r in samples if r['author'] == author)) for author in sorted({r['author'] for r in samples})},
               'class_instances': {part: {c['name']: class_counts[part][c['id']] for c in CLASSES} for part in ('train', 'val')},
               'instances': {part: sum(c.values()) for part, c in class_counts.items()},
               'source_record_count': audit['source_record_count'], 'source_identity_count': audit['raw_identity_count'],
               'excluded_images': audit['excluded_images'], 'unknown_task_issues': audit['unknown_task_issues'],
               'invalid_annotation_count': len(audit['invalid_annotations']),
               'annotation_policy': 'Each original annotation is one instance. Unknown BF/JT tasks MUST be masked out in classification/semantic loss.',
               'canonical_policy': 'NFKC plus trailing copy-number removal; all same-original aliases join leakage groups, no semantic alias guessing.',
               'geometry_policy': 'Original polygon floats map directly to confirmed per-image geometry; RLE retains native contours. Images are copied unchanged.',
               'initialization_policy': 'New split requires fresh COCO initialization; old model checkpoints have seen some new validation images.',
               'ratio_unit': 'source-image identity count; not annotation count',
               'source_annotations': audit['source_annotations']}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('image_count', 'split_id', 'split', 'pools', 'instances', 'image_geometry', 'invalid_annotation_count')}, ensure_ascii=False, indent=2))
    return summary


def parser():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=root.parent / 'fontdata_example/font/train/new')
    p.add_argument('--output', type=Path, default=root / 'dataset_full')
    p.add_argument('--seed', type=int, default=42)
    return p


if __name__ == '__main__':
    build(parser().parse_args())
