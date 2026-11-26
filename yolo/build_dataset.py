"""Export original CalliPhase stage and JT annotations for YOLO segmentation.

Every original non-text annotation is one instance. No phase merging, inferred
stroke identity, or within-image instance deduplication is performed.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

STROKES = (
    '横', '竖', '撇', '捺', '点', '提', '横折', '横钩', '横折钩', '竖钩',
    '竖提', '竖折', '竖弯钩', '斜钩', '弯钩', '卧钩', '撇折',
)
STROKE_NAMES = (
    'heng', 'shu', 'pie', 'na', 'dian', 'ti', 'heng_zhe', 'heng_gou',
    'heng_zhe_gou', 'shu_gou', 'shu_ti', 'shu_zhe', 'shu_wan_gou',
    'xie_gou', 'wan_gou', 'wo_gou', 'pie_zhe',
)
PARTS = (
    '左', '右', '上', '下', '中', '左上', '左下', '右上', '右下', '上左',
    '上右', '下左', '下右', '下中', '中上', '中下', '中左', '中右', '内', '外',
)
PART_NAMES = (
    'left', 'right', 'top', 'bottom', 'center', 'left_top', 'left_bottom',
    'right_top', 'right_bottom', 'top_left', 'top_right', 'bottom_left',
    'bottom_right', 'bottom_center', 'center_top', 'center_bottom',
    'center_left', 'center_right', 'inner', 'outer',
)
STAGES = ('起笔', '中笔', '收笔')
STAGE_NAMES = ('beginning', 'middle', 'ending')
BF_CLASSES = tuple(f'{stroke}-{stage}' for stroke in STROKES for stage in STAGES
                   if not (stroke == '点' and stage == '中笔'))
CLASSES = [
    {'id': index, 'name': f'phase_{STROKE_NAMES[STROKES.index(stroke)]}_{STAGE_NAMES[STAGES.index(stage)]}',
     'name_zh': label, 'kind': 'phase'}
    for index, label in enumerate(BF_CLASSES)
    for stroke, stage in [label.split('-')]
] + [
    {'id': 50 + i, 'name': 'component_' + name, 'name_zh': zh, 'kind': 'component'}
    for i, (zh, name) in enumerate(zip(PARTS, PART_NAMES))
]


def class_id_for(category_name, kind):
    name = re.sub(r'^\d+-', '', category_name)
    if kind == 'BF':
        return BF_CLASSES.index(name)
    if kind == 'JT':
        return 50 + PARTS.index(name)
    raise ValueError(f'Unknown annotation task: {kind}')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def polygon_mask(segmentation, height, width):
    """Decode both COCO polygons and compressed/uncompressed RLE."""
    if isinstance(segmentation, dict):
        from pycocotools import mask as mask_util
        rle = dict(segmentation)
        if rle.get('size') != [height, width]:
            raise ValueError('RLE/image dimensions disagree')
        if isinstance(rle['counts'], list):
            rle = mask_util.frPyObjects(rle, height, width)
        elif isinstance(rle['counts'], str):
            rle['counts'] = rle['counts'].encode('ascii')
        decoded = mask_util.decode(rle)
        return decoded.any(2) if decoded.ndim == 3 else decoded.astype(bool)
    canvas = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    polygons = segmentation or []
    if polygons and isinstance(polygons[0], (int, float)):
        polygons = [polygons]
    for polygon in polygons:
        if len(polygon) < 6 or len(polygon) % 2:
            raise ValueError('Invalid COCO polygon')
        draw.polygon(list(zip(polygon[::2], polygon[1::2])), fill=1)
    return np.asarray(canvas, dtype=bool)


def pad_resize(mask, size=448):
    height, width = mask.shape
    side = max(height, width)
    square = np.zeros((side, side), dtype=np.uint8)
    y, x = (side - height) // 2, (side - width) // 2
    square[y:y + height, x:x + width] = mask
    return np.asarray(Image.fromarray(square).resize(
        (size, size), Image.Resampling.NEAREST)) > 0



def mask_to_polygon(mask, min_iou=0.98):
    """Preserve disjoint components/holes using retraced shortest bridges.

    YOLO txt stores one polygon per instance. A retraced bridge connects contour
    walks without using a convex hull; the rasterized export is checked against
    the entire source mask so no secondary contour silently disappears.
    """
    import cv2
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST,
                                  cv2.CHAIN_APPROX_SIMPLE)
    if any(len(c) < 3 for c in contours):
        raise ValueError('Degenerate contour cannot be preserved in YOLO polygon format')
    contours = [c[:, 0, :].astype(np.int32) for c in contours]
    if not contours:
        raise ValueError('Mask has no polygon with at least three points')
    contours.sort(key=lambda c: abs(cv2.contourArea(c)), reverse=True)
    polygon = contours.pop(0)
    for contour in contours:
        distances = ((polygon[:, None].astype(float) - contour[None]) ** 2).sum(2)
        i, j = np.unravel_index(distances.argmin(), distances.shape)
        loop = np.concatenate((contour[j:], contour[:j + 1]))
        polygon = np.concatenate((polygon[:i + 1], loop, polygon[i:]))
    raster = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(raster, [polygon], 1)
    union = np.logical_or(mask, raster).sum()
    iou = float(np.logical_and(mask, raster).sum() / max(union, 1))
    if iou < min_iou:
        raise ValueError(f'Contour round-trip IoU {iou:.6f} < {min_iou}')
    return polygon, iou


def yolo_label_line(class_id, polygon, size):
    """Encode pixel centers so float32 * size -> int32 recovers each pixel."""
    coordinates = (np.asarray(polygon, dtype=float).reshape(-1) + 0.5) / size
    return str(class_id) + ' ' + ' '.join(f'{x:.8f}' for x in coordinates)


def serialized_label_iou(line, mask):
    import cv2
    values = np.asarray(line.split()[1:], dtype=np.float32).reshape(-1, 2)
    coordinates = (values * np.asarray([mask.shape[1], mask.shape[0]], dtype=np.float32)).astype(np.int32)
    reconstructed = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(reconstructed, [coordinates], 1)
    union = np.logical_or(mask, reconstructed).sum()
    return float(np.logical_and(mask, reconstructed).sum() / max(union, 1))


def source_index(source_root):
    records, annotations = defaultdict(lambda: defaultdict(list)), []
    for ann_path in sorted(source_root.glob('*/annotations/instances_default.json')):
        folder = ann_path.parent.parent
        kind, author = folder.name[-2:], folder.name[:-2]
        if kind not in ('BF', 'JT'):
            continue
        doc = json.loads(ann_path.read_text(encoding='utf-8'))
        categories = {a['id']: a['name'] for a in doc['categories']}
        by_image = defaultdict(list)
        for ann in doc['annotations']:
            by_image[ann['image_id']].append(dict(ann, category_name=categories[ann['category_id']]))
        annotations.append(ann_path)
        for info in doc['images']:
            raw = folder / 'images' / Path(info['file_name']).name
            digest = file_sha(raw)
            with Image.open(raw) as im:
                if im.size != (info['width'], info['height']):
                    raise ValueError(f'Image/COCO shape mismatch: {raw}')
            records[(author, digest)][kind].append({
                'image': info, 'image_path': raw, 'annotation_path': ann_path,
                'annotations': by_image[info['id']],
            })
    return records, annotations


def annotation_signature(record):
    # IDs and annotation order do not define a different annotation version.
    regions = [(a['category_name'], a['segmentation']) for a in record['annotations']]
    return sorted(json.dumps(a, ensure_ascii=False, sort_keys=True) for a in regions)


def holdout_hashes(val_records, source_root):
    rows = json.loads(val_records.read_text(encoding='utf-8'))
    hashes = set()
    for row in rows:
        path = Path(row['target_path'])
        folder = row.get('font_folder', path.parent.parent.name)
        raw = source_root / folder / 'images' / path.name
        hashes.add(file_sha(raw))
    return hashes


def render_record(record, size):
    info = record['image']
    return [(a, pad_resize(polygon_mask(a['segmentation'], info['height'], info['width']), size))
            for a in record['annotations'] if a['category_name'] != 'text']


def denoised_image_source(bf, jt, size, allow_jt_fallback=False):
    """Load an existing Fontify target without any pixel transformation.

    Native annotations use the confirmed centered-square geometry. The target
    PNG is already square/resized and must not be padded or rescaled again.
    """
    candidates = [('BF', bf)] + ([('JT', jt)] if allow_jt_fallback else [])
    for task, record in candidates:
        folder = record['image_path'].parent.parent
        path = folder / 'images_text_denoised' / Path(record['image']['file_name']).name
        if not path.is_file():
            continue
        with Image.open(path) as image:
            if image.size != (size, size):
                raise ValueError(f'Fontify target size differs from label geometry: {path}: {image.size}')
            if image.mode != 'RGB':
                raise ValueError(f'Expected existing RGB Fontify target: {path}: {image.mode}')
            image = image.copy()
        return image, path, {
            'selected_task': task, 'source_sha256': file_sha(path),
            'fallback_reason': 'BF_denoised_file_missing' if task == 'JT' else None,
            'pixel_processing': 'none: source PNG copied byte-for-byte',
        }
    raise FileNotFoundError(f'No existing Fontify denoised target for {bf["image_path"]}')


def exact_mask_rle(mask):
    from pycocotools import mask as mask_util
    encoded = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded['counts'] = encoded['counts'].decode('ascii')
    return encoded


def draw_preview(image, instances, destination, title):
    font = ImageFont.load_default()
    panels = []
    for kind in ('phase', 'component'):
        panel = image.copy().convert('RGBA')
        overlay = Image.new('RGBA', image.size)
        draw = ImageDraw.Draw(overlay)
        count = 0
        for item in instances:
            class_info = CLASSES[item['class_id']]
            if class_info['kind'] != kind:
                continue
            color = ((37 * count + 230) % 220 + 25, (91 * count + 70) % 220 + 25,
                     (63 * count + 140) % 220 + 25)
            points = [tuple(p) for p in item['polygon']]
            draw.polygon(points, fill=(*color, 45), outline=(*color, 255), width=2)
            anchor = min(points, key=lambda p: p[0] + p[1])
            text = f"{count + 1}:{class_info['name'].split('_', 1)[1]}"
            draw.text(anchor, text, font=font, fill=(200, 0, 0, 255), stroke_width=1,
                      stroke_fill='white')
            count += 1
        panel = Image.alpha_composite(panel, overlay).convert('RGB')
        panels.append(panel)
    result = Image.new('RGB', (image.width * 3, image.height + 40), 'white')
    result.paste(image, (0, 40))
    result.paste(panels[0], (image.width, 40))
    result.paste(panels[1], (image.width * 2, 40))
    draw = ImageDraw.Draw(result)
    for x, name in enumerate(('Input', 'Original BF stage instances', 'Original JT component instances')):
        draw.text((x * image.width + 8, 10), name + ' | ' + title, fill='black', font=font)
    result.save(destination)


def build(args):
    source_root, output = args.source_root.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty dataset: {output}')
    selection = None
    prior_excluded = {}
    selection_path = getattr(args, 'selection_manifest', None)
    if selection_path is not None:
        frozen = json.loads(Path(selection_path).read_text(encoding='utf-8'))
        selection = {row['id']: row for row in frozen['images']}
        prior_excluded = {row['id']: row for row in frozen.get('excluded', [])}
    index, ann_files = source_index(source_root)
    val_hashes = holdout_hashes(args.val_records, source_root)
    print(f'Source identities: {len(index)}; dual: {sum(set(v) == {"BF", "JT"} for v in index.values())}', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    preview_dir = args.preview_dir.resolve()
    preview_dir.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val'):
        for directory in ('images', 'labels'):
            (output / directory / split).mkdir(parents=True, exist_ok=True)
    included, excluded = [], []
    class_counts = {split: Counter() for split in ('train', 'val')}
    preview_counts = Counter()
    all_ious = []
    for (author, digest), tasks in sorted(index.items()):
        identity = author + '_' + digest[:16]
        if selection is not None and identity not in selection:
            excluded.append(prior_excluded.get(identity, {
                'id': identity, 'raw_sha256': digest, 'reason': 'not_in_frozen_selection'}))
            continue
        reason = None
        if set(tasks) != {'BF', 'JT'}:
            reason = 'missing_bf_or_jt_annotation'
        elif any(len({tuple(annotation_signature(r)) for r in rows}) > 1 for rows in tasks.values()):
            reason = 'conflicting_annotation_versions'
        if reason:
            excluded.append({'id': identity, 'raw_sha256': digest, 'reason': reason})
            continue
        bf, jt = tasks['BF'][0], tasks['JT'][0]
        instances, task_counts = [], Counter()
        try:
            for kind, record in (('BF', bf), ('JT', jt)):
                for ann, mask in render_record(record, args.size):
                    if not mask.any():
                        raise ValueError('Empty original annotation after resizing')
                    class_id = class_id_for(ann['category_name'], kind)
                    polygon, iou = mask_to_polygon(mask, args.min_contour_iou)
                    label = yolo_label_line(class_id, polygon, args.size)
                    serialized_iou = serialized_label_iou(label, mask)
                    if serialized_iou < args.min_contour_iou:
                        raise ValueError(f'Serialized label IoU {serialized_iou} below threshold')
                    instances.append({'class_id': class_id, 'source_task': kind,
                                      'source_annotation_id': ann['id'],
                                      'source_category_name': ann['category_name'],
                                      'exact_annotation_rle': exact_mask_rle(mask),
                                      'polygon': polygon.tolist(), 'contour_roundtrip_iou': iou,
                                      'serialized_label_roundtrip_iou': serialized_iou})
                    task_counts[kind] += 1
            if not task_counts['BF'] or not task_counts['JT']:
                reason = 'missing_phase_or_component_instances'
        except ValueError as exc:
            excluded.append({'id': identity, 'raw_sha256': digest,
                             'reason': 'annotation_conversion_failed', 'detail': str(exc)})
            continue
        if reason:
            excluded.append({'id': identity, 'raw_sha256': digest, 'reason': reason})
            continue
        info = bf['image']
        split = 'val' if digest in val_hashes else 'train'
        if selection is not None and (selection[identity]['raw_sha256'] != digest
                                      or selection[identity]['split'] != split):
            raise ValueError(f'Frozen sample identity/split changed: {identity}')
        try:
            image, denoised_path, denoised_metadata = denoised_image_source(
                bf, jt, args.size, getattr(args, 'allow_jt_fallback', False))
        except (FileNotFoundError, ValueError) as exc:
            excluded.append({'id': identity, 'raw_sha256': digest,
                             'reason': 'denoised_source_unavailable', 'detail': str(exc)})
            continue
        denoised_metadata['source_path'] = str(denoised_path.relative_to(source_root))
        image_path = Path('images') / split / (identity + '.png')
        label_path = Path('labels') / split / (identity + '.txt')
        shutil.copyfile(denoised_path, output / image_path)
        if file_sha(output / image_path) != denoised_metadata['source_sha256']:
            raise RuntimeError(f'Copied target differs from Fontify source: {identity}')
        lines = []
        for item in instances:
            lines.append(yolo_label_line(item['class_id'], item['polygon'], args.size))
            class_counts[split][item['class_id']] += 1
            all_ious.append(item['contour_roundtrip_iou'])
        (output / label_path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
        provenance = {}
        for kind, record in (('BF', bf), ('JT', jt)):
            provenance[kind] = {'annotation_path': str(record['annotation_path'].relative_to(source_root)),
                                'image_file_name': record['image']['file_name'], 'image_id': record['image']['id']}
        included.append({'id': identity, 'author': author, 'raw_sha256': digest, 'split': split,
                         'image': str(image_path), 'label': str(label_path),
                         'original_size': [info['width'], info['height']], 'sources': provenance,
                         'instances': instances, 'denoised_source': denoised_metadata})
        if preview_counts[(author, split)] < 1:
            draw_preview(image, instances, preview_dir / (identity + '_' + split + '.jpg'), author + '/' + split)
            preview_counts[(author, split)] += 1
        if len(included) % 100 == 0:
            print(f'Exported {len(included)} directly annotated joint images', flush=True)
    for ann_path in ann_files:
        dest = output / 'source_annotations' / ann_path.parent.parent.name / ann_path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ann_path, dest)
    yaml_lines = ['# Portable relative paths; train.py resolves this directory.', 'path: .',
                  'train: images/train', 'val: images/val', 'names:']
    yaml_lines.extend(f"  {item['id']}: {item['name']}" for item in CLASSES)
    (output / 'data.yaml').write_text('\n'.join(yaml_lines) + '\n')
    (output / 'classes.json').write_text(json.dumps(CLASSES, ensure_ascii=False, indent=2) + '\n')
    manifest = {'schema': 'calli-phase-jt-yolo-denoised-v2', 'images': included, 'excluded': excluded}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    selection_out = {'images': [
        {key: row[key] for key in ('id', 'raw_sha256', 'split')} for row in included
    ]}
    (output / 'selection.json').write_text(json.dumps(selection_out, indent=2) + '\n')
    summary = {
        'created_at': datetime.now(timezone.utc).isoformat(), 'source_identity_count': len(index),
        'source_dual_identity_count': sum(set(v) == {'BF', 'JT'} for v in index.values()),
        'image_count': len(included), 'splits': dict(Counter(i['split'] for i in included)),
        'class_count': len(CLASSES), 'class_instances': {
            split: {c['name']: class_counts[split][c['id']] for c in CLASSES} for split in ('train', 'val')},
        'class_instances_total': {split: sum(c.values()) for split, c in class_counts.items()},
        'excluded_reasons': dict(Counter(i['reason'] for i in excluded)),
        'minimum_contour_roundtrip_iou': min(all_ious) if all_ious else None,
        'minimum_serialized_label_roundtrip_iou': min(
            (item['serialized_label_roundtrip_iou'] for row in included for item in row['instances']), default=None),
        'export_coordinate_convention': '(integer_pixel_coordinate + 0.5) / image_size; eight decimals; verified using float32 -> int32 rasterization',
        'geometry': {'method': 'centered_square_padding', 'image_size': args.size,
                     'mask_resize': 'nearest', 'image_resize': 'none: copy existing Fontify target'},
        'image_source': {'directory': 'images_text_denoised', 'preference': 'BF',
                         'allow_jt_fallback': getattr(args, 'allow_jt_fallback', False),
                         'selected_tasks': dict(Counter(row['denoised_source']['selected_task'] for row in included)),
                         'all_outputs_byte_identical_to_source': True},
        'frozen_selection': {
            'enabled': selection is not None,
            'source_sha256': file_sha(selection_path) if selection_path is not None else None,
            'missing_ids': sorted(set(selection or {}) - {row['id'] for row in included}),
        },
        'annotation_policy': {'original_annotation_is_instance': True, 'phase_merging': False,
                              'within_image_deduplication': False, 'bf_classes': 50, 'jt_classes': 20},
        'split_policy': 'Original ep66 validation raw-image SHA256 held out; all other accepted identities train. No test split.',
        'source_annotations': {str(p.relative_to(source_root)): file_sha(p) for p in ann_files},
        'source_val_records_sha256': file_sha(args.val_records),
        'notes': ['Images are unmodified copies of existing Fontify images_text_denoised files; no brightness adjustment, inversion, thresholding, padding or resize is applied to those PNGs.',
                  'Native annotation masks retain the confirmed centered-square-pad and nearest-resize geometry.',
                  'Each original non-text BF phase/JT annotation is one instance; no phase merging or stage completeness requirement.',
                  'Only same-original BF+JT paired images are used, avoiding incomplete task annotations becoming background.',
                  'Conflicting duplicate annotation versions and invalid conversions exclude the entire source identity.',
                  'YOLO polygon export uses retraced zero-area bridges for disconnected contours; rasterization may add one-pixel lines. Exact annotation RLE remains in manifest.',
                  'Classes absent from training cannot be learned; see class_instances before interpreting results.'],
    }
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ['image_count', 'splits', 'class_instances_total', 'excluded_reasons', 'minimum_contour_roundtrip_iou']}, ensure_ascii=False, indent=2))
    return summary


def parser():
    base = Path(__file__).resolve().parent
    project = base.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=project / 'fontdata_example/font/train/new')
    p.add_argument('--output', type=Path, default=base / 'dataset')
    p.add_argument('--preview-dir', type=Path, default=base / 'previews')
    p.add_argument('--val-records', type=Path, default=project / 'outputs/RunB_Fix_Pixel_ep66/协议与配置/validation_records.json')
    p.add_argument('--selection-manifest', type=Path, help='Optional prior manifest/selection.json that freezes image IDs and splits')
    p.add_argument('--allow-jt-fallback', action='store_true', help='Use same-original JT denoised image only if the BF target is missing')
    p.add_argument('--size', type=int, default=448)
    p.add_argument('--min-contour-iou', type=float, default=0.98)
    return p


if __name__ == '__main__':
    build(parser().parse_args())
