"""Export actual Run B slot predictions, then render a small local gallery."""

import argparse
import hashlib
import html
import json
import sys
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def infer(args):
    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(args.repo.resolve()))
    from tools.build_dual_manifest import normalize_record
    from tools.eval_s_baseline import load_model
    from tools.export_dual_blind import tensor_image
    from util.calli_labels import JT_CLASSES
    from util.dual_loss import InstanceHeadLoss
    from util.dual_targets import build_instance_targets, load_cached_targets

    release = args.release
    protocol = json.loads((release / '05_协议与复现/protocol.json').read_text())
    checkpoint = Path(protocol['final_checkpoint'])
    actual_sha = digest(checkpoint)
    if actual_sha != protocol['final_sha256']:
        raise ValueError('Checkpoint does not match the frozen final model')
    hparams = json.loads((release / '05_协议与复现/final_hparams.json').read_text())
    records = json.loads((release / '05_协议与复现/validation_records.json').read_text())
    lookup = {row['target_path']: row for row in map(normalize_record, records)}
    cases = json.loads((release / '03_固定验证集/validation_268/cases.json').read_text())
    selected = {row['case_id']: row for row in cases}
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    model = load_model(str(checkpoint)).cuda().eval()
    matcher = InstanceHeadLoss().cuda()
    results = []
    with torch.inference_mode():
        for case_id in args.cases:
            case = selected[case_id]
            pack = release / case['pack']
            if digest(pack) != case['pack_sha256']:
                raise ValueError('Frozen input hash mismatch')
            with np.load(pack) as arrays:
                source = torch.from_numpy(arrays['source'].copy())[None].cuda()
                visible = torch.from_numpy(arrays['target_visible'].copy())[None].cuda()
            if torch.count_nonzero(visible[:, :, 448:]):
                raise ValueError('Query target must be hidden')
            prediction, _, head = model._dual_full_prediction(source, visible)
            raw = load_cached_targets(hparams['dual_target_cache'], lookup[case['target_path']])
            targets = build_instance_targets(**raw)
            target_tensors = {key: torch.from_numpy(value)[None].cuda()
                              for key, value in targets.items() if isinstance(value, np.ndarray)}
            _, match = matcher(head, target_tensors['instance_occupancy'],
                               target_tensors['instance_valid'], target_tensors['instance_classes'],
                               torch.tensor([True], device='cuda'))
            probability = head['occupancy_logits'].sigmoid()[0].float().cpu().numpy()
            exists = head['existence_logits'].sigmoid()[0].float().cpu().numpy()
            classes = head['class_logits'].softmax(-1)[0].float().cpu().numpy()
            if not all(np.isfinite(a).all() for a in (probability, exists, classes)):
                raise ValueError('Nonfinite model output')
            folder = args.output / case_id
            folder.mkdir()
            tensor_image(source[0, :, 448:]).save(folder / 'content.png')
            tensor_image(visible[0, :, :448]).save(folder / 'reference.png')
            picture = tensor_image(prediction[0, :, 448:])
            picture.save(folder / 'prediction.png')
            frozen_prediction = release / '02_推理图片/validation_268/final' / f'{case_id}.png'
            with Image.open(frozen_prediction) as old:
                pixel_difference = int(np.abs(np.asarray(picture).astype(int)
                                              - np.asarray(old).astype(int)).max())
            if pixel_difference:
                raise ValueError('Prediction differs from frozen final inference')
            with Image.open(release / '03_固定验证集/validation_268/gt' / f'{case_id}.png') as gt:
                gt.save(folder / 'gt.png')
            assignment = match['assignment'][0].cpu().numpy()
            matched = match['matched_slots'][0].cpu().numpy()
            np.savez_compressed(folder / 'predictions.npz', occupancy_probability=probability,
                                existence_probability=exists, class_probability=classes,
                                gt_occupancy=targets['instance_occupancy'],
                                gt_valid=targets['instance_valid'],
                                gt_classes=targets['instance_classes'],
                                display_assignment=assignment, matched_slots=matched)
            slots = []
            for slot in range(4):
                pred_class = int(classes[slot].argmax())
                row = dict(slot=slot + 1, existence=float(exists[slot]),
                           predicted_class=JT_CLASSES[pred_class],
                           class_probability=float(classes[slot, pred_class]), matched=bool(matched[slot]))
                if matched[slot]:
                    index = int(assignment[slot])
                    gt_mask = targets['instance_occupancy'][index].astype(bool)
                    binary = probability[slot] >= 0.5
                    row.update(gt_class=JT_CLASSES[int(targets['instance_classes'][index])],
                               iou_at_05=float((binary & gt_mask).sum() / max((binary | gt_mask).sum(), 1)))
                slots.append(row)
            results.append(dict(**case, slots=slots, png_max_difference_from_release=pixel_difference))
            print(json.dumps(dict(case=case_id, character=case['character'], slots=slots),
                             ensure_ascii=False), flush=True)
    manifest = dict(checkpoint=str(checkpoint), checkpoint_sha256=actual_sha, epoch=66,
                    source_release=str(release), cases=results,
                    selection='Fixed examples selected by character/structure before prediction; not ranked by accuracy',
                    heatmap='Raw sigmoid occupancy, fixed scale 0..1; no existence gating or per-image normalization',
                    gt_matching='Training loss minimum-cost matching, used only after inference for display',
                    source_files={name: digest(args.repo / name) for name in
                                  ('models_train.py', 'util/dual_view.py', 'util/dual_targets.py', 'util/dual_loss.py')})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


def render(args):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    font_path = '/System/Library/Fonts/STHeiti Medium.ttc'
    regular = ImageFont.truetype(font_path, 20)
    small = ImageFont.truetype(font_path, 17)
    title = ImageFont.truetype(font_path, 30)
    manifest = json.loads((args.output / 'manifest.json').read_text())
    anchors = np.array([[8, 9, 23], [52, 18, 99], [152, 45, 92],
                        [232, 107, 44], [253, 198, 66], [255, 255, 210]], dtype=float)

    def heatmap(values, size):
        values = np.clip(values, 0, 1)
        x = values * (len(anchors) - 1)
        left = np.minimum(x.astype(int), len(anchors) - 2)
        rgb = anchors[left] * (1 - (x - left)[..., None]) + anchors[left + 1] * (x - left)[..., None]
        return Image.fromarray(np.rint(rgb).astype('uint8')).resize((size, size), Image.Resampling.NEAREST)

    def text(draw, xy, value, font=regular, fill='#263040'):
        draw.text(xy, value, font=font, fill=fill)

    overview_cell = 160
    overview = Image.new('RGB', (8 * 176 + 24, len(manifest['cases']) * 234 + 100), '#f1f3f7')
    overview_draw = ImageDraw.Draw(overview)
    text(overview_draw, (20, 12), 'Run B ep66：实际部件预测（六个固定验证样例）', title)
    headers = ['内容字', '风格参考', '生成字', '目标真值', '槽位 1', '槽位 2', '槽位 3', '槽位 4']
    for column, label in enumerate(headers):
        text(overview_draw, (20 + 176 * column, 57), label)
    for case_index, case in enumerate(manifest['cases']):
        folder = args.output / case['case_id']
        with np.load(folder / 'predictions.npz') as arrays:
            probability = arrays['occupancy_probability']
            gt_masks = arrays['gt_occupancy']
            assignment = arrays['display_assignment']
            matched = arrays['matched_slots']
        cell, gap, margin = 256, 18, 24
        canvas = Image.new('RGB', (4 * (cell + gap) + 2 * margin - gap, 1120), '#f1f3f7')
        draw = ImageDraw.Draw(canvas)
        text(draw, (margin, 16), f"{case['character']}  |  {case['writer']}  |  case {case['case_id']}  |  Run B ep66", title)
        text(draw, (margin, 58), '上：字形对照   中：模型预测占用概率   下：训练口径的真值占用图（仅作对照）', small)
        image_names = ['content', 'reference', 'prediction', 'gt']
        for column, (name, label) in enumerate(zip(image_names, headers[:4])):
            x = margin + column * (cell + gap)
            text(draw, (x, 94), label)
            with Image.open(folder / f'{name}.png') as pic:
                canvas.paste(pic.resize((cell, cell), Image.Resampling.LANCZOS), (x, 126))
                overview.paste(pic.resize((overview_cell, overview_cell), Image.Resampling.LANCZOS),
                               (20 + 176 * column, 112 + case_index * 234))
        for column, slot in enumerate(case['slots']):
            x = margin + column * (cell + gap)
            text(draw, (x, 407), f"槽位 {column + 1}  存在概率 {slot['existence']:.3f}", small)
            text(draw, (x, 433), f"预测类别：{slot['predicted_class']} ({slot['class_probability']:.2f})", small)
            canvas.paste(heatmap(probability[column], cell), (x, 463))
            if matched[column]:
                label = f"匹配真值：{slot['gt_class']}  IoU={slot['iou_at_05']:.3f}"
                truth = gt_masks[int(assignment[column])]
            else:
                label = '无匹配真值（空槽位）'
                truth = np.zeros((28, 28), dtype=float)
            text(draw, (x, 744), label, small)
            canvas.paste(heatmap(truth, cell), (x, 778))
            ox, oy = 20 + 176 * (column + 4), 112 + case_index * 234
            overview.paste(heatmap(probability[column], overview_cell), (ox, oy))
            text(overview_draw, (ox, oy + 166), f"存在 {slot['existence']:.2f} / {slot['predicted_class']}", small)
        text(draw, (margin, 1058), '色阶统一：黑色=0，黄白=1；原生分辨率 28×28，按格放大，未逐图增强。', small)
        text(draw, (margin, 1084), '四槽位全部展示；预测类别为左右上下等空间角色；真值匹配不参与模型推理。', small)
        canvas.save(folder / 'panel.png')
        text(overview_draw, (20, 88 + case_index * 234),
             f"{case['character']} / {case['writer']} / {case['case_id']}", small)
    text(overview_draw, (20, overview.height - 28), '热力图：原始占用概率，统一 0–1 色阶；存在概率独立显示，包含全部四槽位。', small)
    overview.save(args.output / 'overview.png')
    cards = ''.join(f'<h2>{html.escape(c["character"])} · {c["writer"]} · {c["case_id"]}</h2>'
                    f'<a href="{c["case_id"]}/panel.png"><img src="{c["case_id"]}/panel.png"></a>'
                    for c in manifest['cases'])
    (args.output / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Run B 部件预测</title>'
        '<style>body{max-width:1180px;margin:32px auto;font-family:sans-serif;background:#f1f3f7}'
        'img{width:100%;height:auto}p{line-height:1.7}</style>'
        '<h1>Run B ep66 实际部件预测</h1><p>六个预先选定的验证样例。'
        '热力图是结体头实际输出的 sigmoid 占用概率，统一使用 0–1 色阶。'
        '每例展示全部四个槽位及存在概率；底行为训练匹配口径下的真值，仅用于对照。'
        '原始预测及真值保存在各例 predictions.npz 中。</p>' + cards)
    for path in args.output.rglob('*.png'):
        with Image.open(path) as pic:
            pic.load()
    print(json.dumps(dict(status='rendered', examples=len(manifest['cases']),
                          output=str(args.output.resolve())), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['infer', 'render'])
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--release', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=['0021', '0017', '0018', '0075', '0161', '0071'])
    args = parser.parse_args()
    infer(args) if args.mode == 'infer' else render(args)


if __name__ == '__main__':
    main()
