"""Train the isolated Run B pixel fusion, then evaluate/export against Fix."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def now():
    return datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    original = args.original_root.resolve()
    result = ROOT / 'experiment'
    result.mkdir(exist_ok=False)
    output = ROOT / 'models/runb_fix_pixel'
    original_run = original / 'models/finetune_dual_view_runb_fix_20261015'
    hparams = json.loads((original_run / 'hparams.json').read_text())
    original_checkpoint = original_run / 'checkpoint-best.pth'
    reference_sha = '57ac8276e3468ad92ce2aff033736128b33862a4f922618c51910717166ae852'
    assert sha256(original_checkpoint) == reference_sha
    assert sha256(hparams['finetune']) == '1333aa39dbd96c5812e08a3b970576f39384a995345506e66c600aa5ce680243'
    smoke = json.loads((ROOT / 'pixel_smoke.json').read_text())
    assert smoke['status'] == 'passed'
    state = {'status': 'prepared', 'started': now(), 'original_checkpoint_sha256': reference_sha,
             'original_hparams': hparams, 'smoke_status': 'passed', 'events': [],
             'design': 'same warm initialization, 51 epochs, BF11; local-view polygon/background pixel union; all other Fix settings retained'}

    def update(status, **kwargs):
        state.update(status=status, updated=now(), **kwargs)
        state['events'].append({'time': now(), 'status': status})
        temporary = result / 'status.tmp'
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        temporary.replace(result / 'status.json')
        print(json.dumps({'time': now(), 'status': status, **kwargs}, ensure_ascii=False), flush=True)

    env = os.environ.copy()
    env.update(PYTHON=sys.executable, NPROC='2', BATCH_SIZE='2', MASTER_PORT='29743',
               CUDA_VISIBLE_DEVICES='0,1', OMP_NUM_THREADS='1', DUAL_PIXEL_MASK='1',
               DATA_PATH=hparams['data_path'], TRAIN_MANIFEST=hparams['json_path'][0],
               TARGET_CACHE=hparams['dual_target_cache'], FIXED_PAIRS=hparams['fixed_pair_path'],
               PARENT_CHECKPOINT=hparams['dual_parent_checkpoint'],
               DUAL_ROOT=str(Path(hparams['finetune']).parent), WARM_CHECKPOINT=hparams['finetune'],
               FIXED_SAMPLES=hparams['dual_sample_manifest'],
               TRAIN_CALIBRATION=hparams['dual_calibration_json'],
               EVAL_CALIBRATION=hparams['dual_eval_calibration_json'],
               S_BASELINE=hparams['s_baseline_path'], BLIND_MANIFEST=hparams['dual_blind_manifest'],
               OUTPUT_DIR=str(output), PYTHONUNBUFFERED='1')

    def run(command, logfile):
        with (result / logfile).open('x') as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)

    try:
        update('training', training_output=str(output))
        run(['bash', 'finetune_dual_view_runb.sh'], 'train_console.log')
        logs = [json.loads(line) for line in (output / 'log.txt').read_text().splitlines()]
        assert len(logs) == 51 and [row['epoch'] for row in logs] == list(range(51))
        actual = json.loads((output / 'hparams.json').read_text())
        assert actual['dual_pixel_mask'] is True
        assert actual['batch_size'] * actual['accum_iter'] * actual['world_size'] == 64
        for key in ('epochs', 'seed', 'lr', 'warmup_epochs', 'style_weight', 'edge_weight_final',
                    'jieti_w', 'dual_head_weight', 'dual_lambda_full', 'num_mask_annotations_bf'):
            assert actual[key] == hparams[key], (key, actual[key], hparams[key])
        update('evaluating', completed_epochs=51)
        checkpoints = {'fix': original_checkpoint, 'pixel_best': output / 'checkpoint-best.pth',
                       'pixel_ep50': output / 'checkpoint-50.pth'}
        for name, checkpoint in checkpoints.items():
            command = [sys.executable, 'tools/eval_s_baseline.py', '--checkpoint', str(checkpoint),
                       '--data_path', hparams['data_path'], '--val_json_path', *hparams['val_json_path'],
                       '--fixed_pair_path', hparams['fixed_pair_path'],
                       '--calibration_json', hparams['dual_eval_calibration_json'],
                       '--dual_target_cache', hparams['dual_target_cache'],
                       '--j_definition', 'instance-lower-full-v1',
                       '--s_baseline_path', hparams['s_baseline_path'],
                       '--output_json', str(result / f'{name}_eval.json'), '--seed', '0']
            run(command, f'{name}_eval.log')
        from tools.dual_report import paired_comparison
        denominator = json.loads(Path(hparams['s_baseline_path']).read_text())
        evaluations = {name: json.loads((result / f'{name}_eval.json').read_text()) for name in checkpoints}
        summary = []
        for name, evaluation in evaluations.items():
            row = {'model': name, 'epoch': evaluation['source_epoch'],
                   **{key: evaluation[key] for key in ('L1_JT', 'L1_BF', 'J', 'S')}}
            for key in ('L1_JT', 'L1_BF', 'J', 'S'):
                row[f'{key}_reduction_vs_fix_percent'] = 100 * (1 - row[key] / evaluations['fix'][key])
            summary.append(row)
            if name != 'fix':
                comparison = paired_comparison(evaluation, evaluations['fix'], denominator)
                (result / f'{name}_paired.json').write_text(json.dumps(comparison, indent=2))
        with (result / 'comparison.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        (result / 'comparison.json').write_text(json.dumps(summary, indent=2))
        update('exporting', metrics=summary)
        export_visuals(evaluations, checkpoints, hparams, result)
        assert sha256(original_checkpoint) == reference_sha
        source_files = ['models_train.py', 'main_train.py', 'engine_train.py', 'data/dual_dataset.py',
                        'util/pixel_mask.py', 'finetune_dual_view_runb.sh', __file__]
        state['source_sha256'] = {name: sha256(ROOT / name) for name in source_files}
        update('complete', ended=now(), metrics=summary)
    except Exception as error:
        update('failed', error=repr(error))
        raise


def export_visuals(evaluations, checkpoints, hparams, destination):
    """Export all 240 fixed pairs, retaining float predictions for verification."""
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from tools.eval_s_baseline import build_dataset, load_model
    from tools.export_dual_blind import tensor_image

    args = argparse.Namespace(data_path=hparams['data_path'], val_json_path=hparams['val_json_path'],
                              fixed_pair_path=hparams['fixed_pair_path'],
                              dual_target_cache=hparams['dual_target_cache'],
                              j_definition='instance-lower-full-v1')
    dataset = build_dataset(args)
    folders = ('fix', 'pixel_best', 'pixel_ep50', 'gt', 'reference', 'comparison', 'float_predictions')
    for folder in folders:
        (destination / folder).mkdir()
    for name, checkpoint in checkpoints.items():
        torch.manual_seed(0)
        model = load_model(str(checkpoint)).cuda().eval()
        with torch.inference_mode():
            for index in range(len(dataset)):
                sample = dataset[index]
                prediction = model.predict_full(sample['image'][None].cuda(), sample['target'][None].cuda())
                lower = prediction[0, :, 448:].float().cpu()
                tensor_image(lower).save(destination / name / f'{index:03d}.png')
                np.save(destination / 'float_predictions' / f'{name}_{index:03d}.npy', lower.numpy())
                if name == 'fix':
                    tensor_image(sample['target'][:, 448:]).save(destination / 'gt' / f'{index:03d}.png')
                    tensor_image(sample['target'][:, :448]).save(destination / 'reference' / f'{index:03d}.png')
        del model
        torch.cuda.empty_cache()
    overview = []
    seen = set()
    for index in range(len(dataset)):
        canvas = Image.new('RGB', (448 * 5, 476), 'white')
        draw = ImageDraw.Draw(canvas)
        for column, name in enumerate(('reference', 'fix', 'pixel_best', 'pixel_ep50', 'gt')):
            with Image.open(destination / name / f'{index:03d}.png') as picture:
                canvas.paste(picture, (column * 448, 28))
            draw.text((column * 448 + 8, 6), name, fill='black')
        canvas.save(destination / 'comparison' / f'{index:03d}.png')
        row = evaluations['fix']['per_item'][index]
        identity = (row['kind'], Path(row['target_path']).parts[-3])
        if identity not in seen:
            seen.add(identity)
            overview.append(canvas.resize((1120, 238)))
    sheet = Image.new('RGB', (1120, 238 * len(overview)), 'white')
    for index, frame in enumerate(overview):
        sheet.paste(frame, (0, index * 238))
    sheet.save(destination / 'overview.png')
    html = ['<!doctype html><meta charset="utf-8"><title>Run B Fix / Pixel fusion</title>',
            '<h1>Run B Fix / Pixel fusion</h1><p><a href="comparison.csv">Metrics CSV</a></p>']
    for index in range(len(dataset)):
        html.append(f'<p>{index:03d}</p><a href="comparison/{index:03d}.png"><img loading="lazy" width="1120" src="comparison/{index:03d}.png"></a>')
    (destination / 'index.html').write_text('\n'.join(html))


if __name__ == '__main__':
    main()
