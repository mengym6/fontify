"""Freeze validation inputs, evaluate the paper model/baseline, and export results."""

import argparse
import csv
import hashlib
import html
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SETS = ('historical_240', 'validation_268')
METRICS = ('L1_JT', 'L1_BF', 'J', 'S', 'MAE', 'PSNR', 'SSIM')
FINAL_SHA = '6973168c7c396a9d153d2dc7368ead203952f38839ec43e320166efa65b55340'
BASELINE_SHA = 'eb5f64a3423f51baab3936fbe3fc2dc17fe1462f80de003220bd7148e9a1ecbc'


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_reference(query, rows, preferred=None):
    pool = [r for r in rows if r['calligrapher'] == query['calligrapher']
            and r['pool_type'] == query['pool_type'] and r['character'] != query['character']]
    pool.sort(key=lambda r: r['target_path'])
    if not pool:
        raise ValueError('No same-writer/type different-character reference: ' + query['target_path'])
    if preferred:
        for row in pool:
            if row['target_path'] == preferred['target_path']:
                return row
    seed = int(hashlib.sha256((query['target_path'] + ':reference-v1').encode()).hexdigest()[:16], 16)
    return pool[seed % len(pool)]


def settings():
    return json.loads((ROOT / 'models/runb_fix_pixel_continue20/hparams.json').read_text())


def prepare(output, final_checkpoint, baseline_checkpoint):
    import numpy as np
    import torch
    from tools.build_dual_manifest import normalize_record
    from tools.eval_s_baseline import build_dataset
    from tools.export_dual_blind import tensor_image
    from util.dual_targets import record_source_paths

    h = settings()
    assert sha(final_checkpoint) == FINAL_SHA
    assert sha(baseline_checkpoint) == BASELINE_SHA
    mapped = argparse.Namespace(data_path=h['data_path'], val_json_path=h['val_json_path'],
                               fixed_pair_path=h['fixed_pair_path'],
                               dual_target_cache=h['dual_target_cache'],
                               j_definition='instance-lower-full-v1')
    dataset = build_dataset(mapped)
    raw = sum([json.loads(Path(p).read_text()) for p in h['val_json_path']], [])
    rows = [normalize_record(r) for r in raw]
    assert len(rows) == 268 and len({r['target_path'] for r in rows}) == 268
    verified = set()
    for row in rows:
        for path in record_source_paths(row).values():
            if path in verified:
                continue
            expected = dataset._cache_metadata['source_sha256'].get(path)
            if not expected or sha(Path(h['data_path']) / path) != expected:
                raise ValueError('Unsealed or changed validation input: ' + path)
            verified.add(path)
        if row['has_jt'] and row['target_path'] not in dataset._cache_index:
            raise ValueError('JT cache does not cover the complete validation list')
    lookup = {r['target_path']: r for r in rows}
    protocol = dict(schema='callify-final-frozen-validation-v1', created=stamp(),
                    final_model='Run B Fix (final)', final_epoch=66,
                    final_checkpoint=str(final_checkpoint), final_sha256=FINAL_SHA,
                    baseline_checkpoint=str(baseline_checkpoint), baseline_sha256=BASELINE_SHA,
                    baseline_epoch=50, seed=0, input_size=[896, 448], query_size=[448, 448],
                    preprocessing='approved pad-square 448 geometry; identity validation crop',
                    inference='lower query fully hidden; actual model target has zero lower pixels in normalized space',
                    normalization=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    backend=dict(dtype='float32', matmul_tf32=False, cudnn_tf32=True, cudnn_benchmark=False),
                    reconstruction='ImageNet-normalized lower-query L1 / (valid RGB pixel count + 0.01)',
                    image_metrics='float RGB [0,1], clamp prediction before MAE/PSNR/SSIM; before PNG quantization',
                    structure='instance-lower-full-v1; shared original evaluation coefficients; J on JT only',
                    future_comparison_set='validation_268',
                    historical_set='unchanged original 240 pair orientation, including historical self references',
                    full_validation='each of 268 validation records queried once; same writer/type, different-character reference',
                    bootstrap='10000 paired entry resamples; S stratified by JT/BF, seed0; no multiple-training-seed claim',
                    actual_training=dict(base_epochs=51, added_epochs=20, total_epochs=71,
                                         selected_epoch=66, local_mask='BF polygon pixels union background >=230',
                                         continuation_peak_lr=0.0001, continuation_warmup=0,
                                         method_mainline='shared dual-view generator, instance layout head and structure supervision'),
                    validation_is_model_selection_set=True, source_files_verified=len(verified))
    provenance = output / '05_协议与复现'
    provenance.mkdir(parents=True)
    for name, path in [('evaluation_coefficients.json', h['dual_eval_calibration_json']),
                       ('historical_S_denominator.json', h['s_baseline_path']),
                       ('historical_fixed_pairs.json', h['fixed_pair_path'])]:
        shutil.copy2(path, provenance / name)
    (provenance / 'validation_json').mkdir()
    for p in h['val_json_path']:
        shutil.copy2(p, provenance / 'validation_json' / Path(p).name)
    write_json(provenance / 'validation_records.json', raw)
    write_json(provenance / 'final_hparams.json', h)
    shutil.copy2(__file__, provenance / 'build_callify_final.py')
    source_snapshot = provenance / 'source_code'
    source_snapshot.mkdir()
    for directory in ('data', 'util'):
        shutil.copytree(ROOT / directory, source_snapshot / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('models_train.py', 'models_eval.py', 'main_train.py', 'engine_train.py'):
        shutil.copy2(ROOT / name, source_snapshot / name)
    write_json(provenance / 'runtime.json', dict(torch=torch.__version__, numpy=np.__version__,
                                               cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version()))
    manifests = {}

    def save_case(set_name, index, reference, query, sample):
        folder = output / '03_固定验证集' / set_name
        for name in ('packs', 'gt', 'reference', 'source', 'visible_target'):
            (folder / name).mkdir(parents=True, exist_ok=True)
        case_id = f'{index:04d}'
        target = sample['target'].float()
        visible = target.clone()
        visible[:, 448:] = 0
        mask = sample['mask'].bool().flatten()
        pack = folder / 'packs' / f'{case_id}.npz'
        np.savez_compressed(pack, source=sample['image'].numpy().astype('float32'),
                            target_visible=visible.numpy(), query_gt=target[:, 448:].numpy(),
                            mask=mask.numpy(), voro=sample['voro'].numpy().astype('uint8'),
                            valid_parts=sample['valid_parts'].numpy(),
                            has_jt=np.array(bool(query['has_jt'])))
        tensor_image(target[:, 448:]).save(folder / 'gt' / f'{case_id}.png')
        tensor_image(target[:, :448]).save(folder / 'reference' / f'{case_id}.png')
        tensor_image(sample['image']).save(folder / 'source' / f'{case_id}.png')
        tensor_image(visible).save(folder / 'visible_target' / f'{case_id}.png')
        return dict(case_id=case_id, writer=query['calligrapher'], kind='JT' if query['has_jt'] else 'BF',
                    character=query['character'], target_path=query['target_path'],
                    reference_path=reference['target_path'], query_content_path=query['image_path'],
                    reference_content_path=reference['image_path'], reference_character=reference['character'],
                    self_reference=query['target_path'] == reference['target_path'],
                    pack=str(pack.relative_to(output)), pack_sha256=sha(pack),
                    source_tensor_sha256=hashlib.sha256(sample['image'].numpy().tobytes()).hexdigest(),
                    visible_target_tensor_sha256=hashlib.sha256(visible.numpy().tobytes()).hexdigest(),
                    query_gt_tensor_sha256=hashlib.sha256(target[:, 448:].numpy().tobytes()).hexdigest())

    legacy = []
    for index in range(len(dataset)):
        sample = dataset[index]
        reference = dataset.pairs[index]
        query = dataset._fixed_pairs[reference['target_path']]
        legacy.append(save_case(SETS[0], index, reference, query, sample))
    manifests[SETS[0]] = legacy
    full = []
    for index, query in enumerate(rows):
        reference = select_reference(query, rows, dataset._fixed_pairs.get(query['target_path']))
        crop = (0, 0, 448, 448)
        upper = dataset._transform_row(reference, crop, False)
        lower = dataset._transform_row(query, crop, True)
        mask = torch.zeros(56, 28, dtype=torch.bool)
        mask[28:] = True
        sample = dict(image=torch.cat((upper[0], lower[0]), 1),
                      target=torch.cat((upper[1], lower[1]), 1), mask=mask,
                      **{k: torch.from_numpy(v) for k, v in lower[3].items()})
        full.append(save_case(SETS[1], index, reference, query, sample))
    assert len(full) == 268 and not any(c['self_reference'] for c in full)
    manifests[SETS[1]] = full
    for set_name, cases in manifests.items():
        write_json(output / '03_固定验证集' / set_name / 'cases.json', cases)
        write_csv(output / '03_固定验证集' / set_name / 'cases.csv', cases)
        protocol[set_name] = dict(count=len(cases), cohorts=dict(Counter(c['kind'] for c in cases)),
                                  unique_queries=len({c['target_path'] for c in cases}),
                                  self_references=sum(c['self_reference'] for c in cases),
                                  cases_sha256=sha(output / '03_固定验证集' / set_name / 'cases.json'))
    write_json(provenance / 'protocol.json', protocol)
    training = output / '04_训练记录'
    training.mkdir()
    for label, path in [('final_first51', ROOT / 'models/runb_fix_pixel/log.txt'),
                        ('final_continue20', ROOT / 'models/runb_fix_pixel_continue20/log.txt'),
                        ('baseline', Path(baseline_checkpoint).parent / 'log.txt')]:
        shutil.copy2(path, training / f'{label}.jsonl')
    return protocol


def infer(output, model_name, checkpoint, device):
    import numpy as np
    import torch
    from skimage.metrics import structural_similarity
    from tools.eval_s_baseline import load_model, make_jieti_loss
    from tools.export_dual_blind import tensor_image

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    model = load_model(str(checkpoint)).to(device).eval()
    coef = json.loads((output / '05_协议与复现/evaluation_coefficients.json').read_text())['coef']
    jmod = make_jieti_loss('instance-lower-full-v1', coef).to(device)
    mean = torch.tensor((0.485, 0.456, 0.406), device=device)[None, :, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device=device)[None, :, None, None]
    for set_name in SETS:
        cases = json.loads((output / '03_固定验证集' / set_name / 'cases.json').read_text())
        destination = output / '02_推理图片' / set_name / model_name
        destination.mkdir(parents=True)
        rows = []
        with torch.inference_mode():
            for index, case in enumerate(cases):
                pack_path = output / case['pack']
                assert sha(pack_path) == case['pack_sha256']
                with np.load(pack_path) as p:
                    source = torch.from_numpy(p['source'].copy())[None].to(device)
                    visible = torch.from_numpy(p['target_visible'].copy())[None].to(device)
                    gt = torch.from_numpy(p['query_gt'].copy())[None].to(device)
                    mask = torch.from_numpy(p['mask'].copy())[None].to(device)
                    voro = torch.from_numpy(p['voro'].astype('int64'))[None].to(device)
                    parts = torch.from_numpy(p['valid_parts'].copy())[None].to(device)
                prediction = model.predict_full(source, visible, mask)
                if index == 0:
                    perturbed = visible.clone()
                    perturbed[:, :, 448:] = gt
                    check = model.predict_full(source, perturbed, mask)
                    torch.testing.assert_close(prediction, check, rtol=1e-6, atol=1e-7)
                lower = prediction[:, :, 448:]
                l1 = (lower - gt).abs().sum() / (gt.numel() + 0.01)
                rgb = (lower * std + mean).clamp(0, 1)
                gt_rgb = (gt * std + mean).clamp(0, 1)
                mse = float((rgb - gt_rgb).square().mean())
                a = rgb[0].permute(1, 2, 0).float().cpu().numpy()
                b = gt_rgb[0].permute(1, 2, 0).float().cpu().numpy()
                row = {**case, 'L1': float(l1), 'J': None, 'centroid': None,
                       'logsigma': None, 'shape': None, 'MAE': float((rgb - gt_rgb).abs().mean()),
                       'PSNR': min(120.0, -10 * math.log10(max(mse, 1e-12))),
                       'SSIM': float(structural_similarity(b, a, data_range=1.0, channel_axis=2)),
                       'model': model_name, 'source_checkpoint_sha256': model._checkpoint_sha256,
                       'source_epoch': model._checkpoint_epoch}
                if case['kind'] == 'JT':
                    target = torch.cat((visible[:, :, :448], gt), 2)
                    j, values, _ = jmod(prediction, target, voro, parts,
                                       torch.ones(1, dtype=torch.bool, device=device))
                    row.update(J=float(j), **{k: float(values[k]) for k in ('centroid', 'logsigma', 'shape')})
                assert all(math.isfinite(row[k]) for k in ('L1', 'MAE', 'PSNR', 'SSIM'))
                tensor_image(lower[0]).save(destination / f"{case['case_id']}.png")
                rows.append(row)
                if index % 50 == 0:
                    print(json.dumps(dict(model=model_name, dataset=set_name, completed=index + 1)), flush=True)
        folder = output / '01_数值结果' / set_name
        write_json(folder / f'{model_name}_per_case.json', rows)
        write_csv(folder / f'{model_name}_per_case.csv', rows)
    print(json.dumps({'model': model_name, 'status': 'complete'}), flush=True)


def aggregate(rows):
    jt = [r for r in rows if r['kind'] == 'JT']
    bf = [r for r in rows if r['kind'] == 'BF']
    avg = lambda records, key: sum(r[key] for r in records) / len(records)
    return dict(L1_JT=avg(jt, 'L1'), L1_BF=avg(bf, 'L1'), J=avg(jt, 'J'),
                centroid=avg(jt, 'centroid'), logsigma=avg(jt, 'logsigma'), shape=avg(jt, 'shape'),
                MAE=avg(rows, 'MAE'), PSNR=avg(rows, 'PSNR'), SSIM=avg(rows, 'SSIM'),
                n=len(rows), n_jt=len(jt), n_bf=len(bf))


def paired_statistics(final, baseline, denominator):
    import numpy as np

    assert [r['pack_sha256'] for r in final] == [r['pack_sha256'] for r in baseline]
    rng = np.random.default_rng(0)
    result = {}
    jt = [i for i, r in enumerate(final) if r['kind'] == 'JT']
    bf = [i for i, r in enumerate(final) if r['kind'] == 'BF']
    for name, key, indices in [('L1_JT', 'L1', jt), ('L1_BF', 'L1', bf), ('J', 'J', jt),
                                ('centroid', 'centroid', jt), ('logsigma', 'logsigma', jt),
                                ('shape', 'shape', jt), ('MAE', 'MAE', range(len(final))),
                                ('PSNR', 'PSNR', range(len(final))), ('SSIM', 'SSIM', range(len(final)))]:
        values = np.array([final[i][key] - baseline[i][key] for i in indices])
        means = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
        result[name] = dict(delta=float(values.mean()), ci95=np.quantile(means, [0.025, 0.975]).tolist(),
                            n=len(values), iterations=10000, seed=0)
    a = np.array([(final[i]['L1'] - baseline[i]['L1']) / denominator['L1_JT']
                  + (final[i]['J'] - baseline[i]['J']) / denominator['J'] for i in jt])
    b = np.array([(final[i]['L1'] - baseline[i]['L1']) / denominator['L1_BF'] for i in bf])
    estimates = a[rng.integers(0, len(a), (10000, len(a)))].mean(1)
    estimates += b[rng.integers(0, len(b), (10000, len(b)))].mean(1)
    result['S'] = dict(delta=float(a.mean() + b.mean()), ci95=np.quantile(estimates, [0.025, 0.975]).tolist(),
                       n=len(final), iterations=10000, seed=0, stratified_by='JT/BF')
    return result


def summarize(output):
    from PIL import Image, ImageDraw

    all_summary, ci_tables = {}, {}
    for set_name in SETS:
        folder = output / '01_数值结果' / set_name
        data = {name: json.loads((folder / f'{name}_per_case.json').read_text()) for name in ('baseline', 'final')}
        metrics = {name: aggregate(rows) for name, rows in data.items()}
        denominator = metrics['baseline']
        for values in metrics.values():
            values['S'] = sum(values[k] / denominator[k] for k in ('L1_JT', 'L1_BF', 'J'))
        ci = paired_statistics(data['final'], data['baseline'], denominator)
        ci_tables[set_name] = ci
        all_summary[set_name] = metrics
        write_json(folder / 'summary.json', metrics)
        write_json(folder / 'paired_95CI.json', ci)
        write_json(folder / 'S_denominator.json', {k: denominator[k] for k in ('L1_JT', 'L1_BF', 'J')})
        write_csv(folder / 'summary.csv', [dict(model=k, **v) for k, v in metrics.items()])
        write_csv(folder / 'paired_95CI.csv', [dict(metric=k, delta=v['delta'], ci_lower=v['ci95'][0], ci_upper=v['ci95'][1],
                                                  n=v['n'], iterations=v['iterations']) for k, v in ci.items()])
        grouped = []
        for name, rows in data.items():
            groups = defaultdict(list)
            for row in rows:
                groups[(row['writer'], row['kind'])].append(row)
            for (writer, kind), samples in sorted(groups.items()):
                grouped.append(dict(model=name, writer=writer, kind=kind, n=len(samples),
                                    L1=sum(r['L1'] for r in samples) / len(samples),
                                    J=sum(r['J'] for r in samples) / len(samples) if kind == 'JT' else None,
                                    MAE=sum(r['MAE'] for r in samples) / len(samples),
                                    PSNR=sum(r['PSNR'] for r in samples) / len(samples),
                                    SSIM=sum(r['SSIM'] for r in samples) / len(samples)))
        write_csv(folder / 'by_writer_kind.csv', grouped)
        compare = output / '02_推理图片' / set_name / 'comparison'
        compare.mkdir()
        frozen = output / '03_固定验证集' / set_name
        pages = []
        overview, seen = [], set()
        for index, case in enumerate(data['final']):
            cid = case['case_id']
            canvas = Image.new('RGB', (448 * 4, 480), 'white')
            draw = ImageDraw.Draw(canvas)
            panels = [('Reference', frozen / 'reference' / f'{cid}.png'),
                      ('Baseline', output / '02_推理图片' / set_name / 'baseline' / f'{cid}.png'),
                      ('Run B Fix final', output / '02_推理图片' / set_name / 'final' / f'{cid}.png'),
                      ('GT', frozen / 'gt' / f'{cid}.png')]
            for column, (label, path) in enumerate(panels):
                with Image.open(path) as im:
                    canvas.paste(im, (448 * column, 32))
                draw.text((448 * column + 6, 8), label, fill='black')
            canvas.save(compare / f'{cid}.png')
            if (case['writer'], case['kind']) not in seen:
                seen.add((case['writer'], case['kind']))
                overview.append(canvas.resize((896, 240)))
            pages.append(f'<p>{cid} · {html.escape(case["writer"])} · {html.escape(case["character"])} · {case["kind"]}</p>'
                         f'<a href="comparison/{cid}.png"><img loading="lazy" width="1100" src="comparison/{cid}.png"></a>')
        sheet = Image.new('RGB', (896, 240 * len(overview)), 'white')
        for index, panel in enumerate(overview):
            sheet.paste(panel, (0, index * 240))
        sheet.save(output / '02_推理图片' / set_name / 'overview.png')
        (output / '02_推理图片' / set_name / 'index.html').write_text(
            '<!doctype html><meta charset="utf-8"><title>Baseline / Run B Fix final</title>'
            f'<h1>{set_name}</h1><p>固定顺序完整展示；总览取每书家每类型首例。</p>' + '\n'.join(pages))
    expected = json.loads((ROOT / 'experiment_continue20/continued_best_eval.json').read_text())
    baseline_expected = json.loads((ROOT / 'experiment/baseline_eval.json').read_text()) if (ROOT / 'experiment/baseline_eval.json').exists() else json.loads((output / '05_协议与复现/historical_S_denominator.json').read_text())
    for name, reference in [('final', expected), ('baseline', baseline_expected)]:
        for key in ('L1_JT', 'L1_BF', 'J'):
            if not math.isclose(all_summary[SETS[0]][name][key], reference[key], rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError(f'Historical reconciliation failed: {name}/{key}')
    write_json(output / '01_数值结果/all_summary.json', all_summary)
    make_figures(output, all_summary, ci_tables)
    make_readme_and_index(output, all_summary)


def make_figures(output, summaries, intervals):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    folder = output / '00_汇总与论文图表'
    folder.mkdir()
    colors = ['#637f9c', '#278e83']

    def save(fig, name):
        fig.tight_layout()
        for suffix in ('png', 'svg', 'pdf'):
            fig.savefig(folder / f'{name}.{suffix}', dpi=300, bbox_inches='tight')
        plt.close(fig)

    for set_name in SETS:
        values = summaries[set_name]
        keys = ('L1_JT', 'L1_BF', 'J', 'S')
        baseline = np.array([values['baseline'][k] for k in keys])
        final = np.array([values['final'][k] for k in keys]) / baseline
        x = np.arange(4)
        fig, ax = plt.subplots(figsize=(8.6, 4.7))
        ax.bar(x - 0.18, np.ones(4), 0.36, color=colors[0], label='Baseline vggfix ep50')
        ax.bar(x + 0.18, final, 0.36, color=colors[1], label='Run B Fix final ep66')
        for i, v in enumerate(final):
            ax.text(i + 0.18, v + 0.025, f'{(v-1)*100:+.2f}%', ha='center', fontsize=10)
        ax.set_xticks(x, ['L1 (JT)', 'L1 (BF)', 'J', 'S'])
        ax.set_ylim(0, max(1.25, final.max() + 0.17))
        ax.set_ylabel('Metric / baseline (lower is better)')
        ax.set_title(f'{set_name}: {values["final"]["n"]} fixed pairs')
        ax.legend(frameon=False, loc='lower left')
        save(fig, f'01_baseline_vs_runb_normalized_{set_name}')
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.3))
        for ax, key in zip(axes, keys):
            v = intervals[set_name][key]
            ax.errorbar(v['delta'], 0, xerr=[[v['delta']-v['ci95'][0]], [v['ci95'][1]-v['delta']]],
                        fmt='o', capsize=5, color=colors[1])
            ax.axvline(0, color='gray', linewidth=0.8)
            ax.set_title(key)
            ax.set_yticks([])
            ax.set_xlabel('Final - baseline')
        fig.suptitle(f'{set_name}: paired bootstrap 95% intervals')
        save(fig, f'02_paired_differences_95CI_{set_name}')
    logs = []
    for name in ('final_first51.jsonl', 'final_continue20.jsonl'):
        logs.extend(json.loads(line) for line in (output / '04_训练记录' / name).read_text().splitlines())
    assert [r['epoch'] for r in logs] == list(range(71))
    write_csv(output / '04_训练记录/final_71epochs.csv', logs)
    x = [r['epoch'] for r in logs]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for key, label in [('train_loss', 'Training loss'), ('test_loss', 'Validation loss')]:
        axes[0, 0].plot(x, [r[key] for r in logs], label=label)
    axes[0, 0].legend(frameon=False)
    axes[0, 0].set_title('Final model optimization')
    axes[0, 1].plot(x, [r['test_S'] for r in logs], color=colors[1])
    axes[0, 1].axhline(3, color=colors[0], ls='--', label='Baseline standalone S=3')
    axes[0, 1].legend(frameon=False)
    axes[0, 1].set_title('Training-time validation S')
    for key, label in [('test_L1_JT', 'JT'), ('test_L1_BF', 'BF')]:
        axes[1, 0].plot(x, [r[key] for r in logs], label=label)
    axes[1, 0].legend(frameon=False)
    axes[1, 0].set_title('Training-time validation L1')
    axes[1, 1].plot(x, [r['test_jieti_J'] for r in logs], color=colors[1])
    axes[1, 1].set_title('Training-time validation J')
    for ax in axes.flat:
        ax.axvline(51, color='gray', ls=':', lw=0.8)
        ax.axvline(66, color=colors[1], ls='--', lw=0.8)
        ax.set_xlabel('Epoch (zero based)')
        ax.grid(alpha=0.15)
    save(fig, '03_training_validation_curves')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, set_name in zip(axes, SETS):
        keys = ('centroid', 'logsigma', 'shape')
        for j, name in enumerate(('baseline', 'final')):
            ax.bar(np.arange(3) + (j - 0.5) * 0.35,
                   [summaries[set_name][name][k] for k in keys], 0.35,
                   color=colors[j], label=name)
        ax.set_xticks(np.arange(3), keys)
        ax.set_title(set_name)
        ax.set_ylabel('J component (lower is better)')
        ax.legend(frameon=False)
    save(fig, '04_structure_components')


def make_readme_and_index(output, summaries):
    readme = '''# Callify 投稿定版材料

定版模型：Run B Fix（投稿定版），锁定续训best epoch66；对照为VGG修复baseline epoch50。
方法主线沿用共享双视图、结体实例头与结构监督。实际像素遮盖及51+20轮配置完整保存在复现协议。

## 使用

- 00_汇总与论文图表：每图包含300dpi PNG、可编辑SVG与矢量PDF。
- 01_数值结果：两套口径的汇总、逐样本、分书家／类型和10000次配对bootstrap区间。
- 02_推理图片：baseline、定版模型、全量对照图和按固定顺序选取的总览。
- 03_固定验证集：案例清单、GT、reference、source、可见目标图与实际模型输入NPZ。
- 04_训练记录：完整71轮日志及baseline日志。训练内验证曲线和standalone主表分别标识。
- 05_协议与复现：权重身份／路径／SHA、原验证划分、评测系数、真实配置和执行源码。

后续公平比较统一使用validation_268/cases.json的顺序与packs/输入，不重新抽参考字或改变预处理。
NPZ的source、target_visible、mask是模型输入；query_gt、voro、valid_parts只用于评分，不能传入生成条件。
每个验证记录查询一次；reference来自同书家同类型的不同字符。全部268条保留，无按效果筛图。
historical_240保留原历史固定配对及方向，供旧表对账，含原有自身参考与重复query，不与268条混算。

L1_JT／L1_BF是模型归一化空间的query重建误差；J仅在JT上计算；S为三项除以同队列baseline后求和。
MAE／PSNR／SSIM在反归一化后[0,1]浮点RGB上计算，先clamp预测，不使用8bit PNG重新算数值。
PSNR按逐例dB平均；SSIM使用scikit-image标准默认窗口，data_range=1、channel_axis=2。
两套队列各自保存S分母，baseline均为3，分母不得跨队列复用。
置信区间按配对条目抽样，仅反映当前单seed模型的样本不确定性；验证集参与过训练模型选择，不称独立测试集。
权重二进制保留服务器；交付包保存准确路径、SHA256与strict加载源码。
'''
    (output / 'README_使用说明.md').write_text(readme)
    cards = []
    for set_name in SETS:
        values = summaries[set_name]
        rows = ''.join('<tr><td>' + name + '</td>' + ''.join(f'<td>{values[name][k]:.6f}</td>' for k in METRICS) + '</tr>'
                       for name in ('baseline', 'final'))
        cards.append(f'<h2>{set_name}</h2><p>{values["final"]["n"]}例 · JT {values["final"]["n_jt"]} / BF {values["final"]["n_bf"]}</p>'
                     '<table><tr><th>Model</th>' + ''.join(f'<th>{k}</th>' for k in METRICS) + '</tr>' + rows + '</table>'
                     f'<p><a href="01_数值结果/{set_name}/summary.csv">汇总CSV</a> · '
                     f'<a href="01_数值结果/{set_name}/paired_95CI.csv">配对95%区间</a> · '
                     f'<a href="02_推理图片/{set_name}/index.html">全量对比图</a> · '
                     f'<a href="03_固定验证集/{set_name}/cases.csv">固定案例清单</a></p>'
                     f'<a href="02_推理图片/{set_name}/overview.png"><img width="700" src="02_推理图片/{set_name}/overview.png"></a>')
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Callify Final</title>'
                                      '<style>body{font:16px sans-serif;margin:32px;max-width:1200px}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style>'
                                      '<h1>Run B Fix · 投稿定版</h1><p>定版ep66；baseline vggfix ep50。后续比较统一使用validation_268冻结输入。</p>'
                                      '<p><a href="README_使用说明.md">使用说明</a> · <a href="05_协议与复现/protocol.json">冻结协议</a></p>' + ''.join(cards))


def controller(options):
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    def state(phase, **extra):
        write_json(output / 'status.json', dict(status=phase, updated=stamp(), **extra))
        print(json.dumps(dict(status=phase, **extra), ensure_ascii=False), flush=True)

    try:
        state('freezing_inputs')
        protocol = prepare(output, options.final_checkpoint, options.baseline_checkpoint)
        state('evaluating', counts={s: protocol[s]['count'] for s in SETS})
        jobs = []
        env = dict(os.environ, OMP_NUM_THREADS='2', PYTHONUNBUFFERED='1')
        for name, checkpoint, device in [('final', options.final_checkpoint, 'cuda:0'),
                                          ('baseline', options.baseline_checkpoint, 'cuda:1')]:
            log = (output / f'{name}_inference.log').open('x')
            command = [sys.executable, __file__, '--phase', 'infer', '--output', str(output),
                       '--model-name', name, '--checkpoint', str(checkpoint), '--device', device]
            jobs.append((subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        for process, log in jobs:
            code = process.wait()
            log.close()
            if code:
                raise RuntimeError(f'Inference worker failed with exit code {code}')
        state('summarizing')
        summarize(output)
        entries = {}
        for p in sorted(output.rglob('*')):
            if p.is_file() and p.name != 'status.json':
                entries[str(p.relative_to(output))] = dict(bytes=p.stat().st_size, sha256=sha(p))
        write_json(output / '文件校验_SHA256.json', entries)
        state('complete', ended=stamp(), files=len(entries), payload_bytes=sum(x['bytes'] for x in entries.values()))
    except Exception as error:
        state('failed', error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('all', 'infer'), default='all')
    parser.add_argument('--model-name', choices=('final', 'baseline'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--final-checkpoint', type=Path)
    parser.add_argument('--baseline-checkpoint', type=Path)
    args = parser.parse_args()
    if args.phase == 'infer':
        infer(args.output, args.model_name, args.checkpoint, args.device)
    else:
        controller(args)


if __name__ == '__main__':
    main()
