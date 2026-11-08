"""Resume the pixel-fusion model with optimizer/AMP state for 20 extra epochs."""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def continuation_rate(epoch, start_epoch=51, extra_epochs=20, peak=1e-4):
    progress = min(1.0, max(0.0, (epoch - start_epoch) / extra_epochs))
    return peak * 0.5 * (1 + math.cos(math.pi * progress))


def validate_checkpoint(checkpoint):
    required = {'model', 'optimizer', 'scaler', 'epoch', 'args'}
    if required.difference(checkpoint) or checkpoint['epoch'] != 50:
        raise ValueError('Continuation requires the complete pixel-fusion ep50 checkpoint')
    saved = checkpoint['args']
    saved = saved if isinstance(saved, dict) else vars(saved)
    if not saved.get('dual_pixel_mask') or not saved.get('dual_view'):
        raise ValueError('Checkpoint must be the trained dual-view pixel-fusion model')
    if not checkpoint['optimizer'].get('state'):
        raise ValueError('Optimizer state is missing')
    if saved['epochs'] != 51:
        raise ValueError('Expected the completed 51-epoch source run')
    return saved


def install_training_exports(main_train):
    """Save three actual training forwards at selected epochs, without inference."""
    import torch
    from PIL import Image, ImageDraw
    from tools.export_dual_blind import tensor_image
    from util import misc
    from util.pixel_mask import expand_mask

    original_epoch = main_train.train_one_epoch

    def capture_epoch(*arguments, **keywords):
        epoch = arguments[4] if len(arguments) > 4 else keywords['epoch']
        if epoch not in (51, 55, 60, 65, 70) or not misc.is_main_process():
            return original_epoch(*arguments, **keywords)
        model = arguments[0] if arguments else keywords['model']
        raw = model.module if hasattr(model, 'module') else model
        destination = Path(keywords['args'].output_dir) / 'training_samples' / f'epoch_{epoch:03d}'
        destination.mkdir(parents=True, exist_ok=True)
        count = 0
        calls = 0

        def save_forward(module, inputs, forwarded, output):
            nonlocal count, calls
            calls += 1
            if count >= 3:
                return
            image, target = inputs[:2]
            mask = expand_mask(forwarded['bool_masked_pos'], target, raw.patch_size).bool()
            pen, full = output[8]['pen_prediction'], output[7].detach()
            with torch.no_grad():
                for index in range(min(len(image), 3 - count)):
                    hidden = mask[index]
                    visible = target[index].masked_fill(hidden, 0)
                    composite = torch.where(hidden, pen[index], target[index])
                    values = (image[index], target[index], visible, pen[index], composite, full[index])
                    labels = ('Source', 'GT', 'Masked target', 'P_pen', 'Composite', 'P_full')
                    width, height = target.shape[-1], target.shape[-2]
                    canvas = Image.new('RGB', (width * 6, height + 32), 'white')
                    draw = ImageDraw.Draw(canvas)
                    for column, (value, label) in enumerate(zip(values, labels)):
                        canvas.paste(tensor_image(value.detach()), (column * width, 32))
                        draw.text((column * width + 6, 8), label, fill='black')
                    stem = f'sample_{count:02d}'
                    canvas.save(destination / f'{stem}.png')
                    canvas.resize((1344, (height + 32) // 2)).save(destination / f'{stem}_preview.png')
                    metadata = dict(epoch=epoch, training_forward=calls - 1,
                                    batch_index=index, mask_pixels=int(hidden[0].sum()),
                                    mask_fraction=float(hidden[0].float().mean()),
                                    has_jt=bool(forwarded['dual_targets']['has_jt'][index]),
                                    panels=labels, extra_inference=False,
                                    mask_fill='zero in ImageNet normalized space')
                    (destination / f'{stem}.json').write_text(json.dumps(metadata, indent=2))
                    count += 1
            print(f'[training-export] epoch={epoch} saved={count}/3 directory={destination}', flush=True)

        hook = raw.register_forward_hook(save_forward, with_kwargs=True)
        try:
            return original_epoch(*arguments, **keywords)
        finally:
            hook.remove()

    main_train.train_one_epoch = capture_epoch


def train_worker(options):
    import torch
    import main_train
    from tools import dual_view_protocol
    from util import lr_sched, misc

    checkpoint = torch.load(options.checkpoint, map_location='cpu', weights_only=False)
    saved = validate_checkpoint(checkpoint)
    args = argparse.Namespace(**saved)
    args.resume = str(options.checkpoint.resolve())
    args.auto_resume = False
    args.start_epoch = 51
    args.epochs = 71
    args.lr = options.peak_lr
    args.warmup_epochs = 0
    args.output_dir = str((ROOT / 'models/runb_fix_pixel_continue20').resolve())
    args.log_dir = str(Path(args.output_dir) / 'logs')
    args.rank, args.world_size, args.gpu = -1, 1, 0
    args.dist_url = 'env://'
    args.distributed = False
    args.continuation_start_epoch = 51
    args.continuation_epochs = 20
    args.continuation_peak_lr = options.peak_lr
    args.continuation_schedule = '20-epoch cosine refinement, no warmup'
    args.continuation_source_sha256 = options.checkpoint_sha256
    args.continuation_rng_restored = False
    args.val_tb_image_freq = 5
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    del checkpoint

    original_validation = dual_view_protocol.validate_training_inputs

    def validate_continuation(current):
        if (current.resume != str(options.checkpoint.resolve())
                or current.start_epoch != 51 or current.epochs != 71
                or not current.dual_pixel_mask):
            raise ValueError('Continuation identity changed')
        if digest(options.checkpoint) != options.checkpoint_sha256:
            raise ValueError('Continuation checkpoint changed')
        resume, lr, warmup = current.resume, current.lr, current.warmup_epochs
        try:
            # Keep the original first-run scientific/data checks intact.
            current.resume = ''
            current.lr, current.warmup_epochs = saved['lr'], saved['warmup_epochs']
            return original_validation(current)
        finally:
            current.resume, current.lr, current.warmup_epochs = resume, lr, warmup

    def load_continuation(args, model, model_without_ddp, optimizer,
                          optimizer_d, loss_scaler):
        source = torch.load(args.resume, map_location='cpu', weights_only=False)
        validate_checkpoint(source)
        model_without_ddp.load_state_dict(source['model'], strict=True)
        optimizer.load_state_dict(source['optimizer'])
        loss_scaler.load_state_dict(source['scaler'])
        if optimizer_d is not None:
            raise ValueError('Pixel continuation must retain noGAN')
        if len(optimizer.state) != len(source['optimizer']['state']):
            raise RuntimeError('Optimizer state was not fully restored')
        if loss_scaler.state_dict() != source['scaler']:
            raise RuntimeError('AMP state was not fully restored')
        args.start_epoch = source['epoch'] + 1
        steps = [float(state['step']) for state in optimizer.state.values() if 'step' in state]
        audit = dict(status='restored', source_epoch=source['epoch'], start_epoch=args.start_epoch,
                     optimizer_states=len(optimizer.state), optimizer_step_min=min(steps),
                     optimizer_step_max=max(steps), scaler=loss_scaler.state_dict(),
                     model_loaded_strictly=True, rng_in_source=False,
                     source_sha256=options.checkpoint_sha256, peak_lr=options.peak_lr)
        rank = misc.get_rank()
        (Path(args.output_dir) / f'resume_audit_rank{rank}.json').write_text(json.dumps(audit, indent=2))
        print(json.dumps({'continuation_restore': audit, 'rank': rank}), flush=True)

    def adjust(optimizer, epoch, current):
        lr = continuation_rate(epoch, 51, 20, current.continuation_peak_lr)
        for group in optimizer.param_groups:
            group['lr'] = lr * group.get('lr_scale', 1.0)
        return lr

    dual_view_protocol.validate_training_inputs = validate_continuation
    misc.auto_load_model = load_continuation
    lr_sched.adjust_learning_rate = adjust
    install_training_exports(main_train)
    main_train.main(args, None)


def controller(options):
    import torch
    source = torch.load(options.checkpoint, map_location='cpu', weights_only=False)
    saved = validate_checkpoint(source)
    source_sha = digest(options.checkpoint)
    del source
    result = ROOT / 'experiment_continue20'
    result.mkdir(exist_ok=False)
    training = ROOT / 'models/runb_fix_pixel_continue20'
    state = dict(status='prepared', started=timestamp(), base_epochs=51, total_epochs=71,
                 additional_epochs=20, source_checkpoint=str(options.checkpoint),
                 source_sha256=source_sha, peak_lr=options.peak_lr,
                 training_output=str(training), rng_in_source=False)

    def update(status, **values):
        state.update(status=status, updated=timestamp(), **values)
        temporary = result / 'status.tmp'
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        temporary.replace(result / 'status.json')
        print(json.dumps({'time': timestamp(), 'status': status, **values}), flush=True)

    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0,1', OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')

    def run(command, name):
        with (result / name).open('x') as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)

    try:
        update('training')
        run([sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node=2',
             '--master_port=29745', 'tools/continue_runb_pixel.py', '--worker',
             '--checkpoint', str(options.checkpoint), '--checkpoint-sha256', source_sha,
             '--peak-lr', str(options.peak_lr)], 'train_console.log')
        logs = [json.loads(line) for line in (training / 'log.txt').read_text().splitlines()]
        if [row['epoch'] for row in logs] != list(range(51, 71)):
            raise RuntimeError('Continuation did not finish exactly 20 new epochs')
        update('evaluating', completed_epochs=71)
        checkpoints = {'continued_best': training / 'checkpoint-best.pth',
                       'continued_ep70': training / 'checkpoint-70.pth'}
        for name, checkpoint in checkpoints.items():
            run([sys.executable, 'tools/eval_s_baseline.py', '--checkpoint', str(checkpoint),
                 '--data_path', saved['data_path'], '--val_json_path', *saved['val_json_path'],
                 '--fixed_pair_path', saved['fixed_pair_path'],
                 '--calibration_json', saved['dual_eval_calibration_json'],
                 '--dual_target_cache', saved['dual_target_cache'],
                 '--j_definition', 'instance-lower-full-v1',
                 '--s_baseline_path', saved['s_baseline_path'], '--seed', '0',
                 '--output_json', str(result / f'{name}_eval.json')], f'{name}_eval.log')
        from tools.dual_report import paired_comparison
        from tools.run_runb_pixel_experiment import export_visuals
        evaluations = {
            'fix': json.loads((ROOT / 'experiment/fix_eval.json').read_text()),
            'pixel_ep50': json.loads((ROOT / 'experiment/pixel_ep50_eval.json').read_text()),
            **{name: json.loads((result / f'{name}_eval.json').read_text()) for name in checkpoints},
        }
        denominator = json.loads(Path(saved['s_baseline_path']).read_text())
        summary = []
        for name, evaluation in evaluations.items():
            row = dict(model=name, epoch=evaluation['source_epoch'],
                       **{key: evaluation[key] for key in ('L1_JT', 'L1_BF', 'J', 'S')})
            for reference in ('fix', 'pixel_ep50'):
                for key in ('L1_JT', 'L1_BF', 'J', 'S'):
                    row[f'{key}_reduction_vs_{reference}_percent'] = 100 * (1 - row[key] / evaluations[reference][key])
                if name in checkpoints:
                    comparison = paired_comparison(evaluation, evaluations[reference], denominator)
                    (result / f'{name}_vs_{reference}_paired.json').write_text(json.dumps(comparison, indent=2))
            summary.append(row)
        (result / 'comparison.json').write_text(json.dumps(summary, indent=2))
        with (result / 'comparison.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        update('exporting', metrics=summary)
        # Reuse the complete fixed-pair exporter: original Fix, pre-extension,
        # extended best and GT; end-epoch metrics are also retained above.
        visual_checkpoints = {'fix': Path(evaluations['fix']['source_checkpoint']),
                              'pixel_best': checkpoints['continued_best'],
                              'pixel_ep50': options.checkpoint}
        visual_evaluations = {'fix': evaluations['fix'], 'pixel_best': evaluations['continued_best'],
                              'pixel_ep50': evaluations['pixel_ep50']}
        export_visuals(visual_evaluations, visual_checkpoints, saved, result)
        if digest(options.checkpoint) != source_sha:
            raise RuntimeError('Source checkpoint changed during continuation')
        update('complete', completed_epochs=71, ended=timestamp(), metrics=summary)
    except Exception as error:
        update('failed', error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--peak-lr', type=float, default=1e-4)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--checkpoint-sha256')
    options = parser.parse_args()
    if not 0 < options.peak_lr <= 1e-3:
        raise ValueError('Expected a positive refinement learning rate <=1e-3')
    if options.worker:
        train_worker(options)
    else:
        controller(options)


if __name__ == '__main__':
    main()
