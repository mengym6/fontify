"""Match the two-view tip ViT gradient budget to the replaced Run B edge.

The rule preserves a known gradient scale, not a claim of optimal weight.
All measurements use fixed training pairs/seeds and train-mode bf16 forwards.
The norm is taken after composing pen + lambda_full * full for each batch.
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.dual_view_protocol import (  # noqa: E402
    apply_weights,
    build_dataset,
    check_samples,
    current_hashes,
    fixed_batch,
    forward_terms,
    load_dual_model,
    seed_all,
    sha256_file,
    validate_calibration,
    write_new_json,
)


def norm(loss, parameters, retain_graph):
    import torch

    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True,
    )
    total = loss.new_zeros((), dtype=torch.float64)
    for value in gradients:
        if value is not None:
            total += value.detach().double().square().sum()
    result = total.sqrt().item()
    if not math.isfinite(result):
        raise ValueError('Nonfinite shared ViT detail gradient')
    return result


def reference_edge(model, predictions, target):
    """Use the original autocast-to-FP32 L1, including its exact reductions."""
    from torch.nn import functional as functional

    truth = model.improved_edge_detection(target)
    edges = [functional.l1_loss(model.improved_edge_detection(prediction), truth)
             for prediction in predictions]
    return 0.2 * (edges[0] + model.dual_lambda_full * edges[1])


def measure(model, dataset, samples, device, weighted=None):
    import torch

    from tools.dual_gradient_probe import parameter_groups

    parameters = parameter_groups(model)[0]['vit']
    records = []
    decoder = model.forward_decoder
    for group in ('jt', 'bf_only', 'mixed'):
        rows = samples['groups'][group]
        for start in range(0, len(rows), samples['batch_size']):
            current = rows[start:start + samples['batch_size']]
            seed_all(current[0]['forward_seed'])
            batch = fixed_batch(dataset, current, device)
            predictions = []

            def capture(*args, **kwargs):
                prediction = decoder(*args, **kwargs)
                predictions.append(prediction)
                return prediction

            model.forward_decoder = capture
            try:
                extra = forward_terms(model, batch, epoch=18)
            finally:
                model.forward_decoder = decoder
            assert len(predictions) == 2
            tip = extra['tip_pen'] + model.dual_lambda_full * extra['tip_full']
            if weighted is None:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    edge = reference_edge(
                        model, predictions, batch['target'].to(torch.bfloat16),
                    )
                assert edge.dtype == torch.float32
                edge_norm = norm(edge, parameters, retain_graph=True)
                tip_norm = norm(tip, parameters, retain_graph=False)
            else:
                edge_norm = None
                tip_norm = norm(weighted * tip, parameters, retain_graph=False)
            record = {
                'group': group, 'batch_start': start,
                'n': len(current), 'forward_seed': current[0]['forward_seed'],
                'edge_weighted_vit_norm': edge_norm, 'tip_vit_norm': tip_norm,
                'tip_pen': extra['tip_pen'].detach().item(),
                'tip_full': extra['tip_full'].detach().item(),
                'pen_pool_sums': extra['tip_pen_pools'].cpu().tolist(),
                'full_pool_sums': extra['tip_full_pools'].cpu().tolist(),
            }
            records.append(record)
            del extra, batch, tip
            predictions.clear()
            if weighted is None:
                del edge
            print(json.dumps({k: v for k, v in record.items()
                              if not k.endswith('sums')}), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-path', 'target-cache', 'samples', 'checkpoint',
                 'parent-checkpoint', 'fixed-pair-path', 'eval-coef',
                 's-baseline', 'calibration', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--json-path', nargs='+', required=True)
    parser.add_argument('--val-json-path', nargs='+', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    hashes = current_hashes(args)
    training = validate_calibration(args.calibration, hashes)
    if training.get('image_detail_loss') != 'tip':
        raise ValueError('Fresh initial tip graph calibration required')
    dataset = build_dataset(args)
    samples = check_samples(args.samples, args)
    model, _ = load_dual_model(args.checkpoint, args.device)
    apply_weights(model, training)
    model.enable_tip(0.0)
    model.train()
    initial = measure(model, dataset, samples, args.device)
    mean_edge = sum(r['edge_weighted_vit_norm'] for r in initial) / len(initial)
    mean_tip = sum(r['tip_vit_norm'] for r in initial) / len(initial)
    if min(mean_edge, mean_tip) <= 0:
        raise ValueError('The mean detail gradient must be positive')
    weight = mean_edge / mean_tip
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError('Invalid tip gradient budget weight')
    write_new_json(args.output + '.initial.json', {
        'tip_weight': weight, 'measurements': initial, **hashes,
    })
    # Independent backward with the actual coefficient, rather than only
    # multiplying stored norms; bf16 arithmetic is not exactly linear.
    final = measure(model, dataset, samples, args.device, weighted=weight)
    mean_final = sum(r['tip_vit_norm'] for r in final) / len(final)
    ratio = mean_final / mean_edge
    write_new_json(args.output + '.final.json', {
        'tip_weight': weight, 'budget_ratio': ratio, 'measurements': final,
        **hashes,
    })
    if not math.isclose(ratio, 1.0, rel_tol=1e-3, abs_tol=1e-6):
        raise ValueError(f'Tip weighted gradient remeasurement failed: {ratio}')
    training.update(tip_final_weight=weight, image_detail_loss='tip')
    Path(args.calibration).write_text(json.dumps(training, indent=2) + '\n')
    result = {
        'schema': 'runb-full-tip-v1', 'status': 'validated', **hashes,
        'tip_weight': weight, 'schedule': {'start': 10, 'duration': 8},
        'training_calibration_sha256': sha256_file(args.calibration),
        'rule': 'ratio of means of composed two-view shared ViT gradient norms',
        'edge_final_weight_reference': 0.2,
        'mean_old_weighted_edge_norm': mean_edge, 'mean_raw_tip_norm': mean_tip,
        'mean_new_weighted_tip_norm': mean_final, 'remeasured_ratio': ratio,
        'n_batches': len(initial), 'train_mode': True, 'autocast': 'bfloat16',
        'initial_epoch': 0, 'detail_measurement_epoch': 18,
        'implementation_sha256': {
            name: sha256_file(ROOT / name)
            for name in ('util/tip_loss.py', 'models_train.py')
        },
        'initial_measurements': args.output + '.initial.json',
        'final_measurements': args.output + '.final.json',
        'calibrator_sha256': sha256_file(Path(__file__)),
        'interpretation': 'Gradient scale matching; not an optimized hyperparameter.',
    }
    write_new_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
