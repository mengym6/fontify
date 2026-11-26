"""Joint training followed by BF-only supervision with a frozen shallow backbone."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def resolve_stages(after=None, freeze_layers=None, resume_checkpoint=None):
    """Resume inherits its saved schedule; legacy runs remain joint training."""
    saved = None
    if resume_checkpoint is not None:
        path = Path(resume_checkpoint).expanduser().resolve().parent.parent / 'partial_supervision.json'
        saved = json.loads(path.read_text()).get('staged_training') if path.is_file() else None
        base = saved or {'bf_only_after': 0, 'freeze_layers': 0}
    else:
        base = {'bf_only_after': 40, 'freeze_layers': 4}
    result = {'bf_only_after': base['bf_only_after'] if after is None else after,
              'freeze_layers': base['freeze_layers'] if freeze_layers is None else freeze_layers}
    if any(not isinstance(v, int) or v < 0 for v in result.values()):
        raise ValueError('Stage epoch and freeze layer count must be nonnegative integers')
    result = result if result['bf_only_after'] else None
    if resume_checkpoint is not None and result != saved:
        raise ValueError('Resume must retain the saved staged-training schedule')
    return result


def bf_only_targets(batch, batch_size):
    """Exclude JT GT before assignment, and mask every JT classification channel."""
    import torch
    keep = batch['cls'].reshape(-1) < 50
    result = dict(batch)
    for key in ('batch_idx', 'cls', 'bboxes', 'masks'):
        result[key] = batch[key][keep.to(batch[key].device)]
    valid = batch.get('task_valid', torch.ones(batch_size, 2, dtype=torch.bool))
    result['task_valid'] = valid.clone().bool()
    result['task_valid'][:, 1] = False
    return result


def apply_training_stage(trainer):
    """Called before native _model_train; its BN-freeze mechanism is retained."""
    from ultralytics.utils.torch_utils import unwrap_model
    from partial_supervision import _branches

    config = trainer.staged_training
    active = bool(config and trainer.epoch >= config['bf_only_after'])
    for branch in _branches(trainer.model):
        branch._calli_bf_only = active
    # Validation always supervises both available tasks, including after switch.
    ema = getattr(getattr(trainer, 'ema', None), 'ema', None)
    for branch in _branches(ema):
        branch._calli_bf_only = False
    if not active or getattr(trainer, '_bf_stage_applied', False):
        return

    model = unwrap_model(trainer.model)
    count = config['freeze_layers']
    if count > len(model.yaml['backbone']):
        raise ValueError('Stage freeze must select shallow backbone modules only')
    # Clear pending joint-training gradients at the stage boundary. Frozen
    # parameters have grad=None, so AdamW neither decays nor momentum-updates them.
    trainer.optimizer.zero_grad(set_to_none=True)
    prefixes = [f'model.{index}.' for index in range(count)]
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            parameter.requires_grad_(False)
            parameter.grad = None
    trainer.freeze_layer_names = list(dict.fromkeys([*trainer.freeze_layer_names, *prefixes]))
    trainer._bf_stage_applied = True
    print(f'BF-only stage starts at epoch {trainer.epoch + 1}; '
          f'froze backbone modules 0..{count - 1} including their BatchNorm statistics; '
          'validation still evaluates BF and JT', flush=True)


def save_joint_checkpoint(trainer):
    """Retain one useful boundary checkpoint, in addition to native best/last."""
    config = trainer.staged_training
    if config and trainer.epoch + 1 == config['bf_only_after']:
        shutil.copyfile(trainer.last, Path(trainer.last).with_name('joint_end.pt'))
