"""Scale YOLO26's auxiliary semantic loss without changing its other terms.

The pinned Ultralytics 8.4.218 SegmentLoss26 returns a five-element loss vector
(box, instance-mask, class, L1, semantic) and detached named log items. Both
E2E branches retain their native assigners and schedule. Model/checkpoint
classes are unchanged; only the transient criterion is specialized.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path


SUPPORTED_ULTRALYTICS = '8.4.218'
CONFIG_NAME = 'loss_config.json'
SCHEMA = 'calli-yolo26-semantic-loss-v1'


def _weight(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('semantic_weight must be finite and nonnegative')
    return value


def _saved_weight(path: Path) -> float:
    config = json.loads(path.read_text(encoding='utf-8'))
    if config.get('schema') != SCHEMA:
        raise ValueError(f'Unknown loss configuration schema: {path}')
    return _weight(config['semantic_weight'])


def resolve_semantic_weight(requested: float | None,
                            resume_checkpoint: Path | str | None = None) -> float:
    """Resolve an explicit request or inherit the resumed run's saved weight.

    An absent sidecar identifies a legacy native run (weight 1). Explicitly
    changing loss semantics during resume is rejected; use a new run instead.
    This function deliberately does not import torch or Ultralytics.
    """
    explicit = None if requested is None else _weight(requested)
    if resume_checkpoint is None:
        return 1.0 if explicit is None else explicit
    path = Path(resume_checkpoint).expanduser().resolve().parent.parent / CONFIG_NAME
    saved = _saved_weight(path) if path.is_file() else 1.0
    if explicit is not None and explicit != saved:
        raise ValueError(
            f'Resume semantic_weight={saved}, requested {explicit}. '
            'Omit --sem-weight to inherit it; changing the loss requires a new run.'
        )
    return saved


def scale_semantic_outputs(losses, items: dict, semantic_weight: float):
    """Multiply only index 4 / sem_loss, retaining autograd even at weight 0."""
    weight = _weight(semantic_weight)
    if tuple(losses.shape) != (5,) or 'sem_loss' not in items:
        raise RuntimeError('Expected native YOLO26 five-term segmentation loss')
    if weight == 1.0:
        return losses, items
    scaled = losses * losses.new_tensor([1.0, 1.0, 1.0, 1.0, weight])
    logged = dict(items)
    logged['sem_loss'] = items['sem_loss'] * weight
    return scaled, logged


def _weighted_segment_class(native_class):
    class SemanticWeightedSegmentLoss26(native_class):
        def loss(self, preds, batch):
            losses, items = super().loss(preds, batch)
            return scale_semantic_outputs(losses, items, self._calli_semantic_weight)
    return SemanticWeightedSegmentLoss26


def install_semantic_weight(model, semantic_weight: float) -> dict:
    """Install on one live model, preserving existing E2E criterion state.

    Imports are lazy so --dry-run and config resolution remain torch-free.
    No model parameters or model classes are replaced. Existing native branch
    state (assigners, hyperparameters, caches) is retained without reinitializing.
    """
    import ultralytics
    from ultralytics.utils.loss import E2ELoss, SegmentLoss26
    from ultralytics.utils.torch_utils import unwrap_model

    if ultralytics.__version__ != SUPPORTED_ULTRALYTICS:
        raise RuntimeError(f'Expected ultralytics=={SUPPORTED_ULTRALYTICS}, got {ultralytics.__version__}')
    weight = _weight(semantic_weight)
    model = unwrap_model(model)
    criterion = getattr(model, 'criterion', None)
    if criterion is None:
        criterion = model.init_criterion()
    weighted_class = _weighted_segment_class(SegmentLoss26)

    def convert(branch):
        if not isinstance(branch, SegmentLoss26):
            raise TypeError('Semantic weighting requires the YOLO26 SegmentLoss26 criterion')
        if hasattr(branch, '_calli_semantic_weight'):
            branch._calli_semantic_weight = weight
            return branch
        wrapped = object.__new__(weighted_class)
        wrapped.__dict__ = branch.__dict__.copy()
        wrapped._calli_semantic_weight = weight
        return wrapped

    if isinstance(criterion, E2ELoss):
        criterion.one2many = convert(criterion.one2many)
        criterion.one2one = convert(criterion.one2one)
        branches = 2
    else:
        criterion = convert(criterion)
        branches = 1
    model.criterion = criterion
    return {'semantic_weight': weight, 'branches': branches}


def make_semantic_loss_callback(semantic_weight: float):
    """Return an on_train_start callback for the model and its EMA criterion.

    Ultralytics 8.4.218 saves a deepcopy of EMA and clears its criterion before
    torch.save (trainer.py:810-812); strip_optimizer also clears the criterion.
    The custom transient loss class is therefore not a checkpoint dependency.
    """
    weight = _weight(semantic_weight)

    def on_train_start(trainer):
        path = Path(trainer.save_dir) / CONFIG_NAME
        if path.is_file() and _saved_weight(path) != weight:
            raise ValueError(f'Existing run has a different semantic loss weight: {path}')
        installed = {'model': install_semantic_weight(trainer.model, weight)}
        ema = getattr(getattr(trainer, 'ema', None), 'ema', None)
        if ema is not None:
            installed['ema'] = install_semantic_weight(ema, weight)
        if int(os.environ.get('RANK', '-1')) in (-1, 0):
            config = {
                'schema': SCHEMA, 'semantic_weight': weight,
                'ultralytics_version': SUPPORTED_ULTRALYTICS,
                'terms': ['box_loss', 'seg_loss', 'cls_loss', 'l1_loss', 'sem_loss'],
                'scaled_term': 'sem_loss', 'installed': installed,
                'checkpoint_note': 'Native checkpoint serialization strips criterion; retain this sidecar when resuming.',
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            temporary.replace(path)
            print(f'YOLO26 auxiliary semantic weight: {weight}; saved {path}', flush=True)
    return on_train_start
