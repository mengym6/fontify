"""Strict provenance checks for complete Run B tip replacement."""

import json
import math
from pathlib import Path

from tools.dual_view_protocol import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def validate_tip_calibration(path, weight, training_path, hashes):
    """Reject stale weights, an old image objective or an unmeasured weight."""
    if not path:
        raise ValueError('Explicit tip calibration JSON is required')
    value = json.loads(Path(path).read_text())
    if value.get('schema') != 'runb-full-tip-v1':
        raise ValueError('Complete Run B tip calibration required')
    if value.get('status') != 'validated':
        raise ValueError('Tip calibration has not passed')
    if not math.isfinite(weight) or weight <= 0 or value['tip_weight'] != weight:
        raise ValueError('Tip weight is not the calibrated positive weight')
    if value['schedule'] != {'start': 10, 'duration': 8}:
        raise ValueError('Tip schedule must match original edge')
    if value['training_calibration_sha256'] != sha256_file(training_path):
        raise ValueError('Tip and training calibrations do not match')
    for name, expected in hashes.items():
        if value.get(name) != expected:
            raise ValueError(f'Stale tip calibration: {name}')
    for name, expected in value['implementation_sha256'].items():
        if name not in ('util/tip_loss.py', 'models_train.py'):
            raise ValueError('Unexpected tip implementation path')
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f'Tip implementation changed: {name}')
    train = json.loads(Path(training_path).read_text())
    if (train.get('image_detail_loss') != 'tip'
            or train.get('tip_final_weight') != weight):
        raise ValueError('Training calibration belongs to another objective')
    return value
