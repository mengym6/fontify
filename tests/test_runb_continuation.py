"""Continuation numbering, nonzero refinement schedule and checkpoint identity."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'tools/continue_runb_pixel.py'
SPEC = importlib.util.spec_from_file_location('runb_continuation', SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def checkpoint():
    return dict(model={}, optimizer={'state': {0: {'step': 867}}}, scaler={'scale': 65536},
                epoch=50, args=SimpleNamespace(dual_pixel_mask=True, dual_view=True, epochs=51))


def test_twenty_epochs_continue_after_completed_epoch50():
    saved = MODULE.validate_checkpoint(checkpoint())
    assert saved['epochs'] == 51
    assert list(range(51, 71)) == list(range(51, 51 + 20))
    assert MODULE.continuation_rate(51) == pytest.approx(1e-4)
    assert MODULE.continuation_rate(61) == pytest.approx(5e-5)
    assert MODULE.continuation_rate(71) == pytest.approx(0)
    values = [MODULE.continuation_rate(51 + i / 10) for i in range(201)]
    assert all(a >= b for a, b in zip(values, values[1:]))


@pytest.mark.parametrize('change', ['optimizer', 'scaler', 'epoch', 'pixel'])
def test_rejects_incomplete_or_wrong_source(change):
    value = checkpoint()
    if change == 'epoch':
        value['epoch'] = 49
    elif change == 'pixel':
        value['args'].dual_pixel_mask = False
    else:
        del value[change]
    with pytest.raises(ValueError):
        MODULE.validate_checkpoint(value)
