"""Frozen validation reference policy and metadata identity contracts."""

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / 'tools/build_callify_final.py'
SPEC = importlib.util.spec_from_file_location('callify_final', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def record(character, kind='bf_only', writer='A', suffix=''):
    return dict(character=character, pool_type=kind, calligrapher=writer,
                target_path=f'{writer}/{kind}/{character}{suffix}.png')


def test_reference_is_different_character_same_writer_and_type():
    q = record('甲')
    rows = [q, record('甲', suffix='2'), record('乙'), record('丙', writer='B'), record('丁', kind='jt_only')]
    chosen = MODULE.select_reference(q, rows, preferred=q)
    assert chosen['character'] == '乙'
    assert chosen == MODULE.select_reference(q, list(reversed(rows)))


def test_valid_historical_reference_is_preserved():
    q, r, s = record('甲'), record('乙'), record('丙')
    assert MODULE.select_reference(q, [q, r, s], preferred=s) == s


def test_no_valid_reference_is_an_error_instead_of_self_reference():
    q = record('甲')
    with pytest.raises(ValueError):
        MODULE.select_reference(q, [q, record('甲', suffix='2')])
