"""Exercise actual generic checkpoint dispatch with small real methods."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from util.checkpoint import build_checkpoint_model, load_fontify_checkpoint
from util.runc_v2 import runc_v2_checkpoint_metadata

SPEC = importlib.util.spec_from_file_location(
    "_runc_checkpoint_fixture", Path(__file__).with_name("test_runc_v2_model.py")
)
FIXTURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURE)


def saved_model():
    model = FIXTURE.new_model("ib").eval()
    saved = {
        **runc_v2_checkpoint_metadata(model),
        "model": copy.deepcopy(model.state_dict()),
        "args": {"model": "tiny"},
    }
    return model, saved


def test_generic_factory_strictly_reconstructs_v2_prediction():
    model, saved = saved_model()
    module = SimpleNamespace(tiny=lambda **kwargs: FIXTURE.small_model())
    restored = build_checkpoint_model(saved, model_module=module).eval()
    imgs, tgts, _, _ = FIXTURE.sample_inputs()
    with torch.no_grad():
        torch.testing.assert_close(model.predict_full(imgs, tgts),
                                   restored.predict_full(imgs, tgts),
                                   rtol=0, atol=0)


def test_generic_load_rejects_stage_metadata_mismatch():
    model, saved = saved_model()
    saved["runc_v2_config"]["stage"] = "ia1"
    with pytest.raises(ValueError, match="metadata disagree"):
        load_fontify_checkpoint(model, saved)


def test_generic_load_rejects_missing_new_parameter():
    model, saved = saved_model()
    key = next(key for key in saved["model"] if key.startswith("runc_v2_coarse."))
    del saved["model"][key]
    with pytest.raises(RuntimeError, match="Missing key"):
        load_fontify_checkpoint(model, saved)


def test_generic_load_never_migrates_v2_as_run_b():
    model, saved = saved_model()
    with pytest.raises(ValueError, match="cannot use legacy-to-dual"):
        load_fontify_checkpoint(model, saved, allow_dual_initialization=True)
