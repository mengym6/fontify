"""Policy and failure receipt checks for the unattended clean-stage driver."""

import json
import sys

import pytest

from tools.run_runc_v2_clean_stage_a import POLICY, Executor, phase_config


def base():
    return {"data": {"reference_policy": POLICY}, "checkpoint": "ckpt14.pth",
            "stage": "ia0", "weights": {"head": 9},
            "calibration_record": "obsolete.json", "resume_optimizer": True,
            "stage_start_checkpoint": "obsolete-start.pth",
            "training": {"seed": 0}}


def test_phase_drops_old_selection_and_does_not_mutate_base():
    original = base()
    result = phase_config(original, "ia1", "clean_ia0_step4.pth")
    assert result["checkpoint"] == "clean_ia0_step4.pth"
    assert result["weights"] == {}
    assert "calibration_record" not in result and "resume_optimizer" not in result
    assert "stage_start_checkpoint" not in result
    assert original["weights"] == {"head": 9}
    assert result["data"]["reference_policy"] == POLICY


def test_unsafe_base_fails_before_commands():
    original = base()
    original["data"].clear()
    with pytest.raises(ValueError, match="distinct-original"):
        phase_config(original, "ia0")


def test_actual_child_failure_is_recorded_and_propagated(tmp_path):
    runner = Executor(tmp_path)
    with pytest.raises(RuntimeError, match="failed with exit 7"):
        runner.command("failure", [sys.executable, "-c", "raise SystemExit(7)"])
    record = json.loads((tmp_path / "pipeline.json").read_text())
    assert record["records"][0]["exit_code"] == 7
    assert record["records"][0]["pid"] > 0


def test_successful_command_retains_real_exit_code(tmp_path):
    runner = Executor(tmp_path)
    runner.command("success", [sys.executable, "-c", "print('verified')"])
    record = json.loads((tmp_path / "pipeline.json").read_text())
    assert record["records"][0]["exit_code"] == 0
    assert "verified" in (tmp_path / "success.log").read_text()
