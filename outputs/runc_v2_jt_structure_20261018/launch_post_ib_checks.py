"""Persist final read-only condition audit and one-update Refiner interface check."""

import json
import sys
from pathlib import Path

from tools.run_runc_v2_clean_stage_a import Executor
from tools.runc_v2_runtime import sha256_file, validate_config

ROOT = Path("outputs/runc_v2_jt_structure_20261018")
state = json.loads((ROOT / "clean_run2_ib/pipeline.json").read_text())
assert state["status"] == "stage_b_complete_pending_structure_review"
checkpoint = Path(state["final_checkpoint"])
assert sha256_file(checkpoint) == (
    "f3e39d8f1747ae5449c74e2a83f5d189d00b49edbc3b4da54f25d87b991970d4"
)
output = ROOT / "post_ib_checks"
output.mkdir(exist_ok=False)
base = json.loads(Path("configs/runc_v2/smoke.json").read_text())
config = {key: base[key] for key in (
    "schema", "experiment", "initial_checkpoint_sha256", "model", "data", "backend"
)}
config.update(
    stage="ib", checkpoint=str(checkpoint), weights={},
    training={"seed": 0, "amp": True},
    purpose="One-update interface only; final stage-I acceptance remains pending",
)
validate_config(config, "stage2-smoke")
config_path = output / "stage2_interface_config.json"
config_path.write_text(json.dumps(config, indent=2))
runner = Executor(output)
try:
    runner.command("condition", [
        sys.executable, "tools/runc_v2_condition_audit.py",
        "--config", str(ROOT / "clean_run2_ib/train_ib_config.json"),
        "--checkpoint", str(checkpoint),
        "--output", str(output / "condition"), "--device", "cuda:0",
    ])
    runner.command("stage2", [
        sys.executable, "-m", "tools.runc_v2", "stage2-smoke",
        "--config", str(config_path), "--output", str(output / "stage2"),
    ])
    runner.state("post_checks_complete_pending_structure_review")
except BaseException as error:
    runner.state("failed", error=repr(error))
    raise
