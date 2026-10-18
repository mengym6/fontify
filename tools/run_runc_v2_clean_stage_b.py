"""Run calibrated, frozen-Head IB only after an explicit technical gate."""

import argparse
import json
import sys
from pathlib import Path

from tools.run_runc_v2_clean_stage_a import Executor, phase_config
from tools.runc_v2_runtime import validate_stage_gate


def coarse_health_passed(record):
    """Require actual fixed-input learning; a zero-update report is insufficient."""
    before, after = record["before_fixed_eval"], record["after_fixed_eval"]
    names = ("total", "raw_density", "raw_occupancy")
    flags = {name: after[name] < before[name] for name in names}
    return bool(record.get("frozen_state_passed") and all(flags.values())), flags


def execute(base, checkpoint, gate, output, models, steps=320):
    config = phase_config(base, "ib", checkpoint)
    config["stage_gate_record"] = str(gate)
    validate_stage_gate(config)
    output, models = Path(output), Path(models)
    if output.exists() or models.exists():
        raise ValueError("IB output directories must be new")
    output.mkdir(parents=True)
    runner = Executor(output)
    try:
        config.update(probe_batch_size=2, probe_repeats=2)
        probe = runner.runtime("probe_ib", config, "probe")
        selection, weights = runner.selection("selection_ib", probe,
                                              output / "probe_ib")
        config.update(weights=weights, calibration_record=str(selection),
                      stage_start_checkpoint=str(checkpoint))
        config["training"].update(lr_coarse=3e-4)
        health_config = output / "health_ib_config.json"
        health_config.write_text(json.dumps(config, indent=2))
        runner.command("tiny_ib", [
            sys.executable, "-m", "tools.runc_v2_health",
            "--config", str(health_config), "--output", str(output / "tiny_ib"),
            "--device", "cuda:0", "--steps", "12", "--no-profile",
        ])
        health = json.loads((output / "tiny_ib/smoke.json").read_text())
        passed, flags = coarse_health_passed(health)
        (output / "coarse_learning_gate.json").write_text(json.dumps({
            "passed": passed, "loss_decreases": flags,
            "scope": "fixed two targets; correctness only, not quality acceptance",
        }, indent=2))
        if not passed:
            raise RuntimeError("IB fixed-input learning check failed")
        config["training"].update(
            steps=steps, batch_size=2, accum_iter=8, save_every=40,
            warmup_steps=16, save_initial=True, evaluate_every_save=True,
        )
        runner.runtime("train_ib", config, "train", destination=models, world=2)
        runner.state("stage_b_complete_pending_structure_review",
                     final_checkpoint=str(models / f"checkpoint-step-{steps:06d}.pth"),
                     next_action="P0/Head/condition review; no automatic BF training")
    except BaseException as error:
        runner.state("failed", error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/runc_v2/smoke.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.steps <= 320:
        parser.error("IB budget must be between 1 and 320 updates")
    base = json.loads(Path(args.base_config).read_text())
    if args.execute:
        execute(base, args.checkpoint, args.gate, args.output, args.models, args.steps)
    else:
        print(json.dumps({"dry_run": True, "stage": "ib", "updates": args.steps,
                          "gate_required": True, "no_automatic_bf_training": True}))


if __name__ == "__main__":
    main()
