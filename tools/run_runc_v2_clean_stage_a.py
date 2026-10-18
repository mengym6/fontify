"""Execute clean Run C v2 checks and IA0/IA1, stopping before the IB gate."""

import argparse
import copy
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

from tools.runc_v2_runtime import sha256_file

ROOT = Path(__file__).resolve().parents[1]
POLICY = "distinct-original-reference-v1"


def phase_config(base, stage, checkpoint=None, steps=1, amp=True):
    """Make a fresh, explicitly safe phase without inheriting old selection."""
    if base.get("data", {}).get("reference_policy") != POLICY:
        raise ValueError("clean execution requires distinct-original references")
    result = copy.deepcopy(base)
    result["stage"] = stage
    result["checkpoint"] = checkpoint or base["checkpoint"]
    result["weights"] = {}
    for key in ("calibration_record", "stage_gate_record", "resume_optimizer",
                "stage_start_checkpoint"):
        result.pop(key, None)
    result["training"].update(
        steps=steps, batch_size=1, accum_iter=2, save_every=1,
        amp=amp, evaluate_every_save=False, save_initial=False,
    )
    return result


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class Executor:
    """Persist each child PID, command and exit code independently of chat."""

    def __init__(self, output):
        self.output = Path(output)
        self.records = []

    def state(self, status, **extra):
        value = {"status": status, "time_utc": utc_now(),
                 "controller_pid": os.getpid(), "records": self.records, **extra}
        temporary = self.output / "pipeline.json.tmp"
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(self.output / "pipeline.json")

    def command(self, name, arguments, world=1):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=(
            "0,1" if world == 2 else "0"), OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1")
        record = {"name": name, "command": arguments,
                  "started_utc": utc_now(), "world_size": world}
        self.records.append(record)
        with (self.output / f"{name}.log").open("x") as stream:
            child = subprocess.Popen(arguments, cwd=ROOT, env=environment,
                                     stdout=stream, stderr=subprocess.STDOUT)
            record["pid"] = child.pid
            self.state("running", active=name)
            record["exit_code"] = child.wait()
        record["ended_utc"] = utc_now()
        self.state("running", active=None)
        if record["exit_code"]:
            raise RuntimeError(f"{name} failed with exit {record['exit_code']}")

    def runtime(self, name, config, command, destination=None, world=1):
        path = self.output / f"{name}_config.json"
        with path.open("x") as stream:
            json.dump(config, stream, indent=2)
        arguments = [sys.executable]
        if world == 2:
            arguments += ["-m", "torch.distributed.run", "--standalone",
                          "--nproc_per_node=2"]
        arguments += ["-m", "tools.runc_v2", command, "--config", str(path),
                      "--output", str(destination or self.output / name)]
        self.command(name, arguments, world)
        return path

    def selection(self, name, probe_config, probe_output):
        path = self.output / f"{name}.json"
        self.command(name, [sys.executable, "tools/calibrate_runc_v2.py",
                           "--probe", str(probe_output / "gradient_probe.json"),
                           "--config", str(probe_config), "--output", str(path)])
        return path, json.loads(path.read_text())["selected_weights"]


def execute(base, output, models, ia1_steps=400):
    output, models = Path(output), Path(models)
    if sha256_file(base["checkpoint"]) != base["initial_checkpoint_sha256"]:
        raise ValueError("clean trajectory must restart from verified checkpoint14")
    if output.exists() or models.exists():
        raise ValueError("clean run directories must be new; never overwrite")
    output.mkdir(parents=True)
    models.mkdir(parents=True)
    runner = Executor(output)
    try:
        health_config = output / "health_config.json"
        health_config.write_text(json.dumps(phase_config(base, "ia0"), indent=2))
        runner.command("tiny_ia0", [
            sys.executable, "-m", "tools.runc_v2_health", "--config",
            str(health_config), "--output", str(output / "tiny_ia0"),
            "--device", "cuda:0", "--steps", "12",
        ])
        for stage in ("ia1", "ib"):
            for amp, world, suffix in ((False, 1, "fp32"), (True, 2, "ddp_bf16")):
                config = phase_config(base, stage, steps=2, amp=amp)
                config["smoke_accum_equivalence"] = stage == "ia1"
                runner.runtime(f"smoke_{stage}_{suffix}", config,
                               "smoke", world=world)
        checkpoint = base["checkpoint"]
        for stage, steps in (("ia0", 4), ("ia1", ia1_steps)):
            config = phase_config(base, stage, checkpoint)
            config.update(probe_batch_size=2, probe_repeats=2)
            probe_path = runner.runtime(f"probe_{stage}", config, "probe")
            selection, weights = runner.selection(
                f"selection_{stage}", probe_path, output / f"probe_{stage}",
            )
            config.update(weights=weights, calibration_record=str(selection),
                          stage_start_checkpoint=checkpoint)
            config["training"].update(
                steps=steps, batch_size=2, accum_iter=8,
                save_every=4 if stage == "ia0" else 50,
                warmup_steps=1 if stage == "ia0" else 20,
                save_initial=True, evaluate_every_save=True,
                lr_head=3e-4, lr_vit=3e-6, lr_coarse=3e-4,
                weight_decay=0.05, clip_grad=3.0,
            )
            destination = models / stage
            runner.runtime(f"train_{stage}", config, "train",
                           destination=destination, world=2)
            checkpoint = str(destination / f"checkpoint-step-{steps:06d}.pth")
        runner.state("stage_a_complete_pending_head_gate",
                     final_checkpoint=checkpoint,
                     next_action="Review all Head snapshots before IB; no BF training")
    except BaseException as error:
        runner.state("failed", error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/runc_v2/smoke.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--ia1-steps", type=int, default=400)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.ia1_steps <= 400:
        parser.error("IA1 bounded budget must be between 1 and 400")
    base = json.loads(Path(args.base_config).read_text())
    phase_config(base, "ia0")
    if args.execute:
        execute(base, args.output, args.models, args.ia1_steps)
    else:
        print(json.dumps({"dry_run": True, "policy": POLICY,
                          "ia0_updates": 4, "ia1_updates": args.ia1_steps,
                          "ends_before_ib": True}))


if __name__ == "__main__":
    main()
