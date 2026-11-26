"""Inspect a running ablation queue at a configurable interval."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import subprocess
import time


def process_alive(pid, project):
    if not isinstance(pid, int):
        return False
    try:
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, ProcessLookupError):
        return False
    return any(argument.startswith(str(project).encode() + b"/")
               for argument in arguments)


def inspect_queue(run_root):
    project = run_root.parent
    pipeline = json.loads((run_root / "pipeline_status.json").read_text())
    controller = int((project / "controller.pid").read_text())
    report = {
        "time": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "status": pipeline["status"],
        "controller_alive": process_alive(controller, project),
        "variants": {}, "alerts": [],
    }
    for name, stage in pipeline["variants"].items():
        directory = run_root / name
        rows = []
        log = directory / "log.txt"
        if log.exists():
            for line in log.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # The final append may still be in progress.
                    continue
        evaluations = list((directory / "evaluation").glob("*/metrics.json"))
        report["variants"][name] = {
            "status": stage["status"], "completed_epochs": len(rows),
            "evaluated_checkpoints": len(evaluations),
        }
        if rows:
            for key in ("train_loss", "test_loss", "test_S"):
                value = rows[-1].get(key)
                if value is not None and not math.isfinite(value):
                    report["alerts"].append(f"{name}: nonfinite {key}")
        if stage["status"] in ("training", "evaluating"):
            if not process_alive(stage.get("pid"), project):
                report["alerts"].append(f"{name}: active process is absent")
    if pipeline["status"] == "running" and not report["controller_alive"]:
        report["alerts"].append("Queue controller is absent")
    if pipeline["status"] == "failed":
        report["alerts"].append(pipeline.get("error", "Queue failed"))
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
         "--format=csv,noheader"], capture_output=True, text=True, timeout=30,
    )
    report["gpu"] = gpu.stdout.strip()
    if gpu.returncode:
        report["alerts"].append(gpu.stderr.strip())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=7200)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    previous = None
    while True:
        try:
            report = inspect_queue(args.run_root.resolve())
            progress = (report["status"], tuple(
                (name, value["status"], value["completed_epochs"],
                 value["evaluated_checkpoints"])
                for name, value in report["variants"].items()
            ))
            if report["status"] == "running" and progress == previous:
                report["alerts"].append("No epoch or evaluation progress since the previous inspection")
            previous = progress
            print(json.dumps(report, ensure_ascii=False), flush=True)
            if args.once or report["status"] == "completed":
                return
        except Exception as error:
            print(json.dumps({"status": "inspection_error", "error": str(error)}), flush=True)
            if args.once:
                raise
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
