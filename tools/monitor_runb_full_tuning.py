"""Inspect complete-model training and evaluation jobs once per interval."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.monitor_runb_ablations import process_alive
from tools.run_runb_full_tuning import write_json

TERMINAL = {"completed", "failed", "stopped_after_goal"}


def completed_log_rows(path):
    if not path.exists():
        return []
    rows = []
    with path.open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def inspect(project, names, previous=None):
    report = {
        "time": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "runs": {}, "alerts": [],
    }
    for name in names:
        directory = project / "runs" / name
        path = directory / "status.json"
        if not path.exists():
            report["runs"][name] = {"status": "pending"}
            continue
        status = json.loads(path.read_text())
        active_run = Path(status.get("active_run_dir", str(directory)))
        rows = completed_log_rows(active_run / "training/log.txt")
        completed_epochs = status.get("completed_prefix_epochs", 0) + len(rows)
        results = [json.loads(path.read_text()) for path in
                   (directory / "evaluation").glob("*/comparison.json")]
        if status.get("active_run_dir"):
            results = status.get("evaluated", [])
        current = {
            "status": status["status"], "controller_pid": status.get("pid"),
            "controller_alive": process_alive(status.get("pid"), project),
            "completed_epochs": completed_epochs,
            "epoch_budget": status["config"]["epochs"],
            "evaluated_checkpoints": len(results),
        }
        if rows:
            current["S_select240"] = rows[-1].get("test_S")
            current["train_loss"] = rows[-1].get("train_loss")
            for key in ("train_loss", "test_loss", "test_S"):
                value = rows[-1].get(key)
                if value is not None and not math.isfinite(value):
                    report["alerts"].append(f"{name}: nonfinite {key}")
        if results:
            current["best_evaluated"] = min(
                results, key=lambda result: result["worst_relative_ratio"],
            )
        if current["status"] not in TERMINAL and not current["controller_alive"]:
            report["alerts"].append(f"{name}: active controller is absent")
        if current["status"] == "failed":
            report["alerts"].append(f"{name}: {status.get('error', 'failed')}")
        if completed_epochs > current["epoch_budget"]:
            report["alerts"].append(f"{name}: epoch budget exceeded")
        old = (previous or {}).get("runs", {}).get(name, {})
        if (current["status"] in ("training", "evaluating")
                and old.get("status") == current["status"]
                and old.get("completed_epochs") == completed_epochs
                and old.get("evaluated_checkpoints") == len(results)):
            report["alerts"].append(f"{name}: no progress since previous inspection")
        report["runs"][name] = current
    report["free_disk_bytes"] = shutil.disk_usage(project).free
    if report["free_disk_bytes"] < 5 * 1024 ** 3:
        report["alerts"].append("Less than 5 GiB of disk space remains")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
         "--format=csv,noheader"], capture_output=True, text=True, timeout=30,
    )
    report["gpu"] = gpu.stdout.strip()
    if gpu.returncode:
        report["alerts"].append(gpu.stderr.strip())
    report["all_terminal"] = all(
        row["status"] in TERMINAL for row in report["runs"].values()
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--interval", type=int, default=3600)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    project = args.project.resolve()
    previous = None
    while True:
        try:
            report = inspect(project, args.runs, previous)
            report["interval_seconds"] = args.interval
            report["next_inspection_at"] = (
                datetime.now(timezone(timedelta(hours=8)))
                + timedelta(seconds=args.interval)
            ).isoformat() if not report["all_terminal"] else None
            write_json(project / "hourly_monitor_status.json", report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
            previous = report
            if args.once or report["all_terminal"]:
                return
        except Exception as error:
            failure = {
                "time": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                "status": "inspection_error", "error": repr(error),
            }
            write_json(project / "hourly_monitor_error.json", failure)
            print(json.dumps(failure, ensure_ascii=False), flush=True)
            if args.once:
                raise
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
