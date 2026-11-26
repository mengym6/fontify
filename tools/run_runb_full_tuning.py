"""Train complete Run B candidates and evaluate frozen validation_268."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.dual_view_protocol import sha256_file


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def read_config(path):
    config = json.loads(Path(path).read_text())
    if config["head_weight"] <= 0:
        raise ValueError("Complete-model candidates require positive head supervision")
    if config.get("jieti_weight") is not None and config["jieti_weight"] <= 0:
        raise ValueError("Complete-model candidates require positive geometry supervision")
    if config.get("full_weight") is not None and config["full_weight"] <= 0:
        raise ValueError("Complete-model candidates require positive full-view supervision")
    if config["epochs"] < 1 or config["learning_rate"] <= 0:
        raise ValueError("Training budget and learning rate must be positive")
    prior_epochs = int(config.get("prior_epochs", 0))
    if prior_epochs < 0 or config["epochs"] + prior_epochs > config.get("epoch_limit", config["epochs"]):
        raise ValueError("Training epochs exceed the experiment budget")
    if config.get("initialization", "complete") not in ("complete", "baseline"):
        raise ValueError("Unknown complete-model initialization")
    if config.get("stages"):
        if sum(stage["epochs"] for stage in config["stages"]) != config["epochs"]:
            raise ValueError("Stage budgets must sum to the total epoch budget")
        if any(stage["epochs"] < 1 for stage in config["stages"]):
            raise ValueError("Stage budgets must be positive")
    return config


def train(options):
    import torch
    import main_train
    from tools import dual_view_protocol
    from util.checkpoint import checkpoint_args, checkpoint_is_dual

    config = read_config(options.config)
    if config.get("stages"):
        raise ValueError("Staged experiments use the run command")
    base = json.loads(Path(config["base_hparams"]).read_text())
    if not base.get("dual_view") or not base.get("dual_pixel_mask"):
        raise ValueError("Expected the original Pixel training configuration")
    source = Path(config["checkpoint"]).resolve()
    source_sha = sha256_file(source)
    if source_sha != config["checkpoint_sha256"]:
        raise ValueError("Starting checkpoint identity changed")
    saved = torch.load(source, map_location="cpu", weights_only=False)
    source_args = checkpoint_args(saved)
    from_baseline = config.get("initialization", "complete") == "baseline"
    if from_baseline:
        protocol = json.loads((Path(config["frozen_root"])
                               / "05_协议与复现/protocol.json").read_text())
        if (checkpoint_is_dual(saved)
                or source_sha != protocol["baseline_sha256"]
                or int(saved["epoch"]) != protocol["baseline_epoch"]):
            raise ValueError("Baseline initialization must match the frozen baseline")
    elif (not checkpoint_is_dual(saved)
            or source_args.get("dual_ablation", "none") != "none"):
        raise ValueError("This tuning stage starts from a complete Run B checkpoint")
    source_epoch = int(saved["epoch"])
    root_baseline_sha = source_args.get("full_tuning_root_baseline_sha256")
    if root_baseline_sha is None and source_args.get("dual_initialize_from_baseline"):
        root_baseline_sha = source_args.get("full_tuning_source_sha256")
    if from_baseline:
        root_baseline_sha = source_sha
    elif "prior_epochs" in config:
        expected_prior = int(source_args.get("full_tuning_prior_epochs", 0)) + source_epoch + 1
        protocol = json.loads((Path(config["frozen_root"])
                               / "05_协议与复现/protocol.json").read_text())
        if (expected_prior != config["prior_epochs"]
                or root_baseline_sha != protocol["baseline_sha256"]):
            raise ValueError("Staged checkpoint lineage or cumulative budget disagrees")
    del saved

    args = argparse.Namespace(**base)
    args.dual_ablation = "none"
    args.dual_initialize_from_baseline = from_baseline
    args.dual_fusion_mode = config.get("fusion_mode", "joint")
    args.dual_head_weight_override = float(config["head_weight"])
    args.dual_jieti_weight_override = config.get("jieti_weight")
    args.dual_full_weight_override = config.get("full_weight")
    args.dual_head_detach_backbone = bool(config["detach_head_backbone"])
    args.finetune = str(source)
    args.resume = ""
    args.auto_resume = False
    args.start_epoch = 0
    args.epochs = int(config["epochs"])
    args.lr = float(config["learning_rate"])
    args.warmup_epochs = int(config.get("warmup_epochs", 1))
    args.edge_warmup_epochs = int(config.get("edge_warmup_epochs", 0))
    args.loss_warmup_duration = int(config.get("loss_warmup_duration", 0))
    args.batch_size = 2
    args.accum_iter = 32
    args.num_workers = 4
    args.world_size = 1
    args.rank = -1
    args.gpu = -1
    args.local_rank = -1
    args.distributed = False
    args.dist_url = "env://"
    args.save_freq = int(config.get("save_freq", 5))
    args.save_best = True
    args.save_best_only = False
    args.output_dir = str(Path(config["output_dir"]).resolve() / "training")
    args.log_dir = str(Path(args.output_dir) / "logs")
    args.val_tb_image_freq = 5
    args.full_tuning_config = str(Path(options.config).resolve())
    args.full_tuning_source_sha256 = source_sha
    args.full_tuning_source_epoch = source_epoch
    args.full_tuning_prior_epochs = int(config.get("prior_epochs", 0))
    args.full_tuning_root_baseline_sha256 = root_baseline_sha
    args.full_tuning_optimizer = "fresh AdamW for the changed objective"
    args.full_tuning_validation = "frozen validation_268 development tuning"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    original_validation = dual_view_protocol.validate_training_inputs

    def validate_tuning(current):
        # Reuse the original data, pairing, warm-up, and calibration audit.
        # The authorized candidate explicitly changes the optimization schedule.
        baseline = argparse.Namespace(**base)
        baseline.output_dir = current.output_dir
        baseline.resume = ""
        baseline.auto_resume = False
        for name in ("data_path", "json_path", "val_json_path",
                     "fixed_pair_path", "dual_target_cache",
                     "dual_calibration_json", "dual_eval_calibration_json"):
            if getattr(current, name) != getattr(baseline, name):
                raise ValueError(f"Original data/protocol field changed: {name}")
        result = original_validation(baseline)
        current.dual_blind_manifest_sha256 = baseline.dual_blind_manifest_sha256
        if current.batch_size * current.accum_iter * current.world_size != 64:
            raise ValueError("Candidate effective batch must be 64")
        write_json(Path(current.output_dir) / "tuning_initialization.json", {
            "source": str(source), "source_sha256": source_sha,
            "source_epoch": source_epoch, "config": config,
            "base_hparams_sha256": sha256_file(config["base_hparams"]),
            "optimizer": current.full_tuning_optimizer,
            "data_audit": "original Run B validation passed",
            "initialization": config.get("initialization", "complete"),
            "timestamp": timestamp(),
        })
        return result

    dual_view_protocol.validate_training_inputs = validate_tuning
    try:
        main_train.main(args, None)
    finally:
        dual_view_protocol.validate_training_inputs = original_validation


def compare_result(result, thresholds):
    measured = {"S": result["S"], **result["groups"]["all"]}
    wins = {}
    relative = {}
    for metric, threshold in thresholds.items():
        higher = metric in ("PSNR", "SSIM")
        wins[metric] = (measured[metric] > threshold if higher
                        else measured[metric] < threshold)
        relative[metric] = (threshold / measured[metric] if higher
                            else measured[metric] / threshold)
    return {"metrics": measured, "wins": wins, "all_five_better": all(wins.values()),
            "worst_relative_ratio": max(relative.values())}


def evaluate(options):
    from tools.run_runb_ablations import evaluate as evaluate_checkpoint

    config = read_config(options.config)
    checkpoint = Path(options.checkpoint).resolve()
    if config.get("stages"):
        raise ValueError("Individual evaluations use the corresponding stage configuration")
    destination = Path(config["output_dir"]) / "evaluation" / checkpoint.stem
    evaluate_checkpoint(argparse.Namespace(
        checkpoint=checkpoint, frozen_root=Path(config["frozen_root"]),
        output_dir=destination, validation_set="validation_268",
        visual_case_ids=[],
    ))
    result = json.loads((destination / "metrics.json").read_text())
    comparison = compare_result(result, config["thresholds"])
    if config.get("ablation_thresholds"):
        comparison["vs_ablations"] = compare_result(result, config["ablation_thresholds"])
    comparison.update(checkpoint=str(checkpoint),
                      checkpoint_sha256=result["checkpoint_sha256"],
                      epoch=result["epoch_zero_based"],
                      cumulative_epochs=config.get("prior_epochs", 0) + result["epoch_zero_based"] + 1)
    write_json(destination / "comparison.json", comparison)


def wait_for_prerequisite(config, state, status_path):
    if not config.get("after_run"):
        return
    prerequisite = Path(config["after_run"]) / "status.json"
    while True:
        if prerequisite.exists():
            preceding = json.loads(prerequisite.read_text())
            if preceding["status"] == "completed":
                free_memory = subprocess.check_output([
                    'nvidia-smi', f'--id={config["gpu"]}',
                    '--query-gpu=memory.free', '--format=csv,noheader,nounits',
                ], text=True)
                if int(free_memory.strip()) >= 22000:
                    break
            if preceding["status"] in ("failed", "stopped_after_goal"):
                raise RuntimeError("Preceding experiment did not complete")
        time.sleep(60)
    state.update(status="training", started_at=timestamp())
    write_json(status_path, state)


def run_staged(options, config):
    if config.get("initialization") != "baseline":
        raise ValueError("Bounded staged experiments start from the baseline")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    queued = bool(config.get("after_run"))
    state = {"status": "queued" if queued else "training", "pid": os.getpid(),
             "config": config, "queued_at": timestamp() if queued else None,
             "started_at": None if queued else timestamp(),
             "completed_prefix_epochs": 0, "evaluated": []}
    status_path = output / "status.json"
    write_json(status_path, state)
    source = Path(config["checkpoint"])
    source_sha = config["checkpoint_sha256"]
    completed = 0
    comparisons = []
    try:
        wait_for_prerequisite(config, state, status_path)
        for number, stage in enumerate(config["stages"], 1):
            stage_dir = output / f"stage{number}"
            stage_config = {key: value for key, value in config.items()
                            if key not in ("stages", "after_run")}
            stage_config.update(stage)
            stage_config.update(
                name=f"{config['name']}_stage{number}", output_dir=str(stage_dir),
                checkpoint=str(source), checkpoint_sha256=source_sha,
                initialization="baseline" if number == 1 else "complete",
                prior_epochs=completed,
            )
            config_path = output / f"stage{number}_config.json"
            write_json(config_path, stage_config)
            read_config(config_path)
            state.update(active_run_dir=str(stage_dir), active_stage=number,
                         completed_prefix_epochs=completed)
            write_json(status_path, state)
            with (output / f"stage{number}_controller.log").open("w") as stream:
                subprocess.run([
                    sys.executable, str(Path(__file__).resolve()), "run",
                    "--config", str(config_path),
                ], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            summary = json.loads((stage_dir / "summary.json").read_text())
            for candidate in summary["candidates"]:
                candidate.update(stage=number,
                                 cumulative_epochs=completed + candidate["epoch"] + 1)
                comparisons.append(candidate)
            completed += stage["epochs"]
            source = stage_dir / "training" / f"checkpoint-{stage['epochs'] - 1}.pth"
            source_sha = sha256_file(source)
            state["evaluated"] = comparisons
            write_json(status_path, state)
        best = min(comparisons, key=lambda candidate: candidate["worst_relative_ratio"])
        write_json(output / "summary.json", {
            "config": config, "candidates": comparisons, "best": best,
            "completed_epochs": completed,
            "any_all_five_better": any(c["all_five_better"] for c in comparisons),
            "validation_role": "development selection on the frozen validation_268",
        })
        state.update(status="completed", ended_at=timestamp(), best=best,
                     completed_epochs=completed)
        write_json(status_path, state)
    except Exception as error:
        state.update(status="failed", ended_at=timestamp(), error=repr(error))
        write_json(status_path, state)
        raise


def run(options):
    config = read_config(options.config)
    if config.get("stages"):
        return run_staged(options, config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    queued = bool(config.get("after_run"))
    state = {"status": "queued" if queued else "training",
             "queued_at": timestamp() if queued else None,
             "started_at": None if queued else timestamp(),
             "config": config, "pid": os.getpid(), "evaluated": []}
    status_path = output / "status.json"
    write_json(status_path, state)
    command = [sys.executable, str(Path(__file__).resolve())]
    try:
        wait_for_prerequisite(config, state, status_path)
        with (output / "training_console.log").open("w") as stream:
            subprocess.run(command + ["train", "--config", str(options.config)],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        checkpoints = list((output / "training").glob("checkpoint-*.pth"))
        checkpoints.sort(key=lambda p: (p.stem == "checkpoint-best",
                                       int(p.stem.split("-")[-1])
                                       if p.stem != "checkpoint-best" else 0))
        if not checkpoints:
            raise RuntimeError("Training completed without checkpoints")
        state["status"] = "evaluating"
        write_json(status_path, state)
        comparisons = []
        for checkpoint in checkpoints:
            destination = output / "evaluation" / checkpoint.stem
            destination.mkdir(parents=True, exist_ok=True)
            state["current_checkpoint"] = str(checkpoint)
            write_json(status_path, state)
            with (destination / "console.log").open("w") as stream:
                subprocess.run(command + ["evaluate", "--config", str(options.config),
                                          "--checkpoint", str(checkpoint)],
                               cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            comparison = json.loads((destination / "comparison.json").read_text())
            comparisons.append(comparison)
            state["evaluated"].append(comparison)
            write_json(status_path, state)
        best = min(comparisons, key=lambda item: item["worst_relative_ratio"])
        write_json(output / "summary.json", {
            "config": config, "candidates": comparisons, "best": best,
            "any_all_five_better": any(c["all_five_better"] for c in comparisons),
            "validation_role": "development selection on the frozen validation_268",
        })
        state.update(status="completed", ended_at=timestamp(), best=best)
        write_json(status_path, state)
    except Exception as error:
        state.update(status="failed", ended_at=timestamp(), error=repr(error))
        write_json(status_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "run"):
        child = subparsers.add_parser(name)
        child.add_argument("--config", type=Path, required=True)
        if name == "evaluate":
            child.add_argument("--checkpoint", type=Path, required=True)
    options = parser.parse_args()
    {"train": train, "evaluate": evaluate, "run": run}[options.command](options)


if __name__ == "__main__":
    main()
