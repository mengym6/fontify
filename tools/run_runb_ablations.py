"""Train the three Run B ablations and evaluate their saved checkpoints."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VARIANTS = ("no_j", "no_head", "no_injection")
METRICS = ("PSNR", "LPIPS_VGG", "MAE", "SSIM")
S_EVAL_SOURCE = "frozen validation_268 with its baseline S denominator"
S_EVAL_SOURCES = {
    "validation_268": S_EVAL_SOURCE,
    "historical_240": "frozen historical_240 with its baseline S denominator",
}


def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selection_score(training_dir, epoch):
    """Read the exact validation score used to select this epoch's weights."""
    records = [json.loads(line) for line in
               (Path(training_dir) / "log.txt").read_text().splitlines() if line]
    matches = [row for row in records if row["epoch"] == epoch]
    if len(matches) != 1:
        raise ValueError(f"Expected one validation record for epoch {epoch}")
    score = float(matches[0]["test_S"])
    if not math.isfinite(score):
        raise ValueError(f"Nonfinite selection S at epoch {epoch}")
    return score


def evaluation_score(rows, denominator):
    from tools.eval_s_baseline import absolute_s_score

    jt = [row for row in rows if row["kind"] == "JT"]
    bf = [row for row in rows if row["kind"] == "BF"]
    components = {
        "L1_JT": sum(row["L1"] for row in jt) / len(jt),
        "L1_BF": sum(row["L1"] for row in bf) / len(bf),
        "J": sum(row["J"] for row in jt) / len(jt),
    }
    score, _ = absolute_s_score(components, denominator)
    return score, components


def report(options):
    """Report the requested frozen set and retain the selection score."""
    output = options.output_dir.resolve()
    validation_set = getattr(options, "validation_set", "validation_268")
    expected_source = S_EVAL_SOURCES[validation_set]
    rows = []
    checkpoints = [f"checkpoint-{epoch}" for epoch in (9, 19, 29, 39, 49)]
    checkpoints.append("checkpoint-best")
    for variant in (*VARIANTS, "reference_ep66", "reference_fix_ep50"):
        for checkpoint in checkpoints:
            path = output / variant / "evaluation" / checkpoint / "metrics.json"
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            recorded_set = result.get("validation_set") or Path(
                result.get("validation_cases", "")
            ).parent.name or "validation_268"
            if recorded_set != validation_set:
                raise ValueError(f"Unexpected validation set in {path}")
            score = selection_score(Path(result["checkpoint"]).parent,
                                    result["epoch_zero_based"])
            result["S_select"] = score
            result["S_select_source"] = "log.txt test_S, original Run B fixed 240 pairs"
            if result.get("S_source") != expected_source:
                result["S"] = None
                result["S_source"] = f"pending {validation_set} structure evaluation"
            write_json(path, result)
            for group, metrics in result["groups"].items():
                rows.append({
                    "variant": variant, "checkpoint": checkpoint + ".pth",
                    "completed_epochs": result["completed_epochs"],
                    "validation_set": validation_set,
                    "group": group, **metrics,
                    "S": result["S"] if group == "all" else None,
                    "S_select": score if group == "all" else None,
                })
    if rows:
        write_csv(output / "summary.csv", rows)
    print(json.dumps({"summary_rows": len(rows), "selection_S_included": True}), flush=True)


def train(options):
    import main_train

    saved = json.loads(options.hparams.read_text())
    if not saved.get("dual_view") or not saved.get("dual_pixel_mask"):
        raise ValueError("Expected the original Run B Fix Pixel hparams")
    args = argparse.Namespace(**saved)
    args.dual_ablation = options.variant
    args.epochs = 50
    args.start_epoch = 0
    args.resume = ""
    args.auto_resume = False
    args.batch_size = 2
    args.accum_iter = 16
    args.world_size = 2
    args.rank = -1
    args.gpu = -1
    args.local_rank = -1
    args.distributed = False
    args.dist_url = "env://"
    args.save_freq = 10
    args.save_best = True
    args.save_best_only = False
    args.output_dir = str(options.output_dir.resolve())
    args.log_dir = str(options.output_dir.resolve() / "logs")
    args.ablation_source_hparams = str(options.hparams.resolve())
    args.ablation_selection = "minimum original Run B validation S"
    options.output_dir.mkdir(parents=True, exist_ok=True)
    main_train.main(args, None)


def evaluate(options):
    import lpips
    import numpy as np
    import torch
    from skimage.metrics import structural_similarity
    from PIL import Image

    from tools.eval_s_baseline import load_model, make_jieti_loss

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    validation_set = getattr(options, "validation_set", "validation_268")
    expected_count = {"validation_268": 268, "historical_240": 240}[validation_set]
    cases_path = (options.frozen_root / "03_固定验证集"
                  / validation_set / "cases.json")
    cases = json.loads(cases_path.read_text())
    if len(cases) != expected_count or len({c["case_id"] for c in cases}) != expected_count:
        raise ValueError(f"Expected the complete frozen Run B {validation_set}")
    if validation_set == "validation_268" and len({c["target_path"] for c in cases}) != 268:
        raise ValueError("validation_268 must contain 268 distinct target queries")
    visual_ids = set(getattr(options, "visual_case_ids", []) or [])
    if not visual_ids.issubset({c["case_id"] for c in cases}):
        raise ValueError("Requested visual case is absent from the frozen set")
    if visual_ids:
        (options.output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    metric = lpips.LPIPS(net="vgg", version="0.1").to(device).eval()
    model = load_model(str(options.checkpoint)).to(device).eval()
    coefficients = json.loads((options.frozen_root / "05_协议与复现"
                               / "evaluation_coefficients.json").read_text())["coef"]
    structure_loss = make_jieti_loss("instance-lower-full-v1", coefficients).to(device)
    denominator = json.loads((options.frozen_root / "01_数值结果"
                              / validation_set / "S_denominator.json").read_text())
    mean = torch.tensor((0.485, 0.456, 0.406), device=device)[None, :, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device=device)[None, :, None, None]
    rows = []
    with torch.inference_mode():
        for index, case in enumerate(cases):
            with np.load(options.frozen_root / case["pack"]) as pack:
                source = torch.from_numpy(pack["source"].copy())[None].to(device)
                visible = torch.from_numpy(pack["target_visible"].copy())[None].to(device)
                truth = torch.from_numpy(pack["query_gt"].copy())[None].to(device)
                mask = torch.from_numpy(pack["mask"].copy())[None].to(device)
                voro = torch.from_numpy(pack["voro"].astype("int64"))[None].to(device)
                parts = torch.from_numpy(pack["valid_parts"].copy())[None].to(device)
            height = truth.shape[-2]
            full_prediction = model.predict_full(source, visible, mask)
            prediction = full_prediction[:, :, -height:]
            rgb = (prediction * std + mean).clamp(0, 1)
            gt = (truth * std + mean).clamp(0, 1)
            mse = (rgb - gt).square().mean().item()
            predicted_array = rgb[0].permute(1, 2, 0).cpu().numpy()
            target_array = gt[0].permute(1, 2, 0).cpu().numpy()
            if case["case_id"] in visual_ids:
                image = Image.fromarray((predicted_array * 255).clip(0, 255).astype("uint8"))
                image.save(options.output_dir / "predictions" / f"{case['case_id']}.png")
            row = {
                "case_id": case["case_id"], "kind": case["kind"],
                "writer": case["writer"], "character": case["character"],
                "L1": ((prediction - truth).abs().sum() / (truth.numel() + 0.01)).item(),
                "J": None,
                "PSNR": min(120.0, -10 * math.log10(max(mse, 1e-12))),
                "LPIPS_VGG": metric(rgb * 2 - 1, gt * 2 - 1).item(),
                "MAE": (rgb - gt).abs().mean().item(),
                "SSIM": float(structural_similarity(
                    target_array, predicted_array, data_range=1.0, channel_axis=2,
                )),
            }
            if case["kind"] == "JT":
                target = torch.cat((visible[:, :, :height], truth), dim=2)
                value, _, _ = structure_loss(
                    full_prediction, target, voro, parts,
                    torch.ones(1, dtype=torch.bool, device=device),
                )
                row["J"] = value.item()
            if not all(math.isfinite(row[name]) for name in METRICS):
                raise FloatingPointError(f"Nonfinite image metric at {case['case_id']}")
            rows.append(row)
            if (index + 1) % 50 == 0:
                print(f"Evaluated {index + 1}/{len(cases)}", flush=True)
    groups = {}
    for group in ("all", "JT", "BF"):
        selected = rows if group == "all" else [r for r in rows if r["kind"] == group]
        groups[group] = {
            "count": len(selected),
            **{name: sum(r[name] for r in selected) / len(selected) for name in METRICS},
        }
    score, components = evaluation_score(rows, denominator)
    result = {
        "checkpoint": str(options.checkpoint.resolve()),
        "checkpoint_sha256": model._checkpoint_sha256,
        "epoch_zero_based": model._checkpoint_epoch,
        "completed_epochs": model._checkpoint_epoch + 1,
        "variant": model._checkpoint_args.get("dual_ablation", "none"),
        "validation_set": validation_set,
        "validation_cases": str(cases_path.resolve()),
        "visual_case_ids": sorted(visual_ids),
        "image_space": "448x448 float RGB [0,1], before PNG quantization",
        "lpips": {"network": "vgg", "version": "0.1", "input_range": [-1, 1]},
        "groups": groups,
        "S": score, "S_source": S_EVAL_SOURCES[validation_set],
        "S_components": components, "S_denominator": denominator,
        "S_select": selection_score(options.checkpoint.parent, model._checkpoint_epoch),
        "S_select_source": "log.txt test_S, original Run B fixed 240 pairs",
    }
    options.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(options.output_dir / "per_case.csv", rows)
    write_json(options.output_dir / "metrics.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def refresh_s268(options):
    """Complete missing structure metrics after the training queue exits."""
    output = options.output_dir.resolve()
    status_path = output / "s268_status.json"
    state = {"status": "waiting_for_training", "pid": os.getpid(),
             "started_at": timestamp(), "completed": []}

    def update():
        state["updated_at"] = timestamp()
        write_json(status_path, state)

    update()
    try:
        while True:
            pipeline = json.loads((output / "pipeline_status.json").read_text())
            if pipeline["status"] == "failed":
                raise RuntimeError("Training queue failed; inspect its error before refreshing metrics")
            if pipeline["status"] == "completed":
                break
            time.sleep(60)
        state["status"] = "waiting_for_gpu"
        update()
        while True:
            memory = subprocess.check_output(
                ["nvidia-smi", "--id=0", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"], text=True,
            )
            if int(memory.strip()) < 500:
                break
            time.sleep(60)
        if options.reference_checkpoint:
            reference_dir = output / "reference_ep66/evaluation/checkpoint-best"
            existing = reference_dir / "metrics.json"
            if (not existing.exists()
                    or json.loads(existing.read_text()).get("S_source") != S_EVAL_SOURCE):
                state.update(status="evaluating_reference", current=str(reference_dir))
                update()
                evaluate(argparse.Namespace(
                    checkpoint=options.reference_checkpoint,
                    frozen_root=options.frozen_root,
                    output_dir=reference_dir,
                ))
            reference = json.loads(existing.read_text())
            archived = json.loads((options.frozen_root / "01_数值结果"
                                   / "validation_268" / "summary.json").read_text())["final"]
            actual = {"S": reference["S"], **reference["groups"]["all"]}
            state["reference_deltas"] = {
                key: actual[key] - archived[key] for key in ("S", "PSNR", "MAE", "SSIM")
            }
            if any(not math.isclose(actual[key], archived[key], rel_tol=1e-5, abs_tol=1e-5)
                   for key in state["reference_deltas"]):
                raise RuntimeError("Reference ep66 differs from its archived GPU evaluation")
            update()
        for variant in VARIANTS:
            for path in sorted((output / variant / "evaluation").glob("*/metrics.json")):
                result = json.loads(path.read_text())
                if result.get("S_source") == S_EVAL_SOURCE:
                    continue
                state.update(status="evaluating", current=str(path.parent))
                update()
                with (path.parent / "s268_console.log").open("w") as stream:
                    subprocess.run([
                        sys.executable, str(Path(__file__).resolve()), "evaluate",
                        "--checkpoint", result["checkpoint"],
                        "--frozen-root", str(options.frozen_root.resolve()),
                        "--output-dir", str(path.parent),
                    ], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
                state["completed"].append(str(path.parent))
                report(argparse.Namespace(output_dir=output))
                update()
        report(argparse.Namespace(output_dir=output))
        state.update(status="completed", ended_at=timestamp())
        update()
    except Exception as error:
        state.update(status="failed", error=str(error), ended_at=timestamp())
        update()
        raise


def run(options):
    output = options.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "pipeline_status.json"
    if status_path.exists():
        raise FileExistsError(f"An ablation pipeline already exists at {output}")
    state = {
        "status": "running", "started_at": timestamp(), "epochs_per_variant": 50,
        "variants": {name: {"status": "planned"} for name in VARIANTS},
        "best_selection": "minimum original Run B validation S",
        "periodic_epochs_completed": [10, 20, 30, 40, 50],
        "evaluation": "validation_268, PSNR/VGG-LPIPS/MAE/SSIM",
    }

    def update():
        state["updated_at"] = timestamp()
        write_json(status_path, state)

    def execute(command, log_path, stage):
        with log_path.open("w") as stream:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            stage["pid"] = process.pid
            update()
            code = process.wait()
        stage["returncode"] = code
        if code:
            raise RuntimeError(f"Process exited {code}; see {log_path}")

    update()
    try:
        for variant in VARIANTS:
            stage = state["variants"][variant]
            directory = output / variant
            directory.mkdir(exist_ok=False)
            stage.update(status="training", started_at=timestamp(), output=str(directory))
            state["current_variant"] = variant
            update()
            execute([
                sys.executable, "-m", "torch.distributed.run", "--standalone",
                "--nproc_per_node=2", str(Path(__file__).resolve()), "train",
                "--variant", variant, "--hparams", str(options.hparams.resolve()),
                "--output-dir", str(directory),
            ], directory / "console.log", stage)
            logs = [json.loads(line) for line in (directory / "log.txt").read_text().splitlines()]
            if [row["epoch"] for row in logs] != list(range(50)):
                raise RuntimeError(f"Incomplete epoch sequence for {variant}")
            checkpoints = [directory / f"checkpoint-{epoch}.pth" for epoch in (9, 19, 29, 39, 49)]
            checkpoints.append(directory / "checkpoint-best.pth")
            if set(directory.glob("checkpoint*.pth")) != set(checkpoints):
                raise RuntimeError(f"Unexpected checkpoint inventory for {variant}")
            stage.update(status="evaluating", completed_epochs=50,
                         best_epoch=min(logs, key=lambda row: row["test_S"])["epoch"])
            update()
            for checkpoint in checkpoints:
                result_dir = directory / "evaluation" / checkpoint.stem
                result_dir.mkdir(parents=True)
                execute([
                    sys.executable, str(Path(__file__).resolve()), "evaluate",
                    "--checkpoint", str(checkpoint),
                    "--frozen-root", str(options.frozen_root.resolve()),
                    "--output-dir", str(result_dir),
                ], result_dir / "console.log", stage)
                report(argparse.Namespace(output_dir=output))
            stage.update(status="completed", ended_at=timestamp())
            update()
        state.update(status="completed", ended_at=timestamp())
        update()
    except Exception as error:
        state.update(status="failed", error=str(error), ended_at=timestamp())
        if state.get("current_variant"):
            state["variants"][state["current_variant"]]["status"] = "failed"
        update()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    controller = commands.add_parser("run")
    controller.add_argument("--hparams", type=Path, required=True)
    controller.add_argument("--frozen-root", type=Path, required=True)
    controller.add_argument("--output-dir", type=Path, required=True)
    worker = commands.add_parser("train")
    worker.add_argument("--variant", choices=VARIANTS, required=True)
    worker.add_argument("--hparams", type=Path, required=True)
    worker.add_argument("--output-dir", type=Path, required=True)
    evaluator = commands.add_parser("evaluate")
    evaluator.add_argument("--checkpoint", type=Path, required=True)
    evaluator.add_argument("--frozen-root", type=Path, required=True)
    evaluator.add_argument("--output-dir", type=Path, required=True)
    evaluator.add_argument("--validation-set", choices=tuple(S_EVAL_SOURCES), default="validation_268")
    evaluator.add_argument("--visual-case-ids", nargs="+", default=[])
    reporter = commands.add_parser("report")
    reporter.add_argument("--output-dir", type=Path, required=True)
    reporter.add_argument("--validation-set", choices=tuple(S_EVAL_SOURCES), default="validation_268")
    refresher = commands.add_parser("refresh-s268")
    refresher.add_argument("--output-dir", type=Path, required=True)
    refresher.add_argument("--frozen-root", type=Path, required=True)
    refresher.add_argument("--reference-checkpoint", type=Path)
    options = parser.parse_args()
    {"run": run, "train": train, "evaluate": evaluate,
     "report": report, "refresh-s268": refresh_s268}[options.command](options)


if __name__ == "__main__":
    main()
