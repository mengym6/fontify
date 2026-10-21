"""Durable strict Run B preparation, training, and final evaluation controller."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "fontdata_example"
PROTOCOL = ROOT / "models/dual_view_strict"
OUTPUT = ROOT / "models/runb_strict"
CACHE = DATA / "dual_targets_strict"
MANIFEST = DATA / "train_json_dual_strict/train_dual.json"
FIXED = DATA / "val_pairs_fixed_strict.json"
PARENT = ROOT / "models/vit_base_font/checkpoint-14.pth"
BASELINE = Path(
    "/root/autodl-tmp/fontify_baseline/models/"
    "finetune_no_gan_nojt_no_freeze_baseline_vggfix/checkpoint-50.pth"
)
CTRL = Path(
    "/root/autodl-tmp/fontify/models/finetune_jieti_ctrl_g12/checkpoint-best.pth"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-controller", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if PROTOCOL.exists() and not args.resume_controller:
        raise FileExistsError(PROTOCOL)
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    lock = (PROTOCOL / "controller.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = PROTOCOL / "status.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "completed": [],
            "output": str(OUTPUT),
            "protocol": str(PROTOCOL),
        }
    )
    checkpoint_hash = hashlib.sha256()
    with PARENT.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            checkpoint_hash.update(block)
    expected = "90df3369c8187c3ab6bd9e8a6f7af812eafc87c3657e2f4eff7879f578d99258"
    if checkpoint_hash.hexdigest() != expected:
        raise ValueError("Initialization differs from the historical Run B parent")
    state["parent_checkpoint_sha256"] = expected
    state["source_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    state["configuration"] = {
        "epochs": 51, "seed": 0, "effective_batch": 64,
        "physical_batch_per_gpu": 2, "world_size": 2, "accum_iter": 16,
        "lr": 0.00195699, "warmup_epochs": 10,
        "style_weight": 14.73, "edge_weight_final": 0.2,
        "native_geometry": "pad_square", "reference_policy":
        "distinct-original-reference-v1",
        "historical_comparators_are_not_strict_retrained_baselines": True,
    }
    env = dict(
        os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1"
    )
    for key in ("TIP_WEIGHT", "TIP_CALIBRATION"):
        env.pop(key, None)
    python = sys.executable
    val = [str(path) for path in sorted((DATA / "val_json_strict").glob("*.json"))]
    if len(val) != 18:
        raise ValueError("Missing strict validation lists")

    def save(stage, status, **extra):
        state.update(
            stage=stage,
            status=status,
            pid=os.getpid(),
            time=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            **extra,
        )
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        with (PROTOCOL / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(state) + "\n")

    def run(stage, command, gpu="0", extra_env=None):
        if stage in state["completed"]:
            return
        command = list(map(str, command))
        save(stage, "running", command=command)
        current_env = dict(env, CUDA_VISIBLE_DEVICES=gpu, **(extra_env or {}))
        log = PROTOCOL / f"{stage}.log"
        if log.exists():
            log = PROTOCOL / f"{stage}_retry_{datetime.now().strftime('%H%M%S')}.log"
        with log.open("x") as stream:
            result = subprocess.run(
                command, env=current_env, stdout=stream, stderr=subprocess.STDOUT
            )
        if result.returncode:
            save(stage, "failed", returncode=result.returncode)
            raise SystemExit(result.returncode)
        state["completed"].append(stage)
        save(stage, "complete", returncode=0)

    common = ["--data-path", DATA, "--json-path", MANIFEST, "--target-cache", CACHE]
    eval_common = [
        "--data_path",
        DATA,
        "--val_json_path",
        *val,
        "--fixed_pair_path",
        FIXED,
        "--dual_target_cache",
        CACHE,
    ]
    coef = PROTOCOL / "eval_coef.json"
    denominator = PROTOCOL / "s_baseline_dual.json"
    samples = PROTOCOL / "probe_samples.json"
    warm = PROTOCOL / "head-warmup.pth"
    calibration = PROTOCOL / "train_coef.json"
    run(
        "cpu_tests",
        [
            python,
            "-m",
            "pytest",
            "-q",
            "tests/test_dual_data.py",
            "tests/test_dual_protocol.py",
            "tests/test_dual_runtime_tools.py",
            "tests/test_runb_strict.py",
        ],
    )
    run(
        "prepare_data",
        [
            python,
            "tools/prepare_runb_strict.py",
            "--data",
            DATA,
            "--protocol",
            PROTOCOL,
        ],
    )
    run(
        "build_cache",
        [
            python,
            "tools/build_dual_targets.py",
            "--data-path",
            DATA,
            "--json-paths",
            MANIFEST,
            *val,
            "--fixed-pair-path",
            FIXED,
            "--output-dir",
            CACHE,
            "--native-geometry",
            "pad_square",
            "--strict-reference",
            "--bf-audit-source",
            DATA / "dual_targets_v1",
            "--bf-audit-metadata-sha256",
            "c9e3eec178e2df0dc648058a508ac45f5cc4c77556a821fd26c9e3a1c78486b6",
        ],
    )
    run(
        "samples",
        [
            python,
            "tools/dual_view_protocol.py",
            "samples",
            *common,
            "--output",
            samples,
        ],
    )
    run(
        "eval_calibration",
        [
            python,
            "tools/calibrate_dual_eval.py",
            "--checkpoint",
            CTRL,
            *eval_common,
            "--output_json",
            coef,
        ],
    )
    legacy = ROOT / "models/dual_view_pad_20261014/legacy_seed0_reference.json"
    old_val = sorted((DATA / "val_json_new").glob("*.json"))
    run(
        "denominator",
        [
            python,
            "tools/eval_s_baseline.py",
            "--checkpoint",
            BASELINE,
            *eval_common,
            "--calibration_json",
            coef,
            "--j_definition",
            "instance-lower-full-v1",
            "--as_dual_denominator",
            "--legacy_val_json_path",
            *old_val,
            "--legacy_fixed_pair_path",
            DATA / "val_pairs_fixed.json",
            "--legacy_l1_reference_json",
            legacy,
            "--output_json",
            denominator,
        ],
    )
    run(
        "ctrl_evaluation",
        [
            python,
            "tools/eval_s_baseline.py",
            "--checkpoint",
            CTRL,
            *eval_common,
            "--calibration_json",
            coef,
            "--j_definition",
            "instance-lower-full-v1",
            "--s_baseline_path",
            denominator,
            "--output_json",
            PROTOCOL / "ctrl_best_eval.json",
        ],
    )
    run(
        "blind_manifest",
        [
            python,
            "tools/dual_report.py",
            "blind-manifest",
            "--evaluation",
            denominator,
            "--models",
            "run_b",
            "baseline_vggfix",
            "ctrl_g12",
            "--seed",
            "0",
            "--limit",
            "40",
            "--output",
            PROTOCOL / "blind_manifest.json",
        ],
    )
    run(
        "head_warmup",
        [
            python,
            "tools/dual_view_protocol.py",
            "warmup",
            *common,
            "--samples",
            samples,
            "--checkpoint",
            PARENT,
            "--batch-size",
            "2",
            "--output",
            warm,
        ],
        gpu="1",
    )
    run(
        "training_calibration",
        [
            python,
            "tools/dual_view_protocol.py",
            "calibrate",
            *common,
            "--samples",
            samples,
            "--checkpoint",
            warm,
            "--parent-checkpoint",
            PARENT,
            "--val-json-path",
            *val,
            "--fixed-pair-path",
            FIXED,
            "--eval-coef",
            coef,
            "--s-baseline",
            denominator,
            "--output",
            calibration,
        ],
    )
    run(
        "warm_end_probe",
        [
            python,
            "tools/dual_gradient_probe.py",
            "--point",
            "warm_end",
            "--checkpoint",
            warm,
            "--calibration",
            calibration,
            "--samples",
            samples,
            *common,
            "--output",
            PROTOCOL / "probe_warm_end.json",
        ],
    )
    run(
        "ddp_smoke",
        [
            python,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--master_port=29631",
            "tools/smoke_dual_view.py",
            *common,
            "--checkpoint",
            warm,
            "--train-calibration",
            calibration,
            "--batch-size",
            "2",
            "--iterations",
            "2",
            "--output-json",
            PROTOCOL / "smoke.json",
        ],
        gpu="0,1",
    )
    run(
        "training",
        ["bash", "finetune_dual_view_runb.sh"],
        gpu="0,1",
        extra_env={
            "PYTHON": python,
            "NPROC": "2",
            "BATCH_SIZE": "2",
            "MASTER_PORT": "29632",
            "DATA_PATH": str(DATA),
            "VAL_JSON_DIR": str(DATA / "val_json_strict"),
            "TRAIN_MANIFEST": str(MANIFEST),
            "TARGET_CACHE": str(CACHE),
            "FIXED_PAIRS": str(FIXED),
            "PARENT_CHECKPOINT": str(PARENT),
            "DUAL_ROOT": str(PROTOCOL),
            "OUTPUT_DIR": str(OUTPUT),
        },
    )
    for point in ("ep2", "ep25", "ep50"):
        epoch = point[2:]
        checkpoint = OUTPUT / (
            "checkpoint-50.pth" if epoch == "50" else f"checkpoint-probe-{epoch}.pth"
        )
        run(
            f"probe_{point}",
            [
                python,
                "tools/dual_gradient_probe.py",
                "--point",
                point,
                "--checkpoint",
                checkpoint,
                "--calibration",
                calibration,
                "--samples",
                samples,
                *common,
                "--output",
                PROTOCOL / f"probe_{point}.json",
            ],
        )
    for point in ("best", "50"):
        run(
            f"eval_{point}",
            [
                python,
                "tools/eval_s_baseline.py",
                "--checkpoint",
                OUTPUT / f"checkpoint-{point}.pth",
                *eval_common,
                "--calibration_json",
                coef,
                "--j_definition",
                "instance-lower-full-v1",
                "--s_baseline_path",
                denominator,
                "--output_json",
                PROTOCOL / f"run_b_{point}_eval.json",
            ],
        )
    run(
        "clipping",
        [
            python,
            "tools/dual_report.py",
            "clipping",
            "--log",
            OUTPUT / "dual_diagnostics.jsonl",
            "--output",
            PROTOCOL / "clipping.json",
        ],
    )
    save("automatic_checks_finished", "complete")


if __name__ == "__main__":
    main()
