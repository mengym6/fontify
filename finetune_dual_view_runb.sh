#!/usr/bin/env bash
# Run B starts only after independent warm-up, both calibrations and new S.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
PYTHON=${PYTHON:-python}
NPROC=${NPROC:-4}
BATCH_SIZE=${BATCH_SIZE:-2}
MASTER_PORT=${MASTER_PORT:-29557}
DATA_PATH=${DATA_PATH:-fontdata_example}
VAL_JSON_DIR=${VAL_JSON_DIR:-$DATA_PATH/val_json_clean_v2}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$DATA_PATH/train_json_dual_v1/train_dual.json}
TARGET_CACHE=${TARGET_CACHE:-$DATA_PATH/dual_targets_v1}
FIXED_PAIRS=${FIXED_PAIRS:-$DATA_PATH/val_pairs_fixed_clean.json}
PARENT_CHECKPOINT=${PARENT_CHECKPOINT:-models/vit_base_font/checkpoint-14.pth}
DUAL_ROOT=${DUAL_ROOT:-models/dual_view}
WARM_CHECKPOINT=${WARM_CHECKPOINT:-$DUAL_ROOT/head-warmup.pth}
FIXED_SAMPLES=${FIXED_SAMPLES:-$DUAL_ROOT/probe_samples.json}
TRAIN_CALIBRATION=${TRAIN_CALIBRATION:-$DUAL_ROOT/train_coef.json}
EVAL_CALIBRATION=${EVAL_CALIBRATION:-$DUAL_ROOT/eval_coef.json}
S_BASELINE=${S_BASELINE:-$DUAL_ROOT/s_baseline_dual.json}
BLIND_MANIFEST=${BLIND_MANIFEST:-$DUAL_ROOT/blind_manifest.json}
OUTPUT_DIR=${OUTPUT_DIR:-models/finetune_dual_view_runb}
DETAIL_ARGS=()
EDGE_FINAL=0.2
if [[ -n "${TIP_WEIGHT:-}" ]]; then
    : "${TIP_CALIBRATION:?TIP_CALIBRATION is required with TIP_WEIGHT}"
    DETAIL_ARGS=(--tip_weight "$TIP_WEIGHT" --tip_calibration_json "$TIP_CALIBRATION")
    EDGE_FINAL=0.0
fi

if ! [[ "$NPROC" =~ ^[1-9][0-9]*$ && "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo 'NPROC and BATCH_SIZE must be positive integers' >&2
    exit 2
fi
if (( 64 % (NPROC * BATCH_SIZE) != 0 )); then
    echo 'NPROC * BATCH_SIZE must divide effective batch 64' >&2
    exit 2
fi
ACCUM_ITER=$((64 / (NPROC * BATCH_SIZE)))
shopt -s nullglob
VAL_MANIFESTS=("$VAL_JSON_DIR"/*.json)
if (( ${#VAL_MANIFESTS[@]} == 0 )); then
    echo 'No clean v2 validation manifests found' >&2
    exit 2
fi
for prerequisite in "$TRAIN_MANIFEST" "$PARENT_CHECKPOINT" "$WARM_CHECKPOINT" \
    "$FIXED_SAMPLES" "$TRAIN_CALIBRATION" "$EVAL_CALIBRATION" \
    "$S_BASELINE" "$FIXED_PAIRS" "$BLIND_MANIFEST"; do
    if [[ ! -f "$prerequisite" ]]; then
        echo "Missing prerequisite: $prerequisite; follow docs/dual_view_runb.rst" >&2
        exit 2
    fi
done
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Output directory already exists: $OUTPUT_DIR; select a new run directory" >&2
    exit 2
fi

"$PYTHON" tools/dual_view_protocol.py preflight \
    --data-path "$DATA_PATH" --json-path "$TRAIN_MANIFEST" \
    --target-cache "$TARGET_CACHE" --samples "$FIXED_SAMPLES" \
    --checkpoint "$WARM_CHECKPOINT" --parent-checkpoint "$PARENT_CHECKPOINT" \
    --val-json-path "${VAL_MANIFESTS[@]}" --fixed-pair-path "$FIXED_PAIRS" \
    --eval-coef "$EVAL_CALIBRATION" --s-baseline "$S_BASELINE" \
    --calibration "$TRAIN_CALIBRATION"

# Read-only startup checks: GPU bf16, physical card count, blind-list hash and
# conservative space for seven optimizer checkpoints plus two model snapshots.
"$PYTHON" - "$NPROC" "$WARM_CHECKPOINT" "$OUTPUT_DIR" "$BLIND_MANIFEST" "$FIXED_PAIRS" <<'PY'
import json
import shutil
import sys
from pathlib import Path
import torch
from tools.dual_view_protocol import SCHEMA, sha256_file

nproc, warm, output, blind, fixed = sys.argv[1:]
if not torch.cuda.is_available() or torch.cuda.device_count() < int(nproc):
    raise SystemExit('Insufficient currently available CUDA devices')
if not torch.cuda.is_bf16_supported():
    raise SystemExit('CUDA bf16 is required')
value = json.loads(Path(blind).read_text())
if value.get('schema') != SCHEMA or value.get('purpose') != 'fixed_blind_manifest':
    raise SystemExit('A fixed pre-training blind manifest is required')
if value.get('fixed_pair_sha256') != sha256_file(fixed):
    raise SystemExit('Blind manifest belongs to another fixed validation set')
destination = Path(output).resolve().parent
while not destination.exists():
    destination = destination.parent
budget = int(Path(warm).stat().st_size * (7 * 3 + 2) * 1.2)
free = shutil.disk_usage(destination).free
if free < budget:
    raise SystemExit(f'Insufficient checkpoint disk budget: need {budget}, free {free}')
print(json.dumps({'effective_batch': 64, 'gpu_count': torch.cuda.device_count(),
                  'disk_budget_bytes': budget, 'disk_free_bytes': free,
                  'blind_manifest_sha256': sha256_file(blind)}))
PY

"$PYTHON" -m torch.distributed.run --nproc_per_node="$NPROC" \
    --master_port="$MASTER_PORT" main_train.py \
    --dual_view --dual_target_cache "$TARGET_CACHE" \
    --dual_calibration_json "$TRAIN_CALIBRATION" \
    --dual_eval_calibration_json "$EVAL_CALIBRATION" \
    --dual_parent_checkpoint "$PARENT_CHECKPOINT" \
    --dual_sample_manifest "$FIXED_SAMPLES" \
    --dual_blind_manifest "$BLIND_MANIFEST" \
    --batch_size "$BATCH_SIZE" --accum_iter "$ACCUM_ITER" \
    --model vit_base_patch16_input896x448_win_dec64_8glb_sl1 \
    --num_mask_patches 784 --max_mask_patches_per_block 392 \
    --epochs 51 --warmup_epochs 10 --lr 0.00195699 \
    --clip_grad 3.0 --style_weight 14.73 --layer_decay 0.8 --drop_path 0.1 \
    --input_size 896 448 --augmentation_policy finetune \
    --half_mask_ratio 0 --fusion_lambda 0.5 \
    --edge_warmup_epochs 10 --loss_warmup_duration 8 --edge_weight_final "$EDGE_FINAL" \
    "${DETAIL_ARGS[@]}" \
    --save_freq 10 --save_best --seed 0 \
    --data_path "$DATA_PATH" --json_path "$TRAIN_MANIFEST" \
    --val_json_path "${VAL_MANIFESTS[@]}" \
    --output_dir "$OUTPUT_DIR" --log_dir "$OUTPUT_DIR/logs" \
    --finetune "$WARM_CHECKPOINT" --no_gan --no_jt \
    --num_mask_annotations_bf 11 --mask_coverage_threshold 0.1 \
    --jieti_alpha_jt 1.0 --jieti_struct_pair_prob 0 \
    --jieti_soft_fg linear --jieti_k_max 4 --jieti_pool 224 \
    --jieti_valid_ink 200 --jieti_pred_mass_ratio 0.1 \
    --fixed_pair_path "$FIXED_PAIRS" --s_baseline_path "$S_BASELINE" \
    --vis_every_epoch --val_tb_image_limit 76 --val_tb_images_per_batch 2 \
    --grad_log_interval 1
