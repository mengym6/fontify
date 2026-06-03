#!/usr/bin/env bash
set -euo pipefail

CSV_PATH=${CSV_PATH:-random_search_runs.csv}
STAGE=${STAGE:-short}
RUN_IDS=${RUN_IDS:-all}
DATA_PATH=${DATA_PATH:-fontdata_example}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-models/vit_base_font/checkpoint-14.pth}
BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-models/random_search}
PYTHON_BIN=${PYTHON_BIN:-python}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29555}
SAVE_FREQ=${SAVE_FREQ:-10}
EXPORT_VAL_IMAGES=${EXPORT_VAL_IMAGES:-1}
VAL_IMAGE_LIMIT=${VAL_IMAGE_LIMIT:-4}
VAL_EXPORT_TIMEOUT=${VAL_EXPORT_TIMEOUT:-300}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

usage() {
  cat <<'EOF'
Usage:
  STAGE=short RUN_IDS=all bash run_random_search_finetune.sh
  STAGE=mid RUN_IDS=rs01,rs07,baseline bash run_random_search_finetune.sh
  STAGE=full RUN_IDS=rs01,baseline bash run_random_search_finetune.sh

Environment variables:
  CSV_PATH          Default: random_search_runs.csv
  STAGE             short | mid | full
  RUN_IDS           all, or comma-separated ids from the CSV
  DATA_PATH         Default: fontdata_example
  PRETRAIN_CKPT     Default: models/vit_base_font/checkpoint-14.pth
  BASE_OUTPUT_DIR   Default: models/random_search
  NPROC_PER_NODE    Default: 2
  MASTER_PORT_BASE  Default: 29555
  SAVE_FREQ         Default: 10
  EXPORT_VAL_IMAGES Default: 1
  VAL_IMAGE_LIMIT   Default: 4
  VAL_EXPORT_TIMEOUT Default: 300 seconds; 0 disables timeout; ignored when GNU timeout is unavailable

The CSV controls --batch_size and target_effective_batch.
The launcher computes accum_iter = target_effective_batch / (batch_size * NPROC_PER_NODE).
All runs use --semantic_only_epochs 0, so JT and BF train together from epoch 0.
This launcher intentionally does not pass --auto_resume.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$STAGE" != "short" && "$STAGE" != "mid" && "$STAGE" != "full" ]]; then
  echo "STAGE must be one of: short, mid, full" >&2
  exit 2
fi

if [[ ! -f "$CSV_PATH" ]]; then
  echo "CSV not found: $CSV_PATH" >&2
  exit 2
fi

mkdir -p "$BASE_OUTPUT_DIR"

selected_run() {
  local run_id="$1"
  if [[ "$RUN_IDS" == "all" ]]; then
    return 0
  fi
  local needle=",${run_id},"
  local haystack=",${RUN_IDS},"
  [[ "$haystack" == *"$needle"* ]]
}

run_index=0

while IFS=, read -r run_id batch_size target_effective_batch lr weight_decay layer_decay freeze_blocks drop_path short_epochs mid_epochs full_epochs; do
  if [[ "$run_id" == "run_id" || -z "$run_id" ]]; then
    continue
  fi
  if ! selected_run "$run_id"; then
    continue
  fi

  case "$STAGE" in
    short)
      epochs="$short_epochs"
      ;;
    mid)
      epochs="$mid_epochs"
      ;;
    full)
      epochs="$full_epochs"
      ;;
  esac
  semantic_only_epochs=0

  per_accum_batch=$((batch_size * NPROC_PER_NODE))
  if (( target_effective_batch % per_accum_batch != 0 )); then
    echo "Cannot derive an integer accum_iter for ${run_id}: target_effective_batch=${target_effective_batch}, batch_size=${batch_size}, NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
    exit 2
  fi
  accum_iter=$((target_effective_batch / per_accum_batch))
  if (( accum_iter < 1 )); then
    echo "Invalid accum_iter for ${run_id}: ${accum_iter}" >&2
    exit 2
  fi
  effective_batch_size=$((batch_size * accum_iter * NPROC_PER_NODE))
  master_port=$((MASTER_PORT_BASE + run_index))
  val_image_epoch=$((epochs - 1))
  run_name="${run_id}_eb${effective_batch_size}_bs${batch_size}_acc${accum_iter}_ep${epochs}_sem0"
  output_dir="${BASE_OUTPUT_DIR}/${STAGE}/${run_name}"
  log_dir="${output_dir}/logs"
  mkdir -p "$output_dir" "$log_dir"

  cat > "${output_dir}/run_params.txt" <<EOF
run_id=${run_id}
stage=${STAGE}
epochs=${epochs}
semantic_only_epochs=${semantic_only_epochs}
curriculum=jt_bf_sync_from_epoch0
batch_size=${batch_size}
target_effective_batch=${target_effective_batch}
accum_iter=${accum_iter}
effective_batch_size=${effective_batch_size}
lr=${lr}
weight_decay=${weight_decay}
layer_decay=${layer_decay}
freeze_blocks=${freeze_blocks}
drop_path=${drop_path}
data_path=${DATA_PATH}
pretrain_ckpt=${PRETRAIN_CKPT}
output_dir=${output_dir}
log_dir=${log_dir}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
nproc_per_node=${NPROC_PER_NODE}
master_port=${master_port}
val_tb_image_limit=${VAL_IMAGE_LIMIT}
val_image_epoch=${val_image_epoch}
EOF

  echo "==== Running ${run_id} (${STAGE}) ===="
  echo "output_dir=${output_dir}"
  echo "effective_batch_size=${effective_batch_size}"

  (
    set -x
    "$PYTHON_BIN" -m torch.distributed.launch \
      --nproc_per_node="$NPROC_PER_NODE" \
      --master_port="$master_port" \
      --use_env main_train.py \
      --batch_size "$batch_size" \
      --accum_iter "$accum_iter" \
      --model vit_base_patch16_input896x448_win_dec64_8glb_sl1 \
      --num_mask_patches 784 \
      --max_mask_patches_per_block 392 \
      --epochs "$epochs" \
      --warmup_epochs 1 \
      --lr "$lr" \
      --weight_decay "$weight_decay" \
      --layer_decay "$layer_decay" \
      --drop_path "$drop_path" \
      --clip_grad 1.0 \
      --input_size 896 448 \
      --save_freq "$SAVE_FREQ" \
      --val_tb_image_limit "$VAL_IMAGE_LIMIT" \
      --data_path "${DATA_PATH}/" \
      --json_path "${DATA_PATH}/train_json_new/"*.json \
      --val_json_path "${DATA_PATH}/val_json_new/"*.json \
      --output_dir "$output_dir" \
      --log_dir "$log_dir" \
      --finetune "$PRETRAIN_CKPT" \
      --freeze_encoder \
      --freeze_blocks "$freeze_blocks" \
      --semantic_mask_dir "${DATA_PATH}/font/train/new" \
      --num_mask_annotations_bf 3 \
      --num_mask_annotations_jt 1 \
      --mask_coverage_threshold 0.1 \
      --semantic_only_epochs "$semantic_only_epochs"
  ) 2>&1 | tee "${output_dir}/train.log"

  if [[ "$EXPORT_VAL_IMAGES" == "1" ]]; then
    echo "==== Exporting ${VAL_IMAGE_LIMIT} validation image(s) for ${run_id} ===="
    export_cmd=(
      "$PYTHON_BIN" tools/export_tb_val_images.py
      --log_dir "$log_dir" \
      --output_dir "${output_dir}/val_images" \
      --limit "$VAL_IMAGE_LIMIT" \
      --epoch "$val_image_epoch"
    )
    if [[ "$VAL_EXPORT_TIMEOUT" != "0" ]] && command -v timeout >/dev/null 2>&1; then
      export_cmd=(timeout "$VAL_EXPORT_TIMEOUT" "${export_cmd[@]}")
    fi
    if "${export_cmd[@]}" 2>&1 | tee "${output_dir}/export_val_images.log"; then
      echo "val_images=${output_dir}/val_images"
    else
      echo "Warning: failed to export validation images for ${run_id}" >&2
    fi
  fi

  run_index=$((run_index + 1))
done < "$CSV_PATH"
