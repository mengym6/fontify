#!/usr/bin/env bash
set -euo pipefail

CSV_PATH=${CSV_PATH:-random_search_runs.csv}
RUN_IDS=${RUN_IDS:-all}
DATA_PATH=${DATA_PATH:-fontdata_example}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-models/vit_base_font/checkpoint-14.pth}
BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-models/random_search_three_stage}
PYTHON_BIN=${PYTHON_BIN:-python}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29555}
SAVE_FREQ=${SAVE_FREQ:-10}
EXPORT_VAL_IMAGES=${EXPORT_VAL_IMAGES:-1}
VAL_IMAGE_LIMIT=${VAL_IMAGE_LIMIT:-4}
VAL_TB_IMAGE_FREQ=${VAL_TB_IMAGE_FREQ:-5}
VAL_EXPORT_TIMEOUT=${VAL_EXPORT_TIMEOUT:-300}
STAGE1_EPOCHS=${STAGE1_EPOCHS:-30}
STAGE2_EPOCHS=${STAGE2_EPOCHS:-30}
STAGE3_EPOCHS=${STAGE3_EPOCHS:-30}
DROP_PATH=${DROP_PATH:-0.10}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

usage() {
  cat <<'EOF'
Usage:
  RUN_IDS=all bash run_random_search_finetune.sh
  RUN_IDS=baseline,rs01 bash run_random_search_finetune.sh

Environment variables:
  CSV_PATH          Default: random_search_runs.csv
  RUN_IDS           all, or comma-separated ids from the CSV
  DATA_PATH         Default: fontdata_example
  PRETRAIN_CKPT     Default: models/vit_base_font/checkpoint-14.pth
  BASE_OUTPUT_DIR   Default: models/random_search_three_stage
  NPROC_PER_NODE    Default: 2
  MASTER_PORT_BASE  Default: 29555
  SAVE_FREQ         Default: 10
  EXPORT_VAL_IMAGES Default: 1
  VAL_IMAGE_LIMIT   Default: 4
  VAL_TB_IMAGE_FREQ  Default: 5, write validation TensorBoard images every 5 epochs
  VAL_EXPORT_TIMEOUT Default: 300 seconds; 0 disables timeout; ignored when GNU timeout is unavailable
  STAGE1_EPOCHS     Default: 30
  STAGE2_EPOCHS     Default: 30
  STAGE3_EPOCHS     Default: 30
  DROP_PATH          Default: 0.10, fixed for all random-search runs

CSV columns:
  run_id,batch_size,target_effective_batch,lr,weight_decay,layer_decay,
  s1_random,s1_jt,s1_bf,s2_random,s2_jt,s2_bf,s3_random,s3_jt,s3_bf

Loss schedule expected from models_train.py:
  stage1: freeze full encoder; models_train.py keeps edge/adv at 0
  stage2: freeze first 9 blocks; models_train.py makes edge non-zero from epoch 30
  stage3: unfreeze all; models_train.py makes adv non-zero from epoch 60

This launcher intentionally does not pass --auto_resume.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
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

join_mask_name() {
  local a="$1"
  local b="$2"
  local c="$3"
  printf "%s_%s_%s" "$a" "$b" "$c" | tr '.-' 'p_'
}

run_stage() {
  local run_id="$1"
  local stage_name="$2"
  local stage_index="$3"
  local output_dir="$4"
  local resume_ckpt="$5"
  local total_epochs="$6"
  local freeze_blocks="$7"
  local mask_random="$8"
  local mask_jt="$9"
  local mask_bf="${10}"
  local master_port="${11}"
  local batch_size="${12}"
  local accum_iter="${13}"
  local lr="${14}"
  local weight_decay="${15}"
  local layer_decay="${16}"
  local drop_path="${17}"

  local log_dir="${output_dir}/logs"
  local val_image_epoch=$((total_epochs - 1))
  mkdir -p "$output_dir" "$log_dir"

  cat > "${output_dir}/run_params.txt" <<EOF
run_id=${run_id}
stage=${stage_name}
stage_index=${stage_index}
total_epochs=${total_epochs}
freeze_blocks=${freeze_blocks}
mask_mix_probs=${mask_random},${mask_jt},${mask_bf}
batch_size=${batch_size}
accum_iter=${accum_iter}
lr=${lr}
weight_decay=${weight_decay}
layer_decay=${layer_decay}
drop_path=${drop_path}
data_path=${DATA_PATH}
pretrain_ckpt=${PRETRAIN_CKPT}
resume_ckpt=${resume_ckpt}
output_dir=${output_dir}
log_dir=${log_dir}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
nproc_per_node=${NPROC_PER_NODE}
master_port=${master_port}
val_tb_image_limit=${VAL_IMAGE_LIMIT}
val_tb_image_freq=${VAL_TB_IMAGE_FREQ}
val_image_epoch=${val_image_epoch}
EOF

  local cmd=(
    "$PYTHON_BIN" -m torch.distributed.launch
    --nproc_per_node="$NPROC_PER_NODE"
    --master_port="$master_port"
    --use_env main_train.py
    --batch_size "$batch_size"
    --accum_iter "$accum_iter"
    --model vit_base_patch16_input896x448_win_dec64_8glb_sl1
    --num_mask_patches 784
    --max_mask_patches_per_block 392
    --epochs "$total_epochs"
    --warmup_epochs 1
    --lr "$lr"
    --weight_decay "$weight_decay"
    --layer_decay "$layer_decay"
    --drop_path "$drop_path"
    --clip_grad 1.0
    --input_size 896 448
    --save_freq "$SAVE_FREQ"
    --val_tb_image_limit "$VAL_IMAGE_LIMIT"
    --val_tb_image_freq "$VAL_TB_IMAGE_FREQ"
    --data_path "${DATA_PATH}/"
    --json_path "${DATA_PATH}/train_json_new/"*.json
    --val_json_path "${DATA_PATH}/val_json_new/"*.json
    --output_dir "$output_dir"
    --log_dir "$log_dir"
    --semantic_mask_dir "${DATA_PATH}/font/train/new"
    --num_mask_annotations_bf 3
    --num_mask_annotations_jt 1
    --mask_coverage_threshold 0.1
    --mask_mix_probs "$mask_random" "$mask_jt" "$mask_bf"
  )

  if [[ -n "$resume_ckpt" ]]; then
    cmd+=(--resume "$resume_ckpt")
  else
    cmd+=(--finetune "$PRETRAIN_CKPT")
  fi

  if [[ "$freeze_blocks" != "none" ]]; then
    cmd+=(--freeze_encoder --freeze_blocks "$freeze_blocks")
  fi

  echo "==== Running ${run_id} ${stage_name} ===="
  echo "output_dir=${output_dir}"
  echo "mask_mix_probs=${mask_random},${mask_jt},${mask_bf}"
  echo "freeze_blocks=${freeze_blocks}"

  (
    set -x
    "${cmd[@]}"
  ) 2>&1 | tee "${output_dir}/train.log"

  if [[ "$EXPORT_VAL_IMAGES" == "1" ]]; then
    echo "==== Exporting ${VAL_IMAGE_LIMIT} validation image(s) for ${run_id} ${stage_name} ===="
    local export_cmd=(
      "$PYTHON_BIN" tools/export_tb_val_images.py
      --log_dir "$log_dir"
      --output_dir "${output_dir}/val_images"
      --limit "$VAL_IMAGE_LIMIT"
      --epoch "$val_image_epoch"
    )
    if [[ "$VAL_EXPORT_TIMEOUT" != "0" ]] && command -v timeout >/dev/null 2>&1; then
      export_cmd=(timeout "$VAL_EXPORT_TIMEOUT" "${export_cmd[@]}")
    fi
    if "${export_cmd[@]}" 2>&1 | tee "${output_dir}/export_val_images.log"; then
      echo "val_images=${output_dir}/val_images"
    else
      echo "Warning: failed to export validation images for ${run_id} ${stage_name}" >&2
    fi
  fi
}

run_index=0
stage1_total_epochs=$STAGE1_EPOCHS
stage2_total_epochs=$((STAGE1_EPOCHS + STAGE2_EPOCHS))
stage3_total_epochs=$((STAGE1_EPOCHS + STAGE2_EPOCHS + STAGE3_EPOCHS))

while IFS=, read -r run_id batch_size target_effective_batch lr weight_decay layer_decay \
  s1_random s1_jt s1_bf s2_random s2_jt s2_bf s3_random s3_jt s3_bf; do
  if [[ "$run_id" == "run_id" || -z "$run_id" ]]; then
    continue
  fi
  if ! selected_run "$run_id"; then
    continue
  fi

  per_accum_batch=$((batch_size * NPROC_PER_NODE))
  if (( target_effective_batch % per_accum_batch != 0 )); then
    echo "Cannot derive integer accum_iter for ${run_id}: target_effective_batch=${target_effective_batch}, batch_size=${batch_size}, NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
    exit 2
  fi
  accum_iter=$((target_effective_batch / per_accum_batch))
  if (( accum_iter < 1 )); then
    echo "Invalid accum_iter for ${run_id}: ${accum_iter}" >&2
    exit 2
  fi
  effective_batch_size=$((batch_size * accum_iter * NPROC_PER_NODE))

  mask_tag_s1=$(join_mask_name "$s1_random" "$s1_jt" "$s1_bf")
  mask_tag_s2=$(join_mask_name "$s2_random" "$s2_jt" "$s2_bf")
  mask_tag_s3=$(join_mask_name "$s3_random" "$s3_jt" "$s3_bf")
  run_name="${run_id}_eb${effective_batch_size}_bs${batch_size}_acc${accum_iter}_lr${lr}_wd${weight_decay}_ld${layer_decay}_m${mask_tag_s1}-${mask_tag_s2}-${mask_tag_s3}"
  run_root="${BASE_OUTPUT_DIR}/${run_name}"
  mkdir -p "$run_root"

  cat > "${run_root}/run_params.txt" <<EOF
run_id=${run_id}
stage_epochs=${STAGE1_EPOCHS},${STAGE2_EPOCHS},${STAGE3_EPOCHS}
effective_batch_size=${effective_batch_size}
batch_size=${batch_size}
accum_iter=${accum_iter}
lr=${lr}
weight_decay=${weight_decay}
layer_decay=${layer_decay}
drop_path=${DROP_PATH}
stage1_mask_mix=${s1_random},${s1_jt},${s1_bf}
stage2_mask_mix=${s2_random},${s2_jt},${s2_bf}
stage3_mask_mix=${s3_random},${s3_jt},${s3_bf}
stage1_loss=edge_off_adv_off
stage2_loss=edge_warmup_adv_off
stage3_loss=edge_on_adv_warmup
stage1_freeze=full_encoder
stage2_freeze=blocks_0_9
stage3_freeze=none
EOF

  stage1_dir="${run_root}/stage1"
  stage2_dir="${run_root}/stage2"
  stage3_dir="${run_root}/stage3"

  stage1_port=$((MASTER_PORT_BASE + run_index * 10 + 1))
  stage2_port=$((MASTER_PORT_BASE + run_index * 10 + 2))
  stage3_port=$((MASTER_PORT_BASE + run_index * 10 + 3))

  run_stage "$run_id" "stage1" 1 "$stage1_dir" "" \
    "$stage1_total_epochs" "-1" \
    "$s1_random" "$s1_jt" "$s1_bf" "$stage1_port" \
    "$batch_size" "$accum_iter" "$lr" "$weight_decay" "$layer_decay" "$DROP_PATH"

  stage1_ckpt="${stage1_dir}/checkpoint-$((stage1_total_epochs - 1)).pth"
  if [[ ! -f "$stage1_ckpt" ]]; then
    echo "Missing stage1 checkpoint: $stage1_ckpt" >&2
    exit 3
  fi

  run_stage "$run_id" "stage2" 2 "$stage2_dir" "$stage1_ckpt" \
    "$stage2_total_epochs" "9" \
    "$s2_random" "$s2_jt" "$s2_bf" "$stage2_port" \
    "$batch_size" "$accum_iter" "$lr" "$weight_decay" "$layer_decay" "$DROP_PATH"

  stage2_ckpt="${stage2_dir}/checkpoint-$((stage2_total_epochs - 1)).pth"
  if [[ ! -f "$stage2_ckpt" ]]; then
    echo "Missing stage2 checkpoint: $stage2_ckpt" >&2
    exit 3
  fi

  run_stage "$run_id" "stage3" 3 "$stage3_dir" "$stage2_ckpt" \
    "$stage3_total_epochs" "none" \
    "$s3_random" "$s3_jt" "$s3_bf" "$stage3_port" \
    "$batch_size" "$accum_iter" "$lr" "$weight_decay" "$layer_decay" "$DROP_PATH"

  run_index=$((run_index + 1))
done < "$CSV_PATH"
