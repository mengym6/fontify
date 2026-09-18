#!/bin/bash

set -euo pipefail

EXP_ID=${1:-1}

case "$EXP_ID" in
    1)
        DETAIL_GRADIENT_RATIO=0.0
        DETAIL_NORMALIZE_FLAG=()
        ;;
    2)
        DETAIL_GRADIENT_RATIO=0.1
        DETAIL_NORMALIZE_FLAG=()
        ;;
    3)
        DETAIL_GRADIENT_RATIO=0.1
        DETAIL_NORMALIZE_FLAG=(--detail_per_sample_normalize)
        ;;
    *)
        echo "Usage: $0 {1|2|3}" >&2
        exit 1
        ;;
esac

export CUDA_VISIBLE_DEVICES=0,1

DATA_PATH=fontdata_example
name="finetune_stele_loss_diag_exp${EXP_ID}"
PRETRAIN_CKPT=models/vit_base_font/checkpoint-14.pth

DETAIL_LOSS_WEIGHT=0.05
STRUCTURE_LOSS_WEIGHT=0.05

python -m torch.distributed.launch --nproc_per_node=2 --master_port=29555 \
    --use_env main_train.py \
    --batch_size 2 \
    --accum_iter 32 \
    --model vit_base_patch16_input896x448_win_dec64_8glb_sl1 \
    --num_mask_patches 784 \
    --max_mask_patches_per_block 392 \
    --epochs 35 \
    --warmup_epochs 5 \
    --lr 1e-3 \
    --clip_grad 3.0 \
    --layer_decay 0.8 \
    --drop_path 0.1 \
    --input_size 896 448 \
    --augmentation_policy finetune \
    --adv_warmup_epochs 8 \
    --edge_warmup_epochs 8 \
    --loss_warmup_duration 8 \
    --adv_weight_final 0.4 \
    --edge_weight_final 0.3 \
    --no_gan \
    --structure_loss_weight "${STRUCTURE_LOSS_WEIGHT}" \
    --structure_warmup_epochs 6 \
    --structure_warmup_duration 6 \
    --detail_loss_weight "${DETAIL_LOSS_WEIGHT}" \
    --detail_warmup_epochs 4 \
    --detail_warmup_duration 4 \
    --detail_kernel_size 5 \
    --detail_sigma 1.0 \
    --detail_gradient_ratio "${DETAIL_GRADIENT_RATIO}" \
    "${DETAIL_NORMALIZE_FLAG[@]}" \
    --save_freq 5 \
    --seed 0 \
    --data_path "$DATA_PATH/" \
    --json_path "$DATA_PATH"/train_json_mix/*.json \
    --val_json_path "$DATA_PATH"/val_json_mix/*.json \
    --output_dir "models/$name" \
    --log_dir "models/$name/logs" \
    --finetune "$PRETRAIN_CKPT" \
    --auto_resume \
    --freeze_encoder \
    --freeze_blocks 9 \
    --semantic_mask_dir "$DATA_PATH"/font/train/new \
    --num_mask_annotations_bf 11 \
    --num_mask_annotations_jt 1 \
    --mask_coverage_threshold 0.1 \
    --semantic_only_epochs 0 \
    --val_tb_image_limit 76 \
    --val_tb_images_per_batch 2 \
    --grad_log_interval 5 \
    #--mask_mix_probs 0.8 0.0 0.2
