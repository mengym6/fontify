#!/bin/bash

# Finetune 脚本：用于小数据集（~1200对）微调预训练模型
# 核心改动：降lr、短训练、弱化判别器、关闭数据增强中的颜色抖动

export CUDA_VISIBLE_DEVICES=0,1

DATA_PATH=fontdata_example
name=finetune_no_gan_nojt_no_freeze_baseline_vggfix
NO_JT=1  # 设为 0 时，恢复 JT 语义遮盖

NO_JT_ARGS=()
if [ "$NO_JT" -eq 1 ]; then
    NO_JT_ARGS=(--no_jt)
fi

PRETRAIN_CKPT=models/vit_base_font/checkpoint-14.pth

# 使用旧 JSON 清单与按 type 随机配对，沿用原有遮盖选择逻辑。
# 默认恢复输出目录中的最新 checkpoint。

python -m torch.distributed.launch --nproc_per_node=2 --master_port=29555 \
	--use_env main_train.py  \
    --batch_size 2 \
    --accum_iter 32  \
    --model vit_base_patch16_input896x448_win_dec64_8glb_sl1 \
    --num_mask_patches 784 \
    --max_mask_patches_per_block 392 \
    --epochs 51 \
    --warmup_epochs 5 \
    --lr 1e-3 \
    --clip_grad 3.0 \
    --layer_decay 0.8 \
    --drop_path 0.1 \
    --input_size 896 448 \
    --augmentation_policy finetune \
    --adv_warmup_epochs 8 \
    --edge_warmup_epochs 10 \
    --loss_warmup_duration 8 \
    --adv_weight_final 0.3 \
    --edge_weight_final 0.2 \
    --save_freq 5 \
    --data_path $DATA_PATH/ \
    --json_path $DATA_PATH/train_json_new/*.json \
    --val_json_path $DATA_PATH/val_json_new/*.json \
    --output_dir models/$name \
    --log_dir models/$name/logs \
    --finetune $PRETRAIN_CKPT \
    --auto_resume \
    --no_gan \
    --semantic_mask_dir $DATA_PATH/font/train/new \
    --num_mask_annotations_bf 11 \
    --num_mask_annotations_jt 1 \
    "${NO_JT_ARGS[@]}" \
    --mask_coverage_threshold 0.1 \
    --val_tb_image_limit 76 \
    --val_tb_images_per_batch 2 \
    --grad_log_interval 5
    #--mask_mix_probs 0.8 0.0 0.2
