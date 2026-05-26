#!/bin/bash

# Finetune 脚本：用于小数据集（~1200对）微调预训练模型
# 核心改动：降lr、短训练、弱化判别器、关闭数据增强中的颜色抖动

export CUDA_VISIBLE_DEVICES=0,1

DATA_PATH=fontdata_example
name=finetune_stele
PRETRAIN_CKPT=models/vit_base_font/checkpoint-14.pth

python -m torch.distributed.launch --nproc_per_node=2 --master_port=29555 \
	--use_env main_train.py  \
    --batch_size 2 \
    --accum_iter 48  \
    --model vit_base_patch16_input896x448_win_dec64_8glb_sl1 \
    --num_mask_patches 784 \
    --max_mask_patches_per_block 392 \
    --epochs 5 \
    --warmup_epochs 1 \
    --lr 5e-5 \
    --clip_grad 1.0 \
    --layer_decay 0.8 \
    --drop_path 0.1 \
    --input_size 896 448 \
    --save_freq 1 \
    --data_path $DATA_PATH/ \
    --json_path $DATA_PATH/train_json_new/*.json \
    --val_json_path $DATA_PATH/val_json_new/*.json \
    --output_dir models/$name \
    --log_dir models/$name/logs \
    --no_color_jitter \
    --finetune $PRETRAIN_CKPT \
