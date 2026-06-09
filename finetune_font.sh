#!/bin/bash

# Finetune 脚本：用于小数据集（~1200对）微调预训练模型
# 核心改动：降lr、短训练、弱化判别器、关闭数据增强中的颜色抖动

export CUDA_VISIBLE_DEVICES=0,1

DATA_PATH=fontdata_example
name=finetune_stele
PRETRAIN_CKPT=models/vit_base_font/checkpoint-14.pth

# 手动切换阶段时修改 --mask_mix_probs：
# stage1: 0.7 0.2 0.1; stage2: 0.4 0.3 0.3; stage3: 0.3 0.3 0.4

python -m torch.distributed.launch --nproc_per_node=2 --master_port=29555 \
	--use_env main_train.py  \
    --batch_size 2 \
    --accum_iter 32  \
    --model vit_base_patch16_input896x448_win_dec64_8glb_sl1 \
    --num_mask_patches 784 \
    --max_mask_patches_per_block 392 \
    --epochs 100 \
    --warmup_epochs 1 \
    --lr 5e-5 \
    --weight_decay 0.05 \
    --clip_grad 1.0 \
    --layer_decay 0.7 \
    --drop_path 0.1 \
    --input_size 896 448 \
    --save_freq 5 \
    --data_path $DATA_PATH/ \
    --json_path $DATA_PATH/train_json_new/*.json \
    --val_json_path $DATA_PATH/val_json_new/*.json \
    --output_dir models/$name \
    --log_dir models/$name/logs \
    --finetune $PRETRAIN_CKPT \
    --auto_resume \
    --freeze_encoder \
    --freeze_blocks 9 \
    --semantic_mask_dir $DATA_PATH/font/train/new \
    --num_mask_annotations_bf 3 \
    --num_mask_annotations_jt 1 \
    --mask_coverage_threshold 0.1 \
    --semantic_only_epochs 0 \
    --mask_mix_probs 0.7 0.2 0.1 \
    --jt_edge_loss_weight 0.05 \
    --jt_edge_vec_weight 0.1 \
    --jt_edge_tau 0.45 \
    --jt_edge_temp 0.05
