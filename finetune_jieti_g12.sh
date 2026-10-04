#!/bin/bash

# T1-F（PROGRESS 步骤 13）：用第 12 组（group12_retry1）超参重训 T1，与 baseline 对照。
# 参数逐项照抄第 12 组实际命令（runner.log 启动行 / hparams.json），只改：
#   - 4 卡、accum_iter 8：有效 batch 2×8×4=64，与第 12 组（2 卡、accum 16）相同。
#   - checkpoint 按步骤 13：--save_freq 10 加存 best，不加 --save_best_only。
#   - 输出目录 models/finetune_jieti_g12；seed 用默认值 0，与第 12 组一致。
# T1 配置：不加 --no_jt（JT 二值语义遮盖）。第 12 组没有 --auto_resume，这里也不加。

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

DATA_PATH=fontdata_example
name=finetune_jieti_g12
PRETRAIN_CKPT=models/vit_base_font/checkpoint-14.pth

# 第 12 组超参（search_plan.csv 第 12 行，hparams.json 完整精度）
JIETI_ALPHA_JT=0.585887
JIETI_W=0.73719102
JIETI_P=0.77372
LR=0.00195699
WARMUP_EPOCHS=10

MASTER_PORT=${MASTER_PORT:-29555}

python -m torch.distributed.launch --nproc_per_node=4 --master_port=$MASTER_PORT \
	--use_env main_train.py  \
    --batch_size 2 \
    --accum_iter 8  \
    --model vit_base_patch16_input896x448_win_dec64_8glb_sl1 \
    --num_mask_patches 784 \
    --max_mask_patches_per_block 392 \
    --epochs 51 \
    --warmup_epochs $WARMUP_EPOCHS \
    --lr $LR \
    --clip_grad 3.0 \
    --style_weight 14.73 \
    --layer_decay 0.8 \
    --drop_path 0.1 \
    --input_size 896 448 \
    --augmentation_policy finetune \
    --adv_warmup_epochs 8 \
    --edge_warmup_epochs 10 \
    --loss_warmup_duration 8 \
    --adv_weight_final 0.3 \
    --edge_weight_final 0.2 \
    --save_freq 10 \
    --save_best \
    --data_path $DATA_PATH/ \
    --json_path $DATA_PATH/train_json_new/*.json \
    --val_json_path $DATA_PATH/val_json_new/*.json \
    --output_dir models/$name \
    --log_dir models/$name/logs \
    --finetune $PRETRAIN_CKPT \
    --no_gan \
    --semantic_mask_dir $DATA_PATH/font/train/new \
    --num_mask_annotations_bf 11 \
    --num_mask_annotations_jt 1 \
    --mask_coverage_threshold 0.1 \
    --jieti_loss \
    --jieti_alpha_jt $JIETI_ALPHA_JT \
    --jieti_w $JIETI_W \
    --jieti_struct_pair_prob $JIETI_P \
    --jieti_soft_fg linear \
    --jieti_k_max 4 \
    --jieti_pool 224 \
    --jieti_valid_ink 200 \
    --jieti_pred_mass_ratio 0.1 \
    --jieti_w_centroid 0.9688 \
    --jieti_w_logsigma 0.5409 \
    --jieti_w_shape 1.0 \
    --fixed_pair_path $DATA_PATH/val_pairs_fixed.json \
    --s_baseline_path models/jieti_search/s_baseline.json \
    --vis_every_epoch \
    --val_tb_image_limit 76 \
    --val_tb_images_per_batch 2 \
    --grad_log_interval 5
