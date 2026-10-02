#!/bin/bash

# T1 结体 loss 正式训练（第 1 组 / 中心配置）。
# 基准 = 343f14d 最优配置（无 GAN、全解冻、BF+随机遮盖）。由 finetune_label_head.sh
# 去掉 label head 得到，并按 Q4、Q3、Q12 修改：
#   - Q4：JT 改用 JT 二值语义遮盖，去掉 --no_jt；BF 仍用 BF 二值遮盖。
#   - Q3：JT 走"同书家、同结构组合、异字"配对，p=1（--jieti_struct_pair_prob 1.0）。
#   - Q12：--save_freq 10，额外存 best（--save_best），比较只用 epoch 50。
# α_jt、w、三项相对系数见下。w 为标定后的绝对值，首轮用标定值 w0（见 PROGRESS 步骤 4）。

export CUDA_VISIBLE_DEVICES=0,1

DATA_PATH=fontdata_example
name=finetune_jieti
PRETRAIN_CKPT=models/vit_base_font/checkpoint-14.pth

# 结体首轮超参（Q10）：α_jt=0.5；w 用标定值 w0=0.6423（2026-10-02 ckpt14 标定）。
JIETI_ALPHA_JT=0.5
JIETI_W=0.6423
JIETI_P=1.0

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
    --save_freq 10 \
    --save_best \
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
