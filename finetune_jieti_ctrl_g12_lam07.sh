#!/bin/bash

# T1-F：用第 12 组搜索出的 w/lr/warmup/batch 重跑 T1-C 式对照组（no_jt、p=0、α_jt=1.0）。
# 以 T1-C 对照组 finetune_jieti_ctrl.sh 为模板，保留：
#   - --no_jt：JT 按 baseline 方式随机遮盖，不用 JT 二值语义遮盖；BF 仍用 BF 二值遮盖。
#   - p=0（--jieti_struct_pair_prob 0）：不走同结构配对，全部用 baseline 的同 type 随机配对，不剔除样本。
#   - α_jt=1.0：原 loss 与 baseline 数学等价。
# 换成第 12 组（group12_retry1/hparams.json 完整精度）的值：
#   - w=0.73719102、lr=0.00195699、warmup 10；
#   - 4 卡、accum_iter 8：有效 batch 2×8×4=64，与第 12 组（2 卡、accum 16）相同。
# 注意：相对 baseline 不只多了结体 loss，还多了 lr、batch（64 vs 128）、warmup（10 vs 5）三处差异；
#       相对 T1-C 是 w 和这三项一起变了。
# checkpoint：--save_freq 10 加存 best，不加 --save_best_only。
# 与已审的 finetune_jieti_g12.sh 一致，不加 --auto_resume：重跑会从 epoch 0 开始，不从旧 checkpoint 续训。
# 其余参数与 finetune_jieti_ctrl.sh 完全相同；seed 用默认值 0。
# T1-L：复制自 finetune_jieti_ctrl_g12.sh，只改训练时融合权重 --fusion_lambda 0.7，
#   以及 2 卡、accum_iter 16（有效 batch 2×16×2=64 不变）、输出目录、端口。
#   卡数变了，每卡 seed（seed+rank）和数据切分与 ctrl_g12 不同。
#   S 分母改用 models/jieti_search2/s_baseline.json：models/jieti_search/ 已不在服务器上，
#   路径不存在时 best 会退回按 val 总 loss 选；jieti_search2 那份与原文件数值一致。

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

DATA_PATH=fontdata_example
name=finetune_jieti_ctrl_g12_lam07
PRETRAIN_CKPT=models/vit_base_font/checkpoint-14.pth

# 结体超参：α_jt、p 同 T1-C；w 取第 12 组的值。
JIETI_ALPHA_JT=1.0
JIETI_W=0.73719102
JIETI_P=0
# 第 12 组的 lr / warmup
LR=0.00195699
WARMUP_EPOCHS=10

MASTER_PORT=${MASTER_PORT:-29557}

python -m torch.distributed.launch --nproc_per_node=2 --master_port=$MASTER_PORT \
	--use_env main_train.py  \
    --batch_size 2 \
    --accum_iter 16  \
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
    --no_jt \
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
    --s_baseline_path models/jieti_search2/s_baseline.json \
    --vis_every_epoch \
    --val_tb_image_limit 76 \
    --val_tb_images_per_batch 2 \
    --grad_log_interval 5 \
    --fusion_lambda 0.7
