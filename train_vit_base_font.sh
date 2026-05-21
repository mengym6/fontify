#!/bin/bash

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export MASTER_ADDR=
# export WORLD_SIZE=1
# export RANK=0
export CUDA_VISIBLE_DEVICES=0,1

DATA_PATH=fontdata_example
name=vit_base_font
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29555 \
	--use_env main_train.py  \
    --batch_size 2 \
    --accum_iter 16  \
    --model vit_base_patch16_input896x448_win_dec64_8glb_sl1 \
    --num_mask_patches 784 \
    --max_mask_patches_per_block 392 \
    --epochs 15 \
    --warmup_epochs 1 \
    --lr 1e-3 \
    --clip_grad 3 \
    --layer_decay 0.8 \
    --drop_path 0.1 \
    --input_size 896 448 \
    --save_freq 1 \
    --data_path $DATA_PATH/ \
    --json_path  \
    $DATA_PATH/train_json/font_train_QingniaoHuaguangYaotiFontSimplifiedChinese.json \
    --val_json_path \
    $DATA_PATH/val_json/font_val_QingniaoHuaguangYaotiFontSimplifiedChinese.json \
    --output_dir models/$name \
    --log_dir models/$name/logs \
    --finetune path/to/mae_pretrain_vit_base.pth \
    --auto_resume \
    # --log_wandb \

