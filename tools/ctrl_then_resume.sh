#!/bin/bash
# T1-C 串联：4 卡训练对照组 → 工具口径评测 ep50 → 频带诊断 → 墨量矩诊断 → 恢复搜索。
# 在仓库根目录执行；每步时间戳写入 models/finetune_jieti_ctrl/chain.log。
# 训练没跑到 epoch 50 就停下，不恢复搜索；b/c/d 出错只记一笔，继续往下，保证搜索会恢复。

cd /root/autodl-tmp/fontify || exit 1
source /root/miniconda3/etc/profile.d/conda.sh && conda activate fontify

OUT=models/finetune_jieti_ctrl
CHAIN=$OUT/chain.log
DATA=fontdata_example
BASE_CKPT=/root/autodl-tmp/fontify_baseline/models/finetune_no_gan_nojt_no_freeze_baseline_vggfix/checkpoint-50.pth
CTRL_CKPT=$OUT/checkpoint-50.pth
BAND_OUT=outputs/band_error_ctrl
INK_OUT=outputs/ink_moments_ctrl

log() { echo "[$(date '+%F %T')] $*" >> "$CHAIN"; }

mkdir -p "$OUT"

# a. 4 卡训练（有效 batch 2×16×4=128）
log "a. 开始训练 finetune_jieti_ctrl.sh（GPU0,1,2,3）"
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29555 bash finetune_jieti_ctrl.sh > "$OUT/train.log" 2>&1
log "a. 训练进程退出 rc=$?"

# 完成标志与 runner 一致：log.txt 有 epoch==50 的行
if ! python - "$OUT/log.txt" <<'EOF'
import json, sys
try:
    lines = open(sys.argv[1], encoding="utf-8").read().splitlines()
except OSError:
    sys.exit(1)
for line in lines:
    try:
        if json.loads(line).get("epoch") == 50:
            sys.exit(0)
    except ValueError:
        pass
sys.exit(1)
EOF
then
    log "a. log.txt 里没有 epoch 50，停止：不评测、不恢复搜索"
    exit 1
fi
log "a. log.txt 已有 epoch 50"

# b. 工具口径评测 ep50（参数与 s_baseline 一致，只改 checkpoint / output_json）
if [ -e "$OUT/eval_s_tool_ep50.json" ]; then
    log "b. $OUT/eval_s_tool_ep50.json 已存在，跳过"
else
    log "b. 开始 eval_s_baseline（GPU0）"
    CUDA_VISIBLE_DEVICES=0 python -u tools/eval_s_baseline.py \
        --checkpoint "$CTRL_CKPT" \
        --data_path $DATA/ \
        --val_json_path $DATA/val_json_new/*.json \
        --semantic_mask_dir $DATA/font/train/new \
        --fixed_pair_path $DATA/val_pairs_fixed.json \
        --calibration_json models/jieti_search/calibration_v2.json \
        --output_json "$OUT/eval_s_tool_ep50.json" > "$OUT/eval_s_tool_ep50.log" 2>&1
    log "b. eval_s_baseline 结束 rc=$?"
fi

# c. 频带诊断（输出目录已存在就跳过，不覆盖）
if [ -e "$BAND_OUT" ]; then
    log "c. $BAND_OUT 已存在，跳过"
else
    log "c. 开始 eval_band_error（GPU0）"
    CUDA_VISIBLE_DEVICES=0 python -u tools/eval_band_error.py \
        --baseline_ckpt "$BASE_CKPT" \
        --t1_ckpt "$CTRL_CKPT" \
        --data_path $DATA/ \
        --val_json_path $DATA/val_json_new/*.json \
        --semantic_mask_dir $DATA/font/train/new \
        --fixed_pair_path $DATA/val_pairs_fixed.json \
        --output_dir "$BAND_OUT" > outputs/band_error_ctrl_run.log 2>&1
    log "c. eval_band_error 结束 rc=$?"
fi

# d. 墨量矩诊断：对账用 c 的逐样本结果；BF l1 期望值 = (baseline 0.6168, c 测得的对照组均值)
if [ -e "$INK_OUT" ]; then
    log "d. $INK_OUT 已存在，跳过"
elif [ ! -f "$BAND_OUT/summary.json" ]; then
    log "d. 缺 $BAND_OUT/summary.json，无法对账，跳过"
else
    CTRL_BF_L1=$(python -c "import json;print(json.load(open('$BAND_OUT/summary.json'))['BF']['l1_norm3']['t1'])")
    log "d. 开始 eval_ink_moments（GPU0），expect_bf_l1 = 0.6168 $CTRL_BF_L1"
    CUDA_VISIBLE_DEVICES=0 python -u tools/eval_ink_moments.py \
        --baseline_ckpt "$BASE_CKPT" \
        --t1_ckpt "$CTRL_CKPT" \
        --data_path $DATA/ \
        --val_json_path $DATA/val_json_new/*.json \
        --semantic_mask_dir $DATA/font/train/new \
        --fixed_pair_path $DATA/val_pairs_fixed.json \
        --band_csv "$BAND_OUT/per_image.csv" \
        --expect_bf_l1 0.6168 "$CTRL_BF_L1" \
        --output_dir "$INK_OUT" > outputs/ink_moments_ctrl_run.log 2>&1
    log "d. eval_ink_moments 结束 rc=$?"
fi

# e. 恢复搜索（新 screen jieti_search_r2）
log "e. 启动 screen jieti_search_r2 恢复搜索 group 2-25"
screen -dmS jieti_search_r2 bash -c 'cd /root/autodl-tmp/fontify && source /root/miniconda3/etc/profile.d/conda.sh && conda activate fontify && python -u tools/run_jieti_search.py --output_root models/jieti_search --group1_dir models/finetune_jieti --start_group 2 --end_group 25 --gpu_slots "0,1;2,3" >> models/jieti_search/runner.log 2>&1'
sleep 5
if screen -ls | grep -q jieti_search_r2; then
    log "e. screen jieti_search_r2 已在运行"
else
    log "e. screen jieti_search_r2 未找到，需人工检查"
fi
log "串联结束"
