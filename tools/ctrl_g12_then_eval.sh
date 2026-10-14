#!/bin/bash
# T1-F 串联：4 卡训练 T1-C 式对照组（no_jt、p=0、α_jt=1.0，w/lr/warmup/batch 用第 12 组的值）
#   → 工具口径评测 ep50 与 best → 频带诊断 → 墨量矩诊断。
# 在仓库根目录执行；每步时间戳和 rc 写入 models/finetune_jieti_ctrl_g12/chain.log。
# 训练没跑到 epoch 50 就停下；b/c/d 出错只记一笔，继续往下。不恢复搜索（没有 e 步）。

cd /root/autodl-tmp/fontify || exit 1
source /root/miniconda3/etc/profile.d/conda.sh && conda activate fontify

OUT=models/finetune_jieti_ctrl_g12
CHAIN=$OUT/chain.log
DATA=fontdata_example
BASE_CKPT=/root/autodl-tmp/fontify_baseline/models/finetune_no_gan_nojt_no_freeze_baseline_vggfix/checkpoint-50.pth
T1_CKPT=$OUT/checkpoint-50.pth
BEST_CKPT=$OUT/checkpoint-best.pth
BAND_OUT=outputs/band_error_ctrl_g12
INK_OUT=outputs/ink_moments_ctrl_g12

log() { echo "[$(date '+%F %T')] $*" >> "$CHAIN"; }

mkdir -p "$OUT"

# a. 4 卡训练（有效 batch 2×8×4=64）
log "a. 开始训练 finetune_jieti_ctrl_g12.sh（GPU0,1,2,3）"
CUDA_VISIBLE_DEVICES=0,1,2,3 MASTER_PORT=29555 bash finetune_jieti_ctrl_g12.sh > "$OUT/train.log" 2>&1
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
    log "a. log.txt 里没有 epoch 50，停止：不评测"
    exit 1
fi
log "a. log.txt 已有 epoch 50"

# b. 工具口径评测 ep50 和 best（参数与 s_baseline 一致，只改 checkpoint / output_json）
for TAG in ep50 best; do
    if [ "$TAG" = ep50 ]; then CKPT=$T1_CKPT; else CKPT=$BEST_CKPT; fi
    if [ -e "$OUT/eval_s_tool_$TAG.json" ]; then
        log "b. $OUT/eval_s_tool_$TAG.json 已存在，跳过"
        continue
    fi
    log "b. 开始 eval_s_baseline $TAG（GPU0）：$CKPT"
    CUDA_VISIBLE_DEVICES=0 python -u tools/eval_s_baseline.py \
        --checkpoint "$CKPT" \
        --data_path $DATA/ \
        --val_json_path $DATA/val_json_new/*.json \
        --semantic_mask_dir $DATA/font/train/new \
        --fixed_pair_path $DATA/val_pairs_fixed.json \
        --calibration_json models/jieti_search/calibration_v2.json \
        --output_json "$OUT/eval_s_tool_$TAG.json" > "$OUT/eval_s_tool_$TAG.log" 2>&1
    log "b. eval_s_baseline $TAG 结束 rc=$?"
done

# c. 频带诊断。eval_band_error 在跑模型之前就建好输出目录（exist_ok），
#    所以按 summary.json 是否存在判断跳过；目录已存在但没有 summary.json 时会重跑并覆盖。
if [ -f "$BAND_OUT/summary.json" ]; then
    log "c. $BAND_OUT/summary.json 已存在，跳过"
else
    log "c. 开始 eval_band_error（GPU0）"
    CUDA_VISIBLE_DEVICES=0 python -u tools/eval_band_error.py \
        --baseline_ckpt "$BASE_CKPT" \
        --t1_ckpt "$T1_CKPT" \
        --data_path $DATA/ \
        --val_json_path $DATA/val_json_new/*.json \
        --semantic_mask_dir $DATA/font/train/new \
        --fixed_pair_path $DATA/val_pairs_fixed.json \
        --output_dir "$BAND_OUT" > outputs/band_error_ctrl_g12_run.log 2>&1
    log "c. eval_band_error 结束 rc=$?"
fi

# d. 墨量矩诊断：对账用 c 的逐样本结果；BF l1 期望值 = (baseline 0.6168, c 测得的本组均值)
#    注意：eval_ink_moments 遇到已存在的输出目录会直接退出（不覆盖），
#    若目录在但没有 summary.json（中途失败），需人工处理该目录后重跑。
if [ -f "$INK_OUT/summary.json" ]; then
    log "d. $INK_OUT/summary.json 已存在，跳过"
elif [ ! -f "$BAND_OUT/summary.json" ]; then
    log "d. 缺 $BAND_OUT/summary.json，无法对账，跳过"
else
    T1_BF_L1=$(python -c "import json;print(json.load(open('$BAND_OUT/summary.json'))['BF']['l1_norm3']['t1'])")
    log "d. 开始 eval_ink_moments（GPU0），expect_bf_l1 = 0.6168 $T1_BF_L1"
    CUDA_VISIBLE_DEVICES=0 python -u tools/eval_ink_moments.py \
        --baseline_ckpt "$BASE_CKPT" \
        --t1_ckpt "$T1_CKPT" \
        --data_path $DATA/ \
        --val_json_path $DATA/val_json_new/*.json \
        --semantic_mask_dir $DATA/font/train/new \
        --fixed_pair_path $DATA/val_pairs_fixed.json \
        --band_csv "$BAND_OUT/per_image.csv" \
        --expect_bf_l1 0.6168 "$T1_BF_L1" \
        --output_dir "$INK_OUT" > outputs/ink_moments_ctrl_g12_run.log 2>&1
    log "d. eval_ink_moments 结束 rc=$?"
fi

log "串联结束"
