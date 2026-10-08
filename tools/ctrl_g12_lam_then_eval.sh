#!/bin/bash
# T1-L 串联：训练时融合权重 λ=0.3、0.7 各重训一组（配置照 ctrl_g12，只改 fusion_lambda；2 卡、accum 16）。
#   每组：2 卡训练 → 工具口径评测 ep50 与 best → 频带诊断 → 墨量矩诊断（同 ctrl_g12_then_eval.sh 的 a–d）；
#   两组都完成后：训练时 λ 对照图（0.3 / ctrl_g12 best 作 0.5 / 0.7）。
# 在仓库根目录执行；每步时间戳和 rc 写入各组的 chain.log，总流程写入 outputs/ctrl_g12_lam_chain.log。
# 任何一步失败（rc≠0 或训练没跑到 epoch 50）就停下，脚本以非 0 退出，已产出的 ckpt 不动。
# 所有输出目录必须事先不存在（不覆盖已有结果），存在就停下。
# 评测用 load_model 从 ckpt['args'] 读训练时的 λ；baseline 与 ctrl_g12 的旧 ckpt 没有该字段，按 0.5。
# 原 ctrl_g12 评测用的 models/jieti_search/ 已不在服务器上，calibration 用 models/jieti_search2/ 下的副本。

cd /root/autodl-tmp/fontify || exit 1
source /root/miniconda3/etc/profile.d/conda.sh && conda activate fontify

DATA=fontdata_example
BASE_CKPT=/root/autodl-tmp/fontify_baseline/models/finetune_no_gan_nojt_no_freeze_baseline_vggfix/checkpoint-50.pth
CALIB=models/jieti_search2/calibration_v2.json
S_BASE=models/jieti_search2/s_baseline.json
G12_BEST=models/finetune_jieti_ctrl_g12/checkpoint-best.pth
# ctrl_g12 best 的工具口径评测（models/finetune_jieti_ctrl_g12/eval_s_tool_best.json），作 λ=0.5 对账值
EXPECT_05="0.795421700108619 0.5515879868314817 0.28354447540782746"
FUSION_OUT=outputs/fusion_trained_g12best
MAIN_LOG=outputs/ctrl_g12_lam_chain.log
GPUS=${GPUS:-0,1}

mlog() { echo "[$(date '+%F %T')] $*" >> "$MAIN_LOG"; }

# 启动前检查：所有输出目录 / 文件都不存在
for P in models/finetune_jieti_ctrl_g12_lam03 models/finetune_jieti_ctrl_g12_lam07 \
         outputs/band_error_ctrl_g12_lam03 outputs/ink_moments_ctrl_g12_lam03 \
         outputs/band_error_ctrl_g12_lam07 outputs/ink_moments_ctrl_g12_lam07 \
         "$FUSION_OUT" "$FUSION_OUT.tmp" "$MAIN_LOG"; do
    if [ -e "$P" ]; then
        echo "$P 已存在，不覆盖，停止" >&2
        exit 2
    fi
done
for P in "$BASE_CKPT" "$CALIB" "$S_BASE" "$G12_BEST" $DATA/val_pairs_fixed.json; do
    if [ ! -f "$P" ]; then
        echo "缺 $P，停止" >&2
        exit 2
    fi
done

# run_group TAG PORT：一组 λ 的 a–d，失败返回非 0
run_group() {
    local TAG=$1 PORT=$2
    local OUT=models/finetune_jieti_ctrl_g12_$TAG
    local CHAIN=$OUT/chain.log
    local BAND_OUT=outputs/band_error_ctrl_g12_$TAG
    local INK_OUT=outputs/ink_moments_ctrl_g12_$TAG
    local T1_CKPT=$OUT/checkpoint-50.pth
    local BEST_CKPT=$OUT/checkpoint-best.pth
    log() { echo "[$(date '+%F %T')] $*" >> "$CHAIN"; mlog "$TAG: $*"; }

    mkdir "$OUT" || return 1

    # a. 2 卡训练（有效 batch 2×16×2=64）
    log "a. 开始训练 finetune_jieti_ctrl_g12_$TAG.sh（GPU$GPUS）"
    CUDA_VISIBLE_DEVICES=$GPUS MASTER_PORT=$PORT bash finetune_jieti_ctrl_g12_$TAG.sh > "$OUT/train.log" 2>&1
    local RC=$?
    log "a. 训练进程退出 rc=$RC"
    [ $RC -eq 0 ] || return 1

    # 完成标志与 ctrl_g12_then_eval 一致：log.txt 有 epoch==50 的行
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
        return 1
    fi
    log "a. log.txt 已有 epoch 50"

    # b. 工具口径评测 ep50 和 best（参数与 ctrl_g12_then_eval 的 b 步一致，只改 checkpoint / output_json / calibration 路径）
    local TAG2 CKPT
    for TAG2 in ep50 best; do
        if [ "$TAG2" = ep50 ]; then CKPT=$T1_CKPT; else CKPT=$BEST_CKPT; fi
        log "b. 开始 eval_s_baseline $TAG2（GPU0）：$CKPT"
        CUDA_VISIBLE_DEVICES=0 python -u tools/eval_s_baseline.py \
            --checkpoint "$CKPT" \
            --data_path $DATA/ \
            --val_json_path $DATA/val_json_new/*.json \
            --semantic_mask_dir $DATA/font/train/new \
            --fixed_pair_path $DATA/val_pairs_fixed.json \
            --calibration_json $CALIB \
            --output_json "$OUT/eval_s_tool_$TAG2.json" > "$OUT/eval_s_tool_$TAG2.log" 2>&1
        RC=$?
        log "b. eval_s_baseline $TAG2 结束 rc=$RC"
        [ $RC -eq 0 ] || return 1
    done

    # c. 频带诊断（baseline vs 本组 ep50）
    log "c. 开始 eval_band_error（GPU0）"
    CUDA_VISIBLE_DEVICES=0 python -u tools/eval_band_error.py \
        --baseline_ckpt "$BASE_CKPT" \
        --t1_ckpt "$T1_CKPT" \
        --data_path $DATA/ \
        --val_json_path $DATA/val_json_new/*.json \
        --semantic_mask_dir $DATA/font/train/new \
        --fixed_pair_path $DATA/val_pairs_fixed.json \
        --output_dir "$BAND_OUT" > outputs/band_error_ctrl_g12_${TAG}_run.log 2>&1
    RC=$?
    log "c. eval_band_error 结束 rc=$RC"
    [ $RC -eq 0 ] && [ -f "$BAND_OUT/summary.json" ] || return 1

    # d. 墨量矩诊断：BF l1 期望值 = (baseline 0.6168, c 测得的本组均值)
    local T1_BF_L1
    T1_BF_L1=$(python -c "import json;print(json.load(open('$BAND_OUT/summary.json'))['BF']['l1_norm3']['t1'])") || return 1
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
        --output_dir "$INK_OUT" > outputs/ink_moments_ctrl_g12_${TAG}_run.log 2>&1
    RC=$?
    log "d. eval_ink_moments 结束 rc=$RC"
    [ $RC -eq 0 ] || return 1
    log "本组结束"
}

mlog "串联开始（GPU$GPUS）"
run_group lam03 29556 || { mlog "lam03 失败，停止"; exit 1; }
run_group lam07 29557 || { mlog "lam07 失败，停止"; exit 1; }

# e. 训练时 λ 对照图：每个 ckpt 用自己训练时的 λ 前向；λ=0.5 对账只记录不报错（rtol 1e-3）
mlog "e. 开始 export_fusion_trained（GPU0）"
CUDA_VISIBLE_DEVICES=0 python -u tools/export_fusion_trained.py \
    --pair 0.3 models/finetune_jieti_ctrl_g12_lam03/checkpoint-best.pth \
    --pair 0.5 "$G12_BEST" \
    --pair 0.7 models/finetune_jieti_ctrl_g12_lam07/checkpoint-best.pth \
    --data_path $DATA/ \
    --val_json_path $DATA/val_json_new/*.json \
    --semantic_mask_dir $DATA/font/train/new \
    --fixed_pair_path $DATA/val_pairs_fixed.json \
    --calibration_json $CALIB \
    --s_baseline_json $S_BASE \
    --expect_lambda05 $EXPECT_05 \
    --expect_rtol 1e-3 \
    --output_dir "$FUSION_OUT" > outputs/fusion_trained_g12best_run.log 2>&1
RC=$?
mlog "e. export_fusion_trained 结束 rc=$RC"
[ $RC -eq 0 ] || exit 1

mlog "串联结束"
