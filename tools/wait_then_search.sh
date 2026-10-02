#!/usr/bin/env bash
# 等待 T1 第 1 组训练完成后，自动启动第 2-50 组超参搜索（PROGRESS 步骤 9/12）。
# 完成判据与 run_jieti_search.py 的 is_complete 完全一致：
#   1) models/finetune_jieti/log.txt 中出现 epoch==50 的记录；
#   2) 第 1 组训练进程已退出（pgrep 精确匹配其 output_dir，不误伤搜索组）。
# 全部为只读检查，绝不触碰正在跑的第 1 组。两个条件都满足后才启动 runner。
set -u

REPO=/root/autodl-tmp/fontify
LOG="${REPO}/models/finetune_jieti/log.txt"
# 精确匹配第 1 组进程：搜索组的 output_dir 是 models/jieti_search/groupNN，不会被命中。
PROC_PAT='output_dir models/finetune_jieti( |$)'
INTERVAL=300

cd "${REPO}" || { echo "[$(date)] 无法进入仓库目录 ${REPO}"; exit 1; }

echo "[$(date)] 等待第 1 组完成：日志 ${LOG}，进程模式 '${PROC_PAT}'，每 ${INTERVAL}s 检查一次。"

while true; do
    log_done=0
    if [ -f "${LOG}" ] && grep -q '"epoch": 50}' "${LOG}"; then
        log_done=1
    fi
    proc_gone=0
    if ! pgrep -f "${PROC_PAT}" > /dev/null 2>&1; then
        proc_gone=1
    fi
    if [ "${log_done}" -eq 1 ] && [ "${proc_gone}" -eq 1 ]; then
        echo "[$(date)] 第 1 组已完成（epoch 50 已出现且训练进程已退出），启动搜索。"
        break
    fi
    echo "[$(date)] 尚未就绪：log_done=${log_done} proc_gone=${proc_gone}，${INTERVAL}s 后重试。"
    sleep "${INTERVAL}"
done

# shellcheck disable=SC1091
source /root/miniconda3/etc/profile.d/conda.sh
conda activate fontify
echo "[$(date)] 启动 run_jieti_search.py。"
exec python tools/run_jieti_search.py \
    --output_root models/jieti_search \
    --group1_dir models/finetune_jieti
