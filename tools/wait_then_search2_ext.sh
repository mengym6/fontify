#!/usr/bin/env bash
# T1-S "α_jt 补搜"（PROGRESS T1-S，用户 2026-10-05 定）：等第 2–25 组跑完，自动接着跑第 26–37 组。
#
# 启动条件（两条同时满足）：
#   (a) 主 runner（tools/run_jieti_search2.py，命令行里不带 search_plan_ext.csv 的那个 python 进程）已退出；
#   (b) summary.csv 里第 2–25 组都有 status，且以 done / skipped(done) / done(retryN) 开头。
# 只看 (a) 不够：runner 崩溃也会退出。(a) 满足而 (b) 不满足 → 不启动，写日志后退出（exit 3），由主控处理。
# 启动前检查（失败就每个间隔重试，共 PRECHECK_TRIES 次，仍失败则退出 exit 4，不启动）：
#   4 卡无计算进程、端口 29726–29737 空闲、磁盘 >= 40G、ext 计划 sha256、两个 json 的 sha256 前缀、
#   没有已在跑的 ext runner。
# ext runner 退出后再读一次 summary，第 26–37 组按与 (b) 相同的规则判定：有不合格的组就列出并 exit 5；
# 全部合格才写"补搜完成"。
# 全程只读：不 kill、不删除任何东西。日志：models/jieti_search2/wait_ext.log。
# 不用 set -u：conda 的激活脚本会引用未定义变量。
#
# 测试开关（只在本地 mock 用，服务器上不要设置）：
#   WAIT_EXT_REPO      仓库目录（默认 /root/autodl-tmp/fontify）
#   WAIT_EXT_INTERVAL  检查间隔秒数（默认 300）
#   WAIT_EXT_CONDA_SH  conda.sh 路径；设成 none 则不激活 conda
#   WAIT_EXT_PROC_NET  读端口占用的目录（默认 /proc/net）

REPO="${WAIT_EXT_REPO:-/root/autodl-tmp/fontify}"
INTERVAL="${WAIT_EXT_INTERVAL:-300}"
CONDA_SH="${WAIT_EXT_CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
PROC_NET="${WAIT_EXT_PROC_NET:-/proc/net}"

SEARCH_DIR="models/jieti_search2"
SUMMARY="${SEARCH_DIR}/summary.csv"
EXT_PLAN="${SEARCH_DIR}/search_plan_ext.csv"
EXT_PLAN_SHA256="ccfebc64634643c4028639db580f88f0391b4b25549d0381957d45deb2a49d4b"
S_BASELINE="${SEARCH_DIR}/s_baseline.json"
S_BASELINE_SHA_PREFIX="a692e26921c6"
CALIB="${SEARCH_DIR}/calibration_v2.json"
CALIB_SHA_PREFIX="907392b3a3ec"
PORT_LO=29726
PORT_HI=29737
MIN_FREE_GB=40
PRECHECK_TRIES=3
WAIT_LOG="${SEARCH_DIR}/wait_ext.log"
RUNNER_EXT_LOG="${SEARCH_DIR}/runner_ext.log"

cd "${REPO}" || { echo "[$(date '+%F %T')] 无法进入仓库目录 ${REPO}"; exit 1; }

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "${WAIT_LOG}"
}

if [ "${CONDA_SH}" != "none" ]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}" || { log "无法 source ${CONDA_SH}，退出。"; exit 1; }
    conda activate fontify || { log "conda activate fontify 失败，退出。"; exit 1; }
fi

# python 进程的命令行：<pid> [.../]python[3.x] [选项] tools/run_jieti_search2.py ...
# 只认 python 本身（screen、bash -c 包装进程的命令行以 SCREEN/bash 开头，不会命中）。
RUNNER_RE='^ *[0-9]+ ([^ ]*/)?[Pp]ython[0-9.]* ([^ ]+ )*tools/run_jieti_search2\.py( |$)'

main_runner_pids() {
    ps -eo pid=,args= | grep -E "${RUNNER_RE}" | grep -v 'search_plan_ext\.csv' | awk '{print $1}' | tr '\n' ' '
}

ext_runner_pids() {
    ps -eo pid=,args= | grep -E "${RUNNER_RE}" | grep 'search_plan_ext\.csv' | awk '{print $1}' | tr '\n' ' '
}

# 第 $1–$2 组状态检查（默认 2–25）：全部合格时 exit 0；否则打印不合格的组并 exit 1。
check_summary() {
    python - "${SUMMARY}" "${1:-2}" "${2:-25}" <<'PYEOF'
import csv, re, sys
path = sys.argv[1]
lo, hi = int(sys.argv[2]), int(sys.argv[3])
ok_re = re.compile(r"^(done|skipped\(done\)|done\(retry\d+\))(;|$)")
try:
    with open(path, newline="", encoding="utf-8") as f:
        status = {(r.get("group") or "").strip(): (r.get("status") or "").strip()
                  for r in csv.DictReader(f)}
except OSError as e:
    print("读不到 summary：%s" % e)
    sys.exit(1)
bad, warn = [], []
for g in range(lo, hi + 1):
    st = status.get(str(g))
    if st is None:
        bad.append("%d:缺行" % g)
    elif not ok_re.match(st):
        bad.append("%d:%s" % (g, st or "status 为空"))
    elif "eval_failed" in st:
        warn.append("%d:%s" % (g, st))
if warn:
    print("提示：以下组训练完成但评测失败（不阻止启动，汇总前需补评）：" + "，".join(warn))
if bad:
    print("不合格：" + "，".join(bad))
    sys.exit(1)
print("第 %d–%d 组状态全部合格" % (lo, hi))
PYEOF
}

file_sha256() {
    sha256sum "$1" 2>/dev/null | awk '{print $1}'
}

# 本地端口在 [PORT_LO, PORT_HI] 内的 tcp/tcp6 条目（任意状态），输出端口号列表。
busy_ports() {
    for f in "${PROC_NET}/tcp" "${PROC_NET}/tcp6"; do
        [ -f "${f}" ] || continue
        awk -v lo="${PORT_LO}" -v hi="${PORT_HI}" 'NR > 1 {
            split($2, a, ":"); p = 0; h = toupper(a[2])
            for (i = 1; i <= length(h); i++) p = p * 16 + index("0123456789ABCDEF", substr(h, i, 1)) - 1
            if (p >= lo && p <= hi) print p
        }' "${f}"
    done | sort -u | tr '\n' ' '
}

# 启动前检查：全部通过返回 0，否则把原因写日志并返回 1。
precheck() {
    local fail=0 out sha free_kb free_gb ports ext
    if [ "$(file_sha256 "${EXT_PLAN}")" != "${EXT_PLAN_SHA256}" ]; then
        log "检查失败：${EXT_PLAN} 的 sha256 为 '$(file_sha256 "${EXT_PLAN}")'，应为 ${EXT_PLAN_SHA256}。"; fail=1
    fi
    sha="$(file_sha256 "${S_BASELINE}")"
    if [ "${sha:0:12}" != "${S_BASELINE_SHA_PREFIX}" ]; then
        log "检查失败：${S_BASELINE} 的 sha256 为 '${sha}'，前缀应为 ${S_BASELINE_SHA_PREFIX}。"; fail=1
    fi
    sha="$(file_sha256 "${CALIB}")"
    if [ "${sha:0:12}" != "${CALIB_SHA_PREFIX}" ]; then
        log "检查失败：${CALIB} 的 sha256 为 '${sha}'，前缀应为 ${CALIB_SHA_PREFIX}。"; fail=1
    fi
    if ! out="$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>&1)"; then
        log "检查失败：nvidia-smi 查询失败：${out}"; fail=1
    elif [ -n "$(echo "${out}" | tr -d '[:space:]')" ]; then
        log "检查失败：GPU 上仍有计算进程：$(echo "${out}" | tr '\n' ' ')"; fail=1
    fi
    ports="$(busy_ports)"
    if [ -n "${ports// /}" ]; then
        log "检查失败：端口 ${PORT_LO}–${PORT_HI} 中已被占用：${ports}"; fail=1
    fi
    free_kb="$(df -Pk "${SEARCH_DIR}" | awk 'NR == 2 {print $4}')"
    free_gb=$(( ${free_kb:-0} / 1048576 ))
    if [ "${free_gb}" -lt "${MIN_FREE_GB}" ]; then
        log "检查失败：磁盘剩 ${free_gb}G，少于 ${MIN_FREE_GB}G。"; fail=1
    fi
    ext="$(ext_runner_pids)"
    if [ -n "${ext// /}" ]; then
        log "检查失败：已有 ext runner 在跑（PID ${ext}）。"; fail=1
    fi
    if [ "${fail}" -eq 0 ]; then
        log "启动前检查通过：ext 计划与两个 json 的 sha256 一致，4 卡无计算进程，端口 ${PORT_LO}–${PORT_HI} 空闲，磁盘剩 ${free_gb}G。"
    fi
    return "${fail}"
}

log "开始等待（PID $$）：主 runner 退出且 summary 第 2–25 组全部完成后，启动第 26–37 组；每 ${INTERVAL}s 检查一次。"

while true; do
    pids="$(main_runner_pids)"
    if [ -n "${pids// /}" ]; then
        log "主 runner 仍在运行（PID ${pids}），${INTERVAL}s 后再查。"
        sleep "${INTERVAL}"
        continue
    fi
    if out="$(check_summary 2>&1)"; then
        log "主 runner 已退出；${out}"
        break
    fi
    log "主 runner 已退出，但 summary 第 2–25 组未全部完成：${out}。不启动第 26–37 组，退出，由主控处理。"
    exit 3
done

try=1
until precheck; do
    if [ "${try}" -ge "${PRECHECK_TRIES}" ]; then
        log "启动前检查连续 ${PRECHECK_TRIES} 次未通过，不启动，退出，由主控处理。"
        exit 4
    fi
    try=$(( try + 1 ))
    log "启动前检查未通过，${INTERVAL}s 后重试（第 ${try}/${PRECHECK_TRIES} 次）。"
    sleep "${INTERVAL}"
done

log "启动 run_jieti_search2.py（第 26–37 组），输出追加到 ${RUNNER_EXT_LOG}。"
python -u tools/run_jieti_search2.py --gpu_slots "0,1;2,3" \
    --plan models/jieti_search2/search_plan_ext.csv \
    --start_group 26 --end_group 37 >> "${RUNNER_EXT_LOG}" 2>&1
rc=$?
log "ext runner 退出，rc=${rc}。"
if ! out="$(check_summary 26 37 2>&1)"; then
    log "ext 结束后复查 summary，第 26–37 组未全部合格：${out}。退出，由主控处理。"
    exit 5
fi
log "ext 结束后复查 summary：${out}"
if [ "${rc}" -ne 0 ]; then
    exit "${rc}"
fi
log "补搜完成。"
exit 0
