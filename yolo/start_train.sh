#!/usr/bin/env bash
# 在 AutoDL 上运行：bash start_train.sh
# 只开 TensorBoard：bash start_train.sh --tb-only
# 只核对参数和数据：bash start_train.sh --dry-run
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ================= 在这里调整路径和超参数 =================
# 也可临时覆盖，例如 EPOCHS=50 BATCH=4 bash start_train.sh。
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/fontify_yolo_env/bin/python}"
DATA_YAML="${DATA_YAML:-$SCRIPT_DIR/dataset/data.yaml}"
MODEL="${MODEL:-yolo26s-seg.pt}"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR/runs}"
RUN_NAME="${RUN_NAME:-calli_phase_jieti_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-8}"
IMGSZ="${IMGSZ:-640}"
PATIENCE="${PATIENCE:-25}"
WORKERS="${WORKERS:-2}"
SEED="${SEED:-42}"

# auto 会自行决定优化器和初始学习率，忽略手填 LR0。
# 要手动调整 LR0，请把 OPTIMIZER 改成 AdamW 或 SGD。
OPTIMIZER="${OPTIMIZER:-auto}"
LR0="${LR0:-0.01}"
LRF="${LRF:-0.01}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
COS_LR="${COS_LR:-0}"                    # 1=余弦调度；0=原默认调度
AMP="${AMP:-1}"                          # 1=混合精度；0=关闭
SAVE_PERIOD="${SAVE_PERIOD:-10}"
PREVIEW_EVERY="${PREVIEW_EVERY:-5}"
PREVIEW_COUNT="${PREVIEW_COUNT:-4}"
TB_PORT="${TB_PORT:-6006}"
# ==========================================================

MODE="${1:-train}"
if [[ $# -gt 1 ]]; then
    echo "用法：bash start_train.sh [--dry-run|--tb-only]" >&2
    exit 2
fi
case "$MODE" in train|--dry-run|--tb-only) ;; *)
    echo "未知选项：$MODE；超参数请改脚本顶部或用环境变量覆盖。" >&2
    exit 2 ;;
esac
if [[ "$COS_LR" != 0 && "$COS_LR" != 1 ]] || [[ "$AMP" != 0 && "$AMP" != 1 ]]; then
    echo "COS_LR 和 AMP 只能设为 0 或 1。" >&2
    exit 2
fi
ARGS=(
    --data "$DATA_YAML" --model "$MODEL" --project "$PROJECT_DIR" --name "$RUN_NAME"
    --device "$DEVICE" --epochs "$EPOCHS" --batch "$BATCH" --imgsz "$IMGSZ"
    --patience "$PATIENCE" --workers "$WORKERS" --seed "$SEED"
    --optimizer "$OPTIMIZER" --lr0 "$LR0" --lrf "$LRF"
    --weight-decay "$WEIGHT_DECAY" --warmup-epochs "$WARMUP_EPOCHS"
    --save-period "$SAVE_PERIOD" --preview-every "$PREVIEW_EVERY"
    --preview-count "$PREVIEW_COUNT"
)
[[ "$COS_LR" == 0 ]] || ARGS+=(--cos-lr)
[[ "$AMP" == 1 ]] || ARGS+=(--no-amp)
if [[ "$MODE" == --dry-run ]]; then
    exec "$PYTHON_BIN" "$SCRIPT_DIR/train.py" "${ARGS[@]}" --dry-run
fi
if [[ "$MODE" == train ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/train.py" "${ARGS[@]}" --dry-run >/dev/null
fi

"$PYTHON_BIN" - "$MODE" "$SCRIPT_DIR" "$PROJECT_DIR" "$RUN_NAME" "$TB_PORT" "${ARGS[@]}" <<'PY'
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

mode, root, project, name, port_text, *training_args = sys.argv[1:]
root, project = Path(root).resolve(), Path(project).expanduser().resolve()
port = int(port_text)
if not 1 <= port <= 65535:
    raise ValueError("TB_PORT 必须在 1–65535 范围内")
if not name or Path(name).name != name or name in {".", ".."}:
    raise ValueError("RUN_NAME 应是单个目录名，不含路径")
project.mkdir(parents=True, exist_ok=True)
url = f"http://127.0.0.1:{port}"


def tensorboard_logdir():
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url + "/data/logdir", timeout=2) as response:
        return json.load(response)["logdir"]


with socket.socket() as connection:
    connection.settimeout(1)
    occupied = connection.connect_ex(("127.0.0.1", port)) == 0
if occupied:
    try:
        active_logdir = tensorboard_logdir()
        if Path(active_logdir).resolve() != project:
            raise ValueError(f"当前日志目录为 {active_logdir}，需要 {project}")
    except Exception as error:
        raise RuntimeError(f"端口 {port} 已被其他服务或其他日志目录占用，请更改 TB_PORT。{error}") from error
    print(f"复用现有 TensorBoard：{url}，日志目录 {project}", flush=True)
else:
    tb_log = project / f"tensorboard_{port}.log"
    with tb_log.open("a") as log:
        tb = subprocess.Popen(
            [sys.executable, "-m", "tensorboard.main", "--logdir", str(project),
             "--host", "127.0.0.1", "--port", str(port)],
            cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (project / f"tensorboard_{port}.pid").write_text(str(tb.pid) + "\n")
    for _ in range(30):
        if tb.poll() is not None:
            raise RuntimeError(f"TensorBoard启动失败，请查看 {tb_log}")
        try:
            if Path(tensorboard_logdir()).resolve() == project:
                break
        except Exception:
            time.sleep(0.5)
    else:
        tb.terminate()
        raise RuntimeError(f"TensorBoard未就绪，请查看 {tb_log}")
    print(f"已后台启动 TensorBoard：PID {tb.pid}，{url}，日志 {tb_log}", flush=True)
print(f"本地浏览器访问：先在 VS Code Ports 转发服务器 {port} 端口。", flush=True)
if mode == "--tb-only":
    sys.exit(0)

# 兼容本脚本启动前已有的训练；不依赖某个特定PID文件。
entry = root / "train.py"
if Path("/proc").is_dir():
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            argv = (process / "cmdline").read_bytes().decode().split("\0")
            cwd = (process / "cwd").resolve()
            matches = any((Path(arg) if Path(arg).is_absolute() else cwd / arg).resolve() == entry
                          for arg in argv[1:] if arg.endswith("train.py"))
        except (OSError, UnicodeError):
            continue
        if matches:
            print(f"本目录已有训练 PID {process.name}，TensorBoard已就绪；没有重复启动。", flush=True)
            sys.exit(0)

run_dir, train_log = project / name, project / f"{name}.log"
if run_dir.exists() or train_log.exists():
    raise FileExistsError(f"运行名已存在：{name}，请更改 RUN_NAME；续训使用 train.py --resume。")
environment = os.environ.copy()
environment["PYTHONUNBUFFERED"] = "1"
command = [sys.executable, "-u", str(entry), *training_args]
with train_log.open("x") as log:
    process = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
launch = {"pid": process.pid, "command": command, "started_at": time.time(),
          "log": str(train_log), "run": str(run_dir), "tensorboard": url}
(project / f"{name}_launch.json").write_text(json.dumps(launch, ensure_ascii=False, indent=2) + "\n")
print(f"已后台启动训练 PID {process.pid}", flush=True)
print(f"训练日志：{train_log}\n权重目录：{run_dir / 'weights'}", flush=True)
print(f"查看日志：tail -f {train_log}", flush=True)
PY
