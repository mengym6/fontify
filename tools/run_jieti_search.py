#!/usr/bin/env python
"""T1 结体 loss 超参随机搜索 runner（PROGRESS 步骤 9、10、12 和 C3）。

读取 search_plan.csv，从第 2 组起按 plan 顺序领取运行（每组用 2 卡）。
--gpu_slots "0,1;2,3" 时有多个卡槽：每个槽同一时间只跑一组，槽之间并行。
槽开新组前要求槽内 GPU 上没有任何计算进程，且没有第 1 组（--group1_dir）的训练进程
用到这些卡，不满足就每 --poll_interval 秒查一次。所有 summary.csv 写入、磁盘检查都在
主进程里串行做，没有并发写。
第 1 组复用正式训练输出目录，不重跑；排在最后，等它的训练进程全部退出后再汇总。

每组：以 finetune_jieti.sh 的基准配置为底，只覆盖本组抽到的 6 个超参
（alpha_jt、w、p、lr、accum_iter、warmup_epochs），并带上 --save_best、
--s_baseline_path、--fixed_pair_path、--vis_every_epoch。按 C3 只保留 best。

断点续跑：组目录已存在且 log.txt 里有最后一个 epoch 的记录 → 跳过。
目录存在但未完成（中途崩溃）→ 不删旧目录，输出到新目录 <name>_retry<n> 从头重跑。

磁盘：每组开跑前检查剩余空间（步骤 12）。估算 = 最近一组 best 的实际大小
（没有取 2G）× 2 + TB/vis 余量。不够则打印提示并退出，不删任何文件。

本脚本不依赖 torch，训练由 subprocess 调起 python -m torch.distributed.launch。
--dry_run 只打印命令，不执行，用于本地测试命令拼接与各分支逻辑。

C3：核心代码已加 --save_best_only，打开后跳过周期保存与末轮强制保存，只留
checkpoint-best.pth。runner 对搜索组统一带上这个 flag。
"""

import argparse
import csv
import fcntl
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

GB = 1024 ** 3

# finetune_jieti.sh 的基准配置（除 6 个被搜索的超参外全部固定）。
# 与中心组脚本保持一致；6 个搜索超参由每组计划覆盖。
BASE_ARGS = [
    "--batch_size", "2",
    "--model", "vit_base_patch16_input896x448_win_dec64_8glb_sl1",
    "--num_mask_patches", "784",
    "--max_mask_patches_per_block", "392",
    "--epochs", "51",
    "--clip_grad", "3.0",
    # VGG 修复后的 style 权重：2026-10-02 ckpt14 梯度范数标定（所有组共用）。
    "--style_weight", "14.73",
    "--layer_decay", "0.8",
    "--drop_path", "0.1",
    "--input_size", "896", "448",
    "--augmentation_policy", "finetune",
    "--adv_warmup_epochs", "8",
    "--edge_warmup_epochs", "10",
    "--loss_warmup_duration", "8",
    "--adv_weight_final", "0.3",
    "--edge_weight_final", "0.2",
    "--no_gan",
    "--num_mask_annotations_bf", "11",
    "--num_mask_annotations_jt", "1",
    "--mask_coverage_threshold", "0.1",
    "--jieti_loss",
    "--jieti_soft_fg", "linear",
    "--jieti_k_max", "4",
    "--jieti_pool", "224",
    "--jieti_valid_ink", "200",
    "--jieti_pred_mass_ratio", "0.1",
    # 三项相对系数：2026-10-02 ckpt14 梯度范数标定（所有组共用）。
    "--jieti_w_centroid", "0.9688",
    "--jieti_w_logsigma", "0.5409",
    "--jieti_w_shape", "1.0",
    "--vis_every_epoch",
    "--val_tb_image_limit", "76",
    "--val_tb_images_per_batch", "2",
    "--grad_log_interval", "5",
]

EPOCHS = 51          # 与 BASE_ARGS 的 --epochs 一致
LAST_EPOCH = EPOCHS - 1  # 完成标志：log.txt 里出现 epoch==50 的行

SUMMARY_FIELDS = [
    "group", "alpha_jt", "w", "u", "p", "lr", "accum_iter", "warmup_epochs",
    "best_epoch", "best_S",
    "L1_JT_raw", "L1_JT_ratio",
    "L1_BF_raw", "L1_BF_ratio",
    "J_raw", "J_ratio",
    "status",
]


def read_plan(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_s_baseline(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {"L1_JT": d["L1_JT"], "L1_BF": d["L1_BF"], "J": d["J"]}


def parse_log(log_path):
    """读 log.txt，返回每行解析出的 dict 列表（跳过坏行）。"""
    rows = []
    if not os.path.exists(log_path):
        return rows
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def is_complete(output_dir):
    """完成标志：log.txt 中存在 epoch == LAST_EPOCH 的记录。"""
    for row in parse_log(os.path.join(output_dir, "log.txt")):
        if row.get("epoch") == LAST_EPOCH:
            return True
    return False


def has_run_evidence(output_dir):
    """目录里是否有真正开跑过的痕迹：log.txt 或任意 checkpoint。
    dry-run 残留只有 search_hparams.json，不算开跑过。"""
    if os.path.isfile(os.path.join(output_dir, "log.txt")):
        return True
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            if name.startswith("checkpoint") and name.endswith(".pth"):
                return True
    return False


def fill_missing_s(log_rows, s_base):
    """缺 test_S 的行用 s_baseline 重算，公式同 main_train.py compute_s：
    S = L1_JT/L1_JT_base + L1_BF/L1_BF_base + J/J_base。返回是否重算过。
    第 1 组启动时还没有 s_baseline.json，log.txt 里没有 test_S，靠这里补。"""
    if not s_base:
        return False
    recomputed = False
    for r in log_rows:
        if r.get("test_S") is not None:
            continue
        if all(r.get(k) is not None for k in ("test_L1_JT", "test_L1_BF", "test_jieti_J")):
            r["test_S"] = (r["test_L1_JT"] / s_base["L1_JT"]
                           + r["test_L1_BF"] / s_base["L1_BF"]
                           + r["test_jieti_J"] / s_base["J"])
            recomputed = True
    return recomputed


def pick_best_epoch(log_rows):
    """返回 test_S 最小的那一行；缺 test_S 时退回 test_loss。"""
    scored = [r for r in log_rows if "test_S" in r and r["test_S"] is not None]
    key = "test_S"
    if not scored:
        scored = [r for r in log_rows if "test_loss" in r]
        key = "test_loss"
    if not scored:
        return None, None
    best = min(scored, key=lambda r: r[key])
    return best, key


def estimate_required_bytes(search_root, margin_gb, n_groups=1):
    """（最近一组 best 的大小（没有取 2G）× 2）× n_groups + 余量。
    n_groups = 正在跑的组数 + 1：并发时给还在写 checkpoint 的组也留出空间。"""
    best_size = None
    newest_mtime = -1.0
    if os.path.isdir(search_root):
        for name in os.listdir(search_root):
            ckpt = os.path.join(search_root, name, "checkpoint-best.pth")
            if os.path.isfile(ckpt):
                mt = os.path.getmtime(ckpt)
                if mt > newest_mtime:
                    newest_mtime = mt
                    best_size = os.path.getsize(ckpt)
    if best_size is None:
        best_size = 2 * GB
    return best_size * 2 * n_groups + int(margin_gb * GB)


def disk_ok(output_root, required_bytes):
    """检查 output_root 所在盘的剩余空间是否 >= required_bytes。"""
    probe = output_root
    while probe and not os.path.exists(probe):
        probe = os.path.dirname(probe)
    if not probe:
        probe = "."
    free = shutil.disk_usage(probe).free
    return free >= required_bytes, free


def expand_glob(pattern):
    """展开通配符，和 bash 在 shell 里展开 *.json 的效果一致。
    bash 的通配符按字典序排列，这里用 sorted(glob.glob()) 对齐。
    空匹配直接报错退出，不启动训练（避免把字面 '*.json' 当文件名传下去）。"""
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(f"错误：通配符 {pattern} 没有匹配到任何文件，终止。", file=sys.stderr)
        sys.exit(4)
    return matches


def build_cmd(python_bin, master_port, output_dir, row, args):
    """拼接单组训练命令。6 个超参来自 row，其余来自 BASE_ARGS。"""
    log_dir = os.path.join(output_dir, "logs")
    cmd = [
        python_bin, "-m", "torch.distributed.launch",
        "--nproc_per_node=2",
        "--master_port=%d" % master_port,
        "--use_env", "main_train.py",
    ]
    cmd += BASE_ARGS
    # 搜索超参覆盖。
    cmd += [
        "--accum_iter", str(row["accum_iter"]),
        "--warmup_epochs", str(row["warmup_epochs"]),
        "--lr", str(row["lr"]),
        "--jieti_alpha_jt", str(row["alpha_jt"]),
        "--jieti_w", str(row["w"]),
        "--jieti_struct_pair_prob", str(row["p"]),
    ]
    # 固定路径与开关。
    cmd += [
        "--save_freq", str(args.save_freq),
        "--save_best",
        "--save_best_only",  # C3：搜索组只保留 checkpoint-best.pth
        "--data_path", args.data_path + "/",
    ]
    # json 通配符在 runner 里展开（subprocess 参数列表不经过 shell，* 不会被展开）。
    # 和 finetune_jieti.sh 在 shell 里展开的文件、顺序完全一致（字典序）。
    cmd += ["--json_path"] + expand_glob(os.path.join(args.data_path, "train_json_new", "*.json"))
    cmd += ["--val_json_path"] + expand_glob(os.path.join(args.data_path, "val_json_new", "*.json"))
    cmd += [
        "--output_dir", output_dir,
        "--log_dir", log_dir,
        "--finetune", args.pretrain_ckpt,
        "--semantic_mask_dir", os.path.join(args.data_path, "font", "train", "new"),
        "--fixed_pair_path", args.fixed_pair_path,
        "--s_baseline_path", args.s_baseline_path,
    ]
    return cmd


def write_search_hparams(output_dir, row):
    """把本组抽到的超参另存一份（步骤 10 / hparams.json 由核心代码写）。"""
    os.makedirs(output_dir, exist_ok=True)
    hp = {k: row[k] for k in
          ("group", "alpha_jt", "w", "u", "p", "lr", "accum_iter", "warmup_epochs")}
    with open(os.path.join(output_dir, "search_hparams.json"), "w", encoding="utf-8") as f:
        json.dump(hp, f, ensure_ascii=False, indent=2)


def append_summary(summary_path, record):
    """按 group 幂等写入：已有该组则替换该行，否则追加。
    这样断点续跑/重复运行不会在 summary.csv 里堆叠重复行。
    读-改-写整段持有 <summary>.lock 的排他锁：本进程内是串行写，锁防的是另起一个
    runner（例如事后单独补第 1 组）同时写同一个 summary.csv。"""
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path + ".lock", "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        _append_summary_locked(summary_path, record)


def _append_summary_locked(summary_path, record):
    rows = []
    if os.path.exists(summary_path):
        with open(summary_path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f)]
    key = str(record["group"])
    replaced = False
    for i, r in enumerate(rows):
        if str(r.get("group")) == key:
            rows[i] = record
            replaced = True
            break
    if not replaced:
        rows.append(record)

    def _group_sort_key(r):
        try:
            return (0, int(r["group"]))
        except (ValueError, KeyError, TypeError):
            return (1, str(r.get("group", "")))

    rows.sort(key=_group_sort_key)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in SUMMARY_FIELDS})


def summarize_group(row, output_dir, s_base, status):
    """从 output_dir/log.txt 取 best epoch，组装 summary 行。"""
    log_rows = parse_log(os.path.join(output_dir, "log.txt"))
    if fill_missing_s(log_rows, s_base):
        status = status + ";S_recomputed"
    best, key = pick_best_epoch(log_rows)
    rec = {k: row.get(k, "") for k in
           ("group", "alpha_jt", "w", "u", "p", "lr", "accum_iter", "warmup_epochs")}
    rec["status"] = status
    if best is None:
        rec.update({"best_epoch": "", "best_S": "",
                    "L1_JT_raw": "", "L1_JT_ratio": "",
                    "L1_BF_raw": "", "L1_BF_ratio": "",
                    "J_raw": "", "J_ratio": ""})
        return rec
    rec["best_epoch"] = best.get("epoch", "")
    # 按 test_loss 退回选出的 best 不填 best_S，免得 loss 值混进 S 排名。
    rec["best_S"] = best.get("test_S", "") if key == "test_S" else ""
    l1_jt = best.get("test_L1_JT")
    l1_bf = best.get("test_L1_BF")
    j = best.get("test_jieti_J")
    rec["L1_JT_raw"] = l1_jt if l1_jt is not None else ""
    rec["L1_BF_raw"] = l1_bf if l1_bf is not None else ""
    rec["J_raw"] = j if j is not None else ""
    def _ratio(val, base_key):
        if val is None or not s_base:
            return ""
        denom = s_base[base_key]
        return (val / denom) if denom else ""  # baseline 为 0 时留空，避免除零

    rec["L1_JT_ratio"] = _ratio(l1_jt, "L1_JT")
    rec["L1_BF_ratio"] = _ratio(l1_bf, "L1_BF")
    rec["J_ratio"] = _ratio(j, "J")
    return rec


def prepare_group(row, args, s_base, summary_path, search_root):
    """非第 1 组：判断跳过/重试。已完成返回 None（并写 summary），否则返回 (output_dir, retry)。"""
    group = int(row["group"])
    base_name = "group%02d" % group
    base_dir = os.path.join(search_root, base_name)

    # 断点续跑：已完成直接跳过。
    if is_complete(base_dir):
        print(f"[组 {group}] {base_dir} 已完成，跳过。")
        if not args.dry_run:
            append_summary(summary_path, summarize_group(row, base_dir, s_base, "skipped(done)"))
        return None

    # 目录存在但未完成。区分两种情况：
    #   - 没有开跑痕迹（只有 dry-run 残留的 search_hparams.json）→ 当作从未开跑，
    #     直接用原目录名开跑，不加 _retry。
    #   - 有 log.txt 或 checkpoint（真跑过但没到 epoch 50，崩溃）→ 输出到 _retry<n>，不删旧目录。
    output_dir = base_dir
    retry = 0
    if os.path.exists(base_dir) and has_run_evidence(base_dir):
        retry = 1
        while os.path.exists(os.path.join(search_root, "%s_retry%d" % (base_name, retry))):
            # 已完成的重试目录也算完成。
            cand = os.path.join(search_root, "%s_retry%d" % (base_name, retry))
            if is_complete(cand):
                print(f"[组 {group}] 重试目录 {cand} 已完成，跳过。")
                if not args.dry_run:
                    append_summary(summary_path, summarize_group(row, cand, s_base, "skipped(done)"))
                return None
            retry += 1
        output_dir = os.path.join(search_root, "%s_retry%d" % (base_name, retry))
        print(f"[组 {group}] 旧目录 {base_dir} 未完成，不删除，重跑到 {output_dir}。")
    return output_dir, retry


def parse_gpu_slots(spec):
    """"0,1;2,3" → ["0,1", "2,3"]。None → [None]（单槽，沿用继承的 CUDA_VISIBLE_DEVICES，不查卡）。"""
    if spec is None:
        return [None]
    # 卡号逐个去空格，"0, 1" 也能和 nvidia-smi 的 index 对上。
    slots = [",".join(g.strip() for g in s.split(",")) for s in spec.split(";") if s.strip()]
    for s in slots:
        if len(s.split(",")) != 2:
            print(f"错误：槽 {s} 不是 2 张卡（每组 --nproc_per_node=2）。", file=sys.stderr)
            sys.exit(2)
    return slots


def _nvidia_smi(query_flag, fields):
    out = subprocess.run(["nvidia-smi", query_flag + "=" + fields, "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    return [[c.strip() for c in line.split(",")] for line in out.splitlines() if line.strip()]


def group1_procs(group1_dir):
    """命令行里带 --output_dir <group1_dir> 的进程，返回 [(pid, CUDA_VISIBLE_DEVICES 或 None)]。
    --output_dir 的值按该进程的 cwd 解析成真实路径再比，写法不同（相对/绝对/./）也能认出。
    读不到环境变量时记 None，按占用全部卡处理。"""
    target = os.path.realpath(group1_dir)
    pat = re.compile(r"(?:^| )--output_dir (\S+)")
    found = []
    for pid in os.listdir("/proc") if os.path.isdir("/proc") else []:
        if not pid.isdigit():
            continue
        try:
            with open("/proc/%s/cmdline" % pid, "rb") as f:
                cmdline = f.read().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        hit = False
        for val in pat.findall(cmdline):
            if not os.path.isabs(val):
                try:
                    val = os.path.join(os.readlink("/proc/%s/cwd" % pid), val)
                except OSError:
                    pass
            if os.path.realpath(val) == target:
                hit = True
        if not hit:
            continue
        cvd = None
        try:
            with open("/proc/%s/environ" % pid, "rb") as f:
                for kv in f.read().split(b"\0"):
                    if kv.startswith(b"CUDA_VISIBLE_DEVICES="):
                        cvd = kv.split(b"=", 1)[1].decode()
        except OSError:
            pass
        found.append((int(pid), cvd))
    return found


def slot_busy_reasons(gpus, group1_dir):
    """槽内 GPU 被占用的原因列表；空列表 = 可以开新组。
    1) nvidia-smi 计算进程（按 index→uuid 映射到本槽的卡）；
    2) 第 1 组的进程（CUDA_VISIBLE_DEVICES 与本槽相交，或读不到时视为占用全部卡）。
    nvidia-smi 失败时视为占用，宁可多等。"""
    want = set(gpus.split(","))
    reasons = []
    try:
        idx2uuid = {r[0]: r[1] for r in _nvidia_smi("--query-gpu", "index,uuid")}
        uuid2idx = {u: i for i, u in idx2uuid.items()}
        for uuid, pid in _nvidia_smi("--query-compute-apps", "gpu_uuid,pid"):
            idx = uuid2idx.get(uuid)
            if idx in want:
                reasons.append(f"GPU{idx} 上有进程 {pid}")
    except (OSError, subprocess.CalledProcessError) as e:
        reasons.append(f"nvidia-smi 查询失败（{e}）")
    g1 = [(pid, cvd) for pid, cvd in group1_procs(group1_dir)
          if cvd is None or set(cvd.split(",")) & want]
    if g1:
        # 只给固定文本：dataloader worker 每个 epoch 换 PID，写进来会让等待消息反复变化重打。
        cvds = sorted({str(c) for _, c in g1})
        reasons.append(f"第 1 组训练进程仍在用这些卡（CUDA_VISIBLE_DEVICES={','.join(cvds)}）")
    return reasons


def finish_group(row, output_dir, retry, ret, s_base, summary_path):
    group = int(row["group"])
    if ret != 0:
        print(f"[组 {group}] 训练进程非零退出码 {ret}。", file=sys.stderr)
        status = "failed(rc=%d)" % ret
    elif is_complete(output_dir):
        status = "done" if retry == 0 else "done(retry%d)" % retry
    else:
        status = "incomplete"
    append_summary(summary_path, summarize_group(row, output_dir, s_base, status))
    return status


def summarize_group1(row, args, s_base, summary_path):
    g1_dir = args.group1_dir
    status = "reused" if is_complete(g1_dir) else "reused_incomplete"
    print(f"[组 1] 复用正式训练目录 {g1_dir}（{status}），不重跑。")
    if not args.dry_run:
        append_summary(summary_path, summarize_group(row, g1_dir, s_base, status))
    return status


def now():
    return time.strftime("%m-%d %H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", default="models/jieti_search/search_plan.csv")
    parser.add_argument("--s_baseline_path", default="models/jieti_search/s_baseline.json")
    parser.add_argument("--fixed_pair_path", default="fontdata_example/val_pairs_fixed.json")
    parser.add_argument("--data_path", default="fontdata_example")
    parser.add_argument("--pretrain_ckpt", default="models/vit_base_font/checkpoint-14.pth")
    parser.add_argument("--output_root", default="models/jieti_search",
                        help="组目录与 summary.csv 的根；磁盘检查也看这个盘")
    parser.add_argument("--group1_dir", default="models/finetune_jieti",
                        help="第 1 组复用的正式训练输出目录")
    parser.add_argument("--summary", default=None,
                        help="默认 <output_root>/summary.csv")
    parser.add_argument("--python_bin", default=sys.executable or "python")
    parser.add_argument("--master_port_base", type=int, default=29600)
    parser.add_argument("--save_freq", type=int, default=9999,
                        help="调大以避免中间周期 checkpoint；epoch 0/最后仍会存，见文件头说明")
    parser.add_argument("--disk_margin_gb", type=float, default=5.0,
                        help="TB/vis 余量（GB）")
    parser.add_argument("--start_group", type=int, default=1,
                        help="从第几组开始（含）；默认 1，会处理中心组的汇总")
    parser.add_argument("--end_group", type=int, default=None,
                        help="跑到第几组为止（含）；默认 None 表示不设上限")
    parser.add_argument("--gpu_slots", default=None,
                        help='卡槽，如 "0,1;2,3"。每槽同时只跑一组，槽间并行。'
                             "不给时单槽串行，沿用继承的 CUDA_VISIBLE_DEVICES，不查卡")
    parser.add_argument("--poll_interval", type=float, default=60.0,
                        help="等卡/等训练结束的轮询间隔（秒）")
    parser.add_argument("--dry_run", action="store_true",
                        help="只打印命令和分支判断，不执行训练")
    args = parser.parse_args()

    summary_path = args.summary or os.path.join(args.output_root, "summary.csv")
    plan = read_plan(args.plan)
    s_base = None
    if os.path.exists(args.s_baseline_path):
        s_base = read_s_baseline(args.s_baseline_path)
    else:
        print(f"警告：s_baseline 不存在（{args.s_baseline_path}），比值列将留空。",
              file=sys.stderr)

    search_root = args.output_root
    slots = parse_gpu_slots(args.gpu_slots)
    selected = [r for r in plan
                if int(r["group"]) >= args.start_group
                and (args.end_group is None or int(r["group"]) <= args.end_group)]
    group1_row = next((r for r in selected if int(r["group"]) == 1), None)
    queue = [r for r in selected if int(r["group"]) != 1]

    if args.dry_run:
        # 只演示槽分配（按 plan 顺序轮流），不等卡、不写文件。
        for i, gpus in enumerate(slots):
            if gpus is not None:
                reasons = slot_busy_reasons(gpus, args.group1_dir)
                print(f"[dry_run] 槽 {i}（GPU {gpus}）当前："
                      + ("空闲" if not reasons else "占用，正式运行会等待：" + "；".join(reasons)))
        k = 0
        for row in queue:
            group = int(row["group"])
            prep = prepare_group(row, args, s_base, summary_path, search_root)
            if prep is None:
                print(f"[组 {group}] 状态：skipped(done)")
                continue
            output_dir, _ = prep
            gpus = slots[k % len(slots)]
            k += 1
            required = estimate_required_bytes(search_root, args.disk_margin_gb, min(len(slots), 2))
            ok, free = disk_ok(args.output_root, required)
            print(f"[组 {group}] 磁盘检查：需 {required/GB:.1f}G，剩 {free/GB:.1f}G。")
            cmd = build_cmd(args.python_bin, args.master_port_base + group, output_dir, row, args)
            print(f"[组 {group}] 槽 {(k - 1) % len(slots)} CUDA_VISIBLE_DEVICES={gpus} "
                  f"启动：{' '.join(cmd)}")
            print(f"[组 {group}] 状态：dry_run")
        if group1_row is not None:
            procs = group1_procs(args.group1_dir)
            print(f"[组 1] 训练进程：{procs or '无'}；正式运行会等它们全部退出后再汇总。")
            summarize_group1(group1_row, args, s_base, summary_path)
        return

    # running: 槽号 → (Popen, row, output_dir, retry, 日志文件句柄)
    running = {}
    group1_pending = group1_row is not None
    stop_dispatch = False
    exit_code = 0
    last_wait_msg = {}
    while queue or running or group1_pending:
        # 1) 回收跑完的组（summary 只在主进程里串行写）。
        for si in list(running):
            proc, row, output_dir, retry, lf = running[si]
            ret = proc.poll()
            if ret is None:
                continue
            lf.close()
            del running[si]
            status = finish_group(row, output_dir, retry, ret, s_base, summary_path)
            print(f"[{now()}] [组 {row['group']}] 状态：{status}（槽 {si} 空出）", flush=True)

        # 2) 第 1 组：训练进程全部退出后才汇总，不在它没跑完时标成完成。
        if group1_pending and not group1_procs(args.group1_dir):
            status = summarize_group1(group1_row, args, s_base, summary_path)
            print(f"[{now()}] [组 1] 状态：{status}", flush=True)
            group1_pending = False

        # 3) 给空槽派新组，按 plan 顺序领取。
        for si, gpus in enumerate(slots):
            if stop_dispatch or not queue or si in running:
                continue
            if gpus is not None:
                reasons = slot_busy_reasons(gpus, args.group1_dir)
                if reasons:
                    msg = "；".join(reasons)
                    # 原因不变时不重复打印，免得 runner.log 每分钟刷一行。
                    if last_wait_msg.get(si) != msg:
                        print(f"[{now()}] 槽 {si}（GPU {gpus}）等待（每 {args.poll_interval:.0f}s 查一次）：{msg}",
                              flush=True)
                        last_wait_msg[si] = msg
                    continue
                last_wait_msg.pop(si, None)
            # 跳过已完成的组，直到拿到一组要跑的。
            prep = None
            while queue and prep is None:
                row = queue[0]
                prep = prepare_group(row, args, s_base, summary_path, search_root)
                if prep is None:
                    queue.pop(0)
                    print(f"[组 {row['group']}] 状态：skipped(done)", flush=True)
            if prep is None:
                break
            output_dir, retry = prep
            group = int(row["group"])
            # 磁盘检查（步骤 12）：给正在跑的组和本组都留出 best 的空间。
            required = estimate_required_bytes(search_root, args.disk_margin_gb, len(running) + 1)
            ok, free = disk_ok(args.output_root, required)
            print(f"[{now()}] [组 {group}] 磁盘检查：需 {required/GB:.1f}G，剩 {free/GB:.1f}G。",
                  flush=True)
            if not ok:
                print("磁盘不足，需用户确认；不再派新组，等在跑的组结束后退出。",
                      file=sys.stderr, flush=True)
                stop_dispatch = True
                exit_code = 3
                break
            cmd = build_cmd(args.python_bin, args.master_port_base + group, output_dir, row, args)
            env = os.environ.copy()
            if gpus is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpus
                # 让 CUDA 的卡号和 nvidia-smi 的 index 一致（都按 PCI 总线顺序）。
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            write_search_hparams(output_dir, row)
            os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
            lf = open(os.path.join(output_dir, "train.log"), "a", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
            running[si] = (proc, row, output_dir, retry, lf)
            queue.pop(0)
            print(f"[{now()}] [组 {group}] 槽 {si} CUDA_VISIBLE_DEVICES={gpus} PID {proc.pid} "
                  f"启动：{' '.join(cmd)}", flush=True)

        if stop_dispatch and not running and not group1_pending:
            break
        if not (queue or running or group1_pending):
            break
        time.sleep(args.poll_interval)
    if stop_dispatch:
        print(f"磁盘不足退出，剩 {len(queue)} 组未跑。", file=sys.stderr, flush=True)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
