#!/usr/bin/env python
"""T1-S 第二轮超参搜索 runner（PROGRESS T1-S）。

和旧 runner（tools/run_jieti_search.py，行为不变）的区别：
  - 基准配置是 T1-F 对照组 finetune_jieti_ctrl_g12.sh：--no_jt、p=0、α_jt/w/lr/accum/warmup
    由计划给出；每组 2 卡、--save_freq 9999 --save_best --save_best_only（只存 best）。
  - 每组训练结束、log.txt 有 epoch 50 之后，在本槽第一张卡上串行跑
    tools/eval_s_baseline.py 评 checkpoint-best.pth，输出 <组目录>/eval_s_tool_best.{json,log}；
    参数与 tools/ctrl_g12_then_eval.sh 的 b 步一致，只改 checkpoint 和输出路径。
    评测失败只记进 status，不阻塞下一组。槽要等评测结束才派新组。
  - summary.csv 同时记训练 log 口径和工具口径（eval_s_baseline）的 S 与三个比值；
    排序用工具口径 tool_S（tools/summarize_jieti_search.py --s_column tool_S）。
  - 第 1 组复用 --group1_dir（默认 models/finetune_jieti_ctrl_g12），不训练：工具口径读它的
    eval_s_tool_best.json；log 口径从它的 log.txt 按 s_baseline 分母重算（4 卡，val 240 条，仅供参考）。
  - 输出根目录 models/jieti_search2/，master_port = 29700 + group（旧搜索是 29600 + group）。
  - s_baseline.json 和 calibration_v2.json 默认读 models/jieti_search2/ 下的副本，可用参数覆盖；
    不存在就退出（exit 5），不自己生成。runner 不读写 models/jieti_search/ 下的任何文件。

沿用旧 runner 的安全机制（直接 import 旧 runner 的函数，不复制）：json 通配符在 runner 内展开、
dry-run 不写任何文件、has_run_evidence 和 _retry 规则、磁盘检查、summary 文件锁、
nvidia-smi 查卡。启动方式仍是 python -u。

跳过规则：组目录（或某个 _retryN）已完成且 eval_s_tool_best.json 有效 → 跳过，只刷新 summary；
已完成但没有有效评测结果 → 只补跑评测，不重训。
"""

import argparse
import csv
import fcntl
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_jieti_search as rjs  # noqa: E402  旧 runner，只用它的函数

GB = rjs.GB

# finetune_jieti_ctrl_g12.sh 的基准配置。被搜索的 5 个超参（accum_iter、warmup_epochs、lr、
# jieti_alpha_jt、jieti_w）由计划覆盖；p 固定 0 写在这里，不从计划读。
BASE_ARGS = [
    "--batch_size", "2",
    "--model", "vit_base_patch16_input896x448_win_dec64_8glb_sl1",
    "--num_mask_patches", "784",
    "--max_mask_patches_per_block", "392",
    "--epochs", "51",
    "--clip_grad", "3.0",
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
    "--no_jt",
    "--num_mask_annotations_bf", "11",
    "--num_mask_annotations_jt", "1",
    "--mask_coverage_threshold", "0.1",
    "--jieti_loss",
    "--jieti_struct_pair_prob", "0",
    "--jieti_soft_fg", "linear",
    "--jieti_k_max", "4",
    "--jieti_pool", "224",
    "--jieti_valid_ink", "200",
    "--jieti_pred_mass_ratio", "0.1",
    "--jieti_w_centroid", "0.9688",
    "--jieti_w_logsigma", "0.5409",
    "--jieti_w_shape", "1.0",
    "--vis_every_epoch",
    "--val_tb_image_limit", "76",
    "--val_tb_images_per_batch", "2",
    "--grad_log_interval", "5",
]

PLAN_FIELDS = ("group", "alpha_jt", "w", "u", "p", "lr", "accum_iter", "warmup_epochs")
EVAL_JSON = "eval_s_tool_best.json"
EVAL_LOG = "eval_s_tool_best.log"
GROUP1_NOTE = "4卡，val 240 条，训练 log 口径仅供参考"

SUMMARY_FIELDS = list(PLAN_FIELDS) + [
    "best_epoch",
    "log_S", "log_L1_JT_ratio", "log_L1_BF_ratio", "log_J_ratio",
    "tool_S", "tool_L1_JT_ratio", "tool_L1_BF_ratio", "tool_J_ratio",
    "log_L1_JT_raw", "log_L1_BF_raw", "log_J_raw",
    "tool_L1_JT_raw", "tool_L1_BF_raw", "tool_J_raw", "tool_n_jt", "tool_n_bf",
    "output_dir", "status", "note",
]


# ---------- 读结果 ----------

def read_tool_eval(output_dir):
    """读 eval_s_tool_best.json；文件缺失、坏 json、三项有非有限值时返回 None。"""
    path = os.path.join(output_dir, EVAL_JSON)
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        vals = [float(d[k]) for k in ("L1_JT", "L1_BF", "J")]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not all(math.isfinite(v) for v in vals):
        return None
    return d


def log_best(output_dir, s_base):
    """按 s_baseline 分母从 log.txt 的原始分项重算每个 epoch 的 S，返回 (S 最小的行, 该 S)。
    搜索组训练时用的是同一份分母，重算值等于 test_S；第 1 组靠这里换成新分母。"""
    best, best_s = None, None
    for r in rjs.parse_log(os.path.join(output_dir, "log.txt")):
        if not all(r.get(k) is not None for k in ("test_L1_JT", "test_L1_BF", "test_jieti_J")):
            continue
        s = (r["test_L1_JT"] / s_base["L1_JT"] + r["test_L1_BF"] / s_base["L1_BF"]
             + r["test_jieti_J"] / s_base["J"])
        if best_s is None or s < best_s:
            best, best_s = r, s
    return best, best_s


def build_record(row, output_dir, s_base, status, note=""):
    rec = {k: row.get(k, "") for k in PLAN_FIELDS}
    rec.update({k: "" for k in SUMMARY_FIELDS if k not in rec})
    rec["output_dir"] = output_dir
    rec["status"] = status
    rec["note"] = note
    best, s = log_best(output_dir, s_base)
    if best is not None:
        rec["best_epoch"] = best.get("epoch", "")
        rec["log_S"] = s
        rec["log_L1_JT_raw"] = best["test_L1_JT"]
        rec["log_L1_BF_raw"] = best["test_L1_BF"]
        rec["log_J_raw"] = best["test_jieti_J"]
        rec["log_L1_JT_ratio"] = best["test_L1_JT"] / s_base["L1_JT"]
        rec["log_L1_BF_ratio"] = best["test_L1_BF"] / s_base["L1_BF"]
        rec["log_J_ratio"] = best["test_jieti_J"] / s_base["J"]
    tool = read_tool_eval(output_dir)
    if tool is not None:
        r_jt = tool["L1_JT"] / s_base["L1_JT"]
        r_bf = tool["L1_BF"] / s_base["L1_BF"]
        r_j = tool["J"] / s_base["J"]
        rec.update({"tool_L1_JT_raw": tool["L1_JT"], "tool_L1_BF_raw": tool["L1_BF"],
                    "tool_J_raw": tool["J"], "tool_n_jt": tool.get("n_jt", ""),
                    "tool_n_bf": tool.get("n_bf", ""),
                    "tool_L1_JT_ratio": r_jt, "tool_L1_BF_ratio": r_bf, "tool_J_ratio": r_j,
                    "tool_S": r_jt + r_bf + r_j})
    return rec


def append_summary(summary_path, record):
    """按 group 幂等写入（已有该组就替换），持 <summary>.lock 排他锁读-改-写。
    与旧 runner 的 append_summary 相同，只是列不同。"""
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path + ".lock", "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        rows = []
        if os.path.exists(summary_path):
            with open(summary_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        key = str(record["group"])
        for i, r in enumerate(rows):
            if str(r.get("group")) == key:
                rows[i] = record
                break
        else:
            rows.append(record)

        def _key(r):
            try:
                return (0, int(r["group"]))
            except (ValueError, KeyError, TypeError):
                return (1, str(r.get("group", "")))

        rows.sort(key=_key)
        tmp = summary_path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in SUMMARY_FIELDS})
        os.replace(tmp, summary_path)


# ---------- 命令 ----------

def build_train_cmd(python_bin, master_port, output_dir, row, args):
    cmd = [python_bin, "-m", "torch.distributed.launch",
           "--nproc_per_node=2", "--master_port=%d" % master_port,
           "--use_env", "main_train.py"]
    cmd += BASE_ARGS
    cmd += [
        "--accum_iter", str(row["accum_iter"]),
        "--warmup_epochs", str(row["warmup_epochs"]),
        "--lr", str(row["lr"]),
        "--jieti_alpha_jt", str(row["alpha_jt"]),
        "--jieti_w", str(row["w"]),
        "--save_freq", str(args.save_freq),
        "--save_best",
        "--save_best_only",
        "--data_path", args.data_path + "/",
    ]
    cmd += ["--json_path"] + rjs.expand_glob(os.path.join(args.data_path, "train_json_new", "*.json"))
    cmd += ["--val_json_path"] + rjs.expand_glob(os.path.join(args.data_path, "val_json_new", "*.json"))
    cmd += [
        "--output_dir", output_dir,
        "--log_dir", os.path.join(output_dir, "logs"),
        "--finetune", args.pretrain_ckpt,
        "--semantic_mask_dir", os.path.join(args.data_path, "font", "train", "new"),
        "--fixed_pair_path", args.fixed_pair_path,
        "--s_baseline_path", args.s_baseline_path,
    ]
    return cmd


def build_eval_cmd(python_bin, output_dir, args):
    """与 tools/ctrl_g12_then_eval.sh b 步相同的参数，只改 checkpoint 和 output_json。"""
    return ([python_bin, "-u", "tools/eval_s_baseline.py",
             "--checkpoint", os.path.join(output_dir, "checkpoint-best.pth"),
             "--data_path", args.data_path + "/",
             "--val_json_path"]
            + rjs.expand_glob(os.path.join(args.data_path, "val_json_new", "*.json"))
            + ["--semantic_mask_dir", os.path.join(args.data_path, "font", "train", "new"),
               "--fixed_pair_path", args.fixed_pair_path,
               "--calibration_json", args.calibration_json,
               "--output_json", os.path.join(output_dir, EVAL_JSON)])


def slot_env(gpus, first_only):
    env = os.environ.copy()
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpus.split(",")[0] if first_only else gpus
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env


# ---------- 判断每组要做什么 ----------

def prepare_group(row, search_root):
    """返回 (action, output_dir, retry)，action ∈ skip / eval / train。
    完成判断、has_run_evidence、_retry 规则与旧 runner 的 prepare_group 相同；
    多出来的一点：已完成但没有有效评测结果 → eval（只补评测）。"""
    group = int(row["group"])
    base_name = "group%02d" % group
    base_dir = os.path.join(search_root, base_name)

    def _done(d, retry):
        return ("skip" if read_tool_eval(d) is not None else "eval"), d, retry

    if rjs.is_complete(base_dir):
        return _done(base_dir, 0)
    if not (os.path.exists(base_dir) and rjs.has_run_evidence(base_dir)):
        return "train", base_dir, 0
    retry = 1
    while os.path.exists(os.path.join(search_root, "%s_retry%d" % (base_name, retry))):
        cand = os.path.join(search_root, "%s_retry%d" % (base_name, retry))
        if rjs.is_complete(cand):
            return _done(cand, retry)
        retry += 1
    return "train", os.path.join(search_root, "%s_retry%d" % (base_name, retry)), retry


def train_status(ret, output_dir, retry):
    if ret != 0:
        return "failed(rc=%d)" % ret
    if rjs.is_complete(output_dir):
        return "done" if retry == 0 else "done(retry%d)" % retry
    return "incomplete"


def eval_suffix(ret, output_dir):
    if ret == 0 and read_tool_eval(output_dir) is not None:
        return ""
    if ret != 0:
        return ";eval_failed(rc=%d)" % ret
    return ";eval_failed(no_valid_json)"


def now():
    return time.strftime("%m-%d %H:%M:%S")


def log(msg, err=False):
    print(f"[{now()}] {msg}", file=sys.stderr if err else sys.stdout, flush=True)


# ---------- 主流程 ----------

def check_inputs(args):
    """s_baseline / calibration / fixed_pair / ckpt14 必须存在，缺一个就退出 5，不自己生成。"""
    missing = [p for p in (args.s_baseline_path, args.calibration_json,
                           args.fixed_pair_path, args.pretrain_ckpt) if not os.path.isfile(p)]
    if missing:
        print("错误：输入文件不存在，退出：" + "，".join(missing), file=sys.stderr, flush=True)
        sys.exit(5)


def summarize_group1(row, args, s_base, summary_path):
    g1 = args.group1_dir
    status = "reused" if rjs.is_complete(g1) else "reused_incomplete"
    if read_tool_eval(g1) is None:
        status += ";eval_missing"
    rec = build_record(row, g1, s_base, status, note=GROUP1_NOTE)
    print(f"[组 1] 复用 {g1}（{status}）：log_S={rec['log_S']} best_epoch={rec['best_epoch']} "
          f"tool_S={rec['tool_S']}", flush=True)
    if not args.dry_run:
        append_summary(summary_path, rec)
    return status


def dry_run(args, slots, queue, group1_row, s_base, summary_path):
    """只打印分支判断和命令，不等卡、不写任何文件。"""
    for i, gpus in enumerate(slots):
        if gpus is not None:
            reasons = rjs.slot_busy_reasons(gpus, args.group1_dir)
            print(f"[dry_run] 槽 {i}（GPU {gpus}）当前："
                  + ("空闲" if not reasons else "占用，正式运行会等待：" + "；".join(reasons)))
    if group1_row is not None:
        summarize_group1(group1_row, args, s_base, summary_path)
    k = 0
    for row in queue:
        group = int(row["group"])
        action, output_dir, _ = prepare_group(row, args.output_root)
        if action == "skip":
            print(f"[组 {group}] {output_dir} 已完成且已评测：skip")
            continue
        gpus = slots[k % len(slots)]
        si = k % len(slots)
        k += 1
        if action == "train":
            required = rjs.estimate_required_bytes(args.output_root, args.disk_margin_gb,
                                                   min(len(slots), 2))
            ok, free = rjs.disk_ok(args.output_root, required)
            print(f"[组 {group}] 磁盘检查：需 {required/GB:.1f}G，剩 {free/GB:.1f}G。")
            cmd = build_train_cmd(args.python_bin, args.master_port_base + group,
                                  output_dir, row, args)
            print(f"[组 {group}] 槽 {si} CUDA_VISIBLE_DEVICES={gpus} 训练：{' '.join(cmd)}")
        else:
            print(f"[组 {group}] {output_dir} 已完成但缺评测结果：只补评测")
        ecmd = build_eval_cmd(args.python_bin, output_dir, args)
        first = gpus.split(",")[0] if gpus else None
        print(f"[组 {group}] 槽 {si} CUDA_VISIBLE_DEVICES={first} 评测：{' '.join(ecmd)}")
        print(f"[组 {group}] 状态：dry_run（{action}）")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", default="models/jieti_search2/search_plan.csv")
    parser.add_argument("--s_baseline_path", default="models/jieti_search2/s_baseline.json")
    parser.add_argument("--calibration_json", default="models/jieti_search2/calibration_v2.json",
                        help="eval_s_baseline.py 的 --calibration_json")
    parser.add_argument("--fixed_pair_path", default="fontdata_example/val_pairs_fixed.json")
    parser.add_argument("--data_path", default="fontdata_example")
    parser.add_argument("--pretrain_ckpt", default="models/vit_base_font/checkpoint-14.pth")
    parser.add_argument("--output_root", default="models/jieti_search2")
    parser.add_argument("--group1_dir", default="models/finetune_jieti_ctrl_g12",
                        help="第 1 组复用的目录（不训练）")
    parser.add_argument("--summary", default=None, help="默认 <output_root>/summary.csv")
    parser.add_argument("--python_bin", default=sys.executable or "python")
    parser.add_argument("--master_port_base", type=int, default=29700)
    parser.add_argument("--save_freq", type=int, default=9999)
    parser.add_argument("--disk_margin_gb", type=float, default=5.0)
    parser.add_argument("--start_group", type=int, default=1)
    parser.add_argument("--end_group", type=int, default=None)
    parser.add_argument("--gpu_slots", default=None, help='如 "0,1;2,3"')
    parser.add_argument("--poll_interval", type=float, default=60.0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    check_inputs(args)
    s_base = rjs.read_s_baseline(args.s_baseline_path)
    print(f"s_baseline 分母：L1_JT={s_base['L1_JT']} L1_BF={s_base['L1_BF']} J={s_base['J']}",
          flush=True)
    summary_path = args.summary or os.path.join(args.output_root, "summary.csv")
    slots = rjs.parse_gpu_slots(args.gpu_slots)
    plan = rjs.read_plan(args.plan)
    selected = [r for r in plan
                if int(r["group"]) >= args.start_group
                and (args.end_group is None or int(r["group"]) <= args.end_group)]
    group1_row = next((r for r in selected if int(r["group"]) == 1), None)
    queue = [r for r in selected if int(r["group"]) != 1]

    if args.dry_run:
        dry_run(args, slots, queue, group1_row, s_base, summary_path)
        return

    # running[槽号] = dict(stage=train/eval, proc, row, dir, retry, lf, status)
    running = {}
    group1_pending = group1_row is not None
    stop_dispatch = False
    exit_code = 0
    last_wait_msg = {}

    def start_eval(si, gpus, row, output_dir, retry, status):
        if not os.path.isfile(os.path.join(output_dir, "checkpoint-best.pth")):
            status += ";eval_failed(no_ckpt)"
            append_summary(summary_path, build_record(row, output_dir, s_base, status))
            log(f"[组 {row['group']}] 没有 checkpoint-best.pth，不评测。状态：{status}", err=True)
            return
        cmd = build_eval_cmd(args.python_bin, output_dir, args)
        env = slot_env(gpus, first_only=True)
        lf = open(os.path.join(output_dir, EVAL_LOG), "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        running[si] = dict(stage="eval", proc=proc, row=row, dir=output_dir, retry=retry,
                           lf=lf, status=status)
        log(f"[组 {row['group']}] 槽 {si} CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')} "
            f"PID {proc.pid} 评测：{' '.join(cmd)}")

    while queue or running or group1_pending:
        # 1) 回收：训练结束 → 同槽接评测；评测结束 → 写 summary、空出槽。
        for si in list(running):
            st = running[si]
            ret = st["proc"].poll()
            if ret is None:
                continue
            st["lf"].close()
            del running[si]
            row, d = st["row"], st["dir"]
            if st["stage"] == "train":
                status = train_status(ret, d, st["retry"])
                log(f"[组 {row['group']}] 训练结束 rc={ret}，状态：{status}")
                if status.startswith("done"):
                    # 同槽接评测；没有 best ckpt 时 start_eval 直接写 summary，槽空出
                    start_eval(si, slots[si], row, d, st["retry"], status)
                    continue
                append_summary(summary_path, build_record(row, d, s_base, status))
                log(f"[组 {row['group']}] 状态：{status}（槽 {si} 空出）")
            else:
                status = st["status"] + eval_suffix(ret, d)
                append_summary(summary_path, build_record(row, d, s_base, status))
                log(f"[组 {row['group']}] 评测结束 rc={ret}，状态：{status}（槽 {si} 空出）")

        # 2) 第 1 组：等它的训练进程都不在了再汇总（它早已跑完，正常第一轮就写）。
        if group1_pending and not rjs.group1_procs(args.group1_dir):
            summarize_group1(group1_row, args, s_base, summary_path)
            group1_pending = False

        # 3) 给空槽派活。
        for si, gpus in enumerate(slots):
            if stop_dispatch or not queue or si in running:
                continue
            if gpus is not None:
                reasons = rjs.slot_busy_reasons(gpus, args.group1_dir)
                if reasons:
                    msg = "；".join(reasons)
                    if last_wait_msg.get(si) != msg:
                        log(f"槽 {si}（GPU {gpus}）等待（每 {args.poll_interval:.0f}s 查一次）：{msg}")
                        last_wait_msg[si] = msg
                    continue
                last_wait_msg.pop(si, None)
            while queue and si not in running and not stop_dispatch:
                row = queue.pop(0)
                group = int(row["group"])
                action, output_dir, retry = prepare_group(row, args.output_root)
                if action == "skip":
                    status = "skipped(done)"  # 实际目录记在 output_dir 列
                    append_summary(summary_path, build_record(row, output_dir, s_base, status))
                    log(f"[组 {group}] {output_dir} 已完成且已评测，跳过。")
                    continue
                if action == "eval":
                    log(f"[组 {group}] {output_dir} 已完成但缺评测结果，只补评测。")
                    start_eval(si, gpus, row, output_dir, retry, "skipped(done)")
                    continue
                required = rjs.estimate_required_bytes(args.output_root, args.disk_margin_gb,
                                                       len(running) + 1)
                ok, free = rjs.disk_ok(args.output_root, required)
                log(f"[组 {group}] 磁盘检查：需 {required/GB:.1f}G，剩 {free/GB:.1f}G。")
                if not ok:
                    queue.insert(0, row)
                    log("磁盘不足，需用户确认；不再派新组，等在跑的组结束后退出。", err=True)
                    stop_dispatch = True
                    exit_code = 3
                    break
                if retry:
                    log(f"[组 {group}] 旧目录未完成，不删除，重跑到 {output_dir}。")
                cmd = build_train_cmd(args.python_bin, args.master_port_base + group,
                                      output_dir, row, args)
                env = slot_env(gpus, first_only=False)
                rjs.write_search_hparams(output_dir, row)
                os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
                lf = open(os.path.join(output_dir, "train.log"), "a", encoding="utf-8")
                proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
                running[si] = dict(stage="train", proc=proc, row=row, dir=output_dir,
                                   retry=retry, lf=lf, status="")
                log(f"[组 {group}] 槽 {si} CUDA_VISIBLE_DEVICES={gpus} PID {proc.pid} "
                    f"训练：{' '.join(cmd)}")

        if not (queue and not stop_dispatch) and not running and not group1_pending:
            break
        time.sleep(args.poll_interval)
    if stop_dispatch:
        log(f"磁盘不足退出，剩 {len(queue)} 组未跑。", err=True)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
