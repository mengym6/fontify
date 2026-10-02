#!/usr/bin/env python
"""T1 结体 loss 超参随机搜索 runner（PROGRESS 步骤 9、10、12 和 C3）。

读取 search_plan.csv，从第 2 组起按顺序串行运行（每组用 2 卡，组间串行）。
第 1 组复用正式训练输出目录，不重跑，但照常汇总。

每组：以 finetune_jieti.sh 的基准配置为底，只覆盖本组抽到的 6 个超参
（alpha_jt、w、p、lr、accum_iter、warmup_epochs），并带上 --save_best、
--s_baseline_path、--fixed_pair_path、--vis_every_epoch。按 C3 只保留 best。

断点续跑：组目录已存在且 log.txt 里有最后一个 epoch 的记录 → 跳过。
目录存在但未完成（中途崩溃）→ 不删旧目录，输出到新目录 <name>_retry<n> 从头重跑。

磁盘：每组开跑前检查剩余空间（步骤 12）。估算 = 最近一组 best 的实际大小
（没有取 2G）× 2 + TB/vis 余量。不够则打印提示并退出，不删任何文件。

本脚本不依赖 torch，训练由 subprocess 调起 python -m torch.distributed.launch。
--dry_run 只打印命令，不执行，用于本地测试命令拼接与各分支逻辑。

注意（已交主控）：main_train.py L657 的周期 checkpoint 条件是
`epoch % save_freq == 0 or epoch+1 == epochs`，epoch 0 与最后一个 epoch 无论
save_freq 多大都会存，命令行无法关闭。runner 把 save_freq 设得很大以避免中间
周期 checkpoint，但 epoch 0 和 epoch 50 的 checkpoint 仍会落盘。
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys

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
    "--jieti_w_centroid", "1.0",
    "--jieti_w_logsigma", "1.0",
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


def estimate_required_bytes(search_root, margin_gb):
    """最近一组 best 的大小（没有取 2G）× 2 + 余量。"""
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
    return best_size * 2 + int(margin_gb * GB)


def disk_ok(output_root, required_bytes):
    """检查 output_root 所在盘的剩余空间是否 >= required_bytes。"""
    probe = output_root
    while probe and not os.path.exists(probe):
        probe = os.path.dirname(probe)
    if not probe:
        probe = "."
    free = shutil.disk_usage(probe).free
    return free >= required_bytes, free


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
        "--save_freq", str(args.save_freq),  # 调大以避免中间周期 checkpoint
        "--save_best",
        "--data_path", args.data_path + "/",
        "--json_path", os.path.join(args.data_path, "train_json_new", "*.json"),
        "--val_json_path", os.path.join(args.data_path, "val_json_new", "*.json"),
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
    这样断点续跑/重复运行不会在 summary.csv 里堆叠重复行。"""
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
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
    rec["best_S"] = best.get("test_S", best.get(key, ""))
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


def run_group(row, args, s_base, summary_path, search_root):
    """运行（或跳过/重试）一组，并追加 summary。返回状态字符串。"""
    group = int(row["group"])
    base_name = "group%02d" % group
    base_dir = os.path.join(search_root, base_name)

    # 第 1 组复用正式训练输出目录，不重跑。
    if group == 1:
        g1_dir = args.group1_dir
        if is_complete(g1_dir):
            status = "reused"
        else:
            status = "reused_incomplete"
        print(f"[组 {group}] 复用正式训练目录 {g1_dir}（{status}），不重跑。")
        append_summary(summary_path, summarize_group(row, g1_dir, s_base, status))
        return status

    # 断点续跑：已完成直接跳过。
    if is_complete(base_dir):
        print(f"[组 {group}] {base_dir} 已完成，跳过。")
        append_summary(summary_path, summarize_group(row, base_dir, s_base, "skipped(done)"))
        return "skipped(done)"

    # 目录存在但未完成（崩溃）→ 输出到新目录 _retry<n>，不删旧目录。
    output_dir = base_dir
    retry = 0
    if os.path.exists(base_dir):
        retry = 1
        while os.path.exists(os.path.join(search_root, "%s_retry%d" % (base_name, retry))):
            # 已完成的重试目录也算完成。
            cand = os.path.join(search_root, "%s_retry%d" % (base_name, retry))
            if is_complete(cand):
                print(f"[组 {group}] 重试目录 {cand} 已完成，跳过。")
                append_summary(summary_path, summarize_group(row, cand, s_base, "skipped(done)"))
                return "skipped(done)"
            retry += 1
        output_dir = os.path.join(search_root, "%s_retry%d" % (base_name, retry))
        print(f"[组 {group}] 旧目录 {base_dir} 未完成，不删除，重跑到 {output_dir}。")

    # 磁盘检查（步骤 12）。
    required = estimate_required_bytes(search_root, args.disk_margin_gb)
    ok, free = disk_ok(args.output_root, required)
    print(f"[组 {group}] 磁盘检查：需 {required/GB:.1f}G，剩 {free/GB:.1f}G。")
    if not ok and not args.dry_run:
        print("磁盘不足，需用户确认", file=sys.stderr)
        sys.exit(3)

    write_search_hparams(output_dir, row)
    master_port = args.master_port_base + group
    cmd = build_cmd(args.python_bin, master_port, output_dir, row, args)

    print(f"[组 {group}] 启动：{' '.join(cmd)}")
    if args.dry_run:
        return "dry_run"

    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    train_log = os.path.join(output_dir, "train.log")
    with open(train_log, "a", encoding="utf-8") as lf:
        ret = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if ret != 0:
        print(f"[组 {group}] 训练进程非零退出码 {ret}。", file=sys.stderr)
        status = "failed(rc=%d)" % ret
    elif is_complete(output_dir):
        status = "done" if retry == 0 else "done(retry%d)" % retry
    else:
        status = "incomplete"
    append_summary(summary_path, summarize_group(row, output_dir, s_base, status))
    return status


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
    for row in plan:
        group = int(row["group"])
        if group < args.start_group:
            continue
        status = run_group(row, args, s_base, summary_path, search_root)
        print(f"[组 {group}] 状态：{status}")


if __name__ == "__main__":
    main()
