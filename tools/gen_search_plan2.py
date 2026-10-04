#!/usr/bin/env python
"""生成 T1-S 第二轮超参搜索计划（PROGRESS T1-S）。

以 T1-F 对照组（finetune_jieti_ctrl_g12.sh）为中心，写出
  models/jieti_search2/search_plan.csv       每组超参
  models/jieti_search2/search_plan_spec.json 区间定义、seed、中心组
第 1 组 = 中心组，复用 models/finetune_jieti_ctrl_g12，不重跑；第 2–N 组随机抽样。
旧搜索的 tools/gen_search_plan.py 不动。

抽样分布（用户 2026-10-04 选"中心 ×/÷2"）：
  lr          ~ 对数均匀 [lr_c/2, lr_c·2]
  w           ~ 对数均匀 [w_c/2,  w_c·2]；另记 u = log10(w / 0.6423)，便于和上一轮对比
  accum_iter  ∈ {8, 16, 32}（2 卡、batch 2 → 有效 batch {32, 64, 128}）
  warmup      ∈ {5, 10, 15}
  alpha_jt    ~ U[0.5, 1.0]
  p           固定 0，不搜
每组抽样顺序固定为 alpha_jt, w, lr, accum_iter, warmup_epochs。
只依赖标准库；输出不含时间戳，同一 seed 在本地和服务器上逐字一致。
输出文件已存在时报错退出，不覆盖。
"""

import argparse
import csv
import json
import math
import os
import random
import sys

FIELDS = ["group", "alpha_jt", "w", "u", "p", "lr", "accum_iter", "warmup_epochs", "source"]

ACCUM_CHOICES = [8, 16, 32]
WARMUP_CHOICES = [5, 10, 15]
ALPHA_RANGE = (0.5, 1.0)
W_REF = 0.6423  # 上一轮搜索的 w0，u 以它为参照
CENTER_SOURCE = "reuse:models/finetune_jieti_ctrl_g12"


def loguniform(rng, lo, hi):
    """对数均匀抽样：在 [log lo, log hi] 上均匀，再取指数。"""
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def u_of(w):
    return round(math.log10(w / W_REF), 6)


def build_spec(args):
    return {
        "seed": args.seed,
        "n_groups": args.n_groups,
        "center": {"group": 1, "alpha_jt": args.alpha_c, "w": args.w_c, "lr": args.lr_c,
                   "accum_iter": args.accum_c, "warmup_epochs": args.warmup_c, "p": 0,
                   "source": CENTER_SOURCE},
        "intervals": {
            "lr": {"dist": "loguniform", "lo": args.lr_c / 2, "hi": args.lr_c * 2,
                   "def": "lr_c/2 .. lr_c*2", "csv_format": "%.6g"},
            "w": {"dist": "loguniform", "lo": args.w_c / 2, "hi": args.w_c * 2,
                  "def": "w_c/2 .. w_c*2", "csv_round": 8},
            "alpha_jt": {"dist": "uniform", "lo": ALPHA_RANGE[0], "hi": ALPHA_RANGE[1],
                         "csv_round": 6},
            "accum_iter": {"dist": "choice", "values": ACCUM_CHOICES,
                           "note": "2 卡 x batch 2 → 有效 batch 32/64/128"},
            "warmup_epochs": {"dist": "choice", "values": WARMUP_CHOICES},
            "p": {"dist": "fixed", "value": 0},
        },
        "u_def": "u = log10(w / %s)，由 csv 里四舍五入后的 w 计算" % W_REF,
        "sample_order": ["alpha_jt", "w", "lr", "accum_iter", "warmup_epochs"],
        "generator": "tools/gen_search_plan2.py, random.Random(seed)",
    }


def build_plan(args):
    rng = random.Random(args.seed)
    w_c = round(args.w_c, 8)
    rows = [{
        "group": 1,
        "alpha_jt": round(args.alpha_c, 6),
        "w": w_c,
        "u": u_of(w_c),
        "p": 0,
        "lr": float("%.6g" % args.lr_c),
        "accum_iter": args.accum_c,
        "warmup_epochs": args.warmup_c,
        "source": CENTER_SOURCE,
    }]
    lr_lo, lr_hi = args.lr_c / 2, args.lr_c * 2
    w_lo, w_hi = args.w_c / 2, args.w_c * 2
    for g in range(2, args.n_groups + 1):
        alpha_jt = rng.uniform(*ALPHA_RANGE)
        w = round(loguniform(rng, w_lo, w_hi), 8)
        lr = loguniform(rng, lr_lo, lr_hi)
        accum = rng.choice(ACCUM_CHOICES)
        warmup = rng.choice(WARMUP_CHOICES)
        rows.append({
            "group": g,
            "alpha_jt": round(alpha_jt, 6),
            "w": w,
            "u": u_of(w),
            "p": 0,
            "lr": float("%.6g" % lr),
            "accum_iter": accum,
            "warmup_epochs": warmup,
            "source": "sample",
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=1, help="抽样 seed（上一轮用 0）")
    parser.add_argument("--n_groups", type=int, default=25, help="总组数（含中心组）")
    parser.add_argument("--output", default="models/jieti_search2/search_plan.csv")
    parser.add_argument("--spec_output", default=None,
                        help="区间定义 json，默认与 csv 同目录的 search_plan_spec.json")
    # 中心组 = T1-F 对照组（finetune_jieti_ctrl_g12.sh；4 卡 accum 8 = 2 卡 accum 16）
    parser.add_argument("--alpha_c", type=float, default=1.0)
    parser.add_argument("--w_c", type=float, default=0.73719102)
    parser.add_argument("--lr_c", type=float, default=0.00195699)
    parser.add_argument("--accum_c", type=int, default=16)
    parser.add_argument("--warmup_c", type=int, default=10)
    args = parser.parse_args()

    spec_path = args.spec_output or os.path.join(os.path.dirname(args.output) or ".",
                                                 "search_plan_spec.json")
    for p in (args.output, spec_path):
        if os.path.exists(p):
            print(f"错误：文件已存在，拒绝覆盖：{p}", file=sys.stderr)
            sys.exit(2)

    rows = build_plan(args)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(build_spec(args), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    print(f"写入 {len(rows)} 组到 {args.output}（seed={args.seed}），区间定义 {spec_path}")


if __name__ == "__main__":
    main()
