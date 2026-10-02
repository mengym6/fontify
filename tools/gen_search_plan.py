#!/usr/bin/env python
"""生成 T1 超参随机搜索计划（PROGRESS 步骤 9 / C2）。

用固定 seed 抽样，写出 models/jieti_search/search_plan.csv。第 1 组是步骤 8 的
正式训练（中心配置），第 2-N 组按 C2 的分布随机抽样。只依赖标准库，可本地跑。

抽样分布（C2）：
  alpha_jt     ~ U[0.2, 0.9]
  w = w0·10^u, u ~ U[-1, 0.5]
  p            ~ U[0.5, 1]
  lr           ~ 对数均匀 [3e-4, 3e-3]
  accum_iter   ∈ {8, 16, 32, 64}
  warmup_epochs∈ {5, 8, 10}

w0 由命令行传入（标定后才知道）。csv 同时记录 u 和最终 w。
若输出文件已存在则报错退出，不覆盖。
"""

import argparse
import csv
import math
import os
import random
import sys

# 每组固定的 6 个超参列（加上 group 和 u）。runner 和 summary 都按这些列名读。
FIELDS = ["group", "alpha_jt", "w", "u", "p", "lr", "accum_iter", "warmup_epochs"]

ACCUM_CHOICES = [8, 16, 32, 64]
WARMUP_CHOICES = [5, 8, 10]


def loguniform(rng, lo, hi):
    """对数均匀抽样：在 [log lo, log hi] 上均匀，再取指数。"""
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def build_plan(w0, n_groups, seed, lr0=1e-3, accum0=32, warmup0=5, alpha0=0.5):
    """返回计划行列表（dict）。第 1 组为中心配置，u=0 → w=w0。"""
    rng = random.Random(seed)
    rows = []
    # 第 1 组：步骤 8 的中心配置。
    rows.append({
        "group": 1,
        "alpha_jt": round(alpha0, 6),
        "w": round(w0, 8),
        "u": 0.0,
        "p": 1.0,
        "lr": lr0,
        "accum_iter": accum0,
        "warmup_epochs": warmup0,
    })
    # 抽样顺序固定（alpha_jt, u, p, lr, accum_iter, warmup_epochs），保证可复现。
    for g in range(2, n_groups + 1):
        alpha_jt = rng.uniform(0.2, 0.9)
        u = rng.uniform(-1.0, 0.5)
        w = w0 * (10.0 ** u)
        p = rng.uniform(0.5, 1.0)
        lr = loguniform(rng, 3e-4, 3e-3)
        accum = rng.choice(ACCUM_CHOICES)
        warmup = rng.choice(WARMUP_CHOICES)
        rows.append({
            "group": g,
            "alpha_jt": round(alpha_jt, 6),
            "w": round(w, 8),
            "u": round(u, 6),
            "p": round(p, 6),
            "lr": float("%.6g" % lr),
            "accum_iter": accum,
            "warmup_epochs": warmup,
        })
    return rows


def write_plan(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--w0", type=float, required=True,
                        help="步骤 4 标定出的结体项权重绝对值")
    parser.add_argument("--n_groups", type=int, default=50, help="总组数（含中心组）")
    parser.add_argument("--seed", type=int, default=0, help="抽样 seed，固定可复现")
    parser.add_argument("--output", default="models/jieti_search/search_plan.csv")
    parser.add_argument("--lr0", type=float, default=1e-3, help="中心组 lr")
    parser.add_argument("--accum0", type=int, default=32, help="中心组 accum_iter")
    parser.add_argument("--warmup0", type=int, default=5, help="中心组 warmup_epochs")
    parser.add_argument("--alpha0", type=float, default=0.5, help="中心组 alpha_jt")
    args = parser.parse_args()

    if os.path.exists(args.output):
        print(f"错误：计划文件已存在，拒绝覆盖：{args.output}", file=sys.stderr)
        sys.exit(2)

    rows = build_plan(args.w0, args.n_groups, args.seed,
                      lr0=args.lr0, accum0=args.accum0,
                      warmup0=args.warmup0, alpha0=args.alpha0)
    write_plan(rows, args.output)
    print(f"写入 {len(rows)} 组到 {args.output}（seed={args.seed}, w0={args.w0}）")


if __name__ == "__main__":
    main()
