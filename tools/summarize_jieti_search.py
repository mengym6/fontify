#!/usr/bin/env python
"""汇总 T1 超参搜索结果（PROGRESS 步骤 11）。

读取 summary.csv，按 best_S 升序列出前 5 组；对每个超参，给出前 5 / 前 10 组
的取值范围，以及该超参与 best_S 的 Spearman 相关系数（纯 Python 实现），作为
"大致较优范围"的估计。输出 markdown 到 stdout，同时写一个 txt 文件。

只依赖标准库。
"""

import argparse
import csv
import sys

HPARAMS = ["alpha_jt", "w", "p", "lr", "accum_iter", "warmup_epochs"]


def is_valid_status(status):
    """只纳入训练完整跑完的组：done / done(retryN) / reused / skipped(done)。
    排除 failed(...) / incomplete / reused_incomplete / dry_run 等。
    runner 重算过 S 时状态带 ";S_recomputed" 后缀，判断时去掉。"""
    status = status.split(";")[0]
    if status.startswith("done"):
        return True
    return status in ("reused", "skipped(done)")


def read_summary(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        status = r.get("status", "")
        if not is_valid_status(status):
            continue  # 跳过失败/未完成组，避免未训完的 best_S 混入排名
        try:
            s = float(r["best_S"])
        except (ValueError, KeyError, TypeError):
            continue  # 跳过没有有效 best_S 的组
        rec = {"group": r.get("group", ""), "best_S": s,
               "status": status, "best_epoch": r.get("best_epoch", "")}
        for h in HPARAMS:
            try:
                rec[h] = float(r[h])
            except (ValueError, KeyError, TypeError):
                rec[h] = None
        out.append(rec)
    return out


def rankdata(values):
    """返回平均秩（处理并列），输入是数值列表。"""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 秩从 1 开始，并列取平均
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Spearman 相关系数 = 秩的 Pearson 相关。少于 3 对或零方差返回 None。"""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xr = rankdata([p[0] for p in pairs])
    yr = rankdata([p[1] for p in pairs])
    n = len(xr)
    mx = sum(xr) / n
    my = sum(yr) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xr, yr))
    vx = sum((a - mx) ** 2 for a in xr)
    vy = sum((b - my) ** 2 for b in yr)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


def fmt(v):
    if v is None:
        return "NA"
    if isinstance(v, float):
        return "%.4g" % v
    return str(v)


def value_range(records, h):
    vals = [r[h] for r in records if r.get(h) is not None]
    if not vals:
        return "NA"
    return "[%s, %s]" % (fmt(min(vals)), fmt(max(vals)))


def build_report(records):
    """返回 markdown 文本。records 已过滤出有 best_S 的组。"""
    records = sorted(records, key=lambda r: r["best_S"])
    lines = []
    lines.append("# T1 超参搜索汇总\n")
    lines.append("有效组数（含 best_S）：%d\n" % len(records))

    # 前 5 组表。
    lines.append("## 前 5 组（按 S 升序）\n")
    header = ["组", "S", "best_epoch"] + HPARAMS + ["status"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in records[:5]:
        cells = [fmt(r["group"]), fmt(r["best_S"]), fmt(r.get("best_epoch", ""))] + \
                [fmt(r[h]) for h in HPARAMS] + [r.get("status", "")]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # 各超参较优范围 + Spearman。
    lines.append("## 各超参较优范围与 Spearman（与 S，越负越好）\n")
    header2 = ["超参", "前5范围", "前10范围", "全体范围", "Spearman(ρ, S)"]
    lines.append("| " + " | ".join(header2) + " |")
    lines.append("| " + " | ".join(["---"] * len(header2)) + " |")
    top5 = records[:5]
    top10 = records[:10]
    all_s = [r["best_S"] for r in records]
    for h in HPARAMS:
        rho = spearman([r[h] for r in records], all_s)
        lines.append("| %s | %s | %s | %s | %s |" % (
            h, value_range(top5, h), value_range(top10, h),
            value_range(records, h), fmt(rho)))
    lines.append("")
    lines.append("注：Spearman 为负表示该超参越大 S 越低（越好）；"
                 "组数少时相关系数不稳，只作粗略方向。\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", default="models/jieti_search/summary.csv")
    parser.add_argument("--output", default="models/jieti_search/search_report.txt")
    args = parser.parse_args()

    records = read_summary(args.summary)
    if not records:
        print("没有可汇总的组（summary.csv 为空或无有效 best_S）。", file=sys.stderr)
        sys.exit(1)
    report = build_report(records)
    print(report)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\n已写入 %s" % args.output, file=sys.stderr)


if __name__ == "__main__":
    main()
