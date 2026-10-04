"""T1-V 盲评结果分析：人工结体判断与 J 的方向是否一致。

输入：export_jieti_ab.py 生成的 _key/jieti_ab_key.json、_key/per_sample_J.csv，
以及盲评页导出的 jieti_ab_answers.json。只依赖 numpy 和标准库。

口径（事先定好，评完不改）：
- J 的偏好：ΔJ = J_ctrl − J_base < 0 → J 认为对照组（ctrl）更好，否则认为 baseline 更好。
- 人工偏好：A/B 映射回模型；"持平"单独计数，不进一致率分母。
- 一致率 = 人工偏好与 J 偏好相同的题数 / 非持平题数。
  主检验：top 层（|ΔJ| 最大）的单侧精确二项检验，H0: p = 0.5，H1: p > 0.5。
  rand 层、合并结果、三个分项（centroid / logsigma / shape）的一致率只作描述。
- 另外报告：人工偏好 ctrl 的比例（ctrl 在人眼看来结体是否更好）、持平率、
  选左侧的比例（位置偏差）。
- 置信区间：Clopper-Pearson 精确区间。
"""

import argparse
import csv
import json
import math
from pathlib import Path

J_PARTS = ("centroid", "logsigma", "shape")


def binom_cdf(k, n, p=0.5):
    """P(X <= k)，X ~ Bin(n, p)。"""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def binom_test(k, n, p=0.5, alternative="greater"):
    """精确二项检验 p 值。alternative ∈ {"greater", "less", "two-sided"}。"""
    if n == 0:
        return float("nan")
    if alternative == "greater":
        return 1.0 - binom_cdf(k - 1, n, p)
    if alternative == "less":
        return binom_cdf(k, n, p)
    # 双侧：把概率不大于观测值概率的结果都加起来（与 scipy.stats.binomtest 一致）
    pk = math.comb(n, k) * p ** k * (1 - p) ** (n - k)
    tot = 0.0
    for i in range(n + 1):
        pi = math.comb(n, i) * p ** i * (1 - p) ** (n - i)
        if pi <= pk * (1 + 1e-7):
            tot += pi
    return min(1.0, tot)


def _lower_cp(k, n, alpha):
    """下界：P(X >= k | p) = alpha/2 的解（关于 p 单调增）。"""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 1.0 - binom_cdf(k - 1, n, mid) < alpha / 2:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _upper_cp(k, n, alpha):
    """上界：P(X <= k | p) = alpha/2 的解（关于 p 单调减）。"""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > alpha / 2:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def cp_interval(k, n, alpha=0.05):
    if n == 0:
        return [float("nan"), float("nan")]
    lower = 0.0 if k == 0 else _lower_cp(k, n, alpha)
    upper = 1.0 if k == n else _upper_cp(k, n, alpha)
    return [lower, upper]


def human_pref(item, ans):
    """L/T/R → 'base' / 'ctrl' / 'tie'。"""
    if ans == "T":
        return "tie"
    if ans == "L":
        return item["left"]
    if ans == "R":
        return item["right"]
    raise ValueError(f"item {item['item']}: 未知答案 {ans!r}")


def j_pref(delta):
    return "ctrl" if delta < 0 else "base"


def agree_block(pairs, alternative="greater"):
    """pairs: [(人工偏好, J 偏好)]。统计非持平题的一致率。"""
    n_tie = sum(h == "tie" for h, _ in pairs)
    eff = [(h, j) for h, j in pairs if h != "tie"]
    k = sum(h == j for h, j in eff)
    n = len(eff)
    return {"n_items": len(pairs), "n_tie": n_tie, "n_eff": n, "n_agree": k,
            "agree_rate": k / n if n else float("nan"),
            "ci95": cp_interval(k, n),
            "p_value": binom_test(k, n, 0.5, alternative),
            "alternative": alternative}


def pref_block(prefs):
    """人工偏好 ctrl 的比例（非持平题），双侧检验。"""
    eff = [h for h in prefs if h != "tie"]
    k = sum(h == "ctrl" for h in eff)
    n = len(eff)
    return {"n_eff": n, "n_ctrl": k, "rate_ctrl": k / n if n else float("nan"),
            "ci95": cp_interval(k, n), "p_two_sided": binom_test(k, n, 0.5, "two-sided")}


def analyze(key, answers, per_sample):
    items = key["items"]
    ans = answers["answers"]
    if answers.get("session") != key["session"]:
        raise ValueError(f"session 不一致：答案 {answers.get('session')}，key {key['session']}")
    missing = [it["item"] for it in items if str(it["item"]) not in ans]
    if missing:
        raise ValueError(f"缺 {len(missing)} 题未答：{missing[:10]}")
    by_idx = {int(r["idx"]): r for r in per_sample}

    recs = []
    for it in items:
        a = ans[str(it["item"])]
        r = by_idx[it["idx"]]
        rec = {"item": it["item"], "idx": it["idx"], "stratum": it["stratum"],
               "answer": a, "human": human_pref(it, a), "dJ": float(r["dJ"]),
               "J_pref": j_pref(float(r["dJ"]))}
        for p in J_PARTS:
            rec[f"pref_{p}"] = j_pref(float(r[f"ctrl_{p}"]) - float(r[f"base_{p}"]))
        recs.append(rec)

    out = {"session": key["session"], "n_items": len(recs)}
    strata = {"top": [r for r in recs if r["stratum"] == "top"],
              "rand": [r for r in recs if r["stratum"] == "rand"],
              "all": recs}
    for name, rs in strata.items():
        blk = {"J": agree_block([(r["human"], r["J_pref"]) for r in rs])}
        for p in J_PARTS:
            blk[p] = agree_block([(r["human"], r[f"pref_{p}"]) for r in rs])
        blk["human_prefers_ctrl"] = pref_block([r["human"] for r in rs])
        blk["J_prefers_ctrl"] = sum(r["J_pref"] == "ctrl" for r in rs)
        out[name] = blk
    non_tie = [r for r in recs if r["answer"] != "T"]
    k_left = sum(r["answer"] == "L" for r in non_tie)
    out["side_bias"] = {"n_eff": len(non_tie), "n_left": k_left,
                        "rate_left": k_left / len(non_tie) if non_tie else float("nan"),
                        "p_two_sided": binom_test(k_left, len(non_tie), 0.5, "two-sided")}
    out["primary"] = {"test": "top 层 J 一致率，单侧精确二项检验 H1: p > 0.5",
                      **out["top"]["J"]}
    return out, recs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ab_dir", required=True, help="export_jieti_ab.py 的输出目录")
    parser.add_argument("--answers", required=True, help="盲评页导出的 JSON")
    args = parser.parse_args()

    ab = Path(args.ab_dir)
    key = json.loads((ab / "_key" / "jieti_ab_key.json").read_text(encoding="utf-8"))
    with open(ab / "_key" / "per_sample_J.csv", encoding="utf-8") as f:
        per_sample = list(csv.DictReader(f))
    answers = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    out, recs = analyze(key, answers, per_sample)
    (ab / "analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    with open(ab / "analysis_items.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
