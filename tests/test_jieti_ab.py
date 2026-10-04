"""tools/export_jieti_ab.py 与 tools/analyze_jieti_ab.py 的 numpy 部分单测（可直接 python 运行）。"""

import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.analyze_jieti_ab import (  # noqa: E402
    analyze, binom_cdf, binom_test, cp_interval, human_pref,
)
from tools.export_jieti_ab import (  # noqa: E402
    IMAGENET_MEAN, IMAGENET_STD, assign_blind, build_html, check_mean_j,
    ink_overlay, select_pairs, to_rgb_u8,
)


def _rows(n=105, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        b = float(rng.uniform(0.1, 0.5))
        c = b + float(rng.normal(0, 0.05))
        r = {"idx": 2 * i + 1, "base_J": b, "ctrl_J": c, "dJ": c - b}
        for k in ("centroid", "logsigma", "shape"):
            r[f"base_{k}"] = float(rng.uniform(0, 0.2))
            r[f"ctrl_{k}"] = float(rng.uniform(0, 0.2))
        rows.append(r)
    return rows


def test_select_pairs():
    rows = _rows()
    sel = select_pairs(rows, 30, 20, seed=0)
    idx = [i for i, _ in sel]
    assert len(sel) == 50 and len(set(idx)) == 50
    top = [i for i, s in sel if s == "top"]
    by = {r["idx"]: abs(r["dJ"]) for r in rows}
    thr = min(by[i] for i in top)
    rest = [i for i in by if i not in top]
    assert all(by[i] <= thr for i in rest)  # top 确实是 |ΔJ| 最大的 30 条
    assert select_pairs(rows, 30, 20, seed=0) == sel  # 可复现
    assert select_pairs(rows, 30, 20, seed=1) != sel
    try:
        select_pairs(rows[:10], 30, 20)
        raise AssertionError("样本不足应报错")
    except ValueError:
        pass


def test_assign_blind_balanced_and_complete():
    sel = select_pairs(_rows(), 30, 20, seed=0)
    items = assign_blind(sel, seed=0)
    assert [it["item"] for it in items] == list(range(1, 51))
    assert sorted(it["idx"] for it in items) == sorted(i for i, _ in sel)
    for it in items:
        assert {it["left"], it["right"]} == {"base", "ctrl"}
    n_left_ctrl = sum(it["left"] == "ctrl" for it in items)
    assert 10 <= n_left_ctrl <= 40  # 左右大致随机
    # 题目顺序被打乱：top 层不应全部排在前 30 题
    assert any(it["stratum"] == "rand" for it in items[:30])


def test_to_rgb_and_overlay():
    white = ((1.0 - IMAGENET_MEAN) / IMAGENET_STD)[:, None, None] * np.ones((3, 4, 4))
    black = ((0.0 - IMAGENET_MEAN) / IMAGENET_STD)[:, None, None] * np.ones((3, 4, 4))
    assert (to_rgb_u8(white) == 255).all() and (to_rgb_u8(black) == 0).all()
    w = np.full((2, 2, 3), 255, np.uint8)
    k = np.zeros((2, 2, 3), np.uint8)
    assert (ink_overlay(w, w) == 255).all()           # 都无墨 → 白
    assert (ink_overlay(k, k) == 0).all()             # 都有墨 → 黑
    assert tuple(ink_overlay(k, w)[0, 0]) == (255, 0, 0)  # 只有 GT → 红
    assert tuple(ink_overlay(w, k)[0, 0]) == (0, 0, 255)  # 只有预测 → 蓝


def test_html_no_leak():
    rows = _rows()
    sel = select_pairs(rows, 3, 2, seed=0)
    items = assign_blind(sel, seed=0)
    rng = np.random.default_rng(0)
    imgs = {i: {t: rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
                for t in ("gt", "base", "ctrl")} for i, _ in sel}
    html = build_html(items, imgs, "s0_n5")
    js = html.split("const ITEMS = ", 1)[1].split(";\n", 1)[0]
    # 题目数据里只能有 item 和图片键，不能出现模型名、idx、分层名、ΔJ
    keys = set(re.findall(r'"([a-z_]+)":', js))
    assert keys == {"item", "gt", "a", "b", "a_o", "b_o"}, keys
    for word in ("base", "ctrl", "stratum", "top", "rand", "dJ", "idx"):
        assert f'"{word}"' not in js
    assert "__" not in html.replace("__proto__", "")  # 占位符都已替换
    # A 图必须是 left 指定模型的那张
    import base64
    import io
    from PIL import Image
    import json
    payload = json.loads(js)
    for it, p in zip(items, payload):
        a = np.array(Image.open(io.BytesIO(base64.b64decode(p["a"]))))
        assert (a == imgs[it["idx"]][it["left"]]).all()


def test_check_mean_j():
    rows = [{"base_J": 0.3, "ctrl_J": 0.28}, {"base_J": 0.31, "ctrl_J": 0.29}]
    got, rep = check_mean_j(rows, 0.305, 0.285, 1e-6)
    assert abs(got["base"] - 0.305) < 1e-12
    try:
        check_mean_j(rows, 0.40, None, 0.01)
        raise AssertionError("不一致应报错")
    except RuntimeError:
        pass


def test_binom_exact():
    # 已知值：n=30, k=22，单侧 P(X>=22) = 0.008062...
    assert abs(binom_test(22, 30, 0.5, "greater") - 0.0080624) < 1e-6
    assert abs(binom_test(15, 30, 0.5, "two-sided") - 1.0) < 1e-12
    assert abs(binom_test(0, 10, 0.5, "two-sided") - 2 / 1024) < 1e-12
    assert abs(sum(math.comb(10, i) for i in range(11)) / 1024 - binom_cdf(10, 10)) < 1e-12


def test_cp_interval():
    # Clopper-Pearson 参考值：k=22, n=30 → [0.5411, 0.8772]；k=0, n=10 → [0, 0.3085]
    lo, hi = cp_interval(22, 30)
    assert abs(lo - 0.5411) < 1e-3 and abs(hi - 0.8772) < 1e-3
    lo, hi = cp_interval(0, 10)
    assert lo == 0.0 and abs(hi - 0.3085) < 1e-3
    lo, hi = cp_interval(10, 10)
    assert hi == 1.0 and abs(lo - 0.6915) < 1e-3


def test_analyze_end_to_end():
    rows = _rows()
    sel = select_pairs(rows, 30, 20, seed=0)
    items = assign_blind(sel, seed=0)
    key = {"session": "s0_n50", "items": items}
    by = {r["idx"]: r for r in rows}
    # 构造一个"永远同意 J"的评委：J 偏好 ctrl 时选 ctrl 所在一侧
    ans = {}
    for it in items:
        want = "ctrl" if by[it["idx"]]["dJ"] < 0 else "base"
        ans[str(it["item"])] = "L" if it["left"] == want else "R"
    ans[str(items[0]["item"])] = "T"  # 一题持平
    per = [{k: str(v) for k, v in r.items()} for r in rows]
    out, recs = analyze(key, {"session": "s0_n50", "answers": ans}, per)
    assert out["all"]["J"]["n_tie"] == 1 and out["all"]["J"]["agree_rate"] == 1.0
    assert out["primary"]["p_value"] < 1e-6
    assert human_pref(items[0], "L") == items[0]["left"]
    # 反着答 → 一致率 0
    rev = {k: {"L": "R", "R": "L", "T": "T"}[v] for k, v in ans.items()}
    out2, _ = analyze(key, {"session": "s0_n50", "answers": rev}, per)
    assert out2["all"]["J"]["agree_rate"] == 0.0
    try:
        analyze(key, {"session": "wrong", "answers": ans}, per)
        raise AssertionError("session 不符应报错")
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")
