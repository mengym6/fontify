"""tools/eval_ink_moments.py 数值部分的单测（只依赖 numpy，可直接 python 运行）。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.eval_ink_moments import (  # noqa: E402
    align_centroid, bootstrap_fractions, counterfactual, ink_moments,
    low_energy, moment_qtys, scale_mass, shift_bilinear, soft_ink, spearman,
)


def _blob(h=128, w=128, cx=64.0, cy=64.0, sx=6.0, sy=9.0, amp=0.9):
    """各向异性高斯墨团（居中，尾部截断对称，平移不出界）。"""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    return amp * np.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))


def test_soft_ink_matches_linear_fg():
    g = np.array([[-0.1, 0.0, 0.3, 1.0, 1.2]])
    assert np.allclose(soft_ink(g), [[1.0, 1.0, 0.7, 0.0, 0.0]])


def test_moments_of_blob():
    m = ink_moments(_blob())
    assert abs(m["cx"] - 64.0) < 1e-6 and abs(m["cy"] - 64.0) < 1e-6
    assert abs(m["sx"] - 6.0) < 1e-3 and abs(m["sy"] - 9.0) < 1e-3


def test_known_shift_recovered_with_sign():
    a = _blob()
    for dx, dy in ((3.0, -5.0), (2.3, 1.7), (-4.6, 0.25)):
        b = shift_bilinear(a, dx, dy)
        q = moment_qtys(ink_moments(b), ink_moments(a))
        # +x 向右、+y 向下；质心位移逐分量带符号还原
        assert abs(q["dx"] - dx) < 1e-6 and abs(q["dy"] - dy) < 1e-6
        assert abs(q["disp"] - np.hypot(dx, dy)) < 1e-6
        assert abs(b.sum() - a.sum()) < 1e-9 * a.sum()  # 未出界时墨量守恒


def test_integer_shift_is_exact():
    a = _blob()
    b = shift_bilinear(a, 4.0, -3.0)
    # out[y, x] = a[y + 3, x − 4]
    assert np.allclose(b[10:80, 14:90], a[13:83, 10:86])
    assert np.all(b[:, :4] == 0) and np.all(b[-3:, :] == 0)  # 界外补 0


def test_mass_scaling_and_sign():
    a = _blob()
    mg = ink_moments(a)
    more = 1.3 * a
    q = moment_qtys(ink_moments(more), mg)
    assert abs(q["rel_mass"] - 0.3) < 1e-9          # 墨多 → 正
    assert abs(q["logsx"]) < 1e-9 and abs(q["logsy"]) < 1e-9 and q["disp"] < 1e-9
    q2 = moment_qtys(ink_moments(0.8 * a), mg)
    assert abs(q2["rel_mass"] + 0.2) < 1e-9         # 墨少 → 负
    s = scale_mass(more, mg["mass"])
    assert abs(s.sum() - mg["mass"]) < 1e-9 * mg["mass"]
    assert np.allclose(s, a)


def test_scale_sign():
    a = _blob()
    q = moment_qtys(ink_moments(_blob(sx=7.2, sy=9.0 * 0.8)), ink_moments(a))
    assert abs(q["logsx"] - np.log(1.2)) < 1e-3     # 更宽 → 正
    assert abs(q["logsy"] - np.log(0.8)) < 1e-3     # 更矮 → 负


def test_align_then_low_energy_near_zero():
    gt = _blob()
    pred = shift_bilinear(gt, 3.0, -2.0)            # 纯整数平移
    mg = ink_moments(gt)
    al, adx, ady, resid, kept = align_centroid(pred, mg["cx"], mg["cy"])
    assert abs(adx + 3.0) < 1e-6 and abs(ady - 2.0) < 1e-6 and resid < 1e-6
    assert abs(kept - 1.0) < 1e-9
    assert low_energy(al - gt, 4.0) < 1e-20
    assert low_energy(pred - gt, 4.0) > 1e-4
    # 亚像素：双线性带来 f(1−f) 级别的模糊，能量应降两个数量级以上
    pred2 = shift_bilinear(gt, 2.5, 1.5)
    al2 = align_centroid(pred2, mg["cx"], mg["cy"])[0]
    assert low_energy(al2 - gt, 4.0) < 1e-2 * low_energy(pred2 - gt, 4.0)


def test_counterfactual_shift_plus_mass():
    gt = _blob()
    pred = 1.25 * shift_bilinear(gt, -4.0, 3.0)
    cf = counterfactual(pred, gt, ink_moments(gt), [4.0])
    assert cf["s4_elow_raw"] > cf["s4_elow_pos"] > cf["s4_elow_posmass"]
    assert cf["s4_elow_posmass"] < 1e-20
    assert abs(cf["scale_factor"] - 0.8) < 1e-9
    assert cf["frac_gt1"] == 0.0


def test_align_with_mass_leaving_frame():
    """GT 贴边、pred 居中：对齐要把部分墨推出界，单次平移欠修正，迭代后质心残差仍应很小。"""
    gt = _blob(cx=4.0, cy=64.0, sx=6.0)            # 左侧截断，质心在 x≈6.6
    pred = _blob(cx=64.0, cy=64.0, sx=6.0)
    mg = ink_moments(gt)
    one = shift_bilinear(pred, mg["cx"] - ink_moments(pred)["cx"], 0.0)
    assert abs(ink_moments(one)["cx"] - mg["cx"]) > 0.1  # 单次平移确实不精确
    al, _, _, resid, kept = align_centroid(pred, mg["cx"], mg["cy"])
    assert resid < 1e-3
    assert kept < 0.95


def test_fractions_sum_to_one():
    rng = np.random.default_rng(1)
    raw = rng.normal(1.0, 0.3, 50)
    pos = raw * 0.6
    pm = raw * 0.2
    ms = raw * 0.7
    fr = bootstrap_fractions(raw, pos, pm, ms, n_boot=200)
    tot = fr["position"]["frac"] + fr["mass"]["frac"] + fr["shape"]["frac"]
    assert abs(tot - 1.0) < 1e-12
    assert abs(fr["position"]["frac"] - 0.4) < 1e-12
    assert abs(fr["mass"]["frac"] - 0.4) < 1e-12
    assert abs(fr["shape"]["frac"] - 0.2) < 1e-12
    assert abs(fr["mass_first"]["frac"] - 0.3) < 1e-12


def test_spearman():
    x = np.arange(20.0)
    assert abs(spearman(x, x ** 3) - 1.0) < 1e-12
    assert abs(spearman(x, -x) + 1.0) < 1e-12
    y = np.array([1, 2, 2, 3], dtype=float)
    # 并列取平均秩：与 Pearson(秩) 一致
    assert abs(spearman(np.array([1.0, 2, 3, 4]), y) - 0.9486832980505138) < 1e-12


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")
