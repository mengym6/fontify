"""tools/eval_band_error.py 数值部分的单测（只依赖 numpy，可直接 python 运行）。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.eval_band_error import (  # noqa: E402
    IMAGENET_MEAN, IMAGENET_STD, band_metrics, denorm_to_gray,
    gaussian_blur, gaussian_kernel_1d, paired_bootstrap_ci, summarize,
)


def _reference_blur(img, sigma):
    """逐像素二维卷积参考实现（reflect 边界），用于核对可分离实现。"""
    k1 = gaussian_kernel_1d(sigma)
    k2 = np.outer(k1, k1)
    r = (len(k1) - 1) // 2
    pad = np.pad(img, r, mode="reflect")
    h, w = img.shape
    out = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            out[i, j] = (pad[i:i + 2 * r + 1, j:j + 2 * r + 1] * k2).sum()
    return out


def test_kernel_normalized():
    for s in (1.0, 2.0, 4.0, 8.0):
        assert abs(gaussian_kernel_1d(s).sum() - 1.0) < 1e-12


def test_blur_matches_reference():
    rng = np.random.default_rng(0)
    img = rng.normal(size=(40, 30))
    for s in (1.0, 2.5):
        assert np.allclose(gaussian_blur(img, s), _reference_blur(img, s), atol=1e-12)


def test_blur_constant_and_checkerboard():
    const = np.full((64, 64), 0.37)
    assert np.allclose(gaussian_blur(const, 4.0), 0.37)
    yy, xx = np.mgrid[:64, :64]
    checker = ((yy + xx) % 2) * 2.0 - 1.0  # Nyquist 频率
    assert np.abs(gaussian_blur(checker, 2.0)).max() < 1e-3


def test_energy_closure():
    rng = np.random.default_rng(1)
    err = rng.normal(size=(224, 448)) * 0.1
    m = band_metrics(err, 4.0)
    resid = m["e2_total"] - m["e2_low"] - m["e2_high"] - 2 * m["e2_cross"]
    assert abs(resid) < 1e-12


def test_denorm_white_black():
    white = ((1.0 - IMAGENET_MEAN) / IMAGENET_STD)[:, None, None] * np.ones((3, 4, 4))
    black = ((0.0 - IMAGENET_MEAN) / IMAGENET_STD)[:, None, None] * np.ones((3, 4, 4))
    assert np.allclose(denorm_to_gray(white), 1.0)
    assert np.allclose(denorm_to_gray(black), 0.0)
    over = white * 2  # 超出 [0,1] 时 clamp 生效
    assert np.allclose(denorm_to_gray(over, clamp=True), 1.0)


def test_band_selectivity():
    """位移整块笔画 → 低频占主；边缘锯齿噪声 → 高频占主。"""
    h, w = 224, 448
    gt = np.ones((h, w))
    gt[60:160, 100:140] = 0.0  # 竖笔
    shifted = np.ones((h, w))
    shifted[60:160, 120:160] = 0.0  # 平移 20px
    rng = np.random.default_rng(2)
    jag = gt.copy()
    edge = (np.abs(np.arange(w) - 100) <= 1) | (np.abs(np.arange(w) - 139) <= 1)
    jag[60:160][:, edge] = rng.integers(0, 2, size=(100, edge.sum()))
    m_shift = band_metrics(shifted - gt, 4.0)
    m_jag = band_metrics(jag - gt, 4.0)
    assert m_shift["e2_low"] > m_shift["e2_high"]
    assert m_jag["e2_high"] > m_jag["e2_low"]


def test_bootstrap_ci():
    rng = np.random.default_rng(3)
    d = rng.normal(0.5, 1.0, size=200)
    mean, lo, hi = paired_bootstrap_ci(d, n_boot=2000, seed=0)
    assert lo < mean < hi
    assert lo > 0.2 and hi < 0.8
    z = paired_bootstrap_ci(np.zeros(10))
    assert z == (0.0, 0.0, 0.0)


def test_summarize_roundtrip():
    rng = np.random.default_rng(4)
    rows = []
    for i in range(30):
        kind = "BF" if i % 2 else "JT"
        row = {"idx": i, "kind": kind}
        for tag, scale in (("base", 0.1), ("t1", 0.12)):
            err = rng.normal(size=(32, 32)) * scale
            row[f"{tag}_l1_norm3"] = float(np.abs(err).mean())
            for k, v in band_metrics(err, 2.0).items():
                row[f"{tag}_s2_{k}"] = v
        rows.append(row)
    s = summarize(rows, [2.0], n_boot=200)
    assert s["BF"]["n"] == 15 and s["JT"]["n"] == 15
    assert abs(s["BF"]["sigma_2"]["closure_residual"]) < 1e-12
    assert s["BF"]["sigma_2"]["e2_total"]["delta"] > 0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")
