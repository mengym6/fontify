"""BF/JT 软墨量矩诊断与反事实对齐：baseline 与 T1 在固定 val 上的逐样本配对对比。

T1-D 第 3、4 项。eval_band_error 已确认 T1 的 BF 变差以低频为主；本脚本把低频误差
进一步拆成"位置 / 墨量 / 形状"三部分，并给出最差样本的并排图。只做评测，不训练。

软墨量（下半 query 区域，448×448）：
    g = denorm_to_gray(pred, clamp=True)   与 eval_band_error 同口径的 [0,1] 灰度
    a = clamp(1 − g, 0, 1)                 与 jieti_loss 的 linear 软前景同式
A. 逐样本矩：墨量 M = Σa，质心 c = Σa·(x,y)/M，尺度 σx/σy（绕质心二阶矩开方）。
B. 反事实：预测 a 先亚像素平移使质心对齐 GT（双线性，界外补 a=0 即白底），再乘
   M_gt/M_pred 使墨量相等（不 clamp）；每步在 a 空间重算 e = a_pred − a_gt 的低频能量
       E_low = mean((G_σ * e)²)
   未变换时 e_a = −e_gray，E_low 与 eval_band_error 的 s{σ}_e2_low 逐样本相等，可对账。
C. BF 最差 10 个 / 中位 5 个样本的并排图。

数值部分只依赖 numpy（单测可在无 torch 环境跑）；torch、模型、matplotlib 延迟导入。
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eval_band_error import (  # noqa: E402
    IMAGENET_MEAN, IMAGENET_STD, denorm_to_gray, gaussian_blur,
    paired_bootstrap_ci,
)

# 反事实步骤：raw 原始；pos 质心对齐；posmass 质心对齐 + 墨量归一；mass 只做墨量归一（看顺序敏感性）
STEPS = ("raw", "pos", "posmass", "mass")
# 带符号量（逐模型给均值）；A 部分配对差用的是 DELTA_QTYS（有绝对值的先取绝对值）
SIGNED_QTYS = ("rel_mass", "dx", "dy", "disp", "logsx", "logsy")
DELTA_QTYS = ("abs_rel_mass", "rel_mass", "disp", "abs_logsx", "abs_logsy")


# ---------------------------------------------------------------------------
# numpy 数值部分
# ---------------------------------------------------------------------------

def soft_ink(gray):
    """[0,1] 灰度 → 软墨量 a = clamp(1 − g, 0, 1)。"""
    return np.clip(1.0 - gray, 0.0, 1.0)


def ink_moments(a):
    """软墨量图 a (H, W) 的零/一/二阶矩。x 为列坐标、y 为行坐标（向下为正），单位像素。"""
    m = float(a.sum())
    if not m > 0:
        raise ValueError("墨量为 0，质心无定义")
    h, w = a.shape
    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    px = a.sum(axis=0)
    py = a.sum(axis=1)
    cx = float((px * xs).sum() / m)
    cy = float((py * ys).sum() / m)
    sx = float(np.sqrt((px * (xs - cx) ** 2).sum() / m))
    sy = float(np.sqrt((py * (ys - cy) ** 2).sum() / m))
    return {"mass": m, "cx": cx, "cy": cy, "sx": sx, "sy": sy}


def _int_shift(a, n, axis):
    """整数平移 out[i] = a[i − n]，界外补 0。"""
    b = np.moveaxis(a, axis, 0)
    out = np.zeros_like(b)
    length = b.shape[0]
    if abs(n) < length:
        if n >= 0:
            out[n:] = b[:length - n]
        else:
            out[:length + n] = b[-n:]
    return np.moveaxis(out, 0, axis)


def shift_bilinear(a, dx, dy):
    """亚像素平移（可分离线性插值，界外补 0）：内容向 +x 移 dx、向 +y 移 dy。

    d = n + f（n = floor d）时 out[i] = (1−f)·a[i−n] + f·a[i−n−1]，
    质量不出界时质心恰好移动 d，方差增加 f(1−f)（≤0.25 px²）。
    """
    out = a
    for d, axis in ((dx, 1), (dy, 0)):
        n = int(np.floor(d))
        f = d - n
        out = (1.0 - f) * _int_shift(out, n, axis) + f * _int_shift(out, n + 1, axis)
    return out


def align_centroid(a, target_cx, target_cy, iters=16, tol=1e-3):
    """平移 a 使质心对齐目标。墨量出界会让一次平移不精确，故对原图按累计位移迭代修正。

    返回 (对齐后图, 累计 dx, 累计 dy, 残余质心误差 px, 保留墨量比例)。
    """
    m0 = ink_moments(a)
    tdx = tdy = 0.0
    cur = a
    mom = m0
    for _ in range(iters):
        ex, ey = target_cx - mom["cx"], target_cy - mom["cy"]
        if np.hypot(ex, ey) < tol:
            break
        tdx += ex
        tdy += ey
        cur = shift_bilinear(a, tdx, tdy)
        mom = ink_moments(cur)
    resid = float(np.hypot(target_cx - mom["cx"], target_cy - mom["cy"]))
    return cur, tdx, tdy, resid, mom["mass"] / m0["mass"]


def scale_mass(a, target_mass):
    """墨量缩放到 target_mass，不 clamp（保证墨量严格相等）。"""
    return a * (target_mass / float(a.sum()))


def low_energy(err, sigma):
    """低频误差能量 mean((G_σ * err)²)，与 eval_band_error.band_metrics 的 e2_low 同式。"""
    return float((gaussian_blur(err, sigma) ** 2).mean())


def moment_qtys(mp, mg):
    """预测矩 mp 相对 GT 矩 mg 的带符号量。"""
    dx = mp["cx"] - mg["cx"]
    dy = mp["cy"] - mg["cy"]
    return {
        "rel_mass": (mp["mass"] - mg["mass"]) / mg["mass"],
        "dx": dx, "dy": dy, "disp": float(np.hypot(dx, dy)),
        "logsx": float(np.log(mp["sx"] / mg["sx"])),
        "logsy": float(np.log(mp["sy"] / mg["sy"])),
    }


def counterfactual(a_pred, a_gt, mg, sigmas):
    """对一个模型的预测做反事实对齐，返回各步能量与对齐诊断量。"""
    a_pos, adx, ady, resid, kept = align_centroid(a_pred, mg["cx"], mg["cy"])
    scale = mg["mass"] / float(a_pos.sum())
    a_posmass = a_pos * scale
    a_mass = scale_mass(a_pred, mg["mass"])
    states = {"raw": a_pred, "pos": a_pos, "posmass": a_posmass, "mass": a_mass}
    out = {"align_dx": adx, "align_dy": ady, "align_resid": resid,
           "align_mass_kept": kept, "scale_factor": scale,
           "frac_gt1": float((a_posmass > 1.0).mean())}
    for step, st in states.items():
        err = st - a_gt
        out[f"e2_total_{step}"] = float((err ** 2).mean())
        for sigma in sigmas:
            out[f"s{sigma:g}_elow_{step}"] = low_energy(err, sigma)
    return out


def rank_avg(x):
    """平均秩（并列取平均）。"""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return sums[inv] / counts[inv]


def spearman(x, y):
    rx, ry = rank_avg(x), rank_avg(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def spearman_bootstrap(x, y, n_boot=2000, seed=0, alpha=0.05):
    """Spearman 相关及配对 bootstrap 百分位 CI。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = np.array([spearman(x[i], y[i]) for i in idx])
    lo, hi = np.nanquantile(boots, [alpha / 2, 1 - alpha / 2])
    return spearman(x, y), float(lo), float(hi)


def bootstrap_fractions(d_raw, d_pos, d_posmass, d_mass, n_boot=2000, seed=0,
                        alpha=0.05):
    """低频 Δ 的分解比例（均值之比：各步 Δ 先对样本求均值再相除）及配对 bootstrap CI。

    位置 = (Δraw − Δpos)/Δraw，墨量 = (Δpos − Δposmass)/Δraw，形状 = Δposmass/Δraw，
    三者和为 1。mass_first = (Δraw − Δmass)/Δraw 是先做墨量归一时墨量的解释比例。
    """
    arrs = [np.asarray(v, dtype=np.float64) for v in (d_raw, d_pos, d_posmass, d_mass)]
    n = arrs[0].size

    def fr(m_raw, m_pos, m_pm, m_m):
        return {"position": (m_raw - m_pos) / m_raw,
                "mass": (m_pos - m_pm) / m_raw,
                "shape": m_pm / m_raw,
                "mass_first": (m_raw - m_m) / m_raw}

    point = fr(*[v.mean() for v in arrs])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = fr(*[v[idx].mean(axis=1) for v in arrs])
    out = {}
    for k in point:
        lo, hi = np.quantile(boots[k], [alpha / 2, 1 - alpha / 2])
        out[k] = {"frac": float(point[k]), "ci95": [float(lo), float(hi)]}
    return out


def add_derived(row):
    """补绝对值列。"""
    for tag in ("base", "t1"):
        row[f"{tag}_abs_rel_mass"] = abs(row[f"{tag}_rel_mass"])
        row[f"{tag}_abs_logsx"] = abs(row[f"{tag}_logsx"])
        row[f"{tag}_abs_logsy"] = abs(row[f"{tag}_logsy"])
    row["d_l1_norm3"] = row["t1_l1_norm3"] - row["base_l1_norm3"]
    return row


def _mean_ci(v, n_boot, seed):
    m, lo, hi = paired_bootstrap_ci(v, n_boot, seed)
    return {"mean": m, "ci95": [lo, hi], "ci_excludes_0": bool(lo > 0 or hi < 0)}


def summarize(rows, sigmas, n_boot=2000, seed=0):
    summary = {}
    for kind in ("BF", "JT"):
        kr = [r for r in rows if r["kind"] == kind]
        if not kr:
            continue
        col = lambda k: np.array([r[k] for r in kr], dtype=np.float64)  # noqa: E731
        s = {"n": len(kr)}
        # A：逐模型带符号量 + 配对差
        a_part = {}
        for q in sorted(set(SIGNED_QTYS) | set(DELTA_QTYS)):
            base, t1 = col(f"base_{q}"), col(f"t1_{q}")
            a_part[q] = {"base": _mean_ci(base, n_boot, seed),
                         "t1": _mean_ci(t1, n_boot, seed),
                         "delta": _mean_ci(t1 - base, n_boot, seed)}
        a_part["gt_mass_mean"] = float(col("gt_mass").mean())
        a_part["frac_t1_more_ink_than_base"] = float(
            (col("t1_mass") > col("base_mass")).mean())
        s["moments"] = a_part
        # Spearman：Δl1_norm3 与各 Δ 量
        dl1 = col("d_l1_norm3")
        sp = {}
        targets = {q: col(f"t1_{q}") - col(f"base_{q}") for q in DELTA_QTYS}
        for sigma in sigmas:
            targets[f"s{sigma:g}_elow_raw"] = (col(f"t1_s{sigma:g}_elow_raw")
                                               - col(f"base_s{sigma:g}_elow_raw"))
        for q, v in targets.items():
            r, lo, hi = spearman_bootstrap(dl1, v, n_boot, seed)
            sp[q] = {"rho": r, "ci95": [lo, hi]}
        s["spearman_vs_d_l1_norm3"] = sp
        s["l1_norm3"] = {"base": float(col("base_l1_norm3").mean()),
                         "t1": float(col("t1_l1_norm3").mean()),
                         "delta": _mean_ci(dl1, n_boot, seed)}
        # B：反事实分解
        b_part = {}
        for sigma in sigmas:
            blk = {}
            d = {}
            for step in STEPS:
                key = f"s{sigma:g}_elow_{step}"
                base, t1 = col(f"base_{key}"), col(f"t1_{key}")
                d[step] = t1 - base
                blk[step] = {"base": float(base.mean()), "t1": float(t1.mean()),
                             "delta": _mean_ci(d[step], n_boot, seed)}
            blk["fractions"] = bootstrap_fractions(d["raw"], d["pos"], d["posmass"],
                                                   d["mass"], n_boot, seed)
            # 分母 Δraw 的 CI 含 0 时比例不可解释（点估计与 CI 可翻号、发散）
            blk["fractions_interpretable"] = blk["raw"]["delta"]["ci_excludes_0"]
            b_part[f"sigma_{sigma:g}"] = blk
        etot = {}
        for step in STEPS:
            base, t1 = col(f"base_e2_total_{step}"), col(f"t1_e2_total_{step}")
            etot[step] = {"base": float(base.mean()), "t1": float(t1.mean()),
                          "delta": _mean_ci(t1 - base, n_boot, seed)}
        b_part["e2_total"] = etot
        b_part["align_diag"] = {
            f"{tag}_{k}": {"mean": float(col(f"{tag}_{k}").mean()),
                           "max": float(col(f"{tag}_{k}").max()),
                           "min": float(col(f"{tag}_{k}").min())}
            for tag in ("base", "t1")
            for k in ("align_resid", "align_mass_kept", "scale_factor", "frac_gt1")}
        s["counterfactual"] = b_part
        summary[kind] = s
    return summary


# ---------------------------------------------------------------------------
# 作图（matplotlib 延迟导入；服务器无 CJK 字体，标签只用 ASCII，字形用印刷体小图显示）
# ---------------------------------------------------------------------------

def _diverging_cmap():
    """PuOr 色盲友好发散色图，中心强制为纯白：紫 = 正（多墨），橙 = 负（少墨）。"""
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap
    cols = colormaps["PuOr"](np.linspace(0.0, 1.0, 11))
    cols[5] = (1.0, 1.0, 1.0, 1.0)
    return LinearSegmentedColormap.from_list("PuOr_w", cols, N=256)


def _row_label(rec):
    ch = Path(rec["query_path"]).stem
    code = " ".join(f"U+{ord(c):04X}" for c in ch)
    return (f"#{rec['idx']} {code}\n{rec['writer']}\n"
            f"dL1 = {rec['d_l1_norm3']:+.3f}\n"
            f"mass b/T1: {rec['base_rel_mass']:+.1%} / {rec['t1_rel_mass']:+.1%}\n"
            f"disp b/T1: {rec['base_disp']:.1f} / {rec['t1_disp']:.1f} px")


def plot_rows(recs, imgs, path, panel_in=1.6, dpi=140, title=None):
    """每个样本一行：标签 | 印刷体字 | 参考字 | GT | baseline | T1 | base 误差 | T1 误差 | T1−base。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = _diverging_cmap()
    heads = ["glyph (input)", "style ref", "GT", "baseline", "T1",
             "err base\na_pred \u2212 a_gt", "err T1\na_pred \u2212 a_gt", "a_T1 \u2212 a_base"]
    n = len(recs)
    fig = plt.figure(figsize=(panel_in * (len(heads) + 1.5), panel_in * n + 0.9),
                     dpi=dpi)
    gs = fig.add_gridspec(n + 1, len(heads) + 1,
                          width_ratios=[1.5] + [1.0] * len(heads),
                          height_ratios=[1.0] * n + [0.12],
                          wspace=0.03, hspace=0.06)
    im = None
    for r, rec in enumerate(recs):
        d = imgs[rec["idx"]]
        tax = fig.add_subplot(gs[r, 0])
        tax.axis("off")
        tax.text(0.98, 0.5, _row_label(rec), ha="right", va="center",
                 fontsize=7, family="monospace", transform=tax.transAxes)
        panels = [d["glyph"], d["ref"], d["gt"], d["base"], d["t1"],
                  d["a_base"] - d["a_gt"], d["a_t1"] - d["a_gt"],
                  d["a_t1"] - d["a_base"]]
        for c, p in enumerate(panels):
            ax = fig.add_subplot(gs[r, c + 1])
            if p.ndim == 3:
                ax.imshow(p, interpolation="lanczos")
            else:
                im = ax.imshow(p, cmap=cmap, vmin=-1.0, vmax=1.0,
                               interpolation="lanczos")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(heads[c], fontsize=7)
    cax = fig.add_subplot(gs[n, 4:8])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=6)
    cb.set_label("signed soft-ink difference (fixed scale -1..1): "
                 "purple = more ink, orange = less ink, white = 0", fontsize=7)
    if title:
        fig.suptitle(title, fontsize=9, y=0.995)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# torch / 模型部分（仅在服务器运行）
# ---------------------------------------------------------------------------

def _to_rgb_u8(x_norm):
    img = x_norm * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
    return (np.clip(img, 0.0, 1.0).transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)


def run_models(args):
    import torch

    from data.pairdataset import PairDataset
    from tools.eval_s_baseline import build_val_transform, load_model
    from util.masking_generator import MaskingGenerator

    torch.manual_seed(args.seed)
    # 构造参数与 eval_band_error.run_models 逐项相同（逐样本对账见 reconcile_band）
    dataset = PairDataset(
        args.data_path, args.val_json_path, transform=build_val_transform(),
        masked_position_generator=MaskingGenerator(
            (56, 28), num_masking_patches=784, max_num_patches=392),
        use_two_pairs=True, half_mask_ratio=1.0,
        semantic_mask_dir=args.semantic_mask_dir,
        num_mask_annotations_jt=1, num_mask_annotations_bf=11,
        mask_coverage_threshold=0.1, return_jieti=True,
        fixed_pair_path=args.fixed_pair_path,
    )
    models = {
        "base": load_model(args.baseline_ckpt).to(args.device).eval(),
        "t1": load_model(args.t1_ckpt).to(args.device).eval(),
    }

    rows, imgs = [], {}
    with torch.no_grad():
        for idx in range(len(dataset)):
            image, target, mask, _valid, _voro, _vp, is_jt = dataset[idx]
            kind = "JT" if bool(is_jt.item()) else "BF"
            images = image.unsqueeze(0).to(args.device)
            targets = target.unsqueeze(0).to(args.device)
            bmp = torch.from_numpy(mask).bool().flatten()[None].to(args.device)
            hp = mask.shape[0] // 2 * 16
            if not mask[mask.shape[0] // 2:].all() or mask[:mask.shape[0] // 2].any():
                raise RuntimeError(f"sample {idx}: val mask 不是下半整字全遮")

            tgt_full = targets[0].double().cpu().numpy()
            gt_np = tgt_full[:, hp:, :]
            ref_path = dataset.pairs[idx]["target_path"]
            q = dataset._fixed_pairs[ref_path]
            row = {"idx": idx, "kind": kind,
                   "writer": q["type"].replace("font_", "").replace(kind, ""),
                   "target_path": ref_path, "query_path": q["target_path"]}
            keep = {}
            mg0 = None
            for tag, model in models.items():
                # GT 侧每个模型分支各自重算，再 assert 逐值相同（GT 只依赖数据）
                a_gt = soft_ink(denorm_to_gray(gt_np, clamp=False))
                mg = ink_moments(a_gt)
                if mg0 is None:
                    mg0 = mg
                    row.update({f"gt_{k}": v for k, v in mg.items()})
                elif mg != mg0:
                    raise RuntimeError(f"sample {idx}: GT 矩在两模型分支间不一致")
                latent = model.forward_encoder(images, targets, bmp)
                pred_np = model.forward_decoder(latent)[0].double().cpu().numpy()[:, hp:, :]
                row[f"{tag}_l1_norm3"] = float(np.abs(pred_np - gt_np).mean())
                a_p = soft_ink(denorm_to_gray(pred_np, clamp=True))
                mp = ink_moments(a_p)
                # jieti_loss 口径（RGB 不先截断）的墨量，仅用于量化两种口径的差别
                g_unc = denorm_to_gray(pred_np, clamp=False)
                row[f"{tag}_mass_unclamped_rgb"] = float(soft_ink(g_unc).sum())
                row.update({f"{tag}_{k}": v for k, v in mp.items()})
                row.update({f"{tag}_{k}": v for k, v in moment_qtys(mp, mg).items()})
                row.update({f"{tag}_{k}": v for k, v in
                            counterfactual(a_p, a_gt, mg, args.sigmas).items()})
                if kind == "BF":
                    keep[tag] = _to_rgb_u8(pred_np)
                    keep[f"a_{tag}"] = a_p.astype(np.float32)
                    keep["a_gt"] = a_gt.astype(np.float32)
            if kind == "BF":
                keep["gt"] = _to_rgb_u8(gt_np)
                keep["ref"] = _to_rgb_u8(tgt_full[:, :hp, :])
                keep["glyph"] = _to_rgb_u8(image.double().numpy()[:, hp:, :])
                imgs[idx] = keep
            rows.append(add_derived(row))
            if len(rows) % 20 == 0:
                print(f"[ink] {len(rows)} samples done", flush=True)
    return rows, imgs


def reconcile_band(rows, band_csv, sigmas):
    """与 outputs/band_error/per_image.csv 逐样本对账：l1_norm3 与 s{σ}_e2_low。"""
    with open(band_csv, encoding="utf-8") as f:
        band = {int(r["idx"]): r for r in csv.DictReader(f)}
    if len(band) != len(rows):
        raise RuntimeError(f"样本数不一致：band {len(band)} vs 本次 {len(rows)}")
    worst = {}
    for r in rows:
        b = band[r["idx"]]
        if b["kind"] != r["kind"] or b["query_path"] != r["query_path"]:
            raise RuntimeError(f"sample {r['idx']}: kind/query_path 与 band_error 不一致")
        for tag in ("base", "t1"):
            keys = [(f"{tag}_l1_norm3", f"{tag}_l1_norm3")]
            keys += [(f"{tag}_s{s:g}_e2_low", f"{tag}_s{s:g}_elow_raw") for s in sigmas]
            for bk, rk in keys:
                if bk not in b:
                    raise RuntimeError(f"band csv 缺列 {bk}，无法对账（--sigmas 与上一轮不一致？）")
                rel = abs(float(b[bk]) - r[rk]) / max(abs(float(b[bk])), 1e-12)
                worst[bk.replace(f"{tag}_", "")] = max(worst.get(bk.replace(f"{tag}_", ""), 0.0), rel)
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline_ckpt", required=True)
    parser.add_argument("--t1_ckpt", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--val_json_path", nargs="+", required=True)
    parser.add_argument("--semantic_mask_dir", default=None)
    parser.add_argument("--fixed_pair_path", required=True)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[2.0, 4.0, 8.0])
    parser.add_argument("--band_csv", default="outputs/band_error/per_image.csv",
                        help="上一轮逐样本结果，用于逐样本对账；不存在则报错")
    parser.add_argument("--band_rel_tol", type=float, default=1e-3)
    parser.add_argument("--expect_bf_l1", type=float, nargs=2, default=[0.6168, 0.6754],
                        help="BF l1_norm3 均值期望值（baseline, T1）")
    parser.add_argument("--expect_tol", type=float, default=5e-4)
    parser.add_argument("--n_worst", type=int, default=10)
    parser.add_argument("--n_median", type=int, default=5)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="outputs/ink_moments")
    args = parser.parse_args()

    out = Path(args.output_dir)
    if out.exists():
        sys.exit(f"输出目录已存在，不覆盖：{out}")
    if not Path(args.band_csv).is_file():
        sys.exit(f"找不到对账用的 band_error 逐样本结果：{args.band_csv}")

    t0 = time.time()
    rows, imgs = run_models(args)
    t_fwd = time.time() - t0

    # 对账：逐样本 vs band_error；BF 均值 vs 期望。任一不过就停，不写输出。
    recon = reconcile_band(rows, args.band_csv, args.sigmas)
    print("[ink] 逐样本对账最大相对误差:", json.dumps(recon), flush=True)
    bad = {k: v for k, v in recon.items() if v > args.band_rel_tol}
    bf = [r for r in rows if r["kind"] == "BF"]
    bf_l1 = (float(np.mean([r["base_l1_norm3"] for r in bf])),
             float(np.mean([r["t1_l1_norm3"] for r in bf])))
    print(f"[ink] BF l1_norm3 base={bf_l1[0]:.5f} t1={bf_l1[1]:.5f}", flush=True)
    if bad or any(abs(v - e) > args.expect_tol for v, e in zip(bf_l1, args.expect_bf_l1)):
        sys.exit(f"对账失败：band 超差 {bad}，BF l1 {bf_l1} vs 期望 {args.expect_bf_l1}")

    summary = summarize(rows, args.sigmas, args.n_boot, args.seed)
    summary["reconcile"] = {"band_max_rel_err": recon, "bf_l1_norm3": bf_l1,
                            "expect_bf_l1": args.expect_bf_l1,
                            "gt_moments_identical_across_models": True}
    summary["mass_def_rel_diff_max"] = float(max(
        abs(r[f"{t}_mass_unclamped_rgb"] - r[f"{t}_mass"]) / r[f"{t}_mass"]
        for r in rows for t in ("base", "t1")))

    # C：最差 / 中位样本
    bf_sorted = sorted(bf, key=lambda r: r["d_l1_norm3"], reverse=True)
    worst = bf_sorted[:args.n_worst]
    med = float(np.median([r["d_l1_norm3"] for r in bf]))
    median = sorted(sorted(bf, key=lambda r: abs(r["d_l1_norm3"] - med))[:args.n_median],
                    key=lambda r: r["d_l1_norm3"], reverse=True)
    meta_keys = ("idx", "writer", "query_path", "target_path", "d_l1_norm3",
                 "base_rel_mass", "t1_rel_mass", "base_disp", "t1_disp",
                 "base_logsx", "t1_logsx", "base_logsy", "t1_logsy",
                 "gt_mass", "gt_sx", "gt_sy")
    summary["worst"] = [{k: r[k] for k in meta_keys} for r in worst]
    summary["median"] = {"d_l1_median": med,
                         "samples": [{k: r[k] for k in meta_keys} for r in median]}
    summary["config"] = dict(vars(args))
    summary["timing_s"] = {"forward_and_metrics": t_fwd}

    out.mkdir(parents=True)
    (out / "worst10").mkdir()
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for rank, r in enumerate(worst, 1):
        code = "_".join(f"U{ord(c):04X}" for c in Path(r["query_path"]).stem)
        plot_rows([r], imgs, out / "worst10" / f"{rank:02d}_idx{r['idx']:03d}_{code}.png",
                  panel_in=2.4, dpi=150)
    plot_rows(worst, imgs, out / "grid_worst10.png",
              title=f"BF worst {len(worst)} by dL1_norm3 (T1 - baseline)")
    plot_rows(median, imgs, out / "grid_median5.png",
              title=f"BF {len(median)} samples closest to median dL1_norm3 = {med:+.4f}")
    summary["timing_s"]["total"] = time.time() - t0
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("reconcile", "timing_s")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
