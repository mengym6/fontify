"""BF/JT 重建误差的频带拆分：baseline 与 T1 在固定 val 上的逐样本配对对比。

目的：回答"T1 相对 baseline 的 L1_BF 变差，主要落在低频（墨团位置、粗细、
整体形状）还是高频（边缘、锋芒等细节）"。只做评测，不训练、不改训练文件。

做法（每个样本、每个 sigma）：
    e      = pred_gray - gt_gray          下半 query 区域（整字全遮），[0,1] 灰度
    e_low  = G_sigma * e                  高斯低通，reflect 边界
    e_high = e - e_low
报告 L1（mean|.|）与能量（mean(.^2)）。能量满足闭合恒等式
    E_total = E_low + E_high + 2 * C,     C = mean(e_low * e_high)
交叉项 C 一并输出，保证拆分可对账。

两个模型在同一个 dataset 样本上前向，输入逐位相同，所以配对差
Delta = T1 - baseline 只来自模型。95% 置信区间用逐样本配对 bootstrap。

数值部分只依赖 numpy；torch 与模型代码在 run_models() 里延迟导入，
这样无 torch 的环境也能对统计部分做单测。
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float64)
GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float64)
METRICS = ("l1_total", "l1_low", "l1_high",
           "e2_total", "e2_low", "e2_high", "e2_cross")


# ---------------------------------------------------------------------------
# numpy 数值部分（可在无 torch 环境测试）
# ---------------------------------------------------------------------------

def gaussian_kernel_1d(sigma):
    """归一化一维高斯核，半径 ceil(3*sigma)。"""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(img, sigma):
    """二维可分离高斯低通，reflect 边界。img: (H, W) float64。"""
    k = gaussian_kernel_1d(sigma)
    r = (len(k) - 1) // 2
    if r >= min(img.shape):
        raise ValueError("kernel radius exceeds image size")
    h, w = img.shape
    pad = np.pad(img, ((0, 0), (r, r)), mode="reflect")
    out = np.zeros((h, w), dtype=np.float64)
    for i, wi in enumerate(k):
        out += wi * pad[:, i:i + w]
    pad = np.pad(out, ((r, r), (0, 0)), mode="reflect")
    out2 = np.zeros((h, w), dtype=np.float64)
    for i, wi in enumerate(k):
        out2 += wi * pad[i:i + h, :]
    return out2


def denorm_to_gray(x_norm, clamp=True):
    """(3, H, W) ImageNet 归一化张量 → (H, W) [0,1] 灰度。

    clamp=True 对应保存/观看时的实际输出（像素截断到 [0,1]）。
    """
    img = x_norm * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
    if clamp:
        img = np.clip(img, 0.0, 1.0)
    return np.tensordot(GRAY_WEIGHTS, img, axes=(0, 0))


def band_metrics(err, sigma):
    """对误差图 err (H, W) 做低/高频拆分，返回 METRICS 对应的标量字典。"""
    low = gaussian_blur(err, sigma)
    high = err - low
    return {
        "l1_total": float(np.abs(err).mean()),
        "l1_low": float(np.abs(low).mean()),
        "l1_high": float(np.abs(high).mean()),
        "e2_total": float((err ** 2).mean()),
        "e2_low": float((low ** 2).mean()),
        "e2_high": float((high ** 2).mean()),
        "e2_cross": float((low * high).mean()),
    }


def paired_bootstrap_ci(delta, n_boot=2000, seed=0, alpha=0.05):
    """逐样本配对差 delta 的均值及 bootstrap 百分位置信区间。"""
    delta = np.asarray(delta, dtype=np.float64)
    if delta.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, delta.size, size=(n_boot, delta.size))
    boots = delta[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(delta.mean()), float(lo), float(hi)


def summarize(rows, sigmas, n_boot=2000, seed=0):
    """按 kind × sigma × metric 汇总 baseline/T1 均值与配对差置信区间。"""
    summary = {}
    for kind in ("BF", "JT"):
        kind_rows = [r for r in rows if r["kind"] == kind]
        if not kind_rows:
            continue
        summary[kind] = {"n": len(kind_rows)}
        for sigma in sigmas:
            block = {}
            for m in METRICS:
                key = f"s{sigma:g}_{m}"
                base = np.array([r[f"base_{key}"] for r in kind_rows])
                t1 = np.array([r[f"t1_{key}"] for r in kind_rows])
                mean_d, lo, hi = paired_bootstrap_ci(t1 - base, n_boot, seed)
                block[m] = {
                    "base": float(base.mean()),
                    "t1": float(t1.mean()),
                    "delta": mean_d,
                    "ci95": [lo, hi],
                    "rel_delta": (mean_d / float(base.mean())
                                  if base.mean() != 0 else float("nan")),
                    "ci_excludes_0": bool(lo > 0 or hi < 0),
                }
            # 能量变化的分解：dE_total = dE_low + dE_high + 2 dC（逐项对账）
            d = {m: block[m]["delta"] for m in METRICS}
            block["closure_residual"] = (d["e2_total"] - d["e2_low"]
                                         - d["e2_high"] - 2 * d["e2_cross"])
            summary[kind][f"sigma_{sigma:g}"] = block
        for name in ("l1_norm3",):
            base = np.array([r[f"base_{name}"] for r in kind_rows])
            t1 = np.array([r[f"t1_{name}"] for r in kind_rows])
            mean_d, lo, hi = paired_bootstrap_ci(t1 - base, n_boot, seed)
            summary[kind][name] = {"base": float(base.mean()),
                                   "t1": float(t1.mean()),
                                   "delta": mean_d, "ci95": [lo, hi]}
    return summary


# ---------------------------------------------------------------------------
# torch / 模型部分（仅在服务器运行）
# ---------------------------------------------------------------------------

def run_models(args):
    """两个 checkpoint 在同一批固定 val 样本上前向，返回逐样本指标行。"""
    import torch

    from data.pairdataset import PairDataset
    from tools.eval_s_baseline import build_val_transform, load_model
    from util.masking_generator import MaskingGenerator

    torch.manual_seed(args.seed)
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

    rows = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            image, target, mask, _valid, _voro, _vp, is_jt = dataset[idx]
            kind = "JT" if bool(is_jt.item()) else "BF"
            if args.kind != "all" and kind != args.kind:
                continue
            images = image.unsqueeze(0).to(args.device)
            targets = target.unsqueeze(0).to(args.device)
            bmp = torch.from_numpy(mask).bool().flatten()[None].to(args.device)
            hp = mask.shape[0] // 2 * 16  # 下半 query 起始行（像素）
            if not mask[mask.shape[0] // 2:].all() or mask[:mask.shape[0] // 2].any():
                raise RuntimeError(f"sample {idx}: val mask 不是下半整字全遮")

            gt_np = targets[0].double().cpu().numpy()[:, hp:, :]
            gt_gray = denorm_to_gray(gt_np, clamp=False)
            # target_path 是上半参考字（固定配对的 key）；误差只在下半算，被评测的字是 query_path
            ref_path = dataset.pairs[idx]["target_path"]
            row = {"idx": idx, "kind": kind, "target_path": ref_path,
                   "query_path": dataset._fixed_pairs[ref_path]["target_path"]}
            for tag, model in models.items():
                latent = model.forward_encoder(images, targets, bmp)
                pred = model.forward_decoder(latent)
                pred_np = pred[0].double().cpu().numpy()[:, hp:, :]
                # 与 eval_s_baseline 同口径的 L1（归一化空间、三通道），用于对账
                row[f"{tag}_l1_norm3"] = float(np.abs(pred_np - gt_np).mean())
                err = denorm_to_gray(pred_np, clamp=True) - gt_gray
                for sigma in args.sigmas:
                    for m, v in band_metrics(err, sigma).items():
                        row[f"{tag}_s{sigma:g}_{m}"] = v
            rows.append(row)
            if (len(rows)) % 20 == 0:
                print(f"[band] {len(rows)} samples done", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline_ckpt", required=True)
    parser.add_argument("--t1_ckpt", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--val_json_path", nargs="+", required=True)
    parser.add_argument("--semantic_mask_dir", default=None)
    parser.add_argument("--fixed_pair_path", required=True)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[2.0, 4.0, 8.0],
                        help="高斯低通 sigma（448 分辨率像素）")
    parser.add_argument("--kind", choices=["BF", "JT", "all"], default="all")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="outputs/band_error")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = run_models(args)

    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, args.sigmas, args.n_boot, args.seed)
    summary["config"] = {k: v for k, v in vars(args).items()}
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
