"""推理时融合权重 λ 的诊断（T1-L）：同一 checkpoint、同一份输入，只改 encoder 融合权重。

融合在第 3 个 ViT block（idx 2）之后：F = (1−λ)·F_电脑字 + λ·F_风格字（models_train.fusion_lambda）。
- 数据集构造与 tools/eval_s_baseline.py / export_side_by_side.py 逐项相同，下半整字全遮；
  每条样本只调一次 dataset[idx]，同一个模型依次设各个 λ 前向，各档输入完全相同。
- 图：每行 上半参考字 | 下半 GT | 每个 λ 一列（列头 λ=0.3 等），单张图、按 JT/BF 分页的拼图、
  index.csv，规格与写法沿用 export_side_by_side（临时目录 + rename）。
- 数值：每个 λ 的 L1_JT、L1_BF、J，口径与 eval_s_baseline 完全一致；按 s_baseline 分母算
  三项比值与 S，写 metrics.json；逐样本写 per_sample.csv（idx, kind, lambda, l1, J，BF 的 J 为空）。
- 自检：λ=0.5 的三项必须与 --expect_lambda05 一致（相对差 ≤ --expect_rtol），否则报错退出，
  临时目录删掉，不留输出。
- 列头的 λ 需要带希腊字母的 TTF（Pillow 内置字体没有），默认用服务器上的 DejaVuSans。

numpy/PIL 部分可以在没有 torch 的环境单测；torch 与模型代码在 run_model() 里延迟导入。
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.export_side_by_side as sbs  # noqa: E402
from tools.export_jieti_ab import to_rgb_u8  # noqa: E402
from tools.run_jieti_search import read_s_baseline  # noqa: E402

DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
PER_SAMPLE_FIELDS = ("idx", "kind", "lambda", "l1", "J")
METRIC_KEYS = ("L1_JT", "L1_BF", "J")


# ---------------------------------------------------------------------------
# numpy / PIL 部分（可在无 torch 环境测试）
# ---------------------------------------------------------------------------

def lam_key(lam):
    """λ 的文本形式（列头、json 键、csv 列共用）：0.3 → "0.3"。"""
    return f"{lam:g}"


def column_heads(lambdas):
    return ["ref (upper GT)", "GT (lower)"] + [f"λ={lam_key(lam)}" for lam in lambdas]


def check_lambdas(lambdas):
    keys = [lam_key(lam) for lam in lambdas]
    if len(set(keys)) != len(keys):
        raise ValueError(f"--lambdas 有重复：{lambdas}")
    if "0.5" not in keys:
        raise ValueError("--lambdas 必须包含 0.5（与现有代码对账）")
    if not all(0.0 <= lam <= 1.0 for lam in lambdas):
        raise ValueError(f"--lambdas 须在 [0, 1] 内：{lambdas}")


def summarize(per_sample, lambdas, s_base):
    """per_sample: [{idx, kind, lambda(float), l1, J(None 表示 BF)}]，按 λ 汇总。

    均值与 eval_s_baseline 相同：L1 分 JT/BF 各自 np.mean，J 只在 JT 上 np.mean；
    比值 = 本组 / s_baseline，S = 三项比值之和。
    """
    out = {}
    for lam in lambdas:
        rows = [r for r in per_sample if lam_key(r["lambda"]) == lam_key(lam)]
        l1_jt = [r["l1"] for r in rows if r["kind"] == "JT"]
        l1_bf = [r["l1"] for r in rows if r["kind"] == "BF"]
        j_vals = [r["J"] for r in rows if r["kind"] == "JT"]
        m = {
            "L1_JT": float(np.mean(l1_jt)) if l1_jt else float("nan"),
            "L1_BF": float(np.mean(l1_bf)) if l1_bf else float("nan"),
            "J": float(np.mean(j_vals)) if j_vals else float("nan"),
            "n_jt": len(l1_jt), "n_bf": len(l1_bf),
        }
        for k in METRIC_KEYS:
            m[f"{k}_ratio"] = m[k] / s_base[k]
        m["S"] = sum(m[f"{k}_ratio"] for k in METRIC_KEYS)
        out[lam_key(lam)] = m
    return out


def reconcile(per_lambda, expect, rtol):
    """λ=0.5 的三项与期望值对账；任何一项相对差 > rtol（或非有限）就报错。"""
    got = per_lambda["0.5"]
    rel = {k: abs(got[k] - e) / abs(e) for k, e in zip(METRIC_KEYS, expect)}
    report = {"lambda": 0.5, "expect": dict(zip(METRIC_KEYS, expect)),
              "got": {k: got[k] for k in METRIC_KEYS}, "rel_diff": rel, "rtol": rtol}
    bad = {k: v for k, v in rel.items() if not (math.isfinite(v) and v <= rtol)}
    if bad:
        raise RuntimeError(f"λ=0.5 对账失败（rtol={rtol}）：{json.dumps(report, ensure_ascii=False)}")
    return report


def write_per_sample(path, per_sample):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PER_SAMPLE_FIELDS)
        w.writeheader()
        for r in sorted(per_sample, key=lambda r: (r["idx"], r["lambda"])):
            w.writerow({"idx": r["idx"], "kind": r["kind"], "lambda": lam_key(r["lambda"]),
                        "l1": repr(r["l1"]), "J": "" if r["J"] is None else repr(r["J"])})


def make_finalizer(per_sample, lambdas, s_base, expect, rtol, meta):
    """返回 write_outputs 的 before_rename 回调：汇总、对账、写 metrics.json 与 per_sample.csv。

    对账失败时抛异常，write_outputs 会删掉临时目录；失败前先把汇总打印到 stdout 方便排查。
    """
    def finalize(tmp, recs):
        n_items = len(recs)
        if len(per_sample) != n_items * len(lambdas):
            raise RuntimeError(f"逐样本条数 {len(per_sample)} ≠ {n_items}×{len(lambdas)}")
        per_lambda = summarize(per_sample, lambdas, s_base)
        print("[fusion] per_lambda = " + json.dumps(per_lambda, indent=2), flush=True)
        rep = reconcile(per_lambda, expect, rtol)
        metrics = dict(meta)
        metrics.update({"lambdas": [lam_key(lam) for lam in lambdas], "s_baseline": s_base,
                        "per_lambda": per_lambda, "reconcile": rep})
        (tmp / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
        write_per_sample(tmp / "per_sample.csv", per_sample)
    return finalize


# ---------------------------------------------------------------------------
# torch / 模型部分（仅在服务器运行）
# ---------------------------------------------------------------------------

def run_model(args, lambdas, per_sample):
    """逐条产出 (rec, [ref, gt, pred_λ1, pred_λ2, ...], glyph)，并把逐样本 l1/J 追加到 per_sample。"""
    import torch

    from data.pairdataset import PairDataset
    from tools.eval_s_baseline import build_val_transform, load_model
    from util.jieti_loss import JietiLoss
    from util.masking_generator import MaskingGenerator

    # 不设 torch.manual_seed，且按 eval_s_baseline 的顺序建数据集、加载模型：val 变换里的
    # RandomResizedCrop(scale≈1) 仍从 torch 默认随机流取裁剪参数，顺序一致才能逐样本复现它
    coef = json.loads(Path(args.calibration_json).read_text())["coef"]
    # 构造参数与 eval_s_baseline / export_side_by_side 逐项相同
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
    model = load_model(args.model_ckpt).to(args.device).eval()
    jmod = JietiLoss(w_centroid=coef["centroid"], w_logsigma=coef["logsigma"],
                     w_shape=coef["shape"]).to(args.device)

    with torch.no_grad():
        for idx in range(len(dataset)):
            image, target, mask, _valid, voro, valid_parts, is_jt = dataset[idx]
            kind = "JT" if bool(is_jt.item()) else "BF"
            if not mask[mask.shape[0] // 2:].all() or mask[:mask.shape[0] // 2].any():
                raise RuntimeError(f"sample {idx}: val mask 不是下半整字全遮")
            images = image.unsqueeze(0).to(args.device)
            targets = target.unsqueeze(0).to(args.device)
            bmp = torch.from_numpy(mask).bool().flatten()[None].to(args.device)
            voro_b = voro.unsqueeze(0).to(args.device)
            vp_b = valid_parts.unsqueeze(0).to(args.device)
            is_jt_b = is_jt.view(1).to(args.device)
            hp = mask.shape[0] // 2 * 16  # 下半 query 起始行（像素）
            mask_px = bmp[:, :, None].repeat(1, 1, model.patch_size ** 2 * 3).float()
            mask_px = model.unpatchify(mask_px)

            ref_path = dataset.pairs[idx]["target_path"]
            q = dataset._fixed_pairs[ref_path]
            rec = {"idx": idx, "kind": kind,
                   "writer": q["type"].replace("font_", "").replace(kind, ""),
                   "char": Path(q["target_path"]).stem,
                   "ref_path": ref_path, "query_path": q["target_path"]}
            tgt = target.double().numpy()
            panels = [to_rgb_u8(tgt[:, :hp, :]), to_rgb_u8(tgt[:, hp:, :])]
            for lam in lambdas:
                model.fusion_lambda = lam
                pred = model.forward_decoder(model.forward_encoder(images, targets, bmp))
                # 以下与 eval_s_baseline 逐行相同：不加权重建 L1（遮盖区），J 只在 JT 上算
                l1 = ((pred - targets).abs() * mask_px).sum() / (mask_px.sum() + 1e-2)
                J = None
                if kind == "JT":
                    composite = pred * mask_px + targets * (1 - mask_px)
                    J = float(jmod(composite, targets, voro_b, vp_b, is_jt_b.bool())[0].item())
                per_sample.append({"idx": idx, "kind": kind, "lambda": lam,
                                   "l1": float(l1.item()), "J": J})
                panels.append(to_rgb_u8(pred[0].double().cpu().numpy()[:, hp:, :]))
            glyph = to_rgb_u8(image.double().numpy()[:, hp:, :])
            yield rec, panels, glyph
            if (idx + 1) % 20 == 0:
                print(f"[fusion] {idx + 1}/{len(dataset)} samples done", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model_ckpt", required=True)
    parser.add_argument("--model_name", required=True, help="写进 metrics.json 的模型名")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--val_json_path", nargs="+", required=True)
    parser.add_argument("--semantic_mask_dir", default=None)
    parser.add_argument("--fixed_pair_path", required=True)
    parser.add_argument("--calibration_json", required=True,
                        help="calibrate_jieti.py 的输出，取 coef 作三项系数（同 eval_s_baseline）")
    parser.add_argument("--s_baseline_json", required=True, help="S 的分母")
    parser.add_argument("--expect_lambda05", type=float, nargs=3, required=True,
                        metavar=("L1_JT", "L1_BF", "J"), help="λ=0.5 的对账值（工具口径）")
    parser.add_argument("--expect_rtol", type=float, default=1e-4)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--font_path", default=DEFAULT_FONT, help="列头/行标签用的 TTF（需含 λ）")
    parser.add_argument("--rows_per_page", type=int, default=15)
    parser.add_argument("--grid_cell", type=int, default=256, help="拼图里每个面板的边长（像素）")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # 加载模型之前就检查参数、输入文件与输出目录
    out = Path(args.output_dir)
    sbs.check_output_free(out)
    check_lambdas(args.lambdas)
    for p in (args.font_path, args.calibration_json, args.s_baseline_json):
        if not Path(p).is_file():
            raise FileNotFoundError(p)
    s_base = read_s_baseline(args.s_baseline_json)

    per_sample = []
    meta = {"model_ckpt": args.model_ckpt, "model_name": args.model_name,
            "s_baseline_json": args.s_baseline_json, "calibration_json": args.calibration_json}
    finalize = make_finalizer(per_sample, args.lambdas, s_base, args.expect_lambda05,
                              args.expect_rtol, meta)
    recs, pages = sbs.write_outputs(
        run_model(args, args.lambdas, per_sample), out, column_heads(args.lambdas),
        args.rows_per_page, args.grid_cell, font_path=args.font_path, before_rename=finalize)
    n = {k: sum(r["kind"] == k for r in recs) for k in ("JT", "BF")}
    print(f"[fusion] done: {len(recs)} items (JT {n['JT']} / BF {n['BF']}), "
          f"pages JT {len(pages['JT'])} / BF {len(pages['BF'])} -> {out}", flush=True)


if __name__ == "__main__":
    main()
