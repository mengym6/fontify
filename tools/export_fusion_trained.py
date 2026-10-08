"""训练时融合权重 λ 的对照图（T1-L）：多个 (λ, ckpt)，每个 ckpt 用自己训练时的 λ 前向。

与 export_fusion_sweep（同一 ckpt、推理时改 λ）互补：这里每一列是一个单独训练的模型。
- λ 由 tools/eval_s_baseline.load_model 从 ckpt['args'].fusion_lambda 读出（旧 ckpt 无此字段按 0.5），
  并与 --pair 给的 λ 核对，不一致就报错（防止 ckpt 与 λ 对错）。
- 数据集构造与 eval_s_baseline 逐项相同，下半整字全遮；每条样本只调一次 dataset[idx]，
  各模型对同一份输入前向。建第 1 个模型后保存 torch 随机状态，其余模型建完再恢复，
  使 val 变换的裁剪参数与 eval_s_baseline（只建 1 个模型）取到同一段随机流。
- 图：每行 上半参考字 | 下半 GT | 每个 λ 一列（列头 "λ=0.3 (trained)"，按 λ 升序），
  单张图、按 JT/BF 分页的拼图、index.csv，写法沿用 export_side_by_side（临时目录 + rename）。
- 数值：每个 λ 的 L1_JT、L1_BF、J 与 S，写 metrics.json / per_sample.csv（同 export_fusion_sweep）。
- 对账：λ=0.5 的三项与 --expect_lambda05 比较（相对差 ≤ --expect_rtol），只记录 passed，不报错。

numpy/PIL 部分可以在没有 torch 的环境单测；torch 与模型代码在 run_models() 里延迟导入。
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tools.export_fusion_sweep as fs  # noqa: E402
import tools.export_side_by_side as sbs  # noqa: E402
from tools.export_jieti_ab import to_rgb_u8  # noqa: E402
from tools.run_jieti_search import read_s_baseline  # noqa: E402


# ---------------------------------------------------------------------------
# numpy / PIL 部分（可在无 torch 环境测试）
# ---------------------------------------------------------------------------

def parse_pairs(pairs):
    """--pair 的 [[λ字符串, ckpt], ...] → 按 λ 升序的 [(λ, ckpt)]；λ 须含 0.5、不重复、在 (0,1) 内。"""
    out = sorted(((float(lam), ckpt) for lam, ckpt in pairs), key=lambda p: p[0])
    lambdas = [lam for lam, _ in out]
    fs.check_lambdas(lambdas)
    if not all(0.0 < lam < 1.0 for lam in lambdas):
        raise ValueError(f"训练时的 λ 须在 (0, 1) 内：{lambdas}")
    return out


def column_heads(lambdas):
    return ["ref (upper GT)", "GT (lower)"] + [f"λ={fs.lam_key(lam)} (trained)" for lam in lambdas]


def check_ckpt_lambda(lam, got, ckpt):
    """--pair 给的 λ 与 ckpt['args'] 里训练时的 λ 必须一致。"""
    if fs.lam_key(lam) != fs.lam_key(got):
        raise ValueError(f"{ckpt}：--pair 写的 λ={lam}，ckpt 训练时 λ={got}")


# ---------------------------------------------------------------------------
# torch / 模型部分（仅在服务器运行）
# ---------------------------------------------------------------------------

def run_models(args, pairs, per_sample):
    """逐条产出 (rec, [ref, gt, pred_λ1, pred_λ2, ...], glyph)，并把逐样本 l1/J 追加到 per_sample。"""
    import torch

    from data.pairdataset import PairDataset
    from tools.eval_s_baseline import build_val_transform, load_model
    from util.jieti_loss import JietiLoss
    from util.masking_generator import MaskingGenerator

    # 不设 torch.manual_seed，顺序同 eval_s_baseline：读 coef → 建数据集 → 建模型
    coef = json.loads(Path(args.calibration_json).read_text())["coef"]
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
    models, rng_state = [], None
    for lam, ckpt in pairs:
        model = load_model(ckpt).to(args.device).eval()
        check_ckpt_lambda(lam, model.fusion_lambda, ckpt)
        models.append(model)
        if rng_state is None:
            rng_state = torch.get_rng_state()
    # 多建的模型也会消耗随机流（权重初始化），恢复到只建 1 个模型时的状态
    torch.set_rng_state(rng_state)
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
            mask_px = bmp[:, :, None].repeat(1, 1, models[0].patch_size ** 2 * 3).float()
            mask_px = models[0].unpatchify(mask_px)

            ref_path = dataset.pairs[idx]["target_path"]
            q = dataset._fixed_pairs[ref_path]
            rec = {"idx": idx, "kind": kind,
                   "writer": q["type"].replace("font_", "").replace(kind, ""),
                   "char": Path(q["target_path"]).stem,
                   "ref_path": ref_path, "query_path": q["target_path"]}
            tgt = target.double().numpy()
            panels = [to_rgb_u8(tgt[:, :hp, :]), to_rgb_u8(tgt[:, hp:, :])]
            for (lam, _), model in zip(pairs, models):
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
                print(f"[fusion_trained] {idx + 1}/{len(dataset)} samples done", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pair", nargs=2, action="append", required=True, metavar=("LAMBDA", "CKPT"),
                        help="训练时的 λ 与对应 ckpt，可重复；必须含 λ=0.5")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--val_json_path", nargs="+", required=True)
    parser.add_argument("--semantic_mask_dir", default=None)
    parser.add_argument("--fixed_pair_path", required=True)
    parser.add_argument("--calibration_json", required=True,
                        help="calibrate_jieti.py 的输出，取 coef 作三项系数（同 eval_s_baseline）")
    parser.add_argument("--s_baseline_json", required=True, help="S 的分母")
    parser.add_argument("--expect_lambda05", type=float, nargs=3, required=True,
                        metavar=("L1_JT", "L1_BF", "J"), help="λ=0.5 的对账值（工具口径），只记录不报错")
    parser.add_argument("--expect_rtol", type=float, default=1e-3)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--font_path", default=fs.DEFAULT_FONT, help="列头/行标签用的 TTF（需含 λ）")
    parser.add_argument("--rows_per_page", type=int, default=15)
    parser.add_argument("--grid_cell", type=int, default=256, help="拼图里每个面板的边长（像素）")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # 加载模型之前就检查参数、输入文件与输出目录
    out = Path(args.output_dir)
    sbs.check_output_free(out)
    pairs = parse_pairs(args.pair)
    lambdas = [lam for lam, _ in pairs]
    for p in [args.font_path, args.calibration_json, args.s_baseline_json] + [c for _, c in pairs]:
        if not Path(p).is_file():
            raise FileNotFoundError(p)
    s_base = read_s_baseline(args.s_baseline_json)

    per_sample = []
    meta = {"pairs": [{"lambda": fs.lam_key(lam), "ckpt": c} for lam, c in pairs],
            "s_baseline_json": args.s_baseline_json, "calibration_json": args.calibration_json,
            "lambda_mode": "训练时改 λ（每个 λ 单独训练的 ckpt）"}
    finalize = fs.make_finalizer(per_sample, lambdas, s_base, args.expect_lambda05,
                                 args.expect_rtol, meta, strict=False)
    recs, pages = sbs.write_outputs(
        run_models(args, pairs, per_sample), out, column_heads(lambdas),
        args.rows_per_page, args.grid_cell, font_path=args.font_path, before_rename=finalize)
    n = {k: sum(r["kind"] == k for r in recs) for k in ("JT", "BF")}
    print(f"[fusion_trained] done: {len(recs)} items (JT {n['JT']} / BF {n['BF']}), "
          f"pages JT {len(pages['JT'])} / BF {len(pages['BF'])} -> {out}", flush=True)


if __name__ == "__main__":
    main()
