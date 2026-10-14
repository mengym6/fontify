#!/usr/bin/env python
"""结体 loss 的梯度范数标定（在服务器上用 ckpt14 实跑，步骤 4）。

目的（Q8 + Q10）：
1. 三项相对系数：使 centroid / logsigma / shape 三项的梯度范数（对可训练参数）相当。
   以 shape 为基准，c_i = mean||g_shape|| / mean||g_i||。
2. w0：使 JT 样本上"加权后结体项"的梯度范数约为原 loss 梯度范数的 0.5 倍。
   原 loss 按训练实际口径（用户 2026-10-03 定，方案 1）：
       原 loss_JT = α_jt · (recon + style_weight · style + edge_weight · edge)
   w0 = 0.5 · mean_b ||∇原 loss_JT|| / mean_b ||∇J_合成||，
   J_合成 = c_centroid·L_centroid + c_logsigma·L_logsigma + c_shape·L_shape。
   两个范数都对合成后的 loss 求，不用各项范数的线性和。

实现：每个 batch 对 6 项（centroid、logsigma、shape、recon、style、edge，均未加权）
求梯度，只保存 float64 的 6×6 内积矩阵 G；任意线性组合 cᵀL 的梯度范数 = √(cᵀGc)。
recon/style/edge 直接取 model.forward_loss 的返回值（含 inds_ign 处理），
为拿到未乘 α_jt 的原值，标定时把 jieti_alpha_jt 设为 1.0，α_jt 只在系数向量里乘一次。

对多个 JT batch 求平均。输出 json，并打印可直接填进 finetune_jieti.sh 的值。
本脚本不改任何训练文件，只测量。
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import data.pair_transforms as pair_transforms
import models_train
from data.pairdataset import PairDataset
from util.masking_generator import MaskingGenerator


def build_transform():
    normalize = pair_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
    return pair_transforms.Compose([
        pair_transforms.PadToSquare(fill=255),
        pair_transforms.RandomResizedCrop(448, scale=(0.9999, 1.0), interpolation=3),
        pair_transforms.ToTensor(),
        normalize,
    ])


def load_model(checkpoint_path):
    model = models_train.vit_base_patch16_input896x448_win_dec64_8glb_sl1()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")["model"]
    state_dict = model.state_dict()
    for key in ("decoder_embed.weight", "decoder_embed.bias", "mask_token"):
        if key in checkpoint and checkpoint[key].shape != state_dict[key].shape:
            del checkpoint[key]
    model.load_state_dict(checkpoint, strict=False)
    return model


TERMS = ("centroid", "logsigma", "shape", "recon", "style", "edge")
GRAM_CHUNK = 1 << 22


def accumulate_gram(losses, params):
    """对各项 loss 分别求梯度，返回 float64 的内积矩阵（CPU）。

    逐参数张量把各项梯度拼成 (T, n) 再转 float64 累加 A·Aᵀ，避免整条梯度向量转 float64。
    """
    grads = [torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
             for loss in losses]
    T = len(losses)
    gram = torch.zeros(T, T, dtype=torch.float64, device=params[0].device)
    for j, p in enumerate(params):
        col = [grads[i][j] for i in range(T)]
        if all(x is None for x in col):
            continue  # 不在计算图里的参数（如判别器）梯度恒为 0，对内积无贡献
        flat = [(torch.zeros_like(p) if x is None else x).reshape(-1) for x in col]
        # 分块转 float64：decoder_embed.weight 约 50M 元素，整块 stack+double 会多占数 GB 显存。
        for s in range(0, p.numel(), GRAM_CHUNK):
            a = torch.stack([f[s:s + GRAM_CHUNK] for f in flat]).double()
            gram += a @ a.T
            del a
        del flat, col
    del grads
    return gram.cpu()


def composite_norm(grams, c):
    """每个 batch 求 √(cᵀGc)，再对 batch 取均值。"""
    c = torch.tensor(c, dtype=torch.float64)
    return float(np.mean([torch.sqrt((c @ g @ c).clamp_min(0.0)).item() for g in grams]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--json_path", nargs="+", required=True)
    parser.add_argument("--semantic_mask_dir", default=None)
    parser.add_argument("--num_batches", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", default="models/jieti_search/calibration.json")
    # 以下三个默认值与 finetune_jieti.sh（084d1f4）一致。
    parser.add_argument("--alpha_jt", type=float, default=0.5,
                        help="JT 样本原 loss 的权重 α_jt；默认 0.5 = finetune_jieti.sh 的 JIETI_ALPHA_JT")
    parser.add_argument("--style_weight", type=float, default=14.73,
                        help="style 项权重；默认 14.73 = finetune_jieti.sh 的 --style_weight")
    parser.add_argument("--edge_weight", type=float, default=0.0,
                        help="edge 项权重；默认 0 = finetune_jieti.sh 在 epoch < --edge_warmup_epochs(10) 时的实际值")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset = PairDataset(
        args.data_path, args.json_path, transform=build_transform(),
        masked_position_generator=MaskingGenerator((56, 28), num_masking_patches=784, max_num_patches=392),
        use_two_pairs=True, half_mask_ratio=0.0,
        semantic_mask_dir=args.semantic_mask_dir,
        num_mask_annotations_jt=1, num_mask_annotations_bf=11,
        mask_coverage_threshold=0.1, return_jieti=True,
    )
    # 只取 JT 样本（结体项只在 JT 上有信号）。
    jt_indices = [i for i, p in enumerate(dataset.pairs)
                  if 'JT' in p.get('type', '') and dataset.weights[i] > 0]
    random.shuffle(jt_indices)

    model = load_model(args.checkpoint)
    # 开启结体分支，让 forward_loss 走训练时的 JT 路径并返回三项分项。
    # α_jt 设为 1.0：forward_loss 返回的 recon/style/edge 就是未乘 α_jt 的原值，
    # α_jt 只在下面的系数向量里乘一次。w 只影响 forward_loss 内部的总 loss，这里不用总 loss。
    # 三项系数保持 1.0，取的是 J 未加权的三个分项。
    model.enable_jieti(1.0, 1.0, k_max=4, pool=224, soft_fg="linear", pred_mass_ratio=0.1)
    model.to(args.device).eval()
    params = [p for p in model.parameters() if p.requires_grad]

    grams = []
    cursor = 0
    for _ in range(args.num_batches):
        batch = []
        while len(batch) < args.batch_size and cursor < len(jt_indices):
            batch.append(dataset[jt_indices[cursor]])
            cursor += 1
        if not batch:
            break
        images = torch.stack([b[0] for b in batch]).to(args.device)
        targets = torch.stack([b[1] for b in batch]).to(args.device)
        masks = np.stack([b[2] for b in batch])
        bmp = torch.from_numpy(masks).bool().flatten(1).to(args.device)
        valid = torch.stack([b[3] for b in batch]).to(args.device)
        voro = torch.stack([b[4] for b in batch]).to(args.device)
        valid_parts = torch.stack([b[5] for b in batch]).to(args.device)
        is_jt = torch.stack([b[6] for b in batch]).to(args.device)

        # 与训练同一入口：forward → forward_loss（inds_ign、拼接图、按样本 style/edge）。
        out = model(images, targets, bool_masked_pos=bmp, valid=valid, epoch=0, no_gan=True,
                    voro=voro, valid_parts=valid_parts, is_jt=is_jt)
        jieti_extra = out[8]
        losses = [jieti_extra["centroid"], jieti_extra["logsigma"], jieti_extra["shape"],
                  out[1], out[2], out[3]]  # recon(loss_l1l2)、style(loss_vgg)、edge(loss_edge)
        grams.append(accumulate_gram(losses, params))
        del out, jieti_extra, losses

    G = torch.stack(grams)  # (B,6,6)
    norms = torch.sqrt(torch.diagonal(G, dim1=1, dim2=2).clamp_min(0.0))  # (B,6)
    mean = {n: float(norms[:, i].mean()) for i, n in enumerate(TERMS)}
    # Q8：以 shape 为基准，三项系数使梯度范数相当（先对范数取均值再求比值）。
    base = mean["shape"] if mean["shape"] > 0 else 1.0
    coef = {name: (base / mean[name] if mean[name] > 0 else 1.0)
            for name in ("centroid", "logsigma", "shape")}
    c_j = [coef["centroid"], coef["logsigma"], coef["shape"], 0.0, 0.0, 0.0]

    def orig_vec(alpha, sw, ew):
        # α_jt·(recon + sw·style + ew·edge)：α_jt 与 style_weight 各乘一次。
        return [0.0, 0.0, 0.0, alpha, alpha * sw, alpha * ew]

    # Q10：分母是 J 合成后的梯度范数，分子是训练口径的原 loss 梯度范数。
    j_norm_composite = composite_norm(grams, c_j)
    orig_norm = composite_norm(grams, orig_vec(args.alpha_jt, args.style_weight, args.edge_weight))

    def w0_of(numer):
        return 0.5 * numer / j_norm_composite if j_norm_composite > 0 else 1.0

    w0 = w0_of(orig_norm)
    # 参照值：都由同一组 Gram 矩阵推出，不重新求梯度。
    j_norm_linear = sum(coef[n] * mean[n] for n in ("centroid", "logsigma", "shape"))
    w0_legacy = 0.5 * mean["recon"] / j_norm_linear if j_norm_linear > 0 else 1.0
    w0_edge0p2 = w0_of(composite_norm(grams, orig_vec(args.alpha_jt, args.style_weight, 0.2)))
    w0_alpha1 = w0_of(composite_norm(grams, orig_vec(1.0, args.style_weight, args.edge_weight)))
    # 6×6 平均余弦：逐 batch 求余弦，某项范数为 0 的 batch 记 nan，再取 nanmean。
    denom = norms[:, :, None] * norms[:, None, :]
    cos = torch.where(denom > 0, G / denom.clamp_min(1e-300), torch.full_like(G, float("nan")))
    mean_cos = np.nanmean(cos.numpy(), axis=0)

    result = {
        "mean_grad_norm": mean,
        "coef": coef,          # 填 --jieti_w_centroid/logsigma/shape
        "w0": w0,              # 填 --jieti_w（训练口径原 loss + J 合成范数）
        "num_batches": len(grams),
        "j_norm_linear": j_norm_linear,        # 旧算法分母：Σ c_k·mean_k
        "j_norm_composite": j_norm_composite,  # mean_b √(c_Jᵀ G_b c_J)
        "orig_norm": orig_norm,                # mean_b ||∇ α_jt·(recon + sw·style + ew·edge)||
        "w0_legacy": w0_legacy,                # 旧公式 0.5·mean_recon / j_norm_linear
        "w0_edge0p2": w0_edge0p2,              # edge_weight=0.2 时的 w0
        "w0_alpha1": w0_alpha1,                # α_jt=1（Q10 字面口径）时的 w0
        "mean_cosine": {"terms": list(TERMS),
                        "matrix": [[None if np.isnan(v) else float(v) for v in row]
                                   for row in mean_cos]},
        "config": {"alpha_jt": args.alpha_jt, "style_weight": args.style_weight,
                   "edge_weight": args.edge_weight, "checkpoint": args.checkpoint,
                   "seed": args.seed, "batch_size": args.batch_size},
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\n[填入 finetune_jieti.sh] --jieti_w_centroid {coef['centroid']:.4g} "
          f"--jieti_w_logsigma {coef['logsigma']:.4g} --jieti_w_shape {coef['shape']:.4g} "
          f"--jieti_w {w0:.4g}")


if __name__ == "__main__":
    main()
