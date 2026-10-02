#!/usr/bin/env python
"""结体 loss 的梯度范数标定（在服务器上用 ckpt14 实跑，步骤 4）。

目的（Q8 + Q10）：
1. 三项相对系数：使 centroid / logsigma / shape 三项的梯度范数（对可训练参数）相当。
   以 shape 为基准，c_i = ||g_shape|| / ||g_i||。
2. w0：使 JT 样本上"加权后结体项"的梯度范数约为原 loss 梯度范数的 0.5 倍。
   w0 = 0.5 · ||g_recon_like|| / ||g_J_weighted||，其中 J_weighted 用标定后的三项系数。

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
from util.jieti_loss import JietiLoss
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


def grad_norm(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    g = [x for x in grads if x is not None]
    if not g:
        return 0.0
    return float(torch.sqrt(sum(x.square().sum() for x in g)).item())


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
    model.to(args.device).eval()
    jmod = JietiLoss().to(args.device)
    params = [p for p in model.parameters() if p.requires_grad]

    acc = {"centroid": [], "logsigma": [], "shape": [], "recon": []}
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
        voro = torch.stack([b[4] for b in batch]).to(args.device)
        valid_parts = torch.stack([b[5] for b in batch]).to(args.device)
        is_jt = torch.stack([b[6] for b in batch]).to(args.device)

        latent = model.forward_encoder(images, targets, bmp)
        pred = model.forward_decoder(latent)
        mask_px = bmp[:, :, None].repeat(1, 1, model.patch_size ** 2 * 3)
        mask_px = model.unpatchify(mask_px)
        composite = pred * mask_px + targets * (1 - mask_px)

        _, parts, _ = jmod(composite, targets, voro, valid_parts, is_jt.bool())
        for name in ("centroid", "logsigma", "shape"):
            acc[name].append(grad_norm(parts[name], params))
        # 原 loss 近似：遮盖区重建 smoothl1（结体项要对标的"原 loss"）。
        recon = torch.nn.functional.smooth_l1_loss(
            pred * mask_px, targets * mask_px, reduction="sum", beta=0.01) / (mask_px.sum() + 1e-2)
        acc["recon"].append(grad_norm(recon, params))

    mean = {k: float(np.mean(v)) if v else 0.0 for k, v in acc.items()}
    # Q8：以 shape 为基准，三项系数使梯度范数相当。
    base = mean["shape"] if mean["shape"] > 0 else 1.0
    coef = {name: (base / mean[name] if mean[name] > 0 else 1.0)
            for name in ("centroid", "logsigma", "shape")}
    # 用标定系数后，J 的梯度范数（范数近似：按系数线性组合，忽略夹角）。
    j_norm = sum(coef[n] * mean[n] for n in ("centroid", "logsigma", "shape"))
    # Q10：w0 使加权结体项梯度范数 ≈ 0.5 × 原 loss。
    w0 = 0.5 * mean["recon"] / j_norm if j_norm > 0 else 1.0

    result = {
        "mean_grad_norm": mean,
        "coef": coef,          # 填 --jieti_w_centroid/logsigma/shape
        "w0": w0,              # 填 --jieti_w
        "num_batches": len(acc["shape"]),
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
