#!/usr/bin/env python
"""VGG style loss 的梯度范数标定（在服务器上用 ckpt14 实跑）。

背景：
    原 VGG style loss 对输入做了两次 ImageNet 归一化（bug），修复后 loss_style
    的裸值和梯度都变小。为让 style 的影响力回到修复前水平，按梯度范数标定一个
    全局权重：
        w_style = mean‖∇L_style_修复前‖ / mean‖∇L_style_修复后‖
    baseline 和 T1 共用同一个值。本脚本只测量，不改任何训练文件。

做法（对同一批 batch）：
    1. 修复后：直接调用当前的 vgg_loss(pred_img, target_img)。
    2. 修复前：在调用 vgg 之前，手动对输入再做一次 (x − mean) / std，mean/std 用
       vgg_loss 里的 buffer，等价于复现 bug（不改 vgg_perceptual_loss.py 本身）。
    pred_img / target_img 的来源与 models_train.forward_loss 一致（Resize 到 224）。
    分别对生成器所有 requires_grad 的参数求梯度范数（torch.autograd.grad），
    取 N 个 batch 的均值、中位数、标准差。另报告 recon 的梯度范数与修复前后
    loss_style 的 raw 均值做参照。

输出 json，并打印可直接粘贴的 --style_weight <值>。
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import data.pair_transforms as pair_transforms
import models_train
from data.pairdataset import PairDataset
from util.masking_generator import MaskingGenerator


def build_transform(input_size):
    # 与 finetune_font.sh 的 augmentation_policy=finetune、transform_train 一致。
    normalize = pair_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
    return pair_transforms.Compose([
        pair_transforms.PadToSquare(fill=255),
        pair_transforms.RandomResizedCrop(input_size[1], scale=(0.9999, 1.0), interpolation=3),
        pair_transforms.ToTensor(),
        normalize,
    ])


def load_model(checkpoint_path):
    # 与 finetune_font.sh 一致：vit_base。
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
    parser.add_argument("--input_size", type=int, nargs="+", default=[896, 448])
    parser.add_argument("--num_mask_patches", type=int, default=784)
    parser.add_argument("--max_mask_patches_per_block", type=int, default=392)
    parser.add_argument("--min_mask_patches_per_block", type=int, default=16)
    parser.add_argument("--half_mask_ratio", type=float, default=0.1)
    parser.add_argument("--num_mask_annotations_bf", type=int, default=11)
    parser.add_argument("--num_mask_annotations_jt", type=int, default=1)
    parser.add_argument("--mask_coverage_threshold", type=float, default=0.1)
    parser.add_argument("--no_jt", action="store_true", default=True,
                        help="默认 True，与 finetune_font.sh baseline 配置一致")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", default="models/style_calib/calibration.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    window_size = (args.input_size[0] // 16, args.input_size[1] // 16)
    masked_position_generator = MaskingGenerator(
        window_size, num_masking_patches=args.num_mask_patches,
        max_num_patches=args.max_mask_patches_per_block,
        min_num_patches=args.min_mask_patches_per_block,
    )

    dataset = PairDataset(
        args.data_path, args.json_path, transform=build_transform(args.input_size),
        masked_position_generator=masked_position_generator,
        use_two_pairs=True, half_mask_ratio=args.half_mask_ratio,
        semantic_mask_dir=args.semantic_mask_dir,
        num_mask_annotations_bf=args.num_mask_annotations_bf,
        num_mask_annotations_jt=args.num_mask_annotations_jt,
        mask_coverage_threshold=args.mask_coverage_threshold,
        no_jt=args.no_jt,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    model = load_model(args.checkpoint)
    model.to(args.device).eval()
    # 生成器参数（排除判别器）：style 梯度本就不流向判别器，这里显式排除以对齐"生成器"。
    params = [p for n, p in model.named_parameters()
              if p.requires_grad and not n.startswith("discriminator")]

    vgg_loss = model.vgg_loss
    mean = vgg_loss.mean.to(args.device)
    std = vgg_loss.std.to(args.device)
    transform_vgg = transforms.Compose([transforms.Resize((224, 224))])

    gn_fixed, gn_prefix, gn_recon = [], [], []
    raw_fixed, raw_prefix = [], []

    it = iter(loader)
    for _ in range(args.num_batches):
        try:
            samples, targets, bool_masked_pos, valid = next(it)
        except StopIteration:
            break
        samples = samples.to(args.device).float()
        targets = targets.to(args.device).float()
        bool_masked_pos = bool_masked_pos.flatten(1).to(args.device).bool()
        valid = valid.to(args.device).float()

        latent = model.forward_encoder(samples, targets, bool_masked_pos)
        pred = model.forward_decoder(latent)  # [N, 3, H, W]

        # pred_img / target_img 与 forward_loss 一致：Resize 到 224，float。
        with torch.cuda.amp.autocast(enabled=False):
            pred_img = transform_vgg(pred).float()
            target_img = transform_vgg(targets).float()

            # a. 修复后：当前实现。
            loss_fixed = vgg_loss(pred_img, target_img)
            # b. 修复前：手动再做一次 ImageNet 归一化，复现 bug。
            loss_prefix = vgg_loss((pred_img - mean) / std, (target_img - mean) / std)

        gn_fixed.append(grad_norm(loss_fixed, params))
        gn_prefix.append(grad_norm(loss_prefix, params))
        raw_fixed.append(float(loss_fixed.detach().item()))
        raw_prefix.append(float(loss_prefix.detach().item()))

        # recon 参照：与 forward_loss 的 smoothl1 遮盖区重建一致。
        mask_px = bool_masked_pos[:, :, None].repeat(1, 1, model.patch_size ** 2 * 3).float()
        mask_px = model.unpatchify(mask_px)
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=args.device)[None, :, None, None]
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=args.device)[None, :, None, None]
        inds_ign = ((targets * imagenet_std + imagenet_mean) * (1 - mask_px)).sum((1, 2, 3)) < 100 * 3
        valid_r = valid.clone()
        if inds_ign.sum() > 0:
            valid_r[inds_ign] = 0.
        mask_r = mask_px * valid_r
        recon_el = torch.nn.functional.smooth_l1_loss(pred, targets, reduction="none", beta=0.01)
        recon = (recon_el * mask_r).sum() / (mask_r.sum() + 1e-2)
        gn_recon.append(grad_norm(recon, params))

        del latent, pred, pred_img, target_img, loss_fixed, loss_prefix, recon
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    def stats(v):
        a = np.asarray(v, dtype=np.float64)
        return {"mean": float(a.mean()), "median": float(np.median(a)),
                "std": float(a.std()), "n": int(a.size)} if a.size else {}

    s_fixed, s_prefix, s_recon = stats(gn_fixed), stats(gn_prefix), stats(gn_recon)
    w_style = (s_prefix["mean"] / s_fixed["mean"]) if s_fixed.get("mean") else float("nan")
    raw_ratio = (float(np.mean(raw_prefix)) / float(np.mean(raw_fixed))) if raw_fixed and np.mean(raw_fixed) else float("nan")

    result = {
        "w_style": w_style,                       # 填 --style_weight
        "grad_norm_prefix": s_prefix,             # ‖∇L_style_修复前‖
        "grad_norm_fixed": s_fixed,               # ‖∇L_style_修复后‖
        "grad_norm_recon": s_recon,               # recon 参照
        "raw_loss_style_prefix_mean": float(np.mean(raw_prefix)) if raw_prefix else None,
        "raw_loss_style_fixed_mean": float(np.mean(raw_fixed)) if raw_fixed else None,
        "raw_loss_style_ratio": raw_ratio,
        "num_batches": s_fixed.get("n", 0),
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n[填入 finetune_font.sh] --style_weight {w_style:.4g}")


if __name__ == "__main__":
    main()
