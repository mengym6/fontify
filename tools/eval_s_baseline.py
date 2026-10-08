"""在固定 val 上评测 baseline，产出排序指标 S 的分母 s_baseline.json（步骤 3）。

输出 {"L1_JT", "L1_BF", "J"}：不加权重建 L1 分 JT/BF，结体三项之和 J（用标定出的
三项系数，与训练一致）。下半整字全遮、跳过无候选样本（固定 val 已过滤）。
只做评测，不训练、不改训练文件。
"""

import argparse
import json
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


def build_val_transform():
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
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    checkpoint = ckpt["model"]
    state_dict = model.state_dict()
    for key in ("decoder_embed.weight", "decoder_embed.bias", "mask_token"):
        if key in checkpoint and checkpoint[key].shape != state_dict[key].shape:
            del checkpoint[key]
    model.load_state_dict(checkpoint, strict=False)
    # 融合权重 λ 用训练时的值（随 ckpt['args'] 保存）；旧 ckpt 没有这个字段，按原式 0.5
    model.fusion_lambda = getattr(ckpt.get("args"), "fusion_lambda", 0.5)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--val_json_path", nargs="+", required=True)
    parser.add_argument("--semantic_mask_dir", default=None)
    parser.add_argument("--fixed_pair_path", required=True)
    parser.add_argument("--calibration_json", required=True,
                        help="calibrate_jieti.py 的输出，取 coef 作三项系数")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", default="models/jieti_search/s_baseline.json")
    args = parser.parse_args()

    coef = json.loads(Path(args.calibration_json).read_text())["coef"]
    dataset = PairDataset(
        args.data_path, args.val_json_path, transform=build_val_transform(),
        masked_position_generator=MaskingGenerator((56, 28), num_masking_patches=784, max_num_patches=392),
        use_two_pairs=True, half_mask_ratio=1.0,
        semantic_mask_dir=args.semantic_mask_dir,
        num_mask_annotations_jt=1, num_mask_annotations_bf=11,
        mask_coverage_threshold=0.1, return_jieti=True,
        fixed_pair_path=args.fixed_pair_path,
    )
    model = load_model(args.checkpoint).to(args.device).eval()
    jmod = JietiLoss(w_centroid=coef["centroid"], w_logsigma=coef["logsigma"],
                     w_shape=coef["shape"]).to(args.device)

    l1_jt, l1_bf, j_vals = [], [], []
    with torch.no_grad():
        for idx in range(len(dataset)):
            image, target, mask, valid, voro, valid_parts, is_jt = dataset[idx]
            images = image.unsqueeze(0).to(args.device)
            targets = target.unsqueeze(0).to(args.device)
            bmp = torch.from_numpy(mask).bool().flatten()[None].to(args.device)
            voro = voro.unsqueeze(0).to(args.device)
            valid_parts = valid_parts.unsqueeze(0).to(args.device)
            is_jt_b = is_jt.view(1).to(args.device)

            latent = model.forward_encoder(images, targets, bmp)
            pred = model.forward_decoder(latent)
            mask_px = bmp[:, :, None].repeat(1, 1, model.patch_size ** 2 * 3).float()
            mask_px = model.unpatchify(mask_px)
            # 不加权重建 L1（遮盖区）。
            l1 = ((pred - targets).abs() * mask_px).sum() / (mask_px.sum() + 1e-2)
            if bool(is_jt.item()):
                composite = pred * mask_px + targets * (1 - mask_px)
                J, _, _ = jmod(composite, targets, voro, valid_parts, is_jt_b.bool())
                j_vals.append(float(J.item()))
                l1_jt.append(float(l1.item()))
            else:
                l1_bf.append(float(l1.item()))

    result = {
        "L1_JT": float(np.mean(l1_jt)) if l1_jt else float("nan"),
        "L1_BF": float(np.mean(l1_bf)) if l1_bf else float("nan"),
        "J": float(np.mean(j_vals)) if j_vals else float("nan"),
        "n_jt": len(l1_jt), "n_bf": len(l1_bf),
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
