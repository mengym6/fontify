"""线性探针：检验冻结 encoder 的特征里是否已经线性编码了 JT/BF 语义区域。

输入与推理一致（下半 query 全遮，只读电脑字内容 + 上半风格字），取抽头
[2,5,8,11] 下半 28×28 token，只训练一个 1×1 线性层预测 70 类 patch 区域
（BF 50 + JT 20），在 val 上报每类 IoU 与 soft Dice。

对照：
- prior：只学每类每位置的常数 logit，衡量"位置先验"能拿到多少；
- 随机初始化 ViT：区分"学到的"与"结构本来就有的"；
- visible：不遮盖 query 目标，墨迹可见，作为上界。

用法（仓库根目录）：
    python tools/label_probe.py \
        --train_json fontdata_example/train_json_new/*.json \
        --val_json fontdata_example/val_json_new/*.json \
        --model best_ep45=models/finetune_no_gan_nojt_no_freeze_baseline/checkpoint-45.pth \
        --model random=random --output_dir outputs/label_probe
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util.calli_eval import (  # noqa: E402
    QueryDataset, build_model, encode, fixed_references, read_records,
)
from util.calli_labels import (  # noqa: E402
    NUM_BF, NUM_LABELS, class_names, label_loss, task_channel_mask,
)

TAPS = {"all": slice(None), "z2": slice(0, 768), "z5": slice(768, 1536),
        "z8": slice(1536, 2304), "z11": slice(2304, 3072)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data_path", default="fontdata_example")
    p.add_argument("--train_json", nargs="+", required=True)
    p.add_argument("--val_json", nargs="+", required=True)
    p.add_argument("--model", action="append", required=True,
                   help="name=checkpoint；checkpoint 写 random 表示随机初始化")
    p.add_argument("--visible", nargs="*", default=[],
                   help="这些 name 不遮盖 query 目标（墨迹可见上界）")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--target_threshold", type=float, default=0.1,
                   help="patch 覆盖率超过该值记为正样本，与训练 mask_coverage_threshold 一致")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


class LinearProbe(nn.Module):
    def __init__(self, channels, prior=False):
        super().__init__()
        self.channels = channels
        if prior:
            self.logit = nn.Parameter(torch.full((1, NUM_LABELS, 28, 28), -4.0))
        else:
            dim = len(range(3072)[channels])
            self.linear = nn.Linear(dim, NUM_LABELS)
            nn.init.constant_(self.linear.bias, -4.0)

    def forward(self, feats):
        if hasattr(self, "logit"):
            return self.logit.expand(feats.shape[0], -1, -1, -1)
        return self.linear(feats[..., self.channels].float()).permute(0, 3, 1, 2)


@torch.no_grad()
def extract(model, dataset, device, visible):
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=8)
    feats, labels, kinds = [], [], []
    for batch in loader:
        latent, _ = encode(model, batch, device, visible=visible)
        x = torch.cat(latent, dim=-1)
        feats.append(x[:, x.shape[1] // 2:].to(torch.float16))
        labels.append(batch["label"].to(device))
        kinds.append(batch["kind"].to(device))
    return torch.cat(feats), torch.cat(labels), torch.cat(kinds)


def train_probes(probes, data, args):
    feats, labels, kinds = data
    n = feats.shape[0]
    params = [p for probe in probes.values() for p in probe.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * math.ceil(n / args.batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator().manual_seed(args.seed)
    for _ in range(args.epochs):
        perm = torch.randperm(n, generator=gen).to(feats.device)
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            f, y, k = feats[idx], labels[idx], kinds[idx]
            # 各探针参数互不共享，损失求和等价于分别训练。
            loss = sum(label_loss(probe(f), y, k) for probe in probes.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()


@torch.no_grad()
def evaluate(probe, data, threshold, batch_size=64):
    feats, labels, kinds = data
    stats = {k: torch.zeros(NUM_LABELS, device=feats.device)
             for k in ("inter", "union", "pos", "soft_inter", "soft_sum")}
    for i in range(0, feats.shape[0], batch_size):
        f, y = feats[i:i + batch_size], labels[i:i + batch_size].float()
        task = task_channel_mask(kinds[i:i + batch_size], f.device)[:, :, None, None]
        prob = probe(f).sigmoid() * task
        y = y * task
        # 探针输出的是 patch 覆盖率估计，预测与 GT 用同一覆盖率阈值二值化。
        pb, tb = (prob > threshold).float(), (y > threshold).float()
        stats["inter"] += (pb * tb).sum((0, 2, 3))
        stats["union"] += ((pb + tb) > 0).float().sum((0, 2, 3))
        stats["pos"] += tb.sum((0, 2, 3))
        stats["soft_inter"] += (prob * y).sum((0, 2, 3))
        stats["soft_sum"] += (prob + y).sum((0, 2, 3))
    stats = {k: v.cpu().numpy() for k, v in stats.items()}
    valid = stats["pos"] > 0
    iou = np.where(valid, stats["inter"] / np.maximum(stats["union"], 1), np.nan)
    dice = np.where(valid, 2 * stats["soft_inter"] / np.maximum(stats["soft_sum"], 1e-6), np.nan)
    bf, jt = slice(0, NUM_BF), slice(NUM_BF, NUM_LABELS)
    summary = {
        "mIoU_BF": float(np.nanmean(iou[bf])), "mIoU_JT": float(np.nanmean(iou[jt])),
        "mDice_BF": float(np.nanmean(dice[bf])), "mDice_JT": float(np.nanmean(dice[jt])),
        "classes_BF": int(valid[bf].sum()), "classes_JT": int(valid[jt].sum()),
    }
    per_class = {
        name: {"iou": float(iou[i]), "dice": float(dice[i]), "pos_patches": int(stats["pos"][i])}
        for i, name in enumerate(class_names()) if valid[i]
    }
    return summary, per_class


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train_records = read_records(args.train_json)
    val_records = read_records(args.val_json)
    train_set = QueryDataset(args.data_path, train_records, fixed_references(train_records, args.seed))
    val_set = QueryDataset(args.data_path, val_records, fixed_references(val_records, args.seed))
    print(f"train={len(train_set)} val={len(val_set)}")

    results = {"config": vars(args), "summary": {}, "per_class": {}}
    rows = []
    for spec in args.model:
        name, ckpt = spec.split("=", 1)
        visible = name in args.visible
        start = time.time()
        torch.manual_seed(args.seed)
        model = build_model(None if ckpt == "random" else ckpt, device)
        train_data = extract(model, train_set, device, visible)
        val_data = extract(model, val_set, device, visible)
        del model
        torch.cuda.empty_cache()

        torch.manual_seed(args.seed)
        probes = {tap: LinearProbe(ch).to(device) for tap, ch in TAPS.items()}
        if not results["summary"]:
            # 位置先验与特征无关，只需在第一个模型上训练一次。
            probes["prior"] = LinearProbe(None, prior=True).to(device)
        train_probes(probes, train_data, args)
        for tap, probe in probes.items():
            key = "prior" if tap == "prior" else f"{name}/{tap}"
            summary, per_class = evaluate(probe, val_data, args.target_threshold)
            train_summary, _ = evaluate(probe, train_data, args.target_threshold)
            summary["train_mIoU_BF"] = train_summary["mIoU_BF"]
            summary["train_mIoU_JT"] = train_summary["mIoU_JT"]
            results["summary"][key] = summary
            results["per_class"][key] = per_class
            rows.append((key, summary))
        del train_data, val_data, probes
        torch.cuda.empty_cache()
        print(f"{name}: {time.time() - start:.0f}s", flush=True)
        with open(out / "probe_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    lines = ["| probe | val mIoU BF | val mIoU JT | val mDice BF | val mDice JT | train mIoU BF | train mIoU JT |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key, s in rows:
        lines.append(f"| {key} | {s['mIoU_BF']:.4f} | {s['mIoU_JT']:.4f} | {s['mDice_BF']:.4f} | "
                     f"{s['mDice_JT']:.4f} | {s['train_mIoU_BF']:.4f} | {s['train_mIoU_JT']:.4f} |")
    table = "\n".join(lines)
    (out / "probe_summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
