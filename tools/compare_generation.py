"""生成质量对照：同一批 val 样本、同一固定参考字，比较多个 checkpoint 的生成结果。

输入与推理一致（下半 query 全遮）。输出：
- <output_dir>/<name>/<书家类型>/<字>.png：各模型生成的 448×448 字图；
- <output_dir>/grid/<书家类型>/<字>.png：横向对比图
  [参考书法字 | GT | 模型1 | 模型2 | ...]，供人工评价；
- metrics.json / metrics_summary.md：L1 与 util/font_metrics 结构指标均值。

用法：
    python tools/compare_generation.py \
        --val_json fontdata_example/val_json_new/*.json \
        --model baseline=models/finetune_no_gan_nojt_no_freeze_baseline/checkpoint-45.pth \
        --model label_head=models/finetune_label_head_no_gan_nojt_no_freeze/checkpoint-45.pth \
        --output_dir outputs/compare_generation
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from util.calli_eval import (  # noqa: E402
    QueryDataset, build_model, char_of, encode, fixed_references, read_records,
)
from util.font_metrics import metrics  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data_path", default="fontdata_example")
    p.add_argument("--val_json", nargs="+", required=True)
    p.add_argument("--model", action="append", required=True, help="name=checkpoint")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def to_image(tensor):
    """归一化空间 (3,H,W) → uint8 PIL。"""
    rgb = (tensor.float().cpu() * STD + MEAN).clamp(0, 1)
    return Image.fromarray((rgb.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def main():
    args = parse_args()
    out = Path(args.output_dir)
    device = torch.device(args.device)
    records = read_records(args.val_json)
    dataset = QueryDataset(args.data_path, records, fixed_references(records, args.seed),
                           with_labels=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=8)
    names = [spec.split("=", 1)[0] for spec in args.model]

    scores = {name: [] for name in names}
    for spec in args.model:
        name, ckpt = spec.split("=", 1)
        model = build_model(ckpt, device)
        for batch in loader:
            _, pred = encode(model, batch, device)
            for b, index in enumerate(batch["index"].tolist()):
                record = records[index]
                query_pred = pred[b, :, 448:].float().cpu()
                query_gt = batch["supervision"][b, :, 448:]
                row = metrics(query_pred[None], query_gt[None])
                row["l1"] = (to_tensor01(query_pred) - to_tensor01(query_gt)).abs().mean().item()
                row["type"], row["char"] = record["type"], char_of(record)
                scores[name].append(row)
                path = out / name / record["type"] / f"{char_of(record)}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                to_image(query_pred).save(path)
        del model
        torch.cuda.empty_cache()
        print(f"{name}: generated {len(scores[name])}", flush=True)

    # 对比图：参考书法字 | GT | 各模型生成
    for index, record in enumerate(records):
        ref = records[dataset.refs[index]]
        tiles = [Image.open(Path(args.data_path) / ref["target_path"]).convert("RGB"),
                 Image.open(Path(args.data_path) / record["target_path"]).convert("RGB")]
        tiles += [Image.open(out / n / record["type"] / f"{char_of(record)}.png") for n in names]
        tiles = [t.resize((224, 224), Image.Resampling.BICUBIC) for t in tiles]
        grid = Image.new("RGB", (224 * len(tiles), 244), "white")
        draw = ImageDraw.Draw(grid)
        for i, (tile, label) in enumerate(zip(tiles, ["ref", "GT"] + names)):
            grid.paste(tile, (224 * i, 20))
            draw.text((224 * i + 4, 4), label, fill=(255, 0, 0))
        path = out / "grid" / record["type"] / f"{char_of(record)}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(path)

    keys = ["l1", "centroid", "row", "col", "area", "aspect", "edge_f1", "highpass", "gradient"]
    summary = {}
    for name in names:
        summary[name] = {}
        for subset in ("all", "BF", "JT"):
            rows = [r for r in scores[name] if subset == "all" or r["type"].endswith(subset)]
            summary[name][subset] = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_image": scores}, f, ensure_ascii=False, indent=2)
    lines = ["| model | subset | " + " | ".join(keys) + " |", "|---|---|" + "---:|" * len(keys)]
    for name in names:
        for subset, row in summary[name].items():
            lines.append(f"| {name} | {subset} | " + " | ".join(f"{row[k]:.4f}" for k in keys) + " |")
    table = "\n".join(lines)
    (out / "metrics_summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"对比图目录：{out / 'grid'}（列顺序：ref | GT | {' | '.join(names)}）")


def to_tensor01(tensor):
    return (tensor.float() * STD + MEAN).clamp(0, 1)


if __name__ == "__main__":
    main()
