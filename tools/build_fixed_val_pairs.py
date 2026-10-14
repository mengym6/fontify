#!/usr/bin/env python
"""为固定 val 预抽 pair2（U4）：所有搜索组与 baseline 共用同一份配对。

规则与训练一致（Q3）：JT 走"同书家、同结构组合、异字"配对，无候选者跳过；
BF 走随机同 type 配对。用固定 seed 保证可复现。额外标出 4 张视觉图样本
（固定 2 条 JT + 2 条 BF）。

输出 json：{ target_path: pair2_index }，并在同目录写 _vis_samples.json 列出
4 条视觉样本的 target_path。
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.pairdataset import PairDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--val_json_path", nargs="+", required=True)
    parser.add_argument("--semantic_mask_dir", default=None)
    parser.add_argument("--k_max", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    # 只需配对索引，不做图像变换；return_jieti=True 以建好同结构配对池。
    ds = PairDataset(
        args.data_path, args.val_json_path,
        transform=None, use_two_pairs=True,
        semantic_mask_dir=args.semantic_mask_dir,
        return_jieti=True, jieti_k_max=args.k_max,
    )

    # 值存 pair2 的完整记录（dict），使数据集自包含，不依赖易变的 list 索引，
    # 也不要求参考样本本身留在过滤后的 val 里。
    fixed = {}
    jt_vis, bf_vis = [], []
    for i, pair in enumerate(ds.pairs):
        ptype = pair.get("type", "")
        tpath = pair["target_path"]
        if "JT" in ptype:
            if i in getattr(ds, "_jt_struct_pool", {}):
                j = random.choice(ds._jt_struct_pool[i])
                fixed[tpath] = ds.pairs[j]
                if len(jt_vis) < 2:
                    jt_vis.append(tpath)
            # 无候选者跳过（不写入 fixed，训练侧也会剔除）。
        else:
            pool = ds.pair_type_dict.get(ptype, [])
            if pool:
                j = random.choice(pool)
                fixed[tpath] = ds.pairs[j]
                if len(bf_vis) < 2:
                    bf_vis.append(tpath)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)
    vis_path = os.path.splitext(args.output)[0] + "_vis_samples.json"
    with open(vis_path, "w", encoding="utf-8") as f:
        json.dump({"jt": jt_vis, "bf": bf_vis}, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(fixed)} fixed pairs to {args.output}; vis samples -> {vis_path}")


if __name__ == "__main__":
    main()
