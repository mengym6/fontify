"""Reference agreement; missing rejection support is undefined, never a pass."""

import argparse, json
from pathlib import Path
import torch
from .predict import decode_masks
from .data import CorrespondenceDataset
from .model import load_checkpoint
from .metrics import evaluate_sample, summarize


def evaluate(model, dataset, device="cpu", threshold=0.5, iou_threshold=0.5):
    rows = []
    model.eval()
    with torch.no_grad():
        for x, t in dataset:
            o = model(x[None].to(device), [t["gt_masks"]], [t["groups"]])[0]
            gt = t["gen_masks"].numpy() >= 0.5
            rr = evaluate_sample(
                o["match_logits"].sigmoid().cpu().numpy(),
                decode_masks(o["mask_logits"], dataset.size).cpu().numpy(),
                gt,
                t["matchable"].numpy(),
                t["groups"].numpy(),
                threshold,
                iou_threshold,
            )
            for r in rr:
                r["sample_id"] = t["sample_id"]
                r["gt_id"] = t["gt_ids"][r["index"]]
            rows.extend(rr)
    report = summarize(rows)
    return {
        "reference_type": dataset.data.get("reference_type", "unspecified"),
        "interpretation": "agreement with reference labels, not independently verified real correspondence accuracy",
        "threshold": threshold,
        "iou_threshold": iou_threshold,
        "groups": report,
        "rejection_evaluation_available": report["all"]["unmatchable_support"] > 0,
        "per_region": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["checkpoint", "manifest", "output"]:
        p.add_argument("--" + key, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--iou", type=float, default=0.5)
    a = p.parse_args()
    ds = CorrespondenceDataset(a.manifest, a.split, a.size)
    model = load_checkpoint(a.checkpoint, a.device)
    report = evaluate(model, ds, a.device, a.threshold, a.iou)
    Path(a.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    )
    print(json.dumps(report["groups"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
