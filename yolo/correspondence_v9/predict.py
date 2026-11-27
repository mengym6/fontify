"""Predict gen correspondence/rejection for supplied GT polygons, never gen labels."""

import argparse, json
from pathlib import Path
import torch
from torch.nn import functional as F
from .data import CorrespondenceDataset
from .model import load_checkpoint
from yolo.pairing.predict import rle, restore_mask


def decode_masks(logits, size):
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite predicted masks")
    if not len(logits):
        return logits.new_zeros((0, size, size), dtype=torch.bool)
    return (
        F.interpolate(
            logits[:, None], (size, size), mode="bilinear", align_corners=False
        )[:, 0].sigmoid()
        >= 0.5
    )


def encode_prediction(output, target, size, threshold=0.5):
    scores = output["match_logits"].detach().sigmoid().cpu().tolist()
    masks = decode_masks(output["mask_logits"].detach(), size).cpu().numpy()
    return [
        {
            "gt_id": target["gt_ids"][i],
            "group": "BF" if target["groups"][i] == 0 else "JT",
            "status": "matched" if score >= threshold else "unmatched",
            "match_confidence": score,
            "gen_mask": rle(restore_mask(masks[i], target["original_sizes"][1], size))
            if score >= threshold
            else None,
        }
        for i, score in enumerate(scores)
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["checkpoint", "manifest", "output"]:
        p.add_argument("--" + key, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threshold", type=float, default=0.5)
    a = p.parse_args()
    if not 0 <= a.threshold <= 1:
        raise ValueError("Invalid threshold")
    ds = CorrespondenceDataset(a.manifest, a.split, a.size)
    m = load_checkpoint(a.checkpoint, a.device).eval()
    rows = []
    with torch.no_grad():
        for x, t in ds:
            out = m(x[None].to(a.device), [t["gt_masks"]], [t["groups"]])[0]
            rows.append(
                {
                    "sample_id": t["sample_id"],
                    "regions": encode_prediction(out, t, a.size, a.threshold),
                }
            )
    Path(a.output).write_text(
        json.dumps(
            {"schema": "calli-gt2gen-v9-predictions", "pairs": rows}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
