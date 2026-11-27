"""Predict corresponding region pairs in two images; export shared query IDs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from .data import letterbox
from .model import load_pair_checkpoint


def relation_type(present_a, present_b):
    """Describe generated-side deviations relative to Calli GT."""
    if present_a and present_b:
        return "corresponding"
    if present_b:
        return "extra_in_generated"
    if present_a:
        return "missing_in_generated"
    return "background"


def rle(mask):
    flat = mask.reshape(-1).astype(np.uint8)
    counts = []
    last = 0
    length = 0
    for value in flat:
        if value == last:
            length += 1
        else:
            counts.append(length)
            length = 1
            last = int(value)
    counts.append(length)
    return {"size": list(mask.shape), "order": "C", "counts": counts}


def restore_mask(mask, original, size):
    w, h = original
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    x, y = (size - nw) // 2, (size - nh) // 2
    image = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (size, size), Image.Resampling.NEAREST
    )
    return (
        np.asarray(
            image.crop((x, y, x + nw, y + nh)).resize((w, h), Image.Resampling.NEAREST)
        )
        > 0
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["checkpoint", "image-a", "image-b", "output"]:
        p.add_argument("--" + key, required=True)
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threshold", type=float, default=0.5)
    a = p.parse_args()
    if a.size % 32 or not 0 <= a.threshold <= 1:
        raise ValueError("Invalid size/threshold")
    images, sizes = [], []
    for path in [a.image_a, a.image_b]:
        with Image.open(path) as im:
            sizes.append(im.size)
            images.append(
                torch.from_numpy(
                    np.array(letterbox(im.convert("RGB"), a.size), copy=True)
                )
                .permute(2, 0, 1)
                .float()
                / 255
            )
    model = load_pair_checkpoint(a.checkpoint, a.device).eval()
    with torch.no_grad():
        out = model(torch.stack(images)[None].to(a.device))
    scores = out["presence"][0].sigmoid().cpu()
    masks = out["masks"][0].sigmoid().cpu().numpy() >= 0.5
    records = []
    for q, score in enumerate(scores):
        if score.max() < a.threshold:
            continue
        row = {"pair_id": q, "group": "BF" if out["groups"][q] == 0 else "JT"}
        for i, side in enumerate(["a", "b"]):
            present = bool(score[i] >= a.threshold)
            row[side] = {
                "present": present,
                "confidence": float(score[i]),
                "mask": rle(restore_mask(masks[q, i], sizes[i], a.size))
                if present or (i == 1 and bool(score[0] >= a.threshold))
                else None,
                "mask_role": "observed_region"
                if present
                else (
                    "expected_missing_region"
                    if i == 1 and bool(score[0] >= a.threshold)
                    else None
                ),
            }
        row["relation"] = relation_type(row["a"]["present"], row["b"]["present"])
        row["is_error"] = row["relation"] == "extra_in_generated"
        row["is_missing"] = row["relation"] == "missing_in_generated"
        records.append(row)
    Path(a.output).write_text(
        json.dumps(
            {
                "schema": "calli-pair-predictions-v1",
                "image_a": a.image_a,
                "image_b": a.image_b,
                "pairs": records,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
