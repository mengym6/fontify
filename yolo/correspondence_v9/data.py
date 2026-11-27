"""Stable GT IDs, masks and binary matchability; gen-only errors are not queries."""

import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from yolo.pairing.data import letterbox


def read_data(path):
    data = json.loads(Path(path).read_text())
    if data.get("schema") != "calli-gt2gen-v9-data":
        raise ValueError("Expected v9 data")
    ids = set()
    splits = {}
    for row in data["pairs"]:
        if row["id"] in ids:
            raise ValueError("Duplicate pair ID")
        ids.add(row["id"])
        key = row.get("leak_group", row.get("calli_id", row["id"]))
        splits.setdefault(key, set()).add(row["split"])
        if len(splits[key]) > 1:
            raise ValueError("GT group leaks across splits")
        region_ids = set()
        for g in row["regions"]:
            if g["gt_id"] in region_ids or g["group"] not in ("BF", "JT"):
                raise ValueError("Invalid GT identity/task")
            region_ids.add(g["gt_id"])
            if g["matchable"] is not None and type(g["matchable"]) is not bool:
                raise ValueError("Invalid matchability")
            if not g["gt_polygons"]:
                raise ValueError("GT anchor required")
            if bool(g["gen_polygons"]) != (g["matchable"] is True):
                raise ValueError(
                    "Unmatchable/unknown regions must not have positive gen masks"
                )
            for poly in g["gt_polygons"] + g["gen_polygons"]:
                xy = np.asarray(poly)
                if (
                    xy.ndim != 2
                    or xy.shape[1] != 2
                    or len(xy) < 3
                    or not np.isfinite(xy).all()
                    or (xy < 0).any()
                    or (xy > 1).any()
                ):
                    raise ValueError("Invalid normalized polygons")
    return data


def raster(polygons, original, size):
    w, h = original
    m = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(m)
    for poly in polygons:
        draw.polygon([(int(x * w), int(y * h)) for x, y in poly], fill=255)
    return torch.from_numpy(np.array(letterbox(m, size, True), copy=True)).float() / 255


class CorrespondenceDataset(Dataset):
    def __init__(self, manifest, split="train", size=320):
        self.path = Path(manifest).resolve()
        self.data = read_data(self.path)
        if size % 32:
            raise ValueError("Image size must be divisible by 32")
        self.rows = [r for r in self.data["pairs"] if r["split"] == split]
        self.size = size
        if not self.rows:
            raise ValueError("Empty split")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        images = []
        shapes = []
        for key in ("image_a", "image_b"):
            p = Path(row[key])
            p = p if p.is_absolute() else self.path.parent / p
            with Image.open(p) as im:
                shapes.append(im.size)
                x = np.array(letterbox(im.convert("RGB"), self.size), copy=True)
                images.append(torch.from_numpy(x).permute(2, 0, 1).float() / 255)
        regions = row["regions"]
        n = len(regions)
        gt = (
            torch.stack(
                [raster(g["gt_polygons"], shapes[0], self.size) for g in regions]
            )
            if n
            else torch.zeros(0, self.size, self.size)
        )
        gen = (
            torch.stack(
                [raster(g["gen_polygons"], shapes[1], self.size) for g in regions]
            )
            if n
            else gt.clone()
        )
        return torch.stack(images), {
            "gt_masks": gt,
            "gen_masks": gen,
            "groups": torch.tensor(
                [0 if g["group"] == "BF" else 1 for g in regions], dtype=torch.long
            ),
            "matchable": torch.tensor(
                [-1 if g["matchable"] is None else int(g["matchable"]) for g in regions]
            ),
            "gt_ids": [g["gt_id"] for g in regions],
            "sample_id": row["id"],
            "original_sizes": shapes,
        }


def collate(batch):
    return torch.stack([x for x, t in batch]), [t for x, t in batch]
