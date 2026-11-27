"""Explicit paired-region supervision: missing annotation is not absent region."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


def read_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get("schema") != "calli-paired-regions-v1":
        raise ValueError("Unknown paired-region schema")
    return data


def validate_pairs(data, require_paired=False, allow_pseudo=False):
    seen, supervised, partitions = set(), 0, {}
    for row in data["pairs"]:
        if row["id"] in seen:
            raise ValueError("Duplicate image-pair id")
        seen.add(row["id"])
        key = row.get("leak_group", row.get("calli_id", row["id"]))
        partitions.setdefault(key, set()).add(row["split"])
        if len(partitions[key]) > 1:
            raise ValueError("Calli identity/group crosses train and validation splits")
        if np.asarray(row["complete"]).shape != (2, 2):
            raise ValueError("complete must be [side A/B, task BF/JT]")
        pair_ids = set()
        for region in row["regions"]:
            if region["pair_id"] in pair_ids or region["group"] not in ("BF", "JT"):
                raise ValueError("Duplicate region identity or invalid task")
            pair_ids.add(region["pair_id"])
            visible = []
            for side in ("a", "b"):
                item = region[side]
                presence = item["present"]
                if item.get("mask_role") == "expected_missing_region" and (
                    presence is not False or not item.get("polygons")
                ):
                    raise ValueError(
                        "Expected missing region needs absent ink and nonempty location polygons"
                    )
                if presence is not None and type(presence) is not bool:
                    raise ValueError("present must be true, false or null")
                if presence is None and item.get("polygons"):
                    raise ValueError("Unknown presence cannot have supervised polygons")
                if presence is not None:
                    if item.get("source") not in ("human", "synthetic", "pseudo"):
                        raise ValueError("Supervision must declare its source")
                    if item["source"] == "pseudo" and not allow_pseudo:
                        raise ValueError("Pseudo labels require explicit permission")
                if presence is True and not item.get("polygons"):
                    raise ValueError("Present region requires polygons")
                if (
                    presence is False
                    and item.get("polygons")
                    and item.get("mask_role") != "expected_missing_region"
                ):
                    raise ValueError("Absent region cannot have polygons")
                for poly in item.get("polygons", []):
                    coords = np.asarray(poly, dtype=float)
                    if (
                        coords.ndim != 2
                        or coords.shape[1] != 2
                        or len(coords) < 3
                        or not np.isfinite(coords).all()
                        or (coords < 0).any()
                        or (coords > 1).any()
                    ):
                        raise ValueError(
                            "Polygons must be finite normalized [N,2] coordinates"
                        )
                visible.append(presence)
            if not any(v is True for v in visible):
                raise ValueError("Pair must contain at least one known-present region")
            supervised += all(v is not None for v in visible) and any(
                v is True for v in visible
            )
    if require_paired and not supervised:
        raise ValueError(
            "Image pairs lack generated-side region correspondence supervision; formal training is disabled"
        )
    return supervised


def letterbox(image, size, mask=False):
    w, h = image.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    x, y = (size - nw) // 2, (size - nh) // 2
    canvas = Image.new(
        "L" if mask else "RGB", (size, size), 0 if mask else (255, 255, 255)
    )
    canvas.paste(
        image.resize(
            (nw, nh), Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
        ),
        (x, y),
    )
    return canvas


class PairDataset(Dataset):
    def __init__(self, manifest, split="train", size=320, allow_pseudo=False):
        self.path = Path(manifest).resolve()
        self.data = read_manifest(self.path)
        validate_pairs(self.data, allow_pseudo=allow_pseudo)
        if size % 32:
            raise ValueError("Size must be divisible by 32")
        self.rows = [r for r in self.data["pairs"] if r["split"] == split]
        self.size = size
        if not self.rows:
            raise ValueError("No pairs in requested split")

    def resolve(self, value):
        p = Path(value)
        return p if p.is_absolute() else self.path.parent / p

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        images, originals = [], []
        for key in ("image_a", "image_b"):
            with Image.open(self.resolve(row[key])) as im:
                originals.append(im.size)
                a = np.array(letterbox(im.convert("RGB"), self.size), copy=True)
                images.append(torch.from_numpy(a).permute(2, 0, 1).float() / 255)
        regions = row["regions"]
        masks = torch.zeros(len(regions), 2, self.size, self.size)
        present = torch.full((len(regions), 2), -1.0)
        mask_supervised = torch.zeros(len(regions), 2, dtype=torch.bool)
        groups = torch.tensor(
            [0 if r["group"] == "BF" else 1 for r in regions], dtype=torch.long
        )
        for i, r in enumerate(regions):
            for side, key in enumerate(("a", "b")):
                item = r[key]
                if item["present"] is None:
                    continue
                present[i, side] = float(item["present"])
                if (
                    not item["present"]
                    and item.get("mask_role") != "expected_missing_region"
                ):
                    continue
                mask_supervised[i, side] = True
                w, h = originals[side]
                mask = Image.new("L", (w, h), 0)
                draw = ImageDraw.Draw(mask)
                for poly in item["polygons"]:
                    draw.polygon([(int(x * w), int(y * h)) for x, y in poly], fill=255)
                masks[i, side] = (
                    torch.from_numpy(
                        np.array(letterbox(mask, self.size, True), copy=True)
                    ).float()
                    / 255
                )
        return torch.stack(images), {
            "masks": masks,
            "present": present,
            "mask_supervised": mask_supervised,
            "groups": groups,
            "complete": torch.tensor(row["complete"], dtype=torch.bool),
            "id": row["id"],
        }


def collate_pairs(batch):
    return torch.stack([x[0] for x in batch]), [x[1] for x in batch]
