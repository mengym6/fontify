"""Explicit style identities, split auditing and fixed reference evaluation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REQUIRED = {"image_path", "target_path", "type", "style_id", "character", "glyph_id"}


SPLITS = ("train", "val_seen")


def read_manifest(path):
    """Manifest maps train/val_seen to lists of JSON paths."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    unknown = set(manifest) - set(SPLITS)
    if unknown:
        raise ValueError(f"Unknown manifest splits: {sorted(unknown)}")
    rows = {}
    paths = {}
    for split in SPLITS:
        paths[split] = [str((path.parent / p).resolve()) for p in manifest[split]]
        rows[split] = []
        for source in paths[split]:
            rows[split].extend(json.loads(Path(source).read_text()))
        if not rows[split]:
            raise ValueError(f"Empty split: {split}")
    return rows, paths


def audit(rows, root):
    """Fail on leakage; glyph_id must identify the original before cropping."""
    root = Path(root)
    seen_paths, seen_hashes, seen_glyphs = {}, {}, {}
    fingerprint = hashlib.sha256()
    content_hashes = {}
    for split, records in rows.items():
        fingerprint.update(split.encode())
        for row in records:
            missing = REQUIRED - row.keys()
            if missing or any(
                not isinstance(row[k], str) or not row[k].strip() for k in REQUIRED
            ):
                raise ValueError(f"Missing identity fields: {missing}: {row}")
            if row.get("source_dataset", "calliphase") != "calliphase":
                raise ValueError("Stage 3 accepts only CalliPhase")
            for key in ("image_path", "target_path"):
                path = (root / row[key]).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                if str(path) not in content_hashes:
                    content_hashes[str(path)] = hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                fingerprint.update(content_hashes[str(path)].encode())
            fingerprint.update(json.dumps(row, sort_keys=True).encode())
            target = (root / row["target_path"]).resolve()
            digest = content_hashes[str(target)]
            glyph = (row["style_id"], row["glyph_id"])
            for registry, identity in (
                (seen_paths, str(target)),
                (seen_hashes, digest),
                (seen_glyphs, glyph),
            ):
                if identity in registry and registry[identity] != split:
                    raise ValueError(f"Cross-split target leakage: {target}")
                registry[identity] = split
    train_styles = {r["style_id"] for r in rows["train"]}
    if not {r["style_id"] for r in rows["val_seen"]} <= train_styles:
        raise ValueError("val_seen styles must all occur in train")
    train_chars = {r["character"] for r in rows["train"]}
    if train_chars & {r["character"] for r in rows["val_seen"]}:
        raise ValueError("val_seen must contain characters unseen in training")
    for split, records in rows.items():
        for row in records:
            if not any(
                r["style_id"] == row["style_id"]
                and r["character"] != row["character"]
                and (split != "train" or r["type"] == row["type"])
                for r in records
            ):
                raise ValueError(f"No different-character reference: {row}")
    report = {
        key: {"records": len(value), "styles": len({r["style_id"] for r in value})}
        for key, value in rows.items()
    }
    report["image_and_metadata_sha256"] = fingerprint.hexdigest()
    report["semantic_masks_hashed"] = False
    return report


def image_tensor(path):
    """One deterministic preprocessing path shared by all evaluations."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        side = max(width, height)
        padded = Image.new("RGB", (side, side), "white")
        padded.paste(image, ((side - width) // 2, (side - height) // 2))
        image = padded.resize((448, 448), Image.Resampling.BICUBIC)
        tensor = torch.from_numpy(np.array(image).copy()).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor.float() / 255 - mean) / std


def fixed_sample(root, target, reference, blank=False):
    """The lower input target is always white, never the ground truth."""
    root = Path(root)
    source_ref = image_tensor(root / reference["image_path"])
    source_query = image_tensor(root / target["image_path"])
    style_ref = image_tensor(root / reference["target_path"])
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    white = torch.ones_like(style_ref).sub(mean).div(std)
    if blank:
        style_ref = white
    source = torch.cat((source_ref, source_query), dim=1)
    context = torch.cat((style_ref, white), dim=1)
    supervision = torch.cat(
        (style_ref, image_tensor(root / target["target_path"])), dim=1
    )
    mask = torch.zeros(56, 28, dtype=torch.bool)
    mask[28:] = True
    return source, context, supervision, mask


def reference_cases(records, target):
    """Deterministic references independent of training randomness."""
    records = sorted(
        records, key=lambda r: (r["style_id"], r["character"], r["target_path"])
    )
    records = list({(r["style_id"], r["character"]): r for r in records}.values())
    same = [
        r
        for r in records
        if r["style_id"] == target["style_id"] and r["character"] != target["character"]
    ]
    other = [
        r
        for r in records
        if r["style_id"] != target["style_id"] and r["character"] != target["character"]
    ]
    if not same:
        raise ValueError("Missing correct reference")
    cases = [("correct", same[0], False), ("blank", same[0], True)]
    if len(same) > 1:
        cases.append(("same_style", same[1], False))
    if other:
        cases.append(("other_style", other[0], False))
    return cases
