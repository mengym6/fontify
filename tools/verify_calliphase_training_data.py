"""Verify actual DualDataset tensors and sampled masks on the repaired data."""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dual_dataset import MEAN, STD, DualDataset  # noqa: E402
from util.masking_generator import MaskingGenerator  # noqa: E402


def crop_layer(layer, crop):
    top, left, height, width = crop
    image = Image.fromarray(layer.astype(np.uint8))
    return np.asarray(image.crop((left, top, left + width, top + height)).resize(
        (448, 448), Image.Resampling.NEAREST,
    ))


def verify(data, manifest, cache, output, old_root=None):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    dataset = DualDataset(data, [manifest], cache,
                          MaskingGenerator((56, 28), 784, max_num_patches=392))
    audit = dataset._cache_metadata["bf_coordinate_audit"]["records"]
    visual_indices = sorted(
        (index for index, pair in enumerate(dataset.pairs) if pair["has_bf"]),
        key=lambda index: abs(np.log(np.divide(*audit[
            dataset.pairs[index]["target_path"]
        ]["native_size"]))), reverse=True,
    )[:6]
    failures, counts, visuals = [], Counter(), []
    tensor_hash = hashlib.sha256()
    for index, pair in enumerate(dataset.pairs):
        seed = 17000 + index
        sample = dataset.get_pair(index, index, seed)
        crop = sample["geometry_crop"].tolist()
        tensor_hash.update(sample["target"].numpy().tobytes())
        tensor_hash.update(sample["mask"].numpy().tobytes())
        counts[pair["pool_type"]] += 1
        with Image.open(data / pair["target_path"]) as target:
            pixels = np.asarray(target.convert("RGB"))
        top, left, height, width = crop
        expected = np.asarray(Image.fromarray(pixels).crop(
            (left, top, left + width, top + height),
        ).resize((448, 448), Image.Resampling.NEAREST)).astype(np.float32) / 255
        actual = sample["target"].permute(1, 2, 0).numpy()
        actual = actual * np.asarray(STD) + np.asarray(MEAN)
        target_error = float(np.abs(
            actual - np.concatenate((expected, expected))
        ).max())
        if target_error > 1e-6:
            failures.append({"target": pair["target_path"],
                             "target_error": target_error})
        if not pair["has_bf"]:
            continue
        layers = np.load(data / pair["bf_semantic_mask_path"], allow_pickle=False)
        rng = random.Random(seed)
        semantic = []
        for _ in range(2):
            selected = rng.sample(range(len(layers)), min(11, len(layers)))
            union = (layers[selected] > 0).any(0).astype(np.uint8)
            semantic.append(crop_layer(union, crop))
        joined = np.concatenate(semantic)
        expected_mask = joined.reshape(56, 16, 28, 16).mean((1, 3)) > .1
        if not np.array_equal(expected_mask, sample["mask"].numpy()):
            failures.append({"target": pair["target_path"], "mask_mismatch": True})
        if index in visual_indices:
            mask = np.repeat(np.repeat(expected_mask[28:], 16, 0), 16, 1)
            gt = np.asarray(Image.fromarray(pixels).convert("L").crop(
                (left, top, left + width, top + height),
            ).resize((448, 448), Image.Resampling.NEAREST))
            overlay = gt.copy()
            overlay[mask] = 130
            panels, labels = [gt, overlay], ["Repaired GT", "Actual training mask"]
            if old_root and (old_root / pair["target_path"]).is_file():
                with Image.open(old_root / pair["target_path"]) as old:
                    previous = np.asarray(old.convert("L").crop(
                        (left, top, left + width, top + height),
                    ).resize((448, 448), Image.Resampling.NEAREST)).copy()
                previous[mask] = 130
                panels.insert(0, previous)
                labels.insert(0, "Previous stretched GT + same mask")
            canvas = Image.new("RGB", (448 * len(panels), 480), "white")
            draw = ImageDraw.Draw(canvas)
            for column, (panel, label) in enumerate(zip(panels, labels)):
                canvas.paste(Image.fromarray(panel).convert("RGB"), (448 * column, 32))
                draw.text((448 * column + 5, 8), label, fill="black")
            name = f"sample_{len(visuals):02d}.png"
            canvas.save(output / name)
            visuals.append({"file": name, "target": pair["target_path"],
                            "seed": seed, "crop": crop})
    report = {
        "status": "passed" if not failures else "failed",
        "samples": len(dataset), "pool_counts": dict(counts),
        "native_geometry": dataset.native_geometry,
        "source_seal": dataset.source_seal, "failures": failures,
        "actual_target_and_mask_sha256": tensor_hash.hexdigest(),
        "visuals": visuals,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "visuals"}))
    if failures:
        raise RuntimeError("Actual training data alignment verification failed")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-root", type=Path)
    args = parser.parse_args()
    verify(args.data_path, args.manifest, args.cache, args.output, args.old_root)
