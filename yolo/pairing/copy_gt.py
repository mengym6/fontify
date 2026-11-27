"""Explicitly copy GT region geometry to the generated side as weak targets."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from PIL import Image


def copy_gt(manifest, output):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    data = json.loads(manifest.read_text())
    if data.get("schema") != "calli-paired-regions-v1":
        raise ValueError("Unknown paired-region schema")
    result = deepcopy(data)
    shapes = Counter()
    regions = 0
    for row in result["pairs"]:
        sizes = []
        for key in ["image_a", "image_b"]:
            p = Path(row[key])
            if not p.is_absolute():
                p = manifest.parent / p
            row[key] = str(p.resolve())
            with Image.open(p) as image:
                sizes.append(image.size)
        (wa, ha), (wb, hb) = sizes
        if wa * hb != wb * ha:
            raise ValueError(
                "Different aspect ratios require an explicit geometric mapping: "
                + row["id"]
            )
        shapes[str(sizes)] += 1
        for region in row["regions"]:
            if region["b"]["present"] is not None:
                raise ValueError(
                    "Refusing to overwrite existing generated-side supervision"
                )
            if region["a"]["present"] is not True:
                raise ValueError("Copy mode requires known-present GT regions")
            region["b"] = deepcopy(region["a"])
            region["b"]["source"] = "pseudo"
            region["b"]["method"] = "copy_gt_normalized_coordinates"
            regions += 1
        # Propagate the known GT tasks only, never an unavailable BF/JT task.
        row["complete"][1] = list(row["complete"][0])
    result["supervision_status"] = (
        "WEAK copied GT targets; not verified generated-region correspondences"
    )
    result["copied_gt"] = {
        "source_manifest": str(manifest),
        "source_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "assumptions": "same normalized canvas, same region existence/count; no registration or fine-tag loss",
        "validation_caveat": "Agreement with copied validation masks is not independent correspondence accuracy",
    }
    summary = {
        "paired": len(result["pairs"]),
        "copied_regions": regions,
        "split_counts": dict(Counter(r["split"] for r in result["pairs"])),
        "image_size_pairs": dict(shapes),
        "source": "pseudo",
        "training_requires_allow_pseudo": True,
        "human_verified_correspondences": 0,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "pairs.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(copy_gt(args.manifest, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
