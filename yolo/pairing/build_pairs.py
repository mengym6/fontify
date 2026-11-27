"""Bind selected Run B ep49 images to exact Calli identities, retaining old splits."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(generated, index_csv, dataset, output):
    generated, index_csv, dataset, output = map(
        lambda p: Path(p).resolve(), (generated, index_csv, dataset, output)
    )
    manifest = json.loads((dataset / "manifest.json").read_text())
    lookup = {}
    for row in manifest["images"]:
        for source in row["source_records"]:
            key = (
                source["annotation_path"].split("/")[0],
                Path(source["image_file_name"]).name,
            )
            if key in lookup and lookup[key]["id"] != row["id"]:
                raise ValueError("Ambiguous source identity: " + str(key))
            lookup[key] = row
    pairs, excluded, seen = [], [], set()
    for item in csv.DictReader(index_csv.open(encoding="utf-8-sig")):
        name = item["file"]
        image = generated / name
        if not image.is_file():
            excluded.append({"file": name, "reason": "generated image missing"})
            continue
        if name in seen:
            raise ValueError("Duplicate index filename")
        seen.add(name)
        target = Path(item["query_path"])
        key = (target.parent.parent.name, target.name)
        gt = lookup.get(key)
        if gt is None:
            excluded.append(
                {
                    "file": name,
                    "target": str(target),
                    "reason": "Calli identity excluded/not present in YOLO dataset",
                }
            )
            continue
        regions = []
        # Use exported labels: same geometry as the existing YOLO dataset.
        for i, line in enumerate((dataset / gt["label"]).read_text().splitlines()):
            values = line.split()
            if not values:
                continue
            cid = int(values[0])
            xy = list(map(float, values[1:]))
            poly = [xy[j : j + 2] for j in range(0, len(xy), 2)]
            regions.append(
                {
                    "pair_id": gt["id"] + ":" + str(i),
                    "group": "BF" if cid < 50 else "JT",
                    "original_class_id": cid,
                    "a": {"present": True, "source": "human", "polygons": [poly]},
                    "b": {"present": None, "source": "unannotated", "polygons": []},
                }
            )
        pairs.append(
            {
                "id": Path(name).stem,
                "split": gt["split"],
                "leak_group": gt["leak_group"],
                "calli_id": gt["id"],
                "character": gt["character"],
                "author": gt["author"],
                "image_a": str(dataset / gt["image"]),
                "image_b": str(image),
                "image_a_sha256": digest(dataset / gt["image"]),
                "image_b_sha256": digest(image),
                "target_source_path": str(target),
                "reference_source_path": item["ref_path"],
                "complete": [[gt["has_bf"], gt["has_jt"]], [False, False]],
                "regions": regions,
            }
        )
    extra = sorted(p.name for p in generated.glob("*.png") if p.name not in seen)
    if extra:
        raise ValueError(
            "Generated images absent from authoritative index: " + str(extra[:5])
        )
    # The same character/image group must never enter both dataset splits.
    groups = {}
    for p in pairs:
        groups.setdefault(p["leak_group"], set()).add(p["split"])
    if any(len(v) > 1 for v in groups.values()):
        raise ValueError("Leak group crosses splits")
    document = {
        "schema": "calli-paired-regions-v1",
        "generator": "Run B best ep49 (user selected directory)",
        "generator_checkpoint_sha256": None,
        "generated_directory": str(generated),
        "index_csv": str(index_csv),
        "index_sha256": digest(index_csv),
        "calli_split_id": manifest["split_id"],
        "supervision_status": "image pairs and Calli-side labels ready; generated-side correspondence UNANNOTATED",
        "pairs": pairs,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "pairs.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2)
    )
    summary = {
        "paired": len(pairs),
        "excluded": excluded,
        "split_counts": {
            s: sum(p["split"] == s for p in pairs) for s in ["train", "val"]
        },
        "unique_calli": len({p["calli_id"] for p in pairs}),
        "unique_groups": len(groups),
        "generated_supervised_regions": 0,
        "formal_training_ready": False,
        "note": "No pseudo masks and no copied GT polygons on generated side; split inherited, not independent of generator history.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for x in ["generated", "index-csv", "dataset", "output"]:
        p.add_argument("--" + x, required=True)
    a = p.parse_args()
    print(
        json.dumps(
            build(a.generated, a.index_csv, a.dataset, a.output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
