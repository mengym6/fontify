"""Convert v8 reviewed pairs without reusing copied missing masks as matches."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def convert(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    data = json.loads(source.read_text())
    if data.get("schema") != "calli-paired-regions-v1":
        raise ValueError("Expected v8 paired regions")
    if output.exists():
        raise FileExistsError("Use a fresh v9 manifest path")
    rows = []
    dropped = 0
    for old in data["pairs"]:
        row = {
            k: deepcopy(v)
            for k, v in old.items()
            if k not in ("regions", "complete", "review")
        }
        for key in ("image_a", "image_b"):
            path = Path(row[key])
            row[key] = str(path if path.is_absolute() else source.parent / path)
        row["regions"] = []
        for g in old["regions"]:
            if g["a"]["present"] is not True:
                dropped += 1
                continue
            visible = g["b"]["present"]
            unknown = visible is None or (
                g["b"].get("annotation_role") == "expected_region_not_visible_ink"
                and visible is not False
            )
            matchable = None if unknown else bool(visible)
            row["regions"].append(
                {
                    "gt_id": g["pair_id"],
                    "group": g["group"],
                    "gt_polygons": deepcopy(g["a"]["polygons"]),
                    "matchable": matchable,
                    "gen_polygons": deepcopy(g["b"]["polygons"]) if matchable else [],
                    "reference_type": "reviewed_weak",
                    "source_method": g["b"].get("method", ""),
                }
            )
        rows.append(row)
    result = {
        "schema": "calli-gt2gen-v9-data",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reference_type": "reviewed_weak",
        "dropped_generated_only_regions": dropped,
        "pairs": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    summary = {
        s: dict(
            Counter(
                "matchable"
                if g["matchable"] is True
                else "unmatchable"
                if g["matchable"] is False
                else "unknown"
                for r in rows
                if r["split"] == s
                for g in r["regions"]
            )
        )
        for s in ("train", "val")
    }
    return {"pairs": len(rows), "generated_only_dropped": dropped, "splits": summary}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(json.dumps(convert(a.source, a.output), ensure_ascii=False, indent=2))
