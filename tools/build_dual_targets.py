"""Build fresh raw 448 JT instance caches for train and fixed clean validation."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_dual_manifest import normalize_record
from util.dual_targets import (
    CACHE_SCHEMA,
    DEFINITION,
    SOURCE_SEAL,
    InstanceRenderer,
    build_instance_targets,
    cache_key,
    coordinate_audit,
    load_cached_targets,
    record_source_paths,
    sha256_file,
)


def audit_cache_partitions(cache_dir, output_path):
    """Report NN ink loss separately from overlap-ownership/EDT losses."""
    import numpy as np

    cache_dir, output_path = Path(cache_dir), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Partition audit already exists: {output_path}")
    index = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    valid_count, downsample_count, partition_count = 0, 0, 0
    lost = []
    for entry in index.values():
        pair = {
            "target_path": entry["target_path"],
            "jt_target_path": entry["jt_target_path"],
            "jt_annotation_path": entry["annotation_path"],
        }
        raw = load_cached_targets(cache_dir, pair, index=index, metadata=metadata)
        targets = build_instance_targets(raw["regions"], raw["text"], raw["classes"])
        valid_count += int(targets["instance_valid"].sum())
        downsample_count += int(targets["j_downsample_lost"].sum())
        partition_count += int(targets["j_partition_lost"].sum())
        for slot in np.where(
            targets["j_downsample_lost"] | targets["j_partition_lost"]
        )[0]:
            ink = (raw["regions"][slot] > 0) & (raw["text"] > 0)
            lost.append(
                {
                    "target_path": entry["target_path"],
                    "slot": int(slot),
                    "annotation_id": entry["annotation_ids"][slot],
                    "class_name": entry["class_names"][raw["classes"][slot]],
                    "ink_448": int(ink.sum()),
                    "ink_nearest_224": int(ink[::2, ::2].sum()),
                    "reason": (
                        "nearest-downsample-no-ink"
                        if targets["j_downsample_lost"][slot]
                        else "overlap-ownership-or-EDT-no-partition"
                    ),
                }
            )
    audit = {
        "schema": "dual-partition-audit-v1",
        "definition": DEFINITION,
        "geometry": "identity-448",
        "downsample": "nearest-224",
        "overlap_ownership": "ink-label-map-later-instance-wins",
        "distance": "exact-squared-euclidean",
        "cache_index_sha256": sha256_file(cache_dir / "index.json"),
        "cached_targets": len(index),
        "valid_448_instances": valid_count,
        "j_downsample_lost": downsample_count,
        "j_partition_lost": partition_count,
        "records": lost,
        "supervision": "H-valid-unchanged;J-shielded-by-GT-mass",
    }
    output_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def build_cache(root, records, output_dir, input_paths=(), fixed_pair_path=None):
    """Write arrays plus source/cache hashes, reporting any coordinate failure."""
    root, output_dir = Path(root), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Cache output directory must not exist: {output_dir}")
    renderer = InstanceRenderer(root)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "layers").mkdir()
    index, failures, source_inputs = {}, [], []
    source_hashes = {}
    records = list(records)
    if fixed_pair_path:
        fixed = json.loads(Path(fixed_pair_path).read_text(encoding="utf-8"))
        if not isinstance(fixed, dict):
            raise ValueError("Fixed validation pairs must be a mapping")
        records.extend(fixed.values())

    def digest(relative_path):
        if relative_path not in source_hashes:
            source_hashes[relative_path] = sha256_file(root / relative_path)
        return source_hashes[relative_path]

    for row in records:
        pair = row if "has_jt" in row else normalize_record(row)
        try:
            paths = record_source_paths(pair)
            for path in paths.values():
                digest(path)
            source_inputs.append(paths)
            if not pair["has_jt"] or pair["target_path"] in index:
                continue
            regions, text, classes, entry = renderer.render(pair)
            audit = coordinate_audit(
                root, pair, text, entry, source_hashes=source_hashes
            )
            relative_cache = f"layers/{cache_key(pair)}.npz"
            import numpy as np

            np.savez_compressed(
                output_dir / relative_cache, regions=regions, text=text, classes=classes
            )
            entry.update(
                cache_path=relative_cache,
                cache_sha256=sha256_file(output_dir / relative_cache),
                coordinate_audit=audit,
                target_path=pair["target_path"],
                source_sha256={
                    "annotation": digest(entry["annotation_path"]),
                    "original": digest(audit["original_path"]),
                    "target": digest(pair["target_path"]),
                },
            )
            index[pair["target_path"]] = entry
        except (KeyError, OSError, ValueError) as error:
            failures.append({"target_path": pair["target_path"], "error": str(error)})
    index_path = output_dir / "index.json"
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    audits = [p["coordinate_audit"] for p in index.values()]
    metadata = {
        "definition": DEFINITION,
        "schema": CACHE_SCHEMA,
        "source_seal": SOURCE_SEAL,
        "status": "failed" if failures else "complete",
        "size": 448,
        "k_max": 4,
        "occupancy_mode": "any-ink-in-16x16-patch",
        "deduplication": "native-region-IoU-strictly-greater-than-0.9",
        "instance_ink": "transformed-instance-region-intersect-transformed-JT-text",
        "text_source": "JT-COCO-only",
        "cached_targets": len(index),
        "removed_duplicates": sum(len(p["removed_duplicates"]) for p in index.values()),
        "max_instances": max((p["instance_count"] for p in index.values()), default=0),
        "min_gt_ink_coverage": min(
            (p["gt_ink_coverage"] for p in audits), default=None
        ),
        "min_gt_text_iou": min((p["gt_text_iou"] for p in audits), default=None),
        "max_centroid_offset": max(
            (p["normalized_centroid_offset"] for p in audits), default=None
        ),
        "index_sha256": sha256_file(index_path),
        "input_sha256": {str(path): sha256_file(path) for path in input_paths},
        "source_sha256": source_hashes,
        "source_inputs": source_inputs,
        "failures": failures,
    }
    if fixed_pair_path:
        metadata["fixed_pair_sha256"] = sha256_file(fixed_pair_path)
        import hashlib

        metadata["fixed_pair_md5"] = hashlib.md5(
            Path(fixed_pair_path).read_bytes()
        ).hexdigest()
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if failures:
        raise ValueError(
            f"{len(failures)} cache failures; see {output_dir / 'metadata.json'}"
        )
    audit_cache_partitions(output_dir, output_dir / "downsample_audit.json")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", "--data-path", required=True)
    parser.add_argument("--json_paths", "--json-paths", nargs="+", required=True)
    parser.add_argument("--fixed_pair_path", "--fixed-pair-path")
    parser.add_argument("--output_dir", "--output-dir", required=True)
    args = parser.parse_args(argv)
    paths, records = [], []
    for name in args.json_paths:
        path = Path(name)
        children = sorted(path.glob("*.json")) if path.is_dir() else [path]
        for child in children:
            rows = json.loads(child.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError(f"Expected record list: {child}")
            records.extend(rows)
            paths.append(child)
    metadata = build_cache(
        args.data_path, records, args.output_dir, paths, args.fixed_pair_path
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in metadata.items()
                if key not in ("source_sha256", "input_sha256", "source_inputs")
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
