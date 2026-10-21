"""Prepare strict Run B manifests and safe fixed pairs without touching images."""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_dual_manifest import merge_records, normalize_record  # noqa: E402
from util.strict_reference import references_distinct  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_lists(directory):
    return [
        row
        for path in sorted(Path(directory).glob("*.json"))
        for row in json.loads(path.read_text())
    ]


def filter_unpairable(records, source_hashes):
    """Exclude isolated same-writer/pool targets without falling back to self."""
    pools = {}
    for row in records:
        pools.setdefault((row["calligrapher"], row["pool_type"]), []).append(row)
    retained, excluded = [], []
    for row in records:
        candidates = pools[(row["calligrapher"], row["pool_type"])]
        destination = (
            retained
            if any(
                references_distinct(row, reference, source_hashes)
                for reference in candidates
            )
            else excluded
        )
        destination.append(row)
    return retained, excluded


def prepare(data, protocol):
    """Keep all held-out identities separate; never fall back to self pairing."""
    from PIL import Image

    from data.pairdataset import PairDataset

    train = load_lists(data / "train_json_strict")
    val = load_lists(data / "val_json_strict")
    if (len(train), len(val)) != (1554, 274):
        raise ValueError("Unexpected strict split population")
    hashes, identity_sets = {}, {"train": {}, "val": {}}
    for split, rows in (("train", train), ("val", val)):
        groups = set()
        original_files, original_pixels, target_pixels = set(), set(), set()
        for row in rows:
            pair = normalize_record(row)
            groups.add(
                (
                    pair["calligrapher"],
                    re.sub(r"\d+$", "", Path(pair["target_path"]).stem),
                )
            )
            for name in ("original_path", "target_path"):
                path = pair[name]
                hashes[path] = digest(data / path)
                with Image.open(data / path) as image:
                    image = image.convert("RGB")
                    pixel = hashlib.sha256(
                        str(image.size).encode() + image.tobytes()
                    ).hexdigest()
                if name == "original_path":
                    original_files.add(hashes[path])
                    original_pixels.add(pixel)
                else:
                    target_pixels.add(pixel)
        identity_sets[split] = {
            "groups": groups,
            "original_file_sha256": original_files,
            "original_pixel_sha256": original_pixels,
            "target_pixel_sha256": target_pixels,
        }
    overlap = {
        name: len(identity_sets["train"][name] & identity_sets["val"][name])
        for name in identity_sets["train"]
    }
    if any(overlap.values()):
        raise ValueError(f"Server strict data leakage: {overlap}")

    merged, summary = merge_records(train, data)
    before = len(merged)
    merged, excluded = filter_unpairable(merged, hashes)
    summary["merged_before_safe_reference_filter"] = before
    summary["merged_records"] = len(merged)
    summary["pools"] = dict(Counter(row["pool_type"] for row in merged))
    summary["excluded_without_safe_same_writer_pool_reference"] = [
        {
            "target_path": row["target_path"],
            "writer": row["calligrapher"],
            "pool": row["pool_type"],
        }
        for row in excluded
    ]
    merged_dir = data / "train_json_dual_strict"
    merged_dir.mkdir(exist_ok=False)
    (merged_dir / "train_dual.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n"
    )
    (merged_dir / "summary.txt").write_text(json.dumps(summary, indent=2) + "\n")

    dataset = PairDataset(
        str(data),
        [str(path) for path in sorted((data / "val_json_strict").glob("*.json"))],
        transform=None,
        use_two_pairs=True,
        return_jieti=True,
        jieti_k_max=4,
    )
    rng, fixed, skipped = random.Random(0), {}, []
    for index, row in enumerate(dataset.pairs):
        candidates = (
            dataset._jt_struct_pool.get(index, [])
            if "JT" in row["type"]
            else dataset.pair_type_dict.get(row["type"], [])
        )
        query = normalize_record(row)
        safe = [
            candidate
            for candidate in candidates
            if references_distinct(
                query, normalize_record(dataset.pairs[candidate]), hashes
            )
        ]
        if not safe:
            skipped.append(row["target_path"])
            continue
        fixed[row["target_path"]] = dataset.pairs[rng.choice(safe)]
    path = data / "val_pairs_fixed_strict.json"
    with path.open("x") as stream:
        json.dump(fixed, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    roles = Counter("JT" if "JT" in row["type"] else "BF" for row in fixed.values())
    if not all(roles[role] for role in ("BF", "JT")):
        raise ValueError("Strict fixed evaluation must cover JT and BF")
    report = {
        "train_records": len(train),
        "val_records": len(val),
        "train_merged": summary,
        "server_cross_split_overlap": overlap,
        "fixed_val_pairs": len(fixed),
        "fixed_val_roles": dict(roles),
        "skipped_fixed_queries": sorted(
            set(skipped)
            | {row["target_path"] for row in val if row["target_path"] not in fixed}
        ),
        "fixed_pair_sha256": digest(path),
        "strict_json_sha256": {
            str(path.relative_to(data)): digest(path)
            for path in sorted(data.glob("*_json_strict/*.json"))
        },
        "reference_rule": "same writer/pool, distinct original and target SHA256",
    }
    (protocol / "data_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data, args.protocol)


if __name__ == "__main__":
    main()
