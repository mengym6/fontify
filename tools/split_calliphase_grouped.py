"""Split retained CalliPhase records by calligrapher and character identity."""

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def sha256(path):
    """Hash an asset without changing its bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha256(path):
    """Hash dimensions and decoded RGB pixels, including image verification."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.load()
        return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def record_identity(record):
    """Unify numbered writings, preserving traditional and variant characters."""
    folder = record["font_folder"]
    if not folder.endswith(("BF", "JT")):
        raise ValueError(f"Unknown BF/JT folder: {folder}")
    author, role = folder[:-2], folder[-2:]
    character = re.sub(r"\d+$", "", Path(record["target_path"]).stem)
    if len(character) != 1:
        raise ValueError(f"Expected one character after suffix removal: {record}")
    if Path(record["image_path"]).stem != character:
        raise ValueError(f"Source/target character disagreement: {record}")
    return author, character, role


def load_records(train_input, val_input):
    """Combine both existing splits without adding or dropping any records."""
    records, source_hashes = [], {}
    seen = set()
    for split, directory in (("train", train_input), ("val", val_input)):
        paths = sorted(directory.glob("*.json"))
        if not paths:
            raise ValueError(f"No input JSON lists: {directory}")
        for path in paths:
            source_hashes[str(path.resolve())] = sha256(path)
            rows = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError(f"Expected a record list: {path}")
            for row in rows:
                if row["target_path"] in seen:
                    raise ValueError(f"Repeated input path: {row['target_path']}")
                author, character, role = record_identity(row)
                seen.add(row["target_path"])
                records.append(
                    {
                        "record": row,
                        "old_split": split,
                        "author": author,
                        "character": character,
                        "role": role,
                        "group": (author, character),
                    }
                )
    return sorted(records, key=lambda row: row["record"]["target_path"]), source_hashes


def choose_validation(groups, ratio, rng):
    """Use shuffled groups and subset sums to approach author and BF/JT quotas."""
    shuffled = sorted(groups)
    rng.shuffle(shuffled)
    total = sum(len(groups[key]) for key in shuffled)
    bf_total = sum(row["role"] == "BF" for key in shuffled for row in groups[key])
    jt_total = total - bf_total
    target = round(total * ratio)
    cap = min(total - 1, target + max(len(groups[key]) for key in shuffled))
    # Immutable linked paths make later DP updates independent of earlier states.
    states = {(0, 0): None}
    for key in shuffled:
        size = len(groups[key])
        bf_size = sum(row["role"] == "BF" for row in groups[key])
        for (count, bf_count), path in list(states.items()):
            next_state = (count + size, bf_count + bf_size)
            if next_state[0] <= cap and next_state not in states:
                states[next_state] = (key, path)
    viable = [
        (count, bf_count)
        for count, bf_count in states
        if 0 < count < total
        and (not bf_total or 0 < bf_count < bf_total)
        and (not jt_total or 0 < count - bf_count < jt_total)
    ]
    if not viable:
        raise ValueError("Cannot retain author/task coverage in both splits")
    best = min(
        viable,
        key=lambda state: (
            abs(state[0] - total * ratio),
            abs(state[1] - bf_total * ratio)
            + abs(state[0] - state[1] - jt_total * ratio),
            state,
        ),
    )
    chosen = set()
    path = states[best]
    while path is not None:
        key, path = path
        chosen.add(key)
    return chosen


def assign_splits(records, ratio, seed):
    """Split each author independently; cross-author characters may overlap."""
    by_author = defaultdict(lambda: defaultdict(list))
    for row in records:
        by_author[row["author"]][row["group"]].append(row)
    rng = random.Random(seed)
    assignment = {}
    for author in sorted(by_author):
        groups = by_author[author]
        validation = choose_validation(groups, ratio, rng)
        for key in groups:
            assignment[key] = "val" if key in validation else "train"
    return assignment


def inventory_assets(records, root):
    """Verify all referenced assets and original/target image identities."""
    hashes, identities = {}, {}
    for item in records:
        row = item["record"]
        target = Path(row["target_path"])
        original = target.parent.parent / "images" / target.name
        for field in (
            "image_path",
            "target_path",
            "annotation_path",
            "semantic_mask_path",
        ):
            if field in row:
                path = root / row[field]
                if not path.is_file():
                    raise FileNotFoundError(path)
                key = str(path.resolve())
                if key not in hashes:
                    hashes[key] = sha256(path)
        raw = root / original
        if str(raw.resolve()) not in hashes:
            hashes[str(raw.resolve())] = sha256(raw)
        identities[row["target_path"]] = {
            "original_path": original.as_posix(),
            "original_file_sha256": hashes[str(raw.resolve())],
            "original_pixel_sha256": pixel_sha256(raw),
            "target_file_sha256": hashes[str((root / target).resolve())],
            "target_pixel_sha256": pixel_sha256(root / target),
        }
    return hashes, identities


def audit_split(records, assignment, identities):
    """Reject group leakage and exact original/target image overlap."""
    sides = {
        split: [row for row in records if assignment[row["group"]] == split]
        for split in ("train", "val")
    }
    group_sets = {
        split: {row["group"] for row in rows} for split, rows in sides.items()
    }
    if group_sets["train"] & group_sets["val"]:
        raise ValueError("A calligrapher/character group crosses the split")
    overlap_counts = {"calligrapher_character_groups": 0}
    for field in (
        "original_file_sha256",
        "original_pixel_sha256",
        "target_file_sha256",
        "target_pixel_sha256",
    ):
        hashes = {
            split: {identities[row["record"]["target_path"]][field] for row in rows}
            for split, rows in sides.items()
        }
        overlap = hashes["train"] & hashes["val"]
        overlap_counts[field] = len(overlap)
        if overlap:
            raise ValueError(f"Cross-split image leakage in {field}: {overlap}")
    counts = {
        split: Counter((row["author"], row["role"]) for row in rows)
        for split, rows in sides.items()
    }
    by_author = {}
    for author in sorted({row["author"] for row in records}):
        data = {}
        for split in ("train", "val"):
            bf = counts[split][(author, "BF")]
            jt = counts[split][(author, "JT")]
            if not bf or not jt:
                raise ValueError(f"Missing task coverage: {author}/{split}")
            data[split] = {"BF": bf, "JT": jt, "total": bf + jt}
        data["val_ratio"] = data["val"]["total"] / (
            data["val"]["total"] + data["train"]["total"]
        )
        by_author[author] = data
    moved = Counter(
        f"{row['old_split']}->{assignment[row['group']]}" for row in records
    )
    character_sets = {
        split: {row["character"] for row in rows} for split, rows in sides.items()
    }
    character_statistics = {}
    for split, rows in sides.items():
        source_characters = {Path(row["record"]["image_path"]).stem for row in rows}
        if source_characters != character_sets[split]:
            raise ValueError("Source/target distinct character sets disagree")
        character_statistics[split] = {
            "records": len(rows),
            "distinct_characters": len(character_sets[split]),
            "author_character_groups": len(group_sets[split]),
            "characters": sorted(character_sets[split]),
        }
    character_statistics["shared_characters_across_different_authors"] = len(
        character_sets["train"] & character_sets["val"]
    )
    character_statistics["val_characters_absent_from_train"] = len(
        character_sets["val"] - character_sets["train"]
    )
    return {
        "total_records": len(records),
        "total_groups": len(assignment),
        "train_records": len(sides["train"]),
        "val_records": len(sides["val"]),
        "actual_val_ratio": len(sides["val"]) / len(records),
        "train_groups": len(group_sets["train"]),
        "val_groups": len(group_sets["val"]),
        "cross_split_overlap_counts": overlap_counts,
        "per_author": by_author,
        "old_to_new_records": dict(moved),
        "all_authors_and_both_roles_present_in_both_splits": True,
        "character_statistics": character_statistics,
    }


def write_outputs(records, assignment, identities, destination, report, split_dirs):
    """Save compatible per-folder JSON lists with separate audit metadata."""
    destination.mkdir(parents=True, exist_ok=False)
    folders = sorted({row["record"]["font_folder"] for row in records})
    for split in ("train", "val"):
        directory = split_dirs[split]
        directory.mkdir(parents=True, exist_ok=False)
        for folder in folders:
            rows = [
                row["record"]
                for row in records
                if row["record"]["font_folder"] == folder
                and assignment[row["group"]] == split
            ]
            path = directory / f"font_{split}_{folder}.json"
            path.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    manifest = [
        {
            "target_path": row["record"]["target_path"],
            "author": row["author"],
            "character": row["character"],
            "role": row["role"],
            "old_split": row["old_split"],
            "split": assignment[row["group"]],
            **identities[row["record"]["target_path"]],
        }
        for row in records
    ]
    (destination / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report["output_json_sha256"] = {
        os.path.relpath(path, destination): sha256(path)
        for directory in split_dirs.values()
        for path in sorted(directory.glob("*.json"))
    }
    report["output_directories"] = {
        split: os.path.relpath(directory, destination)
        for split, directory in split_dirs.items()
    }
    (destination / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    """Create a new versioned split, then verify saved records and inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--val-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-output", type=Path)
    parser.add_argument("--val-output", type=Path)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()
    if not 0 < args.val_ratio < 1:
        parser.error("--val-ratio must be strictly between zero and one")
    if args.output.exists():
        raise FileExistsError(f"Output must be a new directory: {args.output}")
    split_dirs = {
        "train": args.train_output or args.output / "train_json",
        "val": args.val_output or args.output / "val_json",
    }
    if len({path.resolve() for path in split_dirs.values()}) != 2:
        raise ValueError("Train/val output directories must differ")
    for directory in split_dirs.values():
        if directory.exists():
            raise FileExistsError(f"JSON output directory already exists: {directory}")
    records, inputs = load_records(args.train_input, args.val_input)
    assets, identities = inventory_assets(records, args.data_root)
    assignment = assign_splits(records, args.val_ratio, args.seed)
    if assignment != assign_splits(records, args.val_ratio, args.seed):
        raise AssertionError("The split is not reproducible")
    report = audit_split(records, assignment, identities)
    report.update(
        {
            "schema": "calliphase-author-character-split-v1",
            "seed": args.seed,
            "requested_val_ratio": args.val_ratio,
            "basis": "Union of retained input train/val lists; no record filtering",
            "grouping": "author + target character after removing numeric suffix",
            "different_author_same_character_may_cross_splits": True,
            "selection": "Seeded shuffled group subset sums for author/BF/JT quotas",
            "input_json_sha256": inputs,
            "referenced_asset_sha256": assets,
            "input_fields_preserved": True,
            "protected_inputs_unchanged": False,
        }
    )
    for path, digest in {**inputs, **assets}.items():
        if sha256(path) != digest:
            raise AssertionError(f"Protected input changed: {path}")
    report["protected_inputs_unchanged"] = True
    write_outputs(records, assignment, identities, args.output, report, split_dirs)
    original = {row["record"]["target_path"]: row["record"] for row in records}
    saved = {}
    for directory in split_dirs.values():
        for path in directory.glob("*.json"):
            for row in json.loads(path.read_text(encoding="utf-8")):
                if row["target_path"] in saved:
                    raise AssertionError("Duplicate output target")
                saved[row["target_path"]] = row
    if saved != original:
        raise AssertionError("Saved outputs changed or omitted input records")
    compact = {
        key: report[key]
        for key in (
            "total_records",
            "total_groups",
            "train_records",
            "val_records",
            "actual_val_ratio",
            "cross_split_overlap_counts",
            "per_author",
        )
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
