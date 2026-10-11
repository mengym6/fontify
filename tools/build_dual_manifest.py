"""Merge clean train records by calligrapher and original-image MD5."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def file_hash(path, algorithm="sha256"):
    """Hash a file without depending on the image decoder."""
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_record(record):
    """Annotate a single original BF/JT record without merging validation."""
    out = dict(record)
    folder = record.get("font_folder", Path(record["target_path"]).parent.parent.name)
    if folder.endswith("BF"):
        kind = "bf"
    elif folder.endswith("JT"):
        kind = "jt"
    else:
        raise ValueError(f"Unknown BF/JT folder: {folder}")
    out.update(
        calligrapher=folder[:-2],
        has_bf=kind == "bf",
        has_jt=kind == "jt",
        pool_type=f"{kind}_only",
        character=Path(record["image_path"]).stem,
        original_path=str(
            Path(record["target_path"]).parent.parent
            / "images"
            / Path(record["target_path"]).name
        ),
    )
    if kind == "bf":
        out["bf_semantic_mask_path"] = record["semantic_mask_path"]
    else:
        out["jt_annotation_path"] = record["annotation_path"]
        out["jt_target_path"] = record["target_path"]
        out["jt_original_path"] = out["original_path"]
    return out


def merge_records(records, root):
    """Return merged records and a reproducible summary; never touch inputs."""
    root = Path(root)
    groups = {}
    for record in records:
        item = normalize_record(record)
        original = root / item["original_path"]
        item["original_md5"] = file_hash(original, "md5")
        key = (item["calligrapher"], item["original_md5"])
        groups.setdefault(key, []).append(item)
    output = []
    duplicates = 0
    for (author, original_md5), items in sorted(groups.items()):
        bf = sorted((p for p in items if p["has_bf"]), key=lambda p: p["target_path"])
        jt = sorted((p for p in items if p["has_jt"]), key=lambda p: p["target_path"])
        duplicates += max(len(bf) - 1, 0) + max(len(jt) - 1, 0)
        # Clean v2 double annotations must refer to the same computer glyph.
        if bf and jt and bf[0]["character"] != jt[0]["character"]:
            raise ValueError(f"BF/JT character disagreement: {items}")
        item = dict((bf or jt)[0])
        item.update(
            has_bf=bool(bf),
            has_jt=bool(jt),
            pool_type="dual" if bf and jt else item["pool_type"],
            record_id=f"{author}:{original_md5}",
            source_records=items,
        )
        if bf:
            item["bf_semantic_mask_path"] = bf[0]["bf_semantic_mask_path"]
        if jt:
            for field in ("jt_annotation_path", "jt_target_path", "jt_original_path"):
                item[field] = jt[0][field]
        output.append(item)
    pools = Counter(p["pool_type"] for p in output)
    by_author = {}
    for item in output:
        pool = by_author.setdefault(item["calligrapher"], Counter())
        pool[item["pool_type"]] += 1
    summary = {
        "schema": "dual-manifest-v1",
        "input_records": len(records),
        "merged_records": len(output),
        "same_kind_duplicates": duplicates,
        "pools": dict(pools),
        "by_calligrapher": {k: dict(v) for k, v in by_author.items()},
        "merge_key": ["calligrapher", "images-file-md5"],
        "double_annotation_target": "BF-images_text_denoised",
    }
    return output, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", "--data-path", required=True)
    parser.add_argument("--input_dir", "--input-dir", required=True)
    parser.add_argument("--output_dir", "--output-dir", required=True)
    args = parser.parse_args(argv)
    source = Path(args.input_dir)
    destination = Path(args.output_dir)
    if destination.exists():
        raise FileExistsError(f"Output directory must not exist: {destination}")
    paths = sorted(source.glob("*.json"))
    if not paths:
        raise ValueError(f"No input JSON files in {source}")
    records = []
    hashes = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list):
            raise ValueError(f"Expected record list: {path}")
        records.extend(rows)
        hashes[path.name] = file_hash(path)
    merged, summary = merge_records(records, args.data_path)
    summary["input_sha256"] = hashes
    destination.mkdir(parents=True, exist_ok=False)
    output = destination / "train_dual.json"
    output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary["manifest_sha256"] = file_hash(output)
    # Keep the report out of glob('*.json') dataset input enumeration.
    (destination / "summary.txt").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
