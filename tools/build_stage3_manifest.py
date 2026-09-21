"""Build the stage-3 CalliPhase manifest with explicit style/character/glyph identities."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect_units(data_root, calli_dir, target_subdir, source_dir):
    """Group usable targets by (writer, character).

    同一书家同一字的 BF/JT 目录以及 `永1`/`永2` 这类重复书写视为同一原始字形的
    不同版本，必须落在同一划分里，因此以 (writer, character) 为最小划分单元。
    """
    units = defaultdict(list)
    dropped = []
    source_names = {p.name for p in (data_root / source_dir).iterdir()}
    folders = sorted(p for p in (data_root / calli_dir).iterdir() if p.is_dir())
    for folder in folders:
        role = folder.name[-2:]
        if role not in ("BF", "JT"):
            continue
        writer = folder.name[:-2]
        target_dir = folder / target_subdir
        if not target_dir.is_dir():
            dropped.append([folder.name, "missing target subdir"])
            continue
        targets = sorted(
            p for p in target_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        for target in targets:
            relative = target.relative_to(data_root).as_posix()
            character = target.stem[0]
            source = f"{character}{target.suffix}"
            mask = folder / "semantic_masks" / f"{target.stem}.npy"
            if source not in source_names:
                dropped.append([relative, "no source glyph"])
                continue
            if not mask.is_file():
                dropped.append([relative, "no semantic mask"])
                continue
            units[(writer, character)].append(
                {
                    "image_path": (source_dir / source).as_posix(),
                    "target_path": relative,
                    "type": role,
                    "style_id": writer,
                    "character": character,
                    "glyph_id": f"{writer}-{target.stem}",
                    "source_dataset": "calliphase",
                }
            )
    return units, dropped


def choose_val_characters(units, val_ratio, min_val, rng):
    """Pick one global val character set holding out about val_ratio per writer.

    val_seen 的字不能出现在任何书家的 train 里，所以优先选被较少书家共用的字，
    每选一个字就同时计入它所有书家的 val 配额。
    """
    writers_of = defaultdict(set)
    chars_of = defaultdict(set)
    for writer, character in units:
        writers_of[character].add(writer)
        chars_of[writer].add(character)
    target = {w: max(min_val, int(len(c) * val_ratio)) for w, c in chars_of.items()}
    count = dict.fromkeys(chars_of, 0)
    val = set()

    def take(character):
        val.add(character)
        for writer in writers_of[character]:
            count[writer] += 1

    order = sorted(writers_of, key=lambda c: (len(writers_of[c]), rng.random()))
    for character in order:
        if all(count[w] < target[w] for w in writers_of[character]):
            take(character)
    for writer in sorted(chars_of):
        while count[writer] < min_val:
            candidates = chars_of[writer] - val
            if not candidates:
                raise ValueError(f"{writer} cannot reach {min_val} val characters")
            take(min(candidates, key=lambda c: (len(writers_of[c]), rng.random())))
    return val


def split_units(units, val_chars):
    train, val = [], []
    for (_, character), records in sorted(units.items()):
        (val if character in val_chars else train).extend(records)
    return train, val


def summarize(train, val):
    """Per-writer counts; also verify the reference constraints audit will enforce."""
    report = {}
    writers = sorted({r["style_id"] for r in train + val})
    for writer in writers:
        row = {}
        for name, records in (("train", train), ("val_seen", val)):
            mine = [r for r in records if r["style_id"] == writer]
            row[name] = {
                "records": len(mine),
                "characters": len({r["character"] for r in mine}),
                "BF": len([r for r in mine if r["type"] == "BF"]),
                "JT": len([r for r in mine if r["type"] == "JT"]),
            }
            for role in ("BF", "JT"):
                chars = {r["character"] for r in mine if r["type"] == role}
                if name == "train" and 0 < len(chars) < 2:
                    raise ValueError(f"{writer}/{role}: train needs 2+ characters")
        if row["val_seen"]["characters"] < 2:
            raise ValueError(f"{writer}: val_seen needs 2+ characters")
        report[writer] = row
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--calli-dir", default="font/train/new")
    parser.add_argument("--target-subdir", default="images_text_denoised")
    parser.add_argument("--source-dir", default="ttf/SourceHanSansSC-Regular")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--min-val-characters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 < args.val_ratio < 1:
        parser.error("--val-ratio must be in (0, 1)")
    data_root = Path(args.data_root).resolve()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    units, dropped = collect_units(
        data_root, Path(args.calli_dir), args.target_subdir, Path(args.source_dir)
    )
    if not units:
        raise ValueError("No usable CalliPhase targets found")
    val_chars = choose_val_characters(
        units, args.val_ratio, args.min_val_characters, random.Random(args.seed)
    )
    train, val = split_units(units, val_chars)
    report = summarize(train, val)
    for name, records in (("train", train), ("val_seen", val)):
        (output / f"{name}.json").write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n"
        )
    (output / "manifest.json").write_text(
        json.dumps({"train": ["train.json"], "val_seen": ["val_seen.json"]}, indent=2)
        + "\n"
    )
    summary = {
        "args": vars(args),
        "train_records": len(train),
        "val_seen_records": len(val),
        "val_fraction": len(val) / (len(train) + len(val)),
        "val_characters": sorted(val_chars),
        "writers": report,
        "dropped": dropped,
    }
    (output / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        f"train={len(train)} val_seen={len(val)} val_fraction={summary['val_fraction']:.3f}"
    )
    for writer, row in report.items():
        print(
            f"{writer}: train {row['train']['records']} records / "
            f"{row['train']['characters']} chars, val_seen {row['val_seen']['records']} "
            f"records / {row['val_seen']['characters']} chars"
        )
    if dropped:
        print(f"dropped {len(dropped)} targets, see split_summary.json")


if __name__ == "__main__":
    main()
