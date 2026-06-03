#!/usr/bin/env python3
"""Export validation image summaries from TensorBoard event files."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TAG_CONTAINS = "val x im_masked y tgt"
EPOCH_RE = re.compile(r"(?:^|[^a-zA-Z0-9])epoch[:_\-/ ]*(-?\d+)(?:[^a-zA-Z0-9]|$)")


@dataclass(frozen=True)
class ImageRecord:
    epoch: int
    step: int
    wall_time: float
    tag: str
    encoded: bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export PNG validation images written by engine_train.evaluate_pt(). "
            "By default, it saves the first four images from the latest val epoch."
        )
    )
    parser.add_argument("--log_dir", required=True, help="TensorBoard log directory.")
    parser.add_argument("--output_dir", default=None, help="Directory for exported PNG files.")
    parser.add_argument("--limit", type=int, default=4, help="Number of images to export.")
    parser.add_argument("--epoch", type=int, default=None, help="Specific epoch to export. Defaults to latest.")
    parser.add_argument("--list_tags", action="store_true", help="List image-like summary tags and exit.")
    parser.add_argument(
        "--tag_contains",
        default="val x; im_masked; y; tgt",
        help="Only image summaries whose tag contains this text are considered.",
    )
    return parser.parse_args()


def event_files(log_dir: Path) -> list[Path]:
    files = sorted(
        (p for p in log_dir.rglob("events.out.tfevents.*") if p.is_file()),
        key=lambda p: (p.stat().st_mtime, str(p)),
    )
    return files


def normalize_tag(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return re.sub(r"_+", "_", normalized)


def tag_matches(tag: str, tag_contains: str) -> bool:
    if tag_contains in tag:
        return True
    return normalize_tag(tag_contains) in normalize_tag(tag)


def tag_epoch(tag: str, tag_contains: str) -> int | None:
    if not tag_matches(tag, tag_contains):
        return None

    match = EPOCH_RE.match(tag)
    if match:
        return int(match.group(1))

    match = re.search(r"(?:^|_)epoch_(-?\d+)(?:_|$)", normalize_tag(tag))
    if match:
        return int(match.group(1))

    if tag.startswith("epoch:"):
        raw = tag.split(" ", 1)[0].split(":", 1)[1]
        try:
            return int(raw)
        except ValueError:
            return None
    return None


def install_numpy_compat_shims() -> None:
    """Keep older TensorBoard imports working with NumPy 2.x."""
    try:
        import numpy as np
    except ModuleNotFoundError:
        return

    if not hasattr(np, "string_"):
        np.string_ = np.bytes_
    if not hasattr(np, "unicode_"):
        np.unicode_ = np.str_


def encoded_image_from_value(value) -> bytes:
    if value.HasField("image"):
        return value.image.encoded_image_string or b""

    if value.HasField("tensor"):
        tensor = value.tensor
        for item in tensor.string_val:
            if item.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF8", b"RIFF")):
                return bytes(item)
        if tensor.string_val:
            return bytes(tensor.string_val[0])

    return b""


def is_image_like_value(value) -> bool:
    if value.HasField("image"):
        return True
    if value.HasField("tensor"):
        plugin_name = value.metadata.plugin_data.plugin_name
        if plugin_name == "images":
            return True
        return bool(encoded_image_from_value(value))
    return False


def load_event_records(files: list[Path]):
    try:
        install_numpy_compat_shims()
        from tensorboard.backend.event_processing import event_file_loader
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "tensorboard is required to read event files. "
            "Use the same Python environment as training, or install requirements.txt."
        ) from exc

    for event_file in files:
        loader = event_file_loader.EventFileLoader(str(event_file))
        for event in loader.Load():
            if not event.summary.value:
                continue
            for value in event.summary.value:
                yield event, value


def list_image_tags(files: list[Path]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for _, value in load_event_records(files):
        if not is_image_like_value(value):
            continue
        counts[value.tag] = counts.get(value.tag, 0) + 1
    return sorted(counts.items(), key=lambda item: item[0])


def format_available_tags(files: list[Path]) -> str:
    tags = list_image_tags(files)
    if not tags:
        return "No image-like summary tags were found in the event files."

    lines = ["Available image-like summary tags:"]
    for tag, count in tags[:80]:
        lines.append(f"  {tag}  ({count})")
    if len(tags) > 80:
        lines.append(f"  ... {len(tags) - 80} more tag(s)")
    return "\n".join(lines)


def iter_images(files: list[Path], tag_contains: str):
    for event, value in load_event_records(files):
        if not is_image_like_value(value):
            continue
        epoch = tag_epoch(value.tag, tag_contains)
        if epoch is None:
            continue
        encoded = encoded_image_from_value(value)
        if not encoded:
            continue
        yield ImageRecord(
            epoch=epoch,
            step=event.step,
            wall_time=event.wall_time,
            tag=value.tag,
            encoded=encoded,
        )


def collect_images(files: list[Path], requested_epoch: int | None, limit: int, tag_contains: str) -> tuple[int, list[ImageRecord]]:
    records: list[ImageRecord] = []
    latest_epoch: int | None = None

    for record in iter_images(files, tag_contains):
        if requested_epoch is not None:
            if record.epoch != requested_epoch:
                continue
            records.append(record)
            if len(records) >= limit:
                return requested_epoch, records
            continue

        if latest_epoch is None or record.epoch > latest_epoch:
            latest_epoch = record.epoch
            records = [record]
        elif record.epoch == latest_epoch and len(records) < limit:
            records.append(record)

    if requested_epoch is not None and not records:
        raise RuntimeError(f"No validation image summaries were found for epoch {requested_epoch}.")
    if latest_epoch is None:
        raise RuntimeError("No validation image summaries were found.")
    return latest_epoch, records


def export_images(records: list[ImageRecord], output_dir: Path, target_epoch: int, limit: int) -> list[Path]:
    if not records:
        raise RuntimeError(f"No validation image summaries were found for epoch {target_epoch}.")

    records.sort(key=lambda record: (record.step, record.wall_time, record.tag))
    selected = records[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    manifest_path = output_dir / "manifest.csv"

    with manifest_path.open("w", newline="") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(["filename", "epoch", "step", "tag", "wall_time"])
        for index, record in enumerate(selected):
            filename = f"val_epoch_{record.epoch:04d}_batch_{record.step:04d}_{index + 1:02d}.png"
            path = output_dir / filename
            path.write_bytes(record.encoded)
            writer.writerow([filename, record.epoch, record.step, record.tag, f"{record.wall_time:.6f}"])
            exported.append(path)

    return exported


def main() -> int:
    args = parse_args()
    log_dir = Path(args.log_dir)

    if args.limit <= 0:
        raise SystemExit("--limit must be positive.")
    if not log_dir.exists():
        raise SystemExit(f"Log directory does not exist: {log_dir}")

    files = event_files(log_dir)
    if not files:
        raise SystemExit(f"No TensorBoard event files found under: {log_dir}")

    if args.list_tags:
        print(format_available_tags(files))
        return 0

    if args.output_dir is None:
        raise SystemExit("--output_dir is required unless --list_tags is used.")

    try:
        target_epoch, records = collect_images(files, args.epoch, args.limit, args.tag_contains)
        exported = export_images(records, Path(args.output_dir), target_epoch, args.limit)
    except RuntimeError as exc:
        raise SystemExit(f"{exc}\n{format_available_tags(files)}") from exc

    print(f"Exported {len(exported)} validation image(s) from epoch {target_epoch} to {args.output_dir}")
    for path in exported:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
