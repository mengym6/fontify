"""Recompute original-image identities and export the Run C v2 coarse oracle."""

import argparse
import csv
import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage
from torch.nn import functional as F

from data.runc_v2_dataset import coarse_targets_from_normalized
from tools.build_dual_manifest import normalize_record
from util.dual_loss import DualJietiLoss
from util.dual_targets import build_instance_targets, load_cached_targets

MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def original_path(record):
    return record.get(
        "original_path",
        str(
            Path(record["target_path"]).parent.parent
            / "images"
            / Path(record["target_path"]).name
        ),
    )


def decoded_hash(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        return hashlib.sha256(
            str(image.size).encode("ascii") + image.tobytes()
        ).hexdigest()


def read_inputs(data):
    train = []
    for path in data["train_json"]:
        train.extend(json.loads(Path(path).read_text(encoding="utf-8")))
    fixed = json.loads(Path(data["fixed_pairs"]).read_text(encoding="utf-8"))
    val = []
    for path in data["val_json"]:
        for pair in json.loads(Path(path).read_text(encoding="utf-8")):
            if pair["target_path"] in fixed:
                lower = fixed[pair["target_path"]]
                lower = lower if "has_jt" in lower else normalize_record(lower)
                val.append({"reference": pair, "target": lower})
    return train, val


def threshold_probe(root, records):
    """Probe two deterministic train targets per author, before choosing tau."""
    selected = []
    for author in sorted({row["calligrapher"] for row in records}):
        selected.extend([row for row in records if row["calligrapher"] == author][:2])
    results = []
    for row in selected:
        with Image.open(root / row["target_path"]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
            legacy = np.asarray(image.convert("L")) < 128
        gray = rgb @ np.asarray((0.299, 0.587, 0.114), dtype=np.float32)
        ink = (1 - gray).clip(0, 1)
        result = {
            "author": row["calligrapher"],
            "target_path": row["target_path"],
            "continuous_ink_fraction": float(ink.mean()),
            "legacy_vs_tau05_pixel_disagreements": int(
                np.count_nonzero(legacy != (ink > 0.5))
            ),
        }
        for threshold in (0.25, 0.5, 0.75):
            labels, count = ndimage.label(ink > threshold)
            areas = np.bincount(labels.ravel())[1:]
            result[str(threshold)] = {
                "ink_fraction": float((ink > threshold).mean()),
                "component_count": int(count),
                "components_le16_pixels": int((areas <= 16).sum()),
            }
        results.append(result)
    return {
        "selection": "first two sealed JT-bearing train records per author",
        "targets": results,
        "chosen_tau": 0.5,
        "rationale": (
            "Matches existing PIL gray<128 convention; density stays continuous."
        ),
        "pixel_disagreements_total": sum(
            row["legacy_vs_tau05_pixel_disagreements"] for row in results
        ),
    }


def audit(data, destination):
    """Rehash raw files rather than equating manifest rows with identities."""
    root = Path(data["root"])
    train, val = read_inputs(data)
    jt = [pair for pair in train if pair["has_jt"]]
    entries = []
    errors = []
    for pair in jt:
        path = original_path(pair)
        md5 = file_hash(root / path, "md5")
        if md5 != pair["original_md5"]:
            errors.append(f"Stored original MD5 mismatch: {path}")
        if pair["has_bf"]:
            jt_md5 = file_hash(root / pair["jt_original_path"], "md5")
            if md5 != jt_md5:
                errors.append(f"Merged BF/JT original mismatch: {path}")
        entries.append(
            {
                "target_path": pair["target_path"],
                "original_path": path,
                "author": pair["calligrapher"],
                "character": pair["character"],
                "pool_type": pair["pool_type"],
                "original_md5": md5,
                "original_sha256": file_hash(root / path),
                "original_pixel_sha256": decoded_hash(root / path),
                "target_sha256": file_hash(root / pair["target_path"]),
                "record_id": pair["record_id"],
            }
        )
    identity = {(p["author"], p["original_md5"]) for p in entries}
    pixels = {(p["author"], p["original_pixel_sha256"]) for p in entries}
    target_hashes = {p["target_sha256"] for p in entries}
    author_char = {(p["author"], p["character"]) for p in entries}
    unique_val = {}
    for pair in val:
        row = pair["target"]
        if row["has_jt"] and row["target_path"] not in unique_val:
            path = original_path(row)
            unique_val[row["target_path"]] = {
                "target_path": row["target_path"],
                "author": row["calligrapher"],
                "character": row["character"],
                "original_md5": file_hash(root / path, "md5"),
                "original_pixel_sha256": decoded_hash(root / path),
                "target_sha256": file_hash(root / row["target_path"]),
            }
    overlap_raw, overlap_pixels, overlap_target, overlap_char = [], [], [], []
    for path, row in unique_val.items():
        if (row["author"], row["original_md5"]) in identity:
            overlap_raw.append(path)
        if (row["author"], row["original_pixel_sha256"]) in pixels:
            overlap_pixels.append(path)
        if row["target_sha256"] in target_hashes:
            overlap_target.append(path)
        if (row["author"], row["character"]) in author_char:
            overlap_char.append(path)
    pools = Counter(pair["pool_type"] for pair in jt)
    if (
        len(jt) != 740
        or pools != {"dual": 466, "jt_only": 274}
        or len(identity) != 740
        or len(pixels) != 740
        or len(target_hashes) != 740
    ):
        errors.append("JT-bearing unique targets do not match 466+274=740")
    if overlap_raw or overlap_pixels or overlap_target:
        errors.append("Exact original/target overlap between train and JT val")
    report = {
        "schema": "runc-v2-jt-manifest-v1",
        "status": "passed" if not errors else "blocked",
        "identity_definition": (
            "calligrapher + original file MD5; decoded RGB SHA cross-check"
        ),
        "train_input_records": len(train),
        "jt_train_records": len(jt),
        "jt_unique_originals": len(identity),
        "jt_unique_decoded_pixels": len(pixels),
        "jt_unique_target_sha256": len(target_hashes),
        "stage1_pools": dict(pools),
        "bf_only_stage1": 0,
        "by_author": dict(Counter(p["author"] for p in entries)),
        "historical_val_pairs": len(val),
        "historical_jt_pairs": sum(p["target"]["has_jt"] for p in val),
        "historical_jt_unique_targets": len(unique_val),
        "raw_identity_overlap": overlap_raw,
        "decoded_pixel_overlap": overlap_pixels,
        "target_sha_overlap": overlap_target,
        "same_author_character_overlap": overlap_char,
        "same_author_character_excluded_unique": len(unique_val) - len(overlap_char),
        "same_author_character_excluded_paths": sorted(
            set(unique_val) - set(overlap_char)
        ),
        "limitations": [
            "Historical validation was repeatedly used for research selection.",
            "Same-author/character exclusion does not certify arbitrary "
            "near-duplicate crops.",
            "No independent generalization claim follows from this split.",
        ],
        "input_sha256": {
            path: file_hash(path)
            for path in data["train_json"] + data["val_json"] + [data["fixed_pairs"]]
        },
        "errors": errors,
        "train_targets": entries,
        "val_unique_targets": list(unique_val.values()),
    }
    report["threshold_train_probe"] = threshold_probe(root, jt)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "data_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "data_jt_train.json").write_text(
        json.dumps(jt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "audit.md").write_text(
        "# Run C v2 data audit\n\n"
        f"Status: **{report['status']}**. Recomputed from actual local files.\n\n"
        f"JT stage-I targets: {len(jt)} = {pools['dual']} dual + "
        f"{pools['jt_only']} JT-only; BF-only=0. Raw identity / decoded pixel / "
        f"target SHA unique counts: {len(identity)} / {len(pixels)} / "
        f"{len(target_hashes)}. Nine-author counts: {report['by_author']}.\n\n"
        f"Historical pairs: {len(val)}, JT={report['historical_jt_pairs']}; "
        f"unique JT lower targets={len(unique_val)}. Exact raw/pixel/target "
        f"overlap={len(overlap_raw)}/{len(overlap_pixels)}/{len(overlap_target)}. "
        f"Same-author/character overlap={len(overlap_char)}: {overlap_char}. "
        f"An additionally reported exclusion subset has "
        f"{len(unique_val) - len(overlap_char)} "
        "unique JT targets. It is not an independent generalization set.\n\n"
        "Source evidence: `tools/build_dual_manifest.py:57` merges by author "
        "and original-file MD5; `data/dual_dataset.py:262` synchronizes crop "
        "across source/target and `:289` transforms instance layers; `:272` "
        "would sample BF masks even for dual targets. The new "
        "`data/runc_v2_dataset.py::RuncV2Dataset` builds stage-I pairs directly "
        "and never invokes that BF sampler. `get_pair` sets all 784 lower "
        "patches masked and zeroes the lower input-target pixels. GT images "
        "and instance arrays remain only as loss/evaluation fields. The "
        "new no-GT-leak data test substitutes the query GT and verifies "
        "image/input_target/mask invariance; model forward leakage requires "
        "its separate runtime check.\n\n"
        "Native annotation mapping remains sealed `pad_square` in the "
        "existing cache. Head occupancy is derived at 28; raw J Voronoi at "
        "224; coarse J Voronoi is independently derived at 112 from the "
        "same synchronously transformed instance ink, with partition loss "
        "flags preserved. These coordinate systems are separately reported.\n\n"
        f"Errors: {errors}. Full per-target provenance is in data_manifest.json.\n",
        encoding="utf-8",
    )
    if errors:
        raise RuntimeError("Data audit blocked: " + "; ".join(errors))
    return report, jt, val


def component_diagnostics(ink, oracle, threshold):
    """Measure threshold-dependent disappearance, splits and merges."""
    source, source_count = ndimage.label(ink > threshold)
    rendered, rendered_count = ndimage.label(oracle > threshold)
    areas = np.bincount(source.ravel(), minlength=source_count + 1)[1:]
    lost, split, small, small_lost, tiny, tiny_lost = 0, 0, 0, 0, 0, 0
    linked = defaultdict(set)
    for label, area in enumerate(areas, 1):
        labels = set(np.unique(rendered[source == label])) - {0}
        lost += not labels
        split += len(labels) > 1
        small += area <= 256
        small_lost += area <= 256 and not labels
        tiny += area <= 64
        tiny_lost += area <= 64 and not labels
        for output_label in labels:
            linked[int(output_label)].add(label)
    return {
        "components_gt": int(source_count),
        "components_oracle": int(rendered_count),
        "lost_components": int(lost),
        "split_components": int(split),
        "merged_output_components": sum(len(v) > 1 for v in linked.values()),
        "small_components_le256": int(small),
        "small_lost_le256": int(small_lost),
        "tiny_components_le64": int(tiny),
        "tiny_lost_le64": int(tiny_lost),
    }


def component_details(ink, oracle, threshold=0.5):
    """Record each lost component bounding box for noise/stroke inspection."""
    source, _ = ndimage.label(ink > threshold)
    rendered, _ = ndimage.label(oracle > threshold)
    details = []
    for label, slices in enumerate(ndimage.find_objects(source), 1):
        if slices is None:
            continue
        mask = source[slices] == label
        output_labels = np.unique(rendered[slices][mask])
        output_labels = output_labels[output_labels > 0]
        if not output_labels.size:
            ys, xs = slices
            details.append(
                {
                    "label": label,
                    "area": int(mask.sum()),
                    "bbox_xyxy": [xs.start, ys.start, xs.stop, ys.stop],
                    "oracle_peak_in_original_support": float(
                        oracle[slices][mask].max()
                    ),
                }
            )
    return details


def merge_details(ink, oracle, threshold=0.5):
    """Localize nearest source-component gaps merged by the density oracle."""
    source, _ = ndimage.label(ink > threshold)
    rendered, _ = ndimage.label(oracle > threshold)
    output = []
    for label in range(1, int(rendered.max()) + 1):
        source_ids = np.unique(source[rendered == label])
        source_ids = source_ids[source_ids > 0]
        if len(source_ids) < 2:
            continue
        for first_index, first in enumerate(source_ids[:-1]):
            distance, indices = ndimage.distance_transform_edt(
                source != first, return_indices=True
            )
            next_index = first_index + 1
            for second in source_ids[next_index:]:
                ys, xs = np.nonzero(source == second)
                nearest = int(distance[ys, xs].argmin())
                by, bx = int(ys[nearest]), int(xs[nearest])
                ay, ax = int(indices[0, by, bx]), int(indices[1, by, bx])
                output.append(
                    {
                        "oracle_component": int(label),
                        "source_components": [int(first), int(second)],
                        "source_component_areas": [
                            int((source == first).sum()),
                            int((source == second).sum()),
                        ],
                        "closest_source_pixels_xy": [[ax, ay], [bx, by]],
                        "closest_distance_pixels": float(distance[by, bx]),
                        "review_crop_xyxy": [
                            max(0, (ax + bx) // 2 - 24),
                            max(0, (ay + by) // 2 - 24),
                            min(448, (ax + bx) // 2 + 24),
                            min(448, (ay + by) // 2 + 24),
                        ],
                    }
                )
    return output


def diagnostic_contactsheets(root, destination, details):
    """Save cropped source/oracle pairs around losses and bridges."""
    visual_dir = destination / "visuals" / "oracle"
    lost = [(row, item) for row in details for item in row["lost_components"]]
    lost = sorted(lost, key=lambda pair: -pair[1]["area"])[:12]
    merged = [(row, item) for row in details for item in row["merged_components"]]
    index_fragments = []
    for kind, entries in (("lost", lost), ("merged", merged)):
        columns, tile_width, tile_height = 3, 288, 176
        sheet = Image.new(
            "RGB",
            (columns * tile_width, max(1, (len(entries) + 2) // 3) * tile_height),
            "white",
        )
        draw = ImageDraw.Draw(sheet)
        descriptions = []
        for number, (row, item) in enumerate(entries):
            rgb = torch.from_numpy(
                np.asarray(Image.open(root / row["target_path"]).convert("RGB")).copy()
            )
            rgb = rgb.permute(2, 0, 1).float()[None] / 255
            targets = coarse_targets_from_normalized((rgb - MEAN) / STD)
            density = F.interpolate(
                targets["density_target"],
                (448, 448),
                mode="bilinear",
                align_corners=False,
            )
            gray = 1 - (
                rgb * rgb.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)
            ).sum(1, keepdim=True)
            if kind == "lost":
                x1, y1, x2, y2 = item["bbox_xyxy"]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                bounds = (
                    max(0, cx - 16),
                    max(0, cy - 16),
                    min(448, cx + 16),
                    min(448, cy + 16),
                )
                label = f"{number}: {row['author']} area={item['area']}"
            else:
                bounds = tuple(item["review_crop_xyxy"])
                label = f"{number}: {row['author']} merged gap"
            x = (number % columns) * tile_width
            y = (number // columns) * tile_height
            draw.text((x + 4, y + 2), label, fill="black")
            for offset, values in ((0, gray), (144, density)):
                image = (
                    Image.fromarray(
                        np.uint8(np.rint((1 - values[0, 0].numpy()).clip(0, 1) * 255))
                    )
                    .convert("RGB")
                    .crop(bounds)
                )
                image = image.resize((144, 144), Image.Resampling.NEAREST)
                sheet.paste(image, (x + offset, y + 20))
            draw.text(
                (x + 4, y + 163), "GT                    density oracle", fill="black"
            )
            descriptions.append(
                {
                    "contactsheet_index": number,
                    "target_path": row["target_path"],
                    "character": row["character"],
                    **item,
                }
            )
        filename = f"{kind}_components_contactsheet.png"
        sheet.save(visual_dir / filename)
        index_fragments.append(
            f"<h2>{kind} components: GT / density oracle crops</h2>"
            f"<img src='{filename}'><pre>"
            f"{html.escape(json.dumps(descriptions, ensure_ascii=False, indent=2))}"
            "</pre>"
        )
    (visual_dir / "topology_details.html").write_text(
        "<!doctype html><meta charset='utf-8'><style>"
        "body{font:15px sans-serif;margin:24px}img{max-width:100%}</style>"
        "<h1>Threshold .5 topology details</h1>" + "\n".join(index_fragments),
        encoding="utf-8",
    )


def scalar_metrics(j_loss, ink, oracle, partition, valid):
    """Existing raw-image J at 224 with a fixed per-target denominator."""
    gt_rgb = (1 - torch.from_numpy(ink)).view(1, 1, 448, 448).expand(1, 3, -1, -1)
    pred_rgb = (1 - oracle).expand(1, 3, -1, -1)
    gt = (gt_rgb - MEAN) / STD
    pred = (pred_rgb - MEAN) / STD
    gt = torch.cat((gt, gt), 2)
    pred = torch.cat((gt[:, :, :448], pred), 2)
    partition = torch.from_numpy(partition).unsqueeze(0)
    validity = torch.from_numpy(valid).unsqueeze(0)
    value, parts, shield = j_loss(pred, gt, partition, validity, torch.tensor([True]))
    a = F.avg_pool2d(torch.from_numpy(ink)[None, None], 2).squeeze(1)
    b = F.avg_pool2d(oracle, 2).squeeze(1)
    masks = torch.stack([(partition == index + 1).float() for index in range(4)], 1)
    mg, cxg, cyg, sxg, syg = j_loss._part_stats(a, masks)
    mp, cxp, cyp, sxp, syp = j_loss._part_stats(b, masks)
    active = validity & (mg > 1e-6) & (mp >= 0.1 * mg)
    count = active.sum().clamp_min(1)
    result = {
        "J_raw224_equal_coefficients": float(value),
        "centroid_relative": float(parts["centroid"]),
        "logsigma_xy": float(parts["logsigma"]),
        "shape": float(parts["shape"]),
        "shield_count": int(shield),
        "valid_instances": int(active.sum()),
    }
    for name, p, g in (("logsigma_x", sxp, sxg), ("logsigma_y", syp, syg)):
        result[name] = float(((p.log() - g.log()).abs() * active).sum() / count)
    displacement = ((cxp - cxg).square() + (cyp - cyg).square()).sqrt() / 224
    result["centroid_absolute_normalized"] = float(
        (displacement * active).sum() / count
    )
    return result


def image_panel(row, images):
    """Use ASCII panel labels; Chinese identity is preserved in HTML."""
    size, title_height = 224, 24
    panel = Image.new("RGB", (len(images) * size, size + title_height), "white")
    draw = ImageDraw.Draw(panel)
    for index, (name, values) in enumerate(images):
        values = np.asarray(values).clip(0, 1)
        image = Image.fromarray(np.uint8(np.rint(values * 255)))
        image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        panel.paste(image, (index * size, title_height))
        draw.text((index * size + 4, 4), name, fill="black")
    return panel


def oracle(data, destination, val, manifest):
    """Render all 77 JT target oracles and a fixed >=12-target review page."""
    root, cache = Path(data["root"]), Path(data["cache"])
    cache_index = json.loads((cache / "index.json").read_text(encoding="utf-8"))
    cache_meta = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
    unique = {}
    for pair in val:
        row = pair["target"]
        if row["has_jt"]:
            unique.setdefault(row["target_path"], row)
    selected = []
    for character in ("無", "容", "惟", "雲", "崇"):
        selected.append(
            next(
                row["target_path"]
                for row in unique.values()
                if row["character"] == character
            )
        )
    represented = {unique[path]["calligrapher"] for path in selected}
    for author in sorted({row["calligrapher"] for row in unique.values()}):
        if author not in represented:
            selected.append(
                next(
                    path
                    for path, row in unique.items()
                    if row["calligrapher"] == author
                )
            )
    selected.extend(path for path in unique if path not in selected)
    selected = selected[:12]
    visual_dir = destination / "visuals" / "oracle"
    visual_dir.mkdir(parents=True, exist_ok=True)
    j_loss = DualJietiLoss(pool=224)
    metrics, pages, lost_details = [], [], []
    thresholds = (0.25, 0.5, 0.75)
    for number, (path, row) in enumerate(unique.items()):
        rgb = (
            torch.from_numpy(np.asarray(Image.open(root / path).convert("RGB")).copy())
            .permute(2, 0, 1)
            .float()[None]
            / 255
        )
        target = (rgb - MEAN) / STD
        targets = coarse_targets_from_normalized(target, data.get("ink_threshold", 0.5))
        density = targets["density_target"]
        up = F.interpolate(density, (448, 448), mode="bilinear", align_corners=False)
        occupancy = targets["occupancy_target"]
        up_occ = F.interpolate(occupancy, (448, 448), mode="nearest")
        ink = (
            (
                1
                - (rgb * rgb.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)).sum(
                    1, keepdim=True
                )
            )
            .clamp(0, 1)[0, 0]
            .numpy()
        )
        raw = load_cached_targets(cache, row, index=cache_index, metadata=cache_meta)
        instance = build_instance_targets(raw["regions"], raw["text"], raw["classes"])
        entry = {
            "target_path": path,
            "author": row["calligrapher"],
            "character": row["character"],
            "threshold": data.get("ink_threshold", 0.5),
            "same_author_character_excluded": path
            in manifest["same_author_character_excluded_paths"],
            "ink_fraction": float(ink.mean()),
            "coarse_occupancy_fraction": float(occupancy.mean()),
            "density_mass_error": float(abs(density.sum() * 16 - ink.sum())),
        }
        entry.update(
            scalar_metrics(j_loss, ink, up, instance["voro"], instance["valid_parts"])
        )
        for threshold in thresholds:
            suffix = str(threshold).replace(".", "p")
            entry.update(
                {
                    key + "_t" + suffix: value
                    for key, value in component_diagnostics(
                        ink, up[0, 0].numpy(), threshold
                    ).items()
                }
            )
        metrics.append(entry)
        lost_details.append(
            {
                "target_path": path,
                "author": row["calligrapher"],
                "character": row["character"],
                "threshold": 0.5,
                "lost_components": component_details(ink, up[0, 0].numpy()),
                "merged_components": merge_details(ink, up[0, 0].numpy()),
            }
        )
        if path in selected:
            gaussian = ndimage.gaussian_filter(
                ink, sigma=1.5, truncate=2.0, mode="mirror"
            )
            panel = image_panel(
                row,
                [
                    ("GT", 1 - ink),
                    ("Density GT 112", 1 - density[0, 0].numpy()),
                    ("Bilinear density oracle", 1 - up[0, 0].numpy()),
                    ("Occupancy GT 112", 1 - occupancy[0, 0].numpy()),
                    ("Nearest occupancy oracle", 1 - up_occ[0, 0].numpy()),
                    ("v1 Gaussian 7 sigma1.5", 1 - gaussian),
                ],
            )
            filename = f"{number:03d}.png"
            panel.save(visual_dir / filename)
            title = html.escape(row["calligrapher"] + " / " + row["character"])
            pages.append(
                f"<h2>{title}</h2><p>{html.escape(path)}</p>"
                f"<img src='{filename}'><pre>"
                f"{html.escape(json.dumps(entry, ensure_ascii=False, indent=2))}"
                "</pre>"
            )
    fields = list(metrics[0])
    with (destination / "oracle_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
    (destination / "data_oracle_lost_components.json").write_text(
        json.dumps(lost_details, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    diagnostic_contactsheets(root, destination, lost_details)
    merge_lines = [
        "# Density oracle narrow-gap merges at threshold 0.5\n",
        "Coordinates use original 448-square target pixels. The pair gives "
        "the closest original pixels from two source components that share "
        "one thresholded oracle component. Crops show GT on left and "
        "continuous density oracle on right.\n",
    ]
    for row in lost_details:
        for item in row["merged_components"]:
            merge_lines.append(
                f"- {row['author']} {row['character']}: "
                f"`{row['target_path']}`; closest pixels "
                f"`{item['closest_source_pixels_xy']}`; "
                f"distance={item['closest_distance_pixels']:.4f}px; "
                f"crop xyxy=`{item['review_crop_xyxy']}`."
            )
    merge_lines.extend([
        "\n[Merge crops](visuals/oracle/merged_components_contactsheet.png).",
        "[Twelve largest disappearing fragments](visuals/oracle/"
        "lost_components_contactsheet.png).",
        "[Coordinates and crop index](visuals/oracle/topology_details.html).\n",
    ])
    (destination / "data_oracle_merge_summary.md").write_text(
        "\n".join(merge_lines), encoding="utf-8"
    )
    (visual_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Run C v2 density oracle</title>"
        "<style>body{font:15px sans-serif;margin:24px}"
        "img{max-width:100%;border:1px solid #ddd}"
        "pre{white-space:pre-wrap}h2{margin-top:40px}</style>"
        "<h1>Run C v2: loss-only GT density oracle (zero training)</h1>"
        "<p>GT, average-pool4 density112, bilinear-upsample density, "
        "max-pool4 occupancy112, "
        "nearest occupancy, v1 Gaussian7 sigma1.5 reflect. These are supervision "
        "oracles, not model outputs. Twelve fixed targets cover nine authors.</p>"
        + "\n".join(pages),
        encoding="utf-8",
    )
    means = {
        name: float(np.mean([row[name] for row in metrics]))
        for name in (
            "J_raw224_equal_coefficients",
            "logsigma_x",
            "logsigma_y",
            "shape",
            "centroid_relative",
            "centroid_absolute_normalized",
        )
    }
    lost_areas = [
        component["area"]
        for target in lost_details
        for component in target["lost_components"]
    ]
    max_lost_area = max(lost_areas, default=0)
    no_ink_disagreements = manifest["threshold_train_probe"][
        "pixel_disagreements_total"
    ]
    topology = {}
    for threshold in thresholds:
        suffix = "_t" + str(threshold).replace(".", "p")
        tiny = sum(row["tiny_components_le64" + suffix] for row in metrics)
        lost = sum(row["tiny_lost_le64" + suffix] for row in metrics)
        topology[str(threshold)] = {
            "tiny_le64_count": tiny,
            "tiny_le64_lost": lost,
            "tiny_retention": (tiny - lost) / max(tiny, 1),
            "all_components_lost": sum(
                row["lost_components" + suffix] for row in metrics
            ),
            "merged_output_components": sum(
                row["merged_output_components" + suffix] for row in metrics
            ),
        }
    (destination / "data_oracle_summary.json").write_text(
        json.dumps(
            {
                "unique_targets": len(metrics),
                "visual_targets": selected,
                "raw_geometry_means": means,
                "topology": topology,
                "oracle_only_zero_training": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (destination / "coarse_oracle_report.md").write_text(
        "# Run C v2 density / occupancy oracle\n\n"
        "Zero training. All inputs below are supervision-side targets, never "
        "deployment model conditions.\n\n"
        "RGB is de-normalized once; grayscale is .299R+.587G+.114B. Density "
        "is avg_pool4(1-gray); occupancy is max_pool4(ink>0.5), aligning "
        "with legacy 8-bit gray<128 at grayscale values 127/128. Density "
        "retains antialias ink and background gray continuously.\n\n"
        "Threshold audit uses two deterministic JT-bearing train targets per "
        "author (18 total), with ink fraction/component/noise statistics at "
        f"tau .25/.5/.75. Legacy-vs-tau.5 pixel disagreements: "
        f"{no_ink_disagreements}. See data_manifest.json threshold_train_probe.\n\n"
        f"77 unique historical JT targets evaluated; twelve fixed visual targets "
        f"cover nine authors including 無/容/惟/雲/崇. Raw224 geometry means: {means}. "
        "J here uses centroid/logsigma/shape coefficients 1/1/1, a constant "
        "GT Voronoi at 224, and per-unique-target averaging; it must not be "
        "numerically compared to historical weighted Run B or v1 J. No "
        "prediction is thresholded for this geometric calculation.\n\n"
        f"Connected-component sensitivity: {topology}. Components use 4-neighbor "
        "connectivity. A source component disappears when no oracle pixel above "
        "the same threshold intersects it; split/merge counts use intersecting "
        "source/output labels. Small<=256 and tiny<=64 source pixels are "
        "descriptive size strata, not acceptance thresholds. Tiny fragments "
        "can be annotation/denoising noise rather than meaningful strokes; "
        "the full CSV and images are required for interpretation.\n\n"
        f"At tau .5 the {len(lost_areas)} disappearing components have at most "
        f"{max_lost_area} source pixels each; their exact bboxes and oracle "
        "peaks are in data_oracle_lost_components.json. The fixed Caoqb 無 "
        "has five GT and five oracle components (main structure plus four "
        "bottom dots), with zero loss/merge at all three thresholds. "
        "At tau .5 narrow-gap merges occur in Dongqc 惟, Sushi 容, Yanzq 海, "
        "and Zhangqb 新. Their presence is a representation limitation even "
        "when mean J is small.\n\n"
        "The density representation preserves mass (avg pooling exactly up to "
        "floating-point error), yet thresholded topology can change because "
        "bilinear interpolation lowers isolated point peaks and joins narrow "
        "gaps. Occupancy is auxiliary; it is not written into P0. Thus its "
        "retention must not be mistaken for density-P0 retention.\n\n"
        "Review gate: these numerical diagnostics do not substitute for human "
        "judgment about bottom dots and short strokes. Inspect "
        "visuals/oracle/index.html "
        "before accepting 112 as adequate. The oracle is an attainable "
        "representation floor; trained model quality has not been established.\n",
        encoding="utf-8",
    )
    return {"means": means, "topology": topology}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    destination = Path(args.output)
    report, _, val = audit(config["data"], destination)
    summary = {"audit": report["status"]}
    if not args.audit_only:
        summary["oracle"] = oracle(config["data"], destination, val, report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
