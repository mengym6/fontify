"""Run B instance targets, rendered from COCO before synchronous augmentation.

The cached arrays are raw 448-square regions and JT text, never old JT NPY
layers. Occupancy and Voronoi targets are derived after augmentation. Occupancy
retains overlapping instances; J assigns overlapping ink to the later instance
using the existing ink_label_map contract before nearest-grid EDT.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from util.calli_labels import JT_CLASSES, normalize_category
from util.jieti_partition import INF, _edt_sq, ink_label_map

DEFINITION = "instance-lower-full-v1"
SOURCE_SEAL = "all-record-inputs-v1"
CACHE_SCHEMA = "dual-target-cache-v2"
NATIVE_GEOMETRIES = ("pad_square", "direct_resize")
K_MAX = 4
IMAGE_SIZE = 448


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_key(pair):
    """Stable cache identity for the exact image GT path."""
    return hashlib.sha256(pair["target_path"].encode("utf-8")).hexdigest()


def record_source_paths(pair):
    """List input files independently of whether a record has JT supervision.

    BF annotation JSON is deliberately excluded: BF masks come from the NPY
    file, while instance regions and text must always come from JT COCO.
    Originals bind the manifest merge identity and JT coordinate audit.
    """
    fields = ["image_path", "target_path", "original_path"]
    if pair["has_bf"]:
        fields.append("bf_semantic_mask_path")
    if pair["has_jt"]:
        fields.extend(
            ["jt_annotation_path", "jt_original_path", "jt_target_path"]
        )
    paths = {}
    for field in fields:
        value = pair.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Record source path missing: {field}")
        paths[field] = value
    return paths


def render_segmentation(segmentation, height, width):
    """Render one COCO polygon/RLE at the annotation's native resolution."""
    if isinstance(segmentation, dict):
        from pycocotools import mask as mask_util

        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_util.frPyObjects(rle, height, width)
        layer = mask_util.decode(rle)
        if layer.ndim == 3:
            layer = layer.any(axis=2)
        return (layer > 0).astype(np.uint8)
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    polygons = segmentation or []
    if polygons and isinstance(polygons[0], (int, float)):
        polygons = [polygons]
    for polygon in polygons:
        if len(polygon) < 6 or len(polygon) % 2:
            raise ValueError("Malformed COCO polygon")
        draw.polygon(list(zip(polygon[::2], polygon[1::2])), fill=1)
    return np.asarray(image, dtype=np.uint8)


def pad_resize_layer(layer, size=IMAGE_SIZE):
    """Use the target preprocessing's center padding and nearest resizing."""
    height, width = layer.shape
    side = max(height, width)
    square = np.zeros((side, side), dtype=np.uint8)
    y, x = (side - height) // 2, (side - width) // 2
    square[y : y + height, x : x + width] = layer
    image = Image.fromarray(square)
    return np.asarray(image.resize((size, size), Image.Resampling.NEAREST))


def resize_native_layer(layer, native_geometry, size=IMAGE_SIZE):
    """Map native COCO geometry to the existing GT's explicit preprocessing."""
    if native_geometry == "pad_square":
        return pad_resize_layer(layer, size)
    if native_geometry == "direct_resize":
        image = Image.fromarray(np.asarray(layer, dtype=np.uint8))
        return np.asarray(image.resize((size, size), Image.Resampling.NEAREST))
    raise ValueError(f"Unknown native geometry: {native_geometry}")


def cache_native_geometry(metadata, entry=None):
    """Validate the sealed native mapping; old v1 caches used square padding."""
    if metadata.get("schema") == CACHE_SCHEMA:
        geometry = metadata.get("native_geometry")
    elif metadata.get("schema") == "dual-target-cache-v1":
        geometry = "pad_square"
        if metadata.get("native_geometry", geometry) != geometry:
            raise ValueError("Legacy instance cache native geometry must be pad_square")
    else:
        raise ValueError("Unknown instance cache schema for native geometry")
    if geometry not in NATIVE_GEOMETRIES:
        raise ValueError(f"Instance cache native geometry missing/invalid: {geometry}")
    if entry is not None:
        entry_geometry = entry.get("native_geometry")
        if metadata.get("schema") == "dual-target-cache-v1":
            entry_geometry = entry.get("native_geometry", "pad_square")
        if entry_geometry != geometry:
            raise ValueError("Instance cache entry native geometry mismatch")
    return geometry


def deduplicate_instances(regions, classes, annotation_ids, threshold=0.9):
    """Keep the first instance when native region IoU is strictly > threshold."""
    kept = []
    removed = []
    for index, region in enumerate(regions):
        duplicate = None
        for previous in kept:
            union = np.count_nonzero(region | regions[previous])
            iou = np.count_nonzero(region & regions[previous]) / max(union, 1)
            if iou > threshold:
                duplicate = previous
                break
        if duplicate is None:
            kept.append(index)
        else:
            removed.append(
                {
                    "annotation_id": annotation_ids[index],
                    "kept_annotation_id": annotation_ids[duplicate],
                }
            )
    return (
        [regions[i] for i in kept],
        [classes[i] for i in kept],
        [annotation_ids[i] for i in kept],
        removed,
    )


class InstanceRenderer:
    """Index COCO files once and render exact JT instances plus JT text."""

    def __init__(self, root, native_geometry="pad_square"):
        if native_geometry not in NATIVE_GEOMETRIES:
            raise ValueError(f"Unknown native geometry: {native_geometry}")
        self.root = Path(root)
        self.native_geometry = native_geometry
        self._indices = {}

    def _index(self, annotation_path):
        if annotation_path not in self._indices:
            path = self.root / annotation_path
            data = json.loads(path.read_text(encoding="utf-8"))
            images = {Path(p["file_name"]).name: p for p in data["images"]}
            by_image = {}
            for annotation in data["annotations"]:
                by_image.setdefault(annotation["image_id"], []).append(annotation)
            categories = {p["id"]: p["name"] for p in data["categories"]}
            self._indices[annotation_path] = (images, by_image, categories)
        return self._indices[annotation_path]

    def render(self, pair):
        annotation_path = pair.get("jt_annotation_path", pair.get("annotation_path"))
        jt_target = pair.get("jt_target_path", pair["target_path"])
        images, by_image, categories = self._index(annotation_path)
        info = images[Path(jt_target).name]
        height, width = int(info["height"]), int(info["width"])
        annotations = sorted(by_image.get(info["id"], []), key=lambda p: p["id"])
        text = np.zeros((height, width), dtype=bool)
        regions, classes, annotation_ids = [], [], []
        for annotation in annotations:
            name = normalize_category(categories[annotation["category_id"]])
            layer = render_segmentation(annotation["segmentation"], height, width) > 0
            if name == "text":
                text |= layer
            else:
                if name not in JT_CLASSES:
                    raise ValueError(f"Unknown JT category {name}: {annotation_path}")
                regions.append(layer)
                classes.append(JT_CLASSES.index(name))
                annotation_ids.append(annotation["id"])
        if not text.any():
            raise ValueError(f"No JT text annotation for {jt_target}")
        regions, classes, annotation_ids, removed = deduplicate_instances(
            regions, classes, annotation_ids
        )
        if len(regions) > K_MAX:
            raise ValueError(
                f"{len(regions)} unique instances exceed {K_MAX}: {jt_target}"
            )
        padded = np.zeros((K_MAX, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        class_array = np.full(K_MAX, -1, dtype=np.int64)
        for slot, (region, class_id) in enumerate(zip(regions, classes)):
            padded[slot] = resize_native_layer(
                region.astype(np.uint8), self.native_geometry
            )
            class_array[slot] = class_id
        metadata = {
            "annotation_path": annotation_path,
            "jt_target_path": jt_target,
            "image_id": info["id"],
            "native_size": [height, width],
            "native_geometry": self.native_geometry,
            "annotation_ids": annotation_ids,
            "removed_duplicates": removed,
            "instance_count": len(regions),
            "class_names": list(JT_CLASSES),
        }
        text = resize_native_layer(text.astype(np.uint8), self.native_geometry)
        return padded, text, class_array, metadata


def coordinate_audit(
    root,
    pair,
    text,
    metadata,
    min_coverage=0.8,
    max_centroid_offset=0.08,
    source_hashes=None,
):
    """Reject mismatched dimensions, padding geometry, or image/label alignment.

    Text and GT need not have identical ink because a merged target uses BF
    denoising. Coverage and normalized centroid displacement are therefore
    recorded, alongside explicit dimensions and raw-image identity.
    """
    root = Path(root)
    target = np.asarray(Image.open(root / pair["target_path"]).convert("L"))
    if target.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"GT must be 448 square: {pair['target_path']}")
    original_path = pair.get("jt_original_path")
    if original_path is None:
        original_path = str(
            Path(metadata["jt_target_path"]).parent.parent
            / "images"
            / Path(metadata["jt_target_path"]).name
        )
    with Image.open(root / original_path) as image:
        if [image.height, image.width] != metadata["native_size"]:
            raise ValueError(f"COCO/raw-image size mismatch: {original_path}")
    # BF and JT native originals must be byte-identical in merged records.
    if pair.get("has_bf") and pair.get("has_jt"):
        if source_hashes is None:
            jt_hash = sha256_file(root / original_path)
            bf_hash = sha256_file(root / pair["original_path"])
        else:
            jt_hash = source_hashes[original_path]
            bf_hash = source_hashes[pair["original_path"]]
        if jt_hash != bf_hash:
            raise ValueError(f"Merged BF/JT originals disagree: {pair['target_path']}")
    ink = target < 128
    if not ink.any():
        raise ValueError(f"Empty image GT: {pair['target_path']}")
    coverage = float(np.count_nonzero(ink & (text > 0)) / np.count_nonzero(ink))
    ink_center = np.asarray(np.nonzero(ink)).mean(axis=1)
    text_center = np.asarray(np.nonzero(text)).mean(axis=1)
    center_offset = float(np.linalg.norm(ink_center - text_center) / IMAGE_SIZE)
    union = np.count_nonzero(ink | (text > 0))
    iou = float(np.count_nonzero(ink & (text > 0)) / max(union, 1))
    if coverage < min_coverage or center_offset > max_centroid_offset:
        raise ValueError(
            f"JT/GT coordinate audit failed: {pair['target_path']} "
            f"coverage={coverage:.4f}, centroid={center_offset:.4f}"
        )
    return {
        "gt_ink_coverage": coverage,
        "gt_text_iou": iou,
        "normalized_centroid_offset": center_offset,
        "min_coverage": min_coverage,
        "max_centroid_offset": max_centroid_offset,
        "original_path": original_path,
        "native_size_verified": True,
        "native_geometry": metadata.get("native_geometry", "pad_square"),
    }


def build_instance_targets(
    regions, text, classes, patch_size=16, pool=224, valid_ink=200
):
    """Derive targets from transformed layers; keep overlapping ink separate."""
    regions = np.asarray(regions) > 0
    text = np.asarray(text) > 0
    ink = regions & text[None]
    height, width = text.shape
    if height != width or height % patch_size:
        raise ValueError("Target shape must be square and divisible by patch size")
    valid = ink.sum((1, 2)) >= valid_ink
    valid &= np.asarray(classes) >= 0
    grid = height // patch_size
    occupancy = ink.reshape(K_MAX, grid, patch_size, grid, patch_size).any((2, 4))
    occupancy &= valid[:, None, None]
    index = (np.arange(pool) * (height / pool)).astype(np.intp)
    # Occupancy/head supervision keeps each region independent. J is a unique
    # ownership partition: preserve the existing later-layer-wins ink labels
    # before nearest downsampling, rather than let nested seeds lose EDT ties.
    independent_small = ink[:, index[:, None], index[None, :]]
    labels = ink_label_map(ink, K_MAX)
    small_labels = labels[np.ix_(index, index)]
    partition = np.zeros((pool, pool), dtype=np.int64)
    best = np.full((pool, pool), INF, dtype=np.float64)
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        distance_transform_edt = None
    for slot in range(K_MAX):
        small_ink = small_labels == slot + 1
        if not valid[slot] or not small_ink.any():
            continue
        distance = (
            _edt_sq(small_ink)
            if distance_transform_edt is None
            else np.rint(distance_transform_edt(~small_ink) ** 2)
        )
        closer = distance < best
        best = np.where(closer, distance, best)
        partition = np.where(closer, slot + 1, partition)
    small_has_ink = independent_small.any((1, 2))
    partition_present = np.array(
        [(partition == slot + 1).any() for slot in range(K_MAX)]
    )
    return {
        "instance_occupancy": occupancy.astype(np.float32),
        "instance_valid": valid,
        "instance_classes": np.asarray(classes, dtype=np.int64),
        "voro": partition,
        "valid_parts": valid.copy(),
        "j_downsample_lost": valid & ~small_has_ink,
        # A distinct condition: overlap ownership/EDT can lose an instance
        # despite independent nearest-downsampled ink still being present.
        "j_partition_lost": valid & small_has_ink & ~partition_present,
    }


def empty_instance_targets(image_size=448, patch_size=16, pool=224):
    """Return explicit zero supervision for a sample without JT annotation."""
    return {
        "instance_occupancy": np.zeros(
            (K_MAX, image_size // patch_size, image_size // patch_size),
            dtype=np.float32,
        ),
        "instance_valid": np.zeros(K_MAX, dtype=bool),
        "instance_classes": np.full(K_MAX, -1, dtype=np.int64),
        "voro": np.zeros((pool, pool), dtype=np.int64),
        "valid_parts": np.zeros(K_MAX, dtype=bool),
        "j_downsample_lost": np.zeros(K_MAX, dtype=bool),
        "j_partition_lost": np.zeros(K_MAX, dtype=bool),
    }


def transform_layers(layers, crop, output_size=448):
    """Apply one shared (top, left, height, width) crop to all binary layers."""
    top, left, height, width = crop
    layers = np.asarray(layers)
    if layers.ndim == 2:
        layers = layers[None]
    transformed = []
    for layer in layers:
        image = Image.fromarray(layer.astype(np.uint8))
        image = image.crop((left, top, left + width, top + height))
        transformed.append(
            np.asarray(
                image.resize((output_size, output_size), Image.Resampling.NEAREST)
            )
        )
    return np.stack(transformed)


def pool_weights(records):
    """Scheme D: every author/pool receives mass sqrt(pool size)."""
    from collections import Counter

    counts = Counter((p["calligrapher"], p["pool_type"]) for p in records)
    return [1.0 / np.sqrt(counts[(p["calligrapher"], p["pool_type"])]) for p in records]


def load_cached_targets(cache_dir, pair, verify=True, index=None, metadata=None):
    """Load raw instance/text arrays, rejecting stale or incomplete caches."""
    cache_dir = Path(cache_dir)
    owns_index = index is None
    if owns_index:
        index = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
    if metadata is None:
        metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("definition") != DEFINITION or metadata.get("status") != "complete":
        raise ValueError("Instance cache definition/status mismatch")
    if owns_index and sha256_file(cache_dir / "index.json") != metadata["index_sha256"]:
        raise ValueError("Instance cache index hash mismatch")
    entry = index[pair["target_path"]]
    cache_native_geometry(metadata, entry)
    annotation = pair.get("jt_annotation_path", pair.get("annotation_path"))
    jt_target = pair.get("jt_target_path", pair["target_path"])
    if (
        entry["annotation_path"] != annotation
        or entry["jt_target_path"] != jt_target
        or entry["class_names"] != list(JT_CLASSES)
    ):
        raise ValueError("JT instance cache record/category mapping mismatch")
    path = cache_dir / entry["cache_path"]
    if verify and sha256_file(path) != entry["cache_sha256"]:
        raise ValueError(f"Instance cache hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in ("regions", "text", "classes")}
