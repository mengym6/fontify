"""Replay BF coordinates and reject masks from a different GT geometry."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from util.dual_targets import render_segmentation, resize_native_layer

AUDIT_SCHEMA = "bf-native-geometry-v1"


def bf_annotation_path(pair):
    """BF masks and annotations belong to the same native-image folder."""
    return str(Path(pair["bf_semantic_mask_path"]).parent.parent
               / "annotations" / "instances_default.json")


def render_bf_layer(segmentation, height, width):
    """Match the existing BF generator's cv2 integer-polygon convention."""
    if isinstance(segmentation, dict):
        return render_segmentation(segmentation, height, width)
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    polygons = segmentation or []
    if polygons and isinstance(polygons[0], (int, float)):
        polygons = [polygons]
    for polygon in polygons:
        if len(polygon) < 6 or len(polygon) % 2:
            raise ValueError("Malformed BF polygon")
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
    return mask


class BFGeometryAuditor:
    """Audit all BF targets, including BF-only and fixed validation references."""

    def __init__(self, root, native_geometry):
        self.root = Path(root)
        self.native_geometry = native_geometry
        self.indices = {}

    def _index(self, annotation_path):
        if annotation_path not in self.indices:
            data = json.loads((self.root / annotation_path).read_text())
            images = {Path(row["file_name"]).name: row for row in data["images"]}
            text_ids = {row["id"] for row in data["categories"]
                        if row["name"] == "text"}
            if len(text_ids) != 1:
                raise ValueError(f"Expected one BF text category: {annotation_path}")
            annotations = {}
            for row in data["annotations"]:
                annotations.setdefault(row["image_id"], []).append(row)
            self.indices[annotation_path] = images, text_ids, annotations
        return self.indices[annotation_path]

    def audit(self, pair):
        """Require every NPY layer and GT text geometry to agree with COCO."""
        annotation_path = bf_annotation_path(pair)
        images, text_ids, annotations = self._index(annotation_path)
        info = images[Path(pair["target_path"]).name]
        height, width = info["height"], info["width"]
        with Image.open(self.root / pair["original_path"]) as original:
            if original.size != (width, height):
                raise ValueError("BF COCO/raw-image size mismatch")
        stored = np.load(self.root / pair["bf_semantic_mask_path"], allow_pickle=False)
        if stored.ndim != 3 or stored.shape[1:] != (448, 448):
            raise ValueError("BF semantic NPY must have shape (K,448,448)")
        text = np.zeros((height, width), dtype=np.uint8)
        ids, different = [], 0
        for annotation in annotations.get(info["id"], []):
            segmentation = annotation["segmentation"]
            if annotation["category_id"] in text_ids:
                text |= render_segmentation(segmentation, height, width)
                continue
            layer = render_bf_layer(segmentation, height, width)
            if not layer.any():
                continue
            index = len(ids)
            if index >= len(stored):
                raise ValueError("BF NPY has fewer layers than native annotations")
            expected = resize_native_layer(layer, self.native_geometry)
            different += int(np.count_nonzero((stored[index] > 0) != (expected > 0)))
            ids.append(annotation["id"])
        if not ids or len(ids) != len(stored) or different:
            raise ValueError(
                f"BF/GT coordinate audit failed: {pair['target_path']} "
                f"geometry={self.native_geometry}, different_pixels={different}, "
                f"native_layers={len(ids)}, stored_layers={len(stored)}"
            )
        if not text.any():
            raise ValueError("BF has no native text annotation")
        # GT uses antialiased text rendering; labels remain nearest-neighbor.
        image = Image.fromarray(text * 255)
        if self.native_geometry == "pad_square":
            side = max(width, height)
            canvas = Image.new("L", (side, side), 0)
            canvas.paste(image, ((side - width) // 2, (side - height) // 2))
            image = canvas
        expected = 255 - np.asarray(image.resize((448, 448), Image.Resampling.LANCZOS))
        with Image.open(self.root / pair["target_path"]) as target:
            actual = np.asarray(target.convert("L"))
        if actual.shape != (448, 448):
            raise ValueError("BF GT must be 448 square")
        mae = float(np.abs(actual.astype(float) - expected.astype(float)).mean() / 255)
        union = (actual < 128) | (expected < 128)
        iou = float(((actual < 128) & (expected < 128)).sum() / max(union.sum(), 1))
        if mae > 0.001 or iou < 0.995:
            raise ValueError(
                f"BF text/GT coordinate audit failed: {pair['target_path']} "
                f"geometry={self.native_geometry}, mae={mae:.6f}, iou={iou:.6f}"
            )
        return {
            "annotation_path": annotation_path,
            "mask_path": pair["bf_semantic_mask_path"],
            "native_geometry": self.native_geometry,
            "native_size": [height, width], "native_size_verified": True,
            "annotation_ids": ids, "different_mask_pixels": different,
            "gt_text_iou": iou, "gt_text_mae": mae,
        }
