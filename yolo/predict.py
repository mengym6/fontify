"""Export Calli phase/JT instances as overlays, binary-mask RLE, and polygon JSON."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def encode_binary_mask(mask) -> dict:
    """Serialize a binary mask without joining disconnected parts or filling holes."""
    import numpy as np
    from pycocotools import mask as mask_utils

    binary = np.asfortranarray(np.asarray(mask) > 0.5, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError("An instance mask must be a two-dimensional array")
    encoded = mask_utils.encode(binary)
    return {
        "size": [int(value) for value in encoded["size"]],
        "counts": encoded["counts"].decode("ascii"),
    }


def chinese_names(path: Path) -> dict[int, str]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("classes", data) if isinstance(data, dict) else data
    if isinstance(rows, list):
        return {
            int(row.get("id", i)): row.get("name_zh", row.get("label_zh", row.get("name", str(i))))
            for i, row in enumerate(rows) if isinstance(row, dict)
        }
    if isinstance(data, dict):
        mapping = data.get("names_zh", data.get("names", {}))
        if isinstance(mapping, list):
            return dict(enumerate(mapping))
        if isinstance(mapping, dict):
            return {int(key): str(value) for key, value in mapping.items()}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True,
                        help="A trained Calli phase/JT best.pt or last.pt (must exist locally)")
    parser.add_argument("--source", type=Path, required=True, help="Image file or image directory")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "predictions" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--classes", type=Path, default=ROOT / "dataset" / "classes.json")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()
    weights, source = args.weights.expanduser().resolve(), args.source.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Trained checkpoint does not exist: {weights}")
    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")
    if not 0 <= args.conf <= 1:
        raise ValueError("conf must lie in [0, 1]")
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")

    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(str(weights))
    if model.task != "segment":
        raise ValueError("Expected a Calli instance-segmentation checkpoint")
    translated = chinese_names(args.classes.expanduser())
    if translated and len(model.names) != len(translated):
        raise ValueError(
            "Checkpoint classes do not match classes.json. Use the trained Calli weights, "
            "not the 80-class COCO initialization, or supply the matching --classes file."
        )
    output.mkdir(parents=True, exist_ok=True)
    index = []
    results = model.predict(
        source=str(source), device=args.device, imgsz=args.imgsz,
        conf=args.conf, max_det=args.max_det, retina_masks=True,
        save=False, stream=True, verbose=False,
    )
    for image_index, result in enumerate(results):
        instances = []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            if result.masks is None or len(result.masks) != len(boxes):
                raise RuntimeError("Segmentation masks and detections are not aligned")
            polygons = result.masks.xy
            normalized = result.masks.xyn
            for i, box in enumerate(boxes):
                class_id = int(box.cls.item())
                name = str(result.names[class_id])
                raw_mask = result.masks.data[i].detach().cpu().numpy()
                if tuple(raw_mask.shape) != tuple(result.orig_shape):
                    raise RuntimeError("retina_masks must provide masks in original-image coordinates")
                instances.append({
                    "class_id": class_id, "class_name": name,
                    "class_name_zh": translated.get(class_id, name),
                    "confidence": float(box.conf.item()),
                    "bbox_xyxy": box.xyxy[0].cpu().tolist(),
                    "mask_rle": encode_binary_mask(raw_mask),
                    "mask_coordinate_space": "original_image",
                    "polygon_xy": polygons[i].tolist(),
                    "polygon_normalized": normalized[i].tolist(),
                })
        stem = f"{image_index:04d}_{Path(result.path).stem}"
        record = {
            "source": str(result.path), "image_shape_hw": list(result.orig_shape),
            "checkpoint": str(weights), "instances": instances,
            "note": (
                "mask_rle losslessly stores the predicted binary mask, including disconnected "
                "parts and holes. polygon_xy/polygon_normalized are convenience contours "
                "and may join components or omit holes. Masks describe visible regions; "
                "they do not recover writing order or hidden strokes."
            ),
        }
        (output / f"{stem}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # Ultralytics plot() returns BGR. Keep checkpoint class names on overlays
        # for portability; Chinese display names remain available in the JSON.
        overlay_bgr = result.plot()
        Image.fromarray(overlay_bgr[..., ::-1]).save(output / f"{stem}.png")
        index.append({"source": str(result.path), "instances": len(instances),
                      "json": f"{stem}.json", "overlay": f"{stem}.png"})
    (output / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(index)} image results to {output}")


if __name__ == "__main__":
    main()
