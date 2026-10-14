"""Stage, seal and install one reproducible CalliPhase coordinate version.

No originals or annotations are regenerated. Derived GT and NPY use native
COCO coordinates, center padding, then 448-square resizing. Changed installed
files are backed up, including on the server. Checkpoints are never touched.
"""

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "calliphase-aligned-pad-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def safe_path(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Invalid relative asset path: {relative}")
    return path


def build(data, output):
    """Rebuild existing target identities; preserve all split/filename choices."""
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(ROOT / "font-preload"))
    denoise = importlib.import_module("denoise_by_text_annotation")
    semantic = importlib.import_module("render_semantic_mask")
    output.mkdir(parents=True)
    staged, files, counts = output / "staged", {}, {}
    no_semantic_layers = []
    for folder in sorted((data / "font/train/new").iterdir()):
        if not folder.name.endswith(("BF", "JT")):
            continue
        annotation_path = folder / "annotations/instances_default.json"
        coco = denoise.load_coco(annotation_path)
        text_id = denoise.find_category_id(coco["categories"], "text")
        text_annotations = denoise.annotations_by_image(coco["annotations"], text_id)
        by_image = {}
        for annotation in coco["annotations"]:
            by_image.setdefault(annotation["image_id"], []).append(annotation)
        files[str(annotation_path.relative_to(data))] = sha256(annotation_path)
        count = 0
        for info in coco["images"]:
            name = Path(info["file_name"]).name
            target = folder / "images_text_denoised" / name
            if not target.is_file():
                continue
            raw = folder / "images" / name
            with Image.open(raw) as image:
                if image.size != (info["width"], info["height"]):
                    raise ValueError(f"COCO/raw dimensions disagree: {raw}")
                text = denoise.render_text_mask(info, text_annotations[info["id"]])
                result = denoise.apply_text_mask(image, text, "mask")
                result = denoise.pad_to_square(result, (255, 255, 255))
                result = result.resize((448, 448), Image.Resampling.LANCZOS)
                relative = str(target.relative_to(data))
                destination = safe_path(staged, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                result.save(destination)
                files[relative] = sha256(destination)
            files[str(raw.relative_to(data))] = sha256(raw)
            layers = semantic.render_single_image_mask(
                info, by_image[info["id"]], text_id, coco["categories"],
                per_annotation=folder.name.endswith("BF"),
            )
            if layers is None:
                if folder.name.endswith("BF"):
                    raise ValueError(f"No BF semantic layers for {raw}")
                no_semantic_layers.append(str(raw.relative_to(data)))
                count += 1
                continue
            layers = np.stack([semantic.pad_and_resize_mask(layer, 448)
                               for layer in layers])
            relative = str((folder / "semantic_masks" / (Path(name).stem + ".npy"))
                           .relative_to(data))
            destination = safe_path(staged, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(destination, layers, allow_pickle=False)
            files[relative] = sha256(destination)
            count += 1
        counts[folder.name] = count
        print(f"staged {folder.name}: {count}", flush=True)
    manifest = {
        "schema": SCHEMA, "native_geometry": "pad_square", "image_size": 448,
        "gt_mode": "mask", "gt_interpolation": "PIL-LANCZOS",
        "label_interpolation": "PIL-NEAREST", "bf_layers": "per-annotation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "counts": counts, "files": files,
        "jt_without_semantic_layers": no_semantic_layers,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def read_manifest(path):
    value = json.loads(Path(path).read_text())
    if value.get("schema") != SCHEMA or value.get("native_geometry") != "pad_square":
        raise ValueError("Unknown aligned-data manifest")
    if not value.get("files"):
        raise ValueError("Empty aligned-data manifest")
    return value


def verify(data, manifest):
    missing, different = [], []
    for relative, expected in manifest["files"].items():
        path = safe_path(data, relative)
        if not path.is_file():
            missing.append(relative)
        elif sha256(path) != expected:
            different.append(relative)
    return {"checked_files": len(manifest["files"]),
            "missing": missing, "different": different,
            "status": "passed" if not missing and not different else "mismatch"}


def install(data, manifest, staged, backup):
    """Validate every input before replacing anything; retain all old bytes."""
    report = verify(data, manifest)
    changed = report["missing"] + report["different"]
    if backup.exists():
        raise FileExistsError(backup)
    for relative in changed:
        source = safe_path(staged, relative)
        if not source.is_file() or sha256(source) != manifest["files"][relative]:
            raise ValueError(f"Staged asset missing/corrupt: {relative}")
    backup.mkdir(parents=True)
    journal = {"status": "installing", "changed": changed, "old_sha256": {}}
    write_json(backup / "installation.json", journal)
    for relative in changed:
        target, source = safe_path(data, relative), safe_path(staged, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            old = safe_path(backup, relative)
            old.parent.mkdir(parents=True, exist_ok=True)
            journal["old_sha256"][relative] = sha256(target)
            shutil.copy2(target, old)
        temporary = target.with_name(target.name + ".geometry-replacement")
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    report = verify(data, manifest)
    journal.update(status=report["status"], verification=report)
    write_json(backup / "installation.json", journal)
    if report["status"] != "passed":
        raise ValueError("Post-installation asset verification failed")
    return journal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "seal", "verify", "install"))
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--staged", type=Path)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--include", nargs="*", default=[])
    args = parser.parse_args()
    if args.action == "build":
        value = build(args.data_path, args.output)
        print(json.dumps({"files": len(value["files"]), "counts": value["counts"]}))
        return
    manifest = read_manifest(args.manifest)
    if args.action == "seal":
        for relative in args.include:
            path = safe_path(args.data_path, relative)
            paths = sorted(path.rglob("*")) if path.is_dir() else [path]
            for item in paths:
                if item.is_file():
                    key = str(item.relative_to(args.data_path))
                    manifest["files"][key] = sha256(item)
        write_json(args.manifest, manifest)
        result = {"status": "sealed", "files": len(manifest["files"])}
    elif args.action == "verify":
        result = verify(args.data_path, manifest)
    else:
        result = install(args.data_path, manifest, args.staged, args.backup)
    if args.output:
        write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("changed", "old_sha256", "different", "missing")}))
    if result.get("status") not in ("passed", "sealed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
