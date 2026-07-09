from pathlib import Path
import shutil


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEW_DIR = REPO_ROOT / "fontdata_example" / "font" / "train" / "new"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def iter_font_dirs(new_dir=DEFAULT_NEW_DIR):
    """Yield real font folders under font/train/new in stable name order."""
    new_dir = Path(new_dir)
    if not new_dir.is_dir():
        raise FileNotFoundError(f"new font directory does not exist: {new_dir}")
    return sorted(
        p for p in new_dir.iterdir()
        if p.is_dir() and not p.name.startswith(("_", "."))
    )


def image_files(input_dir):
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        return []
    return sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def prepare_output_dir(output_dir, clear=True):
    output_dir = Path(output_dir)
    if clear and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def annotation_json_path(font_dir, subdir="annotations", filename="instances_default.json"):
    ann_dir = Path(font_dir) / subdir
    preferred = ann_dir / filename
    if preferred.is_file():
        return preferred
    candidates = sorted(ann_dir.glob("*.json")) if ann_dir.is_dir() else []
    return candidates[0] if candidates else None
