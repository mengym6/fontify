import json
import random
import shutil
from pathlib import Path
from typing import Optional


random.seed(42)

DATA_ROOT = Path(__file__).resolve().parent
NEW_BASE = Path("font/train/new")
TRAIN_OUTPUT_DIR = Path("train_json_new")
VAL_OUTPUT_DIR = Path("val_json_new")
VAL_RATIO = 0.15
SAMPLE_RATIO = 0.5
CLEAR_OUTPUT = True

DEFAULT_SOURCE_DIR = Path("ttf/SourceHanSansSC-Regular")
TARGET_SUBDIR_CANDIDATES = [
    "images_text_denoised",
    "images_white_bg_mask_denoised",
    "images_white_bg",
    "images",
]
ANNOTATIONS_SUBDIR = "annotations"
ANNOTATIONS_FILENAME = "instances_default.json"
SEMANTIC_MASKS_SUBDIR = "semantic_masks"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# 可选：如果某些字体必须使用特定 source 字体，在这里填。
# 未配置的字体自动使用 DEFAULT_SOURCE_DIR。
FOLDER_TO_TTF = {}


def infer_role(folder_name: str) -> str:
    if "BF" in folder_name or "笔法" in folder_name:
        return "BF"
    if "JT" in folder_name or "结体" in folder_name:
        return "JT"
    return ""


def pair_type_for(folder_name: str) -> str:
    role = infer_role(folder_name)
    if role and role not in folder_name:
        return f"font_{folder_name}_{role}"
    return f"font_{folder_name}"


def choose_target_dir(font_dir: Path) -> Optional[Path]:
    for subdir in TARGET_SUBDIR_CANDIDATES:
        candidate = font_dir / subdir
        if candidate.is_dir():
            return candidate
    return None


def choose_source_dir(folder_name: str) -> Path:
    configured = FOLDER_TO_TTF.get(folder_name)
    if configured:
        return Path("ttf") / configured
    return DEFAULT_SOURCE_DIR


def rel(path: Path) -> str:
    return path.as_posix()


def prepare_output_dirs():
    train_abs = DATA_ROOT / TRAIN_OUTPUT_DIR
    val_abs = DATA_ROOT / VAL_OUTPUT_DIR
    if CLEAR_OUTPUT:
        for path in (train_abs, val_abs):
            if path.exists():
                shutil.rmtree(path)
    train_abs.mkdir(parents=True, exist_ok=True)
    val_abs.mkdir(parents=True, exist_ok=True)
    return train_abs, val_abs


def build_pairs_for_font(font_dir: Path):
    folder_name = font_dir.name
    target_dir = choose_target_dir(font_dir)
    if target_dir is None:
        print(f"{folder_name}: 未找到目标图片目录，跳过")
        return []

    source_dir = choose_source_dir(folder_name)
    source_abs = DATA_ROOT / source_dir
    fallback_abs = DATA_ROOT / DEFAULT_SOURCE_DIR
    if not fallback_abs.is_dir():
        print(f"{folder_name}: 缺少默认 source 目录 {fallback_abs}，跳过")
        return []

    source_files = set()
    if source_abs.is_dir():
        source_files = {p.name for p in source_abs.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    fallback_files = {p.name for p in fallback_abs.iterdir() if p.suffix.lower() in IMAGE_EXTS}

    ann_path = font_dir / ANNOTATIONS_SUBDIR / ANNOTATIONS_FILENAME
    rel_ann = rel(NEW_BASE / folder_name / ANNOTATIONS_SUBDIR / ANNOTATIONS_FILENAME) if ann_path.is_file() else None

    pairs = []
    unpaired = []
    fallback_count = 0
    target_files = sorted(p for p in target_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for target_file in target_files:
        source_name = target_file.stem[0] + target_file.suffix
        if source_name in source_files:
            image_path = source_dir / source_name
        elif source_name in fallback_files:
            image_path = DEFAULT_SOURCE_DIR / source_name
            fallback_count += 1
        else:
            unpaired.append(target_file.name)
            continue

        char_name = target_file.stem
        semantic_mask_path = NEW_BASE / folder_name / SEMANTIC_MASKS_SUBDIR / f"{char_name}.npy"
        item = {
            "image_path": rel(image_path),
            "target_path": rel(NEW_BASE / folder_name / target_dir.name / target_file.name),
            "type": pair_type_for(folder_name),
            "font_folder": folder_name,
        }
        if rel_ann:
            item["annotation_path"] = rel_ann
        if (DATA_ROOT / semantic_mask_path).is_file():
            item["semantic_mask_path"] = rel(semantic_mask_path)
        pairs.append(item)

    print(
        f"{folder_name}: target_dir={target_dir.name}, pairs={len(pairs)}, "
        f"fallback={fallback_count}, unpaired={len(unpaired)}"
    )
    if unpaired:
        sample = "、".join(Path(x).stem for x in unpaired[:10])
        print(f"    丢弃(source 无对应字): {sample}{' ...' if len(unpaired) > 10 else ''}")
    return pairs


def sample_pairs_for_font(folder_name: str, pairs: list[dict]) -> list[dict]:
    if not 0 < SAMPLE_RATIO <= 1:
        raise ValueError(f"SAMPLE_RATIO must be in (0, 1], got {SAMPLE_RATIO}")
    if SAMPLE_RATIO == 1 or len(pairs) <= 1:
        return pairs

    sample_count = max(1, int(len(pairs) * SAMPLE_RATIO))
    sampled = random.sample(pairs, sample_count)
    print(f"{folder_name}: random_sample={sample_count}/{len(pairs)}")
    return sampled


def main():
    train_abs, val_abs = prepare_output_dirs()
    new_abs = DATA_ROOT / NEW_BASE
    font_dirs = sorted(p for p in new_abs.iterdir() if p.is_dir() and not p.name.startswith(("_", ".")))

    total_train = 0
    total_val = 0
    for font_dir in font_dirs:
        pairs = build_pairs_for_font(font_dir)
        if not pairs:
            continue
        pairs = sample_pairs_for_font(font_dir.name, pairs)
        random.shuffle(pairs)
        val_count = max(1, int(len(pairs) * VAL_RATIO))
        val_pairs = pairs[:val_count]
        train_pairs = pairs[val_count:]

        train_path = train_abs / f"font_train_{font_dir.name}.json"
        val_path = val_abs / f"font_val_{font_dir.name}.json"
        with train_path.open("w", encoding="utf-8") as fp:
            json.dump(train_pairs, fp, ensure_ascii=False, indent=2)
        with val_path.open("w", encoding="utf-8") as fp:
            json.dump(val_pairs, fp, ensure_ascii=False, indent=2)

        total_train += len(train_pairs)
        total_val += len(val_pairs)

    print(f"完成：train={total_train}, val={total_val}")
    print(f"输出目录：{train_abs} / {val_abs}")


if __name__ == "__main__":
    main()
