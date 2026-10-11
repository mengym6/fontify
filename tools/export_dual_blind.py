"""Export sealed, fixed Run B blind panels with neutral left/right labels.

The answer key and blank rating CSV are separate from anonymous image files.
Both model formats use strict loading and predict_full; annotations are never
forwarded to prediction.  The output directory must not already exist.
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dual_view_protocol import (  # noqa: E402
    SCHEMA,
    read_json,
    seed_all,
    sha256_file,
    write_new_json,
)


def parse_models(entries):
    result = {}
    for entry in entries:
        name, separator, path = entry.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError(
                "models require unique model_name=checkpoint entries"
            )
        result[name] = path
    return result


def validate_manifest(manifest, metadata, models):
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("purpose") != "fixed_blind_manifest"
    ):
        raise ValueError("a fixed new-J blind manifest is required")
    if manifest.get("fixed_pair_sha256") != metadata["fixed_pair_sha256"]:
        raise ValueError(
            "blind manifest belongs to another fixed validation set"
        )
    identities = set()
    for row in manifest["comparisons"]:
        identifier = row["blind_id"]
        if identifier in identities or not identifier.startswith("blind_"):
            raise ValueError("blind identifiers must be unique and anonymous")
        identities.add(identifier)
        if any(
            character not in "abcdefghijklmnopqrstuvwxyz_0123456789"
            for character in identifier
        ):
            raise ValueError("unsafe blind identifier")
        if row["left_model"] == row["right_model"]:
            raise ValueError("a model cannot be compared with itself")
        if {row["left_model"], row["right_model"]} - set(models):
            raise ValueError("missing checkpoint for a blinded model")
        if not isinstance(row["index"], int) or row["index"] < 0:
            raise ValueError(
                "blind index must identify a fixed validation row"
            )
    if not identities:
        raise ValueError("blind manifest is empty")


def validate_checkpoint_seal(checkpoint, manifest_sha256):
    args = checkpoint.get("args")
    args = (
        args
        if isinstance(args, dict)
        else vars(args)
        if args is not None
        else {}
    )
    dual = bool(args.get("dual_view") or checkpoint.get("dual_view_config"))
    if dual and args.get("dual_blind_manifest_sha256") != manifest_sha256:
        raise ValueError(
            "run B checkpoint did not seal this blind manifest before training"
        )
    return dual


def validate_sample(row, sample):
    if sample.get("query_target_path") != row["target_path"]:
        raise ValueError(
            "blind target identity does not match fixed validation"
        )
    if sample.get("reference_target_path") != row["ref_path"]:
        raise ValueError(
            "blind reference identity does not match fixed validation"
        )


def compose_blind(panels, blind_id, cell=448):
    """Create a PIL RGB panel; image labels contain no model identity."""
    from PIL import Image, ImageDraw, ImageFont

    if len(panels) != 4 or cell < 16:
        raise ValueError("four panels and a readable cell size are required")
    header, gap = 36, 4
    canvas = Image.new("RGB", (4 * cell + 3 * gap, cell + header), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, (panel, label) in enumerate(
        zip(panels, ("Reference", "GT", "Left", "Right"))
    ):
        x = column * (cell + gap)
        canvas.paste(panel.convert("RGB").resize((cell, cell)), (x, header))
        draw.text((x + 4, 4), label, fill="black", font=font)
    draw.text((4, 20), blind_id, fill="black", font=font)
    return canvas


def tensor_image(value):
    """ImageNet tensor to RGB without reading labels or model metadata."""
    import torch
    from PIL import Image

    if not torch.isfinite(value).all():
        raise ValueError("nonfinite blind prediction or GT image")
    mean = value.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = value.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    rgb = ((value.float() * std + mean).clamp(0, 1) * 255).round()
    array = rgb.to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def write_rating_template(path, comparisons):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("blind_id", "structure_preference", "pen_preference"),
        )
        writer.writeheader()
        for row in comparisons:
            writer.writerow({"blind_id": row["blind_id"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--val-json-path", nargs="+", required=True)
    parser.add_argument("--fixed-pair-path", required=True)
    parser.add_argument("--target-cache", required=True)
    parser.add_argument("--eval-coef", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cell", type=int, default=448)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(
            "blind output and temporary directories must be new"
        )
    import torch

    from tools.eval_s_baseline import (
        build_dataset,
        eval_metadata,
        load_model,
        validate_eval_coefficients,
    )

    mapped = argparse.Namespace(
        data_path=args.data_path,
        val_json_path=args.val_json_path,
        fixed_pair_path=args.fixed_pair_path,
        dual_target_cache=args.target_cache,
        j_definition=SCHEMA,
    )
    metadata = eval_metadata(mapped)
    validate_eval_coefficients(read_json(args.eval_coef), metadata)
    manifest = read_json(args.manifest)
    models = parse_models(args.models)
    validate_manifest(manifest, metadata, models)
    manifest_hash = sha256_file(args.manifest)
    checkpoints, dual_models = {}, []
    for name, path in models.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if validate_checkpoint_seal(checkpoint, manifest_hash):
            dual_models.append(name)
        checkpoints[name] = {
            "checkpoint_sha256": sha256_file(path),
            "epoch": checkpoint.get("epoch"),
        }
    if not dual_models:
        raise ValueError(
            "blind comparison must include the sealed run B model"
        )
    dataset = build_dataset(mapped, definition=SCHEMA)
    predictions = {}
    identities = {row["index"]: row for row in manifest["comparisons"]}
    # One model on the GPU at a time keeps the export within inference memory.
    for name, path in models.items():
        seed_all(args.seed)
        model = load_model(path).to(args.device).eval()
        with torch.no_grad():
            for index, row in identities.items():
                sample = dataset[index]
                validate_sample(row, sample)
                image = sample["image"][None].to(args.device)
                target = sample["target"][None].to(args.device)
                prediction = model.predict_full(image, target)
                half = prediction.shape[-2] // 2
                predictions[(name, index)] = tensor_image(
                    prediction[0, :, half:]
                )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        image_dir = temporary / "images"
        image_dir.mkdir()
        for row in manifest["comparisons"]:
            sample = dataset[row["index"]]
            validate_sample(row, sample)
            target = sample["target"]
            half = target.shape[-2] // 2
            panels = [
                tensor_image(target[:, :half]),
                tensor_image(target[:, half:]),
                predictions[(row["left_model"], row["index"])],
                predictions[(row["right_model"], row["index"])],
            ]
            compose_blind(panels, row["blind_id"], args.cell).save(
                image_dir / (row["blind_id"] + ".png")
            )
        write_rating_template(
            temporary / "ratings.csv", manifest["comparisons"]
        )
        write_new_json(
            temporary / "answer_key.json",
            {
                "schema": SCHEMA,
                "purpose": "blind_answer_key",
                "manifest_sha256": manifest_hash,
                "models": checkpoints,
                "eval_coef_sha256": sha256_file(args.eval_coef),
                **metadata,
                "comparisons": manifest["comparisons"],
            },
        )
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "images": len(manifest["comparisons"]),
                "manifest_sha256": manifest_hash,
            }
        )
    )


if __name__ == "__main__":
    main()
