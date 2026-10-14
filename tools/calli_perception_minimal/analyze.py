"""Summarize frozen-protocol results and render unselected test examples."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from tools.calli_perception_minimal.probe import PerceptionHead, save_json
from util.calli_labels import class_names


def paired_character_bootstrap(values, baseline, characters):
    """Bootstrap paired per-character means, not dependent reference pairs."""
    delta = np.asarray(values) - np.asarray(baseline)
    groups = sorted(set(characters))
    grouped = np.array([delta[np.array(characters) == char].mean() for char in groups])
    rng = np.random.default_rng(17)
    draws = rng.choice(grouped, (2000, len(grouped)), replace=True).mean(1)
    return dict(
        characters=len(groups),
        mean_delta=float(grouped.mean()),
        ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    torch.set_num_threads(4)
    summary = json.loads((out / "summary.json").read_text())
    records = json.loads((out / "manifest.json").read_text())
    runs = [json.loads(p.read_text()) for p in sorted(out.glob("seed*_results.json"))]
    baseline = json.loads((out / "baselines.json").read_text())
    aggregate = {}
    for kind in ("BF", "JT"):
        aggregate[kind] = {}
        for split in ("train", "dev", "test", "shuffled_input_test"):
            aggregate[kind][split] = {}
            for metric in ("soft_dice", "all_channel_soft_iou", "binary_dice_at_0.5"):
                values = [r[split][kind][metric] for r in runs]
                aggregate[kind][split][metric] = dict(
                    mean=float(np.mean(values)),
                    sd=float(np.std(values, ddof=1)),
                    individual=values,
                )
        characters = [
            r["character"]
            for r in records
            if r["split"] == "test" and r["type"].endswith(kind)
        ]
        # Averaging seed predictions' scores is descriptive, not extra independent data.
        values = np.mean([r["test"][kind]["per_image_soft_dice"] for r in runs], axis=0)
        aggregate[kind]["paired_vs_author_prior"] = paired_character_bootstrap(
            values,
            baseline["author_mean_oracle"][kind]["per_image_soft_dice"],
            characters,
        )
        shuffled = np.mean(
            [r["shuffled_input_test"][kind]["per_image_soft_dice"] for r in runs],
            axis=0,
        )
        aggregate[kind]["paired_vs_shuffled"] = paired_character_bootstrap(
            values, shuffled, characters
        )
    save_json(out / "aggregate.json", aggregate)
    cache = torch.load(out / "cache.pt", map_location="cpu")
    model = PerceptionHead().eval()
    model.load_state_dict(
        torch.load(out / "seed0_best.pt", map_location="cpu")["model"]
    )
    test = [i for i, r in enumerate(records) if r["split"] == "test"]
    train = [i for i, r in enumerate(records) if r["split"] == "train"]
    with torch.no_grad():
        prob = torch.cat(
            [
                model(cache["features"][test[start : start + 16]].float()).sigmoid()
                for start in range(0, len(test), 16)
            ]
        )
    y = cache["targets"][test]
    kinds = cache["kinds"][test]
    channels = []
    for c, name in enumerate(class_names()):
        valid = (kinds == (0 if c < 50 else 1)) & (y[:, c].sum((1, 2)) > 0)
        p, t = prob[valid, c], y[valid, c]
        dice = 2 * (p * t).sum((1, 2)) / (p.sum((1, 2)) + t.sum((1, 2))).clamp_min(1e-8)
        channels.append(
            dict(
                channel=c,
                name=name,
                train_present=int((cache["targets"][train, c].sum((1, 2)) > 0).sum()),
                test_present=int(valid.sum()),
                test_soft_dice=dice.mean().item() if len(dice) else None,
            )
        )
    save_json(out / "per_class_seed0.json", channels)
    chosen = []
    for kind in (0, 1):
        chosen.extend([i for i in test if int(cache["kinds"][i]) == kind][:3])
    canvas = Image.new("RGB", (224 * 4, 254 * len(chosen)), "white")
    draw = ImageDraw.Draw(canvas)
    examples = []
    for row, i in enumerate(chosen):
        target = cache["targets"][i]
        c = int(target.sum((1, 2)).argmax())
        pred = prob[test.index(i), c]
        original = (cache["images"][i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        panels = [Image.fromarray(original)]
        for arr in (target[c], pred, (pred - target[c]).abs()):
            gray = (arr.clamp(0, 1).numpy() * 255).astype(np.uint8)
            panels.append(
                Image.fromarray(gray)
                .convert("RGB")
                .resize((224, 224), Image.Resampling.NEAREST)
            )
        for col, (label, panel) in enumerate(
            zip(("Input", "GT coverage", "Prediction", "Absolute error"), panels)
        ):
            canvas.paste(panel, (col * 224, row * 254 + 30))
            draw.text((col * 224 + 4, row * 254 + 5), f"{label} | ch {c}", fill="black")
        examples.append(
            dict(
                row=row + 1,
                character=records[i]["character"],
                author=records[i]["author"],
                class_name=class_names()[c],
            )
        )
    canvas.save(out / "examples.png")
    save_json(out / "example_labels.json", examples)
    print(
        json.dumps(
            dict(
                aggregate=aggregate,
                elapsed=summary["elapsed_seconds"],
                peak_mb=summary["peak_gpu_allocated_mb"],
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
