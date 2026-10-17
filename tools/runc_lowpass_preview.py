"""Export fixed real-data Gaussian targets for human approval before runc."""

import argparse
import html
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def blur_rgb(tensor, sigma):
    """Apply the proposed seven-pixel Gaussian in RGB with reflect padding."""
    import torch
    from torch.nn import functional as functional

    axis = torch.arange(-3, 4, dtype=tensor.dtype, device=tensor.device)
    kernel = torch.exp(-axis.square() / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :])[None, None]
    kernel = kernel.expand(3, 1, 7, 7)
    return functional.conv2d(
        functional.pad(tensor[None], (3, 3, 3, 3), mode="reflect"),
        kernel, groups=3,
    )[0]


def main():
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from scipy import ndimage

    from tools.eval_s_baseline import DUAL_SCHEMA, build_dataset, file_sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--target-cache", required=True)
    parser.add_argument("--val-json-path", nargs="+", required=True)
    parser.add_argument("--fixed-pair-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    options = SimpleNamespace(
        data_path=args.data_path, dual_target_cache=args.target_cache,
        val_json_path=args.val_json_path, fixed_pair_path=args.fixed_pair_path,
        j_definition=DUAL_SCHEMA,
    )
    dataset = build_dataset(options)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    candidates = []
    samples = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        rgb = (sample["target"][:, 448:] * std + mean).clamp(0, 1)
        gray = rgb.mean(0).numpy()
        foreground = gray < 0.5
        distance = ndimage.distance_transform_edt(foreground)
        thin = foreground & (distance <= 1.25)
        mass = int(foreground.sum())
        if mass:
            score = float(thin.sum() / mass)
            candidates.append((not bool(sample["has_jt"]), score, index))
        samples[index] = (sample, rgb, foreground, distance)
    selected = []
    seen = set()
    # Include historically inspected JT examples, then thin BF/JT candidates.
    for index in (14, 121, 156, 234):
        if index in samples and bool(samples[index][0]["has_jt"]):
            selected.append((index, "historically_inspected_JT"))
            seen.add(samples[index][0]["target_path"])
    selected_writers = {Path(path).parts[-3][:-2] for path in seen}
    for want_bf, count in ((True, 6), (False, 2)):
        eligible = sorted(
            (r for r in candidates if r[0] == want_bf),
            key=lambda row: (-row[1], row[2]),
        )
        added = 0
        writers = set() if want_bf else set(selected_writers)
        for _, _, index in eligible:
            key = samples[index][0]["target_path"]
            writer = Path(key).parts[-3][:-2]
            if key in seen or writer in writers:
                continue
            selected.append((index, "thin_fraction_rank"))
            seen.add(key)
            writers.add(writer)
            added += 1
            if added == count:
                break
    rows = []
    thumbs = []
    sigmas = (1.0, 1.5, 2.0)
    for index, reason in selected:
        sample, rgb, foreground, distance = samples[index]
        versions = [rgb] + [blur_rgb(rgb, sigma) for sigma in sigmas]
        arrays = [value.permute(1, 2, 0).numpy() for value in versions]
        # Stable 96px crop maximizing genuinely thin foreground support.
        thin_support = (foreground & (distance <= 1.25)).astype(float)
        score = ndimage.uniform_filter(thin_support, size=96, mode="constant")
        y, x = np.unravel_index(score.argmax(), score.shape)
        left = int(max(0, min(352, x - 48)))
        top = int(max(0, min(352, y - 48)))
        canvas = Image.new("RGB", (448 * 4, 448 + 32 + 192 + 24), "white")
        draw = ImageDraw.Draw(canvas)
        labels = ["Original GT", "sigma=1.0", "sigma=1.5 (initial)", "sigma=2.0"]
        measurements = []
        original_ink = foreground
        for column, (array, label) in enumerate(zip(arrays, labels)):
            picture = Image.fromarray(np.rint(array * 255).astype(np.uint8))
            canvas.paste(picture, (column * 448, 32))
            crop = picture.crop((left, top, left + 96, top + 96)).resize(
                (192, 192), Image.Resampling.NEAREST,
            )
            canvas.paste(crop, (column * 448 + 128, 480))
            draw.text((column * 448 + 8, 10), label, fill="black")
            draw.text((column * 448 + 8, 676),
                      "same 96px crop, nearest 2x", fill="black")
            ink = array.mean(-1) < 0.5
            _, components = ndimage.label(ink, np.ones((3, 3)))
            measurements.append({
                "sigma": None if column == 0 else sigmas[column - 1],
                "foreground_pixels_at_0_5": int(ink.sum()),
                "components_8_connected_at_0_5": int(components),
                "lost_original_ink_pixels_at_0_5": int((original_ink & ~ink).sum()),
                "added_ink_pixels_at_0_5": int((~original_ink & ink).sum()),
            })
        filename = f"{index:03d}_{'JT' if bool(sample['has_jt']) else 'BF'}.png"
        canvas.save(output / filename)
        thumb = canvas.crop((0, 0, 1792, 480)).resize((896, 240))
        thumbs.append((index, thumb))
        target = Path(args.data_path) / sample["target_path"]
        rows.append({
            "fixed_val_index": index, "kind": "JT" if bool(sample["has_jt"]) else "BF",
            "target_path": sample["target_path"], "target_sha256": file_sha256(target),
            "selection": reason, "zoom_xywh": [left, top, 96, 96],
            "png": filename, "measurements": measurements,
        })
    for start in range(0, len(thumbs), 6):
        page = Image.new("RGB", (896, 264 * len(thumbs[start:start + 6])), "white")
        draw = ImageDraw.Draw(page)
        for line, (index, thumb) in enumerate(thumbs[start:start + 6]):
            draw.text((8, line * 264 + 5), f"Fixed val item {index}", fill="black")
            page.paste(thumb, (0, line * 264 + 24))
        page.save(output / f"contact_{start // 6 + 1}.png")
    payload = {
        "experiment": "runc", "status": "awaiting_user_lowpass_approval",
        "kernel_size": 7, "sigmas": list(sigmas), "padding": "reflect",
        "geometry": "fixed clean val identity 448x448 lower target",
        "fixed_pair_sha256": file_sha256(args.fixed_pair_path),
        "selection_is_diagnostic_not_random_or_generalization_test": True,
        "metrics_are_threshold_proxies_not_topology_ground_truth": True,
        "rows": rows,
    }
    payload["script_sha256"] = file_sha256(__file__)
    (output / "preview.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    entries = "\n".join(
        f'<h2>{r["fixed_val_index"]} / {r["kind"]} / '
        f'{html.escape(r["target_path"])}</h2>'
        f'<a href="{r["png"]}"><img width="100%" src="{r["png"]}"></a>'
        for r in rows
    )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8">'
        '<title>runc Gaussian target review</title>'
        '<style>body{font-family:sans-serif;max-width:1800px;margin:24px auto}'
        'h2{font-size:16px}</style>'
        '<h1>runc: GT | sigma 1.0 | sigma 1.5 | sigma 2.0</h1>'
        '<p>Kernel 7, reflect padding. Top: full 448px glyph. '
        'Bottom: same 96px crop at 2x nearest. Diagnostic selection; '
        'target review before training.</p>' + entries,
        encoding="utf-8",
    )
    files = {p.name: file_sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "sha256.json").write_text(json.dumps(files, indent=2) + "\n")
    print(json.dumps({"count": len(rows), "output": str(output),
                      "status": payload["status"]}))


if __name__ == "__main__":
    main()
