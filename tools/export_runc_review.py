"""Render P0/P1, actual masks and GT-only geometric diagnostics for runc."""

import argparse
import html
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from torch.nn import functional as functional

    from tools.structure_first import (
        MEAN,
        STD,
        build_dataset,
        call_model,
        collate_indices,
        load_model,
        metadata,
        sha256,
        write_json,
    )
    from util.structure_first import set_structure_stage
    from util.structure_first_loss import lowpass_target

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--training-masks", action="store_true")
    parser.add_argument("--all-jt-geometry", action="store_true")
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=[
            14,
            121,
            156,
            234,
            13,
            180,
            51,
            129,
            205,
            27,
            189,
            35,
        ],
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model, _ = load_model(config, device)
    data_config = deepcopy(config)
    if args.training_masks:
        data_config["stage"] = "refine"
        set_structure_stage(model, "refine")
    dataset = build_dataset(data_config, validation=not args.training_masks)
    indices = range(len(dataset)) if args.all_jt_geometry else args.indices
    model.eval()
    mean = torch.tensor(MEAN, device=device)[None, :, None, None]
    std = torch.tensor(STD, device=device)[None, :, None, None]

    def picture(value):
        rgb = (value.float() * std + mean)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(np.rint(rgb * 255).astype(np.uint8))

    def gray(value):
        value = value.detach().float().reshape(1, 1, *value.shape[-2:])
        value = functional.interpolate(value, (448, 448), mode="nearest")[0, 0]
        array = np.rint(value.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")

    rows = []
    with torch.no_grad():
        for index in indices:
            batch = collate_indices(dataset, [index], device, seed=0)
            if args.all_jt_geometry and not bool(batch["has_jt"][0]):
                continue
            outputs = call_model(
                model, batch, config["stage"], inference=not args.training_masks
            )
            p0, gt = outputs["p0"][:, :, 448:], batch["target"][:, :, 448:]
            head = outputs["head"]
            gates = (
                head["occupancy_logits"].float().sigmoid()
                * head["existence_logits"].float().sigmoid()[:, :, None, None]
            )
            filtered = lowpass_target(gt, config["lowpass"])
            panels = [
                ("computer target", picture(batch["image"][:, :, 448:])),
                ("visible style reference", picture(batch["target"][:, :, :448])),
                ("GT: comparison only", picture(gt)),
                ("Gaussian GT: loss only", picture(filtered)),
                ("P0", picture(p0)),
                (
                    "P0 foreground (J definition)",
                    gray(1 - model.dual_jieti_loss._soft_fg(p0)),
                ),
                ("predicted S: occupancy * existence", gray(gates.amax(1))),
            ]
            if bool(batch["has_jt"][0]):
                gt_jt = batch["instance_occupancy"][0].amax(0)
                panels.append(("GT JT: diagnostic only", gray(gt_jt)))
            else:
                missing = Image.new("RGB", (448, 448), "white")
                ImageDraw.Draw(missing).text(
                    (24, 200), "No JT annotation", fill="black"
                )
                panels.append(("GT JT unavailable", missing))
            row = {
                "index": index,
                "target_path": batch["target_path"][0],
                "reference_path": batch["reference_path"][0],
                "has_jt": bool(batch["has_jt"][0]),
                "gt_supplied_to_inference": False,
                "instance_geometry_448px": [],
            }
            geometry = picture(p0)
            draw = ImageDraw.Draw(geometry)
            if row["has_jt"]:
                loss = model.dual_jieti_loss
                part = torch.stack(
                    [(batch["voro"] == slot + 1).float() for slot in range(4)], 1
                )
                pred_fg = functional.adaptive_avg_pool2d(
                    loss._soft_fg(p0)[:, None], (loss.pool, loss.pool)
                )[:, 0]
                true_fg = functional.adaptive_avg_pool2d(
                    loss._soft_fg(gt)[:, None], (loss.pool, loss.pool)
                )[:, 0]
                pred_stats, true_stats = (
                    loss._part_stats(x, part) for x in (pred_fg, true_fg)
                )
                for slot in range(4):
                    if not bool(batch["valid_parts"][0, slot]):
                        continue
                    record = {"instance": slot + 1}
                    for label, stats, color in (
                        ("P0", pred_stats, "red"),
                        ("GT", true_stats, "green"),
                    ):
                        mass, cx, cy, sx, sy = [float(x[0, slot]) for x in stats]
                        factor = 448 / loss.pool
                        cx, cy, sx, sy = [x * factor for x in (cx, cy, sx, sy)]
                        record[label] = {
                            "mass": mass,
                            "cx": cx,
                            "cy": cy,
                            "sigma_x": sx,
                            "sigma_y": sy,
                        }
                        draw.ellipse(
                            (cx - sx, cy - sy, cx + sx, cy + sy), outline=color, width=2
                        )
                        draw.text((cx, cy), str(slot + 1), fill=color)
                    record["mass_shielded"] = (
                        record["P0"]["mass"]
                        < loss.pred_mass_ratio * record["GT"]["mass"]
                    )
                    row["instance_geometry_448px"].append(record)
            panels.append(("GT-Voronoi moments: P0 red / GT green", geometry))
            if "p1" in outputs:
                panels.extend(
                    [
                        ("P1", picture(outputs["p1"][:, :, 448:])),
                        ("actual masked P0", picture(outputs["masked_p0"][:, :, 448:])),
                        ("actual reconstruction mask", gray(outputs["mask"][0])),
                    ]
                )
                row["mask_coverage"] = float(outputs["mask"].float().mean())
            if "bf_mask_stats" in batch:
                row["bf_mask_stats"] = {
                    key: value.detach().cpu().tolist()
                    for key, value in batch["bf_mask_stats"].items()
                }
            canvas = Image.new(
                "RGB", (448 * 4, 480 * ((len(panels) + 3) // 4)), "white"
            )
            draw = ImageDraw.Draw(canvas)
            for number, (label, panel) in enumerate(panels):
                x, y = number % 4 * 448, number // 4 * 480
                draw.text((x + 8, y + 8), label, fill="black")
                canvas.paste(panel, (x, y + 32))
            name = f"{index:03d}.png"
            if not args.all_jt_geometry:
                canvas.save(output / name)
            row["png"] = None if args.all_jt_geometry else name
            rows.append(row)
    write_json(
        output / "review.json",
        {
            "metadata": metadata(config),
            "source_sha256": sha256(__file__),
            "training_mask_diagnostic": args.training_masks,
            "geometry_limitation": (
                "GT Voronoi moments are diagnostics, "
                "not instance segmentation predictions"
            ),
            "rows": rows,
        },
    )
    entries = "\n".join(
        f"<h2>{r['index']}: {html.escape(r['target_path'])}</h2>"
        f'<a href="{r["png"]}"><img width="100%" src="{r["png"]}"></a>'
        for r in rows if r["png"] is not None
    )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>runc review</title>'
        "<style>body{font-family:sans-serif;max-width:1792px;margin:24px auto}</style>"
        "<h1>runc: model outputs and GT-only geometric diagnostics</h1>" + entries,
        encoding="utf-8",
    )
    write_json(
        output / "sha256.json",
        {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    )
    print(json.dumps({"count": len(rows), "output": str(output)}))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    main()
