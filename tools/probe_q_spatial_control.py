"""Frozen Run B Fix: paired Q/G/F interventions, with no optimizer or training.

This executable uses two independent GPU shards, not DDP. Ground truth enters
only post-hoc metrics and the explicitly isolated AO diagnostic branch.
"""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.dual_view_protocol import seed_all  # noqa: E402
from tools.eval_s_baseline import (  # noqa: E402
    DUAL_SCHEMA,
    build_dataset,
    eval_metadata,
    file_sha256,
    load_model,
    make_jieti_loss,
    validate_eval_coefficients,
    validate_s_baseline,
)
from tools.evaluate_q_spatial_probe import (  # noqa: E402
    aggregate,
    dump_json,
    quality,
    quality_context,
    read_csv,
    response_metrics,
    write_csv,
)
from util.q_layout_warp import (  # noqa: E402
    compose_g,
    intervention,
    predicted_supports,
    slot_validity,
    support_diagnostics,
)

CHECKPOINT_SHA = "57ac8276e3468ad92ce2aff033736128b33862a4f922618c51910717166ae852"
RATIOS = (0.75, 0.90, 1.00, 1.10, 1.25, 1.50)
GROUPS = ("A1", "A2", "A3", "A4", "A5")


def state_sha(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def image_rgb(tensor):
    value = tensor.detach().cpu().float().permute(1, 2, 0).numpy()
    value = value * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    return Image.fromarray(np.clip(value * 255, 0, 255).astype(np.uint8))


def mass_image(field):
    values = field.detach().cpu().numpy() if torch.is_tensor(field) else field
    return Image.fromarray(
        np.clip((1 - values) * 255, 0, 255).astype(np.uint8)
    ).convert("RGB")


def montage(rows, path):
    """Same 224px display scale, six columns; labels use portable ASCII."""
    canvas = Image.new("RGB", (224 * 6, len(rows) * 248), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    for y, panels in enumerate(rows):
        for x, (label, image) in enumerate(panels):
            draw.text((224 * x + 4, y * 248 + 4), label, fill="black")
            canvas.paste(image.resize((224, 224)), (224 * x, y * 248 + 22))
    canvas.save(path)


def foreground(jmod, prediction):
    return (
        F.adaptive_avg_pool2d(
            jmod._soft_fg(prediction[:, :, 448:])[:, None], (224, 224)
        )[0, 0]
        .cpu()
        .numpy()
    )


def score(jmod, prediction, target, sample):
    area = sample["valid"][None, :, 448:].to(prediction.device)
    l1 = ((prediction[:, :, 448:] - target[:, :, 448:]).abs() * area).sum()
    result = {"L1": float(l1 / (area.sum() + 1e-2)), "J": None}
    if bool(sample["has_jt"]):
        value, parts, shield = jmod(
            prediction,
            target,
            sample["voro"][None].to(prediction.device),
            sample["valid_parts"][None].to(prediction.device),
            sample["has_jt"][None].to(prediction.device),
        )
        result.update(J=float(value), shielded_parts=int(shield))
        result.update({k: float(parts[k]) for k in ("centroid", "logsigma", "shape")})
    for value in result.values():
        if value is not None and not math.isfinite(value):
            raise FloatingPointError("Nonfinite image metric")
    return result


def decode(model, taps, lower, residual):
    """Replace only the lower half of the four taps, then use native decoder."""
    sizes = [x.shape[-1] for x in taps]
    chunks = lower.split(sizes, -1)
    changed_taps = []
    for original, chunk in zip(taps, chunks):
        half = original.shape[1] // 2
        changed = torch.cat((original[:, :half], chunk), 1)
        if not torch.equal(changed[:, :half], original[:, :half]):
            raise AssertionError("Upper reference tokens changed")
        changed_taps.append(changed)
    return model.forward_decoder(changed_taps, lower_residual=residual)


def wrong_support(support):
    """Equal-area displaced region; roll is a deterministic negative control."""
    height, width = support.shape[-2:]
    offsets = ((height // 2, width // 2), (0, width // 2), (height // 2, 0))
    candidates = [torch.roll(support, xy, (-2, -1)) for xy in offsets]
    overlap = [float(torch.minimum(support, c).sum()) for c in candidates]
    selected = candidates[int(np.argmin(overlap))]
    torch.testing.assert_close(selected.sum(), support.sum(), rtol=1e-6, atol=1e-5)
    return selected, min(overlap) / max(float(support.sum()), 1e-12)


def worker(args):
    seed_all(0)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    output = args.output
    shard = output / f"shard_{args.rank}"
    shard.mkdir()
    mapped = argparse.Namespace(
        data_path=str(ROOT / "fontdata_example"),
        val_json_path=sorted(
            (ROOT / "fontdata_example/val_json_clean_v2").glob("*.json")
        ),
        fixed_pair_path=str(ROOT / "fontdata_example/val_pairs_fixed_clean.json"),
        dual_target_cache=str(ROOT / "fontdata_example/dual_targets_v1"),
        j_definition=DUAL_SCHEMA,
    )
    sealed = json.loads(args.reference_eval.read_text())
    if file_sha256(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Only sealed original-edge Run B Fix ep50 is permitted")
    metadata = eval_metadata(mapped)
    for key, value in metadata.items():
        if sealed[key] != value:
            raise ValueError("Stale evaluation reference: " + key)
    if sealed["source_checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("Wrong reference evaluation checkpoint")
    coef_file = args.prep / "eval_coef.json"
    coefficients = validate_eval_coefficients(
        json.loads(coef_file.read_text()), metadata
    )
    if file_sha256(coef_file) != sealed["eval_coef_sha256"]:
        raise ValueError("Different evaluation coefficients")
    dataset = build_dataset(mapped)
    model = load_model(args.checkpoint).to(device).float().eval().requires_grad_(False)
    if hasattr(model, "g_control") or getattr(model, "tip_weight", None) is not None:
        raise ValueError("G Scale / Tip models are excluded")
    if model._checkpoint_epoch != 50:
        raise ValueError("Wrong source epoch")
    before_state = state_sha(model)
    jmod = make_jieti_loss(DUAL_SCHEMA, coefficients).to(device)
    candidates = set(
        json.loads(args.candidate_summary.read_text())["top20_source_GT_difference"]
    )
    records, instances, selections, blind_keys, blind_scores = [], [], [], [], []
    checks = {
        "rank": args.rank,
        "a0_pixel_exact": 0,
        "identity_pixel_exact": 0,
        "upstream_fixed": 0,
        "gt_leak_checks": 0,
        "sealed_backend_matches": 0,
        "items": 0,
        "checkpoint_before": CHECKPOINT_SHA,
        "state_before": before_state,
    }
    started = time.time()
    with torch.no_grad():
        for index in range(len(dataset)):
            if index % args.shards != args.rank or (args.limit and index >= args.limit):
                continue
            sample = dataset[index]
            image = sample["image"][None].to(device)
            target = sample["target"][None].to(device)
            # Reconcile the sealed cuDNN backend separately, then conduct all
            # paired interventions in strict FP32 with both TF32 flags False.
            torch.backends.cudnn.allow_tf32 = True
            original_backend = model.predict_full(image, target)
            original_metrics = score(jmod, original_backend, target, sample)
            torch.backends.cudnn.allow_tf32 = False
            mask = model.full_target_mask(image)
            taps, quadrants = model.forward_encoder(
                image, target, mask, return_quadrants=True
            )
            full_features = torch.cat(taps, -1)
            lower = full_features[:, full_features.shape[1] // 2 :]
            head = model.dual_head(quadrants, lower)
            checks["tensor_shapes"] = {
                "lower": list(lower.shape),
                "query": list(head["query"].shape),
                "layout": list(head["layout"].shape),
                "occupancy": list(head["occupancy_logits"].shape),
            }
            torch.testing.assert_close(
                compose_g(head, head["occupancy_logits"].sigmoid()),
                head["layout"],
                rtol=0,
                atol=0,
            )
            snapshots = {k: v.clone() for k, v in head.items()}
            residual = model.dual_fusion(head["layout"])
            baseline = decode(model, taps, lower, residual)
            canonical = model.predict_full(image, target)
            torch.testing.assert_close(baseline, canonical, rtol=0, atol=0)
            checks["a0_pixel_exact"] += 1
            if checks["gt_leak_checks"] < 2:
                changed_target = target.clone()
                changed_target[:, :, 448:] = (
                    torch.randn_like(changed_target[:, :, 448:]) * 7
                )
                torch.testing.assert_close(
                    model.predict_full(image, changed_target), baseline, rtol=0, atol=0
                )
                checks["gt_leak_checks"] += 1
            jt = bool(sample["has_jt"])
            path = sample["target_path"]
            folder = Path(path).parent.parent.name
            author = folder[:-2] if folder.endswith(("BF", "JT")) else folder
            glyph_key = author + ":" + Path(path).stem
            base_record = {
                "index": index,
                "kind": "JT" if jt else "BF",
                "target_path": path,
                "reference_path": sample["reference_path"],
                "glyph_key": glyph_key,
                "candidate20": index in candidates,
            }
            gt_fg = foreground(jmod, target)
            base_fg = foreground(jmod, baseline)
            global_context = quality_context(gt_fg)
            base_metrics = score(jmod, baseline, target, sample)
            reference = sealed["per_item"][index]
            if (path, sample["reference_path"], jt) != (
                reference["target_path"],
                reference["ref_path"],
                reference["is_jt"],
            ):
                raise AssertionError("Fixed pairing changed")
            for key in ("L1", "J"):
                if original_metrics[key] is not None and not math.isclose(
                    original_metrics[key], reference[key], rel_tol=1e-6, abs_tol=1e-7
                ):
                    raise AssertionError(f"A0 reference mismatch {index} {key}")
            checks["sealed_backend_matches"] += 1
            records.append(
                {
                    **base_record,
                    "group": "A0",
                    "slot": -1,
                    "axis": "none",
                    "requested_ratio": 1,
                    **base_metrics,
                    "sealed_backend_L1": original_metrics["L1"],
                    "sealed_backend_J": original_metrics["J"],
                    "strict_vs_sealed_pixel_max": float(
                        (baseline - original_backend).abs().max()
                    ),
                    **quality(base_fg, global_context),
                }
            )
            montage(
                [
                    [
                        ("Source", image_rgb(image[0, :, 448:])),
                        ("Style reference", image_rgb(target[0, :, :448])),
                        ("A0", image_rgb(baseline[0, :, 448:])),
                        ("GT: evaluation only", image_rgb(target[0, :, 448:])),
                    ]
                ],
                output / "figures" / f"{index:03d}_baseline.png",
            )
            # Matching is post-hoc and never affects prediction-only selection.
            slot_to_gt = {}
            if jt:
                _, matching = model.dual_head_loss(
                    head,
                    sample["instance_occupancy"][None].to(device),
                    sample["instance_valid"][None].to(device),
                    sample["instance_classes"][None].to(device),
                    sample["has_jt"][None].to(device),
                )
                for slot in range(4):
                    if bool(matching["matched_slots"][0, slot]):
                        slot_to_gt[slot] = int(matching["assignment"][0, slot])
                for slot, gt_index in slot_to_gt.items():
                    region = sample["voro"].numpy() == gt_index + 1
                    instances.append(
                        {
                            **base_record,
                            "group": "A0",
                            "slot": slot,
                            "gt_instance": gt_index,
                            "axis": "none",
                            "requested_ratio": 1,
                            **quality(base_fg, quality_context(gt_fg, region)),
                        }
                    )
            supports = predicted_supports(head)
            valid, existence, area = slot_validity(head, supports)
            for slot in range(4):
                selections.append(
                    {
                        **base_record,
                        "slot": slot,
                        "existence": float(existence[0, slot]),
                        "area": float(area[0, slot]),
                        "prediction_valid": bool(valid[0, slot]),
                        "selected": jt and bool(valid[0, slot]),
                        "matched_gt": slot_to_gt.get(slot),
                        "reason": "BF_no_JT_intervention"
                        if not jt
                        else "valid"
                        if bool(valid[0, slot])
                        else "existence_below_0.5_or_area_below_1_or_nonfinite",
                    }
                )
            # AO is isolated and includes every GT instance, including slots
            # rejected by the prediction-only selection gate.
            controls = [
                (slot, None) for slot in range(4) if jt and bool(valid[0, slot])
            ]
            controls += [(slot, gt_index) for slot, gt_index in slot_to_gt.items()]
            for slot, oracle_gt in controls:
                oracle = oracle_gt is not None
                gt_index = oracle_gt if oracle else slot_to_gt.get(slot)
                support = (
                    sample["instance_occupancy"][gt_index][None].to(device).float()
                    if oracle
                    else supports[:, slot]
                )
                wrong, wrong_overlap = wrong_support(support)
                other = (supports.sum(1) - supports[:, slot]).clamp(0, 1)
                region = (
                    sample["voro"].numpy() == gt_index + 1
                    if gt_index is not None
                    else None
                )
                local_context = (
                    quality_context(gt_fg, region) if region is not None else None
                )
                for axis in (0, 1):
                    panels = {g: [] for g in (("AO",) if oracle else GROUPS)}
                    support_panels = []
                    for ratio in RATIOS:
                        moved, geometry = support_diagnostics(
                            support, other, axis, ratio
                        )
                        support_panels.append(
                            (f"support {ratio:.2f}", mass_image(moved[0]))
                        )
                        for group in panels:
                            chosen = wrong if group == "A5" else support
                            current_geometry = geometry
                            if group == "A5":
                                _, current_geometry = support_diagnostics(
                                    chosen, other, axis, ratio
                                )
                            changed_lower, changed_residual, g_slice, q_slice = (
                                intervention(
                                    model.dual_fusion,
                                    head,
                                    lower,
                                    chosen,
                                    slot,
                                    axis,
                                    ratio,
                                    group,
                                )
                            )
                            normalized = model.dual_fusion.norm(head["layout"])
                            if group in ("A1", "A3", "A4", "A5", "AO"):
                                torch.testing.assert_close(
                                    g_slice, normalized[..., :276], rtol=0, atol=0
                                )
                            if group in ("A2", "A3"):
                                torch.testing.assert_close(
                                    q_slice, normalized[..., 276:], rtol=0, atol=0
                                )
                            if group == "A3":
                                torch.testing.assert_close(
                                    changed_residual, residual, rtol=0, atol=0
                                )
                            prediction = decode(
                                model, taps, changed_lower, changed_residual
                            )
                            if ratio == 1:
                                torch.testing.assert_close(
                                    prediction, baseline, rtol=0, atol=0
                                )
                                checks["identity_pixel_exact"] += 1
                            field = foreground(jmod, prediction)
                            meta = {
                                **base_record,
                                "group": group,
                                "slot": slot,
                                "axis": "xy"[axis],
                                "requested_ratio": ratio,
                                "oracle": oracle,
                                **current_geometry,
                                "wrong_support_overlap": wrong_overlap
                                if group == "A5"
                                else None,
                            }
                            row = {
                                **meta,
                                **score(jmod, prediction, target, sample),
                                **quality(field, global_context),
                                "Q_delta_l2": float(
                                    (q_slice - normalized[..., 276:]).norm()
                                ),
                                "G_delta_l2": float(
                                    (g_slice - normalized[..., :276]).norm()
                                ),
                                "F_delta_l2": float((changed_lower - lower).norm()),
                                "residual_delta_l2": float(
                                    (changed_residual - residual).norm()
                                ),
                                "output_abs_delta": float(
                                    (prediction[:, :, 448:] - baseline[:, :, 448:])
                                    .abs()
                                    .mean()
                                ),
                                "rgb_unclamped_min": float(
                                    (
                                        prediction[:, :, 448:]
                                        * prediction.new_tensor([0.229, 0.224, 0.225])[
                                            None, :, None, None
                                        ]
                                        + prediction.new_tensor([0.485, 0.456, 0.406])[
                                            None, :, None, None
                                        ]
                                    ).min()
                                ),
                                "rgb_unclamped_max": float(
                                    (
                                        prediction[:, :, 448:]
                                        * prediction.new_tensor([0.229, 0.224, 0.225])[
                                            None, :, None, None
                                        ]
                                        + prediction.new_tensor([0.485, 0.456, 0.406])[
                                            None, :, None, None
                                        ]
                                    ).max()
                                ),
                            }
                            records.append(row)
                            if region is not None:
                                instances.append(
                                    {
                                        **meta,
                                        "gt_instance": gt_index,
                                        **response_metrics(
                                            field, base_fg, region, gt_fg, axis
                                        ),
                                        **quality(field, local_context),
                                    }
                                )
                            picture = image_rgb(prediction[0, :, 448:])
                            panels[group].append((f"{group} r={ratio:.2f}", picture))
                            if (
                                index in candidates
                                and not oracle
                                and group in ("A1", "A3", "A4")
                                and ratio in (0.75, 1.5)
                            ):
                                blind_id = (
                                    f"{index:03d}_{slot}_{axis}_{group}_{ratio:.2f}"
                                )
                                hidden = hashlib.sha256(blind_id.encode()).hexdigest()[
                                    :16
                                ]
                                order = random.Random(hidden).randint(0, 1)
                                pair = [image_rgb(baseline[0, :, 448:]), picture]
                                if order:
                                    pair.reverse()
                                blind = Image.new("RGB", (896, 468), "white")
                                d = ImageDraw.Draw(blind)
                                for position, im in enumerate(pair):
                                    d.text(
                                        (448 * position + 4, 3),
                                        "Left" if position == 0 else "Right",
                                        fill="black",
                                    )
                                    blind.paste(im, (448 * position, 20))
                                blind.save(
                                    output / "figures" / ("blind_" + hidden + ".png")
                                )
                                blind_keys.append(
                                    {
                                        "id": hidden,
                                        "index": index,
                                        "slot": slot,
                                        "axis": "xy"[axis],
                                        "group": group,
                                        "ratio": ratio,
                                        "baseline_side": "right" if order else "left",
                                    }
                                )
                                blind_scores.append(
                                    {
                                        "id": hidden,
                                        "tip_clarity_left_1to5": "",
                                        "tip_clarity_right_1to5": "",
                                        "breaks_left": "",
                                        "breaks_right": "",
                                        "ghosting_left": "",
                                        "ghosting_right": "",
                                    }
                                )
                    header = [
                        ("A0", image_rgb(baseline[0, :, 448:])),
                        ("GT: score only", image_rgb(target[0, :, 448:])),
                        ("selected soft support", mass_image(support[0])),
                        ("wrong equal-area support", mass_image(wrong[0])),
                    ]
                    montage(
                        [header, support_panels] + list(panels.values()),
                        output
                        / "figures"
                        / (
                            f"{index:03d}_{'AO' if oracle else 'pred'}"
                            f"_s{slot}_{'xy'[axis]}.png"
                        ),
                    )
            for key in head:
                torch.testing.assert_close(head[key], snapshots[key], rtol=0, atol=0)
            checks["upstream_fixed"] += 1
            checks["items"] += 1
            dump_json(
                shard / "status.json",
                {
                    "status": "running",
                    "completed": checks["items"],
                    "last_index": index,
                    "seconds": time.time() - started,
                    "interventions": len(records),
                },
            )
            print(
                f"rank={args.rank} index={index} items={checks['items']} "
                f"conditions={len(records)} sec={time.time() - started:.1f}",
                flush=True,
            )
    checks["state_after"] = state_sha(model)
    checks["checkpoint_after"] = file_sha256(args.checkpoint)
    if (
        before_state != checks["state_after"]
        or CHECKPOINT_SHA != checks["checkpoint_after"]
    ):
        raise AssertionError("Frozen model/checkpoint changed")
    for name, values in (
        ("per_item", records),
        ("per_instance", instances),
        ("slot_selection", selections),
        ("blind_scores", blind_scores),
    ):
        write_csv(shard / (name + ".csv"), values)
    dump_json(shard / "blind_key.json", blind_keys)
    dump_json(shard / "verification.json", checks)
    dump_json(
        shard / "status.json",
        {
            "status": "complete",
            "items": checks["items"],
            "seconds": time.time() - started,
        },
    )


def launch(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if file_sha256(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("Wrong checkpoint")
    args.output.mkdir(parents=True)
    (args.output / "figures").mkdir()
    den = json.loads((args.prep / "s_baseline_dual.json").read_text())
    reference = json.loads(args.reference_eval.read_text())
    metadata = {
        k: reference[k]
        for k in (
            "fixed_pair_sha256",
            "val_manifest_sha256",
            "target_cache_sha256",
            "eval_coef_sha256",
        )
    }
    validate_s_baseline(den, metadata)
    dump_json(
        args.output / "config.json",
        {
            "ratios": RATIOS,
            "groups": ["A0", *GROUPS, "AO"],
            "seed": 0,
            "precision": "FP32_TF32_disabled",
            "sealed_reconciliation": (
                "Original backend FP32 tensors/cuDNN TF32=True reconciles "
                "separately per item. All paired A0-AO interventions use strict "
                "FP32 with both TF32 flags False; report precision deltas."
            ),
            "s_denominator": {k: den[k] for k in ("L1_JT", "L1_BF", "J")},
            "checkpoint_sha256": CHECKPOINT_SHA,
            "A1_location": (
                "joint LayerNorm output Q slice[276:532], before Linear; "
                "normalized G frozen"
            ),
            "A2_location": (
                "one raw occupancy; recompute LN G slice only; normalized Q frozen"
            ),
            "A3_location": (
                "four lower taps concatenated, before residual addition/"
                "decoder_embed; residual frozen"
            ),
            "warp": (
                "inverse bilinear, zero extension, soft removal/deposition, "
                "overlap normalization, identity bypass"
            ),
            "selection": (
                "prediction existence>=0.5 and soft area>=1 grid cell; "
                "no GT matching filter"
            ),
            "A5": (
                "equal-area toroidal displacement chosen from three half-grid "
                "offsets by lowest overlap"
            ),
            "AO": (
                "GT occupancy specifies Q warp region only; "
                "privileged sensitivity, not inference"
            ),
            "limit": args.limit,
        },
    )
    provenance = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "reference_eval_sha256": file_sha256(args.reference_eval),
        **metadata,
        "candidate_summary_sha256": file_sha256(args.candidate_summary),
        "source_sha256": {
            p: file_sha256(ROOT / p)
            for p in (
                "tools/probe_q_spatial_control.py",
                "tools/evaluate_q_spatial_probe.py",
                "util/q_layout_warp.py",
            )
        },
    }
    dump_json(args.output / "provenance.json", provenance)
    processes, streams = [], []
    devices = args.devices.split(",")
    for rank, device in enumerate(devices):
        command = [
            sys.executable,
            str(Path(__file__)),
            "--worker",
            "--rank",
            str(rank),
            "--shards",
            str(len(devices)),
            "--checkpoint",
            str(args.checkpoint),
            "--reference-eval",
            str(args.reference_eval),
            "--prep",
            str(args.prep),
            "--candidate-summary",
            str(args.candidate_summary),
            "--output",
            str(args.output),
        ]
        if args.limit:
            command += ["--limit", str(args.limit)]
        stream = (args.output / f"worker_{rank}.log").open("w")
        streams.append(stream)
        processes.append(
            subprocess.Popen(
                command,
                cwd=ROOT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": device},
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        )
    codes = [p.wait() for p in processes]
    for stream in streams:
        stream.close()
    if any(codes):
        dump_json(args.output / "status.json", {"status": "failed", "codes": codes})
        raise RuntimeError(f"Probe worker failed: {codes}")
    if args.limit:
        dump_json(
            args.output / "status.json",
            {"status": "smoke_complete", "limit": args.limit},
        )
        return
    aggregate(args.output)
    scores = [
        r
        for s in sorted(args.output.glob("shard_*"))
        for r in read_csv(s / "blind_scores.csv")
    ]
    write_csv(args.output / "blind_scores.csv", scores)
    dump_json(
        args.output / "blind_key.json",
        [
            r
            for s in sorted(args.output.glob("shard_*"))
            for r in json.loads((s / "blind_key.json").read_text())
        ],
    )
    for shard in args.output.glob("shard_*"):
        shutil.rmtree(shard)
    for log in args.output.glob("worker_*.log"):
        log.unlink()
    dump_json(
        args.output / "status.json",
        {"status": "complete", "training_steps": 0, "manual_quality_review": "pending"},
    )
    paths = sorted(
        p
        for p in args.output.rglob("*")
        if p.is_file() and p.name != "sha256_manifest.json"
    )
    dump_json(
        args.output / "sha256_manifest.json",
        {str(p.relative_to(args.output)): file_sha256(p) for p in paths},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-eval", type=Path, required=True)
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    (worker if args.worker else launch)(args)


if __name__ == "__main__":
    main()
