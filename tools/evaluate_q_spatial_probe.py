"""Geometry/quality metrics and final aggregation for the frozen Q probe."""

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def dump_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sigma(field):
    """Soft sigma on 224-grid; return physical 448px units, or missing."""
    mass = float(field.sum())
    if mass <= 1e-6:
        return None
    yy, xx = np.indices(field.shape, dtype=np.float64)
    cx, cy = (field * xx).sum() / mass, (field * yy).sum() / mass
    return np.sqrt(
        [
            (field * (xx - cx) ** 2).sum() / mass,
            (field * (yy - cy) ** 2).sum() / mass,
        ]
    ) * (448 / field.shape[-1])


def fixed_roi(region, margin=8):
    """Baseline GT Voronoi intersection with GT ink is supplied by caller."""
    locations = np.argwhere(region)
    result = np.zeros_like(region, dtype=bool)
    if len(locations):
        low = np.maximum(0, locations.min(0) - margin)
        high = np.minimum(region.shape, locations.max(0) + margin + 1)
        result[low[0] : high[0], low[1] : high[1]] = True
    return result


def quality_context(gt, region=None):
    roi = np.ones_like(gt, dtype=bool) if region is None else region.astype(bool)
    gt_hard = gt >= 0.5
    gt_distance = ndimage.distance_transform_edt(gt_hard)
    skeleton = skeletonize(gt_hard)
    labels, _ = ndimage.label(gt_hard)
    sizes = np.bincount(labels.ravel())
    short = (sizes[labels] <= 32) & gt_hard
    return {
        "roi": roi,
        "gt": gt,
        "hard": gt_hard,
        "thin": skeleton & (gt_distance <= 1.5) & roi,
        "short": short & roi,
        "outside": (ndimage.distance_transform_edt(~gt_hard) > 2) & roi,
    }


def quality(field, context):
    """Evaluation proxies only; these do not replace manual tip assessment."""
    roi, gt = context["roi"], context["gt"]
    hard = field >= 0.5
    edges = np.hypot(ndimage.sobel(field, 0), ndimage.sobel(field, 1)) / 8
    labels, count = ndimage.label(hard & roi)
    skeleton = skeletonize(hard & roi)
    neighbours = ndimage.convolve(skeleton.astype(int), np.ones((3, 3), int)) - skeleton
    hp = field - ndimage.gaussian_filter(field, 1)
    gt_hp = gt - ndimage.gaussian_filter(gt, 1)
    result = {
        "edge_strength": float(edges[roi].mean()) if roi.any() else None,
        "highpass_l1": float(np.abs(hp - gt_hp)[roi].mean()) if roi.any() else None,
        "extra_ink_rate": float((hard & context["outside"]).sum() / max(roi.sum(), 1)),
        "components": int(count),
        "endpoints": int((skeleton & (neighbours == 1)).sum()),
        "ink_mass": float(field[roi].sum()),
    }
    for key in ("thin", "short"):
        mask = context[key]
        result[key + "_recall"] = float(hard[mask].mean()) if mask.any() else None
    del labels
    return result


def response_metrics(field, baseline, region, gt, axis):
    """Fixed GT partition + fixed ROI + hard threshold + neighbour sensitivity."""
    ink = (gt >= 0.5) & region
    roi = fixed_roi(ink)
    nearby = ndimage.binary_dilation(ink, iterations=8)
    result = {}
    for name, current, original, mask in (
        ("soft", field, baseline, region),
        ("hard", field >= 0.5, baseline >= 0.5, region),
        ("roi", field, baseline, roi),
    ):
        before, after = sigma(original * mask), sigma(current * mask)
        result[name + "_pixel_ratio"] = (
            float(after[axis] / before[axis])
            if before is not None and after is not None and before[axis] > 1e-6
            else None
        )
    result["neighbour_intensity_delta"] = (
        float((field - baseline)[nearby & ~region].mean())
        if (nearby & ~region).any()
        else None
    )
    result["local_abs_change"] = float(np.abs(field - baseline)[region].mean())
    return result


def numeric(row, key):
    value = row.get(key)
    return float(value) if value not in (None, "", "None") else None


def distribution(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return {
            "n": 0,
            "median": None,
            "q25": None,
            "q75": None,
            "p10": None,
            "p90": None,
        }
    q = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])
    return dict(
        zip(("p10", "q25", "median", "q75", "p90"), map(float, q)), n=len(values)
    )


def aggregate(output):
    """Merge final shard tables, retaining only final deliverables."""
    output = Path(output)
    shards = sorted(output.glob("shard_*"))
    rows_by_name = {}
    for name in ("per_item", "per_instance", "slot_selection"):
        rows_by_name[name] = [r for s in shards for r in read_csv(s / (name + ".csv"))]
        write_csv(output / (name + ".csv"), rows_by_name[name])
    items, instances = rows_by_name["per_item"], rows_by_name["per_instance"]
    baseline = [r for r in items if r["group"] == "A0"]
    if len(baseline) != 240 or sum(r["kind"] == "JT" for r in baseline) != 101:
        raise ValueError("Incomplete fixed-240 baseline")
    if len({r["index"] for r in baseline}) != 240:
        raise ValueError("Duplicate baseline rows")
    if sum(r["group"] == "A0" for r in instances) != 216:
        raise ValueError("Missing GT baseline instances")
    buckets = defaultdict(list)
    for row in instances:
        if row["group"] == "A0":
            continue
        actual = numeric(row, "support_ratio")
        if actual is None or abs(actual - 1) < 0.20:
            continue
        ratio = numeric(row, "soft_pixel_ratio")
        if ratio is None or ratio <= 0:
            continue
        row["gain"] = math.log(ratio) / math.log(actual)
        for cohort in (
            "all",
            "candidate20" if row["candidate20"] == "True" else "ordinary",
        ):
            key = (
                row["group"],
                row["axis"],
                "enlarge" if actual > 1 else "shrink",
                cohort,
            )
            buckets[key].append(row)
    curves = []
    for key, values in sorted(buckets.items()):
        gains = [r["gain"] for r in values]
        # Dedup by author/character AND GT instance, keeping all paired scales.
        glyphs = defaultdict(list)
        for r in values:
            glyphs[(r["glyph_key"], r["gt_instance"])].append(r["gain"])
        curves.append(
            {
                **dict(zip(("group", "axis", "direction", "cohort"), key)),
                **distribution(gains),
                "direction_correct": float(np.mean(np.asarray(gains) > 0)),
                "opposite_fraction": float(np.mean(np.asarray(gains) < 0)),
                "glyph_instance_dedup_n": len(glyphs),
                "glyph_instance_dedup_gain_median": float(
                    np.median([np.median(v) for v in glyphs.values()])
                ),
                "clipping_over_5pct": float(
                    np.mean(
                        [numeric(r, "mass_clipping_fraction") > 0.05 for r in values]
                    )
                ),
            }
        )
    write_csv(output / "gain_summary.csv", curves)
    response_rows = []
    by_ratio = defaultdict(list)
    for r in instances:
        if r["group"] != "A0":
            by_ratio[(r["group"], r["axis"], r["requested_ratio"])].append(r)
    for key, values in sorted(by_ratio.items()):
        response_rows.append(
            {
                **dict(zip(("group", "axis", "requested_ratio"), key)),
                "n": len(values),
                **{
                    "support_" + k: v
                    for k, v in distribution(
                        [numeric(r, "support_ratio") for r in values]
                    ).items()
                },
                **{
                    "pixel_" + k: v
                    for k, v in distribution(
                        [numeric(r, "soft_pixel_ratio") for r in values]
                    ).items()
                },
                "hard_pixel_ratio_median": distribution(
                    [numeric(r, "hard_pixel_ratio") for r in values]
                )["median"],
                "roi_pixel_ratio_median": distribution(
                    [numeric(r, "roi_pixel_ratio") for r in values]
                )["median"],
            }
        )
    write_csv(output / "response_curves.csv", response_rows)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, ax in zip(("x", "y"), axes):
        for group in ("A1", "A2", "A3", "A4", "A5", "AO"):
            values = sorted(
                (r for r in response_rows if r["group"] == group and r["axis"] == axis),
                key=lambda r: float(r["requested_ratio"]),
            )
            ax.plot(
                [r["support_median"] for r in values],
                [r["pixel_median"] for r in values],
                "o-",
                label=group,
            )
        ax.plot([0.7, 1.6], [0.7, 1.6], "k--", alpha=0.4, label="unit gain")
        ax.set(
            xlabel="Measured support sigma ratio",
            ylabel="Pixel sigma ratio",
            title=axis + " axis (AO privileged)",
        )
        ax.legend(ncol=2)
        ax.grid(alpha=0.2)
    fig.savefig(output / "figures/response_curves.png", dpi=160)
    plt.close(fig)
    config = json.loads((output / "config.json").read_text())
    denominator = config["s_denominator"]
    metrics = {}
    for kind in ("JT", "BF"):
        cohort = [r for r in baseline if r["kind"] == kind]
        metrics["L1_" + kind] = float(np.mean([numeric(r, "L1") for r in cohort]))
    metrics["J"] = float(
        np.mean([numeric(r, "J") for r in baseline if r["kind"] == "JT"])
    )
    metrics["S"] = sum(metrics[k] / denominator[k] for k in ("L1_JT", "L1_BF", "J"))
    # Condition scores average interventions within each image first. Missing
    # valid predictions are explicit no-ops, so every condition has all 240.
    conditions = defaultdict(lambda: defaultdict(list))
    for r in items:
        if r["group"] not in ("A0", "AO"):
            conditions[(r["group"], r["axis"], r["requested_ratio"])][
                r["index"]
            ].append(r)
    scores = []
    for key, images in sorted(conditions.items()):
        jt_l1, bf_l1, js = [], [], []
        for base in baseline:
            current = images.get(base["index"], [base])
            l1 = float(np.mean([numeric(r, "L1") for r in current]))
            (jt_l1 if base["kind"] == "JT" else bf_l1).append(l1)
            if base["kind"] == "JT":
                js.append(float(np.mean([numeric(r, "J") for r in current])))
        values = {
            "L1_JT": float(np.mean(jt_l1)),
            "L1_BF": float(np.mean(bf_l1)),
            "J": float(np.mean(js)),
        }
        values["S"] = sum(values[k] / denominator[k] for k in values)
        scores.append(
            {**dict(zip(("group", "axis", "requested_ratio"), key)), **values}
        )
    write_csv(output / "condition_scores.csv", scores)
    base_by_index = {r["index"]: r for r in baseline}
    quality_rows = []
    for key, images in sorted(conditions.items()):
        for kind in ("JT", "BF"):
            changes = defaultdict(list)
            for base in baseline:
                if base["kind"] != kind:
                    continue
                current = images.get(base["index"], [base])
                for metric in (
                    "edge_strength",
                    "highpass_l1",
                    "extra_ink_rate",
                    "components",
                    "endpoints",
                    "thin_recall",
                    "short_recall",
                ):
                    vals = [numeric(r, metric) for r in current]
                    vals = [v for v in vals if v is not None]
                    old = numeric(base, metric)
                    if vals and old is not None:
                        changes[metric].append(float(np.mean(vals)) - old)
            quality_rows.append(
                {
                    **dict(zip(("group", "axis", "requested_ratio"), key)),
                    "kind": kind,
                    **{
                        k + "_mean_delta": float(np.mean(v)) for k, v in changes.items()
                    },
                }
            )
    write_csv(output / "quality_deltas.csv", quality_rows)
    failures = [r.copy() for r in items if r["group"] != "A0"]
    for r in failures:
        base = base_by_index[r["index"]]
        r["extra_ink_delta"] = numeric(r, "extra_ink_rate") - numeric(
            base, "extra_ink_rate"
        )
        r["endpoints_delta"] = numeric(r, "endpoints") - numeric(base, "endpoints")
    failures.sort(
        key=lambda r: (r["extra_ink_delta"], r["endpoints_delta"]), reverse=True
    )
    write_csv(output / "failure_candidates.csv", failures[:100])
    evidence = {
        "status": "complete",
        "training_steps": 0,
        "baseline": metrics,
        "curves": curves,
        "manual_quality_gate": "pending_blind_human_review",
        "B_authorized": False,
        "validity": [json.loads((s / "verification.json").read_text()) for s in shards],
    }
    dump_json(output / "summary.json", evidence)
    text = [
        "# Run B Fix Q 空间响应：最终诊断",
        "",
        "零训练、FP32、固定240条；101 JT、139 BF、216 GT实例。",
        "",
        f"A0：L1_JT={metrics['L1_JT']:.9f}，L1_BF={metrics['L1_BF']:.9f}，J={metrics['J']:.9f}，S={metrics['S']:.9f}。",
        "",
        "下表只统计实际支持域尺度变化至少20%的条件；gain为像素/支持域的log比值。",
        "",
        "|组|轴|方向|样本域|n|gain中位|IQR|方向正确率|",
        "|---|---|---|---|---:|---:|---|---:|",
    ]
    for r in curves:
        text.append(
            f"|{r['group']}|{r['axis']}|{r['direction']}|{r['cohort']}|{r['n']}|{r['median']:.5f}|{r['q25']:.5f}–{r['q75']:.5f}|{r['direction_correct']:.1%}|"
        )
    text += [
        "",
        "A1/A4的探索性参考门槛为median gain≥0.10且方向正确率≥70%，"
        "同时必须通过笔锋盲评。未完成人工评分前不判定画质通过、不启动B。"
        "AO包含特权GT，仅诊断，排除主表胜出判定。",
        "",
        "S按每张图先平均各单实例干预，再合并完整240条；无合格预测槽的样本是显式no-op。BF无J真值，未填零。",
        "",
        "连接数、端点、细/短笔画召回、额外墨迹和highpass均为224px评测代理，"
        "不等于人工笔锋评价。scale在224px测量并换算为448px；"
        "GT Voronoi不是生成图真实分割。",
        "",
        "旧val被反复用于探索，单checkpoint、单seed，不支持独立泛化或统计显著性结论。最终产物保留CSV、图、来源与校验；无训练checkpoint或临时feature缓存。",
    ]
    (output / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# 最终结果\n\n打开index.html查看图集；summary.md为中文报告，"
        "summary.json为数值结果。\n"
        "per_item.csv包含所有干预的图像指标；per_instance.csv包含所有GT实例基线与已匹配干预。\n"
        "slot_selection.csv记录过滤；response_curves.csv包含所有倍率，"
        "gain_summary.csv仅汇总实际变化≥20%的条件。\n"
        "condition_scores.csv给出全240条同口径L1/J/S。AO为特权诊断，BF的GT字段为空。\n"
        "blind_scores.csv为待人工填写的评分表；盲图不标条件，blind_key.json独立提供答案。\n",
        encoding="utf-8",
    )
    figures = sorted((output / "figures").glob("*.png"))
    body = "".join(
        f'<li><a href="figures/{html.escape(p.name)}">{html.escape(p.stem)}</a>'
        f'<br><img loading="lazy" width="1100" '
        f'src="figures/{html.escape(p.name)}"></li>'
        for p in figures
    )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Q spatial probe</title>'
        "<style>body{font-family:sans-serif}img{max-width:100%}"
        "li{margin:30px 0}</style>"
        '<h1>Q 空间响应诊断</h1><a href="summary.md">中文报告</a> | '
        '<a href="response_curves.csv">响应表</a> | '
        '<a href="condition_scores.csv">完整240评分</a><ul>' + body + "</ul>",
        encoding="utf-8",
    )
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    aggregate(parser.parse_args().output)
