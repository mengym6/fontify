"""Export stage-specific Run C v2 curves and denominator-aware result tables.

The exporter reads saved numerical artifacts only. It neither selects a winner
nor grants stage acceptance. Missing snapshots remain explicitly partial.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

STATISTICS = ("count", "mean", "p50", "p90", "p95")
HEAD_METRICS = (
    "head_soft_iou",
    "head_logsigma_x",
    "head_logsigma_y",
    "head_logsigma",
    "head_centroid_absolute",
    "head_centroid_relative",
    "head_shape",
    "head_existence_recall",
    "head_role_accuracy",
    "head_bce",
    "head_dice",
)
CSV_FIELDS = (
    "stage",
    "optimizer_step",
    "cohort",
    "aggregation",
    "author",
    "metric",
    "count",
    "mean",
    "p50",
    "p90",
    "p95",
    "pair_count",
    "unique_target_count",
    "matched_instance_count",
    "coordinate",
    "training_source_sha",
    "evaluation_source_sha",
    "initial_checkpoint_sha256",
    "start_checkpoint_sha256",
    "snapshot_checkpoint_sha256",
    "evaluated_parameter_sha256",
    "evaluation_path",
)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_tree(value, location="root"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite number in {location}")
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_tree(child, f"{location}[{index}]")


def read_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    finite_tree(data, str(path))
    return data


def quantiles(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {}

    def percentile(fraction):
        index = (len(values) - 1) * fraction
        low = math.floor(index)
        high = math.ceil(index)
        return values[low] + (values[high] - values[low]) * (index - low)

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
    }


def numeric_metrics(rows):
    return sorted(
        {
            name
            for row in rows
            for name, value in row.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and name not in ("index", "instance_count")
        }
    )


def summarize_target_rows(rows):
    """Average references inside each target before target-weighted quantiles."""
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("identity", row["target_path"])].append(row)
    metrics = numeric_metrics(rows)
    target_means = []
    for members in groups.values():
        target_means.append(
            {
                metric: statistics.fmean(
                    row[metric] for row in members if metric in row
                )
                for metric in metrics
                if any(metric in row for row in members)
            }
        )
    return {
        metric: quantiles(row[metric] for row in target_means if metric in row)
        for metric in metrics
    }, len(groups)


def excluded_unique_summary(report, manifest):
    """Keep the risk-excluded unique-target cohort separate from original val."""
    excluded = set(manifest["same_author_character_overlap"])
    retained = [row for row in report["rows"] if row["target_path"] not in excluded]
    summary, count = summarize_target_rows(retained)
    return {
        "metrics": summary,
        "pair_count": len(retained),
        "unique_target_count": count,
        "excluded_target_paths_present": sorted(
            {
                row["target_path"]
                for row in report["rows"]
                if row["target_path"] in excluded
            }
        ),
        "expected_unique_target_count": manifest[
            "same_author_character_excluded_unique"
        ],
        "denominator": "one mean per original target, averaging its references first",
        "limitation": "reused validation after same-author/character exclusion; "
        "not independent generalization; not deduplicated instance tails",
    }


def metric_coordinate(name):
    if name.startswith(("head_", "raw_head")):
        return (
            "Head soft occupancy on 28x28; metric-specific valid-instance denominator"
        )
    if name.startswith(("p0_original448_", "oracle_original448_")):
        return "original448; common GT Voronoi expanded nearest from audited224"
    if name.startswith(("coarse_", "raw_j_")):
        return "density112; directly built GT partition112; oracle density target112"
    if name.startswith(("raw_density", "raw_occupancy")):
        return "density/occupancy112; valid pixel weights then per-character mean"
    return "phase objective; see saved config weights and runner reduction"


def is_head_metric(name):
    return name.startswith(("head_", "raw_head"))


def is_coarse_metric(name):
    return name.startswith(
        (
            "raw_density",
            "raw_occupancy",
            "raw_j_",
            "coarse_",
            "p0_original448_",
            "oracle_original448_",
        )
    )


def snapshot_sha(run_dir, step):
    path = run_dir / f"checkpoint-step-{step:06d}.pth.sha256.json"
    return read_json(path).get("sha256") if path.exists() else None


def read_training(run_dir, warnings):
    path = run_dir / "train.jsonl"
    if not path.exists():
        warnings.append("train.jsonl is absent; no online training curve is available")
        return []
    rows = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        finite_tree(row, f"{path}:{line_number}")
        step = row.get("optimizer_step")
        if type(step) is not int or step < 1 or not isinstance(row.get("losses"), dict):
            raise ValueError(f"Malformed training record at {path}:{line_number}")
        if step in rows and row != rows[step]:
            raise ValueError(f"Conflicting duplicate optimizer_step={step}")
        rows[step] = row
    if not rows:
        warnings.append("train.jsonl contains no optimizer updates")
    return [rows[step] for step in sorted(rows)]


def collect_evaluations(run_dir, manifest, warnings):
    evaluations = []
    paths = sorted(run_dir.glob("*-evaluation-step-*/evaluation.json"))
    paths += sorted(run_dir.glob("evaluation-step-*/evaluation.json"))
    for path in paths:
        match = re.fullmatch(r"(train-)?evaluation-step-(\d+)", path.parent.name)
        if not match:
            continue
        report = read_json(path)
        if not isinstance(report.get("rows"), list) or not report["rows"]:
            warnings.append(f"Empty/missing evaluation rows: {path}")
            continue
        for name in ("summary", "instance_metrics", "by_author_instance_metrics"):
            if not isinstance(report.get(name), dict) or not report[name]:
                warnings.append(f"Missing {name}: {path}")
        for metric in HEAD_METRICS:
            if not report.get("instance_metrics", {}).get(metric):
                warnings.append(f"Missing matched-instance metric {metric}: {path}")
        if not report.get("metadata", {}).get("source_sha"):
            warnings.append(f"Missing evaluation source SHA: {path}")
        cohort = "train18" if match.group(1) else "historical_val"
        item = {
            "optimizer_step": int(match.group(2)),
            "cohort": cohort,
            "path": str(path),
            "report": report,
            "snapshot_checkpoint_sha256": snapshot_sha(run_dir, int(match.group(2))),
        }
        if cohort == "historical_val":
            item["risk_excluded_unique"] = excluded_unique_summary(report, manifest)
            expected = manifest.get("historical_jt_unique_targets", 77)
            observed = report.get("summary", {}).get("unique_target_count")
            if observed != expected:
                warnings.append(
                    f"{path}: observed unique JT={observed}, expected {expected}"
                )
            expected_pairs = manifest.get("historical_jt_pairs")
            observed_pairs = report.get("summary", {}).get("pair_count")
            if expected_pairs is not None and observed_pairs != expected_pairs:
                warnings.append(
                    f"{path}: observed JT pairs={observed_pairs}, "
                    f"expected {expected_pairs}"
                )
            excluded = item["risk_excluded_unique"]
            if (
                excluded["unique_target_count"]
                != excluded["expected_unique_target_count"]
            ):
                warnings.append(
                    f"{path}: risk-excluded unique count is "
                    f"{excluded['unique_target_count']}, expected "
                    f"{excluded['expected_unique_target_count']}"
                )
        evaluations.append(item)
    if not evaluations:
        warnings.append(
            "No nonempty saved evaluations; all fixed-cohort curves are absent"
        )
    return sorted(
        evaluations, key=lambda item: (item["optimizer_step"], item["cohort"])
    )


def table_rows(evaluations, metadata, stage):
    result = []
    for item in evaluations:
        report = item["report"]
        summary = report.get("summary", {})
        evidence = report.get("metadata", {})
        state_sha = report.get("evaluated_parameter_sha256", {})
        if isinstance(state_sha, dict):
            state_sha = state_sha.get("sha256")
        base = {
            "stage": stage,
            "optimizer_step": item["optimizer_step"],
            "cohort": item["cohort"],
            "author": "all",
            "pair_count": summary.get("pair_count"),
            "unique_target_count": summary.get("unique_target_count"),
            "matched_instance_count": report.get("instance_count"),
            "training_source_sha": metadata.get("source_sha"),
            "evaluation_source_sha": evidence.get("source_sha"),
            "initial_checkpoint_sha256": evidence.get("initial_checkpoint_sha256"),
            "start_checkpoint_sha256": evidence.get("checkpoint_sha256"),
            "snapshot_checkpoint_sha256": item["snapshot_checkpoint_sha256"],
            "evaluated_parameter_sha256": state_sha,
            "evaluation_path": item["path"],
        }
        tables = [
            ("pair_means", "all", summary.get("pairs", {})),
            ("unique_target_means", "all", summary.get("unique_targets", {})),
            ("author_character_means", "all", summary.get("author_character", {})),
            ("matched_instances", "all", report.get("instance_metrics", {})),
        ]
        tables.extend(
            ("matched_instances_by_author", author, metrics)
            for author, metrics in report.get("by_author_instance_metrics", {}).items()
        )
        for aggregation, author, metrics in tables:
            for metric, values in metrics.items():
                if not values:
                    continue
                domain = aggregation
                if metric.endswith(("_J_per_character", "_shield")):
                    domain = aggregation.replace(
                        "matched_instances", "per_character_evaluation"
                    )
                result.append(
                    {
                        **base,
                        "aggregation": domain,
                        "author": author,
                        "metric": metric,
                        "coordinate": metric_coordinate(metric),
                        **{key: values.get(key) for key in STATISTICS},
                    }
                )
        if "risk_excluded_unique" in item:
            subset = item["risk_excluded_unique"]
            for metric, values in subset["metrics"].items():
                result.append(
                    {
                        **base,
                        "cohort": "same_author_character_excluded_val",
                        "aggregation": "unique_target_means",
                        "metric": metric,
                        "pair_count": subset["pair_count"],
                        "unique_target_count": subset["unique_target_count"],
                        "matched_instance_count": None,
                        "coordinate": metric_coordinate(metric),
                        **values,
                    }
                )
    return result


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flatten_training(rows):
    result = []
    for row in rows:
        result.append(
            {
                "optimizer_step": row["optimizer_step"],
                "effective_batch": row.get("effective_batch"),
                "gradient_preclip": row.get("gradient_preclip"),
                "elapsed_seconds": row.get("elapsed_seconds"),
                **{f"loss_{key}": value for key, value in row["losses"].items()},
                **{f"lr_{key}": value for key, value in row.get("lr", {}).items()},
            }
        )
    return result


def curve_series(
    rows, metric, cohort, aggregation="matched_instances", statistic="mean"
):
    selected = [
        row
        for row in rows
        if row["metric"] == metric
        and row["cohort"] == cohort
        and row["aggregation"] == aggregation
        and row["author"] == "all"
        and row.get(statistic) is not None
    ]
    return (
        [row["optimizer_step"] for row in selected],
        [row[statistic] for row in selected],
    )


def plot_metrics(output, training, rows, coarse=False):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    metrics = (
        [
            ("raw_density", "Density Smooth L1 [112]"),
            ("raw_occupancy_bce", "Occupancy BCE [112]"),
            ("raw_occupancy_dice", "Occupancy Dice diagnostic [112]"),
            ("coarse_J_per_character", "J coarse [112]"),
            ("coarse_logsigma_x", "Coarse log sigma x [112]"),
            ("coarse_logsigma_y", "Coarse log sigma y [112]"),
            ("coarse_centroid_absolute", "Coarse absolute centroid [112]"),
            ("coarse_centroid_relative", "Coarse relative centroid [112]"),
            ("coarse_shape", "Coarse normalized shape [112]"),
            ("p0_original448_logsigma_x", "P0 log sigma x [original448]"),
            ("p0_original448_logsigma_y", "P0 log sigma y [original448]"),
            ("p0_original448_J_per_character", "P0 vs oracle J [original448]"),
            ("p0_original448_centroid_absolute", "P0 absolute centroid [original448]"),
            ("p0_original448_centroid_relative", "P0 relative centroid [original448]"),
            ("p0_original448_shape", "P0 normalized shape [original448]"),
        ]
        if coarse
        else [
            ("online", "Online train objective; sampled batches"),
            ("head_soft_iou", "Head soft IoU [28]"),
            ("head_logsigma_x", "Head log sigma x [28]"),
            ("head_logsigma_y", "Head log sigma y [28]"),
            ("head_centroid_absolute", "Head absolute centroid [28]"),
            ("head_centroid_relative", "Head relative centroid [28]"),
            ("head_shape", "Head normalized shape [28]"),
            ("head_existence_recall", "Head existence recall"),
            ("head_role_accuracy", "Head role accuracy"),
            ("head_bce", "Head occupancy BCE"),
            ("head_dice", "Head occupancy Dice loss"),
            ("head_logsigma", "Head log sigma x+y [28]"),
        ]
    )
    online_colors = {
        "total": "#7e3e9a",
        "raw_head": "#bc3442",
        "raw_head_geo": "#288b6a",
    }
    columns = 3
    figure, axes = plt.subplots(
        math.ceil(len(metrics) / columns),
        columns,
        figsize=(16, 3.4 * math.ceil(len(metrics) / columns)),
    )
    for axis, (metric, title) in zip(axes.flat, metrics):
        if metric == "online":
            for key in ("total", "raw_head", "raw_head_geo"):
                selected = [row for row in training if key in row["losses"]]
                if selected:
                    axis.plot(
                        [row["optimizer_step"] for row in selected],
                        [row["losses"][key] for row in selected],
                        color=online_colors[key],
                        label=key,
                    )
        else:
            for cohort, color in (
                ("train18", "#bd641a"),
                ("historical_val", "#1e64ac"),
            ):
                aggregation = (
                    "pair_means"
                    if metric.startswith("raw_")
                    else "per_character_evaluation"
                    if metric.endswith(("_J_per_character", "_shield"))
                    else "matched_instances"
                )
                short_cohort = "val101" if cohort == "historical_val" else cohort
                for statistic, style in (("mean", "-"), ("p90", "--"), ("p95", ":")):
                    x, y = curve_series(rows, metric, cohort, aggregation, statistic)
                    if x:
                        axis.plot(
                            x,
                            y,
                            style,
                            color=color,
                            marker=".",
                            label=f"{short_cohort} {statistic}",
                        )
                if metric.startswith("p0_original448_"):
                    oracle_metric = metric.replace(
                        "p0_original448_", "oracle_original448_"
                    )
                    x, y = curve_series(rows, oracle_metric, cohort, aggregation)
                    if x:
                        axis.plot(
                            x,
                            y,
                            "-.",
                            color=color,
                            alpha=0.65,
                            label=f"{cohort} oracle mean",
                        )
            if not coarse:
                for cohort, color in (
                    ("historical_val", "#555555"),
                    ("same_author_character_excluded_val", "#369b53"),
                ):
                    x, y = curve_series(rows, metric, cohort, "unique_target_means")
                    if x:
                        axis.plot(
                            x,
                            y,
                            "-",
                            color=color,
                            linewidth=0.9,
                            label=(
                                "val77" if cohort == "historical_val" else "val74-excl"
                            ),
                        )
        axis.set_title(title)
        axis.set_xlabel("Optimizer updates in this stage")
        axis.grid(alpha=0.25)
        if not axis.lines:
            axis.text(
                0.5, 0.5, "No saved measurements", ha="center", transform=axis.transAxes
            )
    legend = [
        Line2D([0], [0], color="#bd641a", label="train18"),
        Line2D([0], [0], color="#1e64ac", label="val101"),
    ]
    if not coarse:
        legend.extend(
            [
                Line2D([0], [0], color="#555555", label="val77 (unique mean)"),
                Line2D([0], [0], color="#369b53", label="val74-excl (unique mean)"),
            ]
        )
    legend.extend(
        Line2D([0], [0], color="black", linestyle=style, label=name)
        for name, style in (("mean", "-"), ("P90", "--"), ("P95", ":"))
    )
    if coarse:
        legend.append(
            Line2D([0], [0], color="black", linestyle="-.", label="oracle mean")
        )
        caption = (
            "D/occupancy and J: per-character distribution; geometry: "
            "matched-instance distribution. Coordinates 112 and original448 "
            "stay separate.\n"
            "Dash-dot curves are oracle-original448 means, not trained predictions."
        )
    else:
        legend.extend(
            Line2D([0], [0], color=online_colors[key], label=label)
            for key, label in (
                ("total", "online total"),
                ("raw_head", "online H"),
                ("raw_head_geo", "online Hgeo"),
            )
        )
        caption = (
            "train18/val101: matched-instance mean/P90/P95; "
            "val77/val74-excl: unique-target means after averaging references.\n"
            "Online losses use sampled batches. Upper IoU/recall/accuracy quantiles "
            "are not worst-case error tails."
        )
    figure.suptitle(
        "Run C v2 coarse metrics" if coarse else "Run C v2 Head metrics",
        fontsize=14,
        y=0.997,
    )
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.978),
        ncol=5 if not coarse else 6,
        fontsize=9,
        frameon=False,
    )
    figure.text(0.5, 0.008, caption, ha="center", va="bottom", fontsize=9)
    figure.tight_layout(rect=(0, 0.045, 1, 0.925))
    figure.savefig(output / ("coarse_curves.png" if coarse else "curves.png"), dpi=160)
    plt.close(figure)


def markdown_report(stage, status, warnings, training, evaluations, rows, provenance):
    lines = [
        f"# Run C v2 {stage} measured results",
        "",
        f"Export status: **{status}**. Numerical export only; no winner selection "
        "or automatic structural acceptance.",
        "",
        "Online loss uses sampled optimizer batches. Fixed train18, historical "
        "101 JT pairs / 77 unique targets / 216 matched GT instances, and the "
        "same-author-character-excluded 74 unique targets are separate domains. "
        "Per-character means first average valid instances inside each word. "
        "Matched-instance tails pool eligible GT instances; relative-centroid "
        "counts can differ because two valid parts are required. Unique-target "
        "P90/P95 are quantiles across target means, not instance-tail quantiles.",
        "For higher-is-better IoU/recall/accuracy, upper P90/P95 are not worst-case "
        "tails; for geometric errors upper quantiles indicate large-error tails.",
        "",
        "The additional exclusion set removes the three target paths recorded "
        "in data_manifest; it remains reused validation and is not independent "
        "generalization. No slot-index alignment across references is invented.",
        "",
        f"Observed optimizer updates: {len(training)}; last step: "
        f"{max((row['optimizer_step'] for row in training), default='unavailable')}.",
        "",
        "## Snapshot evidence",
        "",
        "| step | cohort | aggregation | pairs | unique | metric n | IoU mean | "
        "log sigma x mean | log sigma y mean | log sigma xy P95 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in evaluations:
        for aggregation in ("matched_instances", "unique_target_means"):
            cohorts = [item["cohort"]]
            if (
                aggregation == "unique_target_means"
                and item["cohort"] == "historical_val"
            ):
                cohorts.append("same_author_character_excluded_val")
            for cohort in cohorts:
                selected = {
                    row["metric"]: row
                    for row in rows
                    if row["optimizer_step"] == item["optimizer_step"]
                    and row["cohort"] == cohort
                    and row["aggregation"] == aggregation
                    and row["author"] == "all"
                }
                iou = selected.get("head_soft_iou", {})

                def cell(metric, statistic="mean"):
                    value = selected.get(metric, {}).get(statistic)
                    return f"{value:.6f}" if value is not None else "unavailable"

                if not iou:
                    continue
                lines.append(
                    f"| {item['optimizer_step']} | {cohort} | {aggregation} | "
                    f"{iou.get('pair_count')} | {iou.get('unique_target_count')} | "
                    f"{iou.get('count')} | {cell('head_soft_iou')} | "
                    f"{cell('head_logsigma_x')} | {cell('head_logsigma_y')} | "
                    f"{cell('head_logsigma', 'p95')} |"
                )
    lines.extend(
        [
            "",
            "## Coordinates and losses",
            "",
            "Head geometry uses soft occupancy on 28x28 with the original "
            "Hungarian assignment. Density/occupancy and training J use 112x112 "
            "and the directly built coarse GT partition. P0/oracle-original448 "
            "metrics share the original448 GT coordinate system and a common "
            "GT Voronoi expanded from224. Oracle-original448 is representation "
            "error; it is not a generated model output. BCE/Dice and weighted "
            "occupancy objective remain separately labeled in CSV. Raw inactive "
            "coarse zeros in IA are not presented as rendered image measurements.",
            "",
            "## Provenance",
            "",
            "```json",
            json.dumps(provenance, ensure_ascii=False, indent=2),
            "```",
            "",
            "Evaluation metadata checkpoint SHA identifies the starting "
            "checkpoint. Snapshot file SHA comes from the saved sidecar "
            "manifest; evaluated parameter-state SHA identifies the model "
            "actually evaluated. Model binaries were not read by this exporter.",
            "",
            "## Limitations and missing evidence",
            "",
        ]
    )
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append(
            "- No missing numerical artifacts detected for the requested budget."
        )
    lines.extend(
        [
            "- Single-seed observations; repeated historical validation selection.",
            "- Human review and final structure acceptance are not performed here.",
            "",
        ]
    )
    return "\n".join(lines)


def export_run(run_dir, output, data_manifest):
    run_dir, output = Path(run_dir), Path(output)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    manifest = read_json(data_manifest)
    for key in (
        "same_author_character_overlap",
        "same_author_character_excluded_unique",
    ):
        if key not in manifest:
            raise ValueError(f"Data manifest lacks exclusion evidence: {key}")
    warnings = []
    metadata_path, config_path = run_dir / "metadata.json", run_dir / "config.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    config = (
        read_json(config_path) if config_path.exists() else metadata.get("config", {})
    )
    if not metadata:
        warnings.append(
            "Training metadata.json is missing; source/checkpoint evidence partial"
        )
    stage = config.get("stage")
    if stage not in ("ia0", "ia1", "ib", "ic"):
        raise ValueError("Run config must identify stage ia0/ia1/ib/ic")
    training = read_training(run_dir, warnings)
    evaluations = collect_evaluations(run_dir, manifest, warnings)
    if any(item["report"].get("stage") != stage for item in evaluations):
        raise ValueError("Evaluation stage differs from the independent run config")
    if not training and not evaluations:
        raise ValueError(
            "No optimizer updates or nonempty evaluation artifacts to export"
        )
    planned = config.get("training", {}).get("steps")
    observed = max((row["optimizer_step"] for row in training), default=0)
    if planned is not None and observed != planned:
        warnings.append(
            f"Requested budget={planned} updates, observed last update={observed}"
        )
    cohorts = {item["cohort"] for item in evaluations}
    for cohort in ("train18", "historical_val"):
        if cohort not in cohorts:
            warnings.append(f"No {cohort} evaluations")
        elif (
            max(
                item["optimizer_step"]
                for item in evaluations
                if item["cohort"] == cohort
            )
            < observed
        ):
            warnings.append(
                f"Latest {cohort} evaluation is earlier than last optimizer update"
            )
    steps_by_cohort = {
        cohort: {
            item["optimizer_step"] for item in evaluations if item["cohort"] == cohort
        }
        for cohort in cohorts
    }
    if (
        len(steps_by_cohort) == 2
        and len(
            set(map(tuple, (sorted(values) for values in steps_by_cohort.values())))
        )
        != 1
    ):
        warnings.append(
            "Train and val snapshots are incomplete or have different steps"
        )
    rows = table_rows(evaluations, metadata, stage)
    head_rows = [row for row in rows if is_head_metric(row["metric"])]
    coarse_rows = [row for row in rows if is_coarse_metric(row["metric"])]
    provenance = {
        "training_source_sha": metadata.get("source_sha"),
        "start_checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "initial_checkpoint_sha256": metadata.get("initial_checkpoint_sha256"),
        "evaluation_source_shas": sorted(
            {
                item["report"].get("metadata", {}).get("source_sha") or "unavailable"
                for item in evaluations
            }
        ),
        "snapshot_file_shas": {
            str(item["optimizer_step"]): item["snapshot_checkpoint_sha256"]
            for item in evaluations
        },
        "data_manifest_sha256": file_sha256(data_manifest),
        "training_backend": metadata.get("backend"),
        "evaluation_backends": [
            {
                "step": item["optimizer_step"],
                "cohort": item["cohort"],
                "backend": item["report"].get("backend"),
                "precision": item["report"].get("precision"),
            }
            for item in evaluations
        ],
    }
    if not provenance["training_source_sha"]:
        warnings.append("Training source SHA is absent")
    if any(item["snapshot_checkpoint_sha256"] is None for item in evaluations):
        warnings.append("Some evaluated snapshots lack checkpoint file-SHA sidecars")
    if stage in ("ib", "ic") and not any(
        row["metric"].startswith("p0_original448_") for row in coarse_rows
    ):
        warnings.append("Rendered stage lacks original448 P0/oracle geometry")
    status = "partial" if warnings else "complete_numerical_export"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "head_metrics.csv", head_rows, CSV_FIELDS)
    if stage in ("ib", "ic"):
        write_csv(output / "coarse_metrics.csv", coarse_rows, CSV_FIELDS)
        plot_metrics(output, training, rows, coarse=True)
    online = flatten_training(training)
    online_fields = sorted({key for row in online for key in row})
    write_csv(
        output / "training_steps.csv", online, online_fields or ["optimizer_step"]
    )
    plot_metrics(output, training, rows)
    report = {
        "status": status,
        "stage": stage,
        "acceptance": "not_performed",
        "run_directory": str(run_dir),
        "warnings": warnings,
        "optimizer_update_count": len(training),
        "last_optimizer_step": observed,
        "provenance": provenance,
        "head_csv_rows": len(head_rows),
        "coarse_csv_rows": len(coarse_rows) if stage in ("ib", "ic") else 0,
        "risk_excluded_unique_by_snapshot": [
            {"optimizer_step": item["optimizer_step"], **item["risk_excluded_unique"]}
            for item in evaluations
            if "risk_excluded_unique" in item
        ],
    }
    (output / "report_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    filename = "ia_results.md" if stage in ("ia0", "ia1") else "ib_results.md"
    (output / filename).write_text(
        markdown_report(
            stage, status, warnings, training, evaluations, rows, provenance
        ),
        encoding="utf-8",
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", required=True, help="one independent stage output directory"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-manifest", required=True)
    args = parser.parse_args(argv)
    report = export_run(args.run, args.output, args.data_manifest)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "stage", "optimizer_update_count", "warnings")
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
