"""Result export keeps pairing, target and instance denominators separate."""

import csv
import json

import pytest
from PIL import Image

from tools.export_runc_v2_report import (
    excluded_unique_summary,
    export_run,
    quantiles,
    summarize_target_rows,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def metrics(value):
    return {
        "head_soft_iou": value / 10,
        "head_logsigma_x": value,
        "head_logsigma_y": value / 2,
        "head_logsigma": value * 1.5,
        "head_centroid_absolute": value / 100,
        "head_centroid_relative": value / 10,
        "head_shape": value / 5,
        "head_bce": value / 3,
        "head_dice": value / 10,
        "head_existence_recall": 1.0,
        "head_role_accuracy": 0.5,
    }


def fixture_report(cohort="fixed_historical_val"):
    rows = []
    for identity, author, value in (
        ("A", "one", 1.0),
        ("A", "one", 3.0),
        ("B", "two", 4.0),
        ("C", "two", 10.0),
    ):
        rows.append(
            {
                "identity": identity,
                "target_path": identity,
                "author": author,
                "character": identity,
                "instance_count": 2,
                **metrics(value),
            }
        )
    pairs = {name: quantiles(row[name] for row in rows) for name in metrics(1)}
    unique, _ = summarize_target_rows(rows)
    instance = {name: quantiles([0.1, 0.4, 0.9]) for name in metrics(1)}
    return {
        "stage": "ia0",
        "cohort": cohort,
        "rows": rows,
        "summary": {
            "pair_count": 4,
            "unique_target_count": 3,
            "pairs": pairs,
            "unique_targets": unique,
            "author_character": unique,
        },
        "instance_count": 8,
        "instance_metrics": instance,
        "by_author_instance_metrics": {"one": instance, "two": instance},
        "evaluated_parameter_sha256": {"sha256": "parameter-state"},
        "metadata": {
            "source_sha": "evaluation-source",
            "checkpoint_sha256": "start-checkpoint",
            "initial_checkpoint_sha256": "ckpt14",
        },
    }


def fixture_run(folder):
    run = folder / "run"
    run.mkdir()
    config = {"stage": "ia0", "training": {"steps": 2}}
    write_json(run / "config.json", config)
    write_json(
        run / "metadata.json",
        {
            "source_sha": "training-source",
            "checkpoint_sha256": "start-checkpoint",
            "initial_checkpoint_sha256": "ckpt14",
            "config": config,
        },
    )
    (run / "train.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "optimizer_step": step,
                    "effective_batch": 32,
                    "losses": {
                        "total": 10 - step,
                        "raw_head": 2.0,
                        "raw_head_geo": 3.0,
                    },
                    "gradient_preclip": 4.0,
                    "lr": {"head": 0.001},
                }
            )
            for step in (1, 2)
        ),
        encoding="utf-8",
    )
    for step in (0, 2):
        write_json(
            run / f"evaluation-step-{step:06d}" / "evaluation.json", fixture_report()
        )
        write_json(
            run / f"train-evaluation-step-{step:06d}" / "evaluation.json",
            fixture_report("fixed_nine_source_train18"),
        )
        write_json(
            run / f"checkpoint-step-{step:06d}.pth.sha256.json",
            {"sha256": f"snapshot-{step}"},
        )
    manifest = folder / "data_manifest.json"
    write_json(
        manifest,
        {
            "same_author_character_overlap": ["C"],
            "same_author_character_excluded_unique": 2,
            "historical_jt_unique_targets": 3,
        },
    )
    return run, manifest


def test_unique_target_average_references_before_quantiles():
    report = fixture_report()
    summary, count = summarize_target_rows(report["rows"])
    assert count == 3
    assert summary["head_logsigma_x"]["mean"] == pytest.approx((2 + 4 + 10) / 3)
    subset = excluded_unique_summary(
        report,
        {
            "same_author_character_overlap": ["C"],
            "same_author_character_excluded_unique": 2,
        },
    )
    assert subset["pair_count"] == 3
    assert subset["unique_target_count"] == 2
    assert subset["metrics"]["head_logsigma_x"]["mean"] == 3
    assert subset["metrics"]["head_logsigma_x"]["p95"] == pytest.approx(3.9)


def test_export_preserves_instance_author_tails_and_checkpoint_meanings(tmp_path):
    run, manifest = fixture_run(tmp_path)
    output = tmp_path / "report"
    result = export_run(run, output, manifest)
    assert result["status"] == "complete_numerical_export"
    assert result["acceptance"] == "not_performed"
    rows = list(csv.DictReader((output / "head_metrics.csv").open()))
    tail = next(
        row
        for row in rows
        if row["aggregation"] == "matched_instances_by_author"
        and row["author"] == "one"
        and row["optimizer_step"] == "2"
    )
    assert tail["p95"] == "0.85"
    assert tail["training_source_sha"] == "training-source"
    assert tail["evaluation_source_sha"] == "evaluation-source"
    assert tail["start_checkpoint_sha256"] == "start-checkpoint"
    assert tail["snapshot_checkpoint_sha256"] == "snapshot-2"
    assert tail["evaluated_parameter_sha256"] == "parameter-state"
    subset = [
        row for row in rows if row["cohort"] == "same_author_character_excluded_val"
    ]
    assert all(row["aggregation"] == "unique_target_means" for row in subset)
    assert all(row["matched_instance_count"] == "" for row in subset)
    with Image.open(output / "curves.png") as image:
        assert image.width > 1000 and image.height > 1000
    assert not (output / "coarse_metrics.csv").exists()
    assert "not independent" in (output / "ia_results.md").read_text()


def test_missing_eval_is_partial_and_empty_run_is_error(tmp_path):
    run, manifest = fixture_run(tmp_path)
    for path in run.glob("*evaluation-step-*/evaluation.json"):
        path.unlink()
    result = export_run(run, tmp_path / "partial", manifest)
    assert result["status"] == "partial"
    assert any("No nonempty" in message for message in result["warnings"])
    (run / "train.jsonl").unlink()
    with pytest.raises(ValueError, match="No optimizer updates"):
        export_run(run, tmp_path / "empty", manifest)


def test_nonfinite_numbers_are_rejected(tmp_path):
    run, manifest = fixture_run(tmp_path)
    path = run / "evaluation-step-000002" / "evaluation.json"
    report = fixture_report()
    report["rows"][0]["head_shape"] = float("nan")
    write_json(path, report)
    with pytest.raises(ValueError, match="Nonfinite"):
        export_run(run, tmp_path / "bad", manifest)


def test_ib_coordinates_and_oracle_are_exported_separately(tmp_path):
    run, manifest = fixture_run(tmp_path)
    config = json.loads((run / "config.json").read_text())
    config["stage"] = "ib"
    write_json(run / "config.json", config)
    for path in run.glob("*evaluation-step-*/evaluation.json"):
        report = json.loads(path.read_text())
        report["stage"] = "ib"
        for name in (
            "coarse_logsigma_x",
            "p0_original448_logsigma_x",
            "oracle_original448_logsigma_x",
            "coarse_J_per_character",
        ):
            report["instance_metrics"][name] = quantiles([0.1, 0.2])
        write_json(path, report)
    output = tmp_path / "ib"
    export_run(run, output, manifest)
    rows = list(csv.DictReader((output / "coarse_metrics.csv").open()))
    assert any(
        row["metric"] == "coarse_logsigma_x" and "density112" in row["coordinate"]
        for row in rows
    )
    assert any(
        row["metric"] == "oracle_original448_logsigma_x"
        and "original448" in row["coordinate"]
        for row in rows
    )
    assert (output / "ib_results.md").exists()
    assert (output / "coarse_curves.png").exists()
    assert any(
        row["metric"] == "coarse_J_per_character"
        and row["aggregation"] == "per_character_evaluation"
        for row in rows
    )


def test_stage_mismatch_is_rejected(tmp_path):
    run, manifest = fixture_run(tmp_path)
    path = run / "evaluation-step-000002" / "evaluation.json"
    report = json.loads(path.read_text())
    report["stage"] = "ib"
    write_json(path, report)
    with pytest.raises(ValueError, match="Evaluation stage differs"):
        export_run(run, tmp_path / "mismatched", manifest)
