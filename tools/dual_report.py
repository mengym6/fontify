"""Run B clipping, paired bootstrap, fixed blind lists and probe summaries.

Completion evidence and research outcomes are separate.  All comparisons
require the same new instance J coefficients and fixed clean validation set.
"""

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dual_view_protocol import (  # noqa: E402
    SCHEMA,
    read_json,
    sha256_file,
    write_new_json,
)
from tools.eval_s_baseline import absolute_s_score  # noqa: E402


def quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return (
        ordered[low] * (high - position) + ordered[high] * (position - low)
        if low != high
        else ordered[low]
    )


def clip_statistics(rows, clip_grad=3.0):
    """Count optimizer-update rows without duplicating micro-batches."""
    values = []
    for row in rows:
        if row.get("pre_clip_grad_norm") in (None, ""):
            continue
        pre = float(row["pre_clip_grad_norm"])
        ratio = float(
            row.get("grad_clip_ratio", min(1.0, clip_grad / max(pre, 1e-12)))
        )
        if not math.isfinite(pre) or not math.isfinite(ratio) or pre < 0:
            raise ValueError("invalid clipping log")
        values.append((pre, ratio))
    if not values:
        raise ValueError("no per-update pre-clip norm records")
    return {
        "n_logged_updates": len(values),
        "trigger_fraction": sum(pre > clip_grad for pre, _ in values)
        / len(values),
        "pre_clip_norm": {
            "median": quantile([v[0] for v in values], 0.5),
            "p90": quantile([v[0] for v in values], 0.9),
            "maximum": max(v[0] for v in values),
        },
        "clip_ratio": {
            "median": quantile([v[1] for v in values], 0.5),
            "p05": quantile([v[1] for v in values], 0.05),
            "p10": quantile([v[1] for v in values], 0.1),
        },
    }


def read_log(path):
    file = Path(path)
    if file.suffix == ".csv":
        with file.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    return [
        json.loads(line)
        for line in file.read_text().splitlines()
        if line.strip()
    ]


def bootstrap(values, iterations=10000, seed=0):
    """Paired resampling of glyph differences; no unpaired sample shuffle."""
    if not values or iterations < 100:
        raise ValueError(
            "nonempty paired values and >=100 bootstrap draws required"
        )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("nonfinite paired differences")
    rng = random.Random(seed)
    n = len(values)
    estimates = [
        sum(rng.choice(values) for _ in range(n)) / n
        for _ in range(iterations)
    ]
    return {
        "n": n,
        "mean_difference": sum(values) / n,
        "ci95": [quantile(estimates, 0.025), quantile(estimates, 0.975)],
        "iterations": iterations,
        "seed": seed,
    }


def _items(value):
    records = value["per_item"]
    result = {}
    for row in records:
        key = (
            row.get("target_path"),
            row.get("ref_path"),
            row.get("index", row.get("idx")),
        )
        if key[0] is None or key in result:
            raise ValueError(
                "per-item records must have unique fixed "
                "target/ref/index identities"
            )
        result[key] = row
    return result


def paired_comparison(run, reference, denominator, iterations=10000, seed=0):
    for record in (run, reference, denominator):
        if record.get("schema") != SCHEMA:
            raise ValueError("old J evaluation cannot be paired with Run B")
    hash_keys = (
        "fixed_pair_sha256",
        "val_manifest_sha256",
        "target_cache_sha256",
        "eval_coef_sha256",
    )
    for key in hash_keys:
        if (
            not run.get(key)
            or run[key] != reference.get(key)
            or run[key] != denominator.get(key)
        ):
            raise ValueError(f"paired evaluation inputs differ: {key}")
    left, right = _items(run), _items(reference)
    if left.keys() != right.keys():
        raise ValueError(
            "evaluations do not contain the same fixed glyph pairs"
        )
    if any(left[key]["kind"] != right[key]["kind"] for key in left):
        raise ValueError("paired evaluations disagree on JT/BF cohort membership")
    result = {
        "direction": (
            "run_minus_reference; negative metric differences favor run"
        ),
        "groups": {},
        "per_item": [],
    }
    for kind in ("all", "JT", "BF"):
        pairs = [
            (key, left[key], right[key])
            for key in left
            if kind == "all" or left[key]["kind"] == kind
        ]
        if not pairs:
            raise ValueError(f"no paired {kind} validation samples")
        stats = {}
        for term in ("L1", "J", "centroid", "logsigma", "shape"):
            if term != "L1" and kind == "BF":
                continue
            active = [
                (a, b)
                for _, a, b in pairs
                if term == "L1" or a["kind"] == "JT"
            ]
            values = [float(a[term]) - float(b[term]) for a, b in active]
            stats[term] = bootstrap(values, iterations, seed)
        result["groups"][kind] = stats
    # S averages THREE cohort means, so preserve JT/BF cohort sizes in each
    # paired bootstrap draw.  Per-glyph J=0 BF values are not diluted into J.
    jt = [(a, right[key]) for key, a in left.items() if a["kind"] == "JT"]
    bf = [(a, right[key]) for key, a in left.items() if a["kind"] == "BF"]

    def cohort_score(column):
        means = {
            "L1_JT": sum(float(row[column]["L1"]) for row in jt) / len(jt),
            "L1_BF": sum(float(row[column]["L1"]) for row in bf) / len(bf),
            "J": sum(float(row[column]["J"]) for row in jt) / len(jt),
        }
        return absolute_s_score(means, denominator)

    run_s, run_components = cohort_score(0)
    reference_s, reference_components = cohort_score(1)
    rng = random.Random(seed)
    s_values = []

    def s_difference(jt_rows, bf_rows):
        return (
            sum((float(a["L1"]) - float(b["L1"])) for a, b in jt_rows)
            / len(jt_rows)
            / denominator["L1_JT"]
            + sum((float(a["L1"]) - float(b["L1"])) for a, b in bf_rows)
            / len(bf_rows)
            / denominator["L1_BF"]
            + sum((float(a["J"]) - float(b["J"])) for a, b in jt_rows)
            / len(jt_rows)
            / denominator["J"]
        )

    for _ in range(iterations):
        s_values.append(
            s_difference(
                [rng.choice(jt) for _ in jt], [rng.choice(bf) for _ in bf]
            )
        )
    result["S"] = {
        "run_S": run_s,
        "reference_S": reference_s,
        "run_components": run_components,
        "reference_components": reference_components,
        "mean_difference": s_difference(jt, bf),
        "ci95": [quantile(s_values, 0.025), quantile(s_values, 0.975)],
        "bootstrap_rule": "paired resampling within fixed JT/BF cohorts",
        "iterations": iterations,
        "seed": seed,
    }
    for key, row in left.items():
        other = right[key]
        result["per_item"].append(
            {
                "identity": list(key),
                "kind": row["kind"],
                "L1_difference": row["L1"] - other["L1"],
                "J_difference": row["J"] - other["J"]
                if row["kind"] == "JT"
                else None,
            }
        )
    return result


def fixed_blind_manifest(items, models, seed=0, limit=40):
    """Generate before training and seal its hash with training inputs."""
    if len(models) < 2 or len(set(models)) != len(models):
        raise ValueError("at least two distinct model labels required")
    rng = random.Random(seed)
    candidates = sorted(
        items, key=lambda row: (row["target_path"], row.get("idx", 0))
    )
    rng.shuffle(candidates)
    chosen = candidates[:limit]
    comparisons = []
    for sample in chosen:
        for reference in models[1:]:
            sides = [models[0], reference]
            rng.shuffle(sides)
            comparisons.append(
                {
                    "blind_id": f"blind_{len(comparisons):04d}",
                    "target_path": sample["target_path"],
                    "ref_path": sample.get("ref_path"),
                    "index": sample.get("index", sample.get("idx")),
                    "left_model": sides[0],
                    "right_model": sides[1],
                }
            )
    return {
        "schema": SCHEMA,
        "purpose": "fixed_blind_manifest",
        "seed": seed,
        "generated_before_training": True,
        "comparisons": comparisons,
        "rating_fields": ["structure_preference", "pen_preference"],
        "allowed_preferences": ["left", "right", "tie"],
    }


def blind_ratings(manifest, rows):
    lookup = {row["blind_id"]: row for row in manifest["comparisons"]}
    counts, seen = {}, set()
    for row in rows:
        identifier = row["blind_id"]
        if identifier not in lookup or identifier in seen:
            raise ValueError("unknown/duplicate blind rating")
        seen.add(identifier)
        pairing = lookup[identifier]
        for field in manifest["rating_fields"]:
            preference = row[field]
            if preference not in ("left", "right", "tie"):
                raise ValueError("blind preferences must be left/right/tie")
            winner = (
                "tie"
                if preference == "tie"
                else pairing[preference + "_model"]
            )
            counts.setdefault(field, Counter())[winner] += 1
    return {
        "rated": len(seen),
        "missing": sorted(lookup.keys() - seen),
        "preferences": {key: dict(value) for key, value in counts.items()},
    }


def probe_summary(files):
    records = [read_json(path) for path in files]
    points = {row["point"]: row for row in records}
    expected = {"warm_end", "ep2", "ep25", "ep50"}
    if points.keys() != expected or len(records) != 4:
        raise ValueError(
            "exactly warm_end/ep2/ep25/ep50 probe outputs required"
        )
    for key in ("sample_manifest_sha256", "calibration_sha256"):
        if (
            len({row.get(key) for row in records}) != 1
            or records[0].get(key) is None
        ):
            raise ValueError(f"probe fixed-input mismatch: {key}")
    result = {
        "points": {},
        "matching_switches": {},
        "interpretation": (
            "offline gradient drift; no claim of training benefit"
        ),
    }
    previous = None
    for point in ("warm_end", "ep2", "ep25", "ep50"):
        row = points[point]
        result["points"][point] = {
            "status": row["status"],
            "fusion_open": row.get("fusion_open"),
            "summary": row.get("summary"),
        }
        if row["status"] != "measured":
            result["points"][point]["reason"] = row.get("reason")
            previous = None
            continue
        matching = {
            (batch["group"], tuple(batch["target_indices"])): batch[
                "slots"
            ].get("matching")
            for batch in row["measurements"]
        }
        slots = {}
        for group in {batch["group"] for batch in row["measurements"]}:
            batches = [
                batch
                for batch in row["measurements"]
                if batch["group"] == group
            ]
            slots[group] = {}
            for field in (
                "active_slots_mean",
                "empty_glyph_fraction",
                "full_glyph_fraction",
                "active_slot_pair_iou_mean",
            ):
                available = [
                    batch
                    for batch in batches
                    if batch["slots"].get(field) is not None
                ]
                count = sum(batch["n"] for batch in available)
                slots[group][field] = (
                    sum(
                        batch["n"] * batch["slots"][field]
                        for batch in available
                    )
                    / count
                    if count
                    else None
                )
        result["points"][point]["slots"] = slots
        if previous is not None:
            valid = [
                key
                for key in matching
                if matching[key] is not None and previous.get(key) is not None
            ]
            result["matching_switches"][point] = {
                "matched_batches": len(valid),
                "changed_batches": sum(
                    matching[key] != previous[key] for key in valid
                ),
            }
        previous = matching
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    clipping = commands.add_parser("clipping")
    clipping.add_argument("--log", required=True)
    clipping.add_argument("--clip-grad", type=float, default=3.0)
    paired = commands.add_parser("paired")
    paired.add_argument("--run", required=True)
    paired.add_argument("--reference", nargs="+", required=True)
    paired.add_argument("--denominator", required=True)
    paired.add_argument("--iterations", type=int, default=10000)
    paired.add_argument("--seed", type=int, default=0)
    blind = commands.add_parser("blind-manifest")
    blind.add_argument("--evaluation", required=True)
    blind.add_argument(
        "--models", nargs="+", default=["run_b", "baseline_vggfix", "ctrl_g12"]
    )
    blind.add_argument("--seed", type=int, default=0)
    blind.add_argument("--limit", type=int, default=40)
    ratings = commands.add_parser("blind-ratings")
    ratings.add_argument("--manifest", required=True)
    ratings.add_argument("--ratings", required=True)
    probes = commands.add_parser("probes")
    probes.add_argument("--probe", nargs=4, required=True)
    for sub in (clipping, paired, blind, ratings, probes):
        sub.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "clipping":
        result = clip_statistics(read_log(args.log), args.clip_grad)
    elif args.command == "paired":
        result = {
            "comparisons": {
                Path(path).stem: paired_comparison(
                    read_json(args.run),
                    read_json(path),
                    read_json(args.denominator),
                    args.iterations,
                    args.seed,
                )
                for path in args.reference
            },
            "process_completion": (
                "requires separate GPU/DDP/51-epoch evidence"
            ),
            "research_effect": (
                "single-seed paired outcomes; no single-factor attribution"
            ),
        }
    elif args.command == "blind-manifest":
        evaluation = read_json(args.evaluation)
        result = fixed_blind_manifest(
            evaluation["per_item"], args.models, args.seed, args.limit
        )
        result["fixed_pair_sha256"] = evaluation["fixed_pair_sha256"]
        result["input_evaluation_sha256"] = sha256_file(args.evaluation)
    elif args.command == "blind-ratings":
        result = blind_ratings(
            read_json(args.manifest), read_log(args.ratings)
        )
        result["blind_manifest_sha256"] = sha256_file(args.manifest)
    else:
        result = probe_summary(args.probe)
    write_new_json(
        args.output, {"schema": SCHEMA, "purpose": args.command, **result}
    )
    print(json.dumps({"output": args.output, "purpose": args.command}))


if __name__ == "__main__":
    main()
