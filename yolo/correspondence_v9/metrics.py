"""Evaluate own-anchor localization and genuine rejection, without rematching IDs."""

from collections import Counter
import numpy as np


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def evaluate_sample(
    probabilities,
    masks,
    truth_masks,
    matchable,
    groups,
    threshold=0.5,
    iou_threshold=0.5,
):
    probabilities = np.asarray(probabilities)
    masks = np.asarray(masks, dtype=bool)
    truth_masks = np.asarray(truth_masks, dtype=bool)
    labels = np.asarray(matchable)
    groups = np.asarray(groups)
    n = len(labels)
    if (
        probabilities.shape != (n,)
        or masks.shape != truth_masks.shape
        or masks.shape[0] != n
    ):
        raise ValueError("Anchor-aligned arrays required")
    if not 0 <= threshold <= 1 or not 0 < iou_threshold <= 1:
        raise ValueError("Invalid thresholds")
    accepted = probabilities >= threshold
    duplicates = set()
    for i in range(n):
        for j in range(i + 1, n):
            if (
                labels[i] == labels[j] == 1
                and groups[i] == groups[j]
                and accepted[i]
                and accepted[j]
                and iou(truth_masks[i], truth_masks[j]) < 0.5
                and iou(masks[i], masks[j]) > 0.9
            ):
                duplicates.update((i, j))
    rows = []
    for i in range(n):
        if labels[i] < 0:
            continue
        own = iou(masks[i], truth_masks[i]) if labels[i] == 1 else None
        other = max(
            [
                iou(masks[i], truth_masks[j])
                for j in range(n)
                if j != i and labels[j] == 1 and groups[j] == groups[i]
            ]
            or [0.0]
        )
        if labels[i] == 1:
            if not accepted[i]:
                status = "false_reject"
            elif own >= iou_threshold and own + 1e-8 >= other and i not in duplicates:
                status = "correct_match"
            else:
                status = "wrong_match"
        else:
            status = "false_accept" if accepted[i] else "correct_reject"
        rows.append(
            {
                "index": i,
                "group": "BF" if groups[i] == 0 else "JT",
                "status": status,
                "iou": own,
                "duplicate_conflict": i in duplicates,
            }
        )
    return rows


def summarize(rows):
    def ratio(a, b):
        return a / b if b else None

    output = {}
    for group in ("BF", "JT", "all"):
        r = [x for x in rows if group == "all" or x["group"] == group]
        c = Counter(x["status"] for x in r)
        positives = c["correct_match"] + c["wrong_match"] + c["false_reject"]
        negatives = c["correct_reject"] + c["false_accept"]
        accepts = c["correct_match"] + c["wrong_match"] + c["false_accept"]
        rejects = c["correct_reject"] + c["false_reject"]
        output[group] = {
            **{
                k: c[k]
                for k in (
                    "correct_match",
                    "wrong_match",
                    "false_reject",
                    "correct_reject",
                    "false_accept",
                )
            },
            "matchable_support": positives,
            "unmatchable_support": negatives,
            "match_precision": ratio(c["correct_match"], accepts),
            "match_recall": ratio(c["correct_match"], positives),
            "unmatchable_precision": ratio(c["correct_reject"], rejects),
            "unmatchable_recall": ratio(c["correct_reject"], negatives),
            "false_reject_rate": ratio(c["false_reject"], positives),
            "false_accept_rate": ratio(c["false_accept"], negatives),
            "duplicate_conflict_queries": sum(x["duplicate_conflict"] for x in r),
            "mean_iou_of_accepted_matchable": float(
                np.mean(
                    [
                        x["iou"]
                        for x in r
                        if x["status"] in ("correct_match", "wrong_match")
                    ]
                )
            )
            if any(x["status"] in ("correct_match", "wrong_match") for x in r)
            else None,
        }
    return output
