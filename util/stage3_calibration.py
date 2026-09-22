"""Pure-Python, deterministic calibration math, independently testable."""

import math
import statistics


def calibrate_measurements(measurements):
    """Balance pixel gradients and match the old parameter-gradient scale."""
    medians = [
        statistics.median(r["pixel_norms"][i] for r in measurements) for i in range(4)
    ]
    if any(not math.isfinite(x) or x <= 0 for x in medians):
        return {"status": "blocked", "reason": "Zero/nonfinite pixel gradient"}
    target = math.exp(sum(math.log(x) for x in medians) / 4)
    coefficients = [target / x for x in medians]
    old, new = [], []
    for row in measurements:
        gram = row["parameter_gram"]
        baseline = row.get("baseline_coefficients", [1.0] * 4)
        old.append(math.sqrt(max(0, sum(
            baseline[i] * baseline[j] * gram[i][j]
            for i in range(4) for j in range(4)
        ))))
        value = sum(
            coefficients[i] * coefficients[j] * gram[i][j]
            for i in range(4)
            for j in range(4)
        )
        new.append(math.sqrt(max(0, value)))
    old_median, new_median = statistics.median(old), statistics.median(new)
    if not all(math.isfinite(x) and x > 0 for x in (old_median, new_median)):
        return {"status": "blocked", "reason": "Degenerate parameter gradient"}
    common_scale = old_median / new_median
    effective = [value * common_scale for value in coefficients]
    report = {
        "status": "ok",
        "structure_coefficients": coefficients,
        "structure_common_scale": common_scale,
        "effective_structure_coefficients": effective,
        "pixel_gradient_medians": medians,
        "original_parameter_gradient_median": old_median,
        "uncorrected_parameter_gradient_median": new_median,
    }
    if not math.isfinite(common_scale) or common_scale <= 0 or any(
        not math.isfinite(value) or value < 0.1 or value > 100
        for value in effective
    ):
        report.update(
            status="blocked", reason="Effective coefficient outside bounds"
        )
    return report
