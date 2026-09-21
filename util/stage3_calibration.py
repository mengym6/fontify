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
    if any(x < 0.1 or x > 100 for x in coefficients):
        return {
            "status": "blocked",
            "reason": "Required coefficient outside bounds",
            "proposed_coefficients": coefficients,
        }
    old, new = [], []
    for row in measurements:
        gram = row["parameter_gram"]
        old.append(math.sqrt(max(0, sum(map(sum, gram)))))
        value = sum(
            coefficients[i] * coefficients[j] * gram[i][j]
            for i in range(4)
            for j in range(4)
        )
        new.append(math.sqrt(max(0, value)))
    old_median, new_median = statistics.median(old), statistics.median(new)
    if not all(math.isfinite(x) and x > 0 for x in (old_median, new_median)):
        return {"status": "blocked", "reason": "Degenerate parameter gradient"}
    return {
        "status": "ok",
        "structure_coefficients": coefficients,
        "structure_common_scale": old_median / new_median,
        "pixel_gradient_medians": medians,
        "original_parameter_gradient_median": old_median,
        "uncorrected_parameter_gradient_median": new_median,
    }
