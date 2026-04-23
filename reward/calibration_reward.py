from typing import List, Dict


def compute_calibration_reward(hypotheses: List[Dict]) -> float:
    """
    R_calibration = 1.0 + calibration_delta

    Uses a proper Brier-score–style calibration measure:
    - For each hypothesis, the "error" is |confidence - outcome|
      where outcome = 1.0 if true, 0.0 if false.
    - Perfect calibration (low error) → multiplier > 1.0
    - Poor calibration (high error) → multiplier < 1.0

    Range: 0.5× to 1.5× (clamped)
    """
    if not hypotheses:
        return 1.0  # Neutral multiplier if no hypotheses declared

    errors = []
    for h in hypotheses:
        confidence = float(h.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        outcome = 1.0 if h.get("was_true", False) else 0.0
        error = (confidence - outcome) ** 2
        errors.append(error)

    mean_brier = sum(errors) / len(errors)

    # Brier score of 0.0 = perfect, 0.25 = random guessing at 50%
    # Map: 0.0 → +0.5 (best), 0.25 → 0.0 (neutral), 0.5 → -0.5 (worst)
    calibration_delta = 0.5 - (2.0 * mean_brier)

    multiplier = 1.0 + calibration_delta
    return max(0.5, min(1.5, multiplier))
