"""
EpistemicOps Reward Model
==========================
Five-component reward: R_era_task, R_calibration, R_teacher_delta,
R_legacy_utility, R_answer_leakage.

Formula: R_total = (R_era_task × R_calibration) + R_teacher_delta + R_legacy_utility + R_answer_leakage
"""

from reward.era_task_reward import compute_era_task_reward
from reward.calibration_reward import compute_calibration_reward
from reward.teacher_delta_reward import compute_teacher_delta_reward
from reward.legacy_utility_reward import compute_legacy_utility_reward
from reward.leakage_penalty import compute_leakage_penalty
from reward.anti_hack_penalty import compute_anti_hack_penalty


def compute_total_reward(
    era_task: float,
    calibration: float,
    teacher_delta: float,
    legacy_utility: float,
    answer_leakage: float,
    anti_hack_penalty: float = 0.0,
) -> dict:
    """
    Combine all reward components per the EpistemicOps formula.

    R_total = (R_era_task × R_calibration) + R_teacher_delta + R_legacy_utility + R_answer_leakage + R_anti_hack_penalty

    Returns dict with all components + total + normalized score.
    """
    adjusted_era_task = era_task * calibration
    r_total = adjusted_era_task + teacher_delta + legacy_utility + answer_leakage + anti_hack_penalty

    # Max possible: (1.0 × 1.5) + 1.0 + 1.0 + 0.0 = 3.5
    max_possible = 3.5
    normalized = max(0.0, r_total / max_possible)

    return {
        "R_era_task": era_task,
        "R_calibration": calibration,
        "R_era_task_adjusted": adjusted_era_task,
        "R_teacher_delta": teacher_delta,
        "R_legacy_utility": legacy_utility,
        "R_answer_leakage": answer_leakage,
        "R_anti_hack_penalty": anti_hack_penalty,
        "R_total": r_total,
        "R_normalized": round(normalized, 4),
        "R_max_possible": max_possible,
    }


__all__ = [
    "compute_total_reward",
    "compute_era_task_reward",
    "compute_calibration_reward",
    "compute_teacher_delta_reward",
    "compute_legacy_utility_reward",
    "compute_leakage_penalty",
    "compute_anti_hack_penalty",
]
