import pytest
from reward import compute_total_reward
from reward.era_task_reward import compute_era_task_reward
from reward.calibration_reward import compute_calibration_reward

def test_era_task_reward():
    required = ["incident_resolved", "slo_breach_avoided", "legacy_doc_written"]
    achieved = ["incident_resolved", "legacy_doc_written"]
    
    score = compute_era_task_reward(achieved, required)
    assert score == 2/3

def test_calibration_reward():
    # Perfect calibration: confident and right, unconfident and wrong
    hypotheses = [
        {"confidence": 0.9, "was_true": True},   # error = (0.9-1.0)^2 = 0.01
        {"confidence": 0.2, "was_true": False}    # error = (0.2-0.0)^2 = 0.04
    ]
    # mean_brier = 0.025, delta = 0.5 - 0.05 = 0.45, multiplier = 1.45
    score = compute_calibration_reward(hypotheses)
    assert score > 1.0  # Should be > 1.0 multiplier for good calibration
    
    # Terrible calibration: confident and wrong
    bad_hypotheses = [
        {"confidence": 0.9, "was_true": False},  # error = 0.81
        {"confidence": 0.1, "was_true": True}    # error = 0.81
    ]
    score_bad = compute_calibration_reward(bad_hypotheses)
    assert score_bad < 1.0  # Should be < 1.0 for poor calibration
    
    # Neutral: no hypotheses
    assert compute_calibration_reward([]) == 1.0

def test_total_reward_computation():
    # Example from spec 14.4
    res = compute_total_reward(
        era_task=0.75,
        calibration=1.3,
        teacher_delta=0.6,
        legacy_utility=0.65,
        answer_leakage=-0.3
    )
    
    # R_total = (0.75 * 1.3) + 0.6 + 0.65 - 0.3 = 0.975 + 0.95 = 1.925
    assert abs(res["R_total"] - 1.925) < 0.001
    assert abs(res["R_normalized"] - (1.925 / 3.5)) < 0.001
