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
        {"confidence": 0.9, "was_true": True},
        {"confidence": 0.2, "was_true": False}
    ]
    
    score = compute_calibration_reward(hypotheses)
    assert score > 1.0 # Should be > 1.0 multiplier

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
