import pytest
from reward import compute_total_reward
from reward.anti_hack_penalty import compute_anti_hack_penalty

def test_compute_total_reward_perfect():
    reward_dict = compute_total_reward(
        era_task=1.0,
        calibration=1.5,
        teacher_delta=1.0,
        legacy_utility=1.0,
        answer_leakage=0.0,
        anti_hack_penalty=0.0
    )
    assert reward_dict["R_total"] == 3.5
    assert reward_dict["R_normalized"] == 1.0

def test_compute_total_reward_with_penalty():
    reward_dict = compute_total_reward(
        era_task=1.0,
        calibration=1.0,
        teacher_delta=0.5,
        legacy_utility=0.5,
        answer_leakage=-1.0,
        anti_hack_penalty=-0.5
    )
    # (1.0 * 1.0) + 0.5 + 0.5 - 1.0 - 0.5 = 0.5
    assert reward_dict["R_total"] == 0.5

def test_anti_hack_penalty():
    # Test valid actions
    action_history = [
        {"action": {"action_type": "call_tool", "payload": {"tool": "get_incident_status"}}}
    ]
    assert compute_anti_hack_penalty(action_history, 40) == 0.0

    # Test hallucinated tool
    action_history = [
        {"action": {"action_type": "call_tool", "payload": {"tool": "hallucinated_tool"}}}
    ]
    assert compute_anti_hack_penalty(action_history, 40) == -0.2

    # Test infinite loop
    action_history = [
        {"action": {"action_type": "write_reasoning", "payload": {"thought": "test"}}},
        {"action": {"action_type": "write_reasoning", "payload": {"thought": "test"}}},
        {"action": {"action_type": "write_reasoning", "payload": {"thought": "test"}}},
        {"action": {"action_type": "write_reasoning", "payload": {"thought": "test"}}},
    ]
    assert compute_anti_hack_penalty(action_history, 40) == -0.5
    
    # Test combined and capped
    action_history = []
    for _ in range(40):
        action_history.append({"action": {"action_type": "call_tool", "payload": {"tool": "hallucinated_tool"}}})
        
    penalty = compute_anti_hack_penalty(action_history, 40)
    assert penalty == -1.0 # Capped at -1.0
