"""
test_grpo_reward.py — Tests for the tiered GRPO reward function.

Key invariants that must ALWAYS hold:
  1. Invalid JSON → 0.0 reward
  2. Hallucinated tool → 0.0 quality (hallucinated tool penalty)
  3. Full, substantive legacy doc > empty legacy doc
  4. Specific drift hypothesis (when drift context present) > vague hypothesis
  5. Known tool with args > known tool without args > unknown tool
  6. Early task complete (step 0) earns negative reward
  7. Content-free legacy earns negative reward
  8. Action type alone cannot earn Tier-2 quality reward (context required)
"""
import pytest
from reward.grpo_reward import (
    compute_grpo_reward,
    compute_grpo_reward_with_components,
    _parse_action,
    _detect_drift_evidence,
    json,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def reward(completion: str, prompt: str = "") -> float:
    return compute_grpo_reward([completion], prompts=[prompt])[0]


def components(completion: str, prompt: str = "") -> dict:
    return compute_grpo_reward_with_components([completion], prompts=[prompt])[0]


# ── Invalid / malformed ───────────────────────────────────────────────────────

def test_invalid_json_zero_reward():
    assert reward("invalid json garbage") == 0.0


def test_empty_string_zero_reward():
    assert reward("") == 0.0


def test_valid_json_but_missing_action_type_zero():
    assert reward('{"payload": {}}') == 0.0


# ── Format scoring ────────────────────────────────────────────────────────────

def test_unknown_action_type_lower_format():
    r = components('{"action_type": "totally_unknown", "payload": {}}')
    assert r["R_format"] < 0.2  # Not full format score


def test_known_action_type_full_format():
    r = components('{"action_type": "write_reasoning", "payload": {"thought": "analyzing..."}}')
    assert r["R_format"] == 0.2


# ── Action quality: call_tool ─────────────────────────────────────────────────

def test_known_tool_with_args_higher_than_no_args():
    with_args = reward(
        '{"action_type": "call_tool", "payload": {"tool": "get_incident_status", "args": {"incident_id": "INC-001"}}}'
    )
    no_args = reward(
        '{"action_type": "call_tool", "payload": {"tool": "get_incident_status", "args": {}}}'
    )
    assert with_args > no_args


def test_hallucinated_tool_zero_quality():
    r = components(
        '{"action_type": "call_tool", "payload": {"tool": "super_magic_api_v999", "args": {}}}'
    )
    assert r["R_action_quality"] == 0.0


# ── Action quality: declare_hypothesis ────────────────────────────────────────

def test_specific_hypothesis_post_drift_higher():
    """Specific hypothesis in drift context should score higher than vague one."""
    drift_prompt = "DRIFT_INJECTION metric_value INVESTIGATING next_cursor"
    specific = reward(
        '{"action_type": "declare_hypothesis", "payload": '
        '{"hypothesis": "metrics-api field renamed from value to metric_value", "confidence": 0.75}}',
        prompt=drift_prompt
    )
    vague = reward(
        '{"action_type": "declare_hypothesis", "payload": '
        '{"hypothesis": "API drift detected", "confidence": 0.5}}',
        prompt=drift_prompt
    )
    assert specific > vague


def test_hypothesis_no_drift_context_lower_score():
    """A hypothesis with no drift evidence in prompt earns zero drift-awareness."""
    r = components(
        '{"action_type": "declare_hypothesis", "payload": '
        '{"hypothesis": "drift in metrics schema", "confidence": 0.6}}',
        prompt=""  # No drift evidence in prompt
    )
    assert r["R_drift_awareness"] == 0.0


def test_overconfident_hypothesis_penalized():
    """Very high confidence (>0.85) hypothesis should earn less than calibrated one."""
    prompt = "DRIFT_INJECTION metric_value"
    overconfident = reward(
        '{"action_type": "declare_hypothesis", "payload": '
        '{"hypothesis": "metrics api renamed value field", "confidence": 0.99}}',
        prompt=prompt
    )
    calibrated = reward(
        '{"action_type": "declare_hypothesis", "payload": '
        '{"hypothesis": "metrics api renamed value field", "confidence": 0.75}}',
        prompt=prompt
    )
    assert calibrated >= overconfident


# ── Action quality: write_legacy ──────────────────────────────────────────────

FULL_LEGACY = """{
    "action_type": "write_legacy",
    "payload": {"content": "SECTION 1: WORLD STATE AT ERA END\\nPayment service P1 latency.\\nSECTION 2: TRUST RATINGS\\nAll stable.\\nSECTION 3: DRIFT EVENTS DETECTED\\nmetrics-api value field renamed to metric_value.\\nSECTION 4: KEY DECISIONS\\nProbed API before assuming code bug.\\nSECTION 5: OPEN ISSUES\\nMonitor notify-api.\\nSECTION 6: RECOMMENDED FIRST ACTIONS\\nCall get_incident_status first."}
}"""

EMPTY_LEGACY = '{"action_type": "write_legacy", "payload": {"content": "SECTION 1: done"}}'


def test_full_legacy_higher_than_empty():
    assert reward(FULL_LEGACY) > reward(EMPTY_LEGACY)


def test_empty_legacy_negative_reward():
    assert reward(EMPTY_LEGACY) < 0.0


def test_full_legacy_reasonable_score():
    r = reward(FULL_LEGACY)
    assert r > 0.5  # Full legacy with all sections should score well


# ── Action quality: declare_task_complete ────────────────────────────────────

def test_early_task_complete_negative():
    """Declaring task complete at step 0 should get a negative reward."""
    r = reward(
        '{"action_type": "declare_task_complete", "payload": {"outcome": "done", "summary": "fixed"}}',
        prompt='"step": 0'
    )
    assert r < 0.0


def test_late_task_complete_positive():
    """Declaring complete after tool calls at a later step is legitimate."""
    r = reward(
        '{"action_type": "declare_task_complete", "payload": '
        '{"outcome": "incident resolved", "summary": "rolled back deployment"}}'  ,
        prompt='"step": 20 call_tool tool_response incidents_resolved'
    )
    assert r > 0.0


# ── Drift awareness scoring ───────────────────────────────────────────────────

def test_drift_awareness_zero_without_context():
    """No drift evidence in prompt → R_drift_awareness must be 0.0."""
    r = components(
        '{"action_type": "declare_hypothesis", '
        '"payload": {"hypothesis": "drift in pagination model", "confidence": 0.7}}',
        prompt="Normal operation, no anomalies"
    )
    assert r["R_drift_awareness"] == 0.0


def test_drift_awareness_positive_with_context():
    """Strong drift context + specific drift hypothesis → R_drift_awareness > 0."""
    r = components(
        '{"action_type": "declare_hypothesis", '
        '"payload": {"hypothesis": "status field type changed from int to string INVESTIGATING", "confidence": 0.75}}',
        prompt="DRIFT_INJECTION INVESTIGATING 204 next_cursor metric_value"
    )
    assert r["R_drift_awareness"] > 0.0


# ── Ordering invariants ───────────────────────────────────────────────────────

def test_good_actions_beat_bad_actions():
    """Good completions must all score higher than invalid JSON."""
    good = [
        reward(FULL_LEGACY),
        reward('{"action_type": "call_tool", "payload": {"tool": "get_incident_status", "args": {"id": "I1"}}}'),
        reward(
            '{"action_type": "declare_hypothesis", "payload": '
            '{"hypothesis": "metrics-api field renamed value to metric_value", "confidence": 0.75}}',
            prompt="DRIFT_INJECTION metric_value"
        ),
    ]
    bad = reward("invalid json garbage")
    for g in good:
        assert g > bad, f"Good reward {g} should beat invalid JSON reward {bad}"


def test_normalize_max_reward_at_most_one():
    """No completion should earn above 1.0 reward."""
    for comp in [FULL_LEGACY,
                 '{"action_type": "call_tool", "payload": {"tool": "get_incident_status", "args": {"id": "X"}}}',
                 '{"action_type": "ready_to_operate", "payload": {"world_model_summary": "All APIs stable."}}']:
        r = reward(comp, prompt="DRIFT_INJECTION metric_value")
        assert r <= 1.0, f"Reward exceeded 1.0: {r} for {comp[:60]}"


# ── _parse_action helper ──────────────────────────────────────────────────────

def test_parse_action_extracts_json_from_markdown():
    text = "```json\n{\"action_type\": \"end_era\", \"payload\": {}}\n```"
    action = _parse_action(text, json.JSONDecoder())
    assert action is not None
    assert action["action_type"] == "end_era"


def test_parse_action_returns_none_for_garbage():
    assert _parse_action("blah blah", json.JSONDecoder()) is None


# ── _detect_drift_evidence helper ────────────────────────────────────────────

def test_detect_drift_evidence_counts_patterns():
    from reward.grpo_reward import _detect_drift_evidence
    prompt = "DRIFT_INJECTION metric_value next_cursor INVESTIGATING"
    count = _detect_drift_evidence(prompt)
    assert count >= 3  # At least 3 patterns: DRIFT_INJECTION, metric_value, next_cursor, INVESTIGATING
