"""
test_anti_hack.py — Tests for the expanded anti-hack penalty module.

Covers: hypothesis spamming, early-completion gaming, legacy spamming,
content-free legacy, loop detection, hallucinated tools, combined cap.
"""
import pytest
from reward.anti_hack_penalty import compute_anti_hack_penalty, compute_anti_hack_penalty_breakdown


def _make_action(action_type: str, payload: dict, step: int = 5) -> dict:
    return {"action": {"action_type": action_type, "payload": payload}, "step": step}


# ── Loop detection ─────────────────────────────────────────────────────────────

def test_loop_detection():
    """4 identical actions in a row should trigger loop penalty."""
    history = [
        _make_action("write_reasoning", {"thought": "thinking..."}, step=i)
        for i in range(4)
    ]
    penalty = compute_anti_hack_penalty(history, 40)
    assert penalty <= -0.5


# ── Hallucinated tool ─────────────────────────────────────────────────────────

def test_hallucinated_tool_penalty():
    """Calling an unknown tool should give a penalty."""
    history = [
        _make_action("call_tool", {"tool": "magic_resolver_v99", "args": {}}, step=5)
    ]
    penalty = compute_anti_hack_penalty(history, 40)
    assert penalty <= -0.2


def test_known_tool_no_penalty():
    """Calling a known tool should NOT trigger hallucinated-tool penalty."""
    history = [
        _make_action("call_tool", {"tool": "get_incident_status", "args": {"incident_id": "INC-001"}}, step=5)
    ]
    penalty = compute_anti_hack_penalty(history, 40)
    assert penalty == 0.0


# ── Early completion gaming ───────────────────────────────────────────────────

def test_early_complete_gaming():
    """declare_task_complete at step < 4 should trigger early-complete penalty."""
    history = [
        _make_action("declare_task_complete", {"outcome": "done", "summary": "all fixed"}, step=1)
    ]
    penalty = compute_anti_hack_penalty(history, 40)
    assert penalty <= -0.30


def test_late_complete_no_gaming_penalty():
    """declare_task_complete at a reasonable step should not get the gaming penalty."""
    history = [
        _make_action("call_tool", {"tool": "get_incident_status", "args": {}}, step=5),
        _make_action("call_tool", {"tool": "resolve_incident", "args": {}}, step=8),
        _make_action("declare_task_complete", {"outcome": "resolved", "summary": "fixed"}, step=12)
    ]
    penalty = compute_anti_hack_penalty(history, 40)
    assert penalty == 0.0  # No penalties


# ── Hypothesis spamming ───────────────────────────────────────────────────────

def test_hypothesis_spam_detection():
    """More than 3 duplicate hypotheses should trigger spam penalty."""
    history = [
        _make_action("declare_hypothesis",
                     {"hypothesis": "API drift detected", "confidence": 0.5}, step=i)
        for i in range(5)  # 5 identical hypotheses
    ]
    penalty = compute_anti_hack_penalty(history, 40)
    assert penalty <= -0.15  # Should hit hyp_spam penalty


def test_diverse_hypotheses_no_spam():
    """Multiple unique hypotheses should not trigger spam penalty."""
    history = [
        _make_action("declare_hypothesis", {"hypothesis": "Connection pool exhausted", "confidence": 0.7}, step=3),
        _make_action("declare_hypothesis", {"hypothesis": "Redis timeout causing cascades", "confidence": 0.6}, step=6),
        _make_action("declare_hypothesis", {"hypothesis": "Deployment caused memory leak", "confidence": 0.5}, step=9),
    ]
    penalty = compute_anti_hack_penalty(history, 40)
    assert penalty == 0.0  # No spam


# ── Legacy spam ───────────────────────────────────────────────────────────────

def test_legacy_spam_penalty():
    """Writing legacy more than 2 times should trigger spam penalty."""
    history = [
        _make_action("write_legacy", {"content": "SECTION 1: state\n" * 20}, step=10),
        _make_action("write_legacy", {"content": "SECTION 1: state\n" * 20}, step=20),
        _make_action("write_legacy", {"content": "SECTION 1: state\n" * 20}, step=30),  # 3rd write
    ]
    penalty = compute_anti_hack_penalty(history, 40)
    assert penalty <= -0.25


# ── Content-free legacy ───────────────────────────────────────────────────────

def test_content_free_legacy_penalty():
    """A legacy doc with < 150 chars should trigger empty-legacy penalty."""
    history = [
        _make_action("write_legacy", {"content": "SECTION 1: done\nSECTION 2: ok"}, step=30)
    ]
    penalty = compute_anti_hack_penalty(history, 40)
    assert penalty <= -0.30


def test_substantial_legacy_no_penalty():
    """A substantial legacy doc should not trigger content-free penalty."""
    long_content = (
        "SECTION 1: WORLD STATE AT ERA END\n"
        "Payment service was experiencing P1 latency spike. Connection pool exhausted.\n"
        "SECTION 2: TRUST RATINGS\n"
        "oncall_engineer: 0.8 (responded promptly). service_owner: 0.9.\n"
        "SECTION 3: DRIFT EVENTS DETECTED\n"
        "incident-api status field changed from int to string 'INVESTIGATING' at step 12.\n"
        "SECTION 4: KEY DECISIONS & RATIONALE\n"
        "Decided to test API schema before assuming code bug.\n"
        "SECTION 5: OPEN ISSUES\n"
        "metrics-api may drift similarly. Watch datapoints field.\n"
        "SECTION 6: RECOMMENDED FIRST ACTIONS\n"
        "Call get_incident_status first to verify schema.\n"
    )
    history = [
        _make_action("write_legacy", {"content": long_content}, step=35)
    ]
    penalty = compute_anti_hack_penalty(history, 40)
    assert penalty == 0.0


# ── Combined and capped ───────────────────────────────────────────────────────

def test_combined_penalties_capped_at_minus_one():
    """Combined penalties should be capped at -1.0."""
    history = []
    for i in range(40):
        history.append(_make_action("call_tool", {"tool": "hallucinated_v999", "args": {}}, step=i))

    penalty = compute_anti_hack_penalty(history, 40)
    assert penalty == -1.0  # Capped


def test_breakdown_has_all_keys():
    """Breakdown dict should always contain all expected keys."""
    history = [_make_action("write_reasoning", {"thought": "ok"}, step=5)]
    breakdown = compute_anti_hack_penalty_breakdown(history, 40)
    EXPECTED_KEYS = {"timeout", "hallucinated_tool", "loop", "hyp_spam",
                     "early_complete", "legacy_spam", "empty_legacy", "total"}
    assert EXPECTED_KEYS == set(breakdown.keys())
