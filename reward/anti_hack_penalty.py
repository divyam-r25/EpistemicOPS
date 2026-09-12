"""
anti_hack_penalty.py — Episode-level anti-reward-hacking penalties.

Penalises degenerate behaviour patterns that can earn reward without
actually solving the task. Called once per era at the end of run_era().

Each penalty is independently observable and logged separately.
"""
from __future__ import annotations

from typing import List


KNOWN_TOOLS = frozenset({
    "get_incident_status", "resolve_incident", "get_metrics",
    "rollback_deployment", "query_logs", "send_notification",
})


def compute_anti_hack_penalty(action_history: list, max_steps: int) -> float:
    """
    Penalise reward-hacking behaviours across the full era trajectory.

    Penalties:
      -0.50  Exceeding max_steps (timeout)
      -0.20  Each hallucinated tool call
      -0.50  Infinite loop (last 4 actions identical)
      -0.30  Hypothesis spamming (>3 identical/near-identical hypotheses)
      -0.30  Early-completion gaming (declare_task_complete at step < 4)
      -0.25  Legacy spamming (write_legacy called >2 times)
      -0.30  Content-free legacy document (<150 chars)

    Returns negative float or 0.0 (capped at -1.0).
    """
    breakdown = _compute_penalty_breakdown(action_history, max_steps)
    total = sum(breakdown.values())
    return max(total, -1.0)


def compute_anti_hack_penalty_breakdown(action_history: list, max_steps: int) -> dict:
    """
    Like compute_anti_hack_penalty but returns each component for logging.

    Returns dict with keys: timeout, hallucinated_tool, loop, hyp_spam,
    early_complete, legacy_spam, empty_legacy, total.
    """
    breakdown = _compute_penalty_breakdown(action_history, max_steps)
    total = max(sum(breakdown.values()), -1.0)
    return {**breakdown, "total": total}


def _compute_penalty_breakdown(action_history: list, max_steps: int) -> dict:
    penalties = {
        "timeout": 0.0,
        "hallucinated_tool": 0.0,
        "loop": 0.0,
        "hyp_spam": 0.0,
        "early_complete": 0.0,
        "legacy_spam": 0.0,
        "empty_legacy": 0.0,
    }

    if not action_history:
        return penalties

    # ── Timeout ───────────────────────────────────────────────────────────────
    if len(action_history) >= max_steps:
        penalties["timeout"] = -0.5

    # ── Analyse per-action ────────────────────────────────────────────────────
    hypothesis_texts: List[str] = []
    legacy_count = 0
    early_complete_seen = False

    for item in action_history:
        action = item.get("action", {})
        action_type = action.get("action_type", "")
        payload = action.get("payload", {})
        step = item.get("step", 999)

        # Hallucinated tool
        if action_type == "call_tool":
            tool = payload.get("tool", "")
            if tool not in KNOWN_TOOLS and ":" not in str(tool):
                penalties["hallucinated_tool"] -= 0.2

        # Hypothesis text collection
        if action_type == "declare_hypothesis":
            text = str(payload.get("hypothesis", "")).strip().lower()
            hypothesis_texts.append(text)

        # Early-completion gaming
        if action_type == "declare_task_complete" and not early_complete_seen:
            if step < 4:
                penalties["early_complete"] = -0.30
                early_complete_seen = True

        # Legacy spam
        if action_type == "write_legacy":
            legacy_count += 1
            content = str(payload.get("content", ""))
            # Content-free legacy (first time)
            if legacy_count == 1 and len(content) < 150:
                penalties["empty_legacy"] = -0.30

    # Legacy spamming: more than 2 write_legacy calls
    if legacy_count > 2:
        penalties["legacy_spam"] = -0.25

    # ── Hypothesis spamming ───────────────────────────────────────────────────
    if len(hypothesis_texts) > 3:
        # Check if more than 3 are duplicates or near-duplicates
        unique = set(hypothesis_texts)
        if len(unique) < len(hypothesis_texts) / 2:
            # More than half are duplicates
            penalties["hyp_spam"] = -0.30
        elif len(hypothesis_texts) > 6:
            # Even many unique hypotheses is suspicious
            penalties["hyp_spam"] = -0.15

    # ── Infinite loop ─────────────────────────────────────────────────────────
    if len(action_history) >= 4:
        last_4 = [str(item.get("action", {})) for item in action_history[-4:]]
        if len(set(last_4)) == 1:
            penalties["loop"] = -0.50

    return penalties
