"""
legacy_utility_reward.py — Reward for the quality and utility of legacy documents.

Previous implementation was circular: it used (r_era_task - 0.15) as the
counterfactual baseline, which always gave +0.15 regardless of document content.

New implementation:
    R_legacy_utility = structural_score × 0.35
                     + drift_capture_score × 0.40
                     + actionability_score × 0.25

Each component is independently measurable from the document text and drift state.
This makes the reward depend on CONTENT QUALITY, not just document existence.

Range: 0.0 to 1.0 (no negative; absence of legacy gives 0.0).
"""
from __future__ import annotations

import re
from typing import List


REQUIRED_SECTIONS = [
    "SECTION 1",  # World state
    "SECTION 2",  # Trust ratings
    "SECTION 3",  # Drift events detected
    "SECTION 4",  # Key decisions
    "SECTION 5",  # Open issues
    "SECTION 6",  # Recommended next-era actions
]

# Tokens that indicate actionable content in the legacy doc
ACTIONABILITY_PATTERNS = [
    r"call\s+get_",      # Tool call recommendations
    r"call\s+query_",
    r"call\s+send_",
    r"check\s+\w+.api",
    r"section\s+6",      # "Section 6" present implies next-era recommendations
    r"first action",
    r"next era",
    r"recommend",
    r"verify",
    r"watch out",
    r"caution",
    r"drift",            # Any mention of drift transfers knowledge
    r"schema",
    r"changed",
    r"renamed",
    r"cursor",
    r"pagination",
    r"rate.limit",
]


def compute_legacy_utility_reward(
    doc_text: str,
    actual_drifts: List[dict],
    structural_score: float = None,
) -> float:
    """
    Compute legacy document utility from content quality.

    Args:
        doc_text:         Full legacy document text (or "" if no doc).
        actual_drifts:    List of drift event dicts that fired this era.
        structural_score: Pre-computed section compliance score (0.0–1.0).
                          If None, computed internally.

    Returns:
        R_legacy_utility in [0.0, 1.0].
    """
    if not doc_text or not doc_text.strip():
        return 0.0

    # ── Component 1: Structural score (35% weight) ────────────────────────────
    if structural_score is None:
        structural_score = _compute_structural_score(doc_text)

    # ── Component 2: Drift capture score (40% weight) ────────────────────────
    drift_capture = _compute_drift_capture(doc_text, actual_drifts)

    # ── Component 3: Actionability score (25% weight) ────────────────────────
    actionability = _compute_actionability(doc_text)

    total = (
        structural_score * 0.35
        + drift_capture * 0.40
        + actionability * 0.25
    )

    return round(min(1.0, max(0.0, total)), 4)


def compute_legacy_utility_reward_breakdown(
    doc_text: str,
    actual_drifts: List[dict],
    structural_score: float = None,
) -> dict:
    """Returns each sub-component for logging/debugging."""
    if not doc_text or not doc_text.strip():
        return {"structural": 0.0, "drift_capture": 0.0, "actionability": 0.0, "total": 0.0}

    if structural_score is None:
        structural_score = _compute_structural_score(doc_text)
    drift_capture = _compute_drift_capture(doc_text, actual_drifts)
    actionability = _compute_actionability(doc_text)
    total = structural_score * 0.35 + drift_capture * 0.40 + actionability * 0.25

    return {
        "structural": round(structural_score, 4),
        "drift_capture": round(drift_capture, 4),
        "actionability": round(actionability, 4),
        "total": round(min(1.0, max(0.0, total)), 4),
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_structural_score(doc_text: str) -> float:
    """Fraction of required sections present (same as legacy_parser compliance)."""
    found = 0
    doc_upper = doc_text.upper()
    for section in REQUIRED_SECTIONS:
        if section in doc_upper:
            found += 1
    return found / len(REQUIRED_SECTIONS)


def _compute_drift_capture(doc_text: str, actual_drifts: List[dict]) -> float:
    """
    How well does the legacy doc capture the actual drift events?

    If no drifts fired: full score (correctly representing a clean era).
    If drifts fired: proportional to which drifts are mentioned.
    """
    if not actual_drifts:
        # No drifts — perfect score if doc exists (correctly records stable state)
        return 1.0

    doc_lower = doc_text.lower()
    captured = 0

    for drift in actual_drifts:
        service = drift.get("target_service", "").lower().replace("-", " ")
        behaviour = drift.get("drifted_behaviour") or {}
        field = str(behaviour.get("field", "")).lower()
        drift_reason = drift.get("drift_reason", "").lower()

        # Service name mentioned
        service_hit = service and service.replace(" ", "") in doc_lower.replace("-", "").replace(" ", "")
        # Affected field mentioned
        field_hit = field and field in doc_lower
        # Any word from drift_reason
        reason_words = [w for w in drift_reason.split() if len(w) > 4]
        reason_hit = any(w in doc_lower for w in reason_words)

        if service_hit and (field_hit or reason_hit):
            captured += 1
        elif service_hit or field_hit:
            captured += 0.5  # Partial credit

    return min(1.0, captured / len(actual_drifts))


def _compute_actionability(doc_text: str) -> float:
    """
    Does the doc give the next agent useful actions to take?

    Checks for actionable patterns and minimum content length.
    """
    if len(doc_text.strip()) < 200:
        return 0.0  # Too short to be actionable

    doc_lower = doc_text.lower()
    hits = sum(
        1 for pattern in ACTIONABILITY_PATTERNS
        if re.search(pattern, doc_lower)
    )

    # Cap at 1.0: hitting 4+ patterns = full actionability score
    return min(1.0, hits / 4)
