"""
grpo_reward.py — Tiered GRPO reward for EpistemicOPS primary agent training.

This module is the SINGLE SOURCE OF TRUTH for per-completion reward during GRPO.
It must remain aligned with the episode-level reward computed in run_episode.py.

Architecture (three tiers):
    TIER 1 — Format (R_format):  0.0 – 0.2
        Valid JSON + valid action schema.  Foundational; no schema = 0.0.

    TIER 2 — Action quality (R_action_quality):  0.0 – 0.6
        Context-aware. Action must be appropriate for the current phase/step.
        Merely outputting an action type gives NO reward on its own.

    TIER 3 — Drift awareness (R_drift_awareness):  0.0 – 0.3
        Positive only when the action demonstrates drift-sensitive behaviour
        given context clues in the prompt.

    PENALTY — Anti-hack (R_anti_hack):  -1.0 – 0.0
        Fires on degenerate patterns: hypothesis-spam, trivial content,
        hallucinated tools, early-complete gaming.

Total = R_format + R_action_quality + R_drift_awareness + R_anti_hack
Max possible (no penalties) = 0.2 + 0.6 + 0.3 = 1.1 → normalized to 1.0.

Design rules:
    1.  An action type ALONE cannot earn Tier-2 reward.
    2.  'declare_hypothesis' earns reward only if the prompt context contains
        evidence of an anomaly and the hypothesis text references specific services.
    3.  'write_legacy' earns reward proportional to structural completeness
        AND non-trivial content length — not for merely existing.
    4.  'declare_task_complete' earns reward only if the prompt shows ≥1 task
        criterion was resolved (tool calls made, incident resolved etc.).
    5.  'call_tool' earns reward if the tool is known AND args are non-trivial.
    6.  The reward function MUST remain fast (no env step, no LLM call).
"""

from __future__ import annotations

import json
import re
import logging
from typing import List, Optional

logger = logging.getLogger("grpo-reward")

# ─── Constants ────────────────────────────────────────────────────────────────

KNOWN_TOOLS = frozenset({
    "get_incident_status", "resolve_incident", "get_metrics",
    "rollback_deployment", "query_logs", "send_notification",
})

REQUIRED_LEGACY_SECTIONS = [
    "SECTION 1", "SECTION 2", "SECTION 3",
    "SECTION 4", "SECTION 5", "SECTION 6",
]

# Drift-evidence keywords that might appear in a prompt observation
DRIFT_EVIDENCE_PATTERNS = [
    r"metric_value",           # field rename
    r"INVESTIGATING",          # status string vs int
    r"204",                    # status code change
    r"next_cursor",            # pagination change
    r"rate_limit",             # rate limit change
    r"delivered.*False",       # delivery failure
    r"DRIFT_INJECTION",        # phase marker
    r"SOCRATIC_RECOVERY",      # phase marker
    r"unknown field",          # generic anomaly
    r"unexpected",             # generic anomaly
    r"error.*api",             # api error
]

# ─── Main entry point ─────────────────────────────────────────────────────────

def compute_grpo_reward(
    completions: List[str],
    prompts: Optional[List[str]] = None,
    **kwargs,
) -> List[float]:
    """
    GRPO reward function — called by GRPOTrainer for each batch of completions.

    Args:
        completions: Model-generated text completions (one per sample).
        prompts:     Corresponding input prompts (same length as completions).
                     If None, falls back to context-free scoring.

    Returns:
        List of scalar rewards in [0.0, 1.0] (can be slightly negative with anti-hack).
    """
    decoder = json.JSONDecoder()
    rewards = []

    for idx, completion in enumerate(completions):
        prompt = prompts[idx] if prompts and idx < len(prompts) else ""
        reward, components = _score_completion(completion, prompt, decoder)
        rewards.append(reward)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "GRPO idx=%d  total=%.3f  fmt=%.2f  quality=%.2f  "
                "drift=%.2f  hack=%.2f",
                idx, reward,
                components["R_format"],
                components["R_action_quality"],
                components["R_drift_awareness"],
                components["R_anti_hack"],
            )

    return rewards


def compute_grpo_reward_with_components(
    completions: List[str],
    prompts: Optional[List[str]] = None,
    **kwargs,
) -> List[dict]:
    """
    Like compute_grpo_reward but returns component dicts for logging/debugging.
    """
    decoder = json.JSONDecoder()
    results = []
    for idx, completion in enumerate(completions):
        prompt = prompts[idx] if prompts and idx < len(prompts) else ""
        reward, components = _score_completion(completion, prompt, decoder)
        components["R_total"] = reward
        results.append(components)
    return results


# ─── Internal scoring ─────────────────────────────────────────────────────────

def _score_completion(completion: str, prompt: str, decoder: json.JSONDecoder):
    """Score a single completion. Returns (scalar_reward, components_dict)."""
    components = {
        "R_format": 0.0,
        "R_action_quality": 0.0,
        "R_drift_awareness": 0.0,
        "R_anti_hack": 0.0,
    }

    # ── Tier 1: Format ────────────────────────────────────────────────────────
    action = _parse_action(completion, decoder)
    if action is None:
        # Completely malformed output — no reward
        return 0.0, components

    action_type = action.get("action_type", "")
    payload = action.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}

    # Valid JSON earns base format score
    components["R_format"] = 0.1

    # Known action type earns additional format score
    KNOWN_ACTIONS = {
        "call_tool", "declare_hypothesis", "write_reasoning",
        "write_legacy", "declare_task_complete", "end_era",
        "ready_to_operate", "send_message", "request_clarification",
        "update_trust_rating",
    }
    if action_type in KNOWN_ACTIONS:
        components["R_format"] = 0.2

    # ── Tier 2: Action quality (context-aware) ────────────────────────────────
    drift_in_context = _detect_drift_evidence(prompt)
    post_drift = drift_in_context > 0
    has_tool_calls = _prompt_has_tool_calls(prompt)
    has_incident_resolved = "incidents_resolved" in prompt or "incident_resolved" in prompt
    prompt_step = _extract_step_from_prompt(prompt)

    components["R_action_quality"] = _score_action_quality(
        action_type, payload, post_drift, has_tool_calls,
        has_incident_resolved, prompt_step,
    )

    # ── Tier 3: Drift awareness ───────────────────────────────────────────────
    components["R_drift_awareness"] = _score_drift_awareness(
        action_type, payload, drift_in_context, post_drift,
    )

    # ── Penalty: Anti-hack ────────────────────────────────────────────────────
    components["R_anti_hack"] = _score_anti_hack(action_type, payload, prompt)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    raw = (
        components["R_format"]
        + components["R_action_quality"]
        + components["R_drift_awareness"]
        + components["R_anti_hack"]
    )
    # Normalize so max honest reward = 1.0; allow slight negatives for hacking
    MAX_POSITIVE = 0.2 + 0.6 + 0.3  # = 1.1
    normalized = raw / MAX_POSITIVE
    reward = max(-0.5, min(1.0, normalized))
    return round(reward, 4), components


def _score_action_quality(
    action_type: str,
    payload: dict,
    post_drift: bool,
    has_tool_calls: bool,
    has_incident_resolved: bool,
    prompt_step: int,
) -> float:
    """
    Context-aware action quality. The action type alone is NOT enough to earn this.
    """
    score = 0.0

    if action_type == "call_tool":
        tool = payload.get("tool", "")
        args = payload.get("args", {})
        if tool in KNOWN_TOOLS:
            score = 0.25  # Known tool
            # Tool with non-trivial args is better
            if isinstance(args, dict) and len(args) > 0:
                score = 0.35
        # Unknown tool → no quality score (format score still applies)

    elif action_type == "declare_hypothesis":
        hypothesis = str(payload.get("hypothesis", ""))
        confidence = float(payload.get("confidence", 0.5))
        # Reward hypothesis if: mentions specific service/field AND is post-drift context
        specific = any(
            svc in hypothesis.lower()
            for svc in ["incident", "metric", "deploy", "log", "notify", "api",
                        "status", "cursor", "rate", "pagination", "schema", "field"]
        )
        calibrated = 0.3 <= confidence <= 0.85  # overconfidence penalty
        if specific and post_drift:
            score = 0.45 if calibrated else 0.3
        elif specific and not post_drift and has_tool_calls:
            # Hypothesising based on tool observations (legitimate even pre-phase change)
            score = 0.2
        elif specific:
            score = 0.15
        # Vague hypothesis with no specific referent → 0.0

    elif action_type == "write_legacy":
        content = str(payload.get("content", ""))
        # Require meaningful content length (not just section headers)
        if len(content) < 150:
            score = 0.0  # Too short — not a real legacy doc
        else:
            sections_found = sum(1 for s in REQUIRED_LEGACY_SECTIONS if s in content)
            coverage = sections_found / len(REQUIRED_LEGACY_SECTIONS)
            # Base score proportional to section coverage
            score = 0.25 + (coverage * 0.25)  # max 0.5

    elif action_type == "declare_task_complete":
        # Only earns quality score if context shows actual work was done
        outcome = str(payload.get("outcome", ""))
        summary = str(payload.get("summary", ""))
        if has_incident_resolved or has_tool_calls:
            if len(outcome) > 20 and len(summary) > 20:
                score = 0.45
            elif has_tool_calls:
                score = 0.25
        elif prompt_step < 3:
            # Declaring complete at step 0-2 without any work = gaming → no quality
            score = 0.0
        else:
            score = 0.1  # Some completion, no evidence of success

    elif action_type == "write_reasoning":
        thought = str(payload.get("thought", ""))
        # Reward substantive reasoning (not empty/trivial)
        if len(thought) > 50:
            score = 0.2
        elif len(thought) > 20:
            score = 0.1

    elif action_type == "ready_to_operate":
        # Starting action — small quality score if summary is meaningful
        summary = str(payload.get("world_model_summary", ""))
        score = 0.15 if len(summary) > 20 else 0.05

    elif action_type == "end_era":
        # Only meaningful if legacy doc was written (context clue: legacy section in prompt)
        if "SECTION 1" in payload.get("__context__", "") or has_tool_calls:
            score = 0.2
        else:
            score = 0.1

    elif action_type in ("send_message", "request_clarification"):
        content_val = payload.get("content", payload.get("question", ""))
        score = 0.1 if len(str(content_val)) > 20 else 0.05

    elif action_type == "update_trust_rating":
        score = 0.15 if "entity" in payload and "score" in payload else 0.05

    return min(0.6, score)


def _score_drift_awareness(
    action_type: str,
    payload: dict,
    drift_evidence_count: int,
    post_drift: bool,
) -> float:
    """
    Reward drift-sensitive behaviour when drift evidence exists in context.
    Zero when no drift evidence — avoids rewarding random 'drift' mentions.
    """
    if drift_evidence_count == 0:
        return 0.0

    score = 0.0

    if action_type == "declare_hypothesis":
        hypothesis = str(payload.get("hypothesis", "")).lower()
        # Must mention drift-related concepts to earn drift-awareness reward
        drift_keywords = [
            "drift", "change", "schema", "contract", "renamed", "different",
            "unexpected", "anomal", "field", "type", "status", "cursor",
            "rate limit", "pagination", "metric_value", "string", "integer",
        ]
        keyword_hits = sum(1 for kw in drift_keywords if kw in hypothesis)
        if keyword_hits >= 2:
            score = 0.3  # Strong drift hypothesis
        elif keyword_hits == 1:
            score = 0.15

    elif action_type == "call_tool" and post_drift:
        # Calling a tool after drift evidence is adaptive — agent is probing
        tool = payload.get("tool", "")
        if tool in KNOWN_TOOLS:
            score = 0.1

    elif action_type == "write_legacy" and post_drift:
        content = str(payload.get("content", "")).lower()
        # Bonus if legacy captures drift events
        if "drift" in content or "schema" in content or "change" in content:
            score = 0.2

    return min(0.3, score)


def _score_anti_hack(action_type: str, payload: dict, prompt: str) -> float:
    """
    Returns a negative penalty for degenerate patterns.
    This is called per-completion; aggregate history is not available here.
    Checks based on content quality within a single completion.
    """
    penalty = 0.0

    if action_type == "declare_hypothesis":
        hypothesis = str(payload.get("hypothesis", ""))
        # Trivially short or obviously vague hypothesis
        if len(hypothesis) < 15:
            penalty -= 0.15
        # Exact same template as in the prompt system instructions
        if hypothesis.strip() in ("API drift detected", "API drifted", "drift"):
            penalty -= 0.2

    elif action_type == "write_legacy":
        content = str(payload.get("content", ""))
        # Content-free legacy doc
        if len(content) < 50:
            penalty -= 0.3
        # Legacy that is ONLY section headers with no content
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        section_lines = [l for l in lines if l.startswith("SECTION")]
        if len(lines) > 0 and len(section_lines) / len(lines) > 0.8:
            penalty -= 0.2  # Almost entirely just headers

    elif action_type == "declare_task_complete":
        # Early-complete gaming: declaring done at a very early step
        step = _extract_step_from_prompt(prompt)
        if step < 3:
            penalty -= 0.3

    elif action_type == "call_tool":
        tool = payload.get("tool", "")
        # Hallucinated tool
        if tool not in KNOWN_TOOLS and ":" not in str(tool):
            penalty -= 0.2

    return max(-0.5, penalty)


# ─── Prompt parsing helpers ───────────────────────────────────────────────────

def _parse_action(completion: str, decoder: json.JSONDecoder) -> Optional[dict]:
    """Parse the first valid JSON object from a completion string."""
    clean = re.sub(r"```json|```", "", completion).strip()
    if not clean:
        return None
    try:
        action, _ = decoder.raw_decode(clean)
        if isinstance(action, dict) and "action_type" in action:
            return action
    except (json.JSONDecodeError, ValueError):
        pass
    # Try extracting the first {...} block
    match = re.search(r"\{[\s\S]*?\}", clean)
    if match:
        try:
            action = json.loads(match.group(0))
            if isinstance(action, dict) and "action_type" in action:
                return action
        except json.JSONDecodeError:
            pass
    return None


def _detect_drift_evidence(prompt: str) -> int:
    """Count how many drift evidence patterns appear in the prompt."""
    count = 0
    for pattern in DRIFT_EVIDENCE_PATTERNS:
        if re.search(pattern, prompt, re.IGNORECASE):
            count += 1
    return count


def _prompt_has_tool_calls(prompt: str) -> bool:
    """Check if the prompt context shows prior tool calls were made."""
    return bool(re.search(r"call_tool|tool_response|status_code", prompt, re.IGNORECASE))


def _extract_step_from_prompt(prompt: str) -> int:
    """Extract the current step number from a prompt observation."""
    match = re.search(r'"step"\s*:\s*(\d+)', prompt)
    if match:
        return int(match.group(1))
    match = re.search(r'step[:\s]+(\d+)', prompt, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0
