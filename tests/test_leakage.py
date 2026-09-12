"""
test_leakage.py — Tests proving hidden ground truth cannot leak into agent observations.

These tests implement the information boundary guarantee:
    GROUND TRUTH (internal drift state)
        ↓
    ENVIRONMENT INTERNAL STATE (WorldState.drift_events_fired)
        ↓ BLOCKED
    OBSERVATION (what agent sees)
        ↓
    ONLY INFORMATION A REAL AGENT SHOULD SEE (tool responses, phase, message)
"""
import asyncio
import os
import pytest

os.environ.setdefault("EPISTEMICOPS_OFFLINE", "true")

from environment.openenv_wrapper import EpistemicOpsEnv
from environment.scenario_loader import ScenarioLoader


@pytest.fixture
def env():
    return EpistemicOpsEnv(offline=True)


@pytest.fixture
def cascading_config():
    loader = ScenarioLoader()
    sc = loader.get_scenario("cascading_incident")
    assert sc is not None, "cascading_incident scenario must exist"
    return sc.model_dump()


# ── Test 1: AWAKENING observation has no drift info ───────────────────────────

def test_awakening_obs_has_no_drift_count(env, cascading_config):
    """Immediately after reset, observation must not contain any drift count or drift state."""
    obs = env.reset(cascading_config, era_id=1)

    FORBIDDEN = {"drifts_detected", "drift_events_fired", "drift_count",
                 "active_drifts", "internal_drift_state"}
    for key in FORBIDDEN:
        assert key not in obs, (
            f"Observation leaked '{key}' on AWAKENING — hidden drift state exposed to agent."
        )


# ── Test 2: observation after tool call has no internal drift counter ─────────

@pytest.mark.asyncio
async def test_tool_call_obs_has_no_drift_count(env, cascading_config):
    """After a tool call, the observation must NOT contain the internal drift count."""
    env.reset(cascading_config, era_id=1)

    # Move to OPERATION phase
    await env.step("primary", {"action_type": "ready_to_operate", "payload": {"world_model_summary": "ready"}})

    # Make a tool call
    obs, _, _, _ = await env.step("primary", {
        "action_type": "call_tool",
        "payload": {"tool": "get_incident_status", "args": {"incident_id": "INC-2041"}}
    })

    FORBIDDEN = {"drifts_detected", "drift_events_fired", "drift_count", "active_drifts"}
    for key in FORBIDDEN:
        assert key not in obs, (
            f"Observation leaked '{key}' after tool call — agent should not see drift count."
        )


# ── Test 3: observation string does not contain drift type names ──────────────

@pytest.mark.asyncio
async def test_obs_does_not_contain_drift_type_identifiers(env, cascading_config):
    """The observation string must not directly name drift event IDs or types."""
    env.reset(cascading_config, era_id=2)
    await env.step("primary", {"action_type": "ready_to_operate", "payload": {"world_model_summary": "ready"}})

    obs, _, _, _ = await env.step("primary", {
        "action_type": "call_tool",
        "payload": {"tool": "get_incident_status", "args": {"incident_id": "INC-2041"}}
    })

    import json
    obs_str = json.dumps(obs)

    # These are internal drift event IDs — never legitimate in an observation
    INTERNAL_IDS = ["DE-001", "DE-002", "DE-005", "DE-006", "DE-007", "DE-008",
                    "DE-MSD-001", "DE-LPC-001", "DRIFT_TYPE", "DRIFT_STATUS",
                    "DRIFT_FIELD_RENAME", "DRIFT_PAGINATION"]
    for drift_id in INTERNAL_IDS:
        assert drift_id not in obs_str, (
            f"Observation contained internal drift ID '{drift_id}' — ground truth leaked!"
        )


# ── Test 4: leakage audit guard fires on forbidden keys ──────────────────────

def test_leakage_audit_guard_fires(env, cascading_config):
    """The _audit_observation_for_leakage guard should raise AssertionError on forbidden keys."""
    env.reset(cascading_config, era_id=1)

    BAD_OBS = {
        "step": 1,
        "phase": "OPERATION",
        "drifts_detected": 2,  # FORBIDDEN: internal drift count
    }

    with pytest.raises(AssertionError, match="LEAKAGE"):
        env._audit_observation_for_leakage(BAD_OBS)


# ── Test 5: phase transition reveals drift occurred (indirect signal only) ────

@pytest.mark.asyncio
async def test_drift_only_visible_through_phase_and_tool_response(env, cascading_config):
    """
    When drift fires, the agent should learn of it ONLY via:
      1. Phase transition to DRIFT_INJECTION or SOCRATIC_RECOVERY
      2. Anomalous tool response (unexpected field/type/status)
    
    NOT via a direct drift counter in the observation.
    """
    env.reset(cascading_config, era_id=3)

    # Advance through steps — drift fires mid-era in era 3
    await env.step("primary", {"action_type": "ready_to_operate", "payload": {"world_model_summary": "ready"}})

    # Run until drift phase or max 30 steps
    for _ in range(30):
        obs, _, done, info = await env.step("primary", {
            "action_type": "call_tool",
            "payload": {"tool": "get_incident_status", "args": {"incident_id": "INC-2089"}}
        })
        if done:
            break
        phase = info.get("phase", "")
        if phase in ("DRIFT_INJECTION", "SOCRATIC_RECOVERY"):
            # Good: phase revealed drift indirectly
            # But obs must still not have internal drift count
            import json
            obs_str = json.dumps(obs)
            assert "drifts_detected" not in obs, "Drift count leaked into observation"
            assert "drift_events_fired" not in obs, "Internal drift list leaked"
            break
