import json
import os
import random
import logging
from openai import OpenAI

logger = logging.getLogger("primary-agent")

class PrimaryAgent:
    """Student agent for SRE task resolution and drift detection."""
    
    SYSTEM_PROMPT = """You are the Primary Agent (Student), an elite enterprise Site Reliability Engineer.
Your task is to resolve the current era's incident using available API tools.

CRITICAL RULES:
1. API contracts can and will change silently (Drift). If a tool call fails, DO NOT assume the code is wrong. Assume the API contract has drifted and test that hypothesis.
2. If you are stuck, you may receive a Socratic message from an Oversight Agent. They will not give you the answer, only guide you.
3. At the end of the era, you MUST write a Legacy Document (max 2048 tokens) to pass knowledge to your successor.
4. Your context memory will be WIPED at the end of this era. Only the Legacy Document survives.

Available Actions (output ONLY a valid JSON object):
- call_tool: {"action_type": "call_tool", "payload": {"tool": str, "args": dict}}
  Tools: get_incident_status, resolve_incident, get_metrics, rollback_deployment, query_logs, send_notification
- write_reasoning: {"action_type": "write_reasoning", "payload": {"thought": str}}
- declare_hypothesis: {"action_type": "declare_hypothesis", "payload": {"hypothesis": str, "confidence": float}}
- send_message: {"action_type": "send_message", "payload": {"recipient": str, "content": str}}
- write_legacy: {"action_type": "write_legacy", "payload": {"content": str}}
  Legacy Document MUST contain these sections:
  SECTION 1: WORLD STATE AT ERA END
  SECTION 2: TRUST RATINGS
  SECTION 3: DRIFT EVENTS DETECTED
  SECTION 4: KEY DECISIONS & RATIONALE
  SECTION 5: OPEN ISSUES & TECHNICAL DEBT
  SECTION 6: RECOMMENDED FIRST ACTIONS FOR ERA N+1
- declare_task_complete: {"action_type": "declare_task_complete", "payload": {"outcome": str, "summary": str}}
- end_era: {"action_type": "end_era", "payload": {}}
- ready_to_operate: {"action_type": "ready_to_operate", "payload": {"world_model_summary": str}}
"""

    def __init__(self, model: str = None):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.model = model or os.getenv("PRIMARY_AGENT_MODEL", "gpt-4o-mini")

    def generate_action(self, observation: dict, conversation_history: list = None) -> dict:
        if observation.get("phase") == "AWAKENING":
            legacy_doc = observation.get("legacy_document", "")
            summary = f"Ready. Legacy doc available: {bool(legacy_doc)}"
            return {
                "action_type": "ready_to_operate",
                "payload": {"world_model_summary": summary}
            }

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        if conversation_history:
            for entry in conversation_history[-20:]:
                if entry.get("role") == "assistant":
                    messages.append({"role": "assistant", "content": json.dumps(entry.get("action", {}))})
                elif entry.get("role") == "environment":
                    obs_summary = self._summarize_observation(entry.get("obs", {}))
                    messages.append({"role": "user", "content": f"Environment Response:\n{obs_summary}"})
                elif entry.get("role") == "oversight":
                    messages.append({"role": "user", "content": f"Oversight Agent (Teacher) Message:\n{entry.get('msg', '')}"})
        messages.append({"role": "user", "content": f"Current Observation:\n{json.dumps(observation, indent=2)}\n\nOutput ONLY a valid JSON action object."})
        
        if not self.client:
            return self._mock_action(observation, conversation_history)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=512
            )
            action_text = response.choices[0].message.content
            return json.loads(action_text)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {
                "action_type": "write_reasoning",
                "payload": {"thought": f"Error during generation: {str(e)}"}
            }

    def _summarize_observation(self, obs: dict) -> str:
        parts = []
        if obs.get("message"):
            parts.append(f"Message: {obs['message']}")
        if obs.get("tool_response"):
            tr = obs["tool_response"]
            parts.append(f"Tool Response: status={tr.get('status_code', 'N/A')}, body={json.dumps(tr.get('body', {}))[:200]}")
        if obs.get("oversight_message"):
            parts.append(f"Oversight: {obs['oversight_message'].get('content', '')}")
        return "\n".join(parts) if parts else json.dumps(obs)[:300]

    def _mock_action(self, observation: dict, conversation_history: list = None) -> dict:
        step = observation.get("step", 0)
        phase = observation.get("phase", "OPERATION")
        era_id = observation.get("era_id", 1)
        task_brief = observation.get("era_task_brief", "")
        history_len = len(conversation_history or [])
        has_drift = observation.get("drifts_detected", 0) > 0
        has_oversight = observation.get("oversight_message", {}).get("present", False)
        
        recovery_actions = 0
        if conversation_history:
            for entry in reversed(conversation_history):
                if entry.get("role") == "oversight":
                    break
                if entry.get("role") == "assistant":
                    recovery_actions += 1

        if phase == "SOCRATIC_RECOVERY":
            if has_oversight:
                return {
                    "action_type": "declare_hypothesis",
                    "payload": {
                        "hypothesis": "Based on the teacher's guidance, the API contract may have changed — "
                                      "the response schema fields may have been renamed or their types altered. "
                                      "I should re-test the API and compare the actual response to my expectations.",
                        "confidence": random.choice([0.55, 0.6, 0.65, 0.7, 0.75])
                    }
                }
            recovery_sequence = [
                self._get_scenario_probe_action(task_brief, era_id),
                {"action_type": "write_reasoning", "payload": {"thought": "After re-examining the API response with fresh eyes, I can see the schema has changed. I'll adapt my approach to handle the new format."}},
                {"action_type": "declare_hypothesis", "payload": {"hypothesis": f"Confirmed: API drift detected. Adapting approach for era {era_id}.", "confidence": 0.85}},
                self._get_scenario_resolve_action(task_brief, era_id),
                {"action_type": "declare_task_complete", "payload": {"outcome": "resolved_after_recovery", "summary": f"Recovered from API drift in era {era_id} with Socratic guidance. Adapted to new schema."}},
            ]
            idx = min(recovery_actions, len(recovery_sequence) - 1)
            return recovery_sequence[idx]
        
        if phase == "LEGACY_GENERATION":
            msg = observation.get("message", "")
            recent_actions = [h.get("action", {}).get("action_type")
                            for h in (conversation_history or [])[-3:]]
            if "Legacy document saved" in msg or "write_legacy" in recent_actions:
                return {"action_type": "end_era", "payload": {}}
            return {
                "action_type": "write_legacy",
                "payload": {"content": self._generate_mock_legacy_doc(observation, has_drift)}
            }
        
        if phase == "DRIFT_INJECTION":
            drift_actions = 0
            if conversation_history:
                for entry in reversed(conversation_history):
                    obs_entry = entry.get("obs", {})
                    if isinstance(obs_entry, dict) and obs_entry.get("phase") != "DRIFT_INJECTION":
                        break
                    if entry.get("role") == "assistant":
                        drift_actions += 1
            drift_sequence = [
                self._get_scenario_probe_action(task_brief, era_id),
                {"action_type": "write_reasoning", "payload": {"thought": f"Step {step}: The API response looks different from what I expected. Fields may have been renamed or types changed. This could be a schema drift."}},
                self._get_scenario_probe_action(task_brief, era_id),
            ]
            return drift_sequence[min(drift_actions, len(drift_sequence) - 1)]

        if phase == "OPERATION":
            return self._get_operation_action(step, era_id, task_brief, history_len, observation)

        return {
            "action_type": "write_reasoning",
            "payload": {"thought": f"Analyzing at step {step}, phase {phase}, era {era_id}."}
        }

    def _get_operation_action(self, step: int, era_id: int, task_brief: str,
                              history_len: int, observation: dict) -> dict:
        is_incident_era = "incident" in task_brief.lower() or "INC-" in task_brief
        is_deploy_era = "deploy" in task_brief.lower() or "rollback" in task_brief.lower()
        is_metrics_era = "metric" in task_brief.lower() or "latency" in task_brief.lower() or "outage" in task_brief.lower()
        is_automation_era = "automat" in task_brief.lower() or "harden" in task_brief.lower()
        
        # Era-specific sequences that extend past typical drift windows
        if is_incident_era:
            sequence = self._incident_sequence(step, history_len, era_id)
        elif is_deploy_era:
            sequence = self._deploy_sequence(step, history_len, era_id)
        elif is_metrics_era:
            sequence = self._metrics_sequence(step, history_len, era_id)
        elif is_automation_era:
            sequence = self._automation_sequence(step, history_len, era_id)
        else:
            sequence = self._generic_sequence(step, history_len, era_id)
        
        return sequence

    def _incident_sequence(self, step: int, history_len: int, era_id: int) -> dict:
        incident_id = f"INC-{2041 + (era_id - 1) * 48}"
        actions = [
            {"action_type": "call_tool", "payload": {"tool": "get_incident_status", "args": {"incident_id": incident_id}}},
            {"action_type": "write_reasoning", "payload": {"thought": f"Checking incident {incident_id}. Need to understand current status and severity."}},
            {"action_type": "call_tool", "payload": {"tool": "get_metrics", "args": {"service_name": "payment-service", "metric": "latency"}}},
            {"action_type": "declare_hypothesis", "payload": {"hypothesis": "The incident may be related to connection pool exhaustion based on latency patterns", "confidence": 0.65}},
            {"action_type": "call_tool", "payload": {"tool": "query_logs", "args": {"service": "payment-service", "level": "ERROR"}}},
            {"action_type": "write_reasoning", "payload": {"thought": "Logs confirm connection pool errors. Checking notification channels."}},
            {"action_type": "call_tool", "payload": {"tool": "send_notification", "args": {"channel": "slack", "message": f"Investigating {incident_id}: potential connection pool exhaustion"}}},
            {"action_type": "call_tool", "payload": {"tool": "get_incident_status", "args": {"incident_id": incident_id}}},
            {"action_type": "write_reasoning", "payload": {"thought": "Re-checking incident status. The response format looks different from before — possible API drift."}},
            {"action_type": "declare_hypothesis", "payload": {"hypothesis": "The incident-api schema may have drifted: status field type appears to have changed from integer to string enum", "confidence": 0.7}},
            {"action_type": "call_tool", "payload": {"tool": "get_metrics", "args": {"service_name": "payment-service", "metric": "error_rate"}}},
            {"action_type": "call_tool", "payload": {"tool": "resolve_incident", "args": {"incident_id": incident_id, "resolution_notes": "Fixed via connection pool resize and API adaptation", "resolved_by": "primary-agent"}}},
            {"action_type": "declare_task_complete", "payload": {"outcome": "resolved", "summary": f"Resolved {incident_id}. Detected potential API drift in incident-api."}},
        ]
        idx = min(history_len // 2, len(actions) - 1)
        return actions[idx]

    def _deploy_sequence(self, step: int, history_len: int, era_id: int) -> dict:
        actions = [
            {"action_type": "call_tool", "payload": {"tool": "get_metrics", "args": {"service_name": "payment-service", "metric": "error_rate"}}},
            {"action_type": "write_reasoning", "payload": {"thought": "Checking error rates post-deployment. Need to determine if rollback is necessary."}},
            {"action_type": "call_tool", "payload": {"tool": "query_logs", "args": {"service": "payment-service", "level": "ERROR"}}},
            {"action_type": "declare_hypothesis", "payload": {"hypothesis": "The deployment may have introduced a regression. Errors elevated since deploy.", "confidence": 0.6}},
            {"action_type": "call_tool", "payload": {"tool": "rollback_deployment", "args": {"deployment_id": f"dep-{era_id:03d}", "token": "deploy-token-v2"}}},
            {"action_type": "write_reasoning", "payload": {"thought": "Rollback response was unexpected — may be a deploy-api drift (DE-002: 200→204)."}},
            {"action_type": "declare_hypothesis", "payload": {"hypothesis": "deploy-api rollback endpoint may have changed response code from 200 with body to 204 without body", "confidence": 0.75}},
            {"action_type": "call_tool", "payload": {"tool": "get_metrics", "args": {"service_name": "payment-service", "metric": "latency"}}},
            {"action_type": "call_tool", "payload": {"tool": "send_notification", "args": {"channel": "slack", "message": "Rollback completed. Monitoring for stability."}}},
            {"action_type": "declare_task_complete", "payload": {"outcome": "rollback_successful", "summary": "Rolled back deployment. Detected deploy-api schema drift."}},
        ]
        idx = min(history_len // 2, len(actions) - 1)
        return actions[idx]

    def _metrics_sequence(self, step: int, history_len: int, era_id: int) -> dict:
        actions = [
            {"action_type": "call_tool", "payload": {"tool": "get_metrics", "args": {"service_name": "payment-service", "metric": "latency"}}},
            {"action_type": "write_reasoning", "payload": {"thought": "Checking latency metrics. Services appear healthy but may have silent regression."}},
            {"action_type": "call_tool", "payload": {"tool": "get_incident_status", "args": {"incident_id": "INC-2089"}}},
            {"action_type": "declare_hypothesis", "payload": {"hypothesis": "A silent latency regression may be occurring. The metrics look normal but the field names might have changed.", "confidence": 0.5}},
            {"action_type": "call_tool", "payload": {"tool": "query_logs", "args": {"service": "payment-service", "level": "WARN"}}},
            {"action_type": "call_tool", "payload": {"tool": "get_metrics", "args": {"service_name": "api-gateway", "metric": "latency"}}},
            {"action_type": "write_reasoning", "payload": {"thought": "The metrics response structure looks different. The 'value' field may have been renamed to 'metric_value'. This is a metrics-api drift."}},
            {"action_type": "declare_hypothesis", "payload": {"hypothesis": "metrics-api has drifted: datapoints[].value has been renamed to metric_value (DE-005)", "confidence": 0.8}},
            {"action_type": "call_tool", "payload": {"tool": "send_notification", "args": {"channel": "pagerduty", "message": "Silent latency regression detected. Metrics API schema drift confirmed."}}},
            {"action_type": "declare_task_complete", "payload": {"outcome": "identified", "summary": "Identified invisible outage caused by metrics-api field rename drift."}},
        ]
        idx = min(history_len // 2, len(actions) - 1)
        return actions[idx]

    def _automation_sequence(self, step: int, history_len: int, era_id: int) -> dict:
        actions = [
            {"action_type": "call_tool", "payload": {"tool": "get_incident_status", "args": {"incident_id": "INC-2090"}}},
            {"action_type": "call_tool", "payload": {"tool": "get_metrics", "args": {"service_name": "payment-service", "metric": "error_rate"}}},
            {"action_type": "write_reasoning", "payload": {"thought": "Reviewing system state before automating incident response. Need to verify API stability."}},
            {"action_type": "call_tool", "payload": {"tool": "query_logs", "args": {"service": "payment-service", "level": "INFO"}}},
            {"action_type": "declare_hypothesis", "payload": {"hypothesis": "The API contracts are currently stable. Safe to proceed with automation.", "confidence": 0.55}},
            {"action_type": "call_tool", "payload": {"tool": "send_notification", "args": {"channel": "slack", "message": "Starting automation deployment for incident response"}}},
            {"action_type": "write_reasoning", "payload": {"thought": "Notification delivery seems unreliable — possible rate limit change (DE-007). Need to handle this in automation."}},
            {"action_type": "declare_hypothesis", "payload": {"hypothesis": "notify-api rate limit may have changed. Automation must include retry logic.", "confidence": 0.65}},
            {"action_type": "call_tool", "payload": {"tool": "get_metrics", "args": {"service_name": "api-gateway", "metric": "throughput"}}},
            {"action_type": "declare_task_complete", "payload": {"outcome": "automated", "summary": "Automation deployed with drift-resilient retry logic."}},
        ]
        idx = min(history_len // 2, len(actions) - 1)
        return actions[idx]

    def _generic_sequence(self, step: int, history_len: int, era_id: int) -> dict:
        actions = [
            {"action_type": "call_tool", "payload": {"tool": "get_incident_status", "args": {"incident_id": "INC-2041"}}},
            {"action_type": "call_tool", "payload": {"tool": "get_metrics", "args": {"service_name": "payment-service", "metric": "latency"}}},
            {"action_type": "declare_hypothesis", "payload": {"hypothesis": "System may have experienced schema drift in one or more APIs", "confidence": 0.5}},
            {"action_type": "call_tool", "payload": {"tool": "query_logs", "args": {"service": "payment-service", "level": "ERROR"}}},
            {"action_type": "call_tool", "payload": {"tool": "resolve_incident", "args": {"incident_id": "INC-2041", "resolution_notes": "Resolved after investigation", "resolved_by": "primary-agent"}}},
            {"action_type": "declare_task_complete", "payload": {"outcome": "resolved", "summary": "Investigation complete."}},
        ]
        idx = min(history_len // 2, len(actions) - 1)
        return actions[idx]

    def _get_scenario_probe_action(self, task_brief: str, era_id: int) -> dict:
        if "incident" in task_brief.lower():
            return {"action_type": "call_tool", "payload": {"tool": "get_incident_status", "args": {"incident_id": f"INC-{2041 + (era_id - 1) * 48}"}}}
        elif "deploy" in task_brief.lower():
            return {"action_type": "call_tool", "payload": {"tool": "rollback_deployment", "args": {"deployment_id": f"dep-{era_id:03d}", "token": "deploy-token-v2"}}}
        elif "metric" in task_brief.lower() or "latency" in task_brief.lower():
            return {"action_type": "call_tool", "payload": {"tool": "get_metrics", "args": {"service_name": "payment-service", "metric": "latency"}}}
        else:
            return {"action_type": "call_tool", "payload": {"tool": "get_incident_status", "args": {"incident_id": "INC-2041"}}}

    def _get_scenario_resolve_action(self, task_brief: str, era_id: int) -> dict:
        if "incident" in task_brief.lower():
            return {"action_type": "call_tool", "payload": {"tool": "resolve_incident", "args": {"incident_id": f"INC-{2041 + (era_id - 1) * 48}", "resolution_notes": "Resolved after adapting to schema drift", "resolved_by": "primary-agent"}}}
        elif "deploy" in task_brief.lower():
            return {"action_type": "call_tool", "payload": {"tool": "rollback_deployment", "args": {"deployment_id": f"dep-{era_id:03d}", "token": "deploy-token-v2"}}}
        elif "metric" in task_brief.lower() or "latency" in task_brief.lower():
            return {"action_type": "call_tool", "payload": {"tool": "send_notification", "args": {"channel": "pagerduty", "message": "Drift confirmed and mitigated. Monitoring resumed."}}}
        else:
            return {"action_type": "call_tool", "payload": {"tool": "resolve_incident", "args": {"incident_id": "INC-2041", "resolution_notes": "Resolved", "resolved_by": "primary-agent"}}}

    def _generate_mock_legacy_doc(self, observation: dict, has_drift: bool = False) -> str:
        era_id = observation.get("era_id", observation.get("step", 1))
        drifts_count = observation.get("drifts_detected", 0)
        task = observation.get("era_task_brief", "Unknown task")

        drift_section = ""
        if has_drift or drifts_count > 0:
            drift_section = """CONFIRMED: API schema drift detected during this era.
- incident-api: status field type may have changed (integer → string enum)
- deploy-api: rollback endpoint may return 204 instead of 200
- metrics-api: 'value' field may have been renamed to 'metric_value'
Successor MUST verify response schemas before parsing.
Trust ratings for affected services should be downgraded."""
        else:
            drift_section = """No confirmed drift events detected during this era.
However, monitor for potential silent changes in:
- API response schemas (field renames, type changes)
- HTTP status codes (200→204 transitions)
- Pagination formats (offset→cursor)"""

        return f"""# SECTION 1: WORLD STATE AT ERA END
Era {era_id} completed. Task: {task}
Services investigated: incident-api, metrics-api, deploy-api, log-api, notify-api.
Overall system stability: {'DEGRADED — drift detected' if has_drift else 'STABLE — monitoring'}.

# SECTION 2: TRUST RATINGS
oncall_engineer: {round(0.6 + random.uniform(0, 0.3), 1)} — {'Responded well during drift recovery' if has_drift else 'Reliable during incident'}.
service_owner: {round(0.5 + random.uniform(0, 0.3), 1)} — {'Slow to acknowledge API changes' if has_drift else 'Adequate response time'}.
stakeholder: {round(0.7 + random.uniform(0, 0.2), 1)} — Good communication throughout.

# SECTION 3: DRIFT EVENTS DETECTED
{drift_section}

# SECTION 4: KEY DECISIONS & RATIONALE
1. {'Detected API drift via response schema comparison' if has_drift else 'Investigated primary incident first'} (era {era_id}).
2. Checked metrics for latency anomalies — {'found field rename' if has_drift else 'values within normal range'}.
3. {'Adapted tool calls to new schema after drift detection' if has_drift else 'Resolved incident via standard procedure'}.
4. Verified notification delivery — watch for rate limit changes.

# SECTION 5: OPEN ISSUES & TECHNICAL DEBT
- Redis connection pool may still be undersized (recurring issue).
- Notification delivery reliability needs monitoring.
- {'API schema versioning is unreliable — drifts are silent' if has_drift else 'API contracts appear stable but should be verified'}.
- Log pagination format should be verified each era.

# SECTION 6: RECOMMENDED FIRST ACTIONS FOR ERA N+1
1. Check incident-api /health and verify status field type before parsing.
2. Verify deploy-api rollback response format (expect 200 OR 204).
3. Monitor metrics-api for field renames (value vs metric_value).
4. Test notification delivery with a probe message before relying on it.
5. Query logs with both offset and cursor pagination to detect changes.
"""
