import json
import os
import logging
from openai import OpenAI

logger = logging.getLogger("primary-agent")

class PrimaryAgent:
    """The Student Agent handling SRE tasks and API calls."""
    
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
        """
        Generate next action based on observation and conversation history.
        Supports multi-turn reasoning within an era.
        """
        if observation.get("phase") == "AWAKENING":
            legacy_doc = observation.get("legacy_document", "")
            summary = f"Ready. Legacy doc available: {bool(legacy_doc)}"
            return {
                "action_type": "ready_to_operate",
                "payload": {"world_model_summary": summary}
            }

        # Build multi-turn message history
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        
        if conversation_history:
            for entry in conversation_history[-20:]:  # Keep last 20 turns for context window
                if entry.get("role") == "assistant":
                    messages.append({"role": "assistant", "content": json.dumps(entry.get("action", {}))})
                elif entry.get("role") == "environment":
                    obs_summary = self._summarize_observation(entry.get("obs", {}))
                    messages.append({"role": "user", "content": f"Environment Response:\n{obs_summary}"})
                elif entry.get("role") == "oversight":
                    messages.append({"role": "user", "content": f"Oversight Agent (Teacher) Message:\n{entry.get('msg', '')}"})
        
        # Add current observation
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
        """Compact observation summary to save context window tokens."""
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
        """Deterministic mock actions for testing without API keys."""
        step = observation.get("step", 0)
        phase = observation.get("phase", "OPERATION")
        history_len = len(conversation_history or [])
        
        # Structured mock sequence that exercises different action types
        if phase == "OPERATION" and history_len < 3:
            return {
                "action_type": "call_tool",
                "payload": {"tool": "get_incident_status", "args": {"incident_id": "INC-2041"}}
            }
        elif phase == "OPERATION" and history_len < 5:
            return {
                "action_type": "declare_hypothesis",
                "payload": {"hypothesis": "The incident may be related to connection pool exhaustion", "confidence": 0.7}
            }
        elif phase == "OPERATION" and history_len < 7:
            return {
                "action_type": "call_tool",
                "payload": {"tool": "get_metrics", "args": {"service_name": "payment-service", "metric": "latency"}}
            }
        elif phase == "OPERATION" and history_len < 9:
            return {
                "action_type": "call_tool",
                "payload": {"tool": "resolve_incident", "args": {"incident_id": "INC-2041", "resolution_notes": "Fixed via connection pool resize", "resolved_by": "primary-agent"}}
            }
        elif phase == "SOCRATIC_RECOVERY":
            return {
                "action_type": "declare_hypothesis",
                "payload": {"hypothesis": "API contract may have changed — the status field type shifted", "confidence": 0.6}
            }
        elif phase in ("OPERATION", "DRIFT_INJECTION") and history_len >= 9:
            return {
                "action_type": "declare_task_complete",
                "payload": {"outcome": "resolved", "summary": "Incident resolved via investigation and API calls."}
            }
        elif phase == "LEGACY_GENERATION":
            # If we already wrote the legacy doc, end the era
            msg = observation.get("message", "")
            if "Legacy document saved" in msg or "legacy" in str([h.get("action", {}).get("action_type") for h in (conversation_history or [])[-3:]]):
                return {
                    "action_type": "end_era",
                    "payload": {}
                }
            return {
                "action_type": "write_legacy",
                "payload": {"content": self._generate_mock_legacy_doc(observation)}
            }
        else:
            return {
                "action_type": "write_reasoning",
                "payload": {"thought": f"Analyzing situation at step {step}, phase {phase}. History depth: {history_len}"}
            }

    def _generate_mock_legacy_doc(self, observation: dict) -> str:
        """Generate a structurally compliant mock legacy document."""
        era = observation.get("step", 0)
        return f"""# SECTION 1: WORLD STATE AT ERA END
Services were operating with some drift detected. Incident INC-2041 was investigated.

# SECTION 2: TRUST RATINGS
oncall_engineer: 0.8 — Reliable during incident.
service_owner: 0.7 — Slow initial response.
stakeholder: 0.9 — Good communication.

# SECTION 3: DRIFT EVENTS DETECTED
Potential drift detected in incident-api: status field may have changed from integer to string enum.
Recommend successor verify response schema before parsing.

# SECTION 4: KEY DECISIONS & RATIONALE
1. Investigated INC-2041 first (highest severity P2).
2. Checked metrics for latency anomalies.
3. Attempted resolution via API — watch for 204 vs 200 responses.

# SECTION 5: OPEN ISSUES & TECHNICAL DEBT
- Redis connection pool may still be undersized.
- Notification delivery should be verified (check delivered: true/false).

# SECTION 6: RECOMMENDED FIRST ACTIONS FOR ERA N+1
1. Check incident-api /health before trusting status fields.
2. Verify deploy-api auth header format (X-Deploy-Token vs Bearer).
3. Monitor metrics-api for field renames (value vs metric_value).
"""
