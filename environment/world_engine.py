from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime, timezone

class Phase(Enum):
    AWAKENING = "AWAKENING"
    OPERATION = "OPERATION"
    DRIFT_INJECTION = "DRIFT_INJECTION"
    SOCRATIC_RECOVERY = "SOCRATIC_RECOVERY"
    LEGACY_GENERATION = "LEGACY_GENERATION"

class WorldState:
    """Represents the complete, serializable state of the world."""
    def __init__(self, era_id: int):
        self.era_id = era_id
        self.step = 0
        self.phase = Phase.AWAKENING
        self.services = {
            "incident-api": {"status": "STABLE"},
            "metrics-api": {"status": "STABLE"},
            "deploy-api": {"status": "STABLE"},
            "log-api": {"status": "STABLE"},
            "notify-api": {"status": "STABLE"}
        }
        self.incident_history = []
        self.technical_debt = []
        self.team_trust_scores = {
            "oncall_engineer": 1.0,
            "service_owner": 1.0,
            "stakeholder": 1.0
        }
        self.deployment_history = []
        self.current_task_brief = ""
        self.legacy_document_store = {}
        self.drift_events_fired = []
        # Criteria tracking — set by the environment during action execution
        self.incidents_resolved = []
        self.deployments_completed = []
        self.notifications_sent = []
        self.hypotheses_declared = []
        self.tool_calls_made = []
        self.legacy_doc_written = False
        self.task_declared_complete = False

    def to_dict(self) -> dict:
        return {
            "era_id": self.era_id,
            "step": self.step,
            "phase": self.phase.value,
            "services": self.services,
            "incident_history": self.incident_history,
            "technical_debt": self.technical_debt,
            "team_trust_scores": self.team_trust_scores,
            "deployment_history": self.deployment_history,
            "current_task_brief": self.current_task_brief,
            "legacy_document_store": self.legacy_document_store,
            "drift_events_fired": [
                d if isinstance(d, dict) else d.model_dump() if hasattr(d, 'model_dump') else str(d)
                for d in self.drift_events_fired
            ],
            "incidents_resolved": self.incidents_resolved,
            "legacy_doc_written": self.legacy_doc_written,
            "task_declared_complete": self.task_declared_complete,
            "hypotheses_declared": self.hypotheses_declared,
            "reward_state": getattr(self, "reward_state", {})
        }

class WorldEngine:
    """Manages state transitions and persistence across the environment."""
    
    def __init__(self):
        self.state: Optional[WorldState] = None
        self.scenario_config: dict = {}
        self.era_config: dict = {}

    def initialize_era(self, scenario_config: dict, era_id: int, previous_legacy_doc: str = None):
        """Set up the world for a new era."""
        self.scenario_config = scenario_config
        
        # Find era config
        self.era_config = next(
            (e for e in scenario_config.get("eras", []) if e.get("era_id") == era_id),
            {}
        )
        
        # Initialize state or carry over persistent components
        if not self.state:
            self.state = WorldState(era_id)
        else:
            # Reset per-era state but keep persistent world state
            self.state.era_id = era_id
            self.state.step = 0
            self.state.phase = Phase.AWAKENING
            self.state.drift_events_fired = []
            self.state.incidents_resolved = []
            self.state.deployments_completed = []
            self.state.notifications_sent = []
            self.state.hypotheses_declared = []
            self.state.tool_calls_made = []
            self.state.legacy_doc_written = False
            self.state.task_declared_complete = False
            # Reset services to stable for new era
            for svc in self.state.services:
                self.state.services[svc] = {"status": "STABLE"}
            
        self.state.current_task_brief = self.era_config.get("task_brief", "")
        
        if previous_legacy_doc:
            self.state.legacy_document_store[f"era_{era_id-1}"] = previous_legacy_doc

    def advance_step(self):
        self.state.step += 1

    def transition_phase(self, new_phase: Phase):
        self.state.phase = new_phase
        
    def record_drift(self, drift_event: dict):
        service = drift_event.get("target_service")
        if service in self.state.services:
            self.state.services[service]["status"] = "DRIFTED"
            self.state.services[service]["drift_fired_at_step"] = self.state.step
            self.state.services[service]["drift_description"] = drift_event.get("drift_reason", "")
        self.state.drift_events_fired.append(drift_event)

    def update_trust(self, entity: str, new_score: float):
        if entity in self.state.team_trust_scores:
            self.state.team_trust_scores[entity] = max(0.0, min(1.0, new_score))

    def validate_hypotheses(self):
        """Retroactively validate hypotheses against actual drift events and state.
        
        This fixes the calibration reward bug where was_true was never set,
        causing calibration to always be 0.52.
        """
        for hypothesis in self.state.hypotheses_declared:
            h_text = hypothesis.get("hypothesis", "").lower()
            was_true = False
            
            # Check if hypothesis about drift is correct
            if any(keyword in h_text for keyword in ["drift", "change", "schema", "contract", "renamed", "different"]):
                if len(self.state.drift_events_fired) > 0:
                    # Check if the hypothesis mentions the right service
                    for drift in self.state.drift_events_fired:
                        drift_dict = drift if isinstance(drift, dict) else drift.model_dump() if hasattr(drift, 'model_dump') else {}
                        service = drift_dict.get("target_service", "").lower()
                        field = (drift_dict.get("drifted_behaviour", {}) or {}).get("field", "").lower()
                        if service and service.replace("-", "") in h_text.replace("-", ""):
                            was_true = True
                            break
                        if field and field in h_text:
                            was_true = True
                            break
                    # Generic drift hypothesis: partially true if any drift occurred
                    if not was_true and len(self.state.drift_events_fired) > 0:
                        was_true = True  # At least drift did happen
            
            # Check if hypothesis about incident is correct
            if "incident" in h_text or "connection" in h_text or "pool" in h_text:
                if len(self.state.incidents_resolved) > 0:
                    was_true = True
                    
            # Check if hypothesis about deployment is correct
            if "deploy" in h_text or "rollback" in h_text:
                if len(self.state.deployments_completed) > 0:
                    was_true = True
            
            hypothesis["was_true"] = was_true

    def evaluate_success_criteria(self, criteria_list: List[str]) -> List[str]:
        """Check which success criteria have been met based on actual state.
        
        Uses real tracked state rather than just checking task_declared_complete
        for most criteria. Some criteria that are hard to verify mechanically
        still use task_declared_complete as a proxy.
        """
        met = []
        for criterion in criteria_list:
            if criterion == "incident_resolved" and len(self.state.incidents_resolved) > 0:
                met.append(criterion)
            elif criterion == "legacy_doc_written" and self.state.legacy_doc_written:
                met.append(criterion)
            elif criterion == "deploy_successful" and any(
                d.get("type") != "rollback" for d in self.state.deployments_completed
            ):
                met.append(criterion)
            elif criterion == "slo_breach_avoided":
                # Considered met if task was completed before max steps
                if self.state.task_declared_complete:
                    met.append(criterion)
            elif criterion == "rollback_successful" and any(
                d.get("type") == "rollback" for d in self.state.deployments_completed
            ):
                met.append(criterion)
            elif criterion == "rollback_complete" and any(
                d.get("type") == "rollback" for d in self.state.deployments_completed
            ):
                met.append(criterion)
            elif criterion == "notifications_delivered" and len(self.state.notifications_sent) > 0:
                met.append(criterion)
            elif criterion == "incident_identified":
                # Met if agent made any tool call to get_incident_status
                if any(tc.get("tool") == "get_incident_status" for tc in self.state.tool_calls_made):
                    met.append(criterion)
            elif criterion == "root_cause_documented":
                # Met if legacy doc is written (it should contain root cause)
                if self.state.legacy_doc_written:
                    met.append(criterion)
            elif criterion in ("automation_deployed", "pipeline_hardened",
                               "gate_implemented", "synthetics_deployed"):
                # These require both task completion AND relevant tool calls
                if self.state.task_declared_complete and len(self.state.tool_calls_made) >= 3:
                    met.append(criterion)
            elif criterion == "proxies_scaled":
                # Met if agent investigated metrics and declared task complete
                has_metrics_call = any(
                    tc.get("tool") == "get_metrics" for tc in self.state.tool_calls_made
                )
                if has_metrics_call and self.state.task_declared_complete:
                    met.append(criterion)
            elif criterion == "trust_drop_explained":
                # Met if agent declared hypothesis about trust
                if any("trust" in h.get("hypothesis", "").lower() for h in self.state.hypotheses_declared):
                    met.append(criterion)
            elif criterion == "audit_completed":
                if self.state.task_declared_complete and self.state.legacy_doc_written:
                    met.append(criterion)
        return met
