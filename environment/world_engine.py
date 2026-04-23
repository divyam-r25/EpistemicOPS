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
            "drift_events_fired": self.drift_events_fired,
            "incidents_resolved": self.incidents_resolved,
            "legacy_doc_written": self.legacy_doc_written,
            "task_declared_complete": self.task_declared_complete,
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
        self.era_config = next((e for e in scenario_config.get("eras", []) if e.get("era_id") == era_id), {})
        
        # Initialize state or carry over persistent components
        if not self.state:
            self.state = WorldState(era_id)
        else:
            self.state.era_id = era_id
            self.state.step = 0
            self.state.phase = Phase.AWAKENING
            self.state.drift_events_fired = []
            
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

    def evaluate_success_criteria(self, criteria_list: List[str]) -> List[str]:
        """Check which success criteria have been met based on actual state."""
        met = []
        for criterion in criteria_list:
            if criterion == "incident_resolved" and len(self.state.incidents_resolved) > 0:
                met.append(criterion)
            elif criterion == "legacy_doc_written" and self.state.legacy_doc_written:
                met.append(criterion)
            elif criterion == "deploy_successful" and len(self.state.deployments_completed) > 0:
                met.append(criterion)
            elif criterion == "slo_breach_avoided":
                # Considered met if task was completed before max_steps
                if self.state.task_declared_complete:
                    met.append(criterion)
            elif criterion == "rollback_successful" and any(
                d.get("type") == "rollback" for d in self.state.deployments_completed
            ):
                met.append(criterion)
            elif criterion == "notifications_delivered" and len(self.state.notifications_sent) > 0:
                met.append(criterion)
            elif criterion in ("automation_deployed", "pipeline_hardened",
                               "root_cause_documented", "gate_implemented",
                               "incident_identified", "proxies_scaled",
                               "trust_drop_explained", "audit_completed",
                               "synthetics_deployed"):
                # These are met if the agent declared the task complete
                if self.state.task_declared_complete:
                    met.append(criterion)
        return met
