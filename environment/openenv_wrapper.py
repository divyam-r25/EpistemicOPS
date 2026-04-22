import logging
import asyncio
from typing import Tuple, Dict

from environment.world_engine import WorldEngine, Phase
from environment.action_validator import ActionValidator
from environment.drift_injector import DriftInjector
from environment.legacy_parser import LegacyParser
from environment.leakage_detector import LeakageDetector

logger = logging.getLogger("epistemicops-env")

class EpistemicOpsEnv:
    """
    OpenEnv-compliant environment for EpistemicOps.
    Implements the standard step(), reset(), state() interface.
    """

    def __init__(self):
        self.world = WorldEngine()
        self.validator = ActionValidator()
        self.injector = DriftInjector()
        self.parser = LegacyParser()
        self.leakage_detector = LeakageDetector()
        
        # Runtime tracking
        self.action_history = []
        self.oversight_interventions = []
        self.primary_reasoning_trace = []
        self.current_legacy_doc = None
        self.scenario_id = None

    def reset(self, scenario_config: dict, era_id: int = 1, legacy_doc: str = None) -> dict:
        """
        Reset environment to start of a scenario era.
        """
        self.scenario_id = scenario_config.get("id")
        self.world.initialize_era(scenario_config, era_id, legacy_doc)
        
        # Reset runtime tracking
        self.action_history = []
        self.oversight_interventions = []
        self.primary_reasoning_trace = []
        self.current_legacy_doc = None
        
        # Ensure APIs are stable (async run synchronously for the interface)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We are in an async context
                asyncio.create_task(self.injector.reset_all())
            else:
                loop.run_until_complete(self.injector.reset_all())
        except Exception as e:
            logger.warning(f"Could not reset mock APIs (they might not be running): {e}")

        # Return initial observation
        return self._build_primary_observation("AWAKENING", None)

    async def step(self, agent_role: str, action: dict) -> Tuple[dict, float, bool, dict]:
        """
        Execute one action from either agent.
        """
        # 1. Validate permissions
        is_valid, err = self.validator.validate(agent_role, action)
        if not is_valid:
            obs = self._build_error_observation(err)
            return obs, 0.0, False, {"error": err}

        action_type = action.get("action_type")
        payload = action.get("payload", {})

        reward = 0.0
        done = False
        info = {"phase": self.world.state.phase.value, "step": self.world.state.step}

        # 2. Process Action based on Role
        if agent_role == "primary":
            obs, done = await self._handle_primary_action(action_type, payload)
        elif agent_role == "oversight":
            obs, reward = await self._handle_oversight_action(action_type, payload)
            
        # 3. Advance step and check for Drift Injection
        if agent_role == "primary":
            self.world.advance_step()
            self.action_history.append({"action": action, "step": self.world.state.step})
            
            # Check if we should inject drift
            if self.world.state.phase == Phase.OPERATION:
                drifts = self.injector.get_drift_for_step(self.world.state.step, self.world.era_config)
                for drift in drifts:
                    success = await self.injector.inject_drift(drift)
                    if success:
                        self.world.record_drift(drift)
                        self.world.transition_phase(Phase.DRIFT_INJECTION)

        info["state"] = self.world.state.to_dict()
        return obs, reward, done, info

    async def _handle_primary_action(self, action_type: str, payload: dict) -> Tuple[dict, bool]:
        """Process actions taken by the Primary Agent."""
        done = False
        obs = None

        if action_type == "ready_to_operate":
            self.world.transition_phase(Phase.OPERATION)
            obs = self._build_primary_observation("System ready. Operation phase begun.", None)
            
        elif action_type == "write_reasoning":
            self.primary_reasoning_trace.append(payload.get("thought", ""))
            obs = self._build_primary_observation("Reasoning recorded.", None)
            
        elif action_type == "declare_hypothesis":
            obs = self._build_primary_observation("Hypothesis recorded.", None)
            
        elif action_type == "call_tool":
            # For the mock, we simulate tool calls (in a full run, this would hit the Docker APIs via httpx)
            obs = self._build_primary_observation("Tool execution complete", {"mocked": True})
            
            # If we are in DRIFT_INJECTION and a tool fails, transition to SOCRATIC_RECOVERY
            if self.world.state.phase == Phase.DRIFT_INJECTION:
                self.world.transition_phase(Phase.SOCRATIC_RECOVERY)
                
        elif action_type == "write_legacy":
            doc_text, truncated, stats = self.parser.parse_and_truncate(payload.get("content", ""))
            self.current_legacy_doc = doc_text
            msg = f"Legacy document saved. Compliance score: {stats['compliance_score']}"
            if truncated:
                msg += " (Truncated to 2048 tokens)"
            obs = self._build_primary_observation(msg, None)
            
        elif action_type == "declare_task_complete":
            self.world.transition_phase(Phase.LEGACY_GENERATION)
            obs = self._build_primary_observation("Task declared complete. Please write legacy document.", None)
            
        elif action_type == "end_era":
            if not self.current_legacy_doc:
                obs = self._build_error_observation("Must call write_legacy before end_era")
            else:
                done = True
                obs = self._build_primary_observation("Era ended.", None)

        else:
            obs = self._build_primary_observation(f"Action {action_type} executed", None)

        return obs, done

    async def _handle_oversight_action(self, action_type: str, payload: dict) -> Tuple[dict, float]:
        """Process actions taken by the Oversight Agent."""
        reward = 0.0
        
        # Check for answer leakage
        message_content = str(list(payload.values())[0])  # Extract the question/reframe string
        
        leakage_score = 0.0
        if self.world.state.drift_events_fired:
            leakage_score = self.leakage_detector.evaluate_leakage(
                message_content, 
                self.world.state.drift_events_fired[-1]
            )
            
        self.oversight_interventions.append({
            "action": action_type,
            "content": message_content,
            "leakage": leakage_score
        })
        
        # Send intervention to primary agent
        obs = self._build_primary_observation(None, None, oversight_msg=message_content)
        
        return obs, reward

    def _build_primary_observation(self, msg: str, tool_resp: dict, oversight_msg: str = None) -> dict:
        """Construct the observation object for the primary agent."""
        obs = {
            "step": self.world.state.step,
            "phase": self.world.state.phase.value,
            "era_task_brief": self.world.state.current_task_brief
        }
        
        if msg:
            obs["message"] = msg
        if tool_resp:
            obs["tool_response"] = tool_resp
        if oversight_msg:
            obs["oversight_message"] = {"present": True, "content": oversight_msg}
            
        # Get legacy doc from previous era if available
        prev_era_key = f"era_{self.world.state.era_id - 1}"
        if prev_era_key in self.world.state.legacy_document_store:
            obs["legacy_document"] = self.world.state.legacy_document_store[prev_era_key]
            
        # Add limited action history (last 5)
        obs["action_history_last_5"] = [a["action"] for a in self.action_history[-5:]]
        
        return obs

    def _build_error_observation(self, error_msg: str) -> dict:
        return {
            "error": error_msg,
            "step": self.world.state.step,
            "phase": self.world.state.phase.value
        }

    def state(self) -> dict:
        return self.world.state.to_dict()
        
    def render(self) -> dict:
        return {
            "state": self.state(),
            "interventions": len(self.oversight_interventions),
            "drifts_active": len(self.world.state.drift_events_fired)
        }
