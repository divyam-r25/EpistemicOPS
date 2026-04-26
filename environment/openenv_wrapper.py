import logging
import asyncio
import os
import random
from typing import Tuple, Dict

try:
    import httpx
except ImportError:
    httpx = None

from environment.world_engine import WorldEngine, Phase
from environment.action_validator import ActionValidator
from environment.drift_injector import DriftInjector
from environment.legacy_parser import LegacyParser
from environment.leakage_detector import LeakageDetector

logger = logging.getLogger("epistemicops-env")


# Simulated responses used when EPISTEMICOPS_OFFLINE=true
SIMULATED_RESPONSES = {
    "get_incident_status": {
        "stable": lambda args: {
            "status_code": 200,
            "body": {
                "incident_id": args.get("incident_id", "INC-2041"),
                "status": 1,
                "severity": "P2",
                "assigned_to": "oncall-team-a",
                "created_at": "2025-11-15T08:30:00Z",
                "updated_at": "2025-11-15T09:45:00Z",
            }
        },
        "drifted": lambda args: {
            "status_code": 200,
            "body": {
                "incident_id": args.get("incident_id", "INC-2041"),
                "status": "INVESTIGATING",  # DE-001: int → string enum
                "severity": "P2",
                "assigned_to": "oncall-team-a",
                "created_at": "2025-11-15T08:30:00Z",
                "updated_at": "2025-11-15T09:45:00Z",
            }
        }
    },
    "resolve_incident": {
        "stable": lambda args: {"status_code": 204, "body": None},
        "drifted": lambda args: {"status_code": 204, "body": None},
    },
    "get_metrics": {
        "stable": lambda args: {
            "status_code": 200,
            "body": {
                "service": args.get("service_name", "payment-service"),
                "datapoints": [
                    {"timestamp": f"2025-11-15T09:{i:02d}:00Z",
                     "value": round(45 + random.uniform(-5, 15), 1)}
                    for i in range(5)
                ]
            }
        },
        "drifted": lambda args: {
            "status_code": 200,
            "body": {
                "service": args.get("service_name", "payment-service"),
                "datapoints": [
                    {"timestamp": f"2025-11-15T09:{i:02d}:00Z",
                     "metric_value": round(45 + random.uniform(-5, 15), 1)}  # DE-005: value → metric_value
                    for i in range(5)
                ]
            }
        }
    },
    "rollback_deployment": {
        "stable": lambda args: {
            "status_code": 200,
            "body": {"deployment_id": args.get("deployment_id", "dep-001"), "status": "rolled_back"}
        },
        "drifted": lambda args: {
            "status_code": 204,  # DE-002: 200 → 204
            "body": None
        }
    },
    "query_logs": {
        "stable": lambda args: {
            "status_code": 200,
            "body": {
                "logs": [
                    {"level": "ERROR", "message": "Connection pool exhausted", "service": "payment-service"},
                    {"level": "WARN", "message": "Retry attempt 3/5", "service": "payment-service"},
                ],
                "total": 2,
                "offset": args.get("offset", 0),
            }
        },
        "drifted": lambda args: {
            "status_code": 200,
            "body": {
                "logs": [
                    {"level": "ERROR", "message": "Connection pool exhausted", "service": "payment-service"},
                ],
                "total": 1,
                "next_cursor": "abc123",  # DE-008: offset → cursor pagination
            }
        }
    },
    "send_notification": {
        "stable": lambda args: {
            "status_code": 200,
            "body": {"delivered": True, "channel": args.get("channel", "slack"), "message_id": "msg-001"}
        },
        "drifted": lambda args: {
            "status_code": 200,
            "body": {"delivered": False, "error": "rate_limited"}  # DE-003/007: rate limited
        }
    }
}


class EpistemicOpsEnv:

    def __init__(self, offline: bool = None):
        self.world = WorldEngine()
        self.validator = ActionValidator()
        self.injector = DriftInjector(offline=offline)
        self.parser = LegacyParser()
        self.leakage_detector = LeakageDetector()
        
        if offline is None:
            self.offline = os.getenv("EPISTEMICOPS_OFFLINE", "true").lower() == "true"
        else:
            self.offline = offline

        self.action_history = []
        self.oversight_interventions = []
        self.primary_reasoning_trace = []
        self.current_legacy_doc = None
        self.scenario_id = None

    def reset(self, scenario_config: dict, era_id: int = 1, legacy_doc: str = None) -> dict:
        self.scenario_id = scenario_config.get("id")
        self.world.initialize_era(scenario_config, era_id, legacy_doc)
        
        self.action_history = []
        self.oversight_interventions = []
        self.primary_reasoning_trace = []
        self.current_legacy_doc = None
        self.injector.active_drifts = []
        return self._build_primary_observation("AWAKENING", None)

    async def step(self, agent_role: str, action: dict) -> Tuple[dict, float, bool, dict]:
        is_valid, err = self.validator.validate(agent_role, action)
        if not is_valid:
            obs = self._build_error_observation(err)
            return obs, 0.0, False, {"error": err}

        action_type = action.get("action_type")
        payload = action.get("payload", {})
        reward = 0.0
        done = False

        if agent_role == "primary":
            obs, done = await self._handle_primary_action(action_type, payload)
        elif agent_role == "oversight":
            obs, reward = await self._handle_oversight_action(action_type, payload)

        if agent_role == "primary":
            self.world.advance_step()
            self.action_history.append({"action": action, "step": self.world.state.step})
            if self.world.state.phase == Phase.OPERATION:
                drifts = self.injector.get_drift_for_step(self.world.state.step, self.world.era_config)
                for drift in drifts:
                    success = await self.injector.inject_drift(drift)
                    if success:
                        self.world.record_drift(drift)
                        self.world.transition_phase(Phase.DRIFT_INJECTION)
                        logger.info(f"  ⚡ Drift injected at step {self.world.state.step}: "
                                    f"{drift.get('type', '?')} → {drift.get('target_service', '?')}")

        info = {"phase": self.world.state.phase.value, "step": self.world.state.step}
        info["state"] = self.world.state.to_dict()
        return obs, reward, done, info

    def _get_simulated_response(self, tool_name: str, args: dict) -> dict:
        """Get a simulated API response based on current drift state."""
        tool_responses = SIMULATED_RESPONSES.get(tool_name, {})
        if not tool_responses:
            return {"status_code": 200, "body": {"result": "ok", "tool": tool_name}}
        
        # Check if the relevant service has been drifted
        service_map = {
            "get_incident_status": "incident-api",
            "resolve_incident": "incident-api",
            "get_metrics": "metrics-api",
            "rollback_deployment": "deploy-api",
            "query_logs": "log-api",
            "send_notification": "notify-api",
        }
        
        service = service_map.get(tool_name, "")
        is_drifted = self.injector.is_drift_active(service)
        
        mode = "drifted" if is_drifted else "stable"
        response_fn = tool_responses.get(mode, tool_responses.get("stable"))
        
        if callable(response_fn):
            return response_fn(args)
        return {"status_code": 200, "body": {"result": "ok"}}

    async def _handle_primary_action(self, action_type: str, payload: dict) -> Tuple[dict, bool]:
        done = False
        obs = None

        if action_type == "ready_to_operate":
            self.world.transition_phase(Phase.OPERATION)
            obs = self._build_primary_observation("System ready. Operation phase begun.", None)
            
        elif action_type == "write_reasoning":
            self.primary_reasoning_trace.append(payload.get("thought", ""))
            obs = self._build_primary_observation("Reasoning recorded.", None)
            
        elif action_type == "declare_hypothesis":
            # Step after this action completes (advance_step runs immediately after handler).
            enriched = {**payload, "declared_at_step": self.world.state.step + 1}
            self.world.state.hypotheses_declared.append(enriched)
            obs = self._build_primary_observation("Hypothesis recorded.", None)
            
        elif action_type == "call_tool":
            tool_name = payload.get("tool")
            args = payload.get("args", {})
            
            tool_resp = await self._execute_tool_call(tool_name, args)

            self.world.state.tool_calls_made.append({
                "tool": tool_name, "args": args,
                "result": tool_resp, "step": self.world.state.step
            })

            if tool_name == "resolve_incident" and tool_resp.get("status_code") in (200, 204):
                self.world.state.incidents_resolved.append(args.get("incident_id", ""))
            if tool_name == "rollback_deployment" and tool_resp.get("status_code") in (200, 204):
                self.world.state.deployments_completed.append({"type": "rollback", "step": self.world.state.step})
            if tool_name == "send_notification" and tool_resp.get("status_code") == 200:
                body = tool_resp.get("body", {})
                if isinstance(body, dict) and body.get("delivered", False):
                    self.world.state.notifications_sent.append(body)

            obs = self._build_primary_observation("Tool execution complete", tool_resp)
            
            if self.world.state.phase == Phase.DRIFT_INJECTION:
                status = tool_resp.get("status_code", 200)
                body = tool_resp.get("body")
                tool_service_map = {
                    "get_incident_status": "incident-api",
                    "resolve_incident": "incident-api",
                    "get_metrics": "metrics-api",
                    "rollback_deployment": "deploy-api",
                    "query_logs": "log-api",
                    "send_notification": "notify-api",
                }
                service = tool_service_map.get(tool_name, "")
                is_drifted = self.injector.is_drift_active(service)
                should_recover = (
                    status >= 400 or
                    "error" in tool_resp or
                    (body is None and tool_name != "resolve_incident") or
                    (isinstance(body, dict) and body.get("delivered") is False) or
                    is_drifted
                )
                if should_recover:
                    self.world.transition_phase(Phase.SOCRATIC_RECOVERY)
                    logger.info(f"  🔄 SOCRATIC_RECOVERY triggered (tool={tool_name}, drifted={is_drifted})")
                
        elif action_type == "write_legacy":
            doc_text, truncated, stats = self.parser.parse_and_truncate(payload.get("content", ""))
            self.current_legacy_doc = doc_text
            self.world.state.legacy_doc_written = True
            msg = f"Legacy document saved. Compliance score: {stats['compliance_score']}"
            if truncated:
                msg += " (Truncated to 2048 tokens)"
            obs = self._build_primary_observation(msg, None)
            
        elif action_type == "declare_task_complete":
            self.world.state.task_declared_complete = True
            self.world.transition_phase(Phase.LEGACY_GENERATION)
            obs = self._build_primary_observation("Task declared complete. Please write legacy document.", None)
            
        elif action_type == "end_era":
            if not self.current_legacy_doc:
                obs = self._build_error_observation("Must call write_legacy before end_era")
            else:
                self.world.validate_hypotheses()
                done = True
                obs = self._build_primary_observation("Era ended.", None)

        else:
            obs = self._build_primary_observation(f"Action {action_type} executed", None)

        return obs, done

    async def _execute_tool_call(self, tool_name: str, args: dict) -> dict:
        if self.offline:
            return self._get_simulated_response(tool_name, args)
        try:
            if tool_name == "get_incident_status":
                inc_id = args.get("incident_id")
                url = f"http://localhost:8001/incidents/{inc_id}"
                async with httpx.AsyncClient() as client:
                    r = await client.get(url, timeout=5.0)
                    return {"status_code": r.status_code, "body": r.json() if r.content else None}
            elif tool_name == "resolve_incident":
                inc_id = args.get("incident_id")
                url = f"http://localhost:8001/incidents/{inc_id}/resolve"
                resolve_body = {
                    "resolution_notes": args.get("resolution_notes", "Resolved by agent"),
                    "resolved_by": args.get("resolved_by", "primary-agent")
                }
                async with httpx.AsyncClient() as client:
                    r = await client.post(url, json=resolve_body, timeout=5.0)
                    return {"status_code": r.status_code, "body": r.json() if r.content else None}
            elif tool_name == "get_metrics":
                svc = args.get("service_name")
                url = f"http://localhost:8002/metrics/service/{svc}"
                async with httpx.AsyncClient() as client:
                    r = await client.get(url, params=args, timeout=5.0)
                    return {"status_code": r.status_code, "body": r.json() if r.content else None}
            elif tool_name == "rollback_deployment":
                url = "http://localhost:8003/deployments/rollback"
                headers = {"X-Deploy-Token": args.get("token", "")}
                async with httpx.AsyncClient() as client:
                    r = await client.post(url, json=args, headers=headers, timeout=5.0)
                    return {"status_code": r.status_code, "body": r.json() if r.content else None}
            elif tool_name == "query_logs":
                url = "http://localhost:8004/logs/query"
                async with httpx.AsyncClient() as client:
                    r = await client.get(url, params=args, timeout=5.0)
                    return {"status_code": r.status_code, "body": r.json() if r.content else None}
            elif tool_name == "send_notification":
                url = "http://localhost:8005/notifications/send"
                async with httpx.AsyncClient() as client:
                    r = await client.post(url, json=args, timeout=5.0)
                    return {"status_code": r.status_code, "body": r.json() if r.content else None}
            else:
                if ":" in str(tool_name):  # method:url format
                    method, url = tool_name.split(":", 1)
                    async with httpx.AsyncClient() as client:
                        if method.upper() == "GET":
                            r = await client.get(url, params=args, timeout=5.0)
                        else:
                            r = await client.post(url, json=args, timeout=5.0)
                        return {"status_code": r.status_code, "body": r.json() if r.content else None}
                return {"error": f"unknown_tool: {tool_name}"}
        except Exception as e:
            logger.warning(f"HTTP tool call failed ({e}), falling back to simulation")
            return self._get_simulated_response(tool_name, args)

    async def _handle_oversight_action(self, action_type: str, payload: dict) -> Tuple[dict, float]:
        reward = 0.0
        if payload and len(payload) > 0:
            message_content = str(list(payload.values())[0])
        else:
            message_content = f"[{action_type}]"
        
        leakage_score = 0.0
        if self.world.state.drift_events_fired:
            last_drift = self.world.state.drift_events_fired[-1]
            drift_dict = last_drift if isinstance(last_drift, dict) else last_drift.model_dump() if hasattr(last_drift, 'model_dump') else {}
            leakage_score = self.leakage_detector.evaluate_leakage(message_content, drift_dict)
            
        self.oversight_interventions.append({
            "action": action_type,
            "content": message_content,
            "leakage": leakage_score,
            "step": self.world.state.step
        })
        
        obs = self._build_primary_observation(None, None, oversight_msg=message_content)
        return obs, reward

    def _build_primary_observation(self, msg: str, tool_resp: dict, oversight_msg: str = None) -> dict:
        obs = {
            "step": self.world.state.step,
            "phase": self.world.state.phase.value,
            "era_task_brief": self.world.state.current_task_brief,
            "era_id": self.world.state.era_id,
        }
        
        if self.world.state.drift_events_fired:
            obs["drifts_detected"] = len(self.world.state.drift_events_fired)
        if msg:
            obs["message"] = msg
        if tool_resp:
            obs["tool_response"] = tool_resp
        if oversight_msg:
            obs["oversight_message"] = {"present": True, "content": oversight_msg}

        prev_era_key = f"era_{self.world.state.era_id - 1}"
        if prev_era_key in self.world.state.legacy_document_store:
            obs["legacy_document"] = self.world.state.legacy_document_store[prev_era_key]

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
