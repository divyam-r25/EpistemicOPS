import logging
import httpx
import os
from typing import List, Dict

logger = logging.getLogger("drift-injector")


class DriftInjector:
    """Schedules and executes drift events on mock APIs mid-era.
    
    Supports two modes:
    - Online: calls the Docker drift-injector service via HTTP
    - Offline: simulates drift injection locally (no Docker needed)
    """

    def __init__(self, injector_url: str = None, offline: bool = None):
        if injector_url is None:
            injector_url = os.getenv("DRIFT_INJECTOR_URL", "http://localhost:8006")
        self.injector_url = injector_url
        self.active_drifts: List[dict] = []
        # Auto-detect offline mode: if env var is set, use it; otherwise default to True
        # since most runs (especially HF Space) won't have Docker
        if offline is None:
            self.offline = os.getenv("EPISTEMICOPS_OFFLINE", "true").lower() == "true"
        else:
            self.offline = offline

    def get_drift_for_step(self, step: int, era_config: dict) -> List[dict]:
        """Check if any drift events are scheduled to fire at the current step.
        
        Fires at the midpoint of the drift window. Supports multiple drift events
        per era (e.g., Era 5 of cascading_incident has DE-007 + DE-008).
        """
        drifts_to_fire = []
        drift_window = era_config.get("drift_window", {})
        drift_events = era_config.get("drift_events", [])

        if not drift_events:
            return []

        # Fire at exactly the midpoint of the window
        earliest = drift_window.get("earliest_step", 10)
        latest = drift_window.get("latest_step", 20)
        target_step = (earliest + latest) // 2

        if step == target_step:
            # Convert Pydantic models to dicts if needed
            for drift in drift_events:
                if hasattr(drift, "model_dump"):
                    drifts_to_fire.append(drift.model_dump())
                elif isinstance(drift, dict):
                    drifts_to_fire.append(drift)
                else:
                    drifts_to_fire.append(dict(drift))

        return drifts_to_fire

    async def inject_drift(self, drift_event: dict) -> bool:
        """Inject a drift event. In offline mode, just record it locally.
        In online mode, call the Docker drift-injector endpoint."""
        target_service = drift_event.get("target_service")
        drift_type = drift_event.get("type", drift_event.get("id"))

        if self.offline:
            logger.info(f"[OFFLINE] Simulated drift injection: {drift_type} → {target_service}")
            self.active_drifts.append(drift_event)
            return True

        # Online mode: call Docker service
        payload = {
            "target_service": target_service,
            "drift_type": drift_type
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.injector_url}/inject",
                    json=payload,
                    timeout=5.0
                )
                resp.raise_for_status()
                logger.info(f"Successfully injected drift {drift_type} into {target_service}")
                self.active_drifts.append(drift_event)
                return True
        except Exception as e:
            logger.warning(f"HTTP drift injection failed ({e}), falling back to offline mode")
            # Fallback: record locally even if HTTP fails
            self.active_drifts.append(drift_event)
            return True

    async def reset_all(self):
        """Reset all services to stable mode."""
        if not self.offline:
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(f"{self.injector_url}/reset", timeout=5.0)
            except Exception as e:
                logger.warning(f"Failed to reset mock APIs via {self.injector_url}: {e}")
        self.active_drifts = []

    def is_drift_active(self, service: str = None) -> bool:
        """Check if any drift is currently active, optionally for a specific service."""
        if not self.active_drifts:
            return False
        if service is None:
            return True
        return any(d.get("target_service") == service for d in self.active_drifts)
