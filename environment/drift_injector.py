import logging
import httpx
from typing import List, Dict

logger = logging.getLogger("drift-injector")

class DriftInjector:
    """Schedules and executes drift events on mock APIs mid-era."""
    
    # Internal URLs for the mock API containers
    SERVICE_URLS = {
        "incident-api": "http://incident-api:8000",
        "metrics-api": "http://metrics-api:8000",
        "deploy-api": "http://deploy-api:8000",
        "log-api": "http://log-api:8000",
        "notify-api": "http://notify-api:8000"
    }

    def __init__(self):
        self.active_drifts = []

    def get_drift_for_step(self, step: int, era_config: dict) -> List[dict]:
        """Check if any drift events are scheduled to fire at the current step."""
        drifts_to_fire = []
        drift_window = era_config.get("drift_window", {})
        
        # Simple deterministic firing for now: fire at exactly the midpoint of the window
        target_step = (drift_window.get("earliest_step", 10) + drift_window.get("latest_step", 20)) // 2
        
        if step == target_step:
            drifts_to_fire = era_config.get("drift_events", [])
            
        return drifts_to_fire

    async def inject_drift(self, drift_event: dict) -> bool:
        """Call the internal Docker endpoint to mutate the API contract."""
        target_service = drift_event.get("target_service")
        drift_type = drift_event.get("type", drift_event.get("id"))
        
        if target_service not in self.SERVICE_URLS:
            logger.error(f"Unknown service target: {target_service}")
            return False
            
        url = f"{self.SERVICE_URLS[target_service]}/internal/drift"
        
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json={"drift_type": drift_type}, timeout=5.0)
                resp.raise_for_status()
                logger.info(f"Successfully injected drift {drift_type} into {target_service}")
                self.active_drifts.append(drift_event)
                return True
        except Exception as e:
            logger.error(f"Failed to inject drift into {target_service}: {str(e)}")
            return False

    async def reset_all(self):
        """Reset all services to stable mode."""
        for service, base_url in self.SERVICE_URLS.items():
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(f"{base_url}/internal/reset", timeout=5.0)
            except Exception:
                pass
        self.active_drifts = []
