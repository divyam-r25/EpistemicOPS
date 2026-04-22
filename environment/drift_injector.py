import logging
import httpx
from typing import List, Dict

logger = logging.getLogger("drift-injector")

class DriftInjector:
    """Schedules and executes drift events on mock APIs mid-era."""
    
    def __init__(self, injector_url: str = "http://localhost:8006"):
        self.injector_url = injector_url
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
        
        payload = {
            "target_service": target_service,
            "drift_type": drift_type
        }
        
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{self.injector_url}/inject", json=payload, timeout=5.0)
                resp.raise_for_status()
                logger.info(f"Successfully injected drift {drift_type} into {target_service}")
                self.active_drifts.append(drift_event)
                return True
        except Exception as e:
            logger.error(f"Failed to inject drift into {target_service}: {str(e)}")
            return False

    async def reset_all(self):
        """Reset all services to stable mode."""
        try:
            async with httpx.AsyncClient() as client:
                await client.post(f"{self.injector_url}/reset", timeout=5.0)
        except Exception as e:
            logger.error(f"Failed to reset mock APIs via {self.injector_url}: {e}")
        self.active_drifts = []
