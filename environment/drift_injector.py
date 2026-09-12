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

    def get_drift_for_step(self, step: int, era_config: dict, seed: int = None) -> List[dict]:
        """Check if any drift events are scheduled to fire at the current step.

        Drift fires at a RANDOM step within the configured drift window (seeded).
        This forces the agent to learn from evidence, not from memorized step numbers.

        Args:
            step:       Current world step.
            era_config: Era configuration dict (contains drift_window and drift_events).
            seed:       Optional seed for reproducible per-episode randomness.
                        If None, uses a hash of era_config for stable-but-varying timing.
        """
        drifts_to_fire = []
        drift_window = era_config.get("drift_window", {})
        drift_events = era_config.get("drift_events", [])

        if not drift_events:
            return []

        earliest = drift_window.get("earliest_step", 10)
        latest = drift_window.get("latest_step", 20)

        # Compute a per-era deterministic target step using seeded RNG.
        # This keeps drift timing reproducible for a given episode seed while
        # varying it across episodes/scenarios — agent cannot memorise step number.
        if seed is None:
            # Fallback: stable hash from era config
            import hashlib, json as _json
            cfg_str = _json.dumps(era_config, sort_keys=True, default=str)
            seed = int(hashlib.md5(cfg_str.encode()).hexdigest()[:8], 16)

        rng = __import__("random").Random(seed)
        target_step = rng.randint(earliest, latest)

        if step == target_step:
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
