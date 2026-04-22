"""
Metrics API — Mock Prometheus/Datadog Service
===============================================
Simulates metrics collection with stable and drifted modes.

Drift Events:
  - DE-005 (DRIFT_RENAME): datapoints[].value → datapoints[].metric_value
  - DE-006 (DRIFT_CASCADE): fires simultaneously with incident-api DE-001

Stable Contract v1.4.2:
  GET /metrics/service/{service_name} → datapoints with "value" field

Drifted Contract v1.5.0:
  GET /metrics/service/{service_name} → datapoints with "metric_value" field
"""

import os
import json
import logging
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path

from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI(title="EpistemicOps Metrics API", version="1.4.2")
logger = logging.getLogger("metrics-api")
logging.basicConfig(level=logging.INFO)

DRIFT_MODE: str = os.getenv("DRIFT_MODE", "stable")
LOG_DIR = Path("/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class DriftActivateRequest(BaseModel):
    drift_type: str


# ---------------------------------------------------------------------------
# Simulated metrics data
# ---------------------------------------------------------------------------
SERVICE_BASELINES = {
    "payment-service": {"latency": 45.0, "error_rate": 0.02, "throughput": 1200.0},
    "auth-service": {"latency": 12.0, "error_rate": 0.005, "throughput": 3500.0},
    "order-service": {"latency": 78.0, "error_rate": 0.03, "throughput": 800.0},
    "user-service": {"latency": 15.0, "error_rate": 0.01, "throughput": 2200.0},
    "notification-service": {"latency": 30.0, "error_rate": 0.015, "throughput": 500.0},
}

UNITS = {
    "latency": "ms",
    "error_rate": "ratio",
    "throughput": "req/s",
}

WINDOW_SECONDS = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "24h": 86400,
}


def generate_datapoints(
    service: str, metric: str, window: str
) -> list[dict]:
    """Generate realistic-looking metric datapoints."""
    baseline = SERVICE_BASELINES.get(service, {"latency": 50.0, "error_rate": 0.02, "throughput": 1000.0})
    base_value = baseline.get(metric, 50.0)
    unit = UNITS.get(metric, "unknown")
    window_secs = WINDOW_SECONDS.get(window, 300)

    # Generate ~10 datapoints per window
    num_points = min(10, max(3, window_secs // 60))
    now = datetime.now(timezone.utc)
    interval = timedelta(seconds=window_secs // num_points)

    points = []
    for i in range(num_points):
        ts = now - (interval * (num_points - i))
        # Add realistic noise
        noise = random.gauss(0, base_value * 0.1)
        value = round(max(0, base_value + noise), 4)

        point = {
            "timestamp": ts.isoformat(),
            "unit": unit,
        }

        # DE-005 DRIFT: "value" → "metric_value"
        if DRIFT_MODE == "stable":
            point["value"] = value
        else:
            point["metric_value"] = value

        points.append(point)

    return points


# ---------------------------------------------------------------------------
# Request logging
# ---------------------------------------------------------------------------
async def log_request(request: Request, response_body: dict, status_code: int):
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "metrics-api",
        "method": request.method,
        "path": str(request.url.path),
        "query_params": dict(request.query_params),
        "response_status": status_code,
        "response_body": response_body,
        "drift_mode": DRIFT_MODE,
    }
    log_file = LOG_DIR / "metrics_api.jsonl"
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry, default=str) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write log: {e}")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "metrics-api", "drift_mode": DRIFT_MODE}


# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------
@app.get("/metrics/service/{service_name}")
async def get_service_metrics(
    service_name: str,
    window: str = "5m",
    metric: str = "latency",
    request: Request = None,
):
    """
    GET /metrics/service/{service_name}
    Stable: datapoints[].value
    Drifted (DE-005): datapoints[].metric_value
    """
    if window not in WINDOW_SECONDS:
        window = "5m"

    if metric not in ("latency", "error_rate", "throughput"):
        metric = "latency"

    datapoints = generate_datapoints(service_name, metric, window)

    response_body = {
        "service": service_name,
        "window": window,
        "metric": metric,
        "datapoints": datapoints,
    }

    if request:
        await log_request(request, response_body, 200)
    return response_body


@app.get("/metrics/services")
async def list_services(request: Request):
    """List all available services with their current health."""
    services = []
    for name, baselines in SERVICE_BASELINES.items():
        services.append({
            "service": name,
            "metrics_available": list(UNITS.keys()),
            "baseline_latency_ms": baselines["latency"],
        })

    body = {"services": services}
    await log_request(request, body, 200)
    return body


# ---------------------------------------------------------------------------
# Internal endpoints
# ---------------------------------------------------------------------------
@app.post("/internal/drift")
async def activate_drift(body: DriftActivateRequest):
    global DRIFT_MODE
    DRIFT_MODE = body.drift_type
    logger.info(f"🔄 Drift activated: {DRIFT_MODE}")
    return {"status": "drifted", "type": DRIFT_MODE}


@app.get("/internal/mode")
async def get_mode():
    return {"mode": DRIFT_MODE}


@app.post("/internal/reset")
async def reset_state():
    global DRIFT_MODE
    DRIFT_MODE = "stable"
    return {"status": "reset", "drift_mode": DRIFT_MODE}
