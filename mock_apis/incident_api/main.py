"""
Incident API — Mock PagerDuty/OpsGenie Service
================================================
Simulates incident management with stable and drifted modes.

Drift Events:
  - DE-001 (DRIFT_TYPE): status field changes from integer to string enum
  - DE-006 (DRIFT_CASCADE): fires simultaneously with metrics-api DE-005

Stable Contract v2.1.0:
  GET /incidents/{incident_id} → status as integer (0,1,2)
  POST /incidents/{incident_id}/resolve → 204 empty body

Drifted Contract v2.2.0:
  GET /incidents/{incident_id} → status as string ("OPEN","INVESTIGATING","RESOLVED")
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="EpistemicOps Incident API", version="2.1.0")
logger = logging.getLogger("incident-api")
logging.basicConfig(level=logging.INFO)

DRIFT_MODE: str = os.getenv("DRIFT_MODE", "stable")
LOG_DIR = Path("/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
STATUS_MAP_STABLE = {0: 0, 1: 1, 2: 2}  # integer codes
STATUS_MAP_DRIFTED = {0: "OPEN", 1: "INVESTIGATING", 2: "RESOLVED"}

INCIDENTS: dict = {
    "INC-2041": {
        "incident_id": "INC-2041",
        "status": 1,
        "severity": "P2",
        "assigned_to": "oncall-team-a",
        "created_at": "2025-11-15T08:30:00Z",
        "updated_at": "2025-11-15T09:45:00Z",
        "root_cause": "redis_connection_pool_exhaustion",
        "resolution": None,
    },
    "INC-2089": {
        "incident_id": "INC-2089",
        "status": 0,
        "severity": "P2",
        "assigned_to": "oncall-team-b",
        "created_at": "2025-11-17T14:20:00Z",
        "updated_at": "2025-11-17T14:20:00Z",
        "root_cause": "UNKNOWN",
        "resolution": None,
    },
    "INC-2090": {
        "incident_id": "INC-2090",
        "status": 0,
        "severity": "P3",
        "assigned_to": "oncall-team-a",
        "created_at": "2025-11-18T10:00:00Z",
        "updated_at": "2025-11-18T10:00:00Z",
        "root_cause": "UNKNOWN",
        "resolution": None,
    },
}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ResolveRequest(BaseModel):
    resolution_notes: str
    resolved_by: str


class CreateIncidentRequest(BaseModel):
    severity: str = "P3"
    assigned_to: str = "oncall-team-a"
    description: str = ""


class DriftActivateRequest(BaseModel):
    drift_type: str


# ---------------------------------------------------------------------------
# Request/response logging
# ---------------------------------------------------------------------------
async def log_request(request: Request, response_body: dict | str, status_code: int):
    """Log every request/response to shared volume for Oversight Agent."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "incident-api",
        "method": request.method,
        "path": str(request.url.path),
        "query_params": dict(request.query_params),
        "response_status": status_code,
        "response_body": response_body,
        "drift_mode": DRIFT_MODE,
    }
    log_file = LOG_DIR / "incident_api.jsonl"
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write log: {e}")


# ---------------------------------------------------------------------------
# Helper: format status based on drift mode
# ---------------------------------------------------------------------------
def format_status(raw_status: int) -> int | str:
    """Return status in the format dictated by current drift mode."""
    if DRIFT_MODE == "stable":
        return STATUS_MAP_STABLE.get(raw_status, raw_status)
    else:
        return STATUS_MAP_DRIFTED.get(raw_status, str(raw_status))


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "incident-api", "drift_mode": DRIFT_MODE}


# ---------------------------------------------------------------------------
# Public endpoints (available to Primary Agent)
# ---------------------------------------------------------------------------
@app.get("/incidents/{incident_id}")
async def get_incident(incident_id: str, request: Request):
    """
    GET /incidents/{incident_id}
    Stable: status as integer (0=open, 1=investigating, 2=resolved)
    Drifted (DE-001): status as string enum ("OPEN", "INVESTIGATING", "RESOLVED")
    """
    if incident_id not in INCIDENTS:
        body = {"error": "incident_not_found"}
        await log_request(request, body, 404)
        raise HTTPException(status_code=404, detail="incident_not_found")

    inc = INCIDENTS[incident_id].copy()

    # Apply drift transformation to status field
    inc["status"] = format_status(inc["status"])

    # Remove internal fields not in API contract
    response_body = {
        "incident_id": inc["incident_id"],
        "status": inc["status"],
        "severity": inc["severity"],
        "assigned_to": inc["assigned_to"],
        "created_at": inc["created_at"],
        "updated_at": inc["updated_at"],
    }

    await log_request(request, response_body, 200)
    return response_body


@app.get("/incidents")
async def list_incidents(request: Request):
    """List all incidents."""
    results = []
    for inc in INCIDENTS.values():
        entry = {
            "incident_id": inc["incident_id"],
            "status": format_status(inc["status"]),
            "severity": inc["severity"],
            "assigned_to": inc["assigned_to"],
            "created_at": inc["created_at"],
            "updated_at": inc["updated_at"],
        }
        results.append(entry)

    body = {"incidents": results, "total": len(results)}
    await log_request(request, body, 200)
    return body


@app.post("/incidents/{incident_id}/resolve")
async def resolve_incident(incident_id: str, body: ResolveRequest, request: Request):
    """
    POST /incidents/{incident_id}/resolve
    Stable: 204 empty body on success
    """
    if incident_id not in INCIDENTS:
        err = {"error": "incident_not_found"}
        await log_request(request, err, 404)
        raise HTTPException(status_code=404, detail="incident_not_found")

    inc = INCIDENTS[incident_id]

    if inc["status"] == 2:
        err = {"error": "already_resolved"}
        await log_request(request, err, 400)
        raise HTTPException(status_code=400, detail="already_resolved")

    # Resolve the incident
    inc["status"] = 2
    inc["resolution"] = body.resolution_notes
    inc["updated_at"] = datetime.now(timezone.utc).isoformat()

    await log_request(request, "resolved", 204)
    return Response(status_code=204)


@app.post("/incidents")
async def create_incident(body: CreateIncidentRequest, request: Request):
    """Create a new incident."""
    inc_id = f"INC-{3000 + len(INCIDENTS)}"
    now = datetime.now(timezone.utc).isoformat()

    INCIDENTS[inc_id] = {
        "incident_id": inc_id,
        "status": 0,
        "severity": body.severity,
        "assigned_to": body.assigned_to,
        "created_at": now,
        "updated_at": now,
        "root_cause": "UNKNOWN",
        "resolution": None,
    }

    resp = {"incident_id": inc_id, "status": format_status(0), "created_at": now}
    await log_request(request, resp, 201)
    return JSONResponse(content=resp, status_code=201)


@app.put("/incidents/{incident_id}/status")
async def update_incident_status(
    incident_id: str, request: Request
):
    """Update incident status."""
    if incident_id not in INCIDENTS:
        raise HTTPException(status_code=404, detail="incident_not_found")

    body = await request.json()
    new_status = body.get("status")

    # Accept both integer and string status values
    if isinstance(new_status, str):
        reverse_map = {"OPEN": 0, "INVESTIGATING": 1, "RESOLVED": 2}
        new_status = reverse_map.get(new_status.upper(), new_status)

    if new_status not in (0, 1, 2):
        raise HTTPException(status_code=400, detail="invalid_status")

    INCIDENTS[incident_id]["status"] = new_status
    INCIDENTS[incident_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

    resp = {
        "incident_id": incident_id,
        "status": format_status(new_status),
        "updated_at": INCIDENTS[incident_id]["updated_at"],
    }
    await log_request(request, resp, 200)
    return resp


# ---------------------------------------------------------------------------
# Internal endpoints (Docker-network only, NOT available to Primary Agent)
# ---------------------------------------------------------------------------
@app.post("/internal/drift")
async def activate_drift(body: DriftActivateRequest):
    """Switch this service to drifted mode. Called by Drift Injector."""
    global DRIFT_MODE
    DRIFT_MODE = body.drift_type
    logger.info(f"🔄 Drift activated: {DRIFT_MODE}")
    return {"status": "drifted", "type": DRIFT_MODE}


@app.get("/internal/mode")
async def get_mode():
    """Return current drift mode."""
    return {"mode": DRIFT_MODE}


@app.post("/internal/reset")
async def reset_state():
    """Reset all incident state for a new era."""
    global DRIFT_MODE, INCIDENTS
    DRIFT_MODE = "stable"
    INCIDENTS = {
        "INC-2041": {
            "incident_id": "INC-2041",
            "status": 1,
            "severity": "P2",
            "assigned_to": "oncall-team-a",
            "created_at": "2025-11-15T08:30:00Z",
            "updated_at": "2025-11-15T09:45:00Z",
            "root_cause": "redis_connection_pool_exhaustion",
            "resolution": None,
        },
        "INC-2089": {
            "incident_id": "INC-2089",
            "status": 0,
            "severity": "P2",
            "assigned_to": "oncall-team-b",
            "created_at": "2025-11-17T14:20:00Z",
            "updated_at": "2025-11-17T14:20:00Z",
            "root_cause": "UNKNOWN",
            "resolution": None,
        },
        "INC-2090": {
            "incident_id": "INC-2090",
            "status": 0,
            "severity": "P3",
            "assigned_to": "oncall-team-a",
            "created_at": "2025-11-18T10:00:00Z",
            "updated_at": "2025-11-18T10:00:00Z",
            "root_cause": "UNKNOWN",
            "resolution": None,
        },
    }
    return {"status": "reset", "drift_mode": DRIFT_MODE}
