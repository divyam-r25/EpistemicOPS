"""
Deploy API — Mock Kubernetes/ArgoCD Service
=============================================
Simulates deployment management with stable and drifted modes.

Drift Events:
  - DE-002 (DRIFT_STATUS): rollback returns 204 empty body instead of 200 with body
  - DE-004 (DRIFT_AUTH): auth header changes from X-Deploy-Token to Authorization: Bearer

Stable Contract v3.0.1:
  POST /deployments/rollback → 200 with {rollback_id, status, eta_seconds}
  Auth: X-Deploy-Token header

Drifted Contract:
  DE-002: POST /deployments/rollback → 204 empty body (action succeeds silently)
  DE-004: Auth header changes to Authorization: Bearer {token}
"""

import os
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="EpistemicOps Deploy API", version="3.0.1")
logger = logging.getLogger("deploy-api")
logging.basicConfig(level=logging.INFO)

DRIFT_MODE: str = os.getenv("DRIFT_MODE", "stable")
LOG_DIR = Path("/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Valid deploy token
VALID_TOKEN = "epistemicops-deploy-token-2025"


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
DEPLOYMENTS = {
    "DEP-441": {
        "deploy_id": "DEP-441",
        "era": 2,
        "service": "payment-service",
        "version": "3.2.1",
        "status": "LIVE",
        "known_issues": ["memory_leak_under_load"],
        "deployed_at": "2025-11-16T12:00:00Z",
    },
    "DEP-440": {
        "deploy_id": "DEP-440",
        "era": 1,
        "service": "payment-service",
        "version": "3.2.0",
        "status": "ROLLED_BACK",
        "known_issues": [],
        "deployed_at": "2025-11-15T10:00:00Z",
    },
}

ROLLBACK_IN_PROGRESS: Optional[str] = None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class RollbackRequest(BaseModel):
    service: str
    target_version: str
    reason: str


class DeployRequest(BaseModel):
    service: str
    version: str
    environment: str = "production"


class DriftActivateRequest(BaseModel):
    drift_type: str


# ---------------------------------------------------------------------------
# Auth validation
# ---------------------------------------------------------------------------
def validate_auth(request: Request) -> bool:
    """Validate authentication based on current drift mode."""
    if DRIFT_MODE == "stable" or DRIFT_MODE == "DE-002":
        # Stable auth: X-Deploy-Token header
        token = request.headers.get("X-Deploy-Token", "")
        return token == VALID_TOKEN
    elif DRIFT_MODE in ("DE-004", "DRIFT_AUTH"):
        # Drifted auth: Authorization: Bearer header
        auth_header = request.headers.get("Authorization", "")
        return auth_header == f"Bearer {VALID_TOKEN}"
    else:
        # Unknown drift — accept both formats
        token = request.headers.get("X-Deploy-Token", "")
        auth_header = request.headers.get("Authorization", "")
        return token == VALID_TOKEN or auth_header == f"Bearer {VALID_TOKEN}"


# ---------------------------------------------------------------------------
# Request logging
# ---------------------------------------------------------------------------
async def log_request(request: Request, response_body, status_code: int):
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "deploy-api",
        "method": request.method,
        "path": str(request.url.path),
        "response_status": status_code,
        "response_body": response_body,
        "drift_mode": DRIFT_MODE,
    }
    log_file = LOG_DIR / "deploy_api.jsonl"
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
    return {"status": "healthy", "service": "deploy-api", "drift_mode": DRIFT_MODE}


# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------
@app.post("/deployments/rollback")
async def rollback_deployment(body: RollbackRequest, request: Request):
    """
    POST /deployments/rollback
    Stable: 200 with body {rollback_id, status, eta_seconds}
    DE-002: 204 empty body (action succeeds but no confirmation data)
    DE-004: Auth header must be 'Authorization: Bearer {token}' instead of 'X-Deploy-Token'
    """
    global ROLLBACK_IN_PROGRESS

    # Auth check
    if not validate_auth(request):
        err = {"error": "invalid_token"}
        await log_request(request, err, 401)
        raise HTTPException(status_code=401, detail="invalid_token")

    # Check for in-progress rollback
    if ROLLBACK_IN_PROGRESS:
        err = {"error": "rollback_in_progress"}
        await log_request(request, err, 409)
        raise HTTPException(status_code=409, detail="rollback_in_progress")

    # Execute rollback
    rollback_id = f"RB-{uuid.uuid4().hex[:8]}"
    ROLLBACK_IN_PROGRESS = rollback_id

    # Update deployment state
    for dep in DEPLOYMENTS.values():
        if dep["service"] == body.service and dep["status"] == "LIVE":
            dep["status"] = "ROLLING_BACK"

    # DE-002 DRIFT: return 204 with empty body instead of 200 with data
    if DRIFT_MODE in ("DE-002", "DRIFT_STATUS"):
        await log_request(request, "empty_body_204", 204)
        return Response(status_code=204)

    # Stable: return 200 with full body
    resp = {
        "rollback_id": rollback_id,
        "status": "INITIATED",
        "eta_seconds": 120,
    }
    await log_request(request, resp, 200)
    return resp


@app.get("/deployments")
async def list_deployments(request: Request):
    """List all deployments."""
    body = {"deployments": list(DEPLOYMENTS.values()), "total": len(DEPLOYMENTS)}
    await log_request(request, body, 200)
    return body


@app.get("/deployments/{deploy_id}")
async def get_deployment(deploy_id: str, request: Request):
    """Get deployment details."""
    if deploy_id not in DEPLOYMENTS:
        raise HTTPException(status_code=404, detail="deployment_not_found")

    body = DEPLOYMENTS[deploy_id]
    await log_request(request, body, 200)
    return body


@app.post("/deployments")
async def create_deployment(body: DeployRequest, request: Request):
    """Create a new deployment."""
    if not validate_auth(request):
        raise HTTPException(status_code=401, detail="invalid_token")

    deploy_id = f"DEP-{500 + len(DEPLOYMENTS)}"
    now = datetime.now(timezone.utc).isoformat()

    DEPLOYMENTS[deploy_id] = {
        "deploy_id": deploy_id,
        "era": 0,
        "service": body.service,
        "version": body.version,
        "status": "DEPLOYING",
        "known_issues": [],
        "deployed_at": now,
    }

    resp = {"deploy_id": deploy_id, "status": "DEPLOYING", "deployed_at": now}
    await log_request(request, resp, 201)
    return JSONResponse(content=resp, status_code=201)


@app.post("/deployments/{deploy_id}/complete")
async def complete_deployment(deploy_id: str, request: Request):
    """Mark deployment as live."""
    global ROLLBACK_IN_PROGRESS

    if deploy_id not in DEPLOYMENTS:
        raise HTTPException(status_code=404, detail="deployment_not_found")

    DEPLOYMENTS[deploy_id]["status"] = "LIVE"
    ROLLBACK_IN_PROGRESS = None

    resp = {"deploy_id": deploy_id, "status": "LIVE"}
    await log_request(request, resp, 200)
    return resp


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
    global DRIFT_MODE, ROLLBACK_IN_PROGRESS, DEPLOYMENTS
    DRIFT_MODE = "stable"
    ROLLBACK_IN_PROGRESS = None
    DEPLOYMENTS = {
        "DEP-441": {
            "deploy_id": "DEP-441",
            "era": 2,
            "service": "payment-service",
            "version": "3.2.1",
            "status": "LIVE",
            "known_issues": ["memory_leak_under_load"],
            "deployed_at": "2025-11-16T12:00:00Z",
        },
        "DEP-440": {
            "deploy_id": "DEP-440",
            "era": 1,
            "service": "payment-service",
            "version": "3.2.0",
            "status": "ROLLED_BACK",
            "known_issues": [],
            "deployed_at": "2025-11-15T10:00:00Z",
        },
    }
    return {"status": "reset", "drift_mode": DRIFT_MODE}
