"""
Notify API — Mock Slack/PagerDuty Notification Service
Drift: DE-003 (silent 200 — delivered:false), DE-007 (rate limit cliff 100→5/min)
"""
import os, json, logging, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="EpistemicOps Notify API", version="2.0.0")
logger = logging.getLogger("notify-api")
logging.basicConfig(level=logging.INFO)

DRIFT_MODE: str = os.getenv("DRIFT_MODE", "stable")
LOG_DIR = Path("/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Rate limiting state
REQUEST_TIMESTAMPS: list[float] = []
STABLE_RATE_LIMIT = 100  # per minute
DRIFTED_RATE_LIMIT = 5   # per minute (DE-007)
NOTIFICATION_COUNTER = 0

class SendNotificationRequest(BaseModel):
    channel: str
    message: str
    urgency: str = "medium"

class DriftActivateRequest(BaseModel):
    drift_type: str

def _check_rate_limit() -> tuple[bool, int]:
    """Check if rate limit is exceeded. Returns (allowed, retry_after_seconds)."""
    now = time.time()
    window_start = now - 60
    # Clean old timestamps
    while REQUEST_TIMESTAMPS and REQUEST_TIMESTAMPS[0] < window_start:
        REQUEST_TIMESTAMPS.pop(0)
    limit = DRIFTED_RATE_LIMIT if DRIFT_MODE in ("DE-007", "DRIFT_RATE") else STABLE_RATE_LIMIT
    if len(REQUEST_TIMESTAMPS) >= limit:
        retry_after = int(60 - (now - REQUEST_TIMESTAMPS[0])) + 1
        return False, max(1, retry_after)
    REQUEST_TIMESTAMPS.append(now)
    return True, 0

async def log_request(request: Request, response_body, status_code: int):
    log_entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "service": "notify-api",
                 "method": request.method, "path": str(request.url.path),
                 "response_status": status_code, "drift_mode": DRIFT_MODE}
    try:
        with open(LOG_DIR / "notify_api.jsonl", "a") as f:
            f.write(json.dumps(log_entry, default=str) + "\n")
    except Exception:
        pass

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "notify-api", "drift_mode": DRIFT_MODE}

@app.post("/notifications/send")
async def send_notification(body: SendNotificationRequest, request: Request):
    """
    POST /notifications/send
    Stable: 200 {notification_id, delivered: true}
    DE-003: 200 {notification_id, delivered: false} — silent failure
    DE-007: rate limit drops from 100/min to 5/min
    """
    global NOTIFICATION_COUNTER

    # Rate limit check (applies in all modes, but limit differs)
    allowed, retry_after = _check_rate_limit()
    if not allowed:
        err = {"error": "rate_limit_exceeded", "retry_after_seconds": retry_after}
        await log_request(request, err, 429)
        raise HTTPException(status_code=429, detail=json.dumps(err))

    if body.urgency not in ("low", "medium", "high"):
        raise HTTPException(status_code=400, detail="invalid_urgency")

    NOTIFICATION_COUNTER += 1
    notification_id = f"NOTIF-{NOTIFICATION_COUNTER:05d}"

    # DE-003: Silent 200 — API says OK but notification not actually delivered
    if DRIFT_MODE in ("DE-003", "DRIFT_SILENT"):
        delivered = False
    else:
        delivered = True

    resp = {"notification_id": notification_id, "delivered": delivered}
    await log_request(request, resp, 200)
    return resp

@app.get("/notifications/status/{notification_id}")
async def get_notification_status(notification_id: str, request: Request):
    """Get status of a sent notification."""
    resp = {"notification_id": notification_id, "status": "delivered" if DRIFT_MODE == "stable" else "pending",
            "checked_at": datetime.now(timezone.utc).isoformat()}
    await log_request(request, resp, 200)
    return resp

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
    global DRIFT_MODE, REQUEST_TIMESTAMPS, NOTIFICATION_COUNTER
    DRIFT_MODE = "stable"
    REQUEST_TIMESTAMPS.clear()
    NOTIFICATION_COUNTER = 0
    return {"status": "reset", "drift_mode": DRIFT_MODE}
