"""
Log API — Mock Splunk/ELK Service
Drift: DE-008 (offset → cursor pagination)
"""
import os, json, logging, hashlib, random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Request, Query
from pydantic import BaseModel

app = FastAPI(title="EpistemicOps Log API", version="1.1.0")
logger = logging.getLogger("log-api")
logging.basicConfig(level=logging.INFO)

DRIFT_MODE: str = os.getenv("DRIFT_MODE", "stable")
LOG_DIR = Path("/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_TEMPLATES = [
    {"severity": "ERROR", "message": "Connection refused to redis-primary:6379"},
    {"severity": "ERROR", "message": "Timeout waiting for response from payment-gateway"},
    {"severity": "ERROR", "message": "OOM killer invoked on payment-service pod"},
    {"severity": "WARN", "message": "Connection pool nearing capacity: 185/200"},
    {"severity": "WARN", "message": "Retry attempt 3/5 for upstream auth-service"},
    {"severity": "INFO", "message": "Deployment DEP-441 health check passed"},
    {"severity": "INFO", "message": "Auto-scaling triggered: 3→5 replicas"},
    {"severity": "ERROR", "message": "500 Internal Server Error on POST /api/v2/charge"},
    {"severity": "ERROR", "message": "Database connection pool exhausted"},
]

SERVICE_LOGS = {}
CURSOR_MAP: dict[str, int] = {}

class DriftActivateRequest(BaseModel):
    drift_type: str

def _generate_logs(service: str, count: int = 200) -> list[dict]:
    if service in SERVICE_LOGS:
        return SERVICE_LOGS[service]
    random.seed(hash(service) % 2**32)
    logs = []
    now = datetime.now(timezone.utc)
    for i in range(count):
        t = random.choice(LOG_TEMPLATES)
        ts = now - timedelta(minutes=random.randint(0, 1440))
        logs.append({"timestamp": ts.isoformat(), "severity": t["severity"],
                      "message": t["message"], "trace_id": f"tr-{i:04d}", "service": service})
    logs.sort(key=lambda x: x["timestamp"], reverse=True)
    SERVICE_LOGS[service] = logs
    return logs

def _make_cursor(offset: int) -> str:
    return hashlib.sha256(f"cursor-{offset}".encode()).hexdigest()[:16]

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "log-api", "drift_mode": DRIFT_MODE}

@app.get("/logs/query")
async def query_logs(request: Request, service: str = Query(default="payment-service"),
                     severity: Optional[str] = None, since: Optional[str] = None,
                     limit: int = Query(default=50, le=200), page: int = Query(default=1, ge=1),
                     cursor: Optional[str] = None):
    all_logs = _generate_logs(service)
    if severity:
        all_logs = [l for l in all_logs if l["severity"] in severity.upper().split(",")]
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
            all_logs = [l for l in all_logs if l["timestamp"] >= since_dt.isoformat()]
        except (ValueError, TypeError):
            pass
    total = len(all_logs)

    if DRIFT_MODE == "stable":
        start = (page - 1) * limit
        page_logs = all_logs[start:start + limit]
        return {"logs": page_logs, "total": total, "page": page, "limit": limit,
                "truncated": (start + limit) < total}
    else:
        offset = CURSOR_MAP.get(cursor, 0) if cursor else 0
        page_logs = all_logs[offset:offset + limit]
        truncated = (offset + limit) < total
        resp = {"logs": page_logs, "total": total, "truncated": truncated, "limit": limit}
        if truncated:
            nc = _make_cursor(offset + limit)
            CURSOR_MAP[nc] = offset + limit
            resp["next_cursor"] = nc
        return resp

@app.post("/internal/drift")
async def activate_drift(body: DriftActivateRequest):
    global DRIFT_MODE
    DRIFT_MODE = body.drift_type
    return {"status": "drifted", "type": DRIFT_MODE}

@app.get("/internal/mode")
async def get_mode():
    return {"mode": DRIFT_MODE}

@app.post("/internal/reset")
async def reset_state():
    global DRIFT_MODE
    DRIFT_MODE = "stable"
    SERVICE_LOGS.clear()
    CURSOR_MAP.clear()
    return {"status": "reset", "drift_mode": DRIFT_MODE}
