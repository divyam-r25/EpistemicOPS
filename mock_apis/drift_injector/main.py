import os
import httpx
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("drift-injector")

app = FastAPI(title="EpistemicOps Drift Injector")

# Internal URLs for the mock API containers
SERVICE_URLS = {
    "incident-api": os.getenv("INCIDENT_API_URL", "http://incident-api:8000"),
    "metrics-api": os.getenv("METRICS_API_URL", "http://metrics-api:8000"),
    "deploy-api": os.getenv("DEPLOY_API_URL", "http://deploy-api:8000"),
    "log-api": os.getenv("LOG_API_URL", "http://log-api:8000"),
    "notify-api": os.getenv("NOTIFY_API_URL", "http://notify-api:8000")
}

class DriftRequest(BaseModel):
    target_service: str
    drift_type: str

@app.post("/inject")
async def inject_drift(request: DriftRequest):
    if request.target_service not in SERVICE_URLS:
        raise HTTPException(status_code=400, detail=f"Unknown service: {request.target_service}")
        
    url = f"{SERVICE_URLS[request.target_service]}/internal/drift"
    logger.info(f"Injecting drift {request.drift_type} to {url}")
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json={"drift_type": request.drift_type}, timeout=5.0)
            resp.raise_for_status()
            return {"status": "success", "message": f"Injected {request.drift_type} into {request.target_service}"}
    except Exception as e:
        logger.error(f"Failed to inject drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_drifts():
    successes = []
    failures = []
    
    async with httpx.AsyncClient() as client:
        for service, base_url in SERVICE_URLS.items():
            try:
                url = f"{base_url}/internal/reset"
                resp = await client.post(url, timeout=5.0)
                resp.raise_for_status()
                successes.append(service)
            except Exception as e:
                logger.error(f"Failed to reset {service}: {e}")
                failures.append(service)
                
    return {
        "status": "success" if not failures else "partial",
        "successes": successes,
        "failures": failures
    }
