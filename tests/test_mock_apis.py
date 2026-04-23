import pytest
import httpx
import os

# These tests expect the docker-compose stack to be running locally
BASE_URL = "http://localhost:8006" # drift-injector
INCIDENT_API = "http://localhost:8001"
METRICS_API = "http://localhost:8002"
DEPLOY_API = "http://localhost:8003"
LOG_API = "http://localhost:8004"
NOTIFY_API = "http://localhost:8005"

@pytest.mark.asyncio
async def test_drift_injector_reset():
    async with httpx.AsyncClient() as client:
        # Check reset endpoint
        resp = await client.post(f"{BASE_URL}/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "incident-api" in data["successes"]

@pytest.mark.asyncio
async def test_incident_api_health():
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{INCIDENT_API}/health")
        assert resp.status_code == 200
        assert resp.json().get("status") == "healthy"

@pytest.mark.asyncio
async def test_drift_injection_incident_api():
    async with httpx.AsyncClient() as client:
        # Ensure it's reset
        await client.post(f"{BASE_URL}/reset")
        
        # Inject drift
        payload = {
            "target_service": "incident-api",
            "drift_type": "DRIFT_TYPE"
        }
        resp = await client.post(f"{BASE_URL}/inject", json=payload)
        assert resp.status_code == 200
        
        # The environment or manual verification would verify the drift's effects.
        # Clean up
        await client.post(f"{BASE_URL}/reset")
