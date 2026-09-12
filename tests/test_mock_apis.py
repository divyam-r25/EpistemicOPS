"""
test_mock_apis.py — Tests for the mock Docker API stack.

These tests require the docker-compose stack to be running.
They are automatically SKIPPED in offline mode (EPISTEMICOPS_OFFLINE=true)
via conftest.py's `skip_if_requires_docker` fixture.
"""
import pytest
import httpx

# These tests expect the docker-compose stack to be running locally
BASE_URL = "http://localhost:8006"  # drift-injector
INCIDENT_API = "http://localhost:8001"
METRICS_API = "http://localhost:8002"
DEPLOY_API = "http://localhost:8003"
LOG_API = "http://localhost:8004"
NOTIFY_API = "http://localhost:8005"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_drift_injector_reset():
    """Drift injector /reset endpoint should return success with all services listed."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{BASE_URL}/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "incident-api" in data["successes"]


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_incident_api_health():
    """Incident API health endpoint should report healthy."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{INCIDENT_API}/health")
        assert resp.status_code == 200
        assert resp.json().get("status") == "healthy"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_drift_injection_incident_api():
    """Injecting drift into incident-api should succeed, then reset cleanly."""
    async with httpx.AsyncClient() as client:
        # Ensure clean state
        await client.post(f"{BASE_URL}/reset")

        # Inject drift
        payload = {
            "target_service": "incident-api",
            "drift_type": "DRIFT_TYPE"
        }
        resp = await client.post(f"{BASE_URL}/inject", json=payload)
        assert resp.status_code == 200

        # Clean up
        await client.post(f"{BASE_URL}/reset")
