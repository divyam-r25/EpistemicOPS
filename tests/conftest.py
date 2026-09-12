"""
conftest.py — Pytest configuration for EpistemicOPS test suite.

Defines marks and fixtures shared across all test files.
"""
import os
import pytest


def pytest_configure(config):
    """Register custom marks."""
    config.addinivalue_line(
        "markers",
        "requires_docker: mark test as requiring Docker services to be running"
    )
    config.addinivalue_line(
        "markers",
        "requires_gpu: mark test as requiring a CUDA GPU"
    )
    config.addinivalue_line(
        "markers",
        "requires_openai: mark test as requiring OPENAI_API_KEY to be set"
    )


@pytest.fixture(scope="session")
def offline_mode() -> bool:
    """Whether the environment is running in offline (no-Docker) mode."""
    return os.getenv("EPISTEMICOPS_OFFLINE", "true").lower() == "true"


@pytest.fixture(autouse=True)
def skip_if_requires_docker(request, offline_mode):
    """Automatically skip tests marked requires_docker when in offline mode."""
    if request.node.get_closest_marker("requires_docker") and offline_mode:
        pytest.skip("Docker services not available in offline mode (set EPISTEMICOPS_OFFLINE=false to run)")


@pytest.fixture(autouse=True)
def skip_if_requires_openai(request):
    """Automatically skip tests marked requires_openai when API key is missing."""
    if request.node.get_closest_marker("requires_openai"):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set — skipping LLM-dependent test")
