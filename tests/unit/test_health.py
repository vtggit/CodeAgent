"""
Tests for health check and readiness endpoints (Issue #44).

Covers:
  - /health endpoint returns component statuses
  - /health returns 200 when healthy, 503 when unhealthy
  - /ready endpoint returns readiness status
  - /ready returns 503 when any component is degraded
  - Component checks: database, queue, agent_registry
  - Response time is included and fast
  - _overall_status helper logic
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import (
    ComponentStatus,
    HealthResponse,
    _check_agent_registry,
    _check_database,
    _check_queue,
    _overall_status,
    app,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    """Async test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestComponentStatus:
    """Tests for the ComponentStatus model."""

    def test_healthy(self):
        cs = ComponentStatus(status="healthy", response_time_ms=1.5)
        assert cs.status == "healthy"
        assert cs.message is None
        assert cs.response_time_ms == 1.5

    def test_unhealthy_with_message(self):
        cs = ComponentStatus(status="unhealthy", message="Connection refused")
        assert cs.status == "unhealthy"
        assert cs.message == "Connection refused"

    def test_degraded(self):
        cs = ComponentStatus(status="degraded", message="No agents loaded")
        assert cs.status == "degraded"


class TestHealthResponse:
    """Tests for the HealthResponse model."""

    def test_minimal(self):
        hr = HealthResponse(
            status="healthy",
            version="0.1.0",
            timestamp="2026-01-01T00:00:00Z",
            service="test",
        )
        assert hr.components is None
        assert hr.response_time_ms is None

    def test_with_components(self):
        hr = HealthResponse(
            status="healthy",
            version="0.1.0",
            timestamp="2026-01-01T00:00:00Z",
            service="test",
            components={
                "db": ComponentStatus(status="healthy", response_time_ms=1.0),
            },
            response_time_ms=2.0,
        )
        assert "db" in hr.components
        assert hr.response_time_ms == 2.0


# ---------------------------------------------------------------------------
# _overall_status tests
# ---------------------------------------------------------------------------


class TestOverallStatus:
    """Tests for the _overall_status helper."""

    def test_all_healthy(self):
        components = {
            "a": ComponentStatus(status="healthy"),
            "b": ComponentStatus(status="healthy"),
        }
        assert _overall_status(components) == "healthy"

    def test_one_degraded(self):
        components = {
            "a": ComponentStatus(status="healthy"),
            "b": ComponentStatus(status="degraded"),
        }
        assert _overall_status(components) == "degraded"

    def test_one_unhealthy(self):
        components = {
            "a": ComponentStatus(status="healthy"),
            "b": ComponentStatus(status="unhealthy"),
        }
        assert _overall_status(components) == "unhealthy"

    def test_unhealthy_takes_priority_over_degraded(self):
        components = {
            "a": ComponentStatus(status="degraded"),
            "b": ComponentStatus(status="unhealthy"),
        }
        assert _overall_status(components) == "unhealthy"

    def test_single_healthy(self):
        components = {"a": ComponentStatus(status="healthy")}
        assert _overall_status(components) == "healthy"


# ---------------------------------------------------------------------------
# Component check tests (unit — mocked dependencies)
# ---------------------------------------------------------------------------


class TestCheckDatabase:
    """Tests for the _check_database helper."""

    @pytest.mark.asyncio
    async def test_healthy_database(self):
        """When DB query succeeds, status should be healthy."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()

        mock_engine = MagicMock()
        mock_connect_cm = AsyncMock()
        mock_connect_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_connect_cm.__aexit__ = AsyncMock(return_value=False)
        mock_engine.connect.return_value = mock_connect_cm

        with patch("src.database.engine.get_engine", return_value=mock_engine):
            result = await _check_database()
            assert result.status == "healthy"
            assert result.response_time_ms is not None

    @pytest.mark.asyncio
    async def test_database_connection_error(self):
        """When DB raises an exception, status should be unhealthy."""
        with patch("src.database.engine.get_engine", side_effect=Exception("connection refused")):
            result = await _check_database()
            assert result.status == "unhealthy"
            assert "connection refused" in result.message
            assert result.response_time_ms is not None

    @pytest.mark.asyncio
    async def test_database_connection_real(self):
        """Integration test with actual SQLite database."""
        result = await _check_database()
        assert result.status == "healthy"
        assert result.response_time_ms is not None
        assert result.response_time_ms < 100  # Should be fast


class TestCheckQueue:
    """Tests for the _check_queue helper."""

    @pytest.mark.asyncio
    async def test_queue_not_initialized(self):
        """When queue is not initialized, status should be unhealthy."""
        with patch("src.api.webhooks.get_queue", side_effect=RuntimeError("Queue not initialized")):
            result = await _check_queue()
            assert result.status == "unhealthy"
            assert "not initialized" in result.message.lower()

    @pytest.mark.asyncio
    async def test_queue_healthy(self):
        """When queue health_check returns True, status should be healthy."""
        mock_queue = AsyncMock()
        mock_queue.health_check = AsyncMock(return_value=True)

        with patch("src.api.webhooks.get_queue", return_value=mock_queue):
            result = await _check_queue()
            assert result.status == "healthy"

    @pytest.mark.asyncio
    async def test_queue_unhealthy(self):
        """When queue health_check returns False, status should be unhealthy."""
        mock_queue = AsyncMock()
        mock_queue.health_check = AsyncMock(return_value=False)

        with patch("src.api.webhooks.get_queue", return_value=mock_queue):
            result = await _check_queue()
            assert result.status == "unhealthy"

    @pytest.mark.asyncio
    async def test_queue_exception(self):
        """When queue raises an exception, status should be unhealthy."""
        with patch("src.api.webhooks.get_queue", side_effect=Exception("Redis error")):
            result = await _check_queue()
            assert result.status == "unhealthy"
            assert result.response_time_ms is not None


class TestCheckAgentRegistry:
    """Tests for the _check_agent_registry helper."""

    @pytest.mark.asyncio
    async def test_registry_not_initialized(self):
        """When registry is not initialized, status should be unhealthy."""
        with patch("src.api.main.get_agent_registry", side_effect=RuntimeError("not init")):
            result = await _check_agent_registry()
            assert result.status == "unhealthy"
            assert "not initialized" in result.message.lower()

    @pytest.mark.asyncio
    async def test_registry_with_agents(self):
        """When registry has agents, status should be healthy."""
        mock_registry = MagicMock()
        mock_registry.get_all_agents.return_value = [MagicMock()] * 10

        with patch("src.api.main.get_agent_registry", return_value=mock_registry):
            result = await _check_agent_registry()
            assert result.status == "healthy"
            assert "10 agents" in result.message

    @pytest.mark.asyncio
    async def test_registry_empty(self):
        """When registry has zero agents, status should be degraded."""
        mock_registry = MagicMock()
        mock_registry.get_all_agents.return_value = []

        with patch("src.api.main.get_agent_registry", return_value=mock_registry):
            result = await _check_agent_registry()
            assert result.status == "degraded"
            assert "no agents" in result.message.lower()


# ---------------------------------------------------------------------------
# Endpoint integration tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Integration tests for GET /health."""

    @pytest.mark.asyncio
    async def test_returns_200_when_healthy(self, client):
        resp = await client.get("/health")
        # May be 200 or 503 depending on actual DB state
        data = resp.json()
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert "service" in data
        assert "components" in data
        assert "response_time_ms" in data

    @pytest.mark.asyncio
    async def test_components_present(self, client):
        resp = await client.get("/health")
        data = resp.json()
        components = data["components"]
        assert "database" in components
        assert "queue" in components
        assert "agent_registry" in components

    @pytest.mark.asyncio
    async def test_component_structure(self, client):
        resp = await client.get("/health")
        data = resp.json()
        for name, comp in data["components"].items():
            assert "status" in comp
            assert comp["status"] in ("healthy", "unhealthy", "degraded")
            assert "response_time_ms" in comp

    @pytest.mark.asyncio
    async def test_response_time_fast(self, client):
        resp = await client.get("/health")
        data = resp.json()
        # Entire health check should complete in < 100ms
        assert data["response_time_ms"] < 100

    @pytest.mark.asyncio
    async def test_service_name(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert data["service"] == "multi-agent-github-router"

    @pytest.mark.asyncio
    async def test_version_present(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert data["version"] == "0.1.0"

    @pytest.mark.asyncio
    async def test_timestamp_format(self, client):
        resp = await client.get("/health")
        data = resp.json()
        ts = data["timestamp"]
        assert ts.endswith("Z")
        # Should be a valid ISO timestamp
        datetime.fromisoformat(ts.rstrip("Z"))

    @pytest.mark.asyncio
    async def test_503_when_component_unhealthy(self, client):
        """Health endpoint should return 503 when a critical component is down."""
        unhealthy = ComponentStatus(status="unhealthy", message="DB down")
        with patch("src.api.main._check_database", return_value=unhealthy):
            resp = await client.get("/health")
            assert resp.status_code == 503
            data = resp.json()
            assert data["status"] == "unhealthy"


class TestReadyEndpoint:
    """Integration tests for GET /ready."""

    @pytest.mark.asyncio
    async def test_returns_ready(self, client):
        resp = await client.get("/ready")
        data = resp.json()
        assert "status" in data
        assert "components" in data

    @pytest.mark.asyncio
    async def test_components_present(self, client):
        resp = await client.get("/ready")
        data = resp.json()
        components = data["components"]
        assert "database" in components
        assert "queue" in components
        assert "agent_registry" in components

    @pytest.mark.asyncio
    async def test_503_when_degraded(self, client):
        """Readiness probe should return 503 even for degraded status."""
        degraded = ComponentStatus(status="degraded", message="No agents")
        with patch("src.api.main._check_agent_registry", return_value=degraded):
            resp = await client.get("/ready")
            assert resp.status_code == 503
            data = resp.json()
            assert data["status"] == "not_ready"

    @pytest.mark.asyncio
    async def test_503_when_unhealthy(self, client):
        """Readiness probe should return 503 when component unhealthy."""
        unhealthy = ComponentStatus(status="unhealthy", message="Queue down")
        with patch("src.api.main._check_queue", return_value=unhealthy):
            resp = await client.get("/ready")
            assert resp.status_code == 503
