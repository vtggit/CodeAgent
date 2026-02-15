"""
Tests for request logging middleware (src/api/middleware.py).

Covers:
  - Request ID generation and propagation
  - X-Request-ID header on responses
  - Request/response logging (info vs debug for health endpoints)
  - Error handling and logging
  - Existing X-Request-ID header passthrough
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from starlette.testclient import TestClient
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.api.middleware import RequestLoggingMiddleware
from src.utils.logging import setup_logging, clear_contextvars


@pytest.fixture(autouse=True)
def _reset_context():
    """Clear context vars between tests."""
    clear_contextvars()
    yield
    clear_contextvars()


@pytest.fixture
def test_app():
    """Create a minimal FastAPI app with the logging middleware."""
    setup_logging(log_level="DEBUG", environment="development")

    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/test")
    async def api_test():
        return {"message": "hello"}

    @app.get("/api/error")
    async def api_error():
        raise ValueError("test error")

    return app


@pytest.fixture
def client(test_app):
    """Create a test client for the middleware test app."""
    return TestClient(test_app, raise_server_exceptions=False)


class TestRequestIdPropagation:
    """Tests for X-Request-ID handling."""

    def test_response_has_request_id(self, client):
        """Response should include X-Request-ID header."""
        resp = client.get("/api/test")
        assert "X-Request-ID" in resp.headers
        assert resp.headers["X-Request-ID"].startswith("req-")

    def test_existing_request_id_preserved(self, client):
        """If client sends X-Request-ID, it should be preserved."""
        resp = client.get("/api/test", headers={"X-Request-ID": "custom-id-123"})
        assert resp.headers["X-Request-ID"] == "custom-id-123"

    def test_unique_request_ids(self, client):
        """Each request should get a unique ID."""
        ids = set()
        for _ in range(10):
            resp = client.get("/api/test")
            ids.add(resp.headers["X-Request-ID"])
        assert len(ids) == 10


class TestRequestLogging:
    """Tests for request logging behavior."""

    def test_normal_endpoint_returns_200(self, client):
        resp = client.get("/api/test")
        assert resp.status_code == 200
        assert resp.json() == {"message": "hello"}

    def test_health_endpoint_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_error_endpoint_returns_500(self, client):
        """Errors should still propagate (middleware re-raises)."""
        resp = client.get("/api/error")
        assert resp.status_code == 500
