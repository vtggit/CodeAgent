"""
FastAPI middleware for structured logging and request tracking.

Provides:
  - Request ID generation and propagation
  - Automatic request/response logging
  - Context variable binding for request-scoped logging
"""

import time
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from src.utils.logging import (
    get_logger,
    bind_contextvars,
    clear_contextvars,
    generate_request_id,
)

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds structured logging to every HTTP request.

    Features:
      - Generates a unique request_id for each request
      - Binds request_id to structlog contextvars for all downstream logs
      - Logs request start and completion with timing
      - Adds X-Request-ID header to responses
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Generate request ID
        request_id = request.headers.get("X-Request-ID") or generate_request_id()

        # Bind request context for all loggers in this async context
        clear_contextvars()
        bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        start_time = time.perf_counter()

        # Log request start (skip noisy health checks at debug level)
        if request.url.path in ("/health", "/ready"):
            logger.debug("request_started")
        else:
            logger.info("request_started",
                        client=request.client.host if request.client else "unknown")

        try:
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            # Log completion
            if request.url.path in ("/health", "/ready"):
                logger.debug(
                    "request_completed",
                    status_code=response.status_code,
                    duration_ms=round(elapsed_ms, 1),
                )
            else:
                logger.info(
                    "request_completed",
                    status_code=response.status_code,
                    duration_ms=round(elapsed_ms, 1),
                )

            return response

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "request_failed",
                status_code=500,
                duration_ms=round(elapsed_ms, 1),
                error=str(exc),
                error_type=type(exc).__name__,
                exc_info=True,
            )
            raise

        finally:
            clear_contextvars()
