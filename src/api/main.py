"""
FastAPI application entry point for Multi-Agent GitHub Issue Routing System.

This module sets up the main FastAPI application with health checks,
error handling, and basic endpoints.
"""

from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Optional
import asyncio
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import routers and configuration
from src.api.webhooks import router as webhooks_router, set_queue
from src.api.deliberation import router as deliberation_router
from src.api.middleware import RequestLoggingMiddleware
from src.queue.factory import create_queue
from src.database.engine import init_db, close_db
from src.config.settings import get_settings
from src.agents.registry import AgentRegistry
from src.utils.logging import setup_logging, get_logger

# Version info
__version__ = "0.1.0"

# Global agent registry instance
_agent_registry: AgentRegistry | None = None

# Logger (initialized after setup_logging is called)
logger = get_logger(__name__)


def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry instance."""
    if _agent_registry is None:
        raise RuntimeError("Agent registry not initialized")
    return _agent_registry

# Create FastAPI app
app = FastAPI(
    title="Multi-Agent GitHub Issue Routing System",
    description=(
        "AI-powered system that automatically routes GitHub issues to a swarm of "
        "50+ specialized AI agents who collaboratively analyze, discuss, and provide "
        "expert recommendations through a moderated multi-round deliberation process."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Request logging middleware (must be added before CORS)
app.add_middleware(RequestLoggingMiddleware)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(webhooks_router)
app.include_router(deliberation_router)


# Response models
class ComponentStatus(BaseModel):
    """Status of a single system component."""

    status: str  # "healthy", "unhealthy", "degraded"
    message: Optional[str] = None
    response_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str  # "healthy", "unhealthy", "degraded"
    version: str
    timestamp: str
    service: str
    components: Optional[Dict[str, ComponentStatus]] = None
    response_time_ms: Optional[float] = None


class InfoResponse(BaseModel):
    """API information response model."""

    name: str
    version: str
    description: str
    documentation: str


# Root endpoint
@app.get("/", response_model=InfoResponse, tags=["Info"])
async def root() -> InfoResponse:
    """
    Get API information.

    Returns basic information about the API including name, version,
    and documentation links.
    """
    return InfoResponse(
        name="Multi-Agent GitHub Issue Routing System",
        version=__version__,
        description=(
            "AI-powered system for collaborative issue analysis using "
            "specialized AI agents"
        ),
        documentation="/docs",
    )


# ---------------------------------------------------------------------------
# Health / Readiness helpers
# ---------------------------------------------------------------------------

async def _check_database() -> ComponentStatus:
    """Check database connectivity by executing a simple query."""
    start = time.perf_counter()
    try:
        from src.database.engine import get_engine
        engine = get_engine()
        from sqlalchemy import text
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        elapsed = (time.perf_counter() - start) * 1000
        return ComponentStatus(status="healthy", response_time_ms=round(elapsed, 1))
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return ComponentStatus(
            status="unhealthy",
            message=str(exc)[:200],
            response_time_ms=round(elapsed, 1),
        )


async def _check_queue() -> ComponentStatus:
    """Check queue backend connectivity."""
    start = time.perf_counter()
    try:
        from src.api.webhooks import get_queue
        queue = get_queue()
        healthy = await queue.health_check()
        elapsed = (time.perf_counter() - start) * 1000
        queue_type = type(queue).__name__
        if healthy:
            return ComponentStatus(
                status="healthy",
                message=queue_type,
                response_time_ms=round(elapsed, 1),
            )
        else:
            return ComponentStatus(
                status="unhealthy",
                message=f"{queue_type}: connection failed",
                response_time_ms=round(elapsed, 1),
            )
    except RuntimeError:
        # Queue not initialized yet
        elapsed = (time.perf_counter() - start) * 1000
        return ComponentStatus(
            status="unhealthy",
            message="Queue not initialized",
            response_time_ms=round(elapsed, 1),
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return ComponentStatus(
            status="unhealthy",
            message=str(exc)[:200],
            response_time_ms=round(elapsed, 1),
        )


async def _check_agent_registry() -> ComponentStatus:
    """Check that agent registry is loaded with agents."""
    start = time.perf_counter()
    try:
        registry = get_agent_registry()
        agents = registry.get_all_agents()
        count = len(agents)
        elapsed = (time.perf_counter() - start) * 1000
        if count > 0:
            return ComponentStatus(
                status="healthy",
                message=f"{count} agents loaded",
                response_time_ms=round(elapsed, 1),
            )
        else:
            return ComponentStatus(
                status="degraded",
                message="No agents loaded",
                response_time_ms=round(elapsed, 1),
            )
    except RuntimeError:
        elapsed = (time.perf_counter() - start) * 1000
        return ComponentStatus(
            status="unhealthy",
            message="Agent registry not initialized",
            response_time_ms=round(elapsed, 1),
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return ComponentStatus(
            status="unhealthy",
            message=str(exc)[:200],
            response_time_ms=round(elapsed, 1),
        )


def _overall_status(components: Dict[str, ComponentStatus]) -> str:
    """Determine overall status from component statuses."""
    statuses = {c.status for c in components.values()}
    if "unhealthy" in statuses:
        return "unhealthy"
    if "degraded" in statuses:
        return "degraded"
    return "healthy"


# Health check endpoint (lightweight — liveness probe)
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unhealthy"},
    },
)
async def health_check(response: Response) -> HealthResponse:
    """
    Liveness health check endpoint.

    Returns the current health status of the service **including**
    component-level status for database, queue, and agent registry.
    Suitable for load balancers, Docker HEALTHCHECK, and Kubernetes
    liveness probes.

    Returns 200 if all critical components are healthy, 503 otherwise.
    """
    start = time.perf_counter()

    # Run all checks concurrently for speed
    db_status, queue_status, registry_status = await asyncio.gather(
        _check_database(),
        _check_queue(),
        _check_agent_registry(),
    )

    components = {
        "database": db_status,
        "queue": queue_status,
        "agent_registry": registry_status,
    }

    overall = _overall_status(components)
    elapsed = (time.perf_counter() - start) * 1000

    if overall == "unhealthy":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status=overall,
        version=__version__,
        timestamp=datetime.utcnow().isoformat() + "Z",
        service="multi-agent-github-router",
        components=components,
        response_time_ms=round(elapsed, 1),
    )


# Readiness check endpoint (heavier — readiness probe)
@app.get(
    "/ready",
    response_model=HealthResponse,
    tags=["Health"],
    responses={
        200: {"description": "Service is ready to accept traffic"},
        503: {"description": "Service is not ready"},
    },
)
async def readiness_check(response: Response) -> HealthResponse:
    """
    Readiness check endpoint.

    Indicates whether the service is ready to accept traffic.
    Checks all dependencies: database, queue, and agent registry.
    Used by Kubernetes readiness probes and load balancers.

    Returns 200 only when ALL components are healthy.
    """
    start = time.perf_counter()

    db_status, queue_status, registry_status = await asyncio.gather(
        _check_database(),
        _check_queue(),
        _check_agent_registry(),
    )

    components = {
        "database": db_status,
        "queue": queue_status,
        "agent_registry": registry_status,
    }

    overall = _overall_status(components)
    elapsed = (time.perf_counter() - start) * 1000

    # Readiness requires all components healthy (degraded → not ready)
    if overall != "healthy":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ready" if overall == "healthy" else "not_ready",
        version=__version__,
        timestamp=datetime.utcnow().isoformat() + "Z",
        service="multi-agent-github-router",
        components=components,
        response_time_ms=round(elapsed, 1),
    )


# Agent registry endpoints
@app.get("/agents", tags=["Agents"])
async def list_agents():
    """
    List all registered agents.

    Returns a summary of all agents in the registry with their
    name, expertise, category, priority, and type.
    """
    try:
        registry = get_agent_registry()
    except RuntimeError:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Agent registry not initialized"},
        )

    agents = []
    for agent in registry.get_all_agents():
        agents.append({
            "name": agent.name,
            "expertise": agent.expertise,
            "category": agent.category,
            "priority": agent.priority,
            "type": agent.agent_type.value,
        })

    return {
        "total": len(agents),
        "agents": agents,
    }


@app.get("/agents/summary", tags=["Agents"])
async def agents_summary():
    """
    Get a summary of the agent registry.

    Returns aggregate information about agents including counts
    by category and type.
    """
    try:
        registry = get_agent_registry()
    except RuntimeError:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Agent registry not initialized"},
        )

    return registry.get_summary()


@app.get("/agents/{agent_name}", tags=["Agents"])
async def get_agent_detail(agent_name: str):
    """
    Get detailed information about a specific agent.

    Args:
        agent_name: The unique agent identifier
    """
    try:
        registry = get_agent_registry()
    except RuntimeError:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Agent registry not initialized"},
        )

    if not registry.has_agent(agent_name):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": f"Agent '{agent_name}' not found"},
        )

    agent = registry.get_agent(agent_name)
    return agent.to_dict()


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for uncaught exceptions.

    Args:
        request: The request that caused the exception
        exc: The exception that was raised

    Returns:
        JSONResponse: Error response with details
    """
    logger.error(
        "unhandled_exception",
        error=str(exc),
        error_type=type(exc).__name__,
        path=str(request.url.path),
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc),
            "type": type(exc).__name__,
        },
    )


# Startup event
@app.on_event("startup")
async def startup_event() -> None:
    """
    Application startup event handler.

    Performs initialization tasks when the application starts:
    - Configures structured logging
    - Loads configuration
    - Initializes queue (Redis or memory fallback)
    - Initializes database connections
    - Loads agent registry
    - Sets up monitoring
    """
    # Initialize structured logging first
    app_settings = get_settings()
    setup_logging(
        log_level=app_settings.log_level,
        environment=app_settings.environment,
    )

    # Re-bind logger after setup
    global logger
    logger = get_logger(__name__)

    logger.info("app_starting", version=__version__)

    # Validate and log configuration
    try:
        warnings = app_settings.validate_for_startup()
        for warning in warnings:
            logger.warning("config_warning", message=warning)
        app_settings.log_configuration_summary()
    except ValueError as e:
        logger.error("config_validation_failed", error=str(e))
        if app_settings.is_production:
            raise

    # Initialize queue (Redis preferred, memory fallback)
    queue = await create_queue(
        queue_name="multi-agent-jobs",
        fallback_to_memory=True,
        redis_url=app_settings.redis_url,
    )
    set_queue(queue)

    # Initialize database (creates tables if they don't exist)
    try:
        await init_db()
        logger.info("database_initialized")
    except Exception as e:
        logger.warning("database_init_error", error=str(e))

    # Load agent registry
    global _agent_registry
    _agent_registry = AgentRegistry()
    try:
        config_path = str(Path(__file__).parent.parent.parent / "config" / "agent_definitions.yaml")
        count = _agent_registry.load_from_yaml(config_path)
        summary = _agent_registry.get_summary()
        logger.info(
            "agent_registry_loaded",
            total_agents=count,
            categories=summary["categories"],
        )
    except Exception as e:
        logger.error("agent_registry_load_error", error=str(e))

    logger.info("app_started", version=__version__)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event() -> None:
    """
    Application shutdown event handler.

    Performs cleanup tasks when the application shuts down:
    - Closes queue connections
    - Closes database connections
    - Flushes metrics
    """
    logger.info("app_shutting_down")

    # Close queue connections
    from src.api.webhooks import get_queue
    try:
        queue = get_queue()
        await queue.close()
        logger.info("queue_closed")
    except Exception as e:
        logger.warning("queue_close_error", error=str(e))

    # Close database connections
    try:
        await close_db()
        logger.info("database_closed")
    except Exception as e:
        logger.warning("database_close_error", error=str(e))

    logger.info("app_shutdown_complete")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
