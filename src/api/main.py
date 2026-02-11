"""
FastAPI application entry point for Multi-Agent GitHub Issue Routing System.

This module sets up the main FastAPI application with health checks,
error handling, and basic endpoints.
"""

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import routers and configuration
from src.api.webhooks import router as webhooks_router, set_queue
from src.queue.factory import create_queue
from src.database.engine import init_db, close_db
from src.config.settings import get_settings

# Version info
__version__ = "0.1.0"

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


# Response models
class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    version: str
    timestamp: str
    service: str


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


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the current health status of the service. This endpoint
    can be used by load balancers, monitoring tools, and orchestration
    systems to verify service availability.

    Returns:
        HealthResponse: Current service health status
    """
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.utcnow().isoformat() + "Z",
        service="multi-agent-github-router",
    )


# Readiness check endpoint
@app.get(
    "/ready",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
)
async def readiness_check() -> HealthResponse:
    """
    Readiness check endpoint.

    Indicates whether the service is ready to accept traffic.
    Unlike /health, this endpoint checks dependencies like
    database and Redis connections.

    Returns:
        HealthResponse: Service readiness status
    """
    # TODO: Add actual readiness checks for:
    # - Database connection
    # - Redis connection
    # - Agent registry loaded

    return HealthResponse(
        status="ready",
        version=__version__,
        timestamp=datetime.utcnow().isoformat() + "Z",
        service="multi-agent-github-router",
    )


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
    - Loads configuration
    - Initializes queue (Redis or memory fallback)
    - Initializes database connections
    - Loads agent registry
    - Sets up monitoring
    """
    print(f"Starting Multi-Agent GitHub Router v{__version__}")

    # Validate and log configuration
    app_settings = get_settings()
    try:
        warnings = app_settings.validate_for_startup()
        for warning in warnings:
            print(f"  ⚠ {warning}")
        app_settings.log_configuration_summary()
    except ValueError as e:
        print(f"  ✗ Configuration validation failed: {e}")
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
    except Exception as e:
        print(f"  ⚠ Database initialization error: {e}")

    # TODO: Add actual startup logic:
    # - Load agent definitions
    # - Initialize monitoring


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
    print("Shutting down Multi-Agent GitHub Router")

    # Close queue connections
    from src.api.webhooks import get_queue
    try:
        queue = get_queue()
        await queue.close()
        print("✓ Queue connections closed")
    except Exception as e:
        print(f"⚠ Error closing queue: {e}")

    # Close database connections
    try:
        await close_db()
    except Exception as e:
        print(f"⚠ Error closing database: {e}")

    # TODO: Add actual shutdown logic:
    # - Flush monitoring metrics


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
