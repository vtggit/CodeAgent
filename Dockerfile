# =============================================================================
# Multi-Agent GitHub Issue Routing System - Dockerfile
# =============================================================================
# Multi-stage build for optimal image size and security.
#
# Build stages:
#   1. builder  - Install dependencies in a venv
#   2. runtime  - Copy only the venv and application code
#
# Usage:
#   docker build -t multi-agent-router .
#   docker run -p 8000:8000 --env-file .env multi-agent-router
#
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder - install Python dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Install build dependencies (needed for some pip packages with C extensions)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Create a virtual environment for clean dependency isolation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies first (layer caching optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: Runtime - lean production image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Metadata labels
LABEL maintainer="CodeAgent Team <noreply@github.com>" \
      description="Multi-Agent GitHub Issue Routing System" \
      version="0.1.0"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Create data directory for SQLite database (owned by appuser)
RUN mkdir -p /app/data && chown -R appuser:appuser /app/data

# Copy application code and configuration files
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser alembic/ ./alembic/
COPY --chown=appuser:appuser alembic.ini .
COPY --chown=appuser:appuser pyproject.toml .

# Switch to non-root user
USER appuser

# Expose the application port
EXPOSE 8000

# Environment variable defaults (can be overridden at runtime)
ENV ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    DATABASE_URL=sqlite:///./data/multi_agent.db \
    HOST=0.0.0.0 \
    PORT=8000

# Health check - verify the application is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application with uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
