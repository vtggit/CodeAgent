"""
Tests for Docker configuration files.

Validates that Dockerfile, docker-compose.yml, docker-compose.override.yml,
and .dockerignore are well-formed and meet all acceptance criteria for
Issues #27 (Docker containerization) and #28 (Docker Compose setup).
"""

import os
import re
from pathlib import Path

import pytest
import yaml

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestDockerfile:
    """Validate the Dockerfile meets all acceptance criteria for Issue #27."""

    @pytest.fixture
    def dockerfile_content(self) -> str:
        dockerfile = PROJECT_ROOT / "Dockerfile"
        assert dockerfile.exists(), "Dockerfile not found in project root"
        return dockerfile.read_text()

    def test_dockerfile_exists(self):
        """Dockerfile should exist in the project root."""
        assert (PROJECT_ROOT / "Dockerfile").exists()

    def test_multi_stage_build(self, dockerfile_content: str):
        """Dockerfile must use multi-stage build (builder + runtime)."""
        stages = re.findall(r"FROM\s+\S+\s+AS\s+(\w+)", dockerfile_content)
        assert len(stages) >= 2, f"Expected at least 2 build stages, found {len(stages)}"
        assert "builder" in stages, "Missing 'builder' stage"
        assert "runtime" in stages, "Missing 'runtime' stage"

    def test_base_image_python_slim(self, dockerfile_content: str):
        """Must be based on python:3.11-slim for optimal size."""
        assert "python:3.11-slim" in dockerfile_content

    def test_non_root_user(self, dockerfile_content: str):
        """Must create and use a non-root user for security."""
        assert "groupadd" in dockerfile_content, "Missing groupadd for non-root user"
        assert "useradd" in dockerfile_content, "Missing useradd for non-root user"
        assert "USER appuser" in dockerfile_content, "Missing USER directive to switch to non-root"

    def test_only_production_dependencies(self, dockerfile_content: str):
        """Only production dependencies should be in the final image."""
        assert "requirements.txt" in dockerfile_content, "Missing requirements.txt installation"
        assert (
            "requirements-dev.txt" not in dockerfile_content
        ), "Dev dependencies should NOT be included"

    def test_healthcheck_configured(self, dockerfile_content: str):
        """HEALTHCHECK instruction must be present."""
        assert "HEALTHCHECK" in dockerfile_content
        assert "/health" in dockerfile_content, "Health check should hit the /health endpoint"

    def test_port_exposed(self, dockerfile_content: str):
        """Port 8000 should be exposed."""
        assert "EXPOSE 8000" in dockerfile_content

    def test_venv_from_builder(self, dockerfile_content: str):
        """Virtual environment should be copied from builder stage."""
        assert "COPY --from=builder /opt/venv /opt/venv" in dockerfile_content

    def test_application_code_copied(self, dockerfile_content: str):
        """Application code should be copied with correct ownership."""
        assert "COPY --chown=appuser:appuser src/" in dockerfile_content
        assert "COPY --chown=appuser:appuser config/" in dockerfile_content
        assert "COPY --chown=appuser:appuser alembic/" in dockerfile_content

    def test_cmd_runs_uvicorn(self, dockerfile_content: str):
        """CMD should run uvicorn with the correct module path."""
        assert "uvicorn" in dockerfile_content
        assert "src.api.main:app" in dockerfile_content

    def test_data_directory_created(self, dockerfile_content: str):
        """Data directory should be created for SQLite database."""
        assert "/app/data" in dockerfile_content

    def test_python_optimization_envs(self, dockerfile_content: str):
        """Python optimization environment variables should be set."""
        assert "PYTHONDONTWRITEBYTECODE=1" in dockerfile_content
        assert "PYTHONUNBUFFERED=1" in dockerfile_content

    def test_documentation_comments(self, dockerfile_content: str):
        """Dockerfile should have documentation comments."""
        # Count comment lines (starting with #)
        comment_lines = [
            line for line in dockerfile_content.split("\n") if line.strip().startswith("#")
        ]
        assert len(comment_lines) >= 10, (
            f"Expected at least 10 comment lines for documentation, found {len(comment_lines)}"
        )

    def test_workdir_set(self, dockerfile_content: str):
        """WORKDIR should be set in runtime stage."""
        assert "WORKDIR /app" in dockerfile_content


class TestDockerignore:
    """Validate the .dockerignore excludes unnecessary files."""

    @pytest.fixture
    def dockerignore_content(self) -> str:
        ignore_file = PROJECT_ROOT / ".dockerignore"
        assert ignore_file.exists(), ".dockerignore not found in project root"
        return ignore_file.read_text()

    def test_dockerignore_exists(self):
        """.dockerignore should exist in the project root."""
        assert (PROJECT_ROOT / ".dockerignore").exists()

    @pytest.mark.parametrize(
        "pattern",
        [
            "venv/",
            ".git",
            "__pycache__/",
            "tests/",
            ".env",
            "*.log",
            "data/",
            "screenshots/",
            ".pytest_cache/",
            ".coverage",
            "htmlcov/",
            "*.db",
            ".vscode/",
            ".idea/",
            "*.egg-info/",
            "requirements-dev.txt",
        ],
    )
    def test_excludes_pattern(self, dockerignore_content: str, pattern: str):
        """Important patterns should be excluded from Docker build context."""
        assert pattern in dockerignore_content, f".dockerignore should exclude '{pattern}'"

    def test_excludes_secrets(self, dockerignore_content: str):
        """Secrets and sensitive files should be excluded."""
        assert ".env" in dockerignore_content
        assert "*.pem" in dockerignore_content
        assert "*.key" in dockerignore_content
        assert "secrets/" in dockerignore_content


class TestDockerCompose:
    """Validate docker-compose.yml meets acceptance criteria for Issue #28."""

    @pytest.fixture
    def compose_config(self) -> dict:
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        assert compose_file.exists(), "docker-compose.yml not found"
        with open(compose_file) as f:
            return yaml.safe_load(f)

    def test_docker_compose_exists(self):
        """docker-compose.yml should exist in project root."""
        assert (PROJECT_ROOT / "docker-compose.yml").exists()

    def test_required_services(self, compose_config: dict):
        """Must define web, worker, and redis services."""
        services = compose_config.get("services", {})
        assert "web" in services, "Missing 'web' service"
        assert "worker" in services, "Missing 'worker' service"
        assert "redis" in services, "Missing 'redis' service"

    def test_redis_service(self, compose_config: dict):
        """Redis service should use alpine image with health check."""
        redis = compose_config["services"]["redis"]
        assert "redis" in redis["image"]
        assert "alpine" in redis["image"]
        assert "healthcheck" in redis
        assert "6379:6379" in redis["ports"]

    def test_web_service_build(self, compose_config: dict):
        """Web service should build from Dockerfile."""
        web = compose_config["services"]["web"]
        assert "build" in web
        assert web["build"]["dockerfile"] == "Dockerfile"

    def test_web_service_port(self, compose_config: dict):
        """Web service should expose port 8000."""
        web = compose_config["services"]["web"]
        assert "8000:8000" in web["ports"]

    def test_web_service_depends_on_redis(self, compose_config: dict):
        """Web service should depend on Redis being healthy."""
        web = compose_config["services"]["web"]
        assert "depends_on" in web
        assert "redis" in web["depends_on"]

    def test_web_service_env_file(self, compose_config: dict):
        """Web service should use .env file."""
        web = compose_config["services"]["web"]
        assert ".env" in web.get("env_file", [])

    def test_web_service_healthcheck(self, compose_config: dict):
        """Web service should have a health check."""
        web = compose_config["services"]["web"]
        assert "healthcheck" in web

    def test_web_service_redis_url_override(self, compose_config: dict):
        """Web service should override REDIS_URL for container networking."""
        web = compose_config["services"]["web"]
        env = web.get("environment", [])
        redis_urls = [e for e in env if "REDIS_URL" in e]
        assert len(redis_urls) > 0, "Web must set REDIS_URL for container networking"
        assert "redis://redis:6379" in redis_urls[0], "REDIS_URL should point to redis service"

    def test_worker_service(self, compose_config: dict):
        """Worker service should use custom command and depend on web + redis."""
        worker = compose_config["services"]["worker"]
        assert "command" in worker
        assert "depends_on" in worker
        assert "redis" in worker["depends_on"]
        assert "web" in worker["depends_on"]

    def test_worker_service_redis_url(self, compose_config: dict):
        """Worker service should override REDIS_URL for container networking."""
        worker = compose_config["services"]["worker"]
        env = worker.get("environment", [])
        redis_urls = [e for e in env if "REDIS_URL" in e]
        assert len(redis_urls) > 0

    def test_volumes_defined(self, compose_config: dict):
        """Named volumes should be defined for data persistence."""
        volumes = compose_config.get("volumes", {})
        assert "redis-data" in volumes, "Missing redis-data volume"
        assert "sqlite-data" in volumes, "Missing sqlite-data volume"

    def test_redis_data_persistence(self, compose_config: dict):
        """Redis should mount a volume for data persistence."""
        redis = compose_config["services"]["redis"]
        volume_mounts = redis.get("volumes", [])
        assert any("redis-data" in v for v in volume_mounts), "Redis should use redis-data volume"

    def test_sqlite_data_persistence(self, compose_config: dict):
        """Web and worker services should share SQLite data volume."""
        web = compose_config["services"]["web"]
        worker = compose_config["services"]["worker"]
        web_vols = web.get("volumes", [])
        worker_vols = worker.get("volumes", [])
        assert any("sqlite-data" in v for v in web_vols), "Web should mount sqlite-data"
        assert any("sqlite-data" in v for v in worker_vols), "Worker should mount sqlite-data"

    def test_network_defined(self, compose_config: dict):
        """Custom network should be defined for service discovery."""
        networks = compose_config.get("networks", {})
        assert len(networks) > 0, "At least one network should be defined"

    def test_services_on_network(self, compose_config: dict):
        """All services should be on the same network."""
        network_name = list(compose_config.get("networks", {}).keys())[0]
        for svc_name, svc in compose_config["services"].items():
            assert "networks" in svc, f"Service '{svc_name}' should be on a network"
            assert network_name in svc["networks"], (
                f"Service '{svc_name}' should be on '{network_name}'"
            )

    def test_restart_policies(self, compose_config: dict):
        """Services should have restart policies."""
        for svc_name in ["redis", "web", "worker"]:
            svc = compose_config["services"][svc_name]
            assert "restart" in svc, f"Service '{svc_name}' should have a restart policy"


class TestDockerComposeOverride:
    """Validate docker-compose.override.yml for local development."""

    @pytest.fixture
    def override_config(self) -> dict:
        override_file = PROJECT_ROOT / "docker-compose.override.yml"
        assert override_file.exists(), "docker-compose.override.yml not found"
        with open(override_file) as f:
            return yaml.safe_load(f)

    def test_override_exists(self):
        """docker-compose.override.yml should exist for local customization."""
        assert (PROJECT_ROOT / "docker-compose.override.yml").exists()

    def test_web_hot_reload(self, override_config: dict):
        """Web service should have hot reload enabled in development."""
        web = override_config["services"]["web"]
        assert "--reload" in web.get("command", "")

    def test_web_volume_mounts(self, override_config: dict):
        """Web service should mount source code for hot reload."""
        web = override_config["services"]["web"]
        volumes = web.get("volumes", [])
        src_mounts = [v for v in volumes if "./src" in v]
        assert len(src_mounts) > 0, "Source code should be mounted for hot reload"

    def test_development_environment(self, override_config: dict):
        """Override should set development environment."""
        web = override_config["services"]["web"]
        env = web.get("environment", [])
        assert any("ENVIRONMENT=development" in e for e in env)

    def test_debug_logging(self, override_config: dict):
        """Override should enable debug logging."""
        web = override_config["services"]["web"]
        env = web.get("environment", [])
        assert any("LOG_LEVEL=DEBUG" in e for e in env)

    def test_worker_development_env(self, override_config: dict):
        """Worker should also use development settings."""
        worker = override_config["services"]["worker"]
        env = worker.get("environment", [])
        assert any("ENVIRONMENT=development" in e for e in env)

    def test_config_volume_mounted(self, override_config: dict):
        """Config directory should be mounted for development."""
        web = override_config["services"]["web"]
        volumes = web.get("volumes", [])
        config_mounts = [v for v in volumes if "./config" in v]
        assert len(config_mounts) > 0, "Config should be mounted for development"
