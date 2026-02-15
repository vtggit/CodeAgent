"""
Configuration management for Multi-Agent GitHub Issue Routing System.

This module provides environment-based configuration using Pydantic Settings.
Supports loading from .env files, environment variables, and provides
startup validation with clear error messages.
"""

import os
from functools import lru_cache
from typing import Optional, List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.utils.logging import get_logger

logger = get_logger(__name__)


class AppSettings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings can be overridden via environment variables or a .env file.
    Priority: Environment variables > .env file > defaults
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore unknown environment variables
    )

    # ======================
    # GitHub Configuration
    # ======================
    github_token: Optional[str] = None
    """GitHub personal access token for API access."""
    github_webhook_secret: Optional[str] = None
    """Secret for validating GitHub webhook signatures."""

    # ======================
    # AI/LLM Configuration
    # ======================
    anthropic_api_key: Optional[str] = None
    """Anthropic API key for Claude models."""
    anthropic_model: str = "claude-sonnet-4-5-20250929"
    """Default Anthropic model to use."""

    # Optional: OpenAI (if using GPT models)
    openai_api_key: Optional[str] = None
    """OpenAI API key for GPT models."""
    openai_model: str = "gpt-4o"
    """Default OpenAI model to use."""

    # Optional: Local LM Studio
    lm_studio_base_url: Optional[str] = None
    """Base URL for local LM Studio instance."""
    lm_studio_model: Optional[str] = None
    """Model name for LM Studio local instance."""

    # Optional: Ollama
    ollama_base_url: Optional[str] = None
    """Base URL for Ollama instance (e.g., http://localhost:11434)."""
    ollama_model: Optional[str] = None
    """Model name for Ollama (e.g., llama3, codellama)."""

    # LLM Rate Limiting
    llm_rate_limit_rpm: int = 60
    """Default rate limit in requests per minute across all providers."""

    # ======================
    # Database Configuration
    # ======================
    database_url: str = "sqlite:///./data/multi_agent.db"
    """Database connection URL (SQLite, PostgreSQL, etc.)."""

    # ======================
    # Redis Configuration
    # ======================
    redis_url: str = "redis://localhost:6379/0"
    """Redis connection URL for message queue."""

    # ======================
    # Application Settings
    # ======================
    log_level: str = "INFO"
    """Logging level (DEBUG, INFO, WARNING, ERROR)."""
    environment: str = "development"
    """Application environment (development, staging, production)."""

    # Deliberation Settings
    max_rounds: int = 10
    """Maximum number of deliberation rounds."""
    convergence_threshold: float = 0.8
    """Threshold for consensus detection (0.0-1.0)."""
    min_value_threshold: float = 0.2
    """Minimum value score for accepting suggestions."""
    timeout_minutes: int = 60
    """Maximum time for a deliberation session in minutes."""

    # ======================
    # Monitoring (Optional)
    # ======================
    sentry_dsn: Optional[str] = None
    """Sentry DSN for error tracking."""
    prometheus_enabled: bool = False
    """Enable Prometheus metrics collection."""
    prometheus_port: int = 9090
    """Port for Prometheus metrics endpoint."""

    # ======================
    # Security
    # ======================
    api_key_secret: Optional[str] = None
    """Secret key for API authentication."""

    # ======================
    # Feature Flags
    # ======================
    enable_cost_tracking: bool = False
    """Enable tracking of LLM API costs."""
    enable_caching: bool = False
    """Enable caching of agent responses."""
    enable_label_filtering: bool = False
    """Enable filtering issues by labels."""

    required_labels: Optional[List[str]] = None
    """List of labels that must be present on issues for processing."""

    # ======================
    # Validators
    # ======================
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate that log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(
                f"Invalid log level '{v}'. Must be one of: {', '.join(sorted(valid_levels))}"
            )
        return upper_v

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment name."""
        valid_envs = {"development", "staging", "production"}
        lower_v = v.lower()
        if lower_v not in valid_envs:
            raise ValueError(
                f"Invalid environment '{v}'. Must be one of: {', '.join(sorted(valid_envs))}"
            )
        return lower_v

    @field_validator("convergence_threshold")
    @classmethod
    def validate_convergence_threshold(cls, v: float) -> float:
        """Validate convergence threshold is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"convergence_threshold must be between 0.0 and 1.0, got {v}"
            )
        return v

    @field_validator("min_value_threshold")
    @classmethod
    def validate_min_value_threshold(cls, v: float) -> float:
        """Validate min value threshold is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"min_value_threshold must be between 0.0 and 1.0, got {v}"
            )
        return v

    @field_validator("max_rounds")
    @classmethod
    def validate_max_rounds(cls, v: int) -> int:
        """Validate max rounds is positive."""
        if v < 1:
            raise ValueError(f"max_rounds must be at least 1, got {v}")
        return v

    @field_validator("timeout_minutes")
    @classmethod
    def validate_timeout_minutes(cls, v: int) -> int:
        """Validate timeout is positive."""
        if v < 1:
            raise ValueError(f"timeout_minutes must be at least 1, got {v}")
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    @property
    def has_github_token(self) -> bool:
        """Check if GitHub token is configured."""
        return bool(self.github_token and self.github_token != "ghp_your_github_personal_access_token_here")

    @property
    def has_anthropic_key(self) -> bool:
        """Check if Anthropic API key is configured."""
        return bool(self.anthropic_api_key and self.anthropic_api_key != "sk-ant-your_anthropic_api_key_here")

    @property
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.openai_api_key and self.openai_api_key != "sk-your_openai_api_key_here")

    @property
    def has_webhook_secret(self) -> bool:
        """Check if webhook secret is configured."""
        return bool(self.github_webhook_secret and self.github_webhook_secret != "your_webhook_secret_here")

    @property
    def available_llm_providers(self) -> list[str]:
        """List of LLM providers that have API keys configured."""
        providers = []
        if self.has_anthropic_key:
            providers.append("anthropic")
        if self.has_openai_key:
            providers.append("openai")
        if self.lm_studio_base_url:
            providers.append("lm_studio")
        if self.ollama_base_url:
            providers.append("ollama")
        return providers

    def validate_for_startup(self) -> list[str]:
        """
        Validate configuration for startup and return warnings.

        Returns a list of warning messages for missing optional configurations.
        Raises ValueError for critical missing configurations in production.
        """
        warnings = []
        errors = []

        # Check critical configurations
        if not self.has_github_token:
            if self.is_production:
                errors.append("GITHUB_TOKEN is required in production")
            else:
                warnings.append(
                    "GITHUB_TOKEN not configured - GitHub integration will not work"
                )

        if not self.has_anthropic_key:
            if self.is_production:
                errors.append("ANTHROPIC_API_KEY is required in production")
            else:
                warnings.append(
                    "ANTHROPIC_API_KEY not configured - AI agent features will not work"
                )

        if not self.has_webhook_secret:
            if self.is_production:
                errors.append("GITHUB_WEBHOOK_SECRET is required in production")
            else:
                warnings.append(
                    "GITHUB_WEBHOOK_SECRET not configured - webhook signature validation disabled"
                )

        # Raise errors for production misconfigurations
        if errors:
            error_msg = "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)

        return warnings

    def log_configuration_summary(self) -> None:
        """Log a summary of the current configuration (without secrets)."""
        logger.info(
            "configuration_summary",
            environment=self.environment,
            log_level=self.log_level,
            database_url=self.database_url,
            redis_url=self.redis_url,
            github_token_configured=self.has_github_token,
            anthropic_key_configured=self.has_anthropic_key,
            openai_key_configured=self.has_openai_key,
            webhook_secret_configured=self.has_webhook_secret,
            available_llm_providers=self.available_llm_providers,
            max_rounds=self.max_rounds,
            convergence_threshold=self.convergence_threshold,
            timeout_minutes=self.timeout_minutes,
            cost_tracking=self.enable_cost_tracking,
            caching=self.enable_caching,
            label_filtering=self.enable_label_filtering,
        )


# Global settings instance (cached)
@lru_cache()
def get_settings() -> AppSettings:
    """
    Get the global application settings (cached).

    Returns:
        AppSettings: The configured application settings.
    """
    return AppSettings()


# Convenience access - create on first import
settings = get_settings()
