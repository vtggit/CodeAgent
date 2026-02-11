"""
Configuration management for Multi-Agent GitHub Issue Routing System.

This module provides environment-based configuration using Pydantic Settings.
"""

import os
from typing import Optional, List

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings can be overridden via environment variables or a .env file.
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

    # Optional: Local LM Studio
    lm_studio_base_url: Optional[str] = None
    """Base URL for local LM Studio instance."""

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


# Global settings instance
settings = AppSettings()


def get_settings() -> AppSettings:
    """
    Get the global application settings.

    Returns:
        AppSettings: The configured application settings.
    """
    return settings
