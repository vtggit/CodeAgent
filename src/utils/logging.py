"""
Structured logging configuration for Multi-Agent GitHub Issue Routing System.

This module configures Structlog as the application's default logging system,
providing:
  - JSON output in production for log aggregation
  - Human-readable colored output in development
  - Context propagation (workflow_id, round, agent_name)
  - Request ID tracking for API requests
  - Integration with standard library logging
  - Automatic filtering of sensitive data (tokens, secrets)

Usage:
    from src.utils.logging import setup_logging, get_logger

    # At application startup:
    setup_logging(log_level="INFO", environment="development")

    # In any module:
    logger = get_logger(__name__)
    logger.info("processing_issue", issue_number=42, agent="ui_architect")

    # With bound context (persists across calls):
    logger = logger.bind(workflow_id="wf-123", round=3)
    logger.info("round_started", agent_count=5)
"""

import logging
import re
import sys
import uuid
from typing import Any, Optional

import structlog
from structlog.types import EventDict, WrappedLogger


# Patterns for sensitive data that should never be logged
_SENSITIVE_PATTERNS = [
    re.compile(r"(ghp_[a-zA-Z0-9]{36,})"),           # GitHub PAT
    re.compile(r"(ghs_[a-zA-Z0-9]{36,})"),           # GitHub App token
    re.compile(r"(sk-ant-[a-zA-Z0-9\-]{40,})"),      # Anthropic API key
    re.compile(r"(sk-[a-zA-Z0-9]{40,})"),             # OpenAI API key
    re.compile(r"(xoxb-[a-zA-Z0-9\-]+)"),            # Slack bot token
    re.compile(r"(Bearer\s+[a-zA-Z0-9\-_.]+)"),      # Bearer tokens
]

_SENSITIVE_KEYS = frozenset({
    "token", "secret", "password", "api_key", "apikey",
    "authorization", "auth", "credentials", "private_key",
    "access_token", "refresh_token", "webhook_secret",
    "anthropic_api_key", "openai_api_key", "github_token",
})


def _sanitize_value(value: Any) -> Any:
    """Redact sensitive values from log output."""
    if isinstance(value, str):
        for pattern in _SENSITIVE_PATTERNS:
            if pattern.search(value):
                return "***REDACTED***"
    return value


def _sanitize_event_dict(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """
    Structlog processor that redacts sensitive data from log events.

    Checks both key names and string values for sensitive patterns.
    """
    sanitized = {}
    for key, value in event_dict.items():
        # Check if the key name indicates sensitive data
        key_lower = key.lower()
        if any(sensitive in key_lower for sensitive in _SENSITIVE_KEYS):
            sanitized[key] = "***REDACTED***"
        else:
            sanitized[key] = _sanitize_value(value)
    return sanitized


def _add_app_context(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add application-level context to every log event."""
    event_dict.setdefault("service", "multi-agent-github-router")
    return event_dict


def _drop_color_message_key(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """
    Remove the 'color_message' key from uvicorn logs.

    Uvicorn logs duplicate message content in 'color_message' which
    clutters JSON output.
    """
    event_dict.pop("color_message", None)
    return event_dict


def setup_logging(
    log_level: str = "INFO",
    environment: str = "development",
    json_output: Optional[bool] = None,
) -> None:
    """
    Configure structured logging for the application.

    Sets up Structlog with appropriate processors and formatters based
    on the environment. In production, outputs JSON for log aggregation.
    In development, outputs human-readable colored output.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        environment: Application environment (development, staging, production)
        json_output: Force JSON output (auto-detected from environment if None)
    """
    # Determine output format
    if json_output is None:
        json_output = environment in ("production", "staging")

    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Shared processors for both structlog and standard library
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.ExtraAdder(),
        _add_app_context,
        _drop_color_message_key,
        _sanitize_event_dict,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        # Production: JSON output
        renderer = structlog.processors.JSONRenderer()
        stdlib_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
            foreign_pre_chain=shared_processors,
        )
    else:
        # Development: colored console output
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            pad_event_to=40,
        )
        stdlib_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
            foreign_pre_chain=shared_processors,
        )

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicate output
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add single handler with structlog formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(stdlib_formatter)
    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)

    # Configure structlog
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Adjust uvicorn logging to go through structlog
    for uvicorn_logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(uvicorn_logger_name)
        uvicorn_logger.handlers = [console_handler]
        uvicorn_logger.propagate = False

    # Also capture warnings through logging
    logging.captureWarnings(True)


def get_logger(name: Optional[str] = None, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.

    Creates a new structlog BoundLogger, optionally pre-bound with
    context variables that will appear in every log message.

    Args:
        name: Logger name (typically __name__)
        **initial_context: Key-value pairs to bind to every log message

    Returns:
        Configured BoundLogger instance

    Example:
        logger = get_logger(__name__)
        logger.info("started", port=8000)

        # With persistent context:
        logger = get_logger(__name__, workflow_id="wf-123")
        logger.info("processing")  # workflow_id included automatically
    """
    log = structlog.get_logger(name)
    if initial_context:
        log = log.bind(**initial_context)
    return log


def bind_contextvars(**kwargs: Any) -> None:
    """
    Bind context variables that will be included in all subsequent log messages.

    This uses structlog's contextvars integration, which means the bound
    values will be available to ALL loggers in the current async context
    (or thread).

    Useful for request-scoped context like request_id, workflow_id, etc.

    Args:
        **kwargs: Key-value pairs to bind

    Example:
        bind_contextvars(request_id="req-abc123", workflow_id="wf-456")
        logger.info("processing")  # request_id and workflow_id included
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_contextvars(*keys: str) -> None:
    """
    Remove context variables from the current context.

    Args:
        *keys: Names of context variables to remove
    """
    structlog.contextvars.unbind_contextvars(*keys)


def clear_contextvars() -> None:
    """Clear all context variables from the current context."""
    structlog.contextvars.clear_contextvars()


def generate_request_id() -> str:
    """
    Generate a unique request ID for tracking.

    Returns:
        A short unique identifier string
    """
    return f"req-{uuid.uuid4().hex[:12]}"
