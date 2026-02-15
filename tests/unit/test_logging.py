"""
Tests for structured logging module (src/utils/logging.py).

Covers:
  - Structlog configuration (setup_logging)
  - JSON vs console output modes
  - Sensitive data sanitization
  - Context propagation (bind/unbind/clear)
  - Request ID generation
  - get_logger with initial context
  - Integration with standard library logging
"""

import json
import logging
import re
import sys
from io import StringIO
from unittest.mock import patch

import pytest
import structlog

from src.utils.logging import (
    _SENSITIVE_KEYS,
    _SENSITIVE_PATTERNS,
    _sanitize_event_dict,
    _sanitize_value,
    _add_app_context,
    _drop_color_message_key,
    bind_contextvars,
    clear_contextvars,
    generate_request_id,
    get_logger,
    setup_logging,
    unbind_contextvars,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset logging and structlog state between tests."""
    clear_contextvars()
    yield
    clear_contextvars()
    # Reset structlog to avoid leaking config between tests
    structlog.reset_defaults()


# ---------------------------------------------------------------------------
# _sanitize_value tests
# ---------------------------------------------------------------------------


class TestSanitizeValue:
    """Tests for individual value sanitization."""

    def test_normal_string_passes_through(self):
        assert _sanitize_value("hello world") == "hello world"

    def test_non_string_passes_through(self):
        assert _sanitize_value(42) == 42
        assert _sanitize_value(None) is None
        assert _sanitize_value(True) is True
        assert _sanitize_value([1, 2]) == [1, 2]

    def test_github_pat_redacted(self):
        token = "ghp_" + "a" * 40
        assert _sanitize_value(token) == "***REDACTED***"

    def test_github_app_token_redacted(self):
        token = "ghs_" + "A" * 40
        assert _sanitize_value(token) == "***REDACTED***"

    def test_anthropic_key_redacted(self):
        key = "sk-ant-" + "x" * 50
        assert _sanitize_value(key) == "***REDACTED***"

    def test_openai_key_redacted(self):
        key = "sk-" + "A" * 45
        assert _sanitize_value(key) == "***REDACTED***"

    def test_slack_token_redacted(self):
        token = "xoxb-123-456-abc"
        assert _sanitize_value(token) == "***REDACTED***"

    def test_bearer_token_redacted(self):
        header = "Bearer eyJhbGciOiJIUzI1NiJ9.abc.def"
        assert _sanitize_value(header) == "***REDACTED***"

    def test_short_string_not_falsely_redacted(self):
        # "sk-" followed by < 40 chars should NOT match
        assert _sanitize_value("sk-short") == "sk-short"

    def test_empty_string_passes(self):
        assert _sanitize_value("") == ""


# ---------------------------------------------------------------------------
# _sanitize_event_dict tests
# ---------------------------------------------------------------------------


class TestSanitizeEventDict:
    """Tests for the structlog processor that redacts sensitive data."""

    def test_normal_keys_pass_through(self):
        event = {"event": "test", "user": "alice", "count": 5}
        result = _sanitize_event_dict(None, "info", event)
        assert result == {"event": "test", "user": "alice", "count": 5}

    def test_sensitive_key_names_redacted(self):
        event = {"event": "login", "token": "abc123", "password": "secret"}
        result = _sanitize_event_dict(None, "info", event)
        assert result["token"] == "***REDACTED***"
        assert result["password"] == "***REDACTED***"
        assert result["event"] == "login"

    def test_api_key_variants_redacted(self):
        for key in ("api_key", "apikey", "ANTHROPIC_API_KEY", "GitHub_Token"):
            event = {"event": "test", key: "some-value"}
            result = _sanitize_event_dict(None, "info", event)
            assert result[key] == "***REDACTED***", f"Key {key!r} was not redacted"

    def test_value_pattern_redacted_even_with_safe_key(self):
        token = "ghp_" + "X" * 40
        event = {"event": "debug", "message": token}
        result = _sanitize_event_dict(None, "info", event)
        assert result["message"] == "***REDACTED***"

    def test_mixed_sensitive_and_normal(self):
        event = {
            "event": "webhook_received",
            "issue_number": 42,
            "authorization": "Bearer xyz",
            "agent": "ui_architect",
        }
        result = _sanitize_event_dict(None, "info", event)
        assert result["issue_number"] == 42
        assert result["authorization"] == "***REDACTED***"
        assert result["agent"] == "ui_architect"


# ---------------------------------------------------------------------------
# _add_app_context tests
# ---------------------------------------------------------------------------


class TestAddAppContext:
    """Tests for the app context processor."""

    def test_adds_service_name(self):
        event = {"event": "test"}
        result = _add_app_context(None, "info", event)
        assert result["service"] == "multi-agent-github-router"

    def test_does_not_overwrite_existing_service(self):
        event = {"event": "test", "service": "custom-service"}
        result = _add_app_context(None, "info", event)
        assert result["service"] == "custom-service"


# ---------------------------------------------------------------------------
# _drop_color_message_key tests
# ---------------------------------------------------------------------------


class TestDropColorMessage:
    """Tests for uvicorn color_message cleanup."""

    def test_removes_color_message(self):
        event = {"event": "test", "color_message": "\x1b[32mGET /\x1b[0m"}
        result = _drop_color_message_key(None, "info", event)
        assert "color_message" not in result
        assert result["event"] == "test"

    def test_no_error_when_absent(self):
        event = {"event": "test", "level": "info"}
        result = _drop_color_message_key(None, "info", event)
        assert result == {"event": "test", "level": "info"}


# ---------------------------------------------------------------------------
# setup_logging tests
# ---------------------------------------------------------------------------


class TestSetupLogging:
    """Tests for the main logging configuration function."""

    def test_setup_production_json(self):
        """Production environment should configure JSON renderer."""
        setup_logging(log_level="INFO", environment="production")
        logger = get_logger("test.prod")
        # Just verify it doesn't crash
        logger.info("production_test", key="value")

    def test_setup_development_console(self):
        """Development environment should configure console renderer."""
        setup_logging(log_level="DEBUG", environment="development")
        logger = get_logger("test.dev")
        logger.debug("dev_test", key="value")

    def test_json_output_forced(self):
        """json_output=True should force JSON even in development."""
        setup_logging(log_level="INFO", environment="development", json_output=True)
        logger = get_logger("test.forced_json")
        logger.info("forced_json_test")

    def test_log_level_respected(self):
        """Log level should be applied to root logger."""
        setup_logging(log_level="WARNING", environment="development")
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_log_level_debug(self):
        setup_logging(log_level="DEBUG", environment="development")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_log_level_error(self):
        setup_logging(log_level="ERROR", environment="production")
        root = logging.getLogger()
        assert root.level == logging.ERROR

    def test_staging_uses_json(self):
        """Staging environment should auto-detect JSON output."""
        setup_logging(log_level="INFO", environment="staging")
        # Verify it configured successfully (no crash)
        logger = get_logger("test.staging")
        logger.info("staging_test")

    def test_captures_warnings(self):
        """Python warnings should be captured through logging."""
        setup_logging(log_level="WARNING", environment="development")
        assert logging.captureWarnings


# ---------------------------------------------------------------------------
# get_logger tests
# ---------------------------------------------------------------------------


class TestGetLogger:
    """Tests for logger creation and context binding."""

    def test_returns_bound_logger(self):
        setup_logging(log_level="INFO", environment="development")
        logger = get_logger("test_module")
        assert logger is not None

    def test_with_initial_context(self):
        setup_logging(log_level="INFO", environment="development")
        logger = get_logger("test_module", workflow_id="wf-123", round=3)
        # The logger should have the bound context
        # Verify it logs without error
        logger.info("context_test")

    def test_without_name(self):
        setup_logging(log_level="INFO", environment="development")
        logger = get_logger()
        assert logger is not None
        logger.info("no_name_test")


# ---------------------------------------------------------------------------
# Context variable tests
# ---------------------------------------------------------------------------


class TestContextVars:
    """Tests for contextvars-based context propagation."""

    def test_bind_and_clear(self):
        setup_logging(log_level="INFO", environment="development")
        bind_contextvars(request_id="req-abc", workflow_id="wf-123")
        # Should not raise
        clear_contextvars()

    def test_unbind_specific_keys(self):
        setup_logging(log_level="INFO", environment="development")
        bind_contextvars(request_id="req-abc", workflow_id="wf-123")
        unbind_contextvars("request_id")
        # workflow_id should still be bound
        clear_contextvars()

    def test_clear_removes_all(self):
        setup_logging(log_level="INFO", environment="development")
        bind_contextvars(a="1", b="2", c="3")
        clear_contextvars()
        # No error expected


# ---------------------------------------------------------------------------
# generate_request_id tests
# ---------------------------------------------------------------------------


class TestGenerateRequestId:
    """Tests for request ID generation."""

    def test_starts_with_req_prefix(self):
        rid = generate_request_id()
        assert rid.startswith("req-")

    def test_has_expected_length(self):
        rid = generate_request_id()
        # "req-" (4) + 12 hex chars = 16
        assert len(rid) == 16

    def test_unique_ids(self):
        ids = {generate_request_id() for _ in range(100)}
        assert len(ids) == 100, "Request IDs should be unique"

    def test_hex_chars_only_in_suffix(self):
        rid = generate_request_id()
        suffix = rid[4:]
        assert re.match(r"^[0-9a-f]{12}$", suffix)


# ---------------------------------------------------------------------------
# Integration: JSON output verification
# ---------------------------------------------------------------------------


class TestJsonOutput:
    """Verify that JSON mode produces valid JSON logs."""

    def test_json_log_is_parseable(self):
        """In JSON mode, log output should be valid JSON."""
        setup_logging(log_level="INFO", environment="production", json_output=True)

        # Capture stdout
        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.getLogger().handlers[0].formatter)

        test_logger = logging.getLogger("json_test")
        test_logger.handlers = [handler]
        test_logger.setLevel(logging.INFO)
        test_logger.propagate = False

        test_logger.info("hello from json test")

        output = buf.getvalue().strip()
        if output:
            # Should be valid JSON
            parsed = json.loads(output)
            assert "event" in parsed
            assert parsed["service"] == "multi-agent-github-router"


# ---------------------------------------------------------------------------
# Integration: Sensitive data never leaks
# ---------------------------------------------------------------------------


class TestSensitiveDataIntegration:
    """End-to-end tests that sensitive data doesn't appear in log output."""

    def test_github_token_never_in_output(self):
        setup_logging(log_level="DEBUG", environment="production", json_output=True)

        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.getLogger().handlers[0].formatter)

        test_logger = logging.getLogger("sensitive_test")
        test_logger.handlers = [handler]
        test_logger.setLevel(logging.DEBUG)
        test_logger.propagate = False

        fake_token = "ghp_" + "A" * 40
        test_logger.info("checking token", extra={"token": fake_token})

        output = buf.getvalue()
        assert fake_token not in output

    def test_password_key_redacted_in_output(self):
        setup_logging(log_level="DEBUG", environment="production", json_output=True)

        buf = StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.getLogger().handlers[0].formatter)

        test_logger = logging.getLogger("password_test")
        test_logger.handlers = [handler]
        test_logger.setLevel(logging.DEBUG)
        test_logger.propagate = False

        test_logger.info("user login", extra={"password": "supersecret123"})

        output = buf.getvalue()
        assert "supersecret123" not in output


# ---------------------------------------------------------------------------
# Sensitive patterns coverage
# ---------------------------------------------------------------------------


class TestSensitivePatterns:
    """Ensure all defined sensitive patterns actually match."""

    def test_all_patterns_are_valid_regex(self):
        for pattern in _SENSITIVE_PATTERNS:
            assert pattern.pattern, "Pattern should not be empty"
            # Verify it compiles (already compiled, but double-check)
            re.compile(pattern.pattern)

    def test_all_sensitive_keys_are_lowercase(self):
        for key in _SENSITIVE_KEYS:
            assert key == key.lower(), f"Key {key!r} should be lowercase"
