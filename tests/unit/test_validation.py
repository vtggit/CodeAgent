"""
Tests for input validation and sanitization (src/utils/validation.py).

Covers:
  - SQL injection detection
  - XSS pattern detection
  - Control character stripping
  - HTML escaping
  - Issue title validation (length, empty, malicious)
  - Issue body validation (length, None, malicious)
  - Label validation (length, empty, count)
  - Webhook payload size limits
  - Unicode and special character handling
"""

import pytest

from src.utils.validation import (
    InputTooLongError,
    MaliciousInputError,
    ValidationError,
    detect_sql_injection,
    detect_xss,
    escape_html,
    sanitize_text,
    strip_control_characters,
    validate_issue_body,
    validate_issue_title,
    validate_label,
    validate_labels,
    validate_webhook_payload_size,
    MAX_ISSUE_TITLE_LENGTH,
    MAX_ISSUE_BODY_LENGTH,
    MAX_LABEL_LENGTH,
    MAX_LABEL_COUNT,
    MAX_WEBHOOK_PAYLOAD_BYTES,
)


# ---------------------------------------------------------------------------
# strip_control_characters tests
# ---------------------------------------------------------------------------


class TestStripControlCharacters:
    """Tests for control character removal."""

    def test_normal_text_unchanged(self):
        assert strip_control_characters("hello world") == "hello world"

    def test_preserves_tab(self):
        assert strip_control_characters("hello\tworld") == "hello\tworld"

    def test_preserves_newline(self):
        assert strip_control_characters("hello\nworld") == "hello\nworld"

    def test_preserves_carriage_return(self):
        assert strip_control_characters("hello\r\nworld") == "hello\r\nworld"

    def test_removes_null_byte(self):
        assert strip_control_characters("hello\x00world") == "helloworld"

    def test_removes_bell(self):
        assert strip_control_characters("hello\x07world") == "helloworld"

    def test_removes_escape(self):
        assert strip_control_characters("hello\x1bworld") == "helloworld"

    def test_removes_delete(self):
        assert strip_control_characters("hello\x7fworld") == "helloworld"

    def test_empty_string(self):
        assert strip_control_characters("") == ""

    def test_unicode_preserved(self):
        assert strip_control_characters("Hello 日本語 🎉") == "Hello 日本語 🎉"


# ---------------------------------------------------------------------------
# escape_html tests
# ---------------------------------------------------------------------------


class TestEscapeHtml:
    """Tests for HTML character escaping."""

    def test_normal_text_unchanged(self):
        assert escape_html("hello world") == "hello world"

    def test_escapes_lt_gt(self):
        assert escape_html("<script>") == "&lt;script&gt;"

    def test_escapes_ampersand(self):
        assert escape_html("foo & bar") == "foo &amp; bar"

    def test_escapes_quotes(self):
        result = escape_html('say "hello"')
        assert "&quot;" in result

    def test_escapes_single_quote(self):
        result = escape_html("it's")
        assert "&#x27;" in result


# ---------------------------------------------------------------------------
# detect_xss tests
# ---------------------------------------------------------------------------


class TestDetectXss:
    """Tests for XSS pattern detection."""

    def test_clean_text(self):
        assert detect_xss("Normal issue title") is None

    def test_script_tag(self):
        assert detect_xss("<script>alert('xss')</script>") is not None

    def test_script_tag_uppercase(self):
        assert detect_xss("<SCRIPT>alert(1)</SCRIPT>") is not None

    def test_javascript_protocol(self):
        assert detect_xss("javascript:alert(1)") is not None

    def test_onclick_handler(self):
        assert detect_xss('<div onclick="alert(1)">') is not None

    def test_onerror_handler(self):
        assert detect_xss('<img onerror="alert(1)">') is not None

    def test_iframe_tag(self):
        assert detect_xss('<iframe src="evil.com">') is not None

    def test_object_tag(self):
        assert detect_xss('<object data="evil.swf">') is not None

    def test_embed_tag(self):
        assert detect_xss('<embed src="evil">') is not None

    def test_svg_with_handler(self):
        assert detect_xss('<svg onload="alert(1)">') is not None

    def test_data_uri(self):
        assert detect_xss('data:text/html,<script>alert(1)</script>') is not None

    def test_vbscript(self):
        assert detect_xss("vbscript:MsgBox") is not None

    def test_markdown_code_blocks_ok(self):
        """Code blocks in markdown should not trigger XSS detection."""
        # Backtick code blocks are fine
        assert detect_xss("Use `onclick` to handle events") is None

    def test_normal_html_entities_ok(self):
        assert detect_xss("Use &lt; for less than") is None


# ---------------------------------------------------------------------------
# detect_sql_injection tests
# ---------------------------------------------------------------------------


class TestDetectSqlInjection:
    """Tests for SQL injection pattern detection."""

    def test_clean_text(self):
        assert detect_sql_injection("Fix the login page bug") is None

    def test_drop_table(self):
        assert detect_sql_injection("; DROP TABLE users") is not None

    def test_or_1_equals_1(self):
        assert detect_sql_injection("' OR 1=1") is not None

    def test_union_select(self):
        assert detect_sql_injection("UNION SELECT * FROM passwords") is not None

    def test_union_all_select(self):
        assert detect_sql_injection("UNION ALL SELECT username FROM users") is not None

    def test_delete_statement(self):
        assert detect_sql_injection("; DELETE FROM users WHERE 1=1") is not None

    def test_sql_comment(self):
        assert detect_sql_injection("admin' --") is not None

    def test_block_comment(self):
        assert detect_sql_injection("admin' /* comment */ OR 1=1") is not None

    def test_normal_dashes_ok(self):
        """Normal text with dashes (not SQL comments) should be fine."""
        assert detect_sql_injection("Fix the login-page bug") is None

    def test_normal_single_quotes_ok(self):
        """Normal English possessives should be fine."""
        assert detect_sql_injection("Update the user's profile page") is None


# ---------------------------------------------------------------------------
# sanitize_text tests
# ---------------------------------------------------------------------------


class TestSanitizeText:
    """Tests for general text sanitization."""

    def test_basic_sanitize(self):
        assert sanitize_text("hello world") == "hello world"

    def test_truncate(self):
        result = sanitize_text("abcdefgh", max_length=5)
        assert len(result) == 5
        assert result == "abcde"

    def test_strip_html(self):
        result = sanitize_text("<b>bold</b>", strip_html=True)
        assert "<b>" not in result

    def test_strip_control(self):
        result = sanitize_text("hello\x00world")
        assert "\x00" not in result

    def test_no_strip_control(self):
        result = sanitize_text("hello\x00world", strip_control=False)
        assert "\x00" in result


# ---------------------------------------------------------------------------
# validate_issue_title tests
# ---------------------------------------------------------------------------


class TestValidateIssueTitle:
    """Tests for issue title validation."""

    def test_valid_title(self):
        assert validate_issue_title("Fix the login bug") == "Fix the login bug"

    def test_strips_whitespace(self):
        assert validate_issue_title("  Fix bug  ") == "Fix bug"

    def test_strips_control_chars(self):
        assert validate_issue_title("Fix\x00 bug") == "Fix bug"

    def test_empty_title_raises(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_issue_title("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_issue_title("   ")

    def test_too_long_raises(self):
        with pytest.raises(InputTooLongError):
            validate_issue_title("A" * (MAX_ISSUE_TITLE_LENGTH + 1))

    def test_xss_in_title_raises(self):
        with pytest.raises(MaliciousInputError, match="XSS"):
            validate_issue_title("<script>alert(1)</script>")

    def test_sql_injection_in_title_raises(self):
        with pytest.raises(MaliciousInputError, match="SQL"):
            validate_issue_title("'; DROP TABLE issues --")

    def test_xss_allowed_when_not_rejecting(self):
        result = validate_issue_title(
            "<script>alert(1)</script>", reject_malicious=False
        )
        assert "<script>" in result

    def test_unicode_title(self):
        title = "日本語のイシュータイトル 🐛"
        assert validate_issue_title(title) == title

    def test_max_length_exactly(self):
        title = "A" * MAX_ISSUE_TITLE_LENGTH
        assert validate_issue_title(title) == title


# ---------------------------------------------------------------------------
# validate_issue_body tests
# ---------------------------------------------------------------------------


class TestValidateIssueBody:
    """Tests for issue body validation."""

    def test_valid_body(self):
        assert validate_issue_body("This is a bug description") == "This is a bug description"

    def test_none_returns_none(self):
        assert validate_issue_body(None) is None

    def test_empty_returns_none(self):
        assert validate_issue_body("") is None

    def test_whitespace_only_returns_none(self):
        assert validate_issue_body("   ") is None

    def test_strips_control_chars(self):
        assert validate_issue_body("Bug\x00 report") == "Bug report"

    def test_too_long_raises(self):
        with pytest.raises(InputTooLongError):
            validate_issue_body("A" * (MAX_ISSUE_BODY_LENGTH + 1))

    def test_xss_in_body_raises(self):
        with pytest.raises(MaliciousInputError, match="XSS"):
            validate_issue_body("<script>steal(cookie)</script>")

    def test_sql_injection_in_body_raises(self):
        with pytest.raises(MaliciousInputError, match="SQL"):
            validate_issue_body("UNION SELECT password FROM users")

    def test_markdown_code_blocks_ok(self):
        """Markdown with code snippets should not trigger false positives."""
        body = "Here is an example:\n```python\nprint('hello')\n```"
        assert validate_issue_body(body) == body


# ---------------------------------------------------------------------------
# validate_label tests
# ---------------------------------------------------------------------------


class TestValidateLabel:
    """Tests for single label validation."""

    def test_valid_label(self):
        assert validate_label("bug") == "bug"

    def test_strips_whitespace(self):
        assert validate_label("  enhancement  ") == "enhancement"

    def test_empty_raises(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_label("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_label("   ")

    def test_too_long_raises(self):
        with pytest.raises(InputTooLongError):
            validate_label("A" * (MAX_LABEL_LENGTH + 1))


class TestValidateLabels:
    """Tests for label list validation."""

    def test_valid_labels(self):
        result = validate_labels(["bug", "high-priority"])
        assert result == ["bug", "high-priority"]

    def test_empty_list(self):
        assert validate_labels([]) == []

    def test_too_many_labels(self):
        with pytest.raises(ValidationError, match="Too many labels"):
            validate_labels(["label"] * (MAX_LABEL_COUNT + 1))


# ---------------------------------------------------------------------------
# validate_webhook_payload_size tests
# ---------------------------------------------------------------------------


class TestValidateWebhookPayloadSize:
    """Tests for webhook payload size validation."""

    def test_small_payload_ok(self):
        validate_webhook_payload_size(b"small payload")

    def test_exactly_at_limit(self):
        validate_webhook_payload_size(b"x" * MAX_WEBHOOK_PAYLOAD_BYTES)

    def test_over_limit_raises(self):
        with pytest.raises(InputTooLongError):
            validate_webhook_payload_size(b"x" * (MAX_WEBHOOK_PAYLOAD_BYTES + 1))


# ---------------------------------------------------------------------------
# Exception tests
# ---------------------------------------------------------------------------


class TestExceptions:
    """Tests for custom exception classes."""

    def test_validation_error_with_field(self):
        err = ValidationError("bad input", field="title")
        assert err.field == "title"
        assert "bad input" in str(err)

    def test_input_too_long_error(self):
        err = InputTooLongError("body", 70000, 65536)
        assert err.field == "body"
        assert err.length == 70000
        assert err.max_length == 65536
        assert "70000" in str(err)
        assert "65536" in str(err)

    def test_malicious_input_error(self):
        err = MaliciousInputError("title", "XSS pattern")
        assert err.field == "title"
        assert err.pattern_type == "XSS pattern"
        assert "malicious" in str(err).lower()

    def test_validation_error_is_value_error(self):
        """ValidationError should be a subclass of ValueError."""
        assert issubclass(ValidationError, ValueError)
        assert issubclass(InputTooLongError, ValidationError)
        assert issubclass(MaliciousInputError, ValidationError)
