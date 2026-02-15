"""
Input validation and sanitization utilities.

Provides comprehensive input validation to prevent:
  - SQL injection (parameterized queries are primary defense;
    this adds pattern detection as defense-in-depth)
  - XSS attacks (HTML tag/script escaping)
  - Denial of service via oversized inputs
  - Injection of control characters

Usage:
    from src.utils.validation import (
        sanitize_text,
        validate_issue_title,
        validate_issue_body,
        validate_label,
        InputTooLongError,
        MaliciousInputError,
    )

    clean_title = validate_issue_title(user_input)
    clean_body = validate_issue_body(user_input)
"""

import html
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum lengths for various inputs
MAX_ISSUE_TITLE_LENGTH = 512
MAX_ISSUE_BODY_LENGTH = 65536  # GitHub's own limit
MAX_LABEL_LENGTH = 256
MAX_LABEL_COUNT = 100
MAX_WEBHOOK_PAYLOAD_BYTES = 5 * 1024 * 1024  # 5 MB

# Characters that should never appear in user-facing text
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Patterns that indicate potential XSS attempts
_XSS_PATTERNS = [
    re.compile(r"<script[\s>]", re.IGNORECASE),
    re.compile(r"javascript\s*:", re.IGNORECASE),
    re.compile(r"on\w+\s*=", re.IGNORECASE),  # onclick=, onerror=, etc.
    re.compile(r"<iframe[\s>]", re.IGNORECASE),
    re.compile(r"<object[\s>]", re.IGNORECASE),
    re.compile(r"<embed[\s>]", re.IGNORECASE),
    re.compile(r"<svg[^>]*\son\w+", re.IGNORECASE),
    re.compile(r"data\s*:\s*text/html", re.IGNORECASE),
    re.compile(r"vbscript\s*:", re.IGNORECASE),
]

# Patterns that look like SQL injection attempts
_SQL_INJECTION_PATTERNS = [
    re.compile(r";\s*(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|EXEC)\s", re.IGNORECASE),
    re.compile(r"'\s*(OR|AND)\s+\d+\s*=\s*\d+", re.IGNORECASE),
    re.compile(r"UNION\s+(ALL\s+)?SELECT", re.IGNORECASE),
    re.compile(r"--\s*$", re.MULTILINE),  # SQL comment at end of line
    re.compile(r"/\*.*?\*/", re.DOTALL),  # SQL block comments used for injection
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ValidationError(ValueError):
    """Base class for input validation errors."""

    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        super().__init__(message)


class InputTooLongError(ValidationError):
    """Raised when input exceeds maximum allowed length."""

    def __init__(self, field: str, length: int, max_length: int):
        self.length = length
        self.max_length = max_length
        super().__init__(
            f"Input for '{field}' is too long ({length} chars, max {max_length})",
            field=field,
        )


class MaliciousInputError(ValidationError):
    """Raised when input contains potentially malicious patterns."""

    def __init__(self, field: str, pattern_type: str):
        self.pattern_type = pattern_type
        super().__init__(
            f"Input for '{field}' contains potentially malicious content "
            f"(detected: {pattern_type})",
            field=field,
        )


# ---------------------------------------------------------------------------
# Core sanitization functions
# ---------------------------------------------------------------------------


def strip_control_characters(text: str) -> str:
    """
    Remove ASCII control characters (except tab, newline, carriage return).

    Args:
        text: Input text.

    Returns:
        Text with control characters removed.
    """
    return _CONTROL_CHAR_RE.sub("", text)


def escape_html(text: str) -> str:
    """
    Escape HTML special characters to prevent XSS.

    Uses Python's html.escape which converts:
      & → &amp;   < → &lt;   > → &gt;   " → &quot;   ' → &#x27;

    Args:
        text: Input text.

    Returns:
        HTML-escaped text.
    """
    return html.escape(text, quote=True)


def detect_xss(text: str) -> Optional[str]:
    """
    Check if text contains potential XSS patterns.

    Args:
        text: Input text to check.

    Returns:
        Description of detected pattern, or None if clean.
    """
    for pattern in _XSS_PATTERNS:
        if pattern.search(text):
            return f"XSS pattern: {pattern.pattern}"
    return None


def detect_sql_injection(text: str) -> Optional[str]:
    """
    Check if text contains potential SQL injection patterns.

    Note: This is defense-in-depth. The primary defense is parameterized
    queries via SQLAlchemy. This catches obvious injection attempts early.

    Args:
        text: Input text to check.

    Returns:
        Description of detected pattern, or None if clean.
    """
    for pattern in _SQL_INJECTION_PATTERNS:
        if pattern.search(text):
            return f"SQL injection pattern: {pattern.pattern}"
    return None


def sanitize_text(
    text: str,
    max_length: Optional[int] = None,
    strip_html: bool = False,
    strip_control: bool = True,
) -> str:
    """
    Apply general sanitization to text input.

    Args:
        text: Input text.
        max_length: If set, truncate text to this length.
        strip_html: If True, escape HTML characters.
        strip_control: If True, remove control characters.

    Returns:
        Sanitized text.
    """
    if strip_control:
        text = strip_control_characters(text)

    if strip_html:
        text = escape_html(text)

    if max_length and len(text) > max_length:
        text = text[:max_length]

    return text


# ---------------------------------------------------------------------------
# Field-specific validators
# ---------------------------------------------------------------------------


def validate_issue_title(
    title: str,
    max_length: int = MAX_ISSUE_TITLE_LENGTH,
    reject_malicious: bool = True,
) -> str:
    """
    Validate and sanitize an issue title.

    Args:
        title: Raw issue title.
        max_length: Maximum allowed length.
        reject_malicious: If True, raise on XSS/SQL injection patterns.

    Returns:
        Sanitized issue title.

    Raises:
        InputTooLongError: If title exceeds max_length.
        MaliciousInputError: If malicious patterns detected.
        ValidationError: If title is empty after sanitization.
    """
    # Strip whitespace and control characters
    title = strip_control_characters(title.strip())

    if not title:
        raise ValidationError("Issue title cannot be empty", field="title")

    if len(title) > max_length:
        raise InputTooLongError("title", len(title), max_length)

    if reject_malicious:
        xss = detect_xss(title)
        if xss:
            raise MaliciousInputError("title", xss)

        sqli = detect_sql_injection(title)
        if sqli:
            raise MaliciousInputError("title", sqli)

    return title


def validate_issue_body(
    body: Optional[str],
    max_length: int = MAX_ISSUE_BODY_LENGTH,
    reject_malicious: bool = True,
) -> Optional[str]:
    """
    Validate and sanitize an issue body.

    Args:
        body: Raw issue body (can be None).
        max_length: Maximum allowed length.
        reject_malicious: If True, raise on XSS/SQL injection patterns.

    Returns:
        Sanitized issue body, or None if input was None/empty.

    Raises:
        InputTooLongError: If body exceeds max_length.
        MaliciousInputError: If malicious patterns detected.
    """
    if body is None:
        return None

    body = strip_control_characters(body.strip())

    if not body:
        return None

    if len(body) > max_length:
        raise InputTooLongError("body", len(body), max_length)

    if reject_malicious:
        xss = detect_xss(body)
        if xss:
            raise MaliciousInputError("body", xss)

        sqli = detect_sql_injection(body)
        if sqli:
            raise MaliciousInputError("body", sqli)

    return body


def validate_label(
    label: str,
    max_length: int = MAX_LABEL_LENGTH,
) -> str:
    """
    Validate and sanitize a single label name.

    Args:
        label: Raw label name.
        max_length: Maximum allowed length.

    Returns:
        Sanitized label name.

    Raises:
        InputTooLongError: If label exceeds max_length.
        ValidationError: If label is empty.
    """
    label = strip_control_characters(label.strip())

    if not label:
        raise ValidationError("Label cannot be empty", field="label")

    if len(label) > max_length:
        raise InputTooLongError("label", len(label), max_length)

    return label


def validate_labels(
    labels: list[str],
    max_count: int = MAX_LABEL_COUNT,
    max_label_length: int = MAX_LABEL_LENGTH,
) -> list[str]:
    """
    Validate and sanitize a list of labels.

    Args:
        labels: Raw label names.
        max_count: Maximum number of labels.
        max_label_length: Maximum length per label.

    Returns:
        List of sanitized label names.

    Raises:
        ValidationError: If too many labels.
        InputTooLongError: If any label exceeds max length.
    """
    if len(labels) > max_count:
        raise ValidationError(
            f"Too many labels ({len(labels)}, max {max_count})",
            field="labels",
        )

    return [validate_label(label, max_label_length) for label in labels]


def validate_webhook_payload_size(payload_bytes: bytes) -> None:
    """
    Validate that a webhook payload doesn't exceed size limits.

    Args:
        payload_bytes: Raw payload bytes.

    Raises:
        InputTooLongError: If payload exceeds maximum size.
    """
    if len(payload_bytes) > MAX_WEBHOOK_PAYLOAD_BYTES:
        raise InputTooLongError(
            "webhook_payload",
            len(payload_bytes),
            MAX_WEBHOOK_PAYLOAD_BYTES,
        )
