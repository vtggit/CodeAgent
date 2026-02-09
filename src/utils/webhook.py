"""
Webhook utilities for GitHub webhook processing.

This module provides functions for validating GitHub webhook signatures
and extracting relevant data from webhook payloads.
"""

import hmac
import hashlib
from typing import Optional


def validate_github_signature(
    payload_body: bytes,
    signature_header: str,
    secret: str,
) -> bool:
    """
    Validate GitHub webhook signature using HMAC SHA-256.

    GitHub sends webhooks with an X-Hub-Signature-256 header containing
    an HMAC signature of the payload. This function verifies that signature.

    Args:
        payload_body: Raw webhook payload body as bytes
        signature_header: Value of X-Hub-Signature-256 header
        secret: Webhook secret configured in GitHub

    Returns:
        True if signature is valid, False otherwise

    Example:
        >>> payload = b'{"action": "opened"}'
        >>> signature = "sha256=abc123..."
        >>> secret = "my-webhook-secret"
        >>> validate_github_signature(payload, signature, secret)
        True
    """
    if not signature_header:
        return False

    # GitHub signature format: "sha256=<hex_digest>"
    if not signature_header.startswith("sha256="):
        return False

    # Extract the hex digest from the header
    expected_signature = signature_header.split("=", 1)[1]

    # Compute HMAC signature of the payload
    mac = hmac.new(
        secret.encode("utf-8"),
        msg=payload_body,
        digestmod=hashlib.sha256,
    )
    computed_signature = mac.hexdigest()

    # Use timing-safe comparison to prevent timing attacks
    return hmac.compare_digest(computed_signature, expected_signature)


def extract_issue_data(payload: dict) -> Optional[dict]:
    """
    Extract relevant issue data from GitHub webhook payload.

    Args:
        payload: GitHub webhook payload as dictionary

    Returns:
        Dictionary with extracted data or None if not an issue event

    Example:
        >>> payload = {"action": "opened", "issue": {"number": 42}}
        >>> extract_issue_data(payload)
        {'action': 'opened', 'issue_number': 42, ...}
    """
    if "issue" not in payload:
        return None

    issue = payload["issue"]

    return {
        "action": payload.get("action"),
        "issue_number": issue.get("number"),
        "issue_title": issue.get("title"),
        "issue_body": issue.get("body"),
        "issue_state": issue.get("state"),
        "issue_labels": [label.get("name") for label in issue.get("labels", [])],
        "issue_url": issue.get("html_url"),
        "repository": payload.get("repository", {}).get("full_name"),
        "sender": payload.get("sender", {}).get("login"),
    }
