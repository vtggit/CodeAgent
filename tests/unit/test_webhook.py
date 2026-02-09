"""
Unit tests for webhook utilities.
"""

import hmac
import hashlib
import pytest
from src.utils.webhook import validate_github_signature, extract_issue_data


class TestValidateGitHubSignature:
    """Tests for GitHub webhook signature validation."""

    def test_valid_signature(self):
        """Test that valid signature passes validation."""
        payload = b'{"action": "opened", "issue": {"number": 42}}'
        secret = "test-secret"

        # Generate valid signature
        mac = hmac.new(secret.encode(), msg=payload, digestmod=hashlib.sha256)
        signature = f"sha256={mac.hexdigest()}"

        assert validate_github_signature(payload, signature, secret) is True

    def test_invalid_signature(self):
        """Test that invalid signature fails validation."""
        payload = b'{"action": "opened"}'
        secret = "test-secret"
        signature = "sha256=invalid_signature_here"

        assert validate_github_signature(payload, signature, secret) is False

    def test_wrong_secret(self):
        """Test that wrong secret fails validation."""
        payload = b'{"action": "opened"}'
        secret = "correct-secret"
        wrong_secret = "wrong-secret"

        # Generate signature with correct secret
        mac = hmac.new(secret.encode(), msg=payload, digestmod=hashlib.sha256)
        signature = f"sha256={mac.hexdigest()}"

        # Validate with wrong secret
        assert validate_github_signature(payload, signature, wrong_secret) is False

    def test_missing_signature(self):
        """Test that missing signature fails validation."""
        payload = b'{"action": "opened"}'
        secret = "test-secret"

        assert validate_github_signature(payload, "", secret) is False
        assert validate_github_signature(payload, None, secret) is False

    def test_invalid_signature_format(self):
        """Test that invalid signature format fails validation."""
        payload = b'{"action": "opened"}'
        secret = "test-secret"

        # Missing "sha256=" prefix
        assert validate_github_signature(payload, "invalid_format", secret) is False

    def test_tampered_payload(self):
        """Test that tampered payload fails validation."""
        original_payload = b'{"action": "opened", "issue": {"number": 42}}'
        tampered_payload = b'{"action": "opened", "issue": {"number": 999}}'
        secret = "test-secret"

        # Generate signature for original payload
        mac = hmac.new(secret.encode(), msg=original_payload, digestmod=hashlib.sha256)
        signature = f"sha256={mac.hexdigest()}"

        # Validate with tampered payload
        assert validate_github_signature(tampered_payload, signature, secret) is False


class TestExtractIssueData:
    """Tests for extracting issue data from webhook payloads."""

    def test_extract_basic_issue_data(self):
        """Test extracting basic issue information."""
        payload = {
            "action": "opened",
            "issue": {
                "number": 42,
                "title": "Test Issue",
                "body": "This is a test",
                "state": "open",
                "labels": [{"name": "bug"}, {"name": "priority-high"}],
                "html_url": "https://github.com/owner/repo/issues/42",
            },
            "repository": {"full_name": "owner/repo"},
            "sender": {"login": "test-user"},
        }

        result = extract_issue_data(payload)

        assert result is not None
        assert result["action"] == "opened"
        assert result["issue_number"] == 42
        assert result["issue_title"] == "Test Issue"
        assert result["issue_body"] == "This is a test"
        assert result["issue_state"] == "open"
        assert result["issue_labels"] == ["bug", "priority-high"]
        assert result["issue_url"] == "https://github.com/owner/repo/issues/42"
        assert result["repository"] == "owner/repo"
        assert result["sender"] == "test-user"

    def test_extract_issue_with_no_labels(self):
        """Test extracting issue data when no labels are present."""
        payload = {
            "action": "opened",
            "issue": {
                "number": 1,
                "title": "No Labels",
                "labels": [],
            },
        }

        result = extract_issue_data(payload)

        assert result is not None
        assert result["issue_labels"] == []

    def test_non_issue_payload(self):
        """Test that non-issue payloads return None."""
        payload = {"action": "created", "comment": {"body": "test comment"}}

        result = extract_issue_data(payload)

        assert result is None

    def test_missing_optional_fields(self):
        """Test extraction with missing optional fields."""
        payload = {
            "issue": {
                "number": 1,
                # Missing title, body, etc.
            }
        }

        result = extract_issue_data(payload)

        assert result is not None
        assert result["issue_number"] == 1
        assert result["issue_title"] is None
        assert result["action"] is None
