"""
Unit tests for the GitHubClient wrapper.

Tests cover:
- Client initialization
- Issue retrieval
- Comment posting
- Label listing
- Rate limit handling
- Retry logic
- Error handling
"""

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.integrations.github_client import (
    GitHubClient,
    IssueData,
    CommentData,
    LabelData,
    RateLimitInfo,
    ClientStats,
    get_github_client,
    close_github_client,
)


# ========================
# Fixtures
# ========================


@pytest.fixture
def mock_github():
    """Create a mock Github instance."""
    with patch("src.integrations.github_client.Github") as MockGithub:
        mock_instance = MagicMock()
        MockGithub.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_auth():
    """Create a mock Auth.Token."""
    with patch("src.integrations.github_client.Auth") as MockAuth:
        yield MockAuth


@pytest.fixture
def client(mock_github, mock_auth):
    """Create a GitHubClient with mocked dependencies."""
    return GitHubClient(token="test-token-12345", max_retries=2, retry_delay=0.01)


@pytest.fixture
def mock_repo():
    """Create a mock repository."""
    repo = MagicMock()
    repo.full_name = "test-owner/test-repo"
    return repo


@pytest.fixture
def mock_issue():
    """Create a mock issue."""
    issue = MagicMock()
    issue.number = 42
    issue.title = "Test Issue"
    issue.body = "Test body"
    issue.state = "open"
    issue.html_url = "https://github.com/test-owner/test-repo/issues/42"
    issue.comments = 5
    issue.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    issue.updated_at = datetime(2026, 1, 2, tzinfo=timezone.utc)

    # Mock user
    issue.user = MagicMock()
    issue.user.login = "test-user"

    # Mock labels
    label1 = MagicMock()
    label1.name = "bug"
    label1.color = "d73a4a"
    label1.description = "Something isn't working"
    label2 = MagicMock()
    label2.name = "priority-1"
    label2.color = "ededed"
    label2.description = ""
    issue.labels = [label1, label2]

    return issue


@pytest.fixture
def mock_comment():
    """Create a mock comment."""
    comment = MagicMock()
    comment.id = 123
    comment.body = "Test comment body"
    comment.user = MagicMock()
    comment.user.login = "commenter"
    comment.created_at = datetime(2026, 1, 3, tzinfo=timezone.utc)
    comment.updated_at = datetime(2026, 1, 3, tzinfo=timezone.utc)
    return comment


@pytest.fixture
def mock_rate_limit():
    """Create a mock rate limit response."""
    rate_limit = MagicMock()
    rate_limit.core.remaining = 4999
    rate_limit.core.limit = 5000
    rate_limit.core.reset = datetime(2026, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
    return rate_limit


# ========================
# Initialization Tests
# ========================


class TestGitHubClientInit:
    """Tests for GitHubClient initialization."""

    def test_init_with_token(self, mock_github, mock_auth):
        """Client initializes with provided token."""
        client = GitHubClient(token="test-token")
        assert client is not None
        mock_auth.Token.assert_called_once_with("test-token")

    def test_init_without_token_raises(self, mock_github, mock_auth):
        """Client raises ValueError when no token available."""
        with patch(
            "src.integrations.github_client.GitHubClient._get_token_from_settings",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="GitHub token is required"):
                GitHubClient(token=None)

    def test_init_custom_retries(self, mock_github, mock_auth):
        """Client accepts custom retry configuration."""
        client = GitHubClient(
            token="test-token", max_retries=5, retry_delay=1.0
        )
        assert client._max_retries == 5
        assert client._retry_delay == 1.0

    def test_init_stats(self, client):
        """Client initializes with clean stats."""
        assert client.stats.api_calls == 0
        assert client.stats.comments_posted == 0
        assert client.stats.errors == 0
        assert client.stats.retries == 0


# ========================
# Issue Tests
# ========================


class TestGetIssue:
    """Tests for get_issue method."""

    def test_get_issue_success(
        self, client, mock_github, mock_repo, mock_issue, mock_rate_limit
    ):
        """Successfully retrieves issue data."""
        mock_github.get_repo.return_value = mock_repo
        mock_repo.get_issue.return_value = mock_issue
        mock_github.get_rate_limit.return_value = mock_rate_limit

        result = client.get_issue("test-owner/test-repo", 42)

        assert isinstance(result, IssueData)
        assert result.number == 42
        assert result.title == "Test Issue"
        assert result.body == "Test body"
        assert result.state == "open"
        assert result.user == "test-user"
        assert result.labels == ["bug", "priority-1"]
        assert result.comments_count == 5
        mock_repo.get_issue.assert_called_once_with(42)

    def test_get_issue_increments_api_calls(
        self, client, mock_github, mock_repo, mock_issue, mock_rate_limit
    ):
        """API call counter is incremented."""
        mock_github.get_repo.return_value = mock_repo
        mock_repo.get_issue.return_value = mock_issue
        mock_github.get_rate_limit.return_value = mock_rate_limit

        client.get_issue("test-owner/test-repo", 42)

        assert client.stats.api_calls == 1

    def test_get_issue_empty_body(
        self, client, mock_github, mock_repo, mock_issue, mock_rate_limit
    ):
        """Handles issue with None body."""
        mock_issue.body = None
        mock_github.get_repo.return_value = mock_repo
        mock_repo.get_issue.return_value = mock_issue
        mock_github.get_rate_limit.return_value = mock_rate_limit

        result = client.get_issue("test-owner/test-repo", 42)
        assert result.body == ""

    def test_get_issue_caches_repo(
        self, client, mock_github, mock_repo, mock_issue, mock_rate_limit
    ):
        """Repository is cached after first access."""
        mock_github.get_repo.return_value = mock_repo
        mock_repo.get_issue.return_value = mock_issue
        mock_github.get_rate_limit.return_value = mock_rate_limit

        client.get_issue("test-owner/test-repo", 42)
        client.get_issue("test-owner/test-repo", 43)

        # get_repo should only be called once (cached)
        mock_github.get_repo.assert_called_once_with("test-owner/test-repo")


# ========================
# Comment Tests
# ========================


class TestPostComment:
    """Tests for post_comment method."""

    def test_post_comment_success(
        self, client, mock_github, mock_repo, mock_issue, mock_comment, mock_rate_limit
    ):
        """Successfully posts a comment."""
        mock_github.get_repo.return_value = mock_repo
        mock_repo.get_issue.return_value = mock_issue
        mock_issue.create_comment.return_value = mock_comment
        mock_github.get_rate_limit.return_value = mock_rate_limit

        result = client.post_comment(
            "test-owner/test-repo", 42, "Hello World!"
        )

        assert isinstance(result, CommentData)
        assert result.id == 123
        assert result.body == "Test comment body"
        assert result.user == "commenter"
        assert client.stats.comments_posted == 1
        mock_issue.create_comment.assert_called_once_with("Hello World!")

    def test_post_comment_empty_body_raises(self, client):
        """Raises ValueError for empty comment body."""
        with pytest.raises(ValueError, match="Comment body cannot be empty"):
            client.post_comment("test-owner/test-repo", 42, "")

    def test_post_comment_whitespace_body_raises(self, client):
        """Raises ValueError for whitespace-only comment body."""
        with pytest.raises(ValueError, match="Comment body cannot be empty"):
            client.post_comment("test-owner/test-repo", 42, "   \n  ")


class TestGetIssueComments:
    """Tests for get_issue_comments method."""

    def test_get_comments_success(
        self, client, mock_github, mock_repo, mock_issue, mock_comment, mock_rate_limit
    ):
        """Successfully retrieves comments."""
        mock_github.get_repo.return_value = mock_repo
        mock_repo.get_issue.return_value = mock_issue
        mock_issue.get_comments.return_value = [mock_comment, mock_comment]
        mock_github.get_rate_limit.return_value = mock_rate_limit

        results = client.get_issue_comments("test-owner/test-repo", 42)

        assert len(results) == 2
        assert all(isinstance(c, CommentData) for c in results)

    def test_get_comments_max_limit(
        self, client, mock_github, mock_repo, mock_issue, mock_comment, mock_rate_limit
    ):
        """Respects max_comments limit."""
        comments = [MagicMock() for _ in range(10)]
        for c in comments:
            c.id = 1
            c.body = "comment"
            c.user = MagicMock()
            c.user.login = "user"
            c.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
            c.updated_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

        mock_github.get_repo.return_value = mock_repo
        mock_repo.get_issue.return_value = mock_issue
        mock_issue.get_comments.return_value = iter(comments)
        mock_github.get_rate_limit.return_value = mock_rate_limit

        results = client.get_issue_comments(
            "test-owner/test-repo", 42, max_comments=3
        )

        assert len(results) == 3


# ========================
# Label Tests
# ========================


class TestListLabels:
    """Tests for list_labels method."""

    def test_list_labels_success(
        self, client, mock_github, mock_repo, mock_rate_limit
    ):
        """Successfully lists labels."""
        label1 = MagicMock()
        label1.name = "bug"
        label1.color = "d73a4a"
        label1.description = "Something broken"

        label2 = MagicMock()
        label2.name = "enhancement"
        label2.color = "a2eeef"
        label2.description = None  # Test None description

        mock_github.get_repo.return_value = mock_repo
        mock_repo.get_labels.return_value = [label1, label2]
        mock_github.get_rate_limit.return_value = mock_rate_limit

        results = client.list_labels("test-owner/test-repo")

        assert len(results) == 2
        assert results[0].name == "bug"
        assert results[0].color == "d73a4a"
        assert results[0].description == "Something broken"
        assert results[1].description == ""  # None converted to empty string


# ========================
# Rate Limit Tests
# ========================


class TestRateLimit:
    """Tests for rate limit handling."""

    def test_get_rate_limit(self, client, mock_github, mock_rate_limit):
        """Successfully retrieves rate limit info."""
        mock_github.get_rate_limit.return_value = mock_rate_limit

        result = client.get_rate_limit()

        assert isinstance(result, RateLimitInfo)
        assert result.remaining == 4999
        assert result.limit == 5000

    def test_rate_limit_triggers_wait(
        self, client, mock_github, mock_rate_limit
    ):
        """Rate limit buffer triggers wait when near limit."""
        # Set remaining below buffer (default 10)
        mock_rate_limit.core.remaining = 5
        mock_rate_limit.core.reset = datetime.now(timezone.utc)
        mock_github.get_rate_limit.return_value = mock_rate_limit

        with patch("src.integrations.github_client.time.sleep") as mock_sleep:
            client._check_rate_limit()
            # Should have called sleep
            assert client.stats.rate_limit_waits == 1


# ========================
# Retry Tests
# ========================


class TestRetryLogic:
    """Tests for retry logic."""

    def test_retry_on_server_error(self, client, mock_github, mock_rate_limit):
        """Retries on 500 server errors."""
        from github import GithubException

        mock_github.get_rate_limit.return_value = mock_rate_limit
        mock_github.get_repo.side_effect = GithubException(
            500, {"message": "Internal Server Error"}, None
        )

        with pytest.raises(GithubException):
            client.get_issue("test-owner/test-repo", 42)

        assert client.stats.retries == 2  # max_retries=2
        assert client.stats.errors == 3  # Initial + 2 retries

    def test_no_retry_on_404(self, client, mock_github, mock_rate_limit):
        """Does not retry on 404 errors."""
        from github import GithubException

        mock_github.get_rate_limit.return_value = mock_rate_limit
        mock_github.get_repo.side_effect = GithubException(
            404, {"message": "Not Found"}, None
        )

        with pytest.raises(GithubException):
            client.get_issue("test-owner/test-repo", 42)

        assert client.stats.retries == 0
        assert client.stats.errors == 1

    def test_no_retry_on_403(self, client, mock_github, mock_rate_limit):
        """Does not retry on 403 permission errors."""
        from github import GithubException

        mock_github.get_rate_limit.return_value = mock_rate_limit
        mock_github.get_repo.side_effect = GithubException(
            403, {"message": "Forbidden"}, None
        )

        with pytest.raises(GithubException):
            client.get_issue("test-owner/test-repo", 42)

        assert client.stats.retries == 0

    def test_retry_succeeds_on_second_attempt(
        self, client, mock_github, mock_repo, mock_issue, mock_rate_limit
    ):
        """Operation succeeds after initial failure."""
        from github import GithubException

        mock_github.get_rate_limit.return_value = mock_rate_limit

        # First call fails, second succeeds
        mock_github.get_repo.side_effect = [
            GithubException(502, {"message": "Bad Gateway"}, None),
            mock_repo,
        ]
        mock_repo.get_issue.return_value = mock_issue

        result = client.get_issue("test-owner/test-repo", 42)

        assert isinstance(result, IssueData)
        assert result.number == 42
        assert client.stats.retries == 1


# ========================
# Context Manager Tests
# ========================


class TestContextManager:
    """Tests for context manager protocol."""

    def test_context_manager(self, mock_github, mock_auth):
        """Client works as context manager."""
        with GitHubClient(token="test-token") as client:
            assert client is not None

    def test_repr(self, client):
        """Client has useful repr."""
        assert "GitHubClient" in repr(client)
        assert "api_calls=0" in repr(client)


# ========================
# Global Client Tests
# ========================


class TestGlobalClient:
    """Tests for module-level client management."""

    def test_close_github_client_when_none(self):
        """close_github_client handles None client gracefully."""
        # Reset global
        import src.integrations.github_client as mod

        mod._global_client = None
        close_github_client()  # Should not raise
