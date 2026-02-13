"""
GitHub Client wrapper for the Multi-Agent GitHub Issue Routing System.

Provides a clean interface around PyGithub for posting comments to issues,
retrieving issue details, listing labels, and handling rate limits with
automatic retry/backoff.

Usage:
    from src.integrations.github_client import GitHubClient

    client = GitHubClient(token="ghp_...")
    issue = client.get_issue("owner/repo", 42)
    client.post_comment("owner/repo", 42, "Hello from the system!")
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from github import Auth, Github, GithubException, RateLimitExceededException
from github.Issue import Issue as GithubIssue
from github.IssueComment import IssueComment as GithubIssueComment
from github.Label import Label as GithubLabel
from github.Repository import Repository as GithubRepository

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


# ========================
# Data Classes
# ========================


@dataclass
class IssueData:
    """Simplified issue data returned by the client."""

    number: int
    title: str
    body: str
    state: str
    labels: list[str]
    created_at: datetime
    updated_at: datetime
    user: str
    url: str
    comments_count: int


@dataclass
class CommentData:
    """Simplified comment data returned by the client."""

    id: int
    body: str
    user: str
    created_at: datetime
    updated_at: datetime


@dataclass
class LabelData:
    """Simplified label data returned by the client."""

    name: str
    color: str
    description: str


@dataclass
class RateLimitInfo:
    """Rate limit status information."""

    remaining: int
    limit: int
    reset_time: datetime
    used: int


@dataclass
class ClientStats:
    """Tracks client usage statistics."""

    api_calls: int = 0
    comments_posted: int = 0
    rate_limit_waits: int = 0
    errors: int = 0
    retries: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ========================
# GitHub Client
# ========================


class GitHubClient:
    """
    Wrapper around PyGithub for GitHub API interactions.

    Provides methods for:
    - Retrieving issue details
    - Posting comments to issues
    - Listing labels on a repository
    - Getting issue comments with pagination
    - Rate limit detection and automatic backoff
    - Error handling with configurable retries

    Args:
        token: GitHub personal access token. If None, reads from settings.
        max_retries: Maximum number of retries on transient failures (default: 3).
        retry_delay: Base delay between retries in seconds (default: 2.0).
        rate_limit_buffer: Minimum remaining API calls before triggering wait (default: 10).
    """

    def __init__(
        self,
        token: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        rate_limit_buffer: int = 10,
    ):
        self._token = token or self._get_token_from_settings()
        if not self._token:
            raise ValueError(
                "GitHub token is required. Set GITHUB_TOKEN environment variable "
                "or pass token parameter."
            )

        self._auth = Auth.Token(self._token)
        self._github = Github(auth=self._auth)
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._rate_limit_buffer = rate_limit_buffer
        self._stats = ClientStats()
        self._repo_cache: dict[str, GithubRepository] = {}

        logger.info("GitHubClient initialized (max_retries=%d)", max_retries)

    @staticmethod
    def _get_token_from_settings() -> Optional[str]:
        """Get GitHub token from application settings."""
        try:
            settings = get_settings()
            if settings.has_github_token:
                return settings.github_token
        except Exception:
            pass
        return None

    # ========================
    # Repository Access
    # ========================

    def _get_repo(self, repo_full_name: str) -> GithubRepository:
        """
        Get a GitHub repository object, with caching.

        Args:
            repo_full_name: Repository in "owner/repo" format.

        Returns:
            GitHub Repository object.

        Raises:
            GithubException: If repository not found or access denied.
        """
        if repo_full_name not in self._repo_cache:
            self._repo_cache[repo_full_name] = self._github.get_repo(repo_full_name)
        return self._repo_cache[repo_full_name]

    # ========================
    # Issue Operations
    # ========================

    def get_issue(self, repo_full_name: str, issue_number: int) -> IssueData:
        """
        Get details of a specific issue.

        Args:
            repo_full_name: Repository in "owner/repo" format.
            issue_number: Issue number.

        Returns:
            IssueData with issue details.

        Raises:
            GithubException: If issue not found or access denied.
        """
        return self._with_retry(
            lambda: self._get_issue_impl(repo_full_name, issue_number),
            f"get_issue({repo_full_name}#{issue_number})",
        )

    def _get_issue_impl(self, repo_full_name: str, issue_number: int) -> IssueData:
        """Internal implementation for get_issue."""
        self._check_rate_limit()
        self._stats.api_calls += 1

        repo = self._get_repo(repo_full_name)
        issue = repo.get_issue(issue_number)

        return IssueData(
            number=issue.number,
            title=issue.title,
            body=issue.body or "",
            state=issue.state,
            labels=[label.name for label in issue.labels],
            created_at=issue.created_at,
            updated_at=issue.updated_at,
            user=issue.user.login if issue.user else "unknown",
            url=issue.html_url,
            comments_count=issue.comments,
        )

    def post_comment(
        self,
        repo_full_name: str,
        issue_number: int,
        body: str,
    ) -> CommentData:
        """
        Post a comment on a GitHub issue.

        Args:
            repo_full_name: Repository in "owner/repo" format.
            issue_number: Issue number.
            body: Comment body (Markdown supported).

        Returns:
            CommentData with the created comment details.

        Raises:
            GithubException: If posting fails.
            ValueError: If body is empty.
        """
        if not body or not body.strip():
            raise ValueError("Comment body cannot be empty")

        return self._with_retry(
            lambda: self._post_comment_impl(repo_full_name, issue_number, body),
            f"post_comment({repo_full_name}#{issue_number})",
        )

    def _post_comment_impl(
        self,
        repo_full_name: str,
        issue_number: int,
        body: str,
    ) -> CommentData:
        """Internal implementation for post_comment."""
        self._check_rate_limit()
        self._stats.api_calls += 1

        repo = self._get_repo(repo_full_name)
        issue = repo.get_issue(issue_number)
        comment = issue.create_comment(body)

        self._stats.comments_posted += 1
        logger.info(
            "Posted comment on %s#%d (comment_id=%d, length=%d)",
            repo_full_name,
            issue_number,
            comment.id,
            len(body),
        )

        return CommentData(
            id=comment.id,
            body=comment.body,
            user=comment.user.login if comment.user else "unknown",
            created_at=comment.created_at,
            updated_at=comment.updated_at,
        )

    def get_issue_comments(
        self,
        repo_full_name: str,
        issue_number: int,
        max_comments: int = 100,
    ) -> list[CommentData]:
        """
        Get comments on a GitHub issue with pagination support.

        Args:
            repo_full_name: Repository in "owner/repo" format.
            issue_number: Issue number.
            max_comments: Maximum number of comments to retrieve (default: 100).

        Returns:
            List of CommentData objects.
        """
        return self._with_retry(
            lambda: self._get_issue_comments_impl(
                repo_full_name, issue_number, max_comments
            ),
            f"get_issue_comments({repo_full_name}#{issue_number})",
        )

    def _get_issue_comments_impl(
        self,
        repo_full_name: str,
        issue_number: int,
        max_comments: int,
    ) -> list[CommentData]:
        """Internal implementation for get_issue_comments."""
        self._check_rate_limit()
        self._stats.api_calls += 1

        repo = self._get_repo(repo_full_name)
        issue = repo.get_issue(issue_number)

        comments = []
        for i, comment in enumerate(issue.get_comments()):
            if i >= max_comments:
                break
            comments.append(
                CommentData(
                    id=comment.id,
                    body=comment.body,
                    user=comment.user.login if comment.user else "unknown",
                    created_at=comment.created_at,
                    updated_at=comment.updated_at,
                )
            )

        logger.debug(
            "Retrieved %d comments from %s#%d",
            len(comments),
            repo_full_name,
            issue_number,
        )
        return comments

    # ========================
    # Label Operations
    # ========================

    def list_labels(self, repo_full_name: str) -> list[LabelData]:
        """
        List all labels on a repository.

        Args:
            repo_full_name: Repository in "owner/repo" format.

        Returns:
            List of LabelData objects.
        """
        return self._with_retry(
            lambda: self._list_labels_impl(repo_full_name),
            f"list_labels({repo_full_name})",
        )

    def _list_labels_impl(self, repo_full_name: str) -> list[LabelData]:
        """Internal implementation for list_labels."""
        self._check_rate_limit()
        self._stats.api_calls += 1

        repo = self._get_repo(repo_full_name)
        labels = []
        for label in repo.get_labels():
            labels.append(
                LabelData(
                    name=label.name,
                    color=label.color,
                    description=label.description or "",
                )
            )

        logger.debug(
            "Retrieved %d labels from %s",
            len(labels),
            repo_full_name,
        )
        return labels

    def get_issue_labels(
        self, repo_full_name: str, issue_number: int
    ) -> list[LabelData]:
        """
        Get labels on a specific issue.

        Args:
            repo_full_name: Repository in "owner/repo" format.
            issue_number: Issue number.

        Returns:
            List of LabelData objects.
        """
        return self._with_retry(
            lambda: self._get_issue_labels_impl(repo_full_name, issue_number),
            f"get_issue_labels({repo_full_name}#{issue_number})",
        )

    def _get_issue_labels_impl(
        self, repo_full_name: str, issue_number: int
    ) -> list[LabelData]:
        """Internal implementation for get_issue_labels."""
        self._check_rate_limit()
        self._stats.api_calls += 1

        repo = self._get_repo(repo_full_name)
        issue = repo.get_issue(issue_number)
        labels = []
        for label in issue.labels:
            labels.append(
                LabelData(
                    name=label.name,
                    color=label.color,
                    description=label.description or "",
                )
            )
        return labels

    # ========================
    # Rate Limit Management
    # ========================

    def get_rate_limit(self) -> RateLimitInfo:
        """
        Get current rate limit status.

        Returns:
            RateLimitInfo with current rate limit details.
        """
        rate_limit = self._github.get_rate_limit()
        core = rate_limit.core

        return RateLimitInfo(
            remaining=core.remaining,
            limit=core.limit,
            reset_time=core.reset,
            used=core.limit - core.remaining,
        )

    def _check_rate_limit(self) -> None:
        """
        Check rate limit and wait if necessary.

        If remaining API calls are below the buffer threshold,
        waits until the rate limit resets.
        """
        try:
            rate_limit = self._github.get_rate_limit()
            core = rate_limit.core

            if core.remaining <= self._rate_limit_buffer:
                wait_seconds = max(
                    0,
                    (core.reset - datetime.now(timezone.utc)).total_seconds() + 1,
                )
                if wait_seconds > 0:
                    logger.warning(
                        "Rate limit nearly exhausted (%d remaining). "
                        "Waiting %.0f seconds until reset.",
                        core.remaining,
                        wait_seconds,
                    )
                    self._stats.rate_limit_waits += 1
                    time.sleep(min(wait_seconds, 300))  # Max 5 min wait
        except Exception as e:
            logger.debug("Could not check rate limit: %s", str(e))

    # ========================
    # Retry Logic
    # ========================

    def _with_retry(self, operation: Any, operation_name: str) -> Any:
        """
        Execute an operation with retry logic.

        Uses exponential backoff for transient failures and
        automatic waiting for rate limit errors.

        Args:
            operation: Callable to execute.
            operation_name: Human-readable name for logging.

        Returns:
            Result of the operation.

        Raises:
            The last exception if all retries are exhausted.
        """
        last_error = None

        for attempt in range(self._max_retries + 1):
            try:
                return operation()
            except RateLimitExceededException as e:
                self._stats.rate_limit_waits += 1
                # Wait for rate limit reset
                reset_time = getattr(e, "headers", {}).get(
                    "X-RateLimit-Reset", None
                )
                if reset_time:
                    wait_seconds = max(
                        0, int(reset_time) - int(time.time()) + 1
                    )
                else:
                    wait_seconds = 60

                wait_seconds = min(wait_seconds, 300)  # Max 5 min wait
                logger.warning(
                    "%s: Rate limit exceeded (attempt %d/%d). "
                    "Waiting %d seconds.",
                    operation_name,
                    attempt + 1,
                    self._max_retries + 1,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                last_error = e
            except GithubException as e:
                last_error = e
                self._stats.errors += 1

                # Don't retry 404s or 403s (permission errors)
                if e.status in (404, 403, 401, 422):
                    logger.error(
                        "%s: GitHub API error %d: %s",
                        operation_name,
                        e.status,
                        str(e.data) if e.data else str(e),
                    )
                    raise

                # Retry on server errors (5xx) and connection issues
                if attempt < self._max_retries:
                    self._stats.retries += 1
                    delay = self._retry_delay * (2 ** attempt)
                    logger.warning(
                        "%s: GitHub API error (attempt %d/%d, status=%s). "
                        "Retrying in %.1f seconds.",
                        operation_name,
                        attempt + 1,
                        self._max_retries + 1,
                        e.status,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "%s: All retries exhausted. Last error: %s",
                        operation_name,
                        str(e),
                    )
                    raise
            except Exception as e:
                last_error = e
                self._stats.errors += 1

                if attempt < self._max_retries:
                    self._stats.retries += 1
                    delay = self._retry_delay * (2 ** attempt)
                    logger.warning(
                        "%s: Unexpected error (attempt %d/%d): %s. "
                        "Retrying in %.1f seconds.",
                        operation_name,
                        attempt + 1,
                        self._max_retries + 1,
                        str(e),
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "%s: All retries exhausted. Last error: %s",
                        operation_name,
                        str(e),
                    )
                    raise

        # Should not reach here, but just in case
        if last_error:
            raise last_error

    # ========================
    # Client Management
    # ========================

    @property
    def stats(self) -> ClientStats:
        """Get client usage statistics."""
        return self._stats

    def close(self) -> None:
        """Close the GitHub client connection."""
        try:
            self._github.close()
            logger.info(
                "GitHubClient closed (api_calls=%d, comments_posted=%d, "
                "errors=%d, retries=%d, rate_limit_waits=%d)",
                self._stats.api_calls,
                self._stats.comments_posted,
                self._stats.errors,
                self._stats.retries,
                self._stats.rate_limit_waits,
            )
        except Exception as e:
            logger.debug("Error closing GitHub client: %s", str(e))

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

    def __repr__(self) -> str:
        return (
            f"GitHubClient(api_calls={self._stats.api_calls}, "
            f"comments_posted={self._stats.comments_posted})"
        )


# ========================
# Module-level convenience
# ========================


_global_client: Optional[GitHubClient] = None


def get_github_client() -> Optional[GitHubClient]:
    """
    Get or create a global GitHubClient instance.

    Returns None if no GitHub token is configured.

    Returns:
        GitHubClient instance, or None if not configured.
    """
    global _global_client

    if _global_client is not None:
        return _global_client

    try:
        settings = get_settings()
        if settings.has_github_token:
            _global_client = GitHubClient(token=settings.github_token)
            return _global_client
    except Exception as e:
        logger.warning("Failed to create GitHubClient: %s", str(e))

    return None


def close_github_client() -> None:
    """Close the global GitHubClient instance."""
    global _global_client
    if _global_client is not None:
        _global_client.close()
        _global_client = None
