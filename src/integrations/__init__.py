"""
Integrations module for external service clients.

Provides wrapper classes for GitHub API, result posting,
and potentially other external services in the future.
"""

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
from src.integrations.result_poster import (
    ResultFormatter,
    ResultPoster,
)

__all__ = [
    "GitHubClient",
    "IssueData",
    "CommentData",
    "LabelData",
    "RateLimitInfo",
    "ClientStats",
    "get_github_client",
    "close_github_client",
    "ResultFormatter",
    "ResultPoster",
]
