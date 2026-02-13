"""
Integrations module for external service clients.

Provides wrapper classes for GitHub API, and potentially
other external services in the future.
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

__all__ = [
    "GitHubClient",
    "IssueData",
    "CommentData",
    "LabelData",
    "RateLimitInfo",
    "ClientStats",
    "get_github_client",
    "close_github_client",
]
