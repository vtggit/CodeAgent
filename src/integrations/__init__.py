"""
Integrations module for external service clients.

Provides wrapper classes for GitHub API, result posting,
LLM provider management, and potentially other external services.
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
from src.integrations.llm_provider import (
    LLMProviderManager,
    ProviderStatus,
    ProviderCostRecord,
    ProviderHealth,
    ProviderRateLimit,
    build_model_string,
    calculate_cost,
    get_provider_manager,
    reset_provider_manager,
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
    "LLMProviderManager",
    "ProviderStatus",
    "ProviderCostRecord",
    "ProviderHealth",
    "ProviderRateLimit",
    "build_model_string",
    "calculate_cost",
    "get_provider_manager",
    "reset_provider_manager",
]
