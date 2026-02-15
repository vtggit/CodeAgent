"""
LLM Provider Manager for Multi-Agent GitHub Issue Routing System.

This module provides a centralized LLM provider management layer built on top
of LiteLLM. It adds:
- Multi-provider support with automatic fallback
- Per-provider rate limiting
- Cost tracking across providers
- Provider health monitoring
- Unified configuration for all agent LLM calls

Usage:
    from src.integrations.llm_provider import LLMProviderManager, get_provider_manager

    manager = get_provider_manager()
    response = await manager.completion(
        model="anthropic/claude-sonnet-4-5-20250929",
        messages=[{"role": "user", "content": "Hello"}],
    )
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Any, Optional

import litellm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Suppress litellm verbose logging
litellm.set_verbose = False


class ProviderStatus(str, Enum):
    """Health status of an LLM provider."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ProviderCostRecord:
    """Tracks cost for a single LLM API call."""
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ProviderHealth:
    """Tracks the health state of an LLM provider."""
    provider: str
    status: ProviderStatus = ProviderStatus.UNKNOWN
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    total_calls: int = 0
    total_failures: int = 0
    average_latency_ms: float = 0.0
    _latency_samples: list[float] = field(default_factory=list)

    def record_success(self, latency_ms: float) -> None:
        """Record a successful API call."""
        self.total_calls += 1
        self.consecutive_failures = 0
        self.last_success = datetime.now(timezone.utc)
        self.status = ProviderStatus.HEALTHY
        self._latency_samples.append(latency_ms)
        # Keep last 100 samples
        if len(self._latency_samples) > 100:
            self._latency_samples = self._latency_samples[-100:]
        self.average_latency_ms = sum(self._latency_samples) / len(self._latency_samples)

    def record_failure(self) -> None:
        """Record a failed API call."""
        self.total_calls += 1
        self.total_failures += 1
        self.consecutive_failures += 1
        self.last_failure = datetime.now(timezone.utc)
        if self.consecutive_failures >= 3:
            self.status = ProviderStatus.UNHEALTHY
        elif self.consecutive_failures >= 1:
            self.status = ProviderStatus.DEGRADED

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "provider": self.provider,
            "status": self.status.value,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "consecutive_failures": self.consecutive_failures,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "average_latency_ms": round(self.average_latency_ms, 2),
        }


@dataclass
class ProviderRateLimit:
    """
    Simple token-bucket rate limiter for a provider.

    Limits requests per minute to prevent API abuse and stay within
    provider-specific rate limits.
    """
    provider: str
    max_requests_per_minute: int = 60
    _request_times: list[float] = field(default_factory=list)

    async def acquire(self) -> None:
        """
        Wait until we can make a request within rate limits.

        Uses a sliding window of 60 seconds to track requests.
        """
        now = time.time()
        # Clean old entries (older than 60 seconds)
        self._request_times = [t for t in self._request_times if now - t < 60]

        if len(self._request_times) >= self.max_requests_per_minute:
            # Wait until the oldest request in the window expires
            oldest = self._request_times[0]
            wait_time = 60 - (now - oldest)
            if wait_time > 0:
                logger.warning(
                    "Rate limit reached for provider %s, waiting %.1fs",
                    self.provider,
                    wait_time,
                )
                await asyncio.sleep(wait_time)

        self._request_times.append(time.time())

    @property
    def current_usage(self) -> int:
        """Get current requests in the window."""
        now = time.time()
        return len([t for t in self._request_times if now - t < 60])


# Default rate limits per provider
DEFAULT_RATE_LIMITS: dict[str, int] = {
    "anthropic": 50,
    "openai": 60,
    "lm_studio": 100,  # Local, can be higher
    "ollama": 100,      # Local, can be higher
}

# Known cost per token (in USD) for common models
# These are approximate and should be updated periodically
MODEL_COSTS: dict[str, dict[str, float]] = {
    "anthropic/claude-sonnet-4-5-20250929": {"input": 0.003 / 1000, "output": 0.015 / 1000},
    "anthropic/claude-haiku-3-5-20241022": {"input": 0.00025 / 1000, "output": 0.00125 / 1000},
    "anthropic/claude-opus-4-20250514": {"input": 0.015 / 1000, "output": 0.075 / 1000},
    "openai/gpt-4o": {"input": 0.005 / 1000, "output": 0.015 / 1000},
    "openai/gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
    "openai/gpt-4-turbo": {"input": 0.01 / 1000, "output": 0.03 / 1000},
}


def build_model_string(provider: str, model: str) -> str:
    """
    Build a litellm-compatible model string from provider and model name.

    Args:
        provider: Provider name (anthropic, openai, lm_studio, ollama)
        model: Model identifier

    Returns:
        litellm-compatible model string (e.g., "anthropic/claude-sonnet-4-5-20250929")
    """
    provider_lower = provider.lower()

    if provider_lower == "anthropic":
        return f"anthropic/{model}"
    elif provider_lower == "openai":
        return f"openai/{model}"
    elif provider_lower in ("lm_studio", "ollama"):
        # Local models use openai-compatible format
        return f"openai/{model}"
    else:
        # Default: try the model name directly
        return model


def calculate_cost(model_string: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate the cost of an API call based on token usage.

    Args:
        model_string: The litellm model string
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens

    Returns:
        Estimated cost in USD
    """
    costs = MODEL_COSTS.get(model_string)
    if not costs:
        return 0.0

    input_cost = prompt_tokens * costs["input"]
    output_cost = completion_tokens * costs["output"]
    return input_cost + output_cost


class LLMProviderManager:
    """
    Central manager for all LLM provider interactions.

    Provides:
    - Multi-provider support with automatic fallback
    - Per-provider rate limiting
    - Cost tracking across providers
    - Provider health monitoring
    - Unified completion API

    Example:
        manager = LLMProviderManager()
        response = await manager.completion(
            model="anthropic/claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": "Hello"}],
        )
    """

    def __init__(
        self,
        enable_cost_tracking: bool = False,
        enable_rate_limiting: bool = True,
        rate_limits: Optional[dict[str, int]] = None,
    ) -> None:
        """
        Initialize the LLM Provider Manager.

        Args:
            enable_cost_tracking: Whether to track costs per call
            enable_rate_limiting: Whether to enforce rate limits
            rate_limits: Custom rate limits per provider (requests/minute)
        """
        self._enable_cost_tracking = enable_cost_tracking
        self._enable_rate_limiting = enable_rate_limiting

        # Provider health tracking
        self._health: dict[str, ProviderHealth] = {}

        # Rate limiters per provider
        self._rate_limiters: dict[str, ProviderRateLimit] = {}
        limits = rate_limits or DEFAULT_RATE_LIMITS
        for provider, limit in limits.items():
            self._rate_limiters[provider] = ProviderRateLimit(
                provider=provider,
                max_requests_per_minute=limit,
            )

        # Cost tracking
        self._cost_records: list[ProviderCostRecord] = []
        self._total_cost_usd: float = 0.0

        logger.info(
            "LLMProviderManager initialized",
            cost_tracking=enable_cost_tracking,
            rate_limiting=enable_rate_limiting,
        )

    def _get_provider_from_model(self, model: str) -> str:
        """Extract provider name from model string."""
        if "/" in model:
            return model.split("/")[0]
        return "unknown"

    def _get_health(self, provider: str) -> ProviderHealth:
        """Get or create health tracker for a provider."""
        if provider not in self._health:
            self._health[provider] = ProviderHealth(provider=provider)
        return self._health[provider]

    def _get_rate_limiter(self, provider: str) -> ProviderRateLimit:
        """Get or create rate limiter for a provider."""
        if provider not in self._rate_limiters:
            limit = DEFAULT_RATE_LIMITS.get(provider, 60)
            self._rate_limiters[provider] = ProviderRateLimit(
                provider=provider,
                max_requests_per_minute=limit,
            )
        return self._rate_limiters[provider]

    def _record_cost(
        self,
        model: str,
        provider: str,
        usage: Optional[Any],
    ) -> Optional[ProviderCostRecord]:
        """Record cost for an API call if cost tracking is enabled."""
        if not self._enable_cost_tracking or not usage:
            return None

        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0

        cost = calculate_cost(model, prompt_tokens, completion_tokens)

        record = ProviderCostRecord(
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
        )

        self._cost_records.append(record)
        self._total_cost_usd += cost

        return record

    async def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        api_base: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        fallback_models: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Make a completion request with automatic fallback support.

        Args:
            model: Primary model string (e.g., "anthropic/claude-sonnet-4-5-20250929")
            messages: Chat messages in OpenAI format
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            api_base: Custom API base URL (for local models)
            tools: Tool definitions for function calling
            fallback_models: Alternative models to try if primary fails
            **kwargs: Additional litellm parameters

        Returns:
            litellm completion response

        Raises:
            Exception: If all models (primary + fallbacks) fail
        """
        models_to_try = [model] + (fallback_models or [])
        last_error: Optional[Exception] = None

        for i, current_model in enumerate(models_to_try):
            provider = self._get_provider_from_model(current_model)
            health = self._get_health(provider)

            # Skip unhealthy providers (unless it's the last option)
            if (
                health.status == ProviderStatus.UNHEALTHY
                and i < len(models_to_try) - 1
            ):
                logger.warning(
                    "Skipping unhealthy provider %s, trying next fallback",
                    provider,
                )
                continue

            try:
                # Rate limiting
                if self._enable_rate_limiting:
                    limiter = self._get_rate_limiter(provider)
                    await limiter.acquire()

                # Build kwargs
                call_kwargs: dict[str, Any] = {
                    "model": current_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **kwargs,
                }

                if api_base:
                    call_kwargs["api_base"] = api_base

                if tools:
                    call_kwargs["tools"] = tools

                # Make the call
                start_time = time.time()
                response = await litellm.acompletion(**call_kwargs)
                latency_ms = (time.time() - start_time) * 1000

                # Record success
                health.record_success(latency_ms)

                # Cost tracking
                usage = getattr(response, "usage", None)
                cost_record = self._record_cost(current_model, provider, usage)

                if cost_record and cost_record.cost_usd > 0:
                    logger.debug(
                        "LLM call cost: $%.6f (%d tokens)",
                        cost_record.cost_usd,
                        cost_record.total_tokens,
                    )

                logger.debug(
                    "LLM completion success: model=%s, latency=%.0fms",
                    current_model,
                    latency_ms,
                )

                return response

            except Exception as e:
                last_error = e
                health.record_failure()

                if i < len(models_to_try) - 1:
                    logger.warning(
                        "LLM call failed for %s, trying fallback: %s",
                        current_model,
                        str(e),
                    )
                else:
                    logger.error(
                        "LLM call failed for %s (no more fallbacks): %s",
                        current_model,
                        str(e),
                    )

        # All models failed
        raise last_error or RuntimeError("All LLM providers failed")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    async def completion_with_retry(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        api_base: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        fallback_models: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Make a completion request with automatic retry and fallback.

        Combines tenacity retry logic with provider fallback for maximum
        reliability.

        Args:
            Same as completion()

        Returns:
            litellm completion response
        """
        return await self.completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            api_base=api_base,
            tools=tools,
            fallback_models=fallback_models,
            **kwargs,
        )

    # ==================
    # Status & Metrics
    # ==================

    def get_provider_health(self, provider: Optional[str] = None) -> dict[str, Any]:
        """
        Get health status for one or all providers.

        Args:
            provider: Specific provider to check, or None for all

        Returns:
            Dictionary with health information
        """
        if provider:
            health = self._get_health(provider)
            return health.to_dict()

        return {
            name: h.to_dict()
            for name, h in self._health.items()
        }

    def get_cost_summary(self) -> dict[str, Any]:
        """
        Get cost tracking summary.

        Returns:
            Dictionary with cost information per provider and totals
        """
        if not self._enable_cost_tracking:
            return {"enabled": False}

        by_provider: dict[str, float] = {}
        by_model: dict[str, float] = {}
        total_tokens = 0

        for record in self._cost_records:
            by_provider[record.provider] = by_provider.get(record.provider, 0) + record.cost_usd
            by_model[record.model] = by_model.get(record.model, 0) + record.cost_usd
            total_tokens += record.total_tokens

        return {
            "enabled": True,
            "total_cost_usd": round(self._total_cost_usd, 6),
            "total_tokens": total_tokens,
            "total_calls": len(self._cost_records),
            "cost_by_provider": {k: round(v, 6) for k, v in by_provider.items()},
            "cost_by_model": {k: round(v, 6) for k, v in by_model.items()},
        }

    def get_rate_limit_status(self) -> dict[str, Any]:
        """
        Get rate limit status for all providers.

        Returns:
            Dictionary with rate limit information
        """
        return {
            name: {
                "max_requests_per_minute": limiter.max_requests_per_minute,
                "current_usage": limiter.current_usage,
            }
            for name, limiter in self._rate_limiters.items()
        }

    def get_full_status(self) -> dict[str, Any]:
        """
        Get complete status including health, costs, and rate limits.

        Returns:
            Comprehensive status dictionary
        """
        return {
            "providers": self.get_provider_health(),
            "costs": self.get_cost_summary(),
            "rate_limits": self.get_rate_limit_status(),
        }

    def reset_cost_tracking(self) -> None:
        """Reset all cost tracking data."""
        self._cost_records.clear()
        self._total_cost_usd = 0.0

    def reset_health(self, provider: Optional[str] = None) -> None:
        """Reset health tracking for one or all providers."""
        if provider and provider in self._health:
            self._health[provider] = ProviderHealth(provider=provider)
        elif not provider:
            self._health.clear()


# ==================
# Global Instance
# ==================

_provider_manager: Optional[LLMProviderManager] = None


def get_provider_manager(
    enable_cost_tracking: Optional[bool] = None,
    enable_rate_limiting: bool = True,
    rate_limits: Optional[dict[str, int]] = None,
) -> LLMProviderManager:
    """
    Get or create the global LLM Provider Manager.

    On first call, creates the manager with the given settings.
    Subsequent calls return the same instance.

    Args:
        enable_cost_tracking: Whether to track costs
        enable_rate_limiting: Whether to enforce rate limits
        rate_limits: Custom rate limits per provider

    Returns:
        The global LLMProviderManager instance
    """
    global _provider_manager

    if _provider_manager is None:
        # Try to read cost tracking setting from app settings
        if enable_cost_tracking is None:
            try:
                from src.config.settings import get_settings
                settings = get_settings()
                enable_cost_tracking = settings.enable_cost_tracking
            except Exception:
                enable_cost_tracking = False

        _provider_manager = LLMProviderManager(
            enable_cost_tracking=enable_cost_tracking,
            enable_rate_limiting=enable_rate_limiting,
            rate_limits=rate_limits,
        )

    return _provider_manager


def reset_provider_manager() -> None:
    """Reset the global provider manager (for testing)."""
    global _provider_manager
    _provider_manager = None
