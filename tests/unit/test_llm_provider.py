"""
Tests for the LLM Provider Manager.

Tests cover:
- Provider manager initialization and configuration
- Model string building from provider/model pairs
- Cost calculation for known models
- Rate limiting with sliding window
- Provider health tracking (success/failure recording)
- Completion with fallback models
- Cost tracking and summary
- Provider status reporting
- Global instance management (get/reset)
- Integration with agent configurations
"""

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.integrations.llm_provider import (
    DEFAULT_RATE_LIMITS,
    MODEL_COSTS,
    LLMProviderManager,
    ProviderCostRecord,
    ProviderHealth,
    ProviderRateLimit,
    ProviderStatus,
    build_model_string,
    calculate_cost,
    get_provider_manager,
    reset_provider_manager,
)
from src.models.agent import (
    AgentConfig,
    AgentType,
    FallbackProviderConfig,
    LLMProviderConfig,
)


# ==================
# Fixtures
# ==================


@pytest.fixture(autouse=True)
def reset_global_manager():
    """Reset the global provider manager before and after each test."""
    reset_provider_manager()
    yield
    reset_provider_manager()


@pytest.fixture
def manager():
    """Create a fresh LLMProviderManager for testing."""
    return LLMProviderManager(
        enable_cost_tracking=True,
        enable_rate_limiting=True,
    )


@pytest.fixture
def manager_no_tracking():
    """Create a manager without cost tracking."""
    return LLMProviderManager(
        enable_cost_tracking=False,
        enable_rate_limiting=False,
    )


@pytest.fixture
def mock_litellm_response():
    """Create a mock litellm completion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Test response content"
    response.model = "anthropic/claude-sonnet-4-5-20250929"
    response.usage = MagicMock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    response.usage.total_tokens = 150
    return response


# ==================
# build_model_string tests
# ==================


class TestBuildModelString:
    """Tests for the build_model_string utility function."""

    def test_anthropic_provider(self):
        """Test Anthropic model string format."""
        result = build_model_string("anthropic", "claude-sonnet-4-5-20250929")
        assert result == "anthropic/claude-sonnet-4-5-20250929"

    def test_openai_provider(self):
        """Test OpenAI model string format."""
        result = build_model_string("openai", "gpt-4o")
        assert result == "openai/gpt-4o"

    def test_lm_studio_provider(self):
        """Test LM Studio uses OpenAI-compatible format."""
        result = build_model_string("lm_studio", "local-model")
        assert result == "openai/local-model"

    def test_ollama_provider(self):
        """Test Ollama uses OpenAI-compatible format."""
        result = build_model_string("ollama", "llama3")
        assert result == "openai/llama3"

    def test_unknown_provider(self):
        """Test unknown provider returns model name directly."""
        result = build_model_string("custom_provider", "my-model")
        assert result == "my-model"

    def test_case_insensitive(self):
        """Test provider name is case-insensitive."""
        result = build_model_string("Anthropic", "claude-sonnet-4-5-20250929")
        assert result == "anthropic/claude-sonnet-4-5-20250929"

    def test_openai_case(self):
        """Test OpenAI case insensitivity."""
        result = build_model_string("OpenAI", "gpt-4")
        assert result == "openai/gpt-4"

    def test_empty_model(self):
        """Test empty model string."""
        result = build_model_string("anthropic", "")
        assert result == "anthropic/"


# ==================
# calculate_cost tests
# ==================


class TestCalculateCost:
    """Tests for the calculate_cost function."""

    def test_known_model_cost(self):
        """Test cost calculation for a known model."""
        cost = calculate_cost(
            "anthropic/claude-sonnet-4-5-20250929",
            prompt_tokens=1000,
            completion_tokens=500,
        )
        assert cost > 0

    def test_unknown_model_zero_cost(self):
        """Test unknown model returns zero cost."""
        cost = calculate_cost(
            "unknown/model",
            prompt_tokens=1000,
            completion_tokens=500,
        )
        assert cost == 0.0

    def test_zero_tokens_zero_cost(self):
        """Test zero tokens results in zero cost."""
        cost = calculate_cost(
            "anthropic/claude-sonnet-4-5-20250929",
            prompt_tokens=0,
            completion_tokens=0,
        )
        assert cost == 0.0

    def test_cost_calculation_accuracy(self):
        """Test cost calculation matches expected values."""
        model = "anthropic/claude-sonnet-4-5-20250929"
        costs = MODEL_COSTS[model]
        prompt_tokens = 1000
        completion_tokens = 500

        expected_cost = (
            prompt_tokens * costs["input"] +
            completion_tokens * costs["output"]
        )

        actual_cost = calculate_cost(model, prompt_tokens, completion_tokens)
        assert abs(actual_cost - expected_cost) < 1e-10


# ==================
# ProviderHealth tests
# ==================


class TestProviderHealth:
    """Tests for the ProviderHealth dataclass."""

    def test_initial_state(self):
        """Test initial health state is unknown."""
        health = ProviderHealth(provider="anthropic")
        assert health.status == ProviderStatus.UNKNOWN
        assert health.consecutive_failures == 0
        assert health.total_calls == 0

    def test_record_success(self):
        """Test recording a successful call."""
        health = ProviderHealth(provider="anthropic")
        health.record_success(latency_ms=150.0)

        assert health.status == ProviderStatus.HEALTHY
        assert health.total_calls == 1
        assert health.consecutive_failures == 0
        assert health.last_success is not None
        assert health.average_latency_ms == 150.0

    def test_record_failure(self):
        """Test recording a failed call."""
        health = ProviderHealth(provider="anthropic")
        health.record_failure()

        assert health.status == ProviderStatus.DEGRADED
        assert health.total_calls == 1
        assert health.total_failures == 1
        assert health.consecutive_failures == 1
        assert health.last_failure is not None

    def test_consecutive_failures_unhealthy(self):
        """Test that 3 consecutive failures marks provider as unhealthy."""
        health = ProviderHealth(provider="anthropic")
        health.record_failure()
        health.record_failure()
        assert health.status == ProviderStatus.DEGRADED

        health.record_failure()
        assert health.status == ProviderStatus.UNHEALTHY
        assert health.consecutive_failures == 3

    def test_success_resets_failures(self):
        """Test that a success resets consecutive failure count."""
        health = ProviderHealth(provider="anthropic")
        health.record_failure()
        health.record_failure()
        assert health.consecutive_failures == 2

        health.record_success(100.0)
        assert health.consecutive_failures == 0
        assert health.status == ProviderStatus.HEALTHY

    def test_average_latency_calculation(self):
        """Test average latency calculation with multiple samples."""
        health = ProviderHealth(provider="anthropic")
        health.record_success(100.0)
        health.record_success(200.0)
        health.record_success(300.0)

        assert health.average_latency_ms == 200.0

    def test_latency_sample_limit(self):
        """Test that latency samples are limited to 100."""
        health = ProviderHealth(provider="anthropic")
        for i in range(150):
            health.record_success(float(i))

        # Should only keep last 100 samples (50-149)
        assert len(health._latency_samples) == 100

    def test_to_dict(self):
        """Test serialization to dictionary."""
        health = ProviderHealth(provider="anthropic")
        health.record_success(100.0)

        d = health.to_dict()
        assert d["provider"] == "anthropic"
        assert d["status"] == "healthy"
        assert d["total_calls"] == 1
        assert d["total_failures"] == 0
        assert d["consecutive_failures"] == 0
        assert d["average_latency_ms"] == 100.0


# ==================
# ProviderRateLimit tests
# ==================


class TestProviderRateLimit:
    """Tests for the ProviderRateLimit class."""

    def test_initial_usage(self):
        """Test initial rate limit usage is zero."""
        limiter = ProviderRateLimit(provider="anthropic", max_requests_per_minute=60)
        assert limiter.current_usage == 0

    @pytest.mark.asyncio
    async def test_acquire_under_limit(self):
        """Test acquiring within rate limit completes immediately."""
        limiter = ProviderRateLimit(provider="anthropic", max_requests_per_minute=60)
        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start

        # Should complete almost immediately
        assert elapsed < 1.0
        assert limiter.current_usage == 1

    @pytest.mark.asyncio
    async def test_multiple_acquires(self):
        """Test multiple acquisitions within limit."""
        limiter = ProviderRateLimit(provider="anthropic", max_requests_per_minute=100)
        for _ in range(5):
            await limiter.acquire()

        assert limiter.current_usage == 5

    def test_expired_requests_cleaned(self):
        """Test that requests older than 60 seconds are cleaned up."""
        limiter = ProviderRateLimit(provider="anthropic", max_requests_per_minute=60)
        # Manually add old timestamps
        old_time = time.time() - 120  # 2 minutes ago
        limiter._request_times = [old_time, old_time + 1, old_time + 2]

        # Current usage should be 0 since all are expired
        assert limiter.current_usage == 0


# ==================
# ProviderCostRecord tests
# ==================


class TestProviderCostRecord:
    """Tests for the ProviderCostRecord dataclass."""

    def test_creation(self):
        """Test creating a cost record."""
        record = ProviderCostRecord(
            provider="anthropic",
            model="anthropic/claude-sonnet-4-5-20250929",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.00375,
        )
        assert record.provider == "anthropic"
        assert record.cost_usd == 0.00375

    def test_to_dict(self):
        """Test serialization to dictionary."""
        record = ProviderCostRecord(
            provider="anthropic",
            model="test-model",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.01,
        )
        d = record.to_dict()
        assert d["provider"] == "anthropic"
        assert d["model"] == "test-model"
        assert d["prompt_tokens"] == 100
        assert d["completion_tokens"] == 50
        assert d["total_tokens"] == 150
        assert d["cost_usd"] == 0.01
        assert "timestamp" in d

    def test_defaults(self):
        """Test default values."""
        record = ProviderCostRecord(
            provider="openai",
            model="gpt-4o",
        )
        assert record.prompt_tokens == 0
        assert record.completion_tokens == 0
        assert record.total_tokens == 0
        assert record.cost_usd == 0.0


# ==================
# LLMProviderManager tests
# ==================


class TestLLMProviderManager:
    """Tests for the LLMProviderManager class."""

    def test_initialization(self, manager):
        """Test manager initializes correctly."""
        assert manager._enable_cost_tracking is True
        assert manager._enable_rate_limiting is True

    def test_initialization_no_tracking(self, manager_no_tracking):
        """Test manager initializes without tracking."""
        assert manager_no_tracking._enable_cost_tracking is False
        assert manager_no_tracking._enable_rate_limiting is False

    def test_custom_rate_limits(self):
        """Test manager with custom rate limits."""
        limits = {"anthropic": 30, "openai": 40}
        manager = LLMProviderManager(rate_limits=limits)
        assert manager._rate_limiters["anthropic"].max_requests_per_minute == 30
        assert manager._rate_limiters["openai"].max_requests_per_minute == 40

    def test_get_provider_from_model(self, manager):
        """Test extracting provider from model string."""
        assert manager._get_provider_from_model("anthropic/claude") == "anthropic"
        assert manager._get_provider_from_model("openai/gpt-4") == "openai"
        assert manager._get_provider_from_model("unknown-model") == "unknown"

    def test_get_health_creates_new(self, manager):
        """Test health tracker creation for new provider."""
        health = manager._get_health("anthropic")
        assert health.provider == "anthropic"
        assert health.status == ProviderStatus.UNKNOWN

    def test_get_health_returns_existing(self, manager):
        """Test health tracker returns same instance."""
        health1 = manager._get_health("anthropic")
        health1.record_success(100.0)
        health2 = manager._get_health("anthropic")
        assert health2.total_calls == 1

    def test_get_rate_limiter_creates_new(self, manager):
        """Test rate limiter creation for new provider."""
        limiter = manager._get_rate_limiter("custom_provider")
        assert limiter.provider == "custom_provider"
        assert limiter.max_requests_per_minute == 60  # default

    def test_get_rate_limiter_known_provider(self, manager):
        """Test rate limiter for known provider uses default limits."""
        limiter = manager._get_rate_limiter("anthropic")
        assert limiter.max_requests_per_minute == DEFAULT_RATE_LIMITS["anthropic"]

    @pytest.mark.asyncio
    async def test_completion_success(self, manager, mock_litellm_response):
        """Test successful completion call."""
        with patch("src.integrations.llm_provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)

            response = await manager.completion(
                model="anthropic/claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert response.choices[0].message.content == "Test response content"
            mock_litellm.acompletion.assert_called_once()

    @pytest.mark.asyncio
    async def test_completion_records_health(self, manager, mock_litellm_response):
        """Test that completion records health on success."""
        with patch("src.integrations.llm_provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)

            await manager.completion(
                model="anthropic/claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hello"}],
            )

            health = manager.get_provider_health("anthropic")
            assert health["status"] == "healthy"
            assert health["total_calls"] == 1

    @pytest.mark.asyncio
    async def test_completion_records_cost(self, manager, mock_litellm_response):
        """Test that completion records cost when tracking is enabled."""
        with patch("src.integrations.llm_provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)

            await manager.completion(
                model="anthropic/claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hello"}],
            )

            costs = manager.get_cost_summary()
            assert costs["enabled"] is True
            assert costs["total_calls"] == 1
            assert costs["total_tokens"] == 150

    @pytest.mark.asyncio
    async def test_completion_no_cost_tracking(self, manager_no_tracking, mock_litellm_response):
        """Test that cost is not tracked when disabled."""
        with patch("src.integrations.llm_provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)

            await manager_no_tracking.completion(
                model="anthropic/claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hello"}],
            )

            costs = manager_no_tracking.get_cost_summary()
            assert costs["enabled"] is False

    @pytest.mark.asyncio
    async def test_completion_with_fallback(self, manager, mock_litellm_response):
        """Test fallback to alternative model on failure."""
        call_count = 0

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Primary provider failed")
            return mock_litellm_response

        with patch("src.integrations.llm_provider.litellm") as mock_litellm:
            mock_litellm.acompletion = mock_completion

            response = await manager.completion(
                model="anthropic/claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hello"}],
                fallback_models=["openai/gpt-4o"],
            )

            assert response.choices[0].message.content == "Test response content"
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_completion_all_fail(self, manager):
        """Test error when all models fail."""
        with patch("src.integrations.llm_provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=Exception("All failed"))

            with pytest.raises(Exception, match="All failed"):
                await manager.completion(
                    model="anthropic/claude-sonnet-4-5-20250929",
                    messages=[{"role": "user", "content": "Hello"}],
                    fallback_models=["openai/gpt-4o"],
                )

    @pytest.mark.asyncio
    async def test_completion_skips_unhealthy(self, manager, mock_litellm_response):
        """Test that unhealthy providers are skipped during fallback."""
        # Mark anthropic as unhealthy
        health = manager._get_health("anthropic")
        for _ in range(3):
            health.record_failure()
        assert health.status == ProviderStatus.UNHEALTHY

        with patch("src.integrations.llm_provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)

            response = await manager.completion(
                model="anthropic/claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hello"}],
                fallback_models=["openai/gpt-4o"],
            )

            # Should have skipped anthropic and gone straight to openai
            call_kwargs = mock_litellm.acompletion.call_args
            assert call_kwargs[1]["model"] == "openai/gpt-4o"

    @pytest.mark.asyncio
    async def test_completion_uses_unhealthy_as_last_resort(self, manager, mock_litellm_response):
        """Test that unhealthy provider is used if it's the only option."""
        # Mark anthropic as unhealthy
        health = manager._get_health("anthropic")
        for _ in range(3):
            health.record_failure()

        with patch("src.integrations.llm_provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)

            # No fallbacks, must use unhealthy provider
            response = await manager.completion(
                model="anthropic/claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert response.choices[0].message.content == "Test response content"

    @pytest.mark.asyncio
    async def test_completion_with_tools(self, manager, mock_litellm_response):
        """Test completion with tool definitions."""
        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        with patch("src.integrations.llm_provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)

            await manager.completion(
                model="anthropic/claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hello"}],
                tools=tools,
            )

            call_kwargs = mock_litellm.acompletion.call_args[1]
            assert "tools" in call_kwargs
            assert call_kwargs["tools"] == tools

    @pytest.mark.asyncio
    async def test_completion_with_api_base(self, manager, mock_litellm_response):
        """Test completion with custom API base URL."""
        with patch("src.integrations.llm_provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)

            await manager.completion(
                model="openai/local-model",
                messages=[{"role": "user", "content": "Hello"}],
                api_base="http://localhost:1234/v1",
            )

            call_kwargs = mock_litellm.acompletion.call_args[1]
            assert call_kwargs["api_base"] == "http://localhost:1234/v1"

    @pytest.mark.asyncio
    async def test_completion_with_retry(self, manager, mock_litellm_response):
        """Test completion_with_retry wrapper."""
        with patch("src.integrations.llm_provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)

            response = await manager.completion_with_retry(
                model="anthropic/claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert response.choices[0].message.content == "Test response content"

    # ==================
    # Status & Metrics
    # ==================

    def test_get_provider_health_all(self, manager):
        """Test getting health for all providers."""
        manager._get_health("anthropic").record_success(100.0)
        manager._get_health("openai").record_failure()

        health = manager.get_provider_health()
        assert "anthropic" in health
        assert "openai" in health
        assert health["anthropic"]["status"] == "healthy"
        assert health["openai"]["status"] == "degraded"

    def test_get_provider_health_single(self, manager):
        """Test getting health for a specific provider."""
        manager._get_health("anthropic").record_success(100.0)

        health = manager.get_provider_health("anthropic")
        assert health["status"] == "healthy"

    def test_get_cost_summary_empty(self, manager):
        """Test cost summary with no calls."""
        costs = manager.get_cost_summary()
        assert costs["enabled"] is True
        assert costs["total_cost_usd"] == 0
        assert costs["total_calls"] == 0

    def test_get_rate_limit_status(self, manager):
        """Test rate limit status reporting."""
        status = manager.get_rate_limit_status()
        assert "anthropic" in status
        assert "openai" in status
        assert status["anthropic"]["max_requests_per_minute"] == DEFAULT_RATE_LIMITS["anthropic"]

    def test_get_full_status(self, manager):
        """Test comprehensive status report."""
        status = manager.get_full_status()
        assert "providers" in status
        assert "costs" in status
        assert "rate_limits" in status

    def test_reset_cost_tracking(self, manager):
        """Test resetting cost data."""
        manager._cost_records.append(
            ProviderCostRecord(provider="test", model="test", cost_usd=1.0)
        )
        manager._total_cost_usd = 1.0

        manager.reset_cost_tracking()
        assert len(manager._cost_records) == 0
        assert manager._total_cost_usd == 0.0

    def test_reset_health_specific(self, manager):
        """Test resetting health for a specific provider."""
        health = manager._get_health("anthropic")
        health.record_success(100.0)
        assert health.total_calls == 1

        manager.reset_health("anthropic")
        health = manager._get_health("anthropic")
        assert health.total_calls == 0

    def test_reset_health_all(self, manager):
        """Test resetting health for all providers."""
        manager._get_health("anthropic").record_success(100.0)
        manager._get_health("openai").record_success(200.0)

        manager.reset_health()
        assert len(manager._health) == 0


# ==================
# Global Instance Tests
# ==================


class TestGlobalInstance:
    """Tests for the global provider manager instance."""

    def test_get_provider_manager_creates_instance(self):
        """Test first call creates a new instance."""
        manager = get_provider_manager(enable_cost_tracking=False)
        assert manager is not None
        assert isinstance(manager, LLMProviderManager)

    def test_get_provider_manager_returns_same(self):
        """Test subsequent calls return the same instance."""
        manager1 = get_provider_manager(enable_cost_tracking=False)
        manager2 = get_provider_manager()
        assert manager1 is manager2

    def test_reset_provider_manager(self):
        """Test resetting the global instance."""
        manager1 = get_provider_manager(enable_cost_tracking=False)
        reset_provider_manager()
        manager2 = get_provider_manager(enable_cost_tracking=True)
        assert manager1 is not manager2


# ==================
# Model Config Integration Tests
# ==================


class TestModelConfigIntegration:
    """Tests for integration with Pydantic model configs."""

    def test_llm_provider_config_defaults(self):
        """Test LLMProviderConfig default values."""
        config = LLMProviderConfig()
        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4-5-20250929"
        assert config.fallback_providers == []
        assert config.rate_limit_rpm is None

    def test_llm_provider_config_with_fallbacks(self):
        """Test LLMProviderConfig with fallback providers."""
        config = LLMProviderConfig(
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            fallback_providers=[
                FallbackProviderConfig(provider="openai", model="gpt-4o"),
                FallbackProviderConfig(
                    provider="lm_studio",
                    model="local-model",
                    base_url="http://localhost:1234/v1",
                ),
            ],
        )
        assert len(config.fallback_providers) == 2
        assert config.fallback_providers[0].provider == "openai"
        assert config.fallback_providers[1].base_url == "http://localhost:1234/v1"

    def test_llm_provider_config_with_rate_limit(self):
        """Test LLMProviderConfig with custom rate limit."""
        config = LLMProviderConfig(rate_limit_rpm=30)
        assert config.rate_limit_rpm == 30

    def test_fallback_provider_config(self):
        """Test FallbackProviderConfig creation."""
        fb = FallbackProviderConfig(
            provider="openai",
            model="gpt-4o-mini",
        )
        assert fb.provider == "openai"
        assert fb.model == "gpt-4o-mini"
        assert fb.base_url is None

    def test_agent_config_with_fallbacks(self):
        """Test AgentConfig with multi-provider LLM config."""
        config = AgentConfig(
            name="test_agent",
            type=AgentType.CLAUDE_SDK_TEXT,
            expertise="Testing",
            llm=LLMProviderConfig(
                provider="anthropic",
                model="claude-sonnet-4-5-20250929",
                fallback_providers=[
                    FallbackProviderConfig(provider="openai", model="gpt-4o"),
                ],
            ),
        )
        assert config.llm.provider == "anthropic"
        assert len(config.llm.fallback_providers) == 1
        assert config.llm.fallback_providers[0].model == "gpt-4o"

    def test_build_model_string_from_config(self):
        """Test building model strings from config objects."""
        config = LLMProviderConfig(provider="openai", model="gpt-4o")
        model_str = build_model_string(config.provider, config.model)
        assert model_str == "openai/gpt-4o"

    def test_build_fallback_strings_from_config(self):
        """Test building fallback model strings from config."""
        config = LLMProviderConfig(
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            fallback_providers=[
                FallbackProviderConfig(provider="openai", model="gpt-4o"),
                FallbackProviderConfig(provider="anthropic", model="claude-haiku-3-5-20241022"),
            ],
        )

        fallbacks = [
            build_model_string(fb.provider, fb.model)
            for fb in config.fallback_providers
        ]
        assert fallbacks == [
            "openai/gpt-4o",
            "anthropic/claude-haiku-3-5-20241022",
        ]


# ==================
# Agent Integration Tests
# ==================


class TestAgentIntegration:
    """Tests for integration with the agent classes."""

    @pytest.mark.asyncio
    async def test_text_agent_uses_provider_manager(self, mock_litellm_response):
        """Test that ClaudeTextAgent uses the LLMProviderManager."""
        from src.agents.claude_text_agent import ClaudeTextAgent

        config = AgentConfig(
            name="test_agent",
            type=AgentType.CLAUDE_SDK_TEXT,
            expertise="Testing",
            llm=LLMProviderConfig(
                provider="anthropic",
                model="claude-sonnet-4-5-20250929",
            ),
        )

        agent = ClaudeTextAgent(config)
        assert agent._provider_manager is not None
        assert agent._llm_model == "anthropic/claude-sonnet-4-5-20250929"
        assert agent._fallback_models == []

    @pytest.mark.asyncio
    async def test_text_agent_with_fallbacks(self):
        """Test ClaudeTextAgent initialization with fallback providers."""
        from src.agents.claude_text_agent import ClaudeTextAgent

        config = AgentConfig(
            name="test_agent",
            type=AgentType.CLAUDE_SDK_TEXT,
            expertise="Testing",
            llm=LLMProviderConfig(
                provider="anthropic",
                model="claude-sonnet-4-5-20250929",
                fallback_providers=[
                    FallbackProviderConfig(provider="openai", model="gpt-4o"),
                ],
            ),
        )

        agent = ClaudeTextAgent(config)
        assert len(agent._fallback_models) == 1
        assert agent._fallback_models[0] == "openai/gpt-4o"

    @pytest.mark.asyncio
    async def test_code_agent_uses_provider_manager(self):
        """Test that ClaudeCodeAgent uses the LLMProviderManager."""
        from src.agents.claude_code_agent import ClaudeCodeAgent

        config = AgentConfig(
            name="test_code_agent",
            type=AgentType.CLAUDE_SDK_CODE,
            expertise="Backend development",
            tools=["Read", "Grep"],
            can_edit_files=True,
            llm=LLMProviderConfig(
                provider="anthropic",
                model="claude-sonnet-4-5-20250929",
            ),
        )

        agent = ClaudeCodeAgent(config)
        assert agent._provider_manager is not None
        assert agent._llm_model == "anthropic/claude-sonnet-4-5-20250929"

    @pytest.mark.asyncio
    async def test_code_agent_with_fallbacks(self):
        """Test ClaudeCodeAgent with fallback providers."""
        from src.agents.claude_code_agent import ClaudeCodeAgent

        config = AgentConfig(
            name="test_code_agent",
            type=AgentType.CLAUDE_SDK_CODE,
            expertise="Backend development",
            tools=["Read", "Grep"],
            llm=LLMProviderConfig(
                provider="anthropic",
                model="claude-sonnet-4-5-20250929",
                fallback_providers=[
                    FallbackProviderConfig(provider="openai", model="gpt-4o"),
                ],
            ),
        )

        agent = ClaudeCodeAgent(config)
        assert len(agent._fallback_models) == 1
        assert agent._fallback_models[0] == "openai/gpt-4o"


# ==================
# Settings Integration Tests
# ==================


class TestSettingsIntegration:
    """Tests for integration with AppSettings."""

    def test_available_llm_providers_default(self):
        """Test available providers with default config."""
        from src.config.settings import AppSettings

        settings = AppSettings(
            anthropic_api_key=None,
            openai_api_key=None,
            _env_file=None,
        )
        assert settings.available_llm_providers == []

    def test_available_llm_providers_anthropic(self):
        """Test available providers with Anthropic key."""
        from src.config.settings import AppSettings

        settings = AppSettings(
            anthropic_api_key="sk-ant-real-key",
            _env_file=None,
        )
        assert "anthropic" in settings.available_llm_providers

    def test_available_llm_providers_openai(self):
        """Test available providers with OpenAI key."""
        from src.config.settings import AppSettings

        settings = AppSettings(
            openai_api_key="sk-real-openai-key",
            _env_file=None,
        )
        assert "openai" in settings.available_llm_providers

    def test_available_llm_providers_local(self):
        """Test available providers with local LM Studio."""
        from src.config.settings import AppSettings

        settings = AppSettings(
            lm_studio_base_url="http://localhost:1234/v1",
            _env_file=None,
        )
        assert "lm_studio" in settings.available_llm_providers

    def test_available_llm_providers_ollama(self):
        """Test available providers with Ollama."""
        from src.config.settings import AppSettings

        settings = AppSettings(
            ollama_base_url="http://localhost:11434",
            _env_file=None,
        )
        assert "ollama" in settings.available_llm_providers

    def test_available_llm_providers_multiple(self):
        """Test available providers with multiple configured."""
        from src.config.settings import AppSettings

        settings = AppSettings(
            anthropic_api_key="sk-ant-real",
            openai_api_key="sk-real-openai",
            lm_studio_base_url="http://localhost:1234",
            _env_file=None,
        )
        providers = settings.available_llm_providers
        assert "anthropic" in providers
        assert "openai" in providers
        assert "lm_studio" in providers

    def test_has_openai_key_false(self):
        """Test has_openai_key returns False for placeholder."""
        from src.config.settings import AppSettings

        settings = AppSettings(
            openai_api_key="sk-your_openai_api_key_here",
            _env_file=None,
        )
        assert settings.has_openai_key is False

    def test_has_openai_key_true(self):
        """Test has_openai_key returns True for real key."""
        from src.config.settings import AppSettings

        settings = AppSettings(
            openai_api_key="sk-real-key-123",
            _env_file=None,
        )
        assert settings.has_openai_key is True

    def test_new_settings_fields(self):
        """Test new settings fields have correct defaults."""
        from src.config.settings import AppSettings

        settings = AppSettings(_env_file=None)
        assert settings.openai_model == "gpt-4o"
        assert settings.lm_studio_model is None
        assert settings.ollama_base_url is None
        assert settings.ollama_model is None
        assert settings.llm_rate_limit_rpm == 60


# ==================
# Edge Cases
# ==================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_completion_empty_messages(self, manager, mock_litellm_response):
        """Test completion with empty messages list."""
        with patch("src.integrations.llm_provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)

            response = await manager.completion(
                model="anthropic/claude-sonnet-4-5-20250929",
                messages=[],
            )
            assert response is not None

    @pytest.mark.asyncio
    async def test_completion_no_usage_in_response(self, manager):
        """Test cost tracking when response has no usage data."""
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "Test"
        response.usage = None  # No usage data

        with patch("src.integrations.llm_provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=response)

            await manager.completion(
                model="anthropic/claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hello"}],
            )

            costs = manager.get_cost_summary()
            assert costs["total_calls"] == 0  # No cost recorded without usage

    def test_provider_health_serialization(self):
        """Test health serialization with no success/failure yet."""
        health = ProviderHealth(provider="test")
        d = health.to_dict()
        assert d["last_success"] is None
        assert d["last_failure"] is None

    @pytest.mark.asyncio
    async def test_completion_preserves_extra_kwargs(self, manager, mock_litellm_response):
        """Test that extra kwargs are passed through to litellm."""
        with patch("src.integrations.llm_provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)

            await manager.completion(
                model="anthropic/claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hello"}],
                top_p=0.9,
                stream=False,
            )

            call_kwargs = mock_litellm.acompletion.call_args[1]
            assert call_kwargs["top_p"] == 0.9
            assert call_kwargs["stream"] is False

    def test_multiple_cost_records(self, manager):
        """Test accumulation of multiple cost records."""
        for i in range(5):
            manager._cost_records.append(
                ProviderCostRecord(
                    provider="anthropic",
                    model="test",
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    cost_usd=0.01,
                )
            )
        manager._total_cost_usd = 0.05

        costs = manager.get_cost_summary()
        assert costs["total_calls"] == 5
        assert costs["total_cost_usd"] == 0.05
        assert costs["total_tokens"] == 750

    @pytest.mark.asyncio
    async def test_rate_limit_not_enforced_when_disabled(self, manager_no_tracking, mock_litellm_response):
        """Test that rate limiting is skipped when disabled."""
        with patch("src.integrations.llm_provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)

            # Should not enforce any rate limit
            for _ in range(5):
                await manager_no_tracking.completion(
                    model="anthropic/claude-sonnet-4-5-20250929",
                    messages=[{"role": "user", "content": "Hello"}],
                )

            assert mock_litellm.acompletion.call_count == 5
