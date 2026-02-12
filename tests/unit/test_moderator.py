"""
Tests for the ModeratorAgent agent selection logic.

Tests cover:
- Keyword-based agent selection for various issue types
- Agent selection result structure and methods
- ModeratorAgent initialization and configuration
- Selection with different issue categories
- Minimum/maximum agent count enforcement
- Compliance agent inclusion
- Fallback from LLM to keyword matching
"""

import os
import pytest

from src.agents.registry import AgentRegistry
from src.orchestration.moderator import (
    AgentSelectionResult,
    ModeratorAgent,
    select_agents_by_keywords,
    MIN_AGENTS,
    MAX_AGENTS,
)


# ==================
# Fixtures
# ==================


@pytest.fixture
def registry():
    """Create an AgentRegistry loaded with production agents."""
    reg = AgentRegistry()
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config",
        "agent_definitions.yaml",
    )
    reg.load_from_yaml(config_path)
    return reg


@pytest.fixture
def moderator(registry):
    """Create a ModeratorAgent with the production registry."""
    return ModeratorAgent(registry=registry)


@pytest.fixture
def ui_issue():
    """Sample UI-related issue."""
    return {
        "title": "Add dark mode toggle to settings page",
        "body": (
            "Users should be able to switch between light and dark mode "
            "from the settings page. The toggle should persist across sessions "
            "using local storage. Need to update CSS variables and ensure "
            "all components support both themes. Must be accessible with "
            "keyboard navigation."
        ),
        "labels": ["enhancement", "ui"],
    }


@pytest.fixture
def security_issue():
    """Sample security-related issue."""
    return {
        "title": "Implement CSRF protection for all API endpoints",
        "body": (
            "We need to add CSRF token validation to all POST, PUT, DELETE "
            "endpoints. This includes generating tokens, validating on the server, "
            "and including tokens in forms and AJAX requests. Also need to audit "
            "existing authentication middleware for vulnerabilities."
        ),
        "labels": ["security", "bug"],
    }


@pytest.fixture
def performance_issue():
    """Sample performance-related issue."""
    return {
        "title": "Optimize database queries causing slow page loads",
        "body": (
            "The dashboard page takes 5+ seconds to load. Profiling shows "
            "N+1 query problems in the user activity feed. We need to add "
            "proper indexing, implement query batching, and add Redis caching "
            "for frequently accessed data."
        ),
        "labels": ["performance", "database"],
    }


@pytest.fixture
def mobile_issue():
    """Sample mobile-related issue."""
    return {
        "title": "Build native iOS and Android push notification support",
        "body": (
            "Implement push notifications for mobile apps. Need to integrate "
            "with Apple Push Notification Service (APNS) for iOS and Firebase "
            "Cloud Messaging (FCM) for Android. Include notification preferences "
            "settings and support for rich notifications with images."
        ),
        "labels": ["mobile", "feature"],
    }


@pytest.fixture
def vague_issue():
    """Issue with minimal context."""
    return {
        "title": "Fix bug",
        "body": "Something is broken",
        "labels": [],
    }


# ==================
# AgentSelectionResult Tests
# ==================


class TestAgentSelectionResult:
    """Tests for AgentSelectionResult data class."""

    def test_basic_properties(self, registry):
        """Test basic properties of AgentSelectionResult."""
        agents = [
            registry.get_agent("system_architect"),
            registry.get_agent("frontend_dev"),
        ]
        reasoning = {
            "system_architect": "Direct relevance",
            "frontend_dev": "Implementation needed",
        }
        result = AgentSelectionResult(
            agents=agents, reasoning=reasoning, method="keyword"
        )

        assert result.count == 2
        assert result.agent_names == ["system_architect", "frontend_dev"]
        assert result.method == "keyword"

    def test_to_dict(self, registry):
        """Test serialization to dictionary."""
        agents = [registry.get_agent("system_architect")]
        result = AgentSelectionResult(
            agents=agents,
            reasoning={"system_architect": "Test reason"},
            method="llm",
        )
        d = result.to_dict()
        assert d["agents"] == ["system_architect"]
        assert d["reasoning"]["system_architect"] == "Test reason"
        assert d["method"] == "llm"
        assert d["count"] == 1


# ==================
# Keyword Selection Tests
# ==================


class TestKeywordSelection:
    """Tests for keyword-based agent selection."""

    def test_ui_issue_selects_ui_agents(self, ui_issue, registry):
        """UI issue should select UI-relevant agents."""
        result = select_agents_by_keywords(
            issue_title=ui_issue["title"],
            issue_body=ui_issue["body"],
            issue_labels=ui_issue["labels"],
            registry=registry,
        )

        names = result.agent_names
        assert "ui_architect" in names
        assert "frontend_dev" in names
        assert result.method == "keyword"

    def test_ui_issue_includes_accessibility(self, ui_issue, registry):
        """UI issue mentioning keyboard navigation selects ADA expert."""
        result = select_agents_by_keywords(
            issue_title=ui_issue["title"],
            issue_body=ui_issue["body"],
            issue_labels=ui_issue["labels"],
            registry=registry,
        )
        names = result.agent_names
        assert "ada_expert" in names

    def test_security_issue_selects_security_agents(self, security_issue, registry):
        """Security issue should select security-relevant agents."""
        result = select_agents_by_keywords(
            issue_title=security_issue["title"],
            issue_body=security_issue["body"],
            issue_labels=security_issue["labels"],
            registry=registry,
        )

        names = result.agent_names
        assert "security_expert" in names
        assert "auth_expert" in names

    def test_performance_issue_selects_perf_agents(self, performance_issue, registry):
        """Performance issue should select performance and database agents."""
        result = select_agents_by_keywords(
            issue_title=performance_issue["title"],
            issue_body=performance_issue["body"],
            issue_labels=performance_issue["labels"],
            registry=registry,
        )

        names = result.agent_names
        assert "performance_expert" in names
        assert "database_expert" in names

    def test_mobile_issue_selects_mobile_agents(self, mobile_issue, registry):
        """Mobile issue should select mobile-relevant agents."""
        result = select_agents_by_keywords(
            issue_title=mobile_issue["title"],
            issue_body=mobile_issue["body"],
            issue_labels=mobile_issue["labels"],
            registry=registry,
        )

        names = result.agent_names
        assert "ios_developer" in names or "android_developer" in names
        assert "notification_expert" in names

    def test_minimum_agent_count(self, vague_issue, registry):
        """Even vague issues should select at least MIN_AGENTS agents."""
        result = select_agents_by_keywords(
            issue_title=vague_issue["title"],
            issue_body=vague_issue["body"],
            issue_labels=vague_issue["labels"],
            registry=registry,
        )

        assert result.count >= MIN_AGENTS

    def test_maximum_agent_count(self, registry):
        """Selection should not exceed MAX_AGENTS."""
        # Create an issue that matches many categories
        result = select_agents_by_keywords(
            issue_title="Full stack security performance database API mobile UI testing deployment",
            issue_body=(
                "We need to implement a comprehensive feature that touches "
                "the frontend UI, backend API, database queries, mobile apps, "
                "security authentication, performance optimization, testing, "
                "and deployment CI/CD pipeline. Also needs accessibility, "
                "internationalization, SEO, email notifications, and search."
            ),
            issue_labels=["security", "performance", "ui", "mobile", "infrastructure"],
            registry=registry,
        )

        assert result.count <= MAX_AGENTS

    def test_compliance_agents_included(self, ui_issue, registry):
        """Compliance agents should be included even if not directly matched."""
        result = select_agents_by_keywords(
            issue_title=ui_issue["title"],
            issue_body=ui_issue["body"],
            issue_labels=ui_issue["labels"],
            registry=registry,
        )

        names = result.agent_names
        # At least one compliance agent should be present
        compliance_present = (
            "security_expert" in names or "qa_engineer" in names
        )
        assert compliance_present

    def test_reasoning_provided(self, ui_issue, registry):
        """Each selected agent should have a selection reason."""
        result = select_agents_by_keywords(
            issue_title=ui_issue["title"],
            issue_body=ui_issue["body"],
            issue_labels=ui_issue["labels"],
            registry=registry,
        )

        for name in result.agent_names:
            assert name in result.reasoning
            assert len(result.reasoning[name]) > 0

    def test_agents_ordered_by_relevance(self, ui_issue, registry):
        """Most relevant agents should appear first."""
        result = select_agents_by_keywords(
            issue_title=ui_issue["title"],
            issue_body=ui_issue["body"],
            issue_labels=ui_issue["labels"],
            registry=registry,
        )

        # First few agents should be UI-related for a UI issue
        top_agents = result.agent_names[:5]
        ui_related = {"ui_architect", "frontend_dev", "ux_designer", "ada_expert", "ui_designer"}
        overlap = set(top_agents) & ui_related
        assert len(overlap) >= 2, f"Expected UI agents in top 5, got {top_agents}"


class TestKeywordSelectionEdgeCases:
    """Edge case tests for keyword selection."""

    def test_empty_issue(self, registry):
        """Empty issue should still return agents (minimum count)."""
        result = select_agents_by_keywords(
            issue_title="",
            issue_body="",
            issue_labels=[],
            registry=registry,
        )
        assert result.count >= MIN_AGENTS

    def test_labels_influence_selection(self, registry):
        """Issue labels should influence agent selection."""
        result_with_labels = select_agents_by_keywords(
            issue_title="Fix issue",
            issue_body="Something needs fixing",
            issue_labels=["security", "authentication"],
            registry=registry,
        )

        result_without_labels = select_agents_by_keywords(
            issue_title="Fix issue",
            issue_body="Something needs fixing",
            issue_labels=[],
            registry=registry,
        )

        # Security labels should bring in security agents
        names_with = result_with_labels.agent_names
        assert "security_expert" in names_with or "auth_expert" in names_with

    def test_case_insensitive_matching(self, registry):
        """Keyword matching should be case-insensitive."""
        result = select_agents_by_keywords(
            issue_title="IMPLEMENT DARK MODE UI TOGGLE",
            issue_body="NEEDS RESPONSIVE CSS AND ACCESSIBILITY",
            issue_labels=[],
            registry=registry,
        )

        names = result.agent_names
        assert "ui_architect" in names or "frontend_dev" in names


# ==================
# ModeratorAgent Tests
# ==================


class TestModeratorAgentInit:
    """Tests for ModeratorAgent initialization."""

    def test_default_init(self):
        """ModeratorAgent initializes with defaults."""
        mod = ModeratorAgent()
        assert mod.model == "claude-sonnet-4-5-20250929"
        assert mod.registry is None

    def test_custom_model(self):
        """ModeratorAgent accepts custom model."""
        mod = ModeratorAgent(model="claude-haiku-3")
        assert mod.model == "claude-haiku-3"

    def test_with_registry(self, registry):
        """ModeratorAgent accepts a registry."""
        mod = ModeratorAgent(registry=registry)
        assert mod.registry is registry

    def test_registry_setter(self, registry):
        """Registry can be set after initialization."""
        mod = ModeratorAgent()
        assert mod.registry is None
        mod.registry = registry
        assert mod.registry is registry


class TestModeratorAgentSelection:
    """Tests for ModeratorAgent.select_relevant_agents()."""

    @pytest.mark.asyncio
    async def test_select_falls_back_to_keyword(self, moderator, ui_issue):
        """Without API key, falls back to keyword selection."""
        # Ensure no API key is set (tests shouldn't call real APIs)
        original_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            result = await moderator.select_relevant_agents(ui_issue)
            assert result.method == "keyword"
            assert result.count >= MIN_AGENTS
            assert result.count <= MAX_AGENTS
            assert "ui_architect" in result.agent_names
        finally:
            if original_key:
                os.environ["ANTHROPIC_API_KEY"] = original_key

    @pytest.mark.asyncio
    async def test_select_requires_registry(self):
        """Raises ValueError if no registry is available."""
        mod = ModeratorAgent()
        with pytest.raises(ValueError, match="No agent registry"):
            await mod.select_relevant_agents({"title": "Test", "body": ""})

    @pytest.mark.asyncio
    async def test_select_with_passed_registry(self, registry, ui_issue):
        """Can pass registry directly to select_relevant_agents."""
        mod = ModeratorAgent()
        result = await mod.select_relevant_agents(ui_issue, registry=registry)
        assert result.count >= MIN_AGENTS

    @pytest.mark.asyncio
    async def test_select_ui_issue(self, moderator, ui_issue):
        """UI issue selects appropriate agents."""
        result = await moderator.select_relevant_agents(ui_issue)
        names = result.agent_names

        assert "ui_architect" in names
        assert "frontend_dev" in names
        assert result.count >= MIN_AGENTS
        assert result.count <= MAX_AGENTS

    @pytest.mark.asyncio
    async def test_select_security_issue(self, moderator, security_issue):
        """Security issue selects appropriate agents."""
        result = await moderator.select_relevant_agents(security_issue)
        names = result.agent_names

        assert "security_expert" in names
        assert result.count >= MIN_AGENTS

    @pytest.mark.asyncio
    async def test_select_performance_issue(self, moderator, performance_issue):
        """Performance issue selects appropriate agents."""
        result = await moderator.select_relevant_agents(performance_issue)
        names = result.agent_names

        assert "performance_expert" in names
        assert result.count >= MIN_AGENTS

    @pytest.mark.asyncio
    async def test_select_mobile_issue(self, moderator, mobile_issue):
        """Mobile issue selects appropriate agents."""
        result = await moderator.select_relevant_agents(mobile_issue)
        names = result.agent_names

        mobile_agents = {"ios_developer", "android_developer", "mobile_web_dev"}
        assert len(set(names) & mobile_agents) >= 1

    @pytest.mark.asyncio
    async def test_selection_result_serializable(self, moderator, ui_issue):
        """Selection result can be serialized to dict."""
        result = await moderator.select_relevant_agents(ui_issue)
        d = result.to_dict()

        assert isinstance(d, dict)
        assert "agents" in d
        assert "reasoning" in d
        assert "method" in d
        assert "count" in d
        assert d["count"] == len(d["agents"])

    @pytest.mark.asyncio
    async def test_different_issues_different_agents(self, moderator, ui_issue, security_issue):
        """Different issue types should select different agent compositions."""
        ui_result = await moderator.select_relevant_agents(ui_issue)
        sec_result = await moderator.select_relevant_agents(security_issue)

        ui_names = set(ui_result.agent_names)
        sec_names = set(sec_result.agent_names)

        # There should be some difference in selected agents
        unique_to_ui = ui_names - sec_names
        unique_to_sec = sec_names - ui_names

        # At least one agent should be unique to each selection
        assert len(unique_to_ui) >= 1 or len(unique_to_sec) >= 1


# ==================
# Integration-style Tests
# ==================


class TestMultiIssueCoverage:
    """Test agent selection across various issue types."""

    @pytest.mark.asyncio
    async def test_devops_issue(self, moderator):
        """DevOps issue selects infrastructure agents."""
        issue = {
            "title": "Set up CI/CD pipeline with Docker and Kubernetes",
            "body": (
                "Need to create GitHub Actions workflows for automated testing, "
                "building Docker images, and deploying to Kubernetes cluster. "
                "Include staging and production environments."
            ),
            "labels": ["infrastructure", "devops"],
        }
        result = await moderator.select_relevant_agents(issue)
        names = result.agent_names
        infra_agents = {"devops_engineer", "cloud_architect", "sre"}
        assert len(set(names) & infra_agents) >= 1

    @pytest.mark.asyncio
    async def test_data_science_issue(self, moderator):
        """Data science issue selects ML/data agents."""
        issue = {
            "title": "Implement recommendation engine using machine learning",
            "body": (
                "Build a product recommendation system using collaborative "
                "filtering and content-based approaches. Need model training "
                "pipeline, A/B testing framework, and analytics dashboard."
            ),
            "labels": ["ml", "feature"],
        }
        result = await moderator.select_relevant_agents(issue)
        names = result.agent_names
        ml_agents = {"ml_engineer", "data_scientist", "analytics_expert", "data_engineer"}
        assert len(set(names) & ml_agents) >= 1

    @pytest.mark.asyncio
    async def test_payment_issue(self, moderator):
        """Payment issue selects payment and security agents."""
        issue = {
            "title": "Integrate Stripe payment processing for subscriptions",
            "body": (
                "Implement subscription billing with Stripe. Need checkout flow, "
                "webhook handling for payment events, invoice generation, and "
                "support for multiple currencies and payment methods."
            ),
            "labels": ["payment", "feature"],
        }
        result = await moderator.select_relevant_agents(issue)
        names = result.agent_names
        assert "payment_systems_expert" in names
        assert "security_expert" in names

    @pytest.mark.asyncio
    async def test_i18n_issue(self, moderator):
        """Internationalization issue selects i18n agents."""
        issue = {
            "title": "Add multi-language support with RTL layout",
            "body": (
                "Implement internationalization using react-intl. Support "
                "English, Spanish, Arabic (RTL), and Japanese. Include "
                "translation workflow and locale-aware date/number formatting."
            ),
            "labels": ["i18n"],
        }
        result = await moderator.select_relevant_agents(issue)
        names = result.agent_names
        assert "i18n_expert" in names

    @pytest.mark.asyncio
    async def test_documentation_issue(self, moderator):
        """Documentation issue selects documentation agents."""
        issue = {
            "title": "Write API documentation with OpenAPI/Swagger",
            "body": (
                "Create comprehensive API documentation using OpenAPI spec. "
                "Include endpoint descriptions, request/response examples, "
                "authentication guides, and a getting started tutorial."
            ),
            "labels": ["documentation"],
        }
        result = await moderator.select_relevant_agents(issue)
        names = result.agent_names
        assert "tech_writer" in names or "api_architect" in names
