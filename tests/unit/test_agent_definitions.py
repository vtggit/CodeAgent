"""
Tests for agent_definitions.yaml configuration file.

Validates that:
- The YAML file exists and is parseable
- All agents have required fields (type, expertise, domain_knowledge)
- Agent types are valid (claude_sdk_text, claude_sdk_code)
- 50+ agents are defined as required by the spec
- All agents load successfully into the registry
- Required agent categories are represented
- All agents from the spec are present
"""

import os

import pytest
import yaml

from src.agents.base import BaseAgent
from src.agents.claude_code_agent import ClaudeCodeAgent
from src.agents.claude_text_agent import ClaudeTextAgent
from src.agents.registry import AgentRegistry
from src.models.agent import AgentType


# Path to the agent definitions YAML
YAML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "config",
    "agent_definitions.yaml",
)


# ==================
# Fixtures
# ==================


@pytest.fixture
def yaml_data():
    """Load and parse the YAML config."""
    with open(YAML_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture
def agents_data(yaml_data):
    """Get the agents dictionary from YAML."""
    return yaml_data["agents"]


@pytest.fixture
def loaded_registry():
    """Load all agents into a registry."""
    registry = AgentRegistry()
    registry.load_from_yaml(YAML_PATH)
    return registry


# ==================
# File and Structure Tests
# ==================


class TestYAMLFile:
    """Tests for the YAML file existence and structure."""

    def test_yaml_file_exists(self):
        """agent_definitions.yaml exists."""
        assert os.path.exists(YAML_PATH), f"File not found: {YAML_PATH}"

    def test_yaml_is_valid(self, yaml_data):
        """YAML file parses successfully."""
        assert isinstance(yaml_data, dict)
        assert "agents" in yaml_data

    def test_agents_section_not_empty(self, agents_data):
        """Agents section contains definitions."""
        assert len(agents_data) > 0


# ==================
# Agent Count Tests
# ==================


class TestAgentCount:
    """Tests for the 50+ agent requirement."""

    def test_minimum_50_agents(self, agents_data):
        """At least 50 agents are defined."""
        assert len(agents_data) >= 50, (
            f"Expected at least 50 agents, found {len(agents_data)}"
        )

    def test_all_agents_load_into_registry(self, loaded_registry):
        """All agents load successfully with zero errors."""
        assert loaded_registry.agent_count >= 50
        assert len(loaded_registry.load_errors) == 0

    def test_registry_agent_count_matches_yaml(self, agents_data, loaded_registry):
        """Registry agent count matches YAML definition count."""
        assert loaded_registry.agent_count == len(agents_data)


# ==================
# Required Fields Tests
# ==================


class TestRequiredFields:
    """Tests that all agents have required fields."""

    def test_all_agents_have_type(self, agents_data):
        """Every agent has a type field."""
        for name, agent in agents_data.items():
            assert "type" in agent, f"Agent '{name}' missing 'type'"

    def test_all_agent_types_valid(self, agents_data):
        """All agent types are valid enum values."""
        valid_types = {t.value for t in AgentType}
        for name, agent in agents_data.items():
            assert agent["type"] in valid_types, (
                f"Agent '{name}' has invalid type '{agent['type']}'. "
                f"Valid: {valid_types}"
            )

    def test_all_agents_have_expertise(self, agents_data):
        """Every agent has an expertise field."""
        for name, agent in agents_data.items():
            assert "expertise" in agent and agent["expertise"], (
                f"Agent '{name}' missing or empty 'expertise'"
            )

    def test_all_agents_have_domain_knowledge(self, agents_data):
        """Every agent has a domain_knowledge field."""
        for name, agent in agents_data.items():
            assert "domain_knowledge" in agent and agent["domain_knowledge"], (
                f"Agent '{name}' missing or empty 'domain_knowledge'"
            )

    def test_all_agents_have_priority(self, agents_data):
        """Every agent has a valid priority (1-10)."""
        for name, agent in agents_data.items():
            priority = agent.get("priority", 0)
            assert 1 <= priority <= 10, (
                f"Agent '{name}' has invalid priority {priority} (must be 1-10)"
            )

    def test_all_agents_have_category(self, agents_data):
        """Every agent has a category field."""
        for name, agent in agents_data.items():
            assert "category" in agent and agent["category"], (
                f"Agent '{name}' missing or empty 'category'"
            )


# ==================
# Spec Completeness Tests
# ==================


class TestSpecCompleteness:
    """Tests that all agents from the design spec are present."""

    # Architecture agents
    def test_architecture_agents(self, agents_data):
        """Required architecture agents are present."""
        required = ["system_architect", "ui_architect", "data_architect", "api_architect"]
        for name in required:
            assert name in agents_data, f"Missing architecture agent: {name}"

    # Development agents
    def test_development_agents(self, agents_data):
        """Required development agents are present."""
        required = ["frontend_dev", "backend_dev", "ios_developer", "android_developer"]
        for name in required:
            assert name in agents_data, f"Missing development agent: {name}"

    # Quality agents
    def test_quality_agents(self, agents_data):
        """Required quality agents are present."""
        required = ["qa_engineer", "security_expert", "ada_expert", "performance_expert"]
        for name in required:
            assert name in agents_data, f"Missing quality agent: {name}"

    # Infrastructure agents
    def test_infrastructure_agents(self, agents_data):
        """Required infrastructure agents are present."""
        required = ["devops_engineer", "cloud_architect", "sre"]
        for name in required:
            assert name in agents_data, f"Missing infrastructure agent: {name}"

    # Domain specialists
    def test_domain_specialists(self, agents_data):
        """Required domain specialist agents are present."""
        required = [
            "ml_engineer", "analytics_expert", "seo_expert",
            "i18n_expert", "search_expert", "blockchain_expert",
            "ai_integration_expert",
        ]
        for name in required:
            assert name in agents_data, f"Missing domain specialist: {name}"

    # Business agents
    def test_business_agents(self, agents_data):
        """Required business agents are present."""
        required = ["product_manager", "ux_researcher", "ux_designer", "tech_writer"]
        for name in required:
            assert name in agents_data, f"Missing business agent: {name}"


# ==================
# Category Distribution Tests
# ==================


class TestCategoryDistribution:
    """Tests for proper category distribution."""

    def test_all_required_categories_present(self, loaded_registry):
        """All required categories have at least one agent."""
        required_categories = [
            "architecture", "development", "quality",
            "infrastructure", "domain", "business",
        ]
        for cat in required_categories:
            agents = loaded_registry.get_agents_by_category(cat)
            assert len(agents) > 0, f"No agents in category '{cat}'"

    def test_multiple_agents_per_major_category(self, loaded_registry):
        """Major categories have multiple agents."""
        for cat in ["architecture", "development", "quality"]:
            agents = loaded_registry.get_agents_by_category(cat)
            assert len(agents) >= 3, (
                f"Category '{cat}' has only {len(agents)} agents (expected >= 3)"
            )


# ==================
# Agent Type Tests
# ==================


class TestAgentTypes:
    """Tests for agent type configuration."""

    def test_code_agents_have_tools(self, agents_data):
        """Code agents have tools configured."""
        for name, agent in agents_data.items():
            if agent["type"] == "claude_sdk_code":
                assert "tools" in agent and len(agent["tools"]) > 0, (
                    f"Code agent '{name}' has no tools configured"
                )

    def test_code_agents_exist(self, agents_data):
        """At least one code agent is defined."""
        code_agents = [
            name for name, agent in agents_data.items()
            if agent["type"] == "claude_sdk_code"
        ]
        assert len(code_agents) >= 1, "No code agents defined"

    def test_code_agents_load_as_correct_type(self, loaded_registry):
        """Code agents are loaded as ClaudeCodeAgent instances."""
        code_agents = loaded_registry.get_agents_by_type(AgentType.CLAUDE_SDK_CODE)
        for agent in code_agents:
            assert isinstance(agent, ClaudeCodeAgent), (
                f"Agent '{agent.name}' should be ClaudeCodeAgent but is {type(agent)}"
            )

    def test_text_agents_load_as_correct_type(self, loaded_registry):
        """Text agents are loaded as ClaudeTextAgent instances."""
        text_agents = loaded_registry.get_agents_by_type(AgentType.CLAUDE_SDK_TEXT)
        for agent in text_agents:
            assert isinstance(agent, ClaudeTextAgent), (
                f"Agent '{agent.name}' should be ClaudeTextAgent but is {type(agent)}"
            )


# ==================
# Domain Knowledge Quality Tests
# ==================


class TestDomainKnowledgeQuality:
    """Tests for domain knowledge content quality."""

    def test_domain_knowledge_minimum_length(self, agents_data):
        """Domain knowledge is substantive (at least 50 characters)."""
        for name, agent in agents_data.items():
            dk = agent.get("domain_knowledge", "")
            assert len(dk) >= 50, (
                f"Agent '{name}' has very short domain_knowledge ({len(dk)} chars)"
            )

    def test_expertise_minimum_length(self, agents_data):
        """Expertise descriptions are substantive (at least 10 characters)."""
        for name, agent in agents_data.items():
            expertise = agent.get("expertise", "")
            assert len(expertise) >= 10, (
                f"Agent '{name}' has very short expertise ({len(expertise)} chars)"
            )

    def test_no_duplicate_expertise(self, agents_data):
        """No two agents have identical expertise."""
        expertise_map = {}
        for name, agent in agents_data.items():
            exp = agent.get("expertise", "").strip()
            if exp in expertise_map:
                pytest.fail(
                    f"Agents '{expertise_map[exp]}' and '{name}' have identical expertise"
                )
            expertise_map[exp] = name
