"""
Tests for the AgentRegistry class.

Tests cover:
- Loading from YAML configuration files
- Loading from dictionaries
- Agent retrieval by name, category, type, and priority
- Configuration validation and error handling
- Hot reload capability
- Registry summary and metadata
"""

import os
import tempfile
import time

import pytest
import yaml

from src.agents.base import BaseAgent
from src.agents.claude_text_agent import ClaudeTextAgent
from src.agents.registry import (
    AgentConfigError,
    AgentNotFoundError,
    AgentRegistry,
    AgentRegistryError,
)
from src.models.agent import AgentConfig, AgentType


# ==================
# Fixtures
# ==================


@pytest.fixture
def sample_agents_yaml():
    """Create a temporary YAML config with sample agent definitions."""
    config = {
        "agents": {
            "test_architect": {
                "type": "claude_sdk_text",
                "expertise": "System architecture and design",
                "domain_knowledge": "Microservices, scalability, design patterns",
                "priority": 9,
                "category": "architecture",
            },
            "test_developer": {
                "type": "claude_sdk_text",
                "expertise": "Backend development",
                "domain_knowledge": "Python, FastAPI, databases",
                "priority": 7,
                "category": "development",
            },
            "test_security": {
                "type": "claude_sdk_text",
                "expertise": "Application security",
                "domain_knowledge": "OWASP, XSS, CSRF, authentication",
                "priority": 8,
                "category": "quality",
            },
            "test_qa": {
                "type": "claude_sdk_text",
                "expertise": "Quality assurance",
                "domain_knowledge": "Testing, automation, pytest",
                "priority": 6,
                "category": "quality",
            },
            "test_product": {
                "type": "claude_sdk_text",
                "expertise": "Product management",
                "domain_knowledge": "Requirements, user stories",
                "priority": 5,
                "category": "business",
            },
        }
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(config, f)
        f.flush()
        yield f.name

    os.unlink(f.name)


@pytest.fixture
def sample_agents_dict():
    """Sample agent definitions as a dictionary."""
    return {
        "agent_alpha": {
            "type": "claude_sdk_text",
            "expertise": "Alpha expertise",
            "domain_knowledge": "Alpha knowledge",
            "priority": 8,
            "category": "architecture",
        },
        "agent_beta": {
            "type": "claude_sdk_text",
            "expertise": "Beta expertise",
            "domain_knowledge": "Beta knowledge",
            "priority": 6,
            "category": "development",
        },
        "agent_gamma": {
            "type": "claude_sdk_text",
            "expertise": "Gamma expertise",
            "domain_knowledge": "Gamma knowledge",
            "priority": 4,
            "category": "development",
        },
    }


@pytest.fixture
def registry():
    """Create an empty AgentRegistry."""
    return AgentRegistry()


@pytest.fixture
def loaded_registry(sample_agents_yaml, registry):
    """Create an AgentRegistry loaded with sample agents."""
    registry.load_from_yaml(sample_agents_yaml)
    return registry


# ==================
# Loading Tests
# ==================


class TestLoadFromYaml:
    """Tests for loading agent definitions from YAML files."""

    def test_load_valid_yaml(self, sample_agents_yaml, registry):
        """Load a valid YAML file and verify all agents are instantiated."""
        count = registry.load_from_yaml(sample_agents_yaml)
        assert count == 5
        assert registry.agent_count == 5

    def test_load_returns_agent_count(self, sample_agents_yaml, registry):
        """load_from_yaml returns the number of agents loaded."""
        count = registry.load_from_yaml(sample_agents_yaml)
        assert count == 5

    def test_load_sets_config_path(self, sample_agents_yaml, registry):
        """After loading, config_path is set."""
        registry.load_from_yaml(sample_agents_yaml)
        assert registry._config_path is not None
        assert registry._config_path.endswith(".yaml")

    def test_load_sets_last_load_time(self, sample_agents_yaml, registry):
        """After loading, last_load_time is set."""
        before = time.time()
        registry.load_from_yaml(sample_agents_yaml)
        after = time.time()
        assert registry.last_load_time is not None
        assert before <= registry.last_load_time <= after

    def test_load_nonexistent_file_raises(self, registry):
        """Loading a nonexistent file raises AgentRegistryError."""
        with pytest.raises(AgentRegistryError, match="not found"):
            registry.load_from_yaml("/nonexistent/path/agents.yaml")

    def test_load_non_yaml_file_raises(self, registry):
        """Loading a non-YAML file raises AgentRegistryError."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not yaml")
            f.flush()
            try:
                with pytest.raises(AgentRegistryError, match="YAML"):
                    registry.load_from_yaml(f.name)
            finally:
                os.unlink(f.name)

    def test_load_invalid_yaml_raises(self, registry):
        """Loading an invalid YAML file raises AgentRegistryError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("invalid: yaml: content: [[[")
            f.flush()
            try:
                with pytest.raises(AgentRegistryError, match="Failed to parse"):
                    registry.load_from_yaml(f.name)
            finally:
                os.unlink(f.name)

    def test_load_empty_yaml_raises(self, registry):
        """Loading an empty YAML file raises AgentConfigError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")
            f.flush()
            try:
                with pytest.raises(AgentConfigError, match="empty"):
                    registry.load_from_yaml(f.name)
            finally:
                os.unlink(f.name)

    def test_load_yaml_without_agents_section_raises(self, registry):
        """Loading YAML without 'agents' key raises AgentConfigError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump({"other_key": "value"}, f)
            f.flush()
            try:
                with pytest.raises(AgentConfigError, match="No 'agents' section"):
                    registry.load_from_yaml(f.name)
            finally:
                os.unlink(f.name)

    def test_load_replaces_previous_agents(self, sample_agents_dict, registry):
        """Loading a new config replaces all previous agents."""
        registry.load_from_dict(sample_agents_dict)
        assert registry.agent_count == 3

        # Load a different set
        new_agents = {
            "new_agent": {
                "type": "claude_sdk_text",
                "expertise": "New expertise",
            }
        }
        registry.load_from_dict(new_agents)
        assert registry.agent_count == 1
        assert "new_agent" in registry
        assert "agent_alpha" not in registry

    def test_load_yml_extension(self, registry):
        """Loading a .yml file works the same as .yaml."""
        config = {
            "agents": {
                "test_agent": {
                    "type": "claude_sdk_text",
                    "expertise": "Test",
                }
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            yaml.dump(config, f)
            f.flush()
            try:
                count = registry.load_from_yaml(f.name)
                assert count == 1
            finally:
                os.unlink(f.name)


class TestLoadFromDict:
    """Tests for loading agent definitions from dictionaries."""

    def test_load_valid_dict(self, sample_agents_dict, registry):
        """Load a valid dictionary of agents."""
        count = registry.load_from_dict(sample_agents_dict)
        assert count == 3
        assert registry.agent_count == 3

    def test_load_empty_dict(self, registry):
        """Loading an empty dict results in zero agents."""
        count = registry.load_from_dict({})
        assert count == 0
        assert registry.agent_count == 0

    def test_load_with_invalid_agent_continues(self, registry):
        """Invalid agents are skipped but others still load."""
        agents = {
            "valid_agent": {
                "type": "claude_sdk_text",
                "expertise": "Valid expertise",
            },
            "invalid_agent": "not a dict",
        }
        count = registry.load_from_dict(agents)
        assert count == 1
        assert "valid_agent" in registry
        assert "invalid_agent" not in registry
        assert len(registry.load_errors) == 1


class TestLoadWithLLMConfig:
    """Tests for agents with custom LLM configurations."""

    def test_agent_with_custom_llm(self, registry):
        """Agent with custom LLM provider settings."""
        agents = {
            "custom_llm_agent": {
                "type": "claude_sdk_text",
                "expertise": "Custom LLM",
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.3,
                    "max_tokens": 2048,
                },
            }
        }
        registry.load_from_dict(agents)
        agent = registry.get_agent("custom_llm_agent")
        assert agent.llm_provider.provider == "openai"
        assert agent.llm_provider.model == "gpt-4"
        assert agent.llm_provider.temperature == 0.3
        assert agent.llm_provider.max_tokens == 2048

    def test_agent_with_default_llm(self, registry):
        """Agent without LLM config gets defaults."""
        agents = {
            "default_llm_agent": {
                "type": "claude_sdk_text",
                "expertise": "Default LLM",
            }
        }
        registry.load_from_dict(agents)
        agent = registry.get_agent("default_llm_agent")
        assert agent.llm_provider.provider == "anthropic"
        assert agent.llm_provider.model == "claude-sonnet-4-5-20250929"


# ==================
# Retrieval Tests
# ==================


class TestGetAgent:
    """Tests for retrieving agents by name."""

    def test_get_existing_agent(self, loaded_registry):
        """Retrieve an agent that exists."""
        agent = loaded_registry.get_agent("test_architect")
        assert agent is not None
        assert agent.name == "test_architect"
        assert isinstance(agent, BaseAgent)

    def test_get_agent_returns_correct_type(self, loaded_registry):
        """Retrieved agent has the correct implementation type."""
        agent = loaded_registry.get_agent("test_architect")
        assert isinstance(agent, ClaudeTextAgent)

    def test_get_agent_has_correct_expertise(self, loaded_registry):
        """Retrieved agent has the expertise from config."""
        agent = loaded_registry.get_agent("test_architect")
        assert agent.expertise == "System architecture and design"

    def test_get_nonexistent_agent_raises(self, loaded_registry):
        """Requesting a nonexistent agent raises AgentNotFoundError."""
        with pytest.raises(AgentNotFoundError, match="not found"):
            loaded_registry.get_agent("nonexistent_agent")

    def test_get_agent_config(self, loaded_registry):
        """Retrieve an agent's configuration."""
        config = loaded_registry.get_agent_config("test_architect")
        assert isinstance(config, AgentConfig)
        assert config.name == "test_architect"
        assert config.priority == 9

    def test_get_nonexistent_config_raises(self, loaded_registry):
        """Requesting a nonexistent agent config raises."""
        with pytest.raises(AgentNotFoundError):
            loaded_registry.get_agent_config("nonexistent")


class TestListAgents:
    """Tests for listing agents."""

    def test_list_agents_returns_sorted_names(self, loaded_registry):
        """list_agents returns sorted agent names."""
        names = loaded_registry.list_agents()
        assert names == sorted(names)
        assert len(names) == 5

    def test_list_agents_contains_all(self, loaded_registry):
        """list_agents includes all loaded agents."""
        names = loaded_registry.list_agents()
        assert "test_architect" in names
        assert "test_developer" in names
        assert "test_security" in names

    def test_list_agents_empty_registry(self, registry):
        """list_agents returns empty list for empty registry."""
        assert registry.list_agents() == []

    def test_get_all_agents(self, loaded_registry):
        """get_all_agents returns all agent instances."""
        agents = loaded_registry.get_all_agents()
        assert len(agents) == 5
        assert all(isinstance(a, BaseAgent) for a in agents)


class TestGetByCategory:
    """Tests for filtering agents by category."""

    def test_get_agents_by_category(self, loaded_registry):
        """Get agents in a specific category."""
        quality_agents = loaded_registry.get_agents_by_category("quality")
        assert len(quality_agents) == 2
        names = [a.name for a in quality_agents]
        assert "test_security" in names
        assert "test_qa" in names

    def test_get_agents_by_category_sorted_by_priority(self, loaded_registry):
        """Agents in a category are sorted by priority (highest first)."""
        quality_agents = loaded_registry.get_agents_by_category("quality")
        priorities = [a.priority for a in quality_agents]
        assert priorities == sorted(priorities, reverse=True)

    def test_get_agents_by_nonexistent_category(self, loaded_registry):
        """Non-existent category returns empty list."""
        agents = loaded_registry.get_agents_by_category("nonexistent")
        assert agents == []

    def test_categories_property(self, loaded_registry):
        """categories property returns sorted unique categories."""
        cats = loaded_registry.categories
        assert "architecture" in cats
        assert "development" in cats
        assert "quality" in cats
        assert "business" in cats
        assert cats == sorted(cats)


class TestGetByType:
    """Tests for filtering agents by type."""

    def test_get_agents_by_type(self, loaded_registry):
        """Get all agents of a given type."""
        text_agents = loaded_registry.get_agents_by_type(AgentType.CLAUDE_SDK_TEXT)
        assert len(text_agents) == 5  # All are text agents

    def test_get_agents_by_unused_type(self, loaded_registry):
        """Get agents of a type with no instances."""
        code_agents = loaded_registry.get_agents_by_type(AgentType.GITHUB_API)
        assert len(code_agents) == 0


class TestGetByPriority:
    """Tests for filtering agents by priority."""

    def test_get_high_priority_agents(self, loaded_registry):
        """Get agents above a priority threshold."""
        high_priority = loaded_registry.get_agents_by_priority(min_priority=8)
        assert len(high_priority) == 2
        names = [a.name for a in high_priority]
        assert "test_architect" in names
        assert "test_security" in names

    def test_get_all_agents_by_priority(self, loaded_registry):
        """min_priority=1 returns all agents."""
        all_agents = loaded_registry.get_agents_by_priority(min_priority=1)
        assert len(all_agents) == 5

    def test_agents_sorted_by_priority_descending(self, loaded_registry):
        """Priority filtering returns agents sorted by priority descending."""
        agents = loaded_registry.get_agents_by_priority(min_priority=1)
        priorities = [a.priority for a in agents]
        assert priorities == sorted(priorities, reverse=True)


class TestHasAgent:
    """Tests for checking agent existence."""

    def test_has_existing_agent(self, loaded_registry):
        """has_agent returns True for existing agents."""
        assert loaded_registry.has_agent("test_architect")

    def test_has_nonexistent_agent(self, loaded_registry):
        """has_agent returns False for nonexistent agents."""
        assert not loaded_registry.has_agent("nonexistent")


# ==================
# Hot Reload Tests
# ==================


class TestHotReload:
    """Tests for hot reload capability."""

    def test_reload_reloads_same_file(self, sample_agents_yaml, registry):
        """reload() re-reads the same config file."""
        registry.load_from_yaml(sample_agents_yaml)
        assert registry.agent_count == 5

        # Reload
        count = registry.reload()
        assert count == 5

    def test_reload_without_load_raises(self, registry):
        """reload() raises if no file was previously loaded."""
        with pytest.raises(AgentRegistryError, match="No configuration file"):
            registry.reload()

    def test_reload_picks_up_changes(self, registry):
        """reload() picks up changes to the config file."""
        config = {
            "agents": {
                "original_agent": {
                    "type": "claude_sdk_text",
                    "expertise": "Original",
                }
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config, f)
            f.flush()
            config_path = f.name

        try:
            registry.load_from_yaml(config_path)
            assert registry.agent_count == 1
            assert "original_agent" in registry

            # Modify the file
            new_config = {
                "agents": {
                    "new_agent": {
                        "type": "claude_sdk_text",
                        "expertise": "New",
                    },
                    "another_agent": {
                        "type": "claude_sdk_text",
                        "expertise": "Another",
                    },
                }
            }
            with open(config_path, "w") as f:
                yaml.dump(new_config, f)

            # Reload and verify changes
            count = registry.reload()
            assert count == 2
            assert "new_agent" in registry
            assert "another_agent" in registry
            assert "original_agent" not in registry
        finally:
            os.unlink(config_path)

    def test_needs_reload_false_initially(self, registry):
        """needs_reload returns False before any load."""
        assert not registry.needs_reload()

    def test_needs_reload_after_file_change(self, registry):
        """needs_reload returns True after file modification."""
        config = {
            "agents": {
                "test": {
                    "type": "claude_sdk_text",
                    "expertise": "Test",
                }
            }
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config, f)
            f.flush()
            config_path = f.name

        try:
            registry.load_from_yaml(config_path)
            assert not registry.needs_reload()

            # Modify the file (touch it with a future mtime)
            time.sleep(0.1)
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            assert registry.needs_reload()
        finally:
            os.unlink(config_path)


# ==================
# Validation Tests
# ==================


class TestConfigValidation:
    """Tests for agent configuration validation."""

    def test_invalid_type_string(self, registry):
        """Agent with invalid type string is skipped with error."""
        agents = {
            "bad_type": {
                "type": "invalid_type",
                "expertise": "Test",
            }
        }
        count = registry.load_from_dict(agents)
        assert count == 0
        assert len(registry.load_errors) == 1

    def test_non_dict_agent_data(self, registry):
        """Agent with non-dict configuration is skipped."""
        agents = {
            "bad_agent": "just a string",
        }
        count = registry.load_from_dict(agents)
        assert count == 0
        assert len(registry.load_errors) == 1

    def test_agent_with_code_type_creates_code_agent(self, registry):
        """Agent with claude_sdk_code type creates ClaudeCodeAgent."""
        from src.agents.claude_code_agent import ClaudeCodeAgent

        agents = {
            "code_agent": {
                "type": "claude_sdk_code",
                "expertise": "Code editing",
                "tools": ["Read", "Grep", "Glob"],
                "can_edit_files": True,
            }
        }
        count = registry.load_from_dict(agents)
        assert count == 1
        agent = registry.get_agent("code_agent")
        assert isinstance(agent, ClaudeCodeAgent)
        assert agent.can_edit_files is True


# ==================
# Summary Tests
# ==================


class TestRegistrySummary:
    """Tests for registry summary and metadata."""

    def test_get_summary(self, loaded_registry):
        """get_summary returns correct metadata."""
        summary = loaded_registry.get_summary()
        assert summary["total_agents"] == 5
        assert "architecture" in summary["categories"]
        assert "development" in summary["categories"]
        assert "quality" in summary["categories"]
        assert summary["categories"]["quality"] == 2
        assert "claude_sdk_text" in summary["types"]
        assert summary["types"]["claude_sdk_text"] == 5
        assert summary["load_errors"] == 0

    def test_empty_registry_summary(self, registry):
        """Summary of empty registry shows zero agents."""
        summary = registry.get_summary()
        assert summary["total_agents"] == 0
        assert summary["categories"] == {}


# ==================
# Dunder Method Tests
# ==================


class TestDunderMethods:
    """Tests for __len__, __contains__, __getitem__, __iter__."""

    def test_len(self, loaded_registry):
        """len() returns agent count."""
        assert len(loaded_registry) == 5

    def test_contains(self, loaded_registry):
        """'in' operator checks agent existence."""
        assert "test_architect" in loaded_registry
        assert "nonexistent" not in loaded_registry

    def test_getitem(self, loaded_registry):
        """Bracket notation retrieves agents."""
        agent = loaded_registry["test_architect"]
        assert agent.name == "test_architect"

    def test_getitem_nonexistent_raises(self, loaded_registry):
        """Bracket notation for nonexistent agent raises."""
        with pytest.raises(AgentNotFoundError):
            _ = loaded_registry["nonexistent"]

    def test_iter(self, loaded_registry):
        """Registry is iterable over agents."""
        agents = list(loaded_registry)
        assert len(agents) == 5
        assert all(isinstance(a, BaseAgent) for a in agents)

    def test_repr(self, loaded_registry):
        """repr() shows useful info."""
        r = repr(loaded_registry)
        assert "AgentRegistry" in r
        assert "agents=5" in r


# ==================
# Production YAML Tests
# ==================


class TestProductionYaml:
    """Tests for loading the actual production agent_definitions.yaml."""

    def test_load_production_config(self, registry):
        """Load the actual production agent_definitions.yaml."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config",
            "agent_definitions.yaml",
        )
        if not os.path.exists(config_path):
            pytest.skip("Production config not found")

        count = registry.load_from_yaml(config_path)
        # The production config has 40+ agents
        assert count >= 40
        assert registry.agent_count >= 40

    def test_production_config_has_expected_categories(self, registry):
        """Production config contains all expected categories."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config",
            "agent_definitions.yaml",
        )
        if not os.path.exists(config_path):
            pytest.skip("Production config not found")

        registry.load_from_yaml(config_path)
        cats = registry.categories
        assert "architecture" in cats
        assert "development" in cats
        assert "quality" in cats
        assert "domain" in cats
        assert "business" in cats

    def test_production_config_all_agents_valid(self, registry):
        """All agents in production config are valid instances."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config",
            "agent_definitions.yaml",
        )
        if not os.path.exists(config_path):
            pytest.skip("Production config not found")

        registry.load_from_yaml(config_path)
        assert len(registry.load_errors) == 0

        for agent in registry:
            assert isinstance(agent, BaseAgent)
            assert agent.name
            assert agent.expertise
