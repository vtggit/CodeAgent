"""
Agent Registry for the Multi-Agent GitHub Issue Routing System.

This module implements the AgentRegistry class that loads agent definitions
from YAML configuration files and manages the agent pool. It supports:

- Loading 50+ agent definitions from config/agent_definitions.yaml
- Factory pattern for creating agent instances based on type
- Agent retrieval by name, category, or type
- Configuration validation on load
- Hot reload capability for config changes
"""

import os
import time
from pathlib import Path
from typing import Any, Optional

import yaml

from src.agents.base import BaseAgent
from src.agents.claude_text_agent import ClaudeTextAgent
from src.models.agent import AgentConfig, AgentType, LLMProviderConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ==================
# Agent Type Registry
# ==================

# Maps AgentType enum values to their concrete implementation classes
AGENT_TYPE_MAP: dict[AgentType, type[BaseAgent]] = {
    AgentType.CLAUDE_SDK_TEXT: ClaudeTextAgent,
    # AgentType.CLAUDE_SDK_CODE: ClaudeCodeAgent,  # Future: Issue #11
    # AgentType.GITHUB_API: GitHubAPIAgent,         # Future
    # AgentType.CUSTOM: CustomAgent,                 # Future
}


class AgentRegistryError(Exception):
    """Base exception for agent registry errors."""

    pass


class AgentConfigError(AgentRegistryError):
    """Raised when an agent configuration is invalid."""

    pass


class AgentNotFoundError(AgentRegistryError):
    """Raised when a requested agent is not found in the registry."""

    pass


class AgentRegistry:
    """
    Registry for managing the pool of specialist agents.

    Loads agent definitions from YAML configuration files, validates them,
    and creates agent instances using the factory pattern. Supports hot reload
    for development workflows.

    Usage:
        registry = AgentRegistry()
        registry.load_from_yaml("config/agent_definitions.yaml")

        agent = registry.get_agent("ui_architect")
        all_agents = registry.list_agents()
        arch_agents = registry.get_agents_by_category("architecture")
    """

    def __init__(self) -> None:
        """Initialize an empty agent registry."""
        self._agents: dict[str, BaseAgent] = {}
        self._configs: dict[str, AgentConfig] = {}
        self._config_path: Optional[str] = None
        self._last_load_time: Optional[float] = None
        self._load_errors: list[str] = []
        logger.info("AgentRegistry initialized")

    # ==================
    # Loading Methods
    # ==================

    def load_from_yaml(self, config_path: str) -> int:
        """
        Load agent definitions from a YAML configuration file.

        Parses the YAML file, validates each agent configuration,
        and creates agent instances using the appropriate factory.

        Args:
            config_path: Path to the agent_definitions.yaml file

        Returns:
            Number of agents successfully loaded

        Raises:
            AgentRegistryError: If the file cannot be read or parsed
            AgentConfigError: If no valid agent definitions are found
        """
        path = Path(config_path)
        if not path.exists():
            raise AgentRegistryError(f"Configuration file not found: {config_path}")

        if not path.suffix in (".yaml", ".yml"):
            raise AgentRegistryError(
                f"Configuration file must be YAML (.yaml or .yml): {config_path}"
            )

        logger.info("Loading agent definitions from: %s", config_path)

        try:
            with open(path, "r") as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise AgentRegistryError(f"Failed to parse YAML file: {e}") from e
        except OSError as e:
            raise AgentRegistryError(f"Failed to read configuration file: {e}") from e

        if not raw_config or not isinstance(raw_config, dict):
            raise AgentConfigError("Configuration file is empty or invalid")

        agents_data = raw_config.get("agents", {})
        if not agents_data:
            raise AgentConfigError("No 'agents' section found in configuration")

        # Clear existing agents for a clean load
        self._agents.clear()
        self._configs.clear()
        self._load_errors.clear()
        self._config_path = str(path.resolve())

        loaded_count = 0
        for agent_name, agent_data in agents_data.items():
            try:
                config = self._parse_agent_config(agent_name, agent_data)
                agent = self._create_agent(config)
                self._agents[agent_name] = agent
                self._configs[agent_name] = config
                loaded_count += 1
            except Exception as e:
                error_msg = f"Failed to load agent '{agent_name}': {e}"
                self._load_errors.append(error_msg)
                logger.warning(error_msg)

        self._last_load_time = time.time()

        if loaded_count == 0:
            raise AgentConfigError(
                f"No agents could be loaded. Errors: {'; '.join(self._load_errors)}"
            )

        logger.info(
            "Loaded %d agents from %s (%d errors)",
            loaded_count,
            config_path,
            len(self._load_errors),
        )

        return loaded_count

    def load_from_dict(self, agents_data: dict[str, dict[str, Any]]) -> int:
        """
        Load agent definitions from a dictionary.

        Useful for testing and programmatic agent creation.

        Args:
            agents_data: Dictionary mapping agent names to their config dicts

        Returns:
            Number of agents successfully loaded
        """
        self._agents.clear()
        self._configs.clear()
        self._load_errors.clear()

        loaded_count = 0
        for agent_name, agent_data in agents_data.items():
            try:
                config = self._parse_agent_config(agent_name, agent_data)
                agent = self._create_agent(config)
                self._agents[agent_name] = agent
                self._configs[agent_name] = config
                loaded_count += 1
            except Exception as e:
                error_msg = f"Failed to load agent '{agent_name}': {e}"
                self._load_errors.append(error_msg)
                logger.warning(error_msg)

        self._last_load_time = time.time()
        logger.info("Loaded %d agents from dict (%d errors)", loaded_count, len(self._load_errors))
        return loaded_count

    # ==================
    # Retrieval Methods
    # ==================

    def get_agent(self, name: str) -> BaseAgent:
        """
        Retrieve an agent by name.

        Args:
            name: The unique agent identifier

        Returns:
            The agent instance

        Raises:
            AgentNotFoundError: If no agent with that name exists
        """
        agent = self._agents.get(name)
        if agent is None:
            available = ", ".join(sorted(self._agents.keys())[:10])
            raise AgentNotFoundError(
                f"Agent '{name}' not found. Available agents: {available}..."
            )
        return agent

    def get_agent_config(self, name: str) -> AgentConfig:
        """
        Retrieve an agent's configuration by name.

        Args:
            name: The unique agent identifier

        Returns:
            The agent's configuration

        Raises:
            AgentNotFoundError: If no agent with that name exists
        """
        config = self._configs.get(name)
        if config is None:
            raise AgentNotFoundError(f"Agent config '{name}' not found")
        return config

    def list_agents(self) -> list[str]:
        """
        List all registered agent names.

        Returns:
            Sorted list of agent name strings
        """
        return sorted(self._agents.keys())

    def get_all_agents(self) -> list[BaseAgent]:
        """
        Get all registered agent instances.

        Returns:
            List of all agent instances, sorted by name
        """
        return [self._agents[name] for name in sorted(self._agents.keys())]

    def get_agents_by_category(self, category: str) -> list[BaseAgent]:
        """
        Get all agents in a given category.

        Args:
            category: Category name (architecture, development, quality, domain, business, infrastructure)

        Returns:
            List of agents in the specified category, sorted by priority (highest first)
        """
        agents = [
            agent for agent in self._agents.values()
            if agent.category == category
        ]
        return sorted(agents, key=lambda a: a.priority, reverse=True)

    def get_agents_by_type(self, agent_type: AgentType) -> list[BaseAgent]:
        """
        Get all agents of a given type.

        Args:
            agent_type: The agent type enum value

        Returns:
            List of agents of the specified type
        """
        return [
            agent for agent in self._agents.values()
            if agent.agent_type == agent_type
        ]

    def get_agents_by_priority(self, min_priority: int = 1) -> list[BaseAgent]:
        """
        Get agents with priority at or above the minimum threshold.

        Args:
            min_priority: Minimum priority level (1-10)

        Returns:
            List of agents meeting the priority threshold, sorted by priority descending
        """
        agents = [
            agent for agent in self._agents.values()
            if agent.priority >= min_priority
        ]
        return sorted(agents, key=lambda a: a.priority, reverse=True)

    def has_agent(self, name: str) -> bool:
        """Check if an agent exists in the registry."""
        return name in self._agents

    @property
    def agent_count(self) -> int:
        """Number of agents in the registry."""
        return len(self._agents)

    @property
    def categories(self) -> list[str]:
        """List of unique categories across all agents."""
        cats = set()
        for agent in self._agents.values():
            if agent.category:
                cats.add(agent.category)
        return sorted(cats)

    # ==================
    # Hot Reload
    # ==================

    def reload(self) -> int:
        """
        Reload agent definitions from the previously loaded config file.

        Useful for development workflows where agent definitions change
        without restarting the server.

        Returns:
            Number of agents loaded after reload

        Raises:
            AgentRegistryError: If no config file was previously loaded
        """
        if not self._config_path:
            raise AgentRegistryError(
                "No configuration file has been loaded yet. Call load_from_yaml() first."
            )

        logger.info("Reloading agent definitions from: %s", self._config_path)
        return self.load_from_yaml(self._config_path)

    def needs_reload(self) -> bool:
        """
        Check if the config file has been modified since last load.

        Returns:
            True if the file has been modified and should be reloaded
        """
        if not self._config_path or not self._last_load_time:
            return False

        try:
            file_mtime = os.path.getmtime(self._config_path)
            return file_mtime > self._last_load_time
        except OSError:
            return False

    # ==================
    # Info & Status
    # ==================

    @property
    def load_errors(self) -> list[str]:
        """List of errors encountered during the last load."""
        return list(self._load_errors)

    @property
    def last_load_time(self) -> Optional[float]:
        """Timestamp of the last successful load."""
        return self._last_load_time

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of the registry state.

        Returns:
            Dictionary with agent counts, categories, and status
        """
        category_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}

        for agent in self._agents.values():
            cat = agent.category or "uncategorized"
            category_counts[cat] = category_counts.get(cat, 0) + 1

            type_key = agent.agent_type.value
            type_counts[type_key] = type_counts.get(type_key, 0) + 1

        return {
            "total_agents": self.agent_count,
            "categories": category_counts,
            "types": type_counts,
            "load_errors": len(self._load_errors),
            "config_path": self._config_path,
            "last_load_time": self._last_load_time,
        }

    # ==================
    # Internal Methods
    # ==================

    def _parse_agent_config(
        self, name: str, data: dict[str, Any]
    ) -> AgentConfig:
        """
        Parse raw YAML data into a validated AgentConfig.

        Args:
            name: Agent name (used as the key in YAML)
            data: Raw configuration dictionary from YAML

        Returns:
            Validated AgentConfig instance

        Raises:
            AgentConfigError: If the configuration is invalid
        """
        if not isinstance(data, dict):
            raise AgentConfigError(
                f"Agent '{name}' configuration must be a dictionary, got {type(data).__name__}"
            )

        # Map YAML type string to AgentType enum
        type_str = data.get("type", "claude_sdk_text")
        try:
            agent_type = AgentType(type_str)
        except ValueError:
            valid_types = [t.value for t in AgentType]
            raise AgentConfigError(
                f"Agent '{name}' has invalid type '{type_str}'. "
                f"Valid types: {valid_types}"
            )

        # Build LLM config if present
        llm_data = data.get("llm", {})
        llm_config = LLMProviderConfig(
            provider=llm_data.get("provider", "anthropic"),
            model=llm_data.get("model", "claude-sonnet-4-5-20250929"),
            base_url=llm_data.get("base_url"),
            temperature=llm_data.get("temperature", 0.7),
            max_tokens=llm_data.get("max_tokens", 4096),
        )

        try:
            config = AgentConfig(
                name=name,
                type=agent_type,
                expertise=data.get("expertise", ""),
                domain_knowledge=data.get("domain_knowledge", ""),
                system_prompt=data.get("system_prompt"),
                llm=llm_config,
                tools=data.get("tools", []),
                can_edit_files=data.get("can_edit_files", False),
                priority=data.get("priority", 5),
                category=data.get("category"),
            )
        except Exception as e:
            raise AgentConfigError(
                f"Agent '{name}' has invalid configuration: {e}"
            ) from e

        return config

    def _create_agent(self, config: AgentConfig) -> BaseAgent:
        """
        Create an agent instance using the factory pattern.

        Maps the agent type to its concrete class and instantiates it.

        Args:
            config: Validated agent configuration

        Returns:
            Concrete agent instance

        Raises:
            AgentConfigError: If the agent type is not supported
        """
        agent_class = AGENT_TYPE_MAP.get(config.type)

        if agent_class is None:
            # For unsupported types, fall back to text agent with a warning
            logger.warning(
                "Agent type '%s' is not yet implemented for '%s'. "
                "Falling back to ClaudeTextAgent.",
                config.type.value,
                config.name,
            )
            agent_class = ClaudeTextAgent

        return agent_class.from_config(config)

    # ==================
    # Dunder Methods
    # ==================

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        return name in self._agents

    def __getitem__(self, name: str) -> BaseAgent:
        return self.get_agent(name)

    def __iter__(self):
        return iter(self._agents.values())

    def __repr__(self) -> str:
        return f"<AgentRegistry(agents={self.agent_count}, categories={self.categories})>"
