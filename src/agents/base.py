"""
Base agent class for the Multi-Agent GitHub Issue Routing System.

This module defines the abstract BaseAgent interface that all specialist
agents must inherit from. It provides the contract for agent participation
in multi-round deliberation workflows.

Each agent must implement:
- should_comment(): Decide whether to participate in the current round
- generate_comment(): Generate a comment for the deliberation

The base class provides:
- Common logging infrastructure
- Serialization/deserialization
- Configuration access
- Conversation context helpers
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from src.models.agent import AgentConfig, AgentDecision, AgentType, LLMProviderConfig
from src.models.workflow import Comment, WorkflowInstance
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all specialist agents.

    Defines the interface for agent participation in deliberation workflows.
    Each agent independently decides when to comment and generates domain-specific
    insights based on the issue and conversation history.

    Attributes:
        name: Unique identifier for this agent
        expertise: Short description of the agent's area of expertise
        domain_knowledge: Detailed knowledge areas, technologies, and skills
        agent_type: Implementation type (CLAUDE_SDK_TEXT, CLAUDE_SDK_CODE, etc.)
        config: Full agent configuration
    """

    def __init__(self, config: AgentConfig) -> None:
        """
        Initialize a base agent from configuration.

        Args:
            config: Agent configuration containing name, expertise, LLM settings, etc.
        """
        self._config = config
        self._logger = get_logger(__name__, agent_name=config.name)
        self._logger.info(
            "agent_initialized",
            agent_type=config.type.value,
            expertise=config.expertise[:60],
        )

    # ==================
    # Required Properties
    # ==================

    @property
    def name(self) -> str:
        """Unique agent identifier."""
        return self._config.name

    @property
    def expertise(self) -> str:
        """Short description of the agent's area of expertise."""
        return self._config.expertise

    @property
    def domain_knowledge(self) -> str:
        """Detailed knowledge areas, technologies, and skills."""
        return self._config.domain_knowledge

    @property
    def agent_type(self) -> AgentType:
        """Agent implementation type."""
        return self._config.type

    # ==================
    # Optional Properties
    # ==================

    @property
    def model(self) -> str:
        """LLM model identifier."""
        return self._config.llm.model

    @property
    def tools(self) -> list[str]:
        """List of tools this agent can use."""
        return self._config.tools

    @property
    def llm_provider(self) -> LLMProviderConfig:
        """LLM provider configuration."""
        return self._config.llm

    @property
    def config(self) -> AgentConfig:
        """Full agent configuration."""
        return self._config

    @property
    def priority(self) -> int:
        """Agent priority for selection (1=lowest, 10=highest)."""
        return self._config.priority

    @property
    def category(self) -> Optional[str]:
        """Agent category (architecture, development, quality, domain, business)."""
        return self._config.category

    @property
    def system_prompt(self) -> Optional[str]:
        """Custom system prompt override for this agent."""
        return self._config.system_prompt

    @property
    def can_edit_files(self) -> bool:
        """Whether this agent can edit files."""
        return self._config.can_edit_files

    # ==================
    # Abstract Methods
    # ==================

    @abstractmethod
    async def should_comment(
        self,
        workflow: WorkflowInstance,
        current_round: int,
    ) -> AgentDecision:
        """
        Decide whether to comment in the current round.

        Each agent evaluates the issue and conversation history to determine
        if they have valuable input to contribute. This allows agents to
        self-select into or out of each round based on relevance.

        Args:
            workflow: Current workflow state including issue and conversation history
            current_round: The current deliberation round number (1-based)

        Returns:
            AgentDecision: Decision object with should_comment, reason,
                          responding_to (list of agent names), and confidence
        """
        ...

    @abstractmethod
    async def generate_comment(
        self,
        workflow: WorkflowInstance,
        current_round: int,
        decision: AgentDecision,
    ) -> Comment:
        """
        Generate a comment for the current round.

        Called only when should_comment() returns True. The agent should
        provide domain-specific analysis, recommendations, or responses
        to other agents' comments.

        Args:
            workflow: Current workflow state including issue and conversation history
            current_round: The current deliberation round number (1-based)
            decision: The decision that led to commenting (includes context)

        Returns:
            Comment: The generated comment with text, references, and metadata
        """
        ...

    # ==================
    # Helper Methods
    # ==================

    def get_issue_context(self, workflow: WorkflowInstance) -> dict[str, Any]:
        """
        Extract relevant issue context from the workflow.

        Provides a convenient summary of the issue for agent evaluation.

        Args:
            workflow: Current workflow state

        Returns:
            Dictionary with issue title, body, labels, and metadata
        """
        issue = workflow.github_issue or {}
        return {
            "title": issue.get("title", ""),
            "body": issue.get("body", ""),
            "labels": issue.get("labels", []),
            "number": issue.get("number"),
            "state": issue.get("state", "open"),
            "url": issue.get("url", issue.get("html_url", "")),
        }

    def get_conversation_history(
        self,
        workflow: WorkflowInstance,
        max_rounds: Optional[int] = None,
    ) -> list[Comment]:
        """
        Get the conversation history, optionally limited to recent rounds.

        Args:
            workflow: Current workflow state
            max_rounds: If set, only return comments from the last N rounds

        Returns:
            List of comments in chronological order
        """
        history = workflow.conversation_history
        if max_rounds is not None and max_rounds > 0:
            min_round = max(1, workflow.current_round - max_rounds + 1)
            history = [c for c in history if c.round >= min_round]
        return history

    def get_round_comments(
        self,
        workflow: WorkflowInstance,
        round_number: int,
    ) -> list[Comment]:
        """
        Get all comments from a specific round.

        Args:
            workflow: Current workflow state
            round_number: The round to get comments from

        Returns:
            List of comments from the specified round
        """
        return [
            c for c in workflow.conversation_history
            if c.round == round_number
        ]

    def get_comments_by_agent(
        self,
        workflow: WorkflowInstance,
        agent_name: str,
    ) -> list[Comment]:
        """
        Get all comments by a specific agent.

        Args:
            workflow: Current workflow state
            agent_name: Name of the agent to get comments from

        Returns:
            List of comments by the specified agent
        """
        return [
            c for c in workflow.conversation_history
            if c.agent == agent_name
        ]

    def get_comments_mentioning(
        self,
        workflow: WorkflowInstance,
    ) -> list[Comment]:
        """
        Get all comments that reference this agent.

        Useful for finding comments that respond to or mention this agent's
        contributions, so the agent can decide whether to respond.

        Args:
            workflow: Current workflow state

        Returns:
            List of comments that reference this agent
        """
        return [
            c for c in workflow.conversation_history
            if self.name in c.references
        ]

    def build_system_prompt(self) -> str:
        """
        Build the system prompt for this agent.

        If a custom system prompt is set in config, use that.
        Otherwise, build a default prompt from the agent's expertise
        and domain knowledge.

        Returns:
            System prompt string for LLM calls
        """
        if self._config.system_prompt:
            return self._config.system_prompt

        return (
            f"You are {self.name}, a specialist agent with expertise in: "
            f"{self.expertise}.\n\n"
            f"Your domain knowledge includes: {self.domain_knowledge}\n\n"
            f"You are participating in a multi-agent deliberation about a GitHub issue. "
            f"Your role is to provide expert analysis from your domain perspective.\n\n"
            f"Guidelines:\n"
            f"- Only comment when you have genuine, valuable insights to share\n"
            f"- Reference other agents by name when responding to their points\n"
            f"- Be concise but thorough in your analysis\n"
            f"- Acknowledge when something is outside your expertise\n"
            f"- Focus on actionable recommendations\n"
            f"- Avoid repeating points already made by other agents"
        )

    def create_comment(
        self,
        text: str,
        round_number: int,
        references: Optional[list[str]] = None,
    ) -> Comment:
        """
        Create a Comment object with this agent's metadata.

        Convenience method to create properly formatted comments.

        Args:
            text: The comment text
            round_number: Current round number
            references: List of agent names being responded to

        Returns:
            Comment object ready for the conversation history
        """
        return Comment(
            round=round_number,
            agent=self.name,
            comment=text,
            references=references or [],
            timestamp=datetime.utcnow(),
        )

    # ==================
    # Serialization
    # ==================

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the agent to a dictionary.

        Returns the agent configuration as a dictionary suitable for
        JSON serialization and storage.

        Returns:
            Dictionary representation of the agent
        """
        return {
            "name": self.name,
            "type": self.agent_type.value,
            "expertise": self.expertise,
            "domain_knowledge": self.domain_knowledge,
            "model": self.model,
            "tools": self.tools,
            "priority": self.priority,
            "category": self.category,
            "can_edit_files": self.can_edit_files,
            "llm": {
                "provider": self.llm_provider.provider,
                "model": self.llm_provider.model,
                "temperature": self.llm_provider.temperature,
                "max_tokens": self.llm_provider.max_tokens,
            },
        }

    @classmethod
    def from_config(cls, config: AgentConfig) -> "BaseAgent":
        """
        Create an agent instance from an AgentConfig.

        This is the primary factory method for creating agents. Subclasses
        can override this to customize initialization.

        Args:
            config: Agent configuration

        Returns:
            Agent instance

        Note:
            This method must be called on a concrete subclass, not BaseAgent directly.
        """
        return cls(config=config)

    # ==================
    # Dunder Methods
    # ==================

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}("
            f"name='{self.name}', "
            f"type={self.agent_type.value}, "
            f"expertise='{self.expertise[:40]}...')>"
        )

    def __str__(self) -> str:
        return f"{self.name} ({self.expertise})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseAgent):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)
