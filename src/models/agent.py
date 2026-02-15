"""
Pydantic models for agent configuration and definitions.

These models define the structure for agent definitions that are loaded
from YAML configuration files, supporting the 50+ specialist agent pool.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AgentType(str, Enum):
    """Supported agent implementation types."""

    CLAUDE_SDK_TEXT = "claude_sdk_text"
    CLAUDE_SDK_CODE = "claude_sdk_code"
    GITHUB_API = "github_api"
    CUSTOM = "custom"


class FallbackProviderConfig(BaseModel):
    """
    Configuration for a fallback LLM provider.

    Used when the primary provider fails or is unavailable.
    """

    provider: str = Field(
        ...,
        description="Fallback provider name (anthropic, openai, lm_studio, ollama)",
    )
    model: str = Field(
        ...,
        description="Fallback model identifier",
    )
    base_url: Optional[str] = Field(
        None,
        description="Custom API base URL for fallback provider",
    )


class LLMProviderConfig(BaseModel):
    """
    LLM provider configuration for an agent.

    Each agent can use a different LLM provider and model based on
    task complexity, cost, privacy, and performance requirements.
    Supports fallback providers for reliability.
    """

    provider: str = Field(
        "anthropic",
        description="LLM provider name (anthropic, openai, lm_studio, ollama)",
    )
    model: str = Field(
        "claude-sonnet-4-5-20250929",
        description="Model identifier",
    )
    base_url: Optional[str] = Field(
        None,
        description="Custom API base URL (for LM Studio, Ollama, etc.)",
    )
    temperature: float = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: int = Field(
        4096,
        ge=1,
        le=200000,
        description="Maximum tokens in response",
    )

    # Multi-provider support
    fallback_providers: list[FallbackProviderConfig] = Field(
        default_factory=list,
        description="Ordered list of fallback providers to try if primary fails",
    )
    rate_limit_rpm: Optional[int] = Field(
        None,
        ge=1,
        le=10000,
        description="Rate limit in requests per minute (overrides provider default)",
    )


class AgentConfig(BaseModel):
    """
    Configuration for a specialist agent.

    Defines an agent's identity, expertise, capabilities, and LLM settings.
    Matches the structure expected in agent_definitions.yaml.
    """

    name: str = Field(..., description="Unique agent identifier")
    type: AgentType = Field(
        AgentType.CLAUDE_SDK_TEXT,
        description="Agent implementation type",
    )
    expertise: str = Field(
        ...,
        description="Short description of the agent's area of expertise",
    )
    domain_knowledge: str = Field(
        "",
        description="Detailed knowledge areas, technologies, and skills",
    )
    system_prompt: Optional[str] = Field(
        None,
        description="Custom system prompt override for this agent",
    )

    # LLM configuration
    llm: LLMProviderConfig = Field(
        default_factory=LLMProviderConfig,
        description="LLM provider and model configuration",
    )

    # Capabilities
    tools: list[str] = Field(
        default_factory=list,
        description="List of tools this agent can use (for code agents)",
    )
    can_edit_files: bool = Field(
        False,
        description="Whether the agent can edit files (code agent feature)",
    )

    # Behavioral hints
    priority: int = Field(
        5,
        ge=1,
        le=10,
        description="Agent priority for selection (1=lowest, 10=highest)",
    )
    category: Optional[str] = Field(
        None,
        description="Agent category (architecture, development, quality, domain, business)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "ui_architect",
                    "type": "claude_sdk_text",
                    "expertise": "User interface architecture and component design",
                    "domain_knowledge": "React, Vue, design systems, responsive design, component libraries, accessibility patterns",
                    "llm": {
                        "provider": "anthropic",
                        "model": "claude-sonnet-4-5-20250929",
                    },
                    "category": "architecture",
                },
                {
                    "name": "frontend_dev",
                    "type": "claude_sdk_code",
                    "expertise": "Frontend implementation",
                    "domain_knowledge": "JavaScript/TypeScript, React hooks, state management",
                    "tools": ["Read", "Grep", "Glob", "Edit", "Write", "Bash"],
                    "can_edit_files": True,
                    "category": "development",
                },
            ]
        }
    }


class AgentParticipation(BaseModel):
    """
    Tracks an agent's participation in a specific workflow.

    Records selection information and participation statistics.
    """

    agent_name: str = Field(..., description="Agent identifier")
    selected: bool = Field(True, description="Whether the agent was selected")
    selection_reason: Optional[str] = Field(
        None,
        description="Why this agent was selected for the workflow",
    )

    # Participation metrics
    total_comments: int = Field(0, ge=0, description="Total comments made")
    rounds_participated: int = Field(0, ge=0, description="Number of rounds with comments")
    first_comment_round: Optional[int] = Field(None, description="Round of first comment")
    last_comment_round: Optional[int] = Field(None, description="Round of last comment")


class AgentDecision(BaseModel):
    """
    An agent's decision about whether to comment in a round.

    Used internally by agents to communicate their participation decision
    to the orchestrator.
    """

    should_comment: bool = Field(
        ...,
        description="Whether the agent wants to comment this round",
    )
    reason: Optional[str] = Field(
        None,
        description="Reasoning for the decision",
    )
    responding_to: list[str] = Field(
        default_factory=list,
        description="Names of agents being responded to",
    )
    confidence: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in the relevance of their input",
    )
