"""
Pydantic models for workflow instances and deliberation state.

These models represent the runtime state of a multi-agent deliberation,
including conversation history, agent selections, and convergence metrics.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class WorkflowStatus(str, Enum):
    """Possible statuses for a workflow instance."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class Comment(BaseModel):
    """
    A single agent comment in the deliberation conversation.

    Represents one agent's contribution in a specific round, including
    references to other agents' comments it responds to.
    """

    round: int = Field(..., ge=1, description="Round number (1-based)")
    agent: str = Field(..., description="Name of the commenting agent")
    comment: str = Field(..., min_length=1, description="The comment text")
    references: list[str] = Field(
        default_factory=list,
        description="List of agent names this comment references/responds to",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the comment was made",
    )
    github_comment_id: Optional[int] = Field(
        None,
        description="GitHub comment ID if posted to the issue",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "round": 1,
                    "agent": "ui_architect",
                    "comment": "I recommend using a toggle component in the header...",
                    "references": [],
                    "timestamp": "2026-02-08T14:31:05Z",
                },
                {
                    "round": 2,
                    "agent": "ada_expert",
                    "comment": "Regarding ui_architect's suggestion, we need to ensure...",
                    "references": ["ui_architect"],
                    "timestamp": "2026-02-08T14:32:03Z",
                },
            ]
        }
    }


class ContinueDecision(BaseModel):
    """
    Result of the moderator's convergence check after each round.

    Determines whether the deliberation should continue or conclude,
    along with supporting metrics and reasoning.
    """

    should_continue: bool = Field(
        ...,
        description="Whether the deliberation should proceed to the next round",
    )
    reason: str = Field(
        ...,
        description="Explanation of why the decision was made",
    )
    convergence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall convergence score (0.0 = no consensus, 1.0 = full consensus)",
    )
    rambling_detected: bool = Field(
        False,
        description="Whether circular arguments or repetition was detected",
    )
    value_trend: str = Field(
        "stable",
        description="Trend of value added per round (increasing, decreasing, stable)",
    )

    @field_validator("value_trend")
    @classmethod
    def validate_value_trend(cls, v: str) -> str:
        """Ensure value_trend is one of the allowed values."""
        allowed = {
            "increasing", "decreasing", "stable", "unknown",
            "converged", "declining", "plateaued", "limited", "productive",
        }
        if v not in allowed:
            raise ValueError(f"value_trend must be one of {allowed}, got '{v}'")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "should_continue": True,
                    "reason": "Discussion is still productive with new insights emerging",
                    "convergence_score": 0.45,
                    "rambling_detected": False,
                    "value_trend": "increasing",
                },
                {
                    "should_continue": False,
                    "reason": "High consensus reached, agents are in agreement",
                    "convergence_score": 0.92,
                    "rambling_detected": False,
                    "value_trend": "decreasing",
                },
            ]
        }
    }


class WorkflowMetrics(BaseModel):
    """
    Runtime metrics for a workflow instance.

    Tracks progress and performance of the deliberation.
    """

    current_round: int = Field(0, ge=0, description="Current round number")
    total_comments: int = Field(0, ge=0, description="Total comments across all rounds")
    convergence_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Latest convergence score",
    )
    participating_agents: int = Field(
        0,
        ge=0,
        description="Number of agents who have commented",
    )
    started_at: Optional[datetime] = Field(None, description="When deliberation started")
    estimated_completion: Optional[datetime] = Field(
        None,
        description="Estimated completion time",
    )


class WorkflowConfig(BaseModel):
    """
    Configuration for a workflow execution.

    Controls the behavior of the deliberation loop.
    """

    max_rounds: int = Field(10, ge=1, le=50, description="Maximum number of rounds")
    convergence_threshold: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Threshold for consensus detection",
    )
    min_value_threshold: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description="Minimum value score for accepting suggestions",
    )
    timeout_minutes: int = Field(
        60,
        ge=1,
        le=1440,
        description="Maximum time for a deliberation session in minutes",
    )


class WorkflowInstance(BaseModel):
    """
    A complete workflow instance representing a deliberation session.

    Contains all state needed to track, persist, and resume a multi-agent
    deliberation including the trigger issue, selected agents, conversation
    history, and runtime metrics.
    """

    instance_id: str = Field(
        ...,
        description="Unique identifier (e.g., 'issue-123-20260208-143022')",
    )
    workflow_id: str = Field(
        "github-issue-deliberation",
        description="Workflow definition identifier",
    )
    status: WorkflowStatus = Field(
        WorkflowStatus.PENDING,
        description="Current workflow status",
    )

    # GitHub issue that triggered the workflow
    github_issue: Optional[dict] = Field(
        None,
        description="GitHub issue data that triggered this workflow",
    )

    # Execution state
    config: WorkflowConfig = Field(
        default_factory=WorkflowConfig,
        description="Workflow execution configuration",
    )
    selected_agents: list[str] = Field(
        default_factory=list,
        description="Names of agents selected for this deliberation",
    )
    conversation_history: list[Comment] = Field(
        default_factory=list,
        description="Full conversation history across all rounds",
    )
    current_round: int = Field(0, ge=0, description="Current round number")

    # Results
    summary: Optional[str] = Field(
        None,
        description="Final synthesized recommendations",
    )
    metrics: WorkflowMetrics = Field(
        default_factory=WorkflowMetrics,
        description="Runtime metrics",
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the workflow was created",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="When the workflow completed",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "instance_id": "issue-42-20260208-143022",
                    "workflow_id": "github-issue-deliberation",
                    "status": "running",
                    "github_issue": {
                        "number": 42,
                        "title": "Add dark mode toggle",
                        "body": "Need dark mode support",
                        "labels": ["enhancement", "ui"],
                    },
                    "selected_agents": [
                        "ui_architect",
                        "ada_expert",
                        "frontend_dev",
                    ],
                    "current_round": 2,
                }
            ]
        }
    }
