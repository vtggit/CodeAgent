"""
SQLAlchemy ORM models for workflow state persistence.

Defines database tables for:
- Workflows: Top-level deliberation instances
- ConversationHistory: Agent comments per round
- Agents: Agent selection records per workflow
- ConvergenceMetrics: Per-round convergence measurements
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Boolean,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class WorkflowRecord(Base):
    """
    Represents a deliberation workflow instance.

    Each workflow is triggered by a GitHub issue and tracks the full
    lifecycle of the multi-agent deliberation process.
    """

    __tablename__ = "workflows"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instance_id = Column(String(255), unique=True, nullable=False, index=True)
    workflow_id = Column(String(255), nullable=False, default="github-issue-deliberation")
    status = Column(
        String(50),
        nullable=False,
        default="pending",
        index=True,
    )  # pending, running, completed, failed, timeout

    # GitHub issue information
    issue_number = Column(Integer, nullable=False)
    issue_title = Column(String(500), nullable=False)
    issue_body = Column(Text, nullable=True)
    issue_url = Column(String(500), nullable=True)
    repository = Column(String(255), nullable=True)
    issue_labels = Column(Text, nullable=True)  # JSON-encoded list

    # Deliberation state
    current_round = Column(Integer, nullable=False, default=0)
    max_rounds = Column(Integer, nullable=False, default=10)
    total_comments = Column(Integer, nullable=False, default=0)

    # Final results
    summary = Column(Text, nullable=True)
    final_convergence_score = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    conversations = relationship(
        "ConversationRecord",
        back_populates="workflow",
        cascade="all, delete-orphan",
        order_by="ConversationRecord.round_number, ConversationRecord.created_at",
    )
    agents = relationship(
        "AgentRecord",
        back_populates="workflow",
        cascade="all, delete-orphan",
    )
    convergence_metrics = relationship(
        "ConvergenceMetricRecord",
        back_populates="workflow",
        cascade="all, delete-orphan",
        order_by="ConvergenceMetricRecord.round_number",
    )

    # Indexes for common queries
    __table_args__ = (
        Index("ix_workflows_status_created", "status", "created_at"),
        Index("ix_workflows_repo_issue", "repository", "issue_number"),
    )

    def __repr__(self) -> str:
        return (
            f"<WorkflowRecord(id={self.id}, instance_id='{self.instance_id}', "
            f"status='{self.status}', round={self.current_round})>"
        )


class ConversationRecord(Base):
    """
    Represents a single agent comment in the deliberation conversation.

    Each record maps to one comment by one agent in one round.
    """

    __tablename__ = "conversation_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    workflow_id = Column(
        Integer,
        ForeignKey("workflows.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    round_number = Column(Integer, nullable=False)
    agent_name = Column(String(255), nullable=False, index=True)
    comment = Column(Text, nullable=False)
    references = Column(Text, nullable=True)  # JSON-encoded list of referenced agents
    github_comment_id = Column(Integer, nullable=True)  # GitHub comment ID if posted

    # Metadata
    should_comment = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationship
    workflow = relationship("WorkflowRecord", back_populates="conversations")

    __table_args__ = (
        Index("ix_conversation_workflow_round", "workflow_id", "round_number"),
    )

    def __repr__(self) -> str:
        return (
            f"<ConversationRecord(id={self.id}, agent='{self.agent_name}', "
            f"round={self.round_number})>"
        )


class AgentRecord(Base):
    """
    Tracks which agents were selected for a workflow and their participation.
    """

    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    workflow_id = Column(
        Integer,
        ForeignKey("workflows.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    agent_name = Column(String(255), nullable=False)
    agent_type = Column(String(100), nullable=True)  # claude_sdk_text, claude_sdk_code, etc.
    expertise = Column(String(500), nullable=True)
    selected = Column(Boolean, nullable=False, default=True)
    selection_reason = Column(Text, nullable=True)

    # Participation metrics
    total_comments = Column(Integer, nullable=False, default=0)
    rounds_participated = Column(Integer, nullable=False, default=0)
    first_comment_round = Column(Integer, nullable=True)
    last_comment_round = Column(Integer, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationship
    workflow = relationship("WorkflowRecord", back_populates="agents")

    __table_args__ = (
        Index("ix_agents_workflow_name", "workflow_id", "agent_name", unique=True),
    )

    def __repr__(self) -> str:
        return (
            f"<AgentRecord(id={self.id}, agent='{self.agent_name}', "
            f"comments={self.total_comments})>"
        )


class ConvergenceMetricRecord(Base):
    """
    Stores convergence metrics measured after each round.

    Used to track how the discussion is progressing toward consensus
    and to make data-driven decisions about when to conclude.
    """

    __tablename__ = "convergence_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    workflow_id = Column(
        Integer,
        ForeignKey("workflows.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    round_number = Column(Integer, nullable=False)

    # Convergence measurements
    convergence_score = Column(Float, nullable=False, default=0.0)
    agreement_level = Column(Float, nullable=True)
    value_added = Column(Float, nullable=True)
    rambling_detected = Column(Boolean, nullable=False, default=False)
    value_trend = Column(String(50), nullable=True)  # increasing, decreasing, stable

    # Round statistics
    comments_this_round = Column(Integer, nullable=False, default=0)
    agents_participated = Column(Integer, nullable=False, default=0)
    new_topics_raised = Column(Integer, nullable=True, default=0)

    # Decision
    should_continue = Column(Boolean, nullable=False, default=True)
    reason = Column(Text, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationship
    workflow = relationship("WorkflowRecord", back_populates="convergence_metrics")

    __table_args__ = (
        Index(
            "ix_convergence_workflow_round",
            "workflow_id",
            "round_number",
            unique=True,
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<ConvergenceMetricRecord(id={self.id}, round={self.round_number}, "
            f"score={self.convergence_score:.2f})>"
        )
