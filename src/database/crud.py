"""
CRUD operations for database models.

Provides async functions for creating, reading, updating, and deleting
workflow records and related data.
"""

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.database.models import (
    WorkflowRecord,
    ConversationRecord,
    AgentRecord,
    ConvergenceMetricRecord,
)


# =====================
# Workflow CRUD
# =====================


async def create_workflow(
    session: AsyncSession,
    instance_id: str,
    issue_number: int,
    issue_title: str,
    issue_body: Optional[str] = None,
    issue_url: Optional[str] = None,
    repository: Optional[str] = None,
    issue_labels: Optional[list[str]] = None,
    max_rounds: int = 10,
    workflow_id: str = "github-issue-deliberation",
) -> WorkflowRecord:
    """
    Create a new workflow record.

    Args:
        session: Database session.
        instance_id: Unique instance identifier.
        issue_number: GitHub issue number.
        issue_title: GitHub issue title.
        issue_body: GitHub issue body text.
        issue_url: GitHub issue URL.
        repository: Repository in owner/repo format.
        issue_labels: List of issue labels.
        max_rounds: Maximum deliberation rounds.
        workflow_id: Workflow definition identifier.

    Returns:
        WorkflowRecord: The created workflow record.
    """
    workflow = WorkflowRecord(
        instance_id=instance_id,
        workflow_id=workflow_id,
        status="pending",
        issue_number=issue_number,
        issue_title=issue_title,
        issue_body=issue_body,
        issue_url=issue_url,
        repository=repository,
        issue_labels=json.dumps(issue_labels) if issue_labels else None,
        max_rounds=max_rounds,
    )
    session.add(workflow)
    await session.flush()
    return workflow


async def get_workflow_by_instance_id(
    session: AsyncSession,
    instance_id: str,
    load_relations: bool = False,
) -> Optional[WorkflowRecord]:
    """
    Get a workflow by its instance ID.

    Args:
        session: Database session.
        instance_id: The workflow instance ID.
        load_relations: Whether to eagerly load relationships.

    Returns:
        WorkflowRecord or None.
    """
    stmt = select(WorkflowRecord).where(
        WorkflowRecord.instance_id == instance_id
    )
    if load_relations:
        stmt = stmt.options(
            selectinload(WorkflowRecord.conversations),
            selectinload(WorkflowRecord.agents),
            selectinload(WorkflowRecord.convergence_metrics),
        )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_workflow_by_id(
    session: AsyncSession,
    workflow_pk: int,
    load_relations: bool = False,
) -> Optional[WorkflowRecord]:
    """
    Get a workflow by its primary key ID.

    Args:
        session: Database session.
        workflow_pk: The workflow primary key.
        load_relations: Whether to eagerly load relationships.

    Returns:
        WorkflowRecord or None.
    """
    stmt = select(WorkflowRecord).where(WorkflowRecord.id == workflow_pk)
    if load_relations:
        stmt = stmt.options(
            selectinload(WorkflowRecord.conversations),
            selectinload(WorkflowRecord.agents),
            selectinload(WorkflowRecord.convergence_metrics),
        )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_workflows(
    session: AsyncSession,
    status: Optional[str] = None,
    repository: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[WorkflowRecord]:
    """
    List workflows with optional filtering.

    Args:
        session: Database session.
        status: Filter by status (pending, running, completed, failed, timeout).
        repository: Filter by repository.
        limit: Maximum number of results.
        offset: Number of results to skip.

    Returns:
        List of WorkflowRecord objects.
    """
    stmt = select(WorkflowRecord).order_by(WorkflowRecord.created_at.desc())

    if status:
        stmt = stmt.where(WorkflowRecord.status == status)
    if repository:
        stmt = stmt.where(WorkflowRecord.repository == repository)

    stmt = stmt.limit(limit).offset(offset)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def update_workflow_status(
    session: AsyncSession,
    workflow_pk: int,
    status: str,
    **kwargs,
) -> None:
    """
    Update a workflow's status and optional fields.

    Args:
        session: Database session.
        workflow_pk: Workflow primary key.
        status: New status value.
        **kwargs: Additional fields to update.
    """
    values = {"status": status, "updated_at": datetime.utcnow()}

    if status == "running" and "started_at" not in kwargs:
        values["started_at"] = datetime.utcnow()
    elif status in ("completed", "failed", "timeout") and "completed_at" not in kwargs:
        values["completed_at"] = datetime.utcnow()

    values.update(kwargs)

    stmt = (
        update(WorkflowRecord)
        .where(WorkflowRecord.id == workflow_pk)
        .values(**values)
    )
    await session.execute(stmt)


async def update_workflow_round(
    session: AsyncSession,
    workflow_pk: int,
    current_round: int,
    total_comments: int,
) -> None:
    """
    Update workflow round progress.

    Args:
        session: Database session.
        workflow_pk: Workflow primary key.
        current_round: Current round number.
        total_comments: Total comments so far.
    """
    stmt = (
        update(WorkflowRecord)
        .where(WorkflowRecord.id == workflow_pk)
        .values(
            current_round=current_round,
            total_comments=total_comments,
            updated_at=datetime.utcnow(),
        )
    )
    await session.execute(stmt)


# =====================
# Conversation CRUD
# =====================


async def add_conversation_entry(
    session: AsyncSession,
    workflow_pk: int,
    round_number: int,
    agent_name: str,
    comment: str,
    references: Optional[list[str]] = None,
    github_comment_id: Optional[int] = None,
    should_comment: bool = True,
) -> ConversationRecord:
    """
    Add a conversation entry (agent comment) to a workflow.

    Args:
        session: Database session.
        workflow_pk: Workflow primary key.
        round_number: Round this comment was made in.
        agent_name: Name of the commenting agent.
        comment: The comment text.
        references: List of agent names referenced.
        github_comment_id: GitHub comment ID if posted.
        should_comment: Whether the agent decided to comment.

    Returns:
        ConversationRecord: The created record.
    """
    record = ConversationRecord(
        workflow_id=workflow_pk,
        round_number=round_number,
        agent_name=agent_name,
        comment=comment,
        references=json.dumps(references) if references else None,
        github_comment_id=github_comment_id,
        should_comment=should_comment,
    )
    session.add(record)
    await session.flush()
    return record


async def get_conversation_history(
    session: AsyncSession,
    workflow_pk: int,
    round_number: Optional[int] = None,
) -> list[ConversationRecord]:
    """
    Get conversation history for a workflow.

    Args:
        session: Database session.
        workflow_pk: Workflow primary key.
        round_number: Optional filter for specific round.

    Returns:
        List of ConversationRecord objects.
    """
    stmt = (
        select(ConversationRecord)
        .where(ConversationRecord.workflow_id == workflow_pk)
        .order_by(
            ConversationRecord.round_number,
            ConversationRecord.created_at,
        )
    )
    if round_number is not None:
        stmt = stmt.where(ConversationRecord.round_number == round_number)

    result = await session.execute(stmt)
    return list(result.scalars().all())


# =====================
# Agent CRUD
# =====================


async def add_agent_to_workflow(
    session: AsyncSession,
    workflow_pk: int,
    agent_name: str,
    agent_type: Optional[str] = None,
    expertise: Optional[str] = None,
    selection_reason: Optional[str] = None,
) -> AgentRecord:
    """
    Record an agent selection for a workflow.

    Args:
        session: Database session.
        workflow_pk: Workflow primary key.
        agent_name: Agent identifier.
        agent_type: Agent type (claude_sdk_text, etc.).
        expertise: Agent expertise description.
        selection_reason: Why this agent was selected.

    Returns:
        AgentRecord: The created record.
    """
    record = AgentRecord(
        workflow_id=workflow_pk,
        agent_name=agent_name,
        agent_type=agent_type,
        expertise=expertise,
        selection_reason=selection_reason,
    )
    session.add(record)
    await session.flush()
    return record


async def update_agent_participation(
    session: AsyncSession,
    workflow_pk: int,
    agent_name: str,
    round_number: int,
) -> None:
    """
    Update an agent's participation metrics after they comment.

    Args:
        session: Database session.
        workflow_pk: Workflow primary key.
        agent_name: Agent identifier.
        round_number: Round number of the comment.
    """
    # Get the agent record
    stmt = select(AgentRecord).where(
        AgentRecord.workflow_id == workflow_pk,
        AgentRecord.agent_name == agent_name,
    )
    result = await session.execute(stmt)
    agent = result.scalar_one_or_none()

    if agent:
        agent.total_comments += 1
        agent.rounds_participated += 1
        if agent.first_comment_round is None:
            agent.first_comment_round = round_number
        agent.last_comment_round = round_number


async def get_workflow_agents(
    session: AsyncSession,
    workflow_pk: int,
) -> list[AgentRecord]:
    """
    Get all agents selected for a workflow.

    Args:
        session: Database session.
        workflow_pk: Workflow primary key.

    Returns:
        List of AgentRecord objects.
    """
    stmt = (
        select(AgentRecord)
        .where(AgentRecord.workflow_id == workflow_pk)
        .order_by(AgentRecord.agent_name)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


# =====================
# Convergence Metrics CRUD
# =====================


async def add_convergence_metric(
    session: AsyncSession,
    workflow_pk: int,
    round_number: int,
    convergence_score: float,
    should_continue: bool,
    reason: Optional[str] = None,
    agreement_level: Optional[float] = None,
    value_added: Optional[float] = None,
    rambling_detected: bool = False,
    value_trend: Optional[str] = None,
    comments_this_round: int = 0,
    agents_participated: int = 0,
    new_topics_raised: int = 0,
) -> ConvergenceMetricRecord:
    """
    Record convergence metrics for a round.

    Args:
        session: Database session.
        workflow_pk: Workflow primary key.
        round_number: Round number.
        convergence_score: Overall convergence score (0.0-1.0).
        should_continue: Whether deliberation should continue.
        reason: Explanation for the decision.
        agreement_level: Level of agreement between agents.
        value_added: Value added by this round.
        rambling_detected: Whether rambling was detected.
        value_trend: Trend of value added (increasing/decreasing/stable).
        comments_this_round: Number of comments in this round.
        agents_participated: Number of agents who commented.
        new_topics_raised: Number of new topics raised.

    Returns:
        ConvergenceMetricRecord: The created record.
    """
    record = ConvergenceMetricRecord(
        workflow_id=workflow_pk,
        round_number=round_number,
        convergence_score=convergence_score,
        should_continue=should_continue,
        reason=reason,
        agreement_level=agreement_level,
        value_added=value_added,
        rambling_detected=rambling_detected,
        value_trend=value_trend,
        comments_this_round=comments_this_round,
        agents_participated=agents_participated,
        new_topics_raised=new_topics_raised,
    )
    session.add(record)
    await session.flush()
    return record


async def get_convergence_metrics(
    session: AsyncSession,
    workflow_pk: int,
) -> list[ConvergenceMetricRecord]:
    """
    Get all convergence metrics for a workflow.

    Args:
        session: Database session.
        workflow_pk: Workflow primary key.

    Returns:
        List of ConvergenceMetricRecord objects ordered by round.
    """
    stmt = (
        select(ConvergenceMetricRecord)
        .where(ConvergenceMetricRecord.workflow_id == workflow_pk)
        .order_by(ConvergenceMetricRecord.round_number)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


# =====================
# Utility Queries
# =====================


async def count_workflows_by_status(
    session: AsyncSession,
) -> dict[str, int]:
    """
    Count workflows grouped by status.

    Returns:
        Dictionary mapping status to count.
    """
    stmt = (
        select(
            WorkflowRecord.status,
            func.count(WorkflowRecord.id),
        )
        .group_by(WorkflowRecord.status)
    )
    result = await session.execute(stmt)
    return {row[0]: row[1] for row in result.all()}


async def get_active_workflows(
    session: AsyncSession,
) -> list[WorkflowRecord]:
    """
    Get all currently active (running) workflows.

    Returns:
        List of running WorkflowRecord objects.
    """
    return await list_workflows(session, status="running")


async def get_interrupted_workflows(
    session: AsyncSession,
) -> list[WorkflowRecord]:
    """
    Get workflows that were interrupted mid-deliberation.

    Returns workflows with status 'running' that may need recovery.
    Includes full relationship data for reconstruction.

    Returns:
        List of WorkflowRecord objects with relations loaded.
    """
    stmt = (
        select(WorkflowRecord)
        .where(WorkflowRecord.status == "running")
        .options(
            selectinload(WorkflowRecord.conversations),
            selectinload(WorkflowRecord.agents),
            selectinload(WorkflowRecord.convergence_metrics),
        )
        .order_by(WorkflowRecord.updated_at.desc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_workflow_for_recovery(
    session: AsyncSession,
    instance_id: str,
) -> Optional[WorkflowRecord]:
    """
    Load a complete workflow record for recovery, with all relations.

    Args:
        session: Database session.
        instance_id: The workflow instance ID to recover.

    Returns:
        WorkflowRecord with conversations, agents, and metrics loaded, or None.
    """
    return await get_workflow_by_instance_id(
        session=session,
        instance_id=instance_id,
        load_relations=True,
    )


async def update_workflow_summary(
    session: AsyncSession,
    workflow_pk: int,
    summary: str,
    final_convergence_score: float,
) -> None:
    """
    Update a workflow's summary and convergence score.

    Args:
        session: Database session.
        workflow_pk: Workflow primary key.
        summary: The synthesized summary.
        final_convergence_score: Final convergence score.
    """
    stmt = (
        update(WorkflowRecord)
        .where(WorkflowRecord.id == workflow_pk)
        .values(
            summary=summary,
            final_convergence_score=final_convergence_score,
            updated_at=datetime.utcnow(),
        )
    )
    await session.execute(stmt)
