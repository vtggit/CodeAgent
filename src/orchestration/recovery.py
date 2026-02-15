"""
Workflow Recovery for the Multi-Agent GitHub Issue Routing System.

This module provides functionality to recover interrupted workflows from
the database and resume deliberation from the last completed round.

Recovery Process:
1. Load the workflow record with all relations from the database
2. Reconstruct the in-memory WorkflowInstance from DB records
3. Determine the last completed round and remaining rounds
4. Resume the deliberation loop from the next round

Design Decisions:
- Recovery is idempotent: running recovery twice on the same workflow
  picks up from the same point
- Conversation history is fully reconstructed from DB records
- Selected agents are re-resolved from the registry using stored names
- If an agent is no longer available, it's skipped with a warning
- Convergence metrics are used to determine continuation state
"""

import json
from datetime import datetime
from typing import Optional

from src.database import crud
from src.database.engine import get_async_session
from src.database.models import (
    AgentRecord,
    ConversationRecord,
    ConvergenceMetricRecord,
    WorkflowRecord,
)
from src.models.workflow import (
    Comment,
    WorkflowConfig,
    WorkflowInstance,
    WorkflowMetrics,
    WorkflowStatus,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class WorkflowRecoveryError(Exception):
    """Raised when workflow recovery fails."""

    pass


def reconstruct_workflow(
    db_workflow: WorkflowRecord,
) -> WorkflowInstance:
    """
    Reconstruct a WorkflowInstance from a database WorkflowRecord.

    Converts the database record and its relationships back into
    the in-memory Pydantic model used by the orchestrator.

    Args:
        db_workflow: WorkflowRecord with conversations, agents, and metrics loaded.

    Returns:
        WorkflowInstance reconstructed from DB state.

    Raises:
        WorkflowRecoveryError: If the record is invalid or cannot be reconstructed.
    """
    if db_workflow is None:
        raise WorkflowRecoveryError("Cannot reconstruct from None record")

    # Map status string to enum
    try:
        status = WorkflowStatus(db_workflow.status)
    except ValueError:
        logger.warning(
            "Unknown status '%s' for workflow %s, defaulting to RUNNING",
            db_workflow.status,
            db_workflow.instance_id,
        )
        status = WorkflowStatus.RUNNING

    # Reconstruct GitHub issue data from stored fields
    github_issue = _reconstruct_issue(db_workflow)

    # Reconstruct conversation history from ConversationRecords
    conversation_history = _reconstruct_conversation(db_workflow.conversations)

    # Reconstruct selected agent names from AgentRecords
    selected_agents = [
        agent.agent_name
        for agent in sorted(db_workflow.agents, key=lambda a: a.agent_name)
    ]

    # Reconstruct workflow config
    config = WorkflowConfig(
        max_rounds=db_workflow.max_rounds,
    )

    # Reconstruct metrics
    metrics = WorkflowMetrics(
        current_round=db_workflow.current_round,
        total_comments=db_workflow.total_comments,
        convergence_score=db_workflow.final_convergence_score or 0.0,
        participating_agents=len(selected_agents),
        started_at=db_workflow.started_at,
    )

    workflow = WorkflowInstance(
        instance_id=db_workflow.instance_id,
        workflow_id=db_workflow.workflow_id,
        status=status,
        github_issue=github_issue,
        config=config,
        selected_agents=selected_agents,
        conversation_history=conversation_history,
        current_round=db_workflow.current_round,
        summary=db_workflow.summary,
        metrics=metrics,
        created_at=db_workflow.created_at,
        completed_at=db_workflow.completed_at,
    )

    logger.info(
        "Reconstructed workflow %s: round=%d, comments=%d, agents=%d",
        workflow.instance_id,
        workflow.current_round,
        len(workflow.conversation_history),
        len(workflow.selected_agents),
    )

    return workflow


def _reconstruct_issue(db_workflow: WorkflowRecord) -> dict:
    """Reconstruct GitHub issue dict from DB record fields."""
    labels = []
    if db_workflow.issue_labels:
        try:
            labels = json.loads(db_workflow.issue_labels)
        except (json.JSONDecodeError, TypeError):
            labels = []

    return {
        "number": db_workflow.issue_number,
        "title": db_workflow.issue_title,
        "body": db_workflow.issue_body or "",
        "url": db_workflow.issue_url or "",
        "html_url": db_workflow.issue_url or "",
        "labels": labels,
        "repository": db_workflow.repository,
    }


def _reconstruct_conversation(
    records: list[ConversationRecord],
) -> list[Comment]:
    """
    Reconstruct Comment list from ConversationRecords.

    Args:
        records: List of ConversationRecord from DB.

    Returns:
        List of Comment objects ordered by round and creation time.
    """
    comments: list[Comment] = []

    # Sort by round_number then by created_at
    sorted_records = sorted(
        records,
        key=lambda r: (r.round_number, r.created_at),
    )

    for record in sorted_records:
        references = []
        if record.references:
            try:
                references = json.loads(record.references)
            except (json.JSONDecodeError, TypeError):
                references = []

        comment = Comment(
            round=record.round_number,
            agent=record.agent_name,
            comment=record.comment,
            references=references,
            timestamp=record.created_at,
            github_comment_id=record.github_comment_id,
        )
        comments.append(comment)

    return comments


def get_last_convergence_state(
    db_workflow: WorkflowRecord,
) -> dict:
    """
    Extract the latest convergence state from a workflow's metrics.

    Useful for determining whether to continue deliberation after recovery.

    Args:
        db_workflow: WorkflowRecord with convergence_metrics loaded.

    Returns:
        Dict with last convergence state, or defaults if no metrics.
    """
    if not db_workflow.convergence_metrics:
        return {
            "last_round": 0,
            "convergence_score": 0.0,
            "should_continue": True,
            "reason": "No convergence data - starting fresh",
            "rambling_detected": False,
            "value_trend": "unknown",
        }

    # Get the latest metric by round number
    sorted_metrics = sorted(
        db_workflow.convergence_metrics,
        key=lambda m: m.round_number,
        reverse=True,
    )
    latest = sorted_metrics[0]

    return {
        "last_round": latest.round_number,
        "convergence_score": latest.convergence_score,
        "should_continue": latest.should_continue,
        "reason": latest.reason or "",
        "rambling_detected": latest.rambling_detected,
        "value_trend": latest.value_trend or "unknown",
    }


async def find_recoverable_workflows() -> list[dict]:
    """
    Find all workflows that can be recovered (status='running').

    Returns:
        List of dicts with workflow summary information.
    """
    session = await get_async_session()
    async with session:
        workflows = await crud.get_interrupted_workflows(session)
        results = []
        for wf in workflows:
            convergence_state = get_last_convergence_state(wf)
            results.append({
                "instance_id": wf.instance_id,
                "issue_number": wf.issue_number,
                "issue_title": wf.issue_title,
                "current_round": wf.current_round,
                "max_rounds": wf.max_rounds,
                "total_comments": wf.total_comments,
                "last_convergence_score": convergence_state["convergence_score"],
                "should_continue": convergence_state["should_continue"],
                "updated_at": wf.updated_at.isoformat() if wf.updated_at else None,
                "db_pk": wf.id,
            })
        return results


async def load_workflow_for_recovery(
    instance_id: str,
) -> Optional[WorkflowInstance]:
    """
    Load and reconstruct a workflow from the database for recovery.

    This is the main entry point for recovering a specific workflow.

    Args:
        instance_id: The workflow instance ID to recover.

    Returns:
        WorkflowInstance if found and reconstructed, None if not found.

    Raises:
        WorkflowRecoveryError: If reconstruction fails.
    """
    session = await get_async_session()
    async with session:
        db_workflow = await crud.get_workflow_for_recovery(
            session=session,
            instance_id=instance_id,
        )

        if db_workflow is None:
            logger.warning("Workflow %s not found in database", instance_id)
            return None

        return reconstruct_workflow(db_workflow)
