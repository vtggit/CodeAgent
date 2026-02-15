"""
API endpoints for the Multi-Agent Deliberation Orchestrator.

Provides endpoints to:
- Trigger a deliberation on a GitHub issue
- Check status of ongoing deliberations
- Retrieve deliberation results
"""

from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

from src.models.workflow import WorkflowConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/deliberation",
    tags=["Deliberation"],
)

# In-memory store for deliberation results (keyed by instance_id)
_deliberation_results: dict = {}
_active_deliberations: dict = {}


class DeliberationRequest(BaseModel):
    """Request to trigger a deliberation on an issue."""

    issue_number: int = Field(..., description="GitHub issue number")
    issue_title: str = Field(..., description="GitHub issue title")
    issue_body: str = Field("", description="GitHub issue body/description")
    issue_labels: list[str] = Field(default_factory=list, description="Issue labels")
    max_rounds: int = Field(10, ge=1, le=50, description="Maximum rounds")
    convergence_threshold: float = Field(
        0.8, ge=0.0, le=1.0, description="Convergence threshold"
    )


class DeliberationTriggerResponse(BaseModel):
    """Response after triggering a deliberation."""

    status: str
    instance_id: str
    message: str


class DeliberationStatusResponse(BaseModel):
    """Response with deliberation status."""

    instance_id: str
    status: str
    current_round: int = 0
    total_comments: int = 0
    message: str = ""


async def _run_deliberation(
    instance_id: str,
    issue_data: dict,
    config: WorkflowConfig,
) -> None:
    """
    Background task to run a deliberation.

    Args:
        instance_id: Unique deliberation instance ID
        issue_data: GitHub issue data
        config: Workflow configuration
    """
    try:
        from src.api.main import get_agent_registry
        from src.orchestration.deliberation import MultiAgentDeliberationOrchestrator
        from src.orchestration.moderator import ModeratorAgent
        from src.orchestration.convergence import ConvergenceDetector

        registry = get_agent_registry()
        moderator = ModeratorAgent(registry=registry)
        detector = ConvergenceDetector(use_llm=False)

        orchestrator = MultiAgentDeliberationOrchestrator(
            registry=registry,
            moderator=moderator,
            convergence_detector=detector,
            config=config,
            persist_to_db=True,
        )

        _active_deliberations[instance_id] = "running"

        result = await orchestrator.deliberate_on_issue(
            issue=issue_data,
            config=config,
        )

        _deliberation_results[result.workflow.instance_id] = result.to_dict()
        _active_deliberations[instance_id] = "completed"

        logger.info(
            "Deliberation %s completed: %d rounds, %d comments",
            result.workflow.instance_id,
            result.total_rounds,
            result.total_comments,
        )

    except Exception as e:
        logger.error("Deliberation %s failed: %s", instance_id, str(e))
        _active_deliberations[instance_id] = "failed"
        _deliberation_results[instance_id] = {
            "instance_id": instance_id,
            "status": "failed",
            "error": str(e),
        }


@router.post(
    "/trigger",
    response_model=DeliberationTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_deliberation(
    request: DeliberationRequest,
    background_tasks: BackgroundTasks,
) -> DeliberationTriggerResponse:
    """
    Trigger a new deliberation on a GitHub issue.

    The deliberation runs asynchronously in the background. Use the
    /deliberation/status/{instance_id} endpoint to check progress.
    """
    from datetime import datetime

    instance_id = f"issue-{request.issue_number}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

    issue_data = {
        "number": request.issue_number,
        "title": request.issue_title,
        "body": request.issue_body,
        "labels": request.issue_labels,
    }

    config = WorkflowConfig(
        max_rounds=request.max_rounds,
        convergence_threshold=request.convergence_threshold,
    )

    background_tasks.add_task(
        _run_deliberation,
        instance_id=instance_id,
        issue_data=issue_data,
        config=config,
    )

    _active_deliberations[instance_id] = "starting"

    return DeliberationTriggerResponse(
        status="accepted",
        instance_id=instance_id,
        message=f"Deliberation started for issue #{request.issue_number}",
    )


@router.get(
    "/status/{instance_id}",
    response_model=DeliberationStatusResponse,
)
async def get_deliberation_status(
    instance_id: str,
) -> DeliberationStatusResponse:
    """
    Get the status of a deliberation.

    Args:
        instance_id: The deliberation instance ID
    """
    status_val = _active_deliberations.get(instance_id)
    if status_val is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Deliberation '{instance_id}' not found",
        )

    result = _deliberation_results.get(instance_id, {})

    return DeliberationStatusResponse(
        instance_id=instance_id,
        status=status_val,
        current_round=result.get("total_rounds", 0),
        total_comments=result.get("total_comments", 0),
        message=result.get("termination_reason", "In progress..."),
    )


@router.get("/results/{instance_id}")
async def get_deliberation_results(instance_id: str):
    """
    Get the full results of a completed deliberation.

    Args:
        instance_id: The deliberation instance ID
    """
    result = _deliberation_results.get(instance_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No results for deliberation '{instance_id}'",
        )
    return result


@router.get("/active")
async def list_active_deliberations():
    """List all active and recent deliberations."""
    return {
        "active": {
            k: v for k, v in _active_deliberations.items()
            if v in ("starting", "running")
        },
        "completed": {
            k: v for k, v in _active_deliberations.items()
            if v == "completed"
        },
        "failed": {
            k: v for k, v in _active_deliberations.items()
            if v == "failed"
        },
        "total": len(_active_deliberations),
    }
