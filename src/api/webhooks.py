"""
GitHub webhook endpoints for the Multi-Agent Issue Routing System.

This module handles incoming GitHub webhooks, validates signatures,
and queues issues for processing by the agent deliberation system.
"""

import json
import time
import uuid
from typing import Any, Optional
from fastapi import APIRouter, Request, Response, Header, HTTPException, status
from pydantic import BaseModel
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.webhook import validate_github_signature, extract_issue_data
from src.queue.base import BaseQueue, QueueJob


# Router for webhook endpoints
router = APIRouter(prefix="/webhook", tags=["Webhooks"])


class WebhookResponse(BaseModel):
    """Response model for webhook acknowledgment."""

    status: str
    message: str
    processed_in_ms: float
    job_id: Optional[str] = None


# Global queue instance (initialized on startup)
_queue: Optional[BaseQueue] = None


def set_queue(queue: BaseQueue) -> None:
    """Set the global queue instance."""
    global _queue
    _queue = queue


def get_queue() -> BaseQueue:
    """Get the global queue instance."""
    if _queue is None:
        raise RuntimeError("Queue not initialized. Call set_queue() first.")
    return _queue


@router.post("/github", response_model=WebhookResponse)
async def github_webhook(
    request: Request,
    x_hub_signature_256: str | None = Header(None),
    x_github_event: str | None = Header(None),
) -> WebhookResponse:
    """
    Receive and process GitHub webhooks.

    This endpoint receives GitHub webhook events, validates the signature,
    extracts relevant issue data, and queues it for async processing.

    Args:
        request: FastAPI request object containing webhook payload
        x_hub_signature_256: GitHub signature header for validation
        x_github_event: GitHub event type (e.g., "issues")

    Returns:
        WebhookResponse: Acknowledgment with processing time

    Raises:
        HTTPException: 401 if signature validation fails
        HTTPException: 400 if payload is invalid
    """
    start_time = time.time()

    # Read raw body for signature validation
    raw_body = await request.body()

    # Get webhook secret from environment (placeholder for now)
    # TODO: Load from configuration in issue #8
    webhook_secret = "dev-webhook-secret"  # Replace with env var

    # Validate GitHub signature
    if x_hub_signature_256:
        if not validate_github_signature(raw_body, x_hub_signature_256, webhook_secret):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid webhook signature",
            )
    else:
        # In development, we might not have a signature
        # TODO: Make this stricter in production
        pass

    # Parse JSON payload
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload",
        )

    # Only process issue events
    if x_github_event != "issues":
        # Acknowledge but don't process
        elapsed_ms = (time.time() - start_time) * 1000
        return WebhookResponse(
            status="ignored",
            message=f"Event type '{x_github_event}' not processed",
            processed_in_ms=elapsed_ms,
        )

    # Extract issue data
    issue_data = extract_issue_data(payload)
    if not issue_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not extract issue data from payload",
        )

    # Only process certain actions (opened, labeled, edited)
    action = issue_data.get("action")
    if action not in ["opened", "labeled", "edited", "reopened"]:
        elapsed_ms = (time.time() - start_time) * 1000
        return WebhookResponse(
            status="ignored",
            message=f"Action '{action}' not processed",
            processed_in_ms=elapsed_ms,
        )

    # Queue for async processing using Redis/Memory queue
    queue = get_queue()

    # Create job
    job = QueueJob(
        job_id=str(uuid.uuid4()),
        job_type="webhook_issue",
        payload={
            "event_type": x_github_event,
            "action": action,
            "issue_data": issue_data,
            "received_at": time.time(),
        },
        priority=1,  # Default priority
        max_retries=3,
    )

    # Enqueue job
    success = await queue.enqueue(job)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to queue webhook for processing",
        )

    # Calculate response time
    elapsed_ms = (time.time() - start_time) * 1000

    return WebhookResponse(
        status="queued",
        message=f"Issue #{issue_data['issue_number']} queued for processing",
        processed_in_ms=elapsed_ms,
        job_id=job.job_id,
    )


@router.get("/queue/status", tags=["Webhooks"])
async def queue_status() -> dict[str, Any]:
    """
    Get current queue status (development endpoint).

    Returns queue depth, dead letter count, and health status.

    Returns:
        Dictionary with queue statistics
    """
    queue = get_queue()

    return {
        "queue_depth": await queue.get_queue_depth(),
        "dead_letter_count": await queue.get_dead_letter_count(),
        "healthy": await queue.health_check(),
        "queue_type": type(queue).__name__,
    }
