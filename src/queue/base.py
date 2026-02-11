"""
Abstract base class for message queue implementations.

This module defines the interface that all queue implementations must follow,
allowing the system to swap between Redis, RabbitMQ, SQS, or other queue backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from datetime import datetime


class QueueJob:
    """Represents a job in the queue."""

    def __init__(
        self,
        job_id: str,
        job_type: str,
        payload: dict[str, Any],
        priority: int = 0,
        max_retries: int = 3,
        retry_count: int = 0,
        created_at: Optional[datetime] = None,
        scheduled_for: Optional[datetime] = None,
    ):
        """
        Initialize a queue job.

        Args:
            job_id: Unique identifier for the job
            job_type: Type of job (e.g., "webhook_issue", "deliberation")
            payload: Job data
            priority: Job priority (higher = more important)
            max_retries: Maximum retry attempts
            retry_count: Current retry attempt number
            created_at: When the job was created
            scheduled_for: When the job should be processed
        """
        self.job_id = job_id
        self.job_type = job_type
        self.payload = payload
        self.priority = priority
        self.max_retries = max_retries
        self.retry_count = retry_count
        self.created_at = created_at or datetime.utcnow()
        self.scheduled_for = scheduled_for

    def to_dict(self) -> dict[str, Any]:
        """Serialize job to dictionary."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "payload": self.payload,
            "priority": self.priority,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "created_at": self.created_at.isoformat(),
            "scheduled_for": self.scheduled_for.isoformat() if self.scheduled_for else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueueJob":
        """Deserialize job from dictionary."""
        return cls(
            job_id=data["job_id"],
            job_type=data["job_type"],
            payload=data["payload"],
            priority=data.get("priority", 0),
            max_retries=data.get("max_retries", 3),
            retry_count=data.get("retry_count", 0),
            created_at=datetime.fromisoformat(data["created_at"]),
            scheduled_for=datetime.fromisoformat(data["scheduled_for"]) if data.get("scheduled_for") else None,
        )


class BaseQueue(ABC):
    """Abstract base class for queue implementations."""

    @abstractmethod
    async def enqueue(self, job: QueueJob) -> bool:
        """
        Add a job to the queue.

        Args:
            job: Job to enqueue

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def dequeue(self, timeout: int = 0) -> Optional[QueueJob]:
        """
        Remove and return a job from the queue.

        Args:
            timeout: How long to wait for a job (0 = non-blocking)

        Returns:
            Job if available, None otherwise
        """
        pass

    @abstractmethod
    async def requeue(self, job: QueueJob, delay_seconds: int = 0) -> bool:
        """
        Re-add a failed job to the queue with incremented retry count.

        Args:
            job: Job to requeue
            delay_seconds: Delay before job becomes available

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def move_to_dead_letter(self, job: QueueJob) -> bool:
        """
        Move a permanently failed job to dead letter queue.

        Args:
            job: Job that exhausted retries

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def get_queue_depth(self) -> int:
        """
        Get number of jobs waiting in queue.

        Returns:
            Number of jobs in queue
        """
        pass

    @abstractmethod
    async def get_dead_letter_count(self) -> int:
        """
        Get number of jobs in dead letter queue.

        Returns:
            Number of failed jobs
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if queue backend is healthy.

        Returns:
            True if healthy, False otherwise
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connections and cleanup resources."""
        pass
