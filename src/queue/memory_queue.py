"""
In-memory message queue implementation (fallback).

This module provides a simple in-memory queue for development and testing
when Redis is not available. NOT suitable for production use.
"""

import asyncio
from typing import Optional
from datetime import datetime
from collections import deque

from .base import BaseQueue, QueueJob


class MemoryQueue(BaseQueue):
    """In-memory implementation of message queue (development only)."""

    def __init__(self, queue_name: str = "jobs"):
        """
        Initialize in-memory queue.

        Args:
            queue_name: Name for the queue (for logging)
        """
        self.queue_name = queue_name
        self._main_queue: deque[QueueJob] = deque()
        self._delayed_queue: deque[QueueJob] = deque()
        self._dead_letter: deque[QueueJob] = deque()
        self._lock = asyncio.Lock()

    async def enqueue(self, job: QueueJob) -> bool:
        """Add a job to the queue."""
        try:
            async with self._lock:
                # If job is scheduled for future, add to delayed queue
                if job.scheduled_for and job.scheduled_for > datetime.utcnow():
                    self._delayed_queue.append(job)
                else:
                    # Insert in priority order (higher priority first)
                    inserted = False
                    for i, existing_job in enumerate(self._main_queue):
                        if job.priority > existing_job.priority:
                            self._main_queue.insert(i, job)
                            inserted = True
                            break
                    if not inserted:
                        self._main_queue.append(job)

            return True

        except Exception as e:
            print(f"Error enqueueing job: {e}")
            return False

    async def dequeue(self, timeout: int = 0) -> Optional[QueueJob]:
        """Remove and return highest priority job from queue."""
        try:
            # First, process delayed jobs
            await self._process_delayed_jobs()

            async with self._lock:
                if self._main_queue:
                    return self._main_queue.popleft()

            return None

        except Exception as e:
            print(f"Error dequeuing job: {e}")
            return None

    async def _process_delayed_jobs(self) -> None:
        """Move delayed jobs that are ready to main queue."""
        try:
            async with self._lock:
                now = datetime.utcnow()
                ready_jobs = []

                # Find all ready jobs
                for job in list(self._delayed_queue):
                    if job.scheduled_for and job.scheduled_for <= now:
                        ready_jobs.append(job)
                        self._delayed_queue.remove(job)

                # Add to main queue in priority order
                for job in ready_jobs:
                    job.scheduled_for = None  # Clear scheduled time
                    inserted = False
                    for i, existing_job in enumerate(self._main_queue):
                        if job.priority > existing_job.priority:
                            self._main_queue.insert(i, job)
                            inserted = True
                            break
                    if not inserted:
                        self._main_queue.append(job)

        except Exception as e:
            print(f"Error processing delayed jobs: {e}")

    async def requeue(self, job: QueueJob, delay_seconds: int = 0) -> bool:
        """Re-add a failed job to the queue with incremented retry count."""
        try:
            # Increment retry count
            job.retry_count += 1

            # Check if job has exhausted retries
            if job.retry_count > job.max_retries:
                return await self.move_to_dead_letter(job)

            # Calculate delay with exponential backoff (only if delay not specified)
            if delay_seconds <= 0:
                # Exponential backoff: 2^retry_count minutes
                delay_seconds = min(2 ** job.retry_count * 60, 3600)  # Max 1 hour

            # Schedule for future processing (or immediate if delay_seconds is small)
            from datetime import timedelta
            if delay_seconds > 0:
                job.scheduled_for = datetime.utcnow() + timedelta(seconds=delay_seconds)
            else:
                job.scheduled_for = None  # Immediate processing

            # Re-enqueue
            return await self.enqueue(job)

        except Exception as e:
            print(f"Error requeuing job: {e}")
            return False

    async def move_to_dead_letter(self, job: QueueJob) -> bool:
        """Move a permanently failed job to dead letter queue."""
        try:
            async with self._lock:
                self._dead_letter.append(job)
            return True

        except Exception as e:
            print(f"Error moving job to dead letter queue: {e}")
            return False

    async def get_queue_depth(self) -> int:
        """Get number of jobs waiting in queue."""
        try:
            async with self._lock:
                return len(self._main_queue)
        except Exception as e:
            print(f"Error getting queue depth: {e}")
            return 0

    async def get_dead_letter_count(self) -> int:
        """Get number of jobs in dead letter queue."""
        try:
            async with self._lock:
                return len(self._dead_letter)
        except Exception as e:
            print(f"Error getting dead letter count: {e}")
            return 0

    async def health_check(self) -> bool:
        """Check if queue is healthy (always true for memory queue)."""
        return True

    async def close(self) -> None:
        """Close queue (no-op for memory queue)."""
        pass
