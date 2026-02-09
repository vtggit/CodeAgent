"""
Redis-based message queue implementation.

This module implements the BaseQueue interface using Redis as the backend,
with support for priority queues, delayed jobs, and dead letter queues.
"""

import json
import asyncio
from typing import Optional
from datetime import datetime, timedelta
import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError

from .base import BaseQueue, QueueJob


class RedisQueue(BaseQueue):
    """Redis implementation of message queue."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        queue_name: str = "jobs",
        max_connections: int = 10,
    ):
        """
        Initialize Redis queue.

        Args:
            redis_url: Redis connection URL
            queue_name: Name prefix for queue keys
            max_connections: Maximum Redis connection pool size
        """
        self.redis_url = redis_url
        self.queue_name = queue_name
        self.max_connections = max_connections
        self._redis: Optional[Redis] = None
        self._connected = False

        # Queue key names
        self.main_queue_key = f"{queue_name}:main"
        self.processing_key = f"{queue_name}:processing"
        self.dead_letter_key = f"{queue_name}:dlq"
        self.delayed_key = f"{queue_name}:delayed"

    async def _ensure_connected(self) -> Redis:
        """Ensure Redis connection is established."""
        if self._redis is None or not self._connected:
            try:
                self._redis = await redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=self.max_connections,
                )
                # Test connection
                await self._redis.ping()
                self._connected = True
            except RedisError as e:
                self._connected = False
                raise ConnectionError(f"Failed to connect to Redis: {e}")

        return self._redis

    async def enqueue(self, job: QueueJob) -> bool:
        """
        Add a job to the queue.

        Jobs are stored in a sorted set with priority as score (higher = processed first).
        """
        try:
            redis_client = await self._ensure_connected()

            # Serialize job
            job_data = json.dumps(job.to_dict())

            # If job is scheduled for future, add to delayed queue
            if job.scheduled_for and job.scheduled_for > datetime.utcnow():
                timestamp = job.scheduled_for.timestamp()
                await redis_client.zadd(self.delayed_key, {job_data: timestamp})
            else:
                # Add to main queue with priority as score
                await redis_client.zadd(self.main_queue_key, {job_data: -job.priority})

            return True

        except Exception as e:
            print(f"Error enqueueing job: {e}")
            return False

    async def dequeue(self, timeout: int = 0) -> Optional[QueueJob]:
        """
        Remove and return highest priority job from queue.

        This uses ZPOPMAX to atomically get the highest priority job.
        """
        try:
            redis_client = await self._ensure_connected()

            # First, check for delayed jobs that are ready
            await self._process_delayed_jobs()

            # Pop highest priority job (ZPOPMAX returns highest score)
            # Since we use negative priority, we need ZPOPMIN
            result = await redis_client.zpopmin(self.main_queue_key, count=1)

            if not result:
                return None

            # Result is list of (member, score) tuples
            job_data, _ = result[0]

            # Deserialize job
            job_dict = json.loads(job_data)
            job = QueueJob.from_dict(job_dict)

            # Add to processing set for tracking
            await redis_client.zadd(
                self.processing_key,
                {job_data: datetime.utcnow().timestamp()}
            )

            return job

        except Exception as e:
            print(f"Error dequeuing job: {e}")
            return None

    async def _process_delayed_jobs(self) -> None:
        """Move delayed jobs that are ready to main queue."""
        try:
            redis_client = await self._ensure_connected()
            now = datetime.utcnow().timestamp()

            # Get all delayed jobs that are ready (score <= now)
            ready_jobs = await redis_client.zrangebyscore(
                self.delayed_key,
                min=0,
                max=now,
                withscores=False
            )

            if ready_jobs:
                # Remove from delayed queue and add to main queue
                for job_data in ready_jobs:
                    job_dict = json.loads(job_data)
                    job = QueueJob.from_dict(job_dict)

                    # Remove from delayed queue
                    await redis_client.zrem(self.delayed_key, job_data)

                    # Add to main queue
                    await redis_client.zadd(
                        self.main_queue_key,
                        {job_data: -job.priority}
                    )

        except Exception as e:
            print(f"Error processing delayed jobs: {e}")

    async def requeue(self, job: QueueJob, delay_seconds: int = 0) -> bool:
        """
        Re-add a failed job to the queue with incremented retry count.

        Uses exponential backoff for delays if not specified.
        """
        try:
            redis_client = await self._ensure_connected()

            # Increment retry count
            job.retry_count += 1

            # Check if job has exhausted retries
            if job.retry_count > job.max_retries:
                return await self.move_to_dead_letter(job)

            # Calculate delay with exponential backoff
            if delay_seconds == 0:
                # Exponential backoff: 2^retry_count minutes
                delay_seconds = min(2 ** job.retry_count * 60, 3600)  # Max 1 hour

            # Schedule for future processing
            job.scheduled_for = datetime.utcnow() + timedelta(seconds=delay_seconds)

            # Remove from processing set
            job_data = json.dumps(job.to_dict())
            await redis_client.zrem(self.processing_key, job_data)

            # Re-enqueue
            return await self.enqueue(job)

        except Exception as e:
            print(f"Error requeuing job: {e}")
            return False

    async def move_to_dead_letter(self, job: QueueJob) -> bool:
        """Move a permanently failed job to dead letter queue."""
        try:
            redis_client = await self._ensure_connected()

            # Serialize job
            job_data = json.dumps(job.to_dict())

            # Add to dead letter queue with timestamp
            await redis_client.zadd(
                self.dead_letter_key,
                {job_data: datetime.utcnow().timestamp()}
            )

            # Remove from processing set if present
            await redis_client.zrem(self.processing_key, job_data)

            return True

        except Exception as e:
            print(f"Error moving job to dead letter queue: {e}")
            return False

    async def get_queue_depth(self) -> int:
        """Get number of jobs waiting in queue."""
        try:
            redis_client = await self._ensure_connected()
            count = await redis_client.zcard(self.main_queue_key)
            return count or 0
        except Exception as e:
            print(f"Error getting queue depth: {e}")
            return 0

    async def get_dead_letter_count(self) -> int:
        """Get number of jobs in dead letter queue."""
        try:
            redis_client = await self._ensure_connected()
            count = await redis_client.zcard(self.dead_letter_key)
            return count or 0
        except Exception as e:
            print(f"Error getting dead letter count: {e}")
            return 0

    async def health_check(self) -> bool:
        """Check if Redis is healthy."""
        try:
            redis_client = await self._ensure_connected()
            await redis_client.ping()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._connected = False
