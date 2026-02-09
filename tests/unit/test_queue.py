"""
Unit tests for message queue implementations.

Tests both Redis and Memory queue implementations to ensure
they follow the BaseQueue interface correctly.
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from src.queue.base import QueueJob
from src.queue.memory_queue import MemoryQueue
from src.queue.redis_queue import RedisQueue


@pytest.fixture
def memory_queue():
    """Create a memory queue for testing."""
    return MemoryQueue(queue_name="test-queue")


@pytest.fixture
async def redis_queue():
    """Create a Redis queue for testing (skip if Redis not available)."""
    queue = RedisQueue(
        redis_url="redis://localhost:6379/15",  # Use test database
        queue_name="test-queue"
    )

    # Check if Redis is available
    try:
        if await queue.health_check():
            yield queue
            # Cleanup: clear test queues
            if queue._redis:
                await queue._redis.delete(queue.main_queue_key)
                await queue._redis.delete(queue.dead_letter_key)
                await queue._redis.delete(queue.delayed_key)
            await queue.close()
        else:
            pytest.skip("Redis not available")
    except Exception:
        pytest.skip("Redis not available")


class TestQueueJob:
    """Test QueueJob serialization and deserialization."""

    def test_job_creation(self):
        """Test creating a job."""
        job = QueueJob(
            job_id="test-123",
            job_type="webhook_issue",
            payload={"issue_number": 42, "title": "Test"},
            priority=5,
        )

        assert job.job_id == "test-123"
        assert job.job_type == "webhook_issue"
        assert job.payload["issue_number"] == 42
        assert job.priority == 5
        assert job.max_retries == 3
        assert job.retry_count == 0

    def test_job_serialization(self):
        """Test job to_dict and from_dict."""
        original = QueueJob(
            job_id="test-456",
            job_type="deliberation",
            payload={"workflow_id": "abc"},
            priority=10,
            max_retries=5,
            retry_count=2,
        )

        # Serialize
        data = original.to_dict()
        assert data["job_id"] == "test-456"
        assert data["job_type"] == "deliberation"
        assert data["priority"] == 10
        assert data["retry_count"] == 2

        # Deserialize
        restored = QueueJob.from_dict(data)
        assert restored.job_id == original.job_id
        assert restored.job_type == original.job_type
        assert restored.payload == original.payload
        assert restored.priority == original.priority
        assert restored.retry_count == original.retry_count


class TestMemoryQueue:
    """Test in-memory queue implementation."""

    @pytest.mark.asyncio
    async def test_enqueue_dequeue(self, memory_queue):
        """Test basic enqueue and dequeue operations."""
        job = QueueJob(
            job_id="test-1",
            job_type="test",
            payload={"data": "value"},
        )

        # Enqueue
        success = await memory_queue.enqueue(job)
        assert success is True

        # Check queue depth
        depth = await memory_queue.get_queue_depth()
        assert depth == 1

        # Dequeue
        retrieved = await memory_queue.dequeue()
        assert retrieved is not None
        assert retrieved.job_id == "test-1"
        assert retrieved.payload["data"] == "value"

        # Queue should be empty now
        depth = await memory_queue.get_queue_depth()
        assert depth == 0

    @pytest.mark.asyncio
    async def test_priority_ordering(self, memory_queue):
        """Test that jobs are dequeued by priority."""
        # Add jobs with different priorities
        job1 = QueueJob(job_id="low", job_type="test", payload={}, priority=1)
        job2 = QueueJob(job_id="high", job_type="test", payload={}, priority=10)
        job3 = QueueJob(job_id="med", job_type="test", payload={}, priority=5)

        await memory_queue.enqueue(job1)
        await memory_queue.enqueue(job2)
        await memory_queue.enqueue(job3)

        # Should dequeue in priority order: high, med, low
        first = await memory_queue.dequeue()
        assert first.job_id == "high"

        second = await memory_queue.dequeue()
        assert second.job_id == "med"

        third = await memory_queue.dequeue()
        assert third.job_id == "low"

    @pytest.mark.asyncio
    async def test_requeue_with_retry(self, memory_queue):
        """Test requeuing failed jobs increments retry count."""
        job = QueueJob(
            job_id="retry-test",
            job_type="test",
            payload={},
            max_retries=3,
            retry_count=0,
        )

        await memory_queue.enqueue(job)
        retrieved = await memory_queue.dequeue()

        # Requeue should increment retry count
        success = await memory_queue.requeue(retrieved)
        assert success is True
        assert retrieved.retry_count == 1

    @pytest.mark.asyncio
    async def test_dead_letter_queue(self, memory_queue):
        """Test moving exhausted jobs to dead letter queue."""
        job = QueueJob(
            job_id="dlq-test",
            job_type="test",
            payload={},
            max_retries=2,
            retry_count=2,  # Already at max
        )

        await memory_queue.enqueue(job)
        retrieved = await memory_queue.dequeue()

        # This should move to DLQ
        success = await memory_queue.requeue(retrieved)
        assert success is True

        # Check DLQ count
        dlq_count = await memory_queue.get_dead_letter_count()
        assert dlq_count == 1

        # Main queue should be empty
        depth = await memory_queue.get_queue_depth()
        assert depth == 0

    @pytest.mark.asyncio
    async def test_delayed_jobs(self, memory_queue):
        """Test scheduling jobs for future processing."""
        job = QueueJob(
            job_id="delayed-test",
            job_type="test",
            payload={},
            scheduled_for=datetime.utcnow() + timedelta(seconds=2),
        )

        await memory_queue.enqueue(job)

        # Should not be available immediately
        retrieved = await memory_queue.dequeue()
        assert retrieved is None

        # Wait for delay to pass
        await asyncio.sleep(2.1)

        # Now should be available
        retrieved = await memory_queue.dequeue()
        assert retrieved is not None
        assert retrieved.job_id == "delayed-test"

    @pytest.mark.asyncio
    async def test_health_check(self, memory_queue):
        """Test health check always returns True for memory queue."""
        healthy = await memory_queue.health_check()
        assert healthy is True


class TestRedisQueue:
    """Test Redis queue implementation (requires Redis)."""

    @pytest.mark.asyncio
    async def test_enqueue_dequeue(self, redis_queue):
        """Test basic enqueue and dequeue operations."""
        job = QueueJob(
            job_id="redis-test-1",
            job_type="test",
            payload={"data": "value"},
        )

        # Enqueue
        success = await redis_queue.enqueue(job)
        assert success is True

        # Check queue depth
        depth = await redis_queue.get_queue_depth()
        assert depth == 1

        # Dequeue
        retrieved = await redis_queue.dequeue()
        assert retrieved is not None
        assert retrieved.job_id == "redis-test-1"
        assert retrieved.payload["data"] == "value"

        # Queue should be empty now
        depth = await redis_queue.get_queue_depth()
        assert depth == 0

    @pytest.mark.asyncio
    async def test_priority_ordering(self, redis_queue):
        """Test that jobs are dequeued by priority."""
        # Add jobs with different priorities
        job1 = QueueJob(job_id="low", job_type="test", payload={}, priority=1)
        job2 = QueueJob(job_id="high", job_type="test", payload={}, priority=10)
        job3 = QueueJob(job_id="med", job_type="test", payload={}, priority=5)

        await redis_queue.enqueue(job1)
        await redis_queue.enqueue(job2)
        await redis_queue.enqueue(job3)

        # Should dequeue in priority order: high, med, low
        first = await redis_queue.dequeue()
        assert first.job_id == "high"

        second = await redis_queue.dequeue()
        assert second.job_id == "med"

        third = await redis_queue.dequeue()
        assert third.job_id == "low"

    @pytest.mark.asyncio
    async def test_health_check(self, redis_queue):
        """Test health check returns True when Redis is connected."""
        healthy = await redis_queue.health_check()
        assert healthy is True


