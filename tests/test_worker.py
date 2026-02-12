"""
Tests for the background worker process.

Tests cover:
  - Worker initialization and lifecycle
  - Job processing (webhook_issue and deliberation types)
  - Retry logic and error handling
  - Graceful shutdown
  - Worker stats tracking
  - GitHub result posting
  - Concurrent job processing
"""

import asyncio
import signal
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.orchestration.worker import Worker, WorkerStats
from src.queue.base import QueueJob
from src.queue.memory_queue import MemoryQueue


# ========================
# Fixtures
# ========================


@pytest.fixture
def memory_queue():
    """Create an in-memory queue for testing."""
    return MemoryQueue(queue_name="test-jobs")


@pytest.fixture
def mock_registry():
    """Create a mock agent registry."""
    registry = MagicMock()
    registry.load_from_yaml.return_value = 10
    registry.get_all_agents.return_value = []
    registry.get_summary.return_value = {"categories": {}, "types": {}}
    return registry


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator."""
    orchestrator = AsyncMock()

    # Create a mock deliberation result
    mock_result = MagicMock()
    mock_result.total_rounds = 3
    mock_result.total_comments = 8
    mock_result.final_convergence_score = 0.85
    mock_result.termination_reason = "convergence"
    mock_result.agent_participation = {"ui_architect": 3, "frontend_dev": 2}
    mock_result.summary = "Test summary"
    mock_result.duration_seconds = 5.0
    mock_result.workflow.status.value = "completed"
    mock_result.workflow.conversation_history = []

    orchestrator.deliberate_on_issue.return_value = mock_result
    return orchestrator


@pytest.fixture
def worker(memory_queue, mock_registry, mock_orchestrator):
    """Create a worker with mocked dependencies."""
    return Worker(
        worker_id="test-worker-001",
        poll_interval=0.1,
        max_concurrent_jobs=1,
        queue=memory_queue,
        registry=mock_registry,
        orchestrator=mock_orchestrator,
    )


def make_webhook_job(
    issue_number: int = 42,
    title: str = "Add dark mode toggle",
    action: str = "opened",
) -> QueueJob:
    """Helper to create a test webhook issue job."""
    return QueueJob(
        job_id=f"test-job-{issue_number}",
        job_type="webhook_issue",
        payload={
            "event_type": "issues",
            "action": action,
            "issue_data": {
                "issue_number": issue_number,
                "title": title,
                "body": "Please add dark mode support",
                "labels": ["enhancement", "ui"],
                "html_url": f"https://github.com/test/repo/issues/{issue_number}",
            },
            "received_at": 1707350000.0,
        },
        priority=1,
        max_retries=3,
    )


# ========================
# WorkerStats Tests
# ========================


class TestWorkerStats:
    """Tests for WorkerStats tracking."""

    def test_initial_stats(self):
        """Stats start at zero."""
        stats = WorkerStats()
        assert stats.jobs_processed == 0
        assert stats.jobs_failed == 0
        assert stats.jobs_retried == 0
        assert stats.total_processing_time == 0.0
        assert stats.last_job_at is None
        assert stats.last_error is None

    def test_avg_processing_time_no_jobs(self):
        """Average processing time is 0 when no jobs processed."""
        stats = WorkerStats()
        assert stats.avg_processing_time == 0.0

    def test_avg_processing_time_with_jobs(self):
        """Average processing time calculated correctly."""
        stats = WorkerStats()
        stats.jobs_processed = 4
        stats.total_processing_time = 20.0
        assert stats.avg_processing_time == 5.0

    def test_uptime_seconds(self):
        """Uptime calculated from started_at."""
        stats = WorkerStats()
        # Should be at least 0 (just created)
        assert stats.uptime_seconds >= 0.0

    def test_to_dict(self):
        """Stats serialization includes all fields."""
        stats = WorkerStats()
        stats.jobs_processed = 5
        stats.jobs_failed = 1
        stats.jobs_retried = 2
        stats.total_processing_time = 25.0

        d = stats.to_dict()
        assert d["jobs_processed"] == 5
        assert d["jobs_failed"] == 1
        assert d["jobs_retried"] == 2
        assert d["avg_processing_time_seconds"] == 5.0
        assert "started_at" in d
        assert "uptime_seconds" in d


# ========================
# Worker Initialization Tests
# ========================


class TestWorkerInit:
    """Tests for Worker initialization."""

    def test_default_worker_id(self):
        """Worker auto-generates ID if not provided."""
        worker = Worker()
        assert worker.worker_id.startswith("worker-")
        assert len(worker.worker_id) > 7

    def test_custom_worker_id(self):
        """Worker uses provided ID."""
        worker = Worker(worker_id="my-worker")
        assert worker.worker_id == "my-worker"

    def test_default_configuration(self):
        """Worker has sensible defaults."""
        worker = Worker()
        assert worker.poll_interval == 2.0
        assert worker.max_concurrent_jobs == 1
        assert not worker.is_running
        assert not worker.is_shutting_down
        assert worker.active_job_count == 0

    def test_custom_configuration(self):
        """Worker accepts custom configuration."""
        worker = Worker(
            poll_interval=5.0,
            max_concurrent_jobs=3,
        )
        assert worker.poll_interval == 5.0
        assert worker.max_concurrent_jobs == 3


# ========================
# Job Processing Tests
# ========================


class TestJobProcessing:
    """Tests for job processing logic."""

    @pytest.mark.asyncio
    async def test_process_webhook_issue_job(self, worker, mock_orchestrator):
        """Worker correctly processes webhook_issue jobs."""
        job = make_webhook_job()
        await worker._process_job(job)

        # Orchestrator should be called with issue data
        mock_orchestrator.deliberate_on_issue.assert_called_once()
        call_args = mock_orchestrator.deliberate_on_issue.call_args[0][0]
        assert call_args["number"] == 42
        assert call_args["title"] == "Add dark mode toggle"
        assert call_args["labels"] == ["enhancement", "ui"]

    @pytest.mark.asyncio
    async def test_process_deliberation_job(self, worker, mock_orchestrator):
        """Worker correctly processes deliberation jobs."""
        job = QueueJob(
            job_id="test-deliberation-1",
            job_type="deliberation",
            payload={
                "issue": {
                    "number": 10,
                    "title": "Test deliberation",
                    "body": "Test body",
                    "labels": [],
                },
            },
        )
        await worker._process_job(job)
        mock_orchestrator.deliberate_on_issue.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_unknown_job_type(self, worker, mock_orchestrator):
        """Worker ignores unknown job types without crashing."""
        job = QueueJob(
            job_id="test-unknown-1",
            job_type="unknown_type",
            payload={},
        )
        await worker._process_job(job)
        mock_orchestrator.deliberate_on_issue.assert_not_called()

    @pytest.mark.asyncio
    async def test_stats_updated_on_success(self, worker):
        """Worker stats updated after successful job processing."""
        job = make_webhook_job()
        await worker._process_job(job)

        assert worker.stats.jobs_processed == 1
        assert worker.stats.jobs_failed == 0
        assert worker.stats.last_job_at is not None
        assert worker.stats.total_processing_time > 0

    @pytest.mark.asyncio
    async def test_stats_updated_on_failure(self, worker, mock_orchestrator):
        """Worker stats updated after failed job processing."""
        mock_orchestrator.deliberate_on_issue.side_effect = Exception("Test error")
        job = make_webhook_job()
        await worker._process_job(job)

        assert worker.stats.jobs_processed == 0
        assert worker.stats.jobs_failed == 1
        assert worker.stats.last_error == "Test error"

    @pytest.mark.asyncio
    async def test_active_jobs_tracking(self, worker, mock_orchestrator):
        """Active jobs are tracked during processing."""
        # Make the orchestrator slow so we can check active count
        async def slow_deliberation(*args, **kwargs):
            assert worker.active_job_count == 1
            return mock_orchestrator.deliberate_on_issue.return_value

        mock_orchestrator.deliberate_on_issue.side_effect = slow_deliberation

        job = make_webhook_job()
        await worker._process_job(job)

        # After processing, active count should be 0
        assert worker.active_job_count == 0


# ========================
# Retry Logic Tests
# ========================


class TestRetryLogic:
    """Tests for job retry and dead letter queue."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, worker, mock_orchestrator, memory_queue):
        """Failed jobs are retried with exponential backoff."""
        mock_orchestrator.deliberate_on_issue.side_effect = Exception("Transient error")

        job = make_webhook_job()
        job.max_retries = 3
        job.retry_count = 0

        await worker._process_job(job)

        # Job should be requeued (retry_count incremented by requeue)
        assert worker.stats.jobs_retried == 1
        assert worker.stats.jobs_failed == 1

    @pytest.mark.asyncio
    async def test_dead_letter_on_max_retries(self, worker, mock_orchestrator, memory_queue):
        """Jobs that exhaust retries go to dead letter queue."""
        mock_orchestrator.deliberate_on_issue.side_effect = Exception("Permanent error")

        job = make_webhook_job()
        job.max_retries = 3
        job.retry_count = 3  # Already at max

        await worker._process_job(job)

        # Should be in dead letter queue
        dead_count = await memory_queue.get_dead_letter_count()
        assert dead_count == 1
        assert worker.stats.jobs_retried == 0  # Not retried, moved to DLQ


# ========================
# Graceful Shutdown Tests
# ========================


class TestGracefulShutdown:
    """Tests for graceful shutdown behavior."""

    @pytest.mark.asyncio
    async def test_stop_sets_flags(self, worker):
        """Calling stop() sets shutdown flags."""
        worker._running = True
        await worker.stop()
        assert worker.is_shutting_down
        assert not worker.is_running

    @pytest.mark.asyncio
    async def test_double_stop_is_safe(self, worker):
        """Calling stop() twice doesn't cause errors."""
        worker._running = True
        await worker.stop()
        await worker.stop()
        assert worker.is_shutting_down

    @pytest.mark.asyncio
    async def test_poll_loop_exits_on_shutdown(self, worker, memory_queue):
        """Poll loop exits when shutdown is requested."""
        worker._running = True
        worker._queue = memory_queue

        # Schedule stop after a short delay
        async def delayed_stop():
            await asyncio.sleep(0.2)
            await worker.stop()

        asyncio.create_task(delayed_stop())

        # Poll loop should exit without hanging
        await asyncio.wait_for(worker._poll_loop(), timeout=2.0)

    @pytest.mark.asyncio
    async def test_poll_loop_processes_then_exits(self, worker, memory_queue, mock_orchestrator):
        """Poll loop processes available jobs before exiting."""
        job = make_webhook_job()
        await memory_queue.enqueue(job)

        worker._running = True
        worker._queue = memory_queue

        # Schedule stop after processing has time
        async def delayed_stop():
            await asyncio.sleep(0.3)
            await worker.stop()

        asyncio.create_task(delayed_stop())

        await asyncio.wait_for(worker._poll_loop(), timeout=3.0)

        # Job should have been processed
        assert worker.stats.jobs_processed == 1


# ========================
# Worker Status Tests
# ========================


class TestWorkerStatus:
    """Tests for worker status reporting."""

    def test_get_status(self, worker):
        """Status includes all expected fields."""
        status = worker.get_status()

        assert status["worker_id"] == "test-worker-001"
        assert status["running"] is False
        assert status["shutting_down"] is False
        assert status["active_jobs"] == 0
        assert status["queue_type"] == "MemoryQueue"
        assert "stats" in status

    def test_status_reflects_state(self, worker):
        """Status reflects current worker state."""
        worker._running = True
        worker._active_jobs.add("job-1")

        status = worker.get_status()
        assert status["running"] is True
        assert status["active_jobs"] == 1


# ========================
# GitHub Comment Formatting Tests
# ========================


class TestGitHubCommentFormatting:
    """Tests for formatting deliberation results as GitHub comments."""

    def test_format_comment_structure(self, worker, mock_orchestrator):
        """GitHub comment has expected structure."""
        result = mock_orchestrator.deliberate_on_issue.return_value
        comment = worker._format_github_comment(result)

        assert "## Multi-Agent Deliberation Results" in comment
        assert "**Rounds:** 3" in comment
        assert "**Comments:** 8" in comment
        assert "**Convergence:** 85%" in comment
        assert "### Agent Participation" in comment
        assert "**ui_architect**" in comment
        assert "### Summary" in comment
        assert "Test summary" in comment

    def test_format_comment_no_participation(self, worker):
        """Comment handles empty participation gracefully."""
        result = MagicMock()
        result.total_rounds = 0
        result.total_comments = 0
        result.final_convergence_score = 0.0
        result.termination_reason = "no agents"
        result.agent_participation = {}
        result.summary = ""
        result.duration_seconds = 0.0
        result.workflow.status.value = "completed"

        comment = worker._format_github_comment(result)
        assert "## Multi-Agent Deliberation Results" in comment
        assert "Agent Participation" not in comment


# ========================
# Integration-style Tests
# ========================


class TestWorkerIntegration:
    """Integration-style tests for the worker with queue."""

    @pytest.mark.asyncio
    async def test_enqueue_and_process(self, worker, memory_queue, mock_orchestrator):
        """Worker picks up and processes jobs from queue."""
        # Enqueue two jobs
        job1 = make_webhook_job(issue_number=1, title="First issue")
        job2 = make_webhook_job(issue_number=2, title="Second issue")
        await memory_queue.enqueue(job1)
        await memory_queue.enqueue(job2)

        # Process both jobs
        worker._running = True
        worker._queue = memory_queue

        async def stop_after_processing():
            # Wait until both jobs are processed
            for _ in range(50):
                if worker.stats.jobs_processed >= 2:
                    await worker.stop()
                    return
                await asyncio.sleep(0.1)
            await worker.stop()

        asyncio.create_task(stop_after_processing())
        await asyncio.wait_for(worker._poll_loop(), timeout=5.0)

        assert worker.stats.jobs_processed == 2
        assert mock_orchestrator.deliberate_on_issue.call_count == 2

    @pytest.mark.asyncio
    async def test_queue_empty_then_receives_job(self, worker, memory_queue, mock_orchestrator):
        """Worker waits on empty queue, then processes when job arrives."""
        worker._running = True
        worker._queue = memory_queue

        async def add_job_later():
            await asyncio.sleep(0.3)
            job = make_webhook_job()
            await memory_queue.enqueue(job)

            # Wait for processing
            for _ in range(30):
                if worker.stats.jobs_processed >= 1:
                    await worker.stop()
                    return
                await asyncio.sleep(0.1)
            await worker.stop()

        asyncio.create_task(add_job_later())
        await asyncio.wait_for(worker._poll_loop(), timeout=5.0)

        assert worker.stats.jobs_processed == 1
