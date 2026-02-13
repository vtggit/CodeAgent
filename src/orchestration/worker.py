"""
Background Worker Process for the Multi-Agent GitHub Issue Routing System.

This module implements a worker that consumes jobs from the message queue
(Redis or in-memory fallback) and triggers multi-agent deliberations via
the orchestration engine.

The worker runs as a standalone process, decoupled from the webhook server,
enabling independent scaling of webhook reception and job processing.

Features:
  - Polls queue for new jobs with configurable interval
  - Graceful shutdown on SIGTERM/SIGINT
  - Job retry with exponential backoff
  - Configurable concurrency (sequential or parallel)
  - Health monitoring via status endpoint
  - Posts results back to GitHub issues (when configured)

Usage:
  python -m src.orchestration.worker

  Environment variables:
    WORKER_POLL_INTERVAL  - Seconds between queue polls (default: 2)
    WORKER_CONCURRENCY    - Max concurrent jobs (default: 1)
    WORKER_ID             - Unique worker identifier (auto-generated)
"""

import asyncio
import logging
import os
import signal
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.registry import AgentRegistry
from src.config.settings import get_settings
from src.database.engine import init_db, close_db
from src.orchestration.deliberation import (
    DeliberationResult,
    MultiAgentDeliberationOrchestrator,
)
from src.orchestration.moderator import ModeratorAgent
from src.integrations.github_client import GitHubClient, get_github_client, close_github_client
from src.integrations.result_poster import ResultFormatter, ResultPoster
from src.queue.base import BaseQueue, QueueJob
from src.queue.factory import create_queue

logger = logging.getLogger(__name__)


class WorkerStats:
    """Tracks worker performance metrics."""

    def __init__(self) -> None:
        self.started_at: datetime = datetime.utcnow()
        self.jobs_processed: int = 0
        self.jobs_failed: int = 0
        self.jobs_retried: int = 0
        self.total_processing_time: float = 0.0
        self.last_job_at: Optional[datetime] = None
        self.last_error: Optional[str] = None

    @property
    def uptime_seconds(self) -> float:
        """How long the worker has been running."""
        return (datetime.utcnow() - self.started_at).total_seconds()

    @property
    def avg_processing_time(self) -> float:
        """Average job processing time in seconds."""
        if self.jobs_processed == 0:
            return 0.0
        return self.total_processing_time / self.jobs_processed

    def to_dict(self) -> dict[str, Any]:
        """Serialize stats to a dictionary."""
        return {
            "started_at": self.started_at.isoformat(),
            "uptime_seconds": round(self.uptime_seconds, 1),
            "jobs_processed": self.jobs_processed,
            "jobs_failed": self.jobs_failed,
            "jobs_retried": self.jobs_retried,
            "avg_processing_time_seconds": round(self.avg_processing_time, 2),
            "total_processing_time_seconds": round(self.total_processing_time, 2),
            "last_job_at": self.last_job_at.isoformat() if self.last_job_at else None,
            "last_error": self.last_error,
        }


class Worker:
    """
    Background job processor for the Multi-Agent deliberation system.

    Consumes jobs from the message queue and triggers deliberations
    via the MultiAgentDeliberationOrchestrator. Runs independently
    from the webhook server for scalable processing.

    Example:
        worker = Worker()
        await worker.start()
    """

    def __init__(
        self,
        worker_id: Optional[str] = None,
        poll_interval: float = 2.0,
        max_concurrent_jobs: int = 1,
        queue: Optional[BaseQueue] = None,
        registry: Optional[AgentRegistry] = None,
        orchestrator: Optional[MultiAgentDeliberationOrchestrator] = None,
    ) -> None:
        """
        Initialize the worker.

        Args:
            worker_id: Unique identifier for this worker instance
            poll_interval: Seconds between queue polls
            max_concurrent_jobs: Maximum number of concurrent jobs (1 = sequential)
            queue: Pre-configured queue instance (auto-created if None)
            registry: Pre-configured agent registry (auto-loaded if None)
            orchestrator: Pre-configured orchestrator (auto-created if None)
        """
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.poll_interval = poll_interval
        self.max_concurrent_jobs = max_concurrent_jobs

        # Components (initialized during start())
        self._queue = queue
        self._registry = registry
        self._orchestrator = orchestrator

        # State
        self._running = False
        self._shutting_down = False
        self._active_jobs: set[str] = set()
        self._semaphore: Optional[asyncio.Semaphore] = None
        self.stats = WorkerStats()

        self._logger = logging.getLogger(f"{__name__}.{self.worker_id}")

    @property
    def is_running(self) -> bool:
        """Whether the worker is currently running."""
        return self._running

    @property
    def is_shutting_down(self) -> bool:
        """Whether the worker is in the process of shutting down."""
        return self._shutting_down

    @property
    def active_job_count(self) -> int:
        """Number of jobs currently being processed."""
        return len(self._active_jobs)

    # ========================
    # Lifecycle Management
    # ========================

    async def start(self) -> None:
        """
        Start the worker process.

        Initializes all components and begins polling the queue.
        Blocks until shutdown is requested.
        """
        self._logger.info(
            "Starting worker %s (poll_interval=%.1fs, concurrency=%d)",
            self.worker_id,
            self.poll_interval,
            self.max_concurrent_jobs,
        )

        try:
            # Initialize components
            await self._initialize()

            # Set up signal handlers
            self._setup_signal_handlers()

            # Start processing loop
            self._running = True
            self._logger.info("Worker %s is ready and polling for jobs", self.worker_id)

            await self._poll_loop()

        except Exception as e:
            self._logger.error("Worker failed to start: %s", str(e), exc_info=True)
            raise
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """
        Request graceful shutdown.

        Sets the shutting_down flag. The poll loop will finish
        processing any active jobs before exiting.
        """
        if self._shutting_down:
            return

        self._logger.info(
            "Graceful shutdown requested for worker %s "
            "(active jobs: %d)",
            self.worker_id,
            self.active_job_count,
        )

        self._shutting_down = True
        self._running = False

    async def _initialize(self) -> None:
        """Initialize all worker components."""
        settings = get_settings()

        # Initialize database
        self._logger.info("Initializing database...")
        try:
            await init_db()
            self._logger.info("Database initialized")
        except Exception as e:
            self._logger.warning("Database initialization error: %s", str(e))

        # Initialize queue
        if self._queue is None:
            self._logger.info("Connecting to message queue...")
            self._queue = await create_queue(
                redis_url=settings.redis_url,
                queue_name="multi-agent-jobs",
                fallback_to_memory=True,
            )
            self._logger.info(
                "Queue connected: %s", type(self._queue).__name__
            )

        # Initialize agent registry
        if self._registry is None:
            self._logger.info("Loading agent registry...")
            self._registry = AgentRegistry()
            config_path = str(
                Path(__file__).parent.parent.parent / "config" / "agent_definitions.yaml"
            )
            try:
                count = self._registry.load_from_yaml(config_path)
                self._logger.info("Agent registry loaded: %d agents", count)
            except Exception as e:
                self._logger.error("Failed to load agent registry: %s", str(e))
                raise

        # Initialize orchestrator
        if self._orchestrator is None:
            moderator = ModeratorAgent(registry=self._registry)
            self._orchestrator = MultiAgentDeliberationOrchestrator(
                registry=self._registry,
                moderator=moderator,
                persist_to_db=True,
            )
            self._logger.info("Orchestrator initialized")

        # Initialize concurrency semaphore
        self._semaphore = asyncio.Semaphore(self.max_concurrent_jobs)

    async def _cleanup(self) -> None:
        """Clean up resources on shutdown."""
        self._logger.info("Cleaning up worker resources...")

        # Wait for active jobs to complete (with timeout)
        if self._active_jobs:
            self._logger.info(
                "Waiting for %d active job(s) to complete...",
                len(self._active_jobs),
            )
            # Give active jobs 30 seconds to finish
            for _ in range(30):
                if not self._active_jobs:
                    break
                await asyncio.sleep(1)

            if self._active_jobs:
                self._logger.warning(
                    "Forcing shutdown with %d active job(s)",
                    len(self._active_jobs),
                )

        # Close queue connection
        if self._queue:
            try:
                await self._queue.close()
                self._logger.info("Queue connection closed")
            except Exception as e:
                self._logger.warning("Error closing queue: %s", str(e))

        # Close database
        try:
            await close_db()
            self._logger.info("Database connection closed")
        except Exception as e:
            self._logger.warning("Error closing database: %s", str(e))

        self._running = False
        self._logger.info(
            "Worker %s shutdown complete. Stats: %s",
            self.worker_id,
            self.stats.to_dict(),
        )

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()

        def _handle_signal(sig: signal.Signals) -> None:
            self._logger.info("Received signal %s", sig.name)
            asyncio.ensure_future(self.stop())

        try:
            loop.add_signal_handler(signal.SIGTERM, lambda: _handle_signal(signal.SIGTERM))
            loop.add_signal_handler(signal.SIGINT, lambda: _handle_signal(signal.SIGINT))
            self._logger.debug("Signal handlers registered (SIGTERM, SIGINT)")
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            self._logger.warning(
                "Signal handlers not supported on this platform"
            )

    # ========================
    # Main Processing Loop
    # ========================

    async def _poll_loop(self) -> None:
        """
        Main loop that polls the queue and processes jobs.

        Runs until stop() is called. Uses a semaphore to limit
        concurrent job processing.
        """
        while self._running and not self._shutting_down:
            try:
                # Check queue health periodically
                if not await self._queue.health_check():
                    self._logger.warning("Queue health check failed, retrying...")
                    await asyncio.sleep(self.poll_interval * 2)
                    continue

                # Try to dequeue a job
                job = await self._queue.dequeue(timeout=0)

                if job is None:
                    # No job available, wait and try again
                    await asyncio.sleep(self.poll_interval)
                    continue

                # Process the job (respecting concurrency limits)
                if self.max_concurrent_jobs > 1:
                    # Concurrent mode: spawn task
                    asyncio.create_task(self._process_with_semaphore(job))
                else:
                    # Sequential mode: process inline
                    await self._process_job(job)

            except asyncio.CancelledError:
                self._logger.info("Poll loop cancelled")
                break
            except Exception as e:
                self._logger.error(
                    "Unexpected error in poll loop: %s",
                    str(e),
                    exc_info=True,
                )
                # Wait before retrying to prevent tight error loops
                await asyncio.sleep(self.poll_interval * 3)

    async def _process_with_semaphore(self, job: QueueJob) -> None:
        """Process a job with concurrency limiting via semaphore."""
        async with self._semaphore:
            await self._process_job(job)

    # ========================
    # Job Processing
    # ========================

    async def _process_job(self, job: QueueJob) -> None:
        """
        Process a single job from the queue.

        Routes the job to the appropriate handler based on job_type,
        handles errors, and manages retries.

        Args:
            job: The queue job to process
        """
        job_id = job.job_id
        self._active_jobs.add(job_id)
        start_time = time.time()

        self._logger.info(
            "Processing job %s (type=%s, retry=%d/%d)",
            job_id,
            job.job_type,
            job.retry_count,
            job.max_retries,
        )

        try:
            # Route to appropriate handler
            if job.job_type == "webhook_issue":
                await self._handle_webhook_issue(job)
            elif job.job_type == "deliberation":
                await self._handle_deliberation(job)
            else:
                self._logger.warning(
                    "Unknown job type: %s (job_id=%s)",
                    job.job_type,
                    job_id,
                )
                return

            # Job completed successfully
            elapsed = time.time() - start_time
            self.stats.jobs_processed += 1
            self.stats.total_processing_time += elapsed
            self.stats.last_job_at = datetime.utcnow()

            self._logger.info(
                "Job %s completed in %.1fs",
                job_id,
                elapsed,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            self.stats.jobs_failed += 1
            self.stats.last_error = str(e)

            self._logger.error(
                "Job %s failed after %.1fs: %s",
                job_id,
                elapsed,
                str(e),
                exc_info=True,
            )

            # Retry the job if retries remain
            await self._retry_job(job, str(e))

        finally:
            self._active_jobs.discard(job_id)

    async def _retry_job(self, job: QueueJob, error: str) -> None:
        """
        Retry a failed job with exponential backoff.

        Args:
            job: The failed job
            error: Error message from the failure
        """
        if job.retry_count >= job.max_retries:
            self._logger.warning(
                "Job %s exhausted all retries (%d). Moving to dead letter queue.",
                job.job_id,
                job.max_retries,
            )
            await self._queue.move_to_dead_letter(job)
            return

        # Calculate backoff delay: 2^retry * 30 seconds, max 15 minutes
        delay = min(2 ** job.retry_count * 30, 900)
        self._logger.info(
            "Retrying job %s in %ds (attempt %d/%d)",
            job.job_id,
            delay,
            job.retry_count + 1,
            job.max_retries,
        )

        success = await self._queue.requeue(job, delay_seconds=delay)
        if success:
            self.stats.jobs_retried += 1
        else:
            self._logger.error(
                "Failed to requeue job %s",
                job.job_id,
            )

    # ========================
    # Job Handlers
    # ========================

    async def _handle_webhook_issue(self, job: QueueJob) -> None:
        """
        Handle a webhook_issue job by triggering deliberation.

        Extracts issue data from the job payload and passes it
        to the orchestrator for multi-agent deliberation.

        Args:
            job: Queue job with webhook issue payload
        """
        payload = job.payload
        issue_data = payload.get("issue_data", {})
        action = payload.get("action", "unknown")

        issue_number = issue_data.get("issue_number", 0)
        issue_title = issue_data.get("title", "Unknown")

        self._logger.info(
            "Handling webhook issue: #%d - %s (action=%s)",
            issue_number,
            issue_title[:80],
            action,
        )

        # Build issue dict for orchestrator
        issue = {
            "number": issue_number,
            "title": issue_title,
            "body": issue_data.get("body", ""),
            "labels": issue_data.get("labels", []),
            "html_url": issue_data.get("html_url"),
            "url": issue_data.get("url"),
            "repository": issue_data.get("repository"),
        }

        # Run deliberation
        result = await self._orchestrator.deliberate_on_issue(issue)

        self._logger.info(
            "Deliberation complete for issue #%d: %d rounds, %d comments, "
            "convergence=%.2f, reason=%s",
            issue_number,
            result.total_rounds,
            result.total_comments,
            result.final_convergence_score,
            result.termination_reason,
        )

        # Post results back to GitHub (if configured)
        await self._post_results_to_github(issue_number, result)

    async def _handle_deliberation(self, job: QueueJob) -> None:
        """
        Handle a direct deliberation job.

        This is for manually triggered deliberations (e.g., via API).

        Args:
            job: Queue job with deliberation payload
        """
        payload = job.payload
        issue = payload.get("issue", {})

        if not issue:
            self._logger.warning("Deliberation job missing issue data")
            return

        result = await self._orchestrator.deliberate_on_issue(issue)

        self._logger.info(
            "Direct deliberation complete: %d rounds, %d comments",
            result.total_rounds,
            result.total_comments,
        )

    async def _post_results_to_github(
        self,
        issue_number: int,
        result: DeliberationResult,
    ) -> None:
        """
        Post deliberation results back to the GitHub issue as a comment.

        Uses the ResultPoster for comprehensive Markdown formatting and
        GitHubClient for robust API interaction with automatic rate limit
        handling and retries.

        Args:
            issue_number: GitHub issue number
            result: Deliberation result to post
        """
        poster = ResultPoster()
        repo_name = os.getenv("GITHUB_REPOSITORY")

        comment = poster.post_results(repo_name, issue_number, result)

        if comment:
            self._logger.info(
                "Posted deliberation results to issue #%d (comment_id=%d)",
                issue_number,
                comment.id,
            )
        else:
            self._logger.debug(
                "Result posting skipped for issue #%d (client not available or error)",
                issue_number,
            )

    # ========================
    # Health & Status
    # ========================

    def get_status(self) -> dict[str, Any]:
        """
        Get the current worker status.

        Returns:
            Dictionary with worker status information
        """
        return {
            "worker_id": self.worker_id,
            "running": self._running,
            "shutting_down": self._shutting_down,
            "active_jobs": self.active_job_count,
            "queue_type": type(self._queue).__name__ if self._queue else None,
            "stats": self.stats.to_dict(),
        }


# ========================
# Entry Point
# ========================


async def run_worker() -> None:
    """
    Run the worker process.

    Reads configuration from environment variables and starts
    the worker. This is the main entry point for the worker process.
    """
    # Configure logging
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Read configuration
    poll_interval = float(os.getenv("WORKER_POLL_INTERVAL", "2"))
    concurrency = int(os.getenv("WORKER_CONCURRENCY", "1"))
    worker_id = os.getenv("WORKER_ID")

    logger.info("=" * 60)
    logger.info("Multi-Agent GitHub Issue Routing System - Worker")
    logger.info("=" * 60)

    worker = Worker(
        worker_id=worker_id,
        poll_interval=poll_interval,
        max_concurrent_jobs=concurrency,
    )

    await worker.start()


def main() -> None:
    """Synchronous entry point for the worker process."""
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
    except Exception as e:
        logger.error("Worker crashed: %s", str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
