"""
Queue factory for creating appropriate queue implementation.

This module provides a factory function that automatically selects
the best available queue backend (Redis preferred, memory fallback).
"""

import os
from typing import Optional
from .base import BaseQueue
from .redis_queue import RedisQueue
from .memory_queue import MemoryQueue

from src.utils.logging import get_logger

logger = get_logger(__name__)


async def create_queue(
    redis_url: Optional[str] = None,
    queue_name: str = "jobs",
    fallback_to_memory: bool = True,
) -> BaseQueue:
    """
    Create a queue instance, preferring Redis but falling back to memory if needed.

    Args:
        redis_url: Redis connection URL (defaults to REDIS_URL env var)
        queue_name: Name for the queue
        fallback_to_memory: Whether to fall back to memory queue if Redis fails

    Returns:
        Queue instance (Redis or Memory)

    Raises:
        ConnectionError: If Redis fails and fallback_to_memory is False
    """
    # Get Redis URL from parameter or environment
    if redis_url is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Try to create Redis queue
    try:
        redis_queue = RedisQueue(redis_url=redis_url, queue_name=queue_name)

        # Test connection
        if await redis_queue.health_check():
            logger.info("queue_connected", queue_type="redis", redis_url=redis_url)
            return redis_queue

        # Redis not available
        if fallback_to_memory:
            logger.warning("queue_fallback", reason="redis_unavailable", queue_type="memory")
            return MemoryQueue(queue_name=queue_name)
        else:
            raise ConnectionError(f"Failed to connect to Redis at {redis_url}")

    except Exception as e:
        if fallback_to_memory:
            logger.warning("queue_fallback", reason=str(e), queue_type="memory")
            return MemoryQueue(queue_name=queue_name)
        else:
            raise ConnectionError(f"Failed to connect to Redis: {e}")


def get_queue_sync(
    redis_url: Optional[str] = None,
    queue_name: str = "jobs",
) -> BaseQueue:
    """
    Synchronous wrapper for development/testing.

    Note: This returns a MemoryQueue since we can't do async initialization.
    For production, use create_queue() in an async context.

    Args:
        redis_url: Ignored (for signature compatibility)
        queue_name: Name for the queue

    Returns:
        In-memory queue instance
    """
    logger.warning("queue_sync_fallback", queue_type="memory")
    return MemoryQueue(queue_name=queue_name)
