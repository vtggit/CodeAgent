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
            print(f"✓ Connected to Redis queue: {redis_url}")
            return redis_queue

        # Redis not available
        if fallback_to_memory:
            print(f"⚠ Redis not available, using in-memory queue (dev mode)")
            return MemoryQueue(queue_name=queue_name)
        else:
            raise ConnectionError(f"Failed to connect to Redis at {redis_url}")

    except Exception as e:
        if fallback_to_memory:
            print(f"⚠ Redis error ({e}), using in-memory queue (dev mode)")
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
    print(f"⚠ Using synchronous queue creation, falling back to memory queue")
    return MemoryQueue(queue_name=queue_name)
