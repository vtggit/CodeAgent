#!/usr/bin/env python3
"""Test Redis connection."""

import redis

try:
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    result = r.ping()
    print(f"Redis connected: {result}")
except redis.ConnectionError as e:
    print(f"Redis not available: {e}")
except Exception as e:
    print(f"Error: {e}")
