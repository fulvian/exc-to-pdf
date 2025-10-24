#!/usr/bin/env python3
"""
Rate limiter utilities for DevStream memory operations.

This module provides async rate limiting to prevent SQLite lock contention
and API rate limit violations using the aiolimiter library (GCRA algorithm).

Context7 Research:
- Library: /mjpieters/aiolimiter (Trust Score 9.6)
- Algorithm: Leaky bucket (Generic Cell Rate Algorithm)
- Performance: <5ms overhead per operation

Usage:
    async with memory_rate_limiter:
        await mcp_client.store_memory(content)

    # Non-blocking check:
    if memory_rate_limiter.has_capacity():
        async with memory_rate_limiter:
            await mcp_client.store_memory(content)
"""

# /// script
# dependencies = [
#     "aiolimiter>=1.0.0",
# ]
# ///

from typing import Dict, Any
import time
from collections import deque
from aiolimiter import AsyncLimiter


class MemoryRateLimiter:
    """
    Rate limiter wrapper for DevStream memory operations.

    Provides both blocking and non-blocking rate limiting patterns
    to prevent SQLite lock contention from excessive memory operations.

    Attributes:
        limiter: AsyncLimiter instance from aiolimiter
        max_rate: Maximum operations per time period
        time_period: Time period in seconds
        total_operations: Total operations attempted
        throttled_operations: Operations delayed by rate limiter
    """

    def __init__(self, max_rate: int, time_period: float = 1.0):
        """
        Initialize rate limiter.

        Args:
            max_rate: Maximum number of operations per time period
            time_period: Time period in seconds (default: 1.0)

        Note:
            Uses aiolimiter's AsyncLimiter with GCRA (Generic Cell Rate Algorithm).
            This provides precise rate control with minimal overhead (<5ms).

            PERF-005 Memory Optimization:
            - Sliding window statistics to prevent unbounded growth
            - Object pooling for stats dictionary
            - Variable reuse to minimize temporary allocations
        """
        self.limiter = AsyncLimiter(max_rate, time_period)
        self.max_rate = max_rate
        self.time_period = time_period

        # PERF-005: Bounded sliding window statistics (Context7 pattern)
        self._operation_times = deque(maxlen=max_rate * 2)  # Keep last 2*rate operations
        self._throttle_times = deque(maxlen=max_rate)      # Keep last rate throttles
        self._total_operations = 0
        self._throttled_operations = 0

        # PERF-005: Object pooling for stats dictionary (Context7 pattern)
        self._stats_cache = None
        self._stats_cache_time = 0.0
        self._cache_ttl = 0.1  # Cache stats for 100ms to reduce allocations

    async def __aenter__(self):
        """
        Async context manager entry - acquire rate limiter capacity.

        Blocks until capacity is available. Tracks statistics.

        Returns:
            Self for context manager pattern
        """
        # PERF-005: Hidden mutability pattern - reuse time variable
        current_time = time.time()
        start_time = current_time
        await self.limiter.acquire()

        # PERF-005: Variable reuse - avoid temporary float creation
        current_time = time.time()
        acquire_duration = current_time - start_time

        # PERF-005: Sliding window statistics (Context7 pattern)
        self._operation_times.append(current_time)
        self._total_operations += 1
        self._stats_cache_time = 0.0  # Invalidate cache

        # Track throttled operations (if we waited >10ms, we were throttled)
        if acquire_duration > 0.01:
            self._throttle_times.append(current_time)
            self._throttled_operations += 1

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - no cleanup needed."""
        return False

    def has_capacity(self) -> bool:
        """
        Check if rate limiter has capacity without blocking.

        Returns:
            True if capacity available, False if at limit

        Note:
            Uses aiolimiter's internal state to check capacity.
            Does NOT consume capacity - use with async context manager.
        """
        # aiolimiter's has_capacity checks if we can acquire without blocking
        return self.limiter.has_capacity()

    def get_current_rate(self) -> float:
        """
        Calculate current operations per second rate.

        Returns:
            Current rate in operations per second (0.0 if no recent activity)

        Note:
            PERF-005: Uses sliding window calculation for accurate rate.
            Calculates rate based on recent operations in time_period window.
        """
        if not self._operation_times:
            return 0.0

        # PERF-005: Hidden mutability pattern - reuse time variable
        current_time = time.time()
        cutoff_time = current_time - self.time_period

        # Count operations within time window
        recent_ops = 0
        for op_time in self._operation_times:
            if op_time > cutoff_time:
                recent_ops += 1

        # PERF-005: Avoid division by zero and limit to max_rate
        if recent_ops == 0:
            return 0.0

        return min(recent_ops / self.time_period, self.max_rate)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limiter statistics.

        Returns:
            Dictionary with utilization metrics:
                - max_rate: Maximum operations per time period
                - time_period: Time period in seconds
                - total_operations: Total operations attempted
                - throttled_operations: Operations delayed by rate limiter
                - throttle_rate: Percentage of throttled operations
                - current_rate: Current operations per second

        Note:
            PERF-005: Object pooling pattern - cache stats to reduce allocations.
        """
        current_time = time.time()

        # PERF-005: Object pooling - reuse cached stats if still valid
        if (self._stats_cache is not None and
            current_time - self._stats_cache_time < self._cache_ttl):
            return self._stats_cache

        # PERF-005: Hidden mutability pattern - reuse variables
        total_ops = self._total_operations
        throttled_ops = self._throttled_operations

        # Calculate throttle rate safely
        if total_ops > 0:
            throttle_rate = (throttled_ops / total_ops * 100)
            throttle_rate_str = f"{throttle_rate:.1f}%"
        else:
            throttle_rate = 0.0
            throttle_rate_str = "0.0%"

        # PERF-005: Reuse dictionary object (Context7 pattern)
        if self._stats_cache is None:
            self._stats_cache = {}

        # Update dictionary in-place to avoid creating new object
        self._stats_cache.update({
            "max_rate": self.max_rate,
            "time_period": self.time_period,
            "total_operations": total_ops,
            "throttled_operations": throttled_ops,
            "throttle_rate": throttle_rate_str,
            "current_rate": self.get_current_rate(),
        })

        self._stats_cache_time = current_time
        return self._stats_cache


# Global rate limiter instances

# Memory operations rate limiter
# Limit: 10 operations/second
# Rationale: SQLite handles ~50-100 concurrent writes/sec in WAL mode,
# but memory operations involve complex queries (semantic search, RRF).
# 10 ops/sec provides 5-10x safety margin to prevent lock contention.
memory_rate_limiter = MemoryRateLimiter(max_rate=10, time_period=1.0)

# Ollama embedding rate limiter
# Limit: 5 operations/second
# Rationale: Ollama API typically rate-limits at 10-20 req/sec.
# 5 ops/sec provides 2-4x safety margin for embedding generation.
# Embedding generation is CPU-intensive (~100-200ms per request).
ollama_rate_limiter = MemoryRateLimiter(max_rate=5, time_period=1.0)


# Convenience functions for direct usage

async def acquire_memory_capacity() -> MemoryRateLimiter:
    """
    Acquire memory operation capacity (blocking).

    Usage:
        async with await acquire_memory_capacity():
            await mcp_client.store_memory(content)

    Returns:
        MemoryRateLimiter context manager
    """
    return memory_rate_limiter


async def acquire_ollama_capacity() -> MemoryRateLimiter:
    """
    Acquire Ollama embedding capacity (blocking).

    Usage:
        async with await acquire_ollama_capacity():
            embedding = await ollama_client.generate_embedding(text)

    Returns:
        MemoryRateLimiter context manager
    """
    return ollama_rate_limiter


def has_memory_capacity() -> bool:
    """
    Check if memory operations have capacity (non-blocking).

    Returns:
        True if capacity available, False if at limit

    Usage:
        if has_memory_capacity():
            async with memory_rate_limiter:
                await mcp_client.store_memory(content)
        else:
            logger.warning("Memory rate limit exceeded, skipping storage")
    """
    return memory_rate_limiter.has_capacity()


def has_ollama_capacity() -> bool:
    """
    Check if Ollama operations have capacity (non-blocking).

    Returns:
        True if capacity available, False if at limit
    """
    return ollama_rate_limiter.has_capacity()


def get_memory_stats() -> Dict[str, Any]:
    """
    Get memory rate limiter statistics.

    Returns:
        Dictionary with utilization metrics
    """
    return memory_rate_limiter.get_stats()


def get_ollama_stats() -> Dict[str, Any]:
    """
    Get Ollama rate limiter statistics.

    Returns:
        Dictionary with utilization metrics
    """
    return ollama_rate_limiter.get_stats()


def get_rate_limiter_stats() -> Dict[str, Dict[str, Any]]:
    """
    Get statistics for all rate limiters.

    Returns:
        Dictionary with stats for both memory and ollama limiters:
            {
                "memory": {...},
                "ollama": {...}
            }

    Usage:
        stats = get_rate_limiter_stats()
        print(f"Memory rate: {stats['memory']['current_rate']} ops/sec")
        print(f"Ollama rate: {stats['ollama']['current_rate']} ops/sec")
    """
    return {
        "memory": get_memory_stats(),
        "ollama": get_ollama_stats()
    }
