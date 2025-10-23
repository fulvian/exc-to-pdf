"""
Memory monitoring and resource management utilities for performance optimization.

This module provides memory monitoring, garbage collection management,
and resource limiting capabilities for handling large Excel files efficiently.
"""

import gc
import logging
import os
import psutil
import threading
import time
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any

import structlog

logger = structlog.get_logger()


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    current_mb: float
    peak_mb: float
    available_mb: float
    percent_used: float
    process_mb: float


@dataclass
class ResourceLimits:
    """Resource usage limits."""
    max_memory_mb: int
    max_cpu_percent: float = 80.0
    gc_threshold_mb: int = 100  # Force GC when memory increases by this amount


class MemoryMonitor:
    """
    Memory usage monitoring and management system.

    Provides real-time memory monitoring, automatic garbage collection,
    and memory limit enforcement for large file processing.
    """

    def __init__(self, limits: ResourceLimits):
        """
        Initialize memory monitor with resource limits.

        Args:
            limits: Resource usage limits
        """
        self.limits = limits
        self._process = psutil.Process(os.getpid())
        self._peak_memory = 0.0
        self._last_gc_memory = 0.0
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: Dict[str, Callable[[], None]] = {}

        logger.info("Memory monitor initialized", limits=limits)

    def get_memory_stats(self) -> MemoryStats:
        """
        Get current memory usage statistics.

        Returns:
            MemoryStats object with current usage information
        """
        try:
            # System memory
            memory = psutil.virtual_memory()
            system_current_mb = memory.used / 1024 / 1024
            system_available_mb = memory.available / 1024 / 1024
            system_percent = memory.percent

            # Process memory
            process_memory = self._process.memory_info()
            process_mb = process_memory.rss / 1024 / 1024

            # Update peak
            if process_mb > self._peak_memory:
                self._peak_memory = process_mb

            return MemoryStats(
                current_mb=process_mb,
                peak_mb=self._peak_memory,
                available_mb=system_available_mb,
                percent_used=system_percent,
                process_mb=process_mb
            )

        except Exception as e:
            logger.warning("Failed to get memory stats", error=str(e))
            return MemoryStats(
                current_mb=0.0,
                peak_mb=self._peak_memory,
                available_mb=0.0,
                percent_used=0.0,
                process_mb=0.0
            )

    def check_memory_limits(self) -> bool:
        """
        Check if current memory usage exceeds limits.

        Returns:
            True if within limits, False if exceeded
        """
        stats = self.get_memory_stats()

        # Check process memory limit
        if stats.current_mb > self.limits.max_memory_mb:
            logger.warning(
                "Memory limit exceeded",
                current_mb=stats.current_mb,
                limit_mb=self.limits.max_memory_mb
            )
            return False

        # Check system memory availability
        if stats.available_mb < 100:  # Leave 100MB system buffer
            logger.warning(
                "Low system memory",
                available_mb=stats.available_mb
            )
            return False

        return True

    def force_garbage_collection(self) -> int:
        """
        Force garbage collection to free memory.

        Returns:
            Number of objects collected
        """
        try:
            # Run full garbage collection
            collected = gc.collect()

            stats = self.get_memory_stats()
            logger.debug(
                "Garbage collection completed",
                objects_collected=collected,
                memory_mb=stats.current_mb
            )

            self._last_gc_memory = stats.current_mb
            return collected

        except Exception as e:
            logger.warning("Garbage collection failed", error=str(e))
            return 0

    def start_monitoring(self, interval: float = 1.0) -> None:
        """
        Start background memory monitoring.

        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring:
            logger.warning("Memory monitoring already started")
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()

        logger.info("Memory monitoring started", interval=interval)

    def stop_monitoring(self) -> None:
        """Stop background memory monitoring."""
        if not self._monitoring:
            return

        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        logger.info("Memory monitoring stopped")

    def _monitor_loop(self, interval: float) -> None:
        """
        Background monitoring loop.

        Args:
            interval: Monitoring interval in seconds
        """
        while self._monitoring:
            try:
                stats = self.get_memory_stats()

                # Check memory limits
                if not self.check_memory_limits():
                    logger.error("Memory limits exceeded, stopping monitoring")
                    self._trigger_memory_limit_callbacks()
                    break

                # Auto garbage collection if needed
                if (stats.current_mb - self._last_gc_memory) > self.limits.gc_threshold_mb:
                    self.force_garbage_collection()

                # Log memory usage periodically
                logger.debug(
                    "Memory usage",
                    current_mb=stats.current_mb,
                    peak_mb=stats.peak_memory,
                    percent_used=stats.percent_used
                )

                time.sleep(interval)

            except Exception as e:
                logger.error("Memory monitoring error", error=str(e))
                time.sleep(interval)

    def _trigger_memory_limit_callbacks(self) -> None:
        """Trigger memory limit exceeded callbacks."""
        for name, callback in self._callbacks.items():
            try:
                callback()
                logger.debug("Memory limit callback triggered", callback=name)
            except Exception as e:
                logger.warning(
                    "Memory limit callback failed",
                    callback=name,
                    error=str(e)
                )

    def add_memory_limit_callback(self, name: str, callback: Callable[[], None]) -> None:
        """
        Add callback for memory limit exceeded events.

        Args:
            name: Callback name
            callback: Function to call when memory limit exceeded
        """
        self._callbacks[name] = callback
        logger.debug("Memory limit callback added", callback=name)

    def remove_memory_limit_callback(self, name: str) -> None:
        """
        Remove memory limit callback.

        Args:
            name: Callback name to remove
        """
        if name in self._callbacks:
            del self._callbacks[name]
            logger.debug("Memory limit callback removed", callback=name)

    def __enter__(self) -> "MemoryMonitor":
        """Context manager entry."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[object]) -> None:
        """Context manager exit."""
        self.stop_monitoring()


def get_system_memory_info() -> Dict[str, float]:
    """
    Get system memory information.

    Returns:
        Dictionary with memory information in MB
    """
    try:
        memory = psutil.virtual_memory()
        return {
            "total_mb": memory.total / 1024 / 1024,
            "available_mb": memory.available / 1024 / 1024,
            "used_mb": memory.used / 1024 / 1024,
            "percent_used": memory.percent
        }
    except Exception as e:
        logger.warning("Failed to get system memory info", error=str(e))
        return {}


def optimize_gc_settings() -> None:
    """Optimize garbage collection settings for large file processing."""
    try:
        # Set GC thresholds for better memory management
        gc.set_threshold(700, 10, 10)  # Default is (700, 10, 10)

        # Enable GC debug mode if in debug logging
        try:
            if logger.isEnabledFor(logging.DEBUG):
                gc.set_debug(gc.DEBUG_STATS)
        except AttributeError:
            # Fallback if isEnabledFor is not available
            import structlog
            if structlog.get_logger().isEnabledFor(logging.DEBUG):
                gc.set_debug(gc.DEBUG_STATS)

        logger.info("GC settings optimized")

    except Exception as e:
        logger.warning("Failed to optimize GC settings", error=str(e))