"""
Progress tracking and user feedback system for long-running operations.

This module provides progress bars, ETA calculations, and status updates
for Excel processing operations, improving user experience for large files.
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Dict, Any
from pathlib import Path

import structlog

logger = structlog.get_logger()


class ProgressStatus(Enum):
    """Progress operation status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressUpdate:
    """Progress update information."""
    current: int
    total: int
    percentage: float
    rate: float  # items per second
    eta_seconds: Optional[float]
    elapsed_seconds: float
    status: ProgressStatus
    message: Optional[str] = None


@dataclass
class OperationStats:
    """Operation performance statistics."""
    items_processed: int = 0
    items_total: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    last_update_time: Optional[float] = None
    last_update_count: int = 0


class ProgressTracker:
    """
    Progress tracking system with ETA calculation and rate monitoring.

    Provides real-time progress updates, performance metrics,
    and customizable progress reporting for long-running operations.
    """

    def __init__(
        self,
        operation_name: str,
        total_items: int,
        update_interval: float = 1.0,
        show_progress: bool = True,
        custom_formatter: Optional[Callable[[ProgressUpdate], str]] = None
    ):
        """
        Initialize progress tracker.

        Args:
            operation_name: Name of the operation being tracked
            total_items: Total number of items to process
            update_interval: Minimum interval between progress updates (seconds)
            show_progress: Whether to show progress output
            custom_formatter: Custom progress message formatter
        """
        self.operation_name = operation_name
        self.total_items = total_items
        self.update_interval = update_interval
        self.show_progress = show_progress
        self.custom_formatter = custom_formatter

        self.stats = OperationStats(items_total=total_items)
        self.status = ProgressStatus.PENDING
        self._callbacks: Dict[str, Callable[[ProgressUpdate], None]] = {}
        self._last_log_time = 0.0

        logger.info(
            "Progress tracker initialized",
            operation=operation_name,
            total_items=total_items
        )

    def start(self) -> None:
        """Start progress tracking."""
        self.status = ProgressStatus.RUNNING
        self.stats.start_time = time.time()
        self.stats.last_update_time = self.stats.start_time
        self._last_log_time = self.stats.start_time

        logger.info(
            "Operation started",
            operation=self.operation_name,
            total_items=self.total_items
        )

        self._update_progress(0, "Starting operation")

    def update(
        self,
        current: int,
        message: Optional[str] = None,
        force_update: bool = False
    ) -> None:
        """
        Update progress with current item count.

        Args:
            current: Current number of items processed
            message: Optional status message
            force_update: Force update regardless of interval
        """
        if self.status != ProgressStatus.RUNNING:
            logger.warning(
                "Update called on non-running tracker",
                status=self.status.value,
                operation=self.operation_name
            )
            return

        current_time = time.time()

        # Check if we should update based on interval
        time_since_last = current_time - (self._last_log_time or 0)
        if not force_update and time_since_last < self.update_interval:
            return

        # Update statistics
        self.stats.items_processed = current
        self.stats.last_update_time = current_time
        self.stats.last_update_count = current

        self._update_progress(current, message)

    def increment(
        self,
        increment: int = 1,
        message: Optional[str] = None
    ) -> None:
        """
        Increment progress by specified amount.

        Args:
            increment: Number of items to increment by
            message: Optional status message
        """
        current = self.stats.items_processed + increment
        self.update(current, message)

    def complete(self, message: Optional[str] = None) -> None:
        """
        Mark operation as completed.

        Args:
            message: Optional completion message
        """
        self.status = ProgressStatus.COMPLETED
        self.stats.end_time = time.time()
        self.stats.items_processed = self.total_items

        elapsed = self.stats.end_time - (self.stats.start_time or 0)

        logger.info(
            "Operation completed",
            operation=self.operation_name,
            items_processed=self.stats.items_processed,
            elapsed_seconds=elapsed,
            rate=self.stats.items_processed / elapsed if elapsed > 0 else 0
        )

        self._update_progress(self.total_items, message or "Operation completed")

    def fail(self, error_message: str) -> None:
        """
        Mark operation as failed.

        Args:
            error_message: Error message describing the failure
        """
        self.status = ProgressStatus.FAILED
        self.stats.end_time = time.time()

        logger.error(
            "Operation failed",
            operation=self.operation_name,
            items_processed=self.stats.items_processed,
            error=error_message
        )

        self._update_progress(self.stats.items_processed, f"Failed: {error_message}")

    def cancel(self, message: Optional[str] = None) -> None:
        """
        Cancel the operation.

        Args:
            message: Optional cancellation message
        """
        self.status = ProgressStatus.CANCELLED
        self.stats.end_time = time.time()

        logger.info(
            "Operation cancelled",
            operation=self.operation_name,
            items_processed=self.stats.items_processed
        )

        self._update_progress(self.stats.items_processed, message or "Operation cancelled")

    def pause(self, message: Optional[str] = None) -> None:
        """
        Pause the operation.

        Args:
            message: Optional pause message
        """
        self.status = ProgressStatus.PAUSED
        self._update_progress(self.stats.items_processed, message or "Operation paused")

    def resume(self, message: Optional[str] = None) -> None:
        """
        Resume the operation.

        Args:
            message: Optional resume message
        """
        self.status = ProgressStatus.RUNNING
        self._update_progress(self.stats.items_processed, message or "Operation resumed")

    def _update_progress(self, current: int, message: Optional[str] = None) -> None:
        """
        Internal method to update progress and notify callbacks.

        Args:
            current: Current progress count
            message: Optional status message
        """
        current_time = time.time()

        # Calculate progress metrics
        percentage = (current / self.total_items * 100) if self.total_items > 0 else 0

        # Calculate rate and ETA
        elapsed_seconds = current_time - (self.stats.start_time or current_time)
        rate = current / elapsed_seconds if elapsed_seconds > 0 else 0

        eta_seconds = None
        if rate > 0 and current < self.total_items:
            remaining_items = self.total_items - current
            eta_seconds = remaining_items / rate

        # Create progress update
        update = ProgressUpdate(
            current=current,
            total=self.total_items,
            percentage=percentage,
            rate=rate,
            eta_seconds=eta_seconds,
            elapsed_seconds=elapsed_seconds,
            status=self.status,
            message=message
        )

        # Log progress if enabled
        if self.show_progress and current_time - self._last_log_time >= self.update_interval:
            self._log_progress(update)
            self._last_log_time = current_time

        # Notify callbacks
        self._notify_callbacks(update)

    def _log_progress(self, update: ProgressUpdate) -> None:
        """
        Log progress update.

        Args:
            update: Progress update to log
        """
        if self.custom_formatter:
            message = self.custom_formatter(update)
            logger.info(message)
        else:
            eta_str = f"ETA: {self._format_time(update.eta_seconds)}" if update.eta_seconds else "ETA: --"
            rate_str = f"{update.rate:.1f} items/s" if update.rate > 0 else "-- items/s"

            message = (
                f"{self.operation_name}: "
                f"{update.current:,}/{update.total:,} "
                f"({update.percentage:.1f}%) "
                f"Rate: {rate_str} "
                f"{eta_str}"
            )

            if update.message:
                message += f" - {update.message}"

            logger.info(message)

    def _notify_callbacks(self, update: ProgressUpdate) -> None:
        """
        Notify all registered callbacks.

        Args:
            update: Progress update to send
        """
        for name, callback in self._callbacks.items():
            try:
                callback(update)
            except Exception as e:
                logger.warning(
                    "Progress callback failed",
                    callback=name,
                    error=str(e)
                )

    def add_callback(self, name: str, callback: Callable[[ProgressUpdate], None]) -> None:
        """
        Add progress update callback.

        Args:
            name: Callback name
            callback: Function to call with progress updates
        """
        self._callbacks[name] = callback
        logger.debug("Progress callback added", callback=name, operation=self.operation_name)

    def remove_callback(self, name: str) -> None:
        """
        Remove progress callback.

        Args:
            name: Callback name to remove
        """
        if name in self._callbacks:
            del self._callbacks[name]
            logger.debug("Progress callback removed", callback=name, operation=self.operation_name)

    @staticmethod
    def _format_time(seconds: Optional[float]) -> str:
        """
        Format time duration in human readable format.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string
        """
        if seconds is None or seconds < 0:
            return "--"

        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.0f}m"
        else:
            hours = seconds / 3600
            remaining_minutes = (seconds % 3600) / 60
            return f"{hours:.0f}h {remaining_minutes:.0f}m"

    def get_summary(self) -> Dict[str, Any]:
        """
        Get operation summary.

        Returns:
            Dictionary with operation summary
        """
        elapsed = 0.0
        if self.stats.start_time:
            if self.stats.end_time:
                elapsed = self.stats.end_time - self.stats.start_time
            else:
                elapsed = time.time() - self.stats.start_time

        return {
            "operation": self.operation_name,
            "status": self.status.value,
            "items_processed": self.stats.items_processed,
            "items_total": self.stats.items_total,
            "percentage": (self.stats.items_processed / self.stats.items_total * 100) if self.stats.items_total > 0 else 0,
            "elapsed_seconds": elapsed,
            "start_time": self.stats.start_time,
            "end_time": self.stats.end_time
        }


class MultiProgressTracker:
    """
    Multi-operation progress tracker.

    Manages multiple progress trackers and provides consolidated progress reporting.
    """

    def __init__(self, operation_name: str, show_progress: bool = True):
        """
        Initialize multi-progress tracker.

        Args:
            operation_name: Name of the overall operation
            show_progress: Whether to show progress output
        """
        self.operation_name = operation_name
        self.show_progress = show_progress
        self.trackers: Dict[str, ProgressTracker] = {}
        self._total_items = 0
        self._total_processed = 0

    def add_tracker(self, name: str, total_items: int) -> ProgressTracker:
        """
        Add a new progress tracker.

        Args:
            name: Tracker name
            total_items: Total items for this tracker

        Returns:
            Created ProgressTracker
        """
        tracker = ProgressTracker(
            f"{self.operation_name}/{name}",
            total_items,
            show_progress=False,  # We'll handle display
            custom_formatter=self._format_multi_progress
        )

        self.trackers[name] = tracker
        self._total_items += total_items

        return tracker

    def _format_multi_progress(self, update: ProgressUpdate) -> str:
        """
        Format multi-operation progress message.

        Args:
            update: Progress update

        Returns:
            Formatted progress message
        """
        # Calculate overall progress
        total_processed = sum(t.stats.items_processed for t in self.trackers.values())
        total_items = sum(t.stats.items_total for t in self.trackers.values())

        overall_percentage = (total_processed / total_items * 100) if total_items > 0 else 0

        # Count completed operations
        completed = sum(1 for t in self.trackers.values() if t.status == ProgressStatus.COMPLETED)
        total_ops = len(self.trackers)

        return (
            f"{self.operation_name}: "
            f"{total_processed:,}/{total_items:,} ({overall_percentage:.1f}%) "
            f"Operations: {completed}/{total_ops}"
        )

    def get_summary(self) -> Dict[str, Any]:
        """
        Get overall operation summary.

        Returns:
            Dictionary with overall summary
        """
        total_processed = sum(t.stats.items_processed for t in self.trackers.values())
        total_items = sum(t.stats.items_total for t in self.trackers.values())

        completed_ops = [name for name, t in self.trackers.items() if t.status == ProgressStatus.COMPLETED]
        failed_ops = [name for name, t in self.trackers.items() if t.status == ProgressStatus.FAILED]

        return {
            "operation": self.operation_name,
            "total_items": total_items,
            "total_processed": total_processed,
            "percentage": (total_processed / total_items * 100) if total_items > 0 else 0,
            "operations_completed": len(completed_ops),
            "operations_failed": len(failed_ops),
            "operations_total": len(self.trackers),
            "completed_operations": completed_ops,
            "failed_operations": failed_ops
        }