"""
Session Tracker using AnyIO task groups for real-time progress updates.

Provides structured concurrency with cancellation handling for session progress
tracking using Context7 AnyIO patterns.
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import uuid

import anyio
import structlog

from .session_manager import SessionManager

# LOG-004: Use AnyIO Lock for better async synchronization (Context7 pattern)
_session_lock = anyio.Lock()

logger = structlog.get_logger(__name__)


@dataclass
class SessionMetrics:
    """Real-time session metrics data model."""
    session_id: str
    tokens_used: int
    files_modified: int
    tasks_completed: int
    last_activity: datetime
    operations_per_second: float = 0.0
    average_operation_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'session_id': self.session_id,
            'tokens_used': self.tokens_used,
            'files_modified': self.files_modified,
            'tasks_completed': self.tasks_completed,
            'last_activity': self.last_activity.isoformat(),
            'operations_per_second': self.operations_per_second,
            'average_operation_time': self.average_operation_time
        }


class TrackingException(Exception):
    """Session tracking exception."""
    pass


class SessionTracker:
    """Real-time session progress tracker using AnyIO task groups."""

    def __init__(self, session_manager: SessionManager) -> None:
        """Initialize tracker with session manager.

        Args:
            session_manager: SessionManager instance for database operations
        """
        self.session_manager = session_manager
        self._logger = structlog.get_logger(__name__).bind(component="SessionTracker")

        # Tracking state
        self._active_trackers: Dict[str, anyio.CancelScope] = {}
        self._metrics_cache: Dict[str, SessionMetrics] = {}
        self._operation_times: Dict[str, List[float]] = {}

        # Task group for managing all tracking tasks
        self._main_task_group: Optional[Any] = None
        self._shutdown_event = anyio.Event()

    async def start_tracking(self, session_id: str) -> None:
        """Start tracking session progress.

        Args:
            session_id: Session identifier to track

        Raises:
            TrackingException: If tracking already exists or session not found
        """
        # LOG-004: Fix race condition - use lock for thread-safe check-and-act
        async with _session_lock:
            if session_id in self._active_trackers:
                raise TrackingException(f"Already tracking session {session_id}")

        # Verify session exists and is active
        session = await self.session_manager.get_session(session_id)
        if not session:
            raise TrackingException(f"Session {session_id} not found")
        if session.status != 'active':
            raise TrackingException(f"Session {session_id} is not active")

        try:
            # Create cancellation scope for this session
            cancel_scope = anyio.CancelScope()

            # LOG-004: Use lock for thread-safe dictionary updates
            async with _session_lock:
                # Initialize metrics
                self._metrics_cache[session_id] = SessionMetrics(
                    session_id=session_id,
                    tokens_used=session.tokens_used or 0,
                    files_modified=session.files_modified or 0,
                    tasks_completed=session.tasks_completed or 0,
                    last_activity=datetime.now()
                )
                self._operation_times[session_id] = []

                # Store cancel scope for later cancellation
                self._active_trackers[session_id] = cancel_scope

            # Start tracking tasks as background tasks
            async def run_tracking_tasks() -> None:
                try:
                    with cancel_scope:
                        async with anyio.create_task_group() as tg:
                            tg.start_soon(self._track_session_progress, session_id)
                            tg.start_soon(self._monitor_session_health, session_id)
                            tg.start_soon(self._update_metrics_periodically, session_id)
                except Exception as e:
                    self._logger.error(
                        "Background tracking task failed",
                        extra={"session_id": session_id, "error": str(e)}
                    )

            # Start background task without awaiting it
            asyncio.create_task(run_tracking_tasks())

            self._logger.info(
                "Started tracking session",
                extra={"session_id": session_id}
            )

        except Exception as e:
            # Cleanup on failure
            self._active_trackers.pop(session_id, None)
            self._metrics_cache.pop(session_id, None)
            self._operation_times.pop(session_id, None)

            self._logger.error(
                "Failed to start tracking",
                extra={"session_id": session_id, "error": str(e)}
            )
            raise TrackingException(f"Failed to start tracking session {session_id}: {e}") from e

    async def stop_tracking(self, session_id: str) -> None:
        """Stop tracking session.

        Args:
            session_id: Session identifier to stop tracking
        """
        # LOG-004: Fix race condition - use lock for thread-safe dictionary access
        async with _session_lock:
            cancel_scope = self._active_trackers.pop(session_id, None)

            if cancel_scope:
                # Clean up caches atomically with scope removal
                self._metrics_cache.pop(session_id, None)
                self._operation_times.pop(session_id, None)

        if cancel_scope:
            # Cancel the tracking scope outside the lock to avoid blocking
            cancel_scope.cancel()

            self._logger.info(
                "Stopped tracking session",
                extra={"session_id": session_id}
            )

    async def update_progress(self, session_id: str, **metrics: Any) -> None:
        """Update session progress metrics.

        Args:
            session_id: Session identifier
            **metrics: Metrics to update (tokens_used, files_modified, tasks_completed)
        """
        # LOG-004: Fix race condition - use AnyIO ResourceGuard for exclusive access
        async with _session_lock:
            if session_id not in self._active_trackers:
                self._logger.warning(
                    "Attempted to update untracked session",
                    extra={"session_id": session_id}
                )
                return

        start_time = datetime.now()

        try:
            # Update database
            await self.session_manager.update_session(session_id, **metrics)

            # Update local cache
            cached_metrics = self._metrics_cache.get(session_id)
            if cached_metrics:
                for key, value in metrics.items():
                    if hasattr(cached_metrics, key):
                        setattr(cached_metrics, key, value)
                cached_metrics.last_activity = start_time

            # Track operation time
            operation_time = (datetime.now() - start_time).total_seconds()
            self._operation_times[session_id].append(operation_time)

            # Keep only last 100 operation times
            if len(self._operation_times[session_id]) > 100:
                self._operation_times[session_id] = self._operation_times[session_id][-100:]

            self._logger.debug(
                "Updated session progress",
                extra={"session_id": session_id, "metrics": list(metrics.keys())}
            )

        except Exception as e:
            self._logger.error(
                "Failed to update progress",
                extra={"session_id": session_id, "metrics": metrics, "error": str(e)}
            )
            raise TrackingException(f"Failed to update progress for {session_id}: {e}") from e

    async def get_metrics(self, session_id: str) -> Optional[SessionMetrics]:
        """Get current metrics for session.

        Args:
            session_id: Session identifier

        Returns:
            SessionMetrics or None if not tracked
        """
        return self._metrics_cache.get(session_id)

    async def get_all_metrics(self) -> List[SessionMetrics]:
        """Get metrics for all tracked sessions.

        Returns:
            List of SessionMetrics for all active tracking sessions
        """
        return list(self._metrics_cache.values())

    async def cleanup_inactive_sessions(self, max_inactive_minutes: int = 30) -> int:
        """Clean up tracking for inactive sessions.

        Args:
            max_inactive_minutes: Maximum inactivity before cleanup

        Returns:
            Number of sessions cleaned up
        """
        # LOG-004: Fix race condition - get snapshot under lock, then cleanup
        async with _session_lock:
            # Create snapshot of sessions to cleanup
            sessions_to_cleanup = [
                session_id for session_id, metrics in list(self._metrics_cache.items())
                if (datetime.now() - metrics.last_activity) > timedelta(minutes=max_inactive_minutes)
            ]

        cleanup_count = 0
        for session_id in sessions_to_cleanup:
            await self.stop_tracking(session_id)
            cleanup_count += 1

        if cleanup_count > 0:
            self._logger.info(
                "Cleaned up inactive sessions",
                extra={"count": cleanup_count, "max_inactive_minutes": max_inactive_minutes}
            )

        return cleanup_count

    async def shutdown(self) -> None:
        """Shutdown tracker and clean up all active tracking."""
        self._shutdown_event.set()

        # Cancel all active tracking
        for session_id in list(self._active_trackers.keys()):
            await self.stop_tracking(session_id)

        self._logger.info("Session tracker shutdown completed")

    async def _track_session_progress(self, session_id: str) -> None:
        """Track session progress in real-time.

        Args:
            session_id: Session identifier
        """
        self._logger.debug(
            "Started progress tracking task",
            extra={"session_id": session_id}
        )

        try:
            while not self._shutdown_event.is_set() and session_id in self._active_trackers:
                # Check if session is still active
                session = await self.session_manager.get_session(session_id)
                if not session or session.status != 'active':
                    self._logger.info(
                        "Session no longer active, stopping tracking",
                        extra={"session_id": session_id, "status": session.status if session else "not_found"}
                    )
                    break

                # Update metrics from database
                cached_metrics = self._metrics_cache.get(session_id)
                if cached_metrics:
                    cached_metrics.tokens_used = session.tokens_used or 0
                    cached_metrics.files_modified = session.files_modified or 0
                    cached_metrics.tasks_completed = session.tasks_completed or 0

                # Wait before next check
                await anyio.sleep(5)  # Check every 5 seconds

        except anyio.get_cancelled_exc_class():
            self._logger.debug(
                "Progress tracking cancelled",
                extra={"session_id": session_id}
            )
            raise
        except Exception as e:
            self._logger.error(
                "Error in progress tracking",
                extra={"session_id": session_id, "error": str(e)}
            )
        finally:
            self._logger.debug(
                "Progress tracking task ended",
                extra={"session_id": session_id}
            )

    async def _monitor_session_health(self, session_id: str) -> None:
        """Monitor session health and detect issues.

        Args:
            session_id: Session identifier
        """
        self._logger.debug(
            "Started health monitoring task",
            extra={"session_id": session_id}
        )

        try:
            while not self._shutdown_event.is_set() and session_id in self._active_trackers:
                cached_metrics = self._metrics_cache.get(session_id)
                if not cached_metrics:
                    break

                # Check for stale session (no activity for too long)
                inactive_time = datetime.now() - cached_metrics.last_activity
                if inactive_time > timedelta(minutes=10):
                    self._logger.warning(
                        "Session appears stale",
                        extra={
                            "session_id": session_id,
                            "inactive_minutes": int(inactive_time.total_seconds() / 60)
                        }
                    )

                # Calculate performance metrics
                operation_times = self._operation_times.get(session_id, [])
                if operation_times:
                    # Calculate operations per second (last minute)
                    recent_operations = [
                        t for t in operation_times
                        if (datetime.now() - timedelta(seconds=60)) <
                           (cached_metrics.last_activity - timedelta(seconds=t))
                    ]
                    cached_metrics.operations_per_second = len(recent_operations) / 60.0

                    # Calculate average operation time
                    cached_metrics.average_operation_time = sum(operation_times[-10:]) / len(operation_times[-10:])

                # Wait before next health check
                await anyio.sleep(30)  # Check every 30 seconds

        except anyio.get_cancelled_exc_class():
            self._logger.debug(
                "Health monitoring cancelled",
                extra={"session_id": session_id}
            )
            raise
        except Exception as e:
            self._logger.error(
                "Error in health monitoring",
                extra={"session_id": session_id, "error": str(e)}
            )

    async def _update_metrics_periodically(self, session_id: str) -> None:
        """Periodically update and cache metrics.

        Args:
            session_id: Session identifier
        """
        self._logger.debug(
            "Started periodic metrics update task",
            extra={"session_id": session_id}
        )

        try:
            while not self._shutdown_event.is_set() and session_id in self._active_trackers:
                # Log current metrics periodically
                cached_metrics = self._metrics_cache.get(session_id)
                if cached_metrics:
                    self._logger.info(
                        "Session metrics update",
                        extra={
                            "session_id": session_id,
                            "tokens_used": cached_metrics.tokens_used,
                            "files_modified": cached_metrics.files_modified,
                            "tasks_completed": cached_metrics.tasks_completed,
                            "ops_per_second": round(cached_metrics.operations_per_second, 2),
                            "avg_op_time": round(cached_metrics.average_operation_time, 3)
                        }
                    )

                # Wait before next update
                await anyio.sleep(60)  # Update every minute

        except anyio.get_cancelled_exc_class():
            self._logger.debug(
                "Periodic metrics update cancelled",
                extra={"session_id": session_id}
            )
            raise
        except Exception as e:
            self._logger.error(
                "Error in periodic metrics update",
                extra={"session_id": session_id, "error": str(e)}
            )

    @asynccontextmanager
    async def track_operation(self, session_id: str, operation_name: str) -> Any:
        """Context manager for tracking individual operations.

        Args:
            session_id: Session identifier
            operation_name: Name of the operation being tracked

        Yields:
            None
        """
        start_time = datetime.now()
        operation_id = str(uuid.uuid4())

        self._logger.debug(
            "Started tracking operation",
            extra={
                "session_id": session_id,
                "operation_id": operation_id,
                "operation_name": operation_name
            }
        )

        try:
            yield
        finally:
            duration = (datetime.now() - start_time).total_seconds()

            self._logger.debug(
                "Completed tracking operation",
                extra={
                    "session_id": session_id,
                    "operation_id": operation_id,
                    "operation_name": operation_name,
                    "duration_seconds": round(duration, 3)
                }
            )

            # Update operation times
            if session_id in self._operation_times:
                self._operation_times[session_id].append(duration)

                # Keep only recent operations
                if len(self._operation_times[session_id]) > 100:
                    self._operation_times[session_id] = self._operation_times[session_id][-100:]