#!/usr/bin/env -S .devstream/bin/python
# -*- coding: utf-8 -*-

"""
Session Event Log - Event Sourcing Implementation v2

Implements append-only event log for session data using Context7 patterns.
Provides thread-safe in-memory event storage with zero database queries
during session operations.

Context7 Patterns Applied:
- pyeventsourcing: Append-only event log pattern
- Event structure: epoch timestamps, type discriminator, data payload
- Thread-safe operations with asyncio.Lock

Event Types:
- file_modified: {"path": str, "tool": str, "size_bytes": int}
- task_completed: {"task_id": str, "title": str}
- task_started: {"task_id": str, "title": str}
- decision: {"content": str, "category": str}
- learning: {"content": str, "importance": str}
- error: {"error_type": str, "message": str}
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SessionEvent:
    """
    Single session event with epoch timestamp.

    CRITICAL: Always use epoch timestamps (float seconds) to avoid
    timezone complexity and datetime parsing bugs.
    """
    timestamp: float  # Epoch seconds from time.time()
    type: str         # Event discriminator
    data: Dict[str, Any]  # Event payload

    def __post_init__(self):
        """Validate event structure."""
        if not isinstance(self.timestamp, float):
            raise TypeError("timestamp must be float (epoch seconds)")
        if not isinstance(self.type, str) or not self.type.strip():
            raise ValueError("type must be non-empty string")
        if not isinstance(self.data, dict):
            raise TypeError("data must be dict")


class SessionEventLog:
    """
    Thread-safe append-only event log for a single session.

    Provides in-memory event storage with async lock protection.
    Follows Context7 append-only pattern from pyeventsourcing.
    """

    def __init__(self, session_id: str):
        """
        Initialize event log for session.

        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id
        self.events: List[SessionEvent] = []
        self._lock = asyncio.Lock()

        logger.debug("session_event_log_created", session_id=session_id)

    async def record_event(self, event_type: str, data: Dict[str, Any]) -> SessionEvent:
        """
        Record event to log (append-only pattern).

        Creates event with current epoch timestamp and appends to log.
        Thread-safe with async lock protection.

        Args:
            event_type: Event type discriminator
            data: Event payload dictionary

        Returns:
            Created SessionEvent instance

        Raises:
            ValueError: If event_type is empty
            TypeError: If data is not dict
        """
        if not isinstance(event_type, str) or not event_type.strip():
            raise ValueError("event_type must be non-empty string")
        if not isinstance(data, dict):
            raise TypeError("data must be dict")

        async with self._lock:
            # CRITICAL: Use epoch timestamp only (no datetime objects)
            event = SessionEvent(
                timestamp=time.time(),
                type=event_type,
                data=data.copy()  # Defensive copy
            )

            self.events.append(event)

            logger.debug(
                "event_recorded",
                session_id=self.session_id,
                event_type=event_type,
                event_count=len(self.events),
                timestamp=event.timestamp
            )

            return event

    def get_all_events(self) -> List[SessionEvent]:
        """
        Return all events in chronological order.

        Returns:
            Copy of events list (preserves order)
        """
        return self.events.copy()

    def get_events_by_type(self, event_type: str) -> List[SessionEvent]:
        """
        Filter events by type.

        Args:
            event_type: Event type to filter

        Returns:
            List of events matching type
        """
        return [event for event in self.events if event.type == event_type]

    def get_event_count(self) -> int:
        """Get total number of events in log."""
        return len(self.events)

    def get_time_range(self) -> Optional[tuple[float, float]]:
        """
        Get time range of events.

        Returns:
            Tuple of (start_time, end_time) or None if no events
        """
        if not self.events:
            return None
        return (self.events[0].timestamp, self.events[-1].timestamp)


# Global session registry (singleton per session)
_session_logs: Dict[str, SessionEventLog] = {}
_registry_lock = asyncio.Lock()


async def get_session_log(session_id: str) -> SessionEventLog:
    """
    Get or create session log (thread-safe singleton).

    Implements registry pattern to ensure one log per session.
    Thread-safe with global lock.

    Args:
        session_id: Session identifier

    Returns:
        SessionEventLog instance for session
    """
    async with _registry_lock:
        if session_id not in _session_logs:
            _session_logs[session_id] = SessionEventLog(session_id)
            logger.debug("session_log_created", session_id=session_id)
        else:
            logger.debug("session_log_reused", session_id=session_id)

        return _session_logs[session_id]


async def close_session_log(session_id: str) -> Optional[SessionEventLog]:
    """
    Close and remove log from registry.

    Removes log from global registry to prevent memory leaks.
    Returns the closed log for final processing if needed.

    Args:
        session_id: Session identifier to close

    Returns:
        Removed SessionEventLog or None if not found
    """
    async with _registry_lock:
        log = _session_logs.pop(session_id, None)
        if log:
            logger.debug(
                "session_log_closed",
                session_id=session_id,
                event_count=log.get_event_count()
            )
        else:
            logger.debug("session_log_not_found", session_id=session_id)

        return log


async def get_all_active_sessions() -> List[str]:
    """
    Get list of all active session IDs.

    Returns:
        List of session IDs with active logs
    """
    async with _registry_lock:
        return list(_session_logs.keys())


async def cleanup_all_logs() -> int:
    """
    Clean up all session logs (emergency cleanup).

    Removes all logs from registry and returns count.
    Used for emergency cleanup or testing.

    Returns:
        Number of logs cleaned up
    """
    async with _registry_lock:
        count = len(_session_logs)
        _session_logs.clear()

        logger.warning("all_session_logs_cleaned", count=count)
        return count


# Context7 validation patterns
def validate_event_structure(event: SessionEvent) -> bool:
    """
    Validate event structure (Context7 pattern).

    Args:
        event: Event to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        # Check timestamp is positive float
        if not isinstance(event.timestamp, float) or event.timestamp <= 0:
            return False

        # Check type is non-empty string
        if not isinstance(event.type, str) or not event.type.strip():
            return False

        # Check data is dict
        if not isinstance(event.data, dict):
            return False

        return True
    except Exception:
        return False


def validate_session_log(log: SessionEventLog) -> bool:
    """
    Validate session log integrity.

    Args:
        log: Session log to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        # Check session ID
        if not isinstance(log.session_id, str) or not log.session_id.strip():
            return False

        # Check events list
        if not isinstance(log.events, list):
            return False

        # Validate all events
        for event in log.events:
            if not validate_event_structure(event):
                return False

        # Check chronological order
        for i in range(1, len(log.events)):
            if log.events[i].timestamp < log.events[i-1].timestamp:
                return False  # Events out of order

        return True
    except Exception:
        return False


# Debug utilities (for testing and debugging)
def get_registry_stats() -> Dict[str, Any]:
    """
    Get registry statistics (for debugging).

    Returns:
        Dictionary with registry stats
    """
    return {
        "active_sessions": len(_session_logs),
        "session_ids": list(_session_logs.keys()),
        "total_events": sum(len(log.events) for log in _session_logs.values())
    }


if __name__ == "__main__":
    # Simple test when run directly
    import asyncio

    async def test_event_log():
        """Test basic event log functionality."""
        session_id = "test-session"

        # Get log
        log = await get_session_log(session_id)

        # Record events
        await log.record_event("file_modified", {"path": "test.py", "tool": "Write"})
        await log.record_event("task_completed", {"title": "Test task"})

        # Check events
        events = log.get_all_events()
        print(f"Recorded {len(events)} events")

        # Close log
        await close_session_log(session_id)
        print("Event log test completed")

    asyncio.run(test_event_log())