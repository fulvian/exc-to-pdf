#!/usr/bin/env -S .devstream/bin/python
# -*- coding: utf-8 -*-

"""
SessionEnd Hook v2 - Event Sourcing Implementation

Replaces complex post-hoc inference with event-driven aggregation.
Zero database queries during session, single write at end.

Context7 Patterns Applied:
- eventsourcing.nodejs: Array.reduce() aggregation pattern
- pyeventsourcing: Append-only event log processing
- Epoch timestamps: Avoid datetime complexity

Workflow:
1. Get event log from registry
2. Aggregate events (zero queries)
3. Generate markdown
4. Store in memory via MCP (1x write)
5. Write marker file (atomic)
6. Close event log
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cchooks
import structlog

from utils.atomic_file_writer import write_atomic
from utils.devstream_base import DevStreamHookBase
from sessions.session_event_log import SessionEvent, get_session_log, close_session_log

logger = structlog.get_logger(__name__)


@dataclass
class SessionSummaryData:
    """
    Aggregated session statistics from events.

    Contains all data needed for markdown summary generation.
    All timestamps stored as epoch seconds.
    """
    session_id: str
    started_at: float  # Epoch
    ended_at: float
    duration_seconds: float

    # Counters
    files_modified: int
    tasks_completed: int
    tasks_started: int
    decisions_made: int
    learnings_captured: int
    errors_occurred: int

    # Samples (top N)
    file_list: List[str]
    completed_task_titles: List[str]
    started_task_titles: List[str]
    decision_list: List[str]
    learning_list: List[str]
    error_list: List[str]

    # Additional metrics
    total_events: int
    unique_event_types: int


class EventAggregator:
    """
    Event-driven aggregation using Context7 reduce pattern.

    Implements eventsourcing.nodejs Array.reduce() pattern to
    transform event stream into aggregated summary data.
    Zero database queries - pure in-memory processing.
    """

    @staticmethod
    def aggregate(events: List[SessionEvent]) -> SessionSummaryData:
        """
        Aggregate events into summary data (zero database queries).

        CRITICAL: Use reduce pattern - iterate events, accumulate state.
        This is the Context7 eventsourcing.nodejs aggregation pattern.

        Args:
            events: Chronological list of session events

        Returns:
            Aggregated session summary data

        Raises:
            ValueError: If events list is empty
        """
        if not events:
            raise ValueError("Cannot aggregate empty event list")

        # Ensure events are sorted by timestamp (CRITICAL for chronological processing)
        events = sorted(events, key=lambda e: e.timestamp)

        # Initialize counters
        files_modified = 0
        tasks_completed = 0
        tasks_started = 0
        decisions_made = 0
        learnings_captured = 0
        errors_occurred = 0

        # Initialize accumulators (use sets for deduplication)
        file_set = set()
        completed_task_titles = []
        started_task_titles = []
        decisions = []
        learnings = []
        errors = []

        # Reduce events into state (Context7 pattern)
        for event in events:
            if event.type == "file_modified":
                files_modified += 1
                path = event.data.get("path", "unknown")
                file_set.add(path)

            elif event.type == "task_completed":
                tasks_completed += 1
                title = event.data.get("title", "Untitled")
                completed_task_titles.append(title)

            elif event.type == "task_started":
                tasks_started += 1
                title = event.data.get("title", "Untitled")
                started_task_titles.append(title)

            elif event.type == "decision":
                decisions_made += 1
                content = event.data.get("content", "No content")
                category = event.data.get("category", "general")
                decisions.append(f"[{category}] {content}")

            elif event.type == "learning":
                learnings_captured += 1
                content = event.data.get("content", "No content")
                importance = event.data.get("importance", "normal")
                learnings.append(f"[{importance}] {content}")

            elif event.type == "error":
                errors_occurred += 1
                error_type = event.data.get("error_type", "unknown")
                message = event.data.get("message", "No message")
                errors.append(f"[{error_type}] {message}")

        # Extract session ID from any event if available
        session_id = "unknown"
        for event in events:
            if "session_id" in event.data:
                session_id = event.data["session_id"]
                break

        # Calculate time metrics
        started_at = events[0].timestamp
        ended_at = events[-1].timestamp
        duration_seconds = ended_at - started_at

        # Get unique event types
        event_types = set(event.type for event in events)

        return SessionSummaryData(
            session_id=session_id,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration_seconds,

            # Counters
            files_modified=files_modified,
            tasks_completed=tasks_completed,
            tasks_started=tasks_started,
            decisions_made=decisions_made,
            learnings_captured=learnings_captured,
            errors_occurred=errors_occurred,

            # Samples (limit to prevent extremely long summaries)
            file_list=list(file_set)[:10],  # Top 10 files
            completed_task_titles=completed_task_titles[:10],  # Top 10 tasks
            started_task_titles=started_task_titles[:5],  # Top 5 tasks
            decision_list=decisions[:5],  # Top 5 decisions
            learning_list=learnings[:5],  # Top 5 learnings
            error_list=errors[:3],  # Top 3 errors

            # Additional metrics
            total_events=len(events),
            unique_event_types=len(event_types)
        )


class SummaryGenerator:
    """
    Generate markdown summary from aggregated data.

    Creates human-readable session summary using epoch timestamps
    converted to local time for display only.
    """

    @staticmethod
    def generate_markdown(data: SessionSummaryData) -> str:
        """
        Generate markdown-formatted session summary.

        CRITICAL: Use datetime.fromtimestamp(epoch) for display.
        Never store datetime objects internally.

        Args:
            data: Aggregated session summary data

        Returns:
            Markdown-formatted session summary
        """
        # Convert epoch timestamps to human-readable format (display only)
        started = datetime.fromtimestamp(data.started_at).strftime("%Y-%m-%d %H:%M:%S")
        ended = datetime.fromtimestamp(data.ended_at).strftime("%Y-%m-%d %H:%M:%S")
        duration_min = int(data.duration_seconds / 60)
        duration_sec = int(data.duration_seconds % 60)

        md = f"""# DevStream Session Summary

**Session**: {data.session_id}
**Started**: {started}
**Ended**: {ended}
**Duration**: {duration_min}m {duration_sec}s

---

## 📊 Work Accomplished

### Files Modified: {data.files_modified}
"""

        # Add file list
        if data.file_list:
            md += "\n```\n"
            for file_path in data.file_list:
                md += f"• {file_path}\n"
            md += "```\n"
        else:
            md += "\n_No files modified_\n"

        # Add tasks section
        md += f"""
### Tasks Completed: {data.tasks_completed}
"""
        if data.completed_task_titles:
            md += "\n"
            for i, title in enumerate(data.completed_task_titles, 1):
                md += f"{i}. {title}\n"
        else:
            md += "\n_No tasks completed_\n"

        # Add tasks started (if any)
        if data.started_task_titles:
            md += f"""
### Tasks Started: {len(data.started_task_titles)}
"""
            for title in data.started_task_titles:
                md += f"• {title}\n"

        # Add decisions section
        if data.decision_list:
            md += f"""
## 🎯 Key Decisions

"""
            for i, decision in enumerate(data.decision_list, 1):
                md += f"{i}. {decision}\n"

        # Add learnings section
        if data.learning_list:
            md += f"""
## 💡 Lessons Learned

"""
            for i, learning in enumerate(data.learning_list, 1):
                md += f"{i}. {learning}\n"

        # Add errors section
        if data.error_list:
            md += f"""
## 🚨 Errors Encountered

"""
            for i, error in enumerate(data.error_list, 1):
                md += f"{i}. {error}\n"

        # Add metrics section
        md += f"""
## 📈 Session Metrics

- **Total Events**: {data.total_events}
- **Event Types**: {data.unique_event_types}
- **Files Modified**: {data.files_modified}
- **Tasks Completed**: {data.tasks_completed}
- **Decisions Made**: {data.decisions_made}
- **Learnings Captured**: {data.learnings_captured}
- **Errors Occurred**: {data.errors_occurred}

---

_Generated by DevStream Event Sourcing Session Summary v2 on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}_
"""

        return md


class SessionEndHookV2(DevStreamHookBase):
    """
    SessionEnd hook v2 - Event Sourcing implementation.

    Processes session end using event-driven aggregation instead of
    complex post-hoc database queries.
    """

    def __init__(self):
        """Initialize SessionEnd hook v2."""
        super().__init__("session_end_v2")

    async def process_session_end(self, session_id: str) -> bool:
        """
        Process session end workflow with Event Sourcing.

        Args:
            session_id: Session identifier to process

        Returns:
            True if processing succeeded, False otherwise
        """
        if not self.should_run():
            self.debug_log("SessionEnd v2 disabled")
            return False

        try:
            self.debug_log(f"Processing session end for {session_id}")

            # Step 1: Get event log from registry
            event_log = await get_session_log(session_id)
            events = event_log.get_all_events()

            if not events:
                self.debug_log("No events - empty session")
                return False

            self.debug_log(f"Found {len(events)} events to process")

            # Step 2: Aggregate events (zero queries)
            aggregator = EventAggregator()
            summary_data = aggregator.aggregate(events)

            self.debug_log(
                f"Aggregated {len(events)} events: "
                f"{summary_data.files_modified} files, "
                f"{summary_data.tasks_completed} tasks completed"
            )

            # Step 3: Generate markdown
            generator = SummaryGenerator()
            summary_markdown = generator.generate_markdown(summary_data)

            # Step 4: Store in memory via MCP (1x database write)
            if self.is_memory_store_enabled():
                try:
                    # Import here to avoid circular imports
                    import sys
                    sys.path.append(str(Path(__file__).parent.parent))
                    sys.path.append(str(Path(__file__).parent.parent / 'context'))
                    try:
                        from mcp_client import get_mcp_client
                    except ImportError:
                        # Fallback for testing without MCP
                        get_mcp_client = None

                    mcp_client = get_mcp_client()
                    if mcp_client:
                        result = await self.safe_mcp_call(
                            mcp_client,
                            "devstream_store_memory",
                            {
                                "content": summary_markdown,
                                "content_type": "context",
                                "keywords": ["session", "summary", session_id, "event-sourcing", "v2"]
                            }
                        )
                        if result:
                            self.debug_log("Session summary stored in memory")
                        else:
                            self.warning_feedback("Failed to store session summary in memory")
                except Exception as e:
                    self.warning_feedback(f"Memory store unavailable: {e}")
            else:
                self.debug_log("Memory store disabled")

            # Step 5: Write marker file (atomic)
            marker_file = Path.home() / ".claude" / "state" / f"devstream_session_{session_id}.txt"
            marker_written = await write_atomic(marker_file, summary_markdown)

            if marker_written:
                self.debug_log(f"Marker file written: {marker_file}")
            else:
                self.warning_feedback("Failed to write session marker file")

            # Step 6: Close event log
            closed_log = await close_session_log(session_id)
            if closed_log:
                self.debug_log("Event log closed successfully")

            # Success feedback (verbose only)
            self.success_feedback(
                f"Session ended: {summary_data.tasks_completed} tasks, "
                f"{summary_data.files_modified} files, {summary_data.total_events} events"
            )

            return True

        except Exception as e:
            self.error_feedback(f"Session end processing failed: {e}")
            self.debug_log(f"Session end error details: {e}", exc_info=True)
            return False


# Hook entry point
async def main():
    """Main entry point for SessionEnd hook v2."""
    hook = SessionEndHookV2()

    # Get session ID from environment
    session_id = os.environ.get("CLAUDE_SESSION_ID", f"session-{int(time.time())}")

    # Process session end
    success = await hook.process_session_end(session_id)

    # Exit with appropriate code
    exit_code = 0 if success else 1
    exit(exit_code)


if __name__ == "__main__":
    # Run hook when executed directly
    asyncio.run(main())