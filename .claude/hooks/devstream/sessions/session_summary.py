"""
Session Summary generator with structlog context binding.

Provides markdown summary generation with automatic session context
propagation using Context7 structlog patterns.
"""

import json
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import structlog

from .session_manager import SessionManager
from .session_tracker import SessionTracker

logger = structlog.get_logger(__name__)


class SummaryException(Exception):
    """Session summary generation exception."""
    pass


class SessionSummary:
    """Session summary generator with structlog context binding."""

    def __init__(self, session_manager: SessionManager) -> None:
        """Initialize summary generator.

        Args:
            session_manager: SessionManager instance for database operations
        """
        self.session_manager = session_manager
        self._logger = structlog.get_logger(__name__).bind(component="SessionSummary")

    async def generate_summary(self, session_id: str) -> str:
        """Generate markdown summary for session.

        Args:
            session_id: Session identifier

        Returns:
            Markdown formatted session summary

        Raises:
            SummaryException: If session not found or summary generation fails
        """
        # Bind session context to structlog for all subsequent logs
        self.bind_session_context(session_id)

        try:
            self._logger.info(
                "Starting session summary generation",
                extra={"session_id": session_id}
            )

            # Get session data
            session = await self.session_manager.get_session(session_id)
            if not session:
                raise SummaryException(f"Session {session_id} not found")

            # Generate summary sections
            summary_sections = []

            # Header
            summary_sections.append(await self._generate_header(session))

            # Session Overview
            summary_sections.append(await self._generate_overview(session))

            # Activity Timeline
            summary_sections.append(await self._generate_activity_timeline(session))

            # Performance Metrics
            summary_sections.append(await self._generate_performance_metrics(session))

            # Key Events
            summary_sections.append(await self._generate_key_events(session))

            # Usage Statistics
            summary_sections.append(await self._generate_usage_statistics(session))

            # Footer
            summary_sections.append(self._generate_footer())

            # Combine all sections
            full_summary = "\n\n".join(summary_sections)

            self._logger.info(
                "Session summary generated successfully",
                extra={
                    "session_id": session_id,
                    "summary_length": len(full_summary)
                }
            )

            return full_summary

        except Exception as e:
            self._logger.error(
                "Failed to generate session summary",
                extra={"session_id": session_id, "error": str(e)}
            )
            raise SummaryException(f"Failed to generate summary for {session_id}: {e}") from e
        finally:
            # Clear context when done
            self._clear_session_context()

    def bind_session_context(self, session_id: str) -> None:
        """Bind session context to structlog.

        Args:
            session_id: Session identifier
        """
        # Clear any existing context
        structlog.contextvars.clear_contextvars()

        # Bind session context variables
        structlog.contextvars.bind_contextvars(
            session_id=session_id,
            component="SessionSummary",
            operation="summary_generation"
        )

        self._logger.debug(
            "Session context bound to structlog",
            extra={"session_id": session_id}
        )

    def _clear_session_context(self) -> None:
        """Clear session context from structlog."""
        structlog.contextvars.clear_contextvars()

    async def _generate_header(self, session: Any) -> str:
        """Generate summary header section.

        Args:
            session: Session object

        Returns:
            Header markdown section
        """
        session_id = session.id
        status_emoji = "✅" if session.status == "completed" else "🔄"

        # Create session title with status
        title = f"{status_emoji} Session Summary: {session_id}"

        # Create metadata line
        started_at = session.started_at.strftime("%Y-%m-%d %H:%M:%S UTC") if session.started_at else "Unknown"
        duration = self._calculate_duration(session)

        metadata = f"**Started:** {started_at} | **Duration:** {duration} | **Status:** {session.status.title()}"

        return f"# {title}\n\n{metadata}"

    async def _generate_overview(self, session: Any) -> str:
        """Generate session overview section.

        Args:
            session: Session object

        Returns:
            Overview markdown section
        """
        sections = []

        # Status and Duration
        sections.append(f"**Status:** {session.status.title()}")
        sections.append(f"**Duration:** {self._calculate_duration(session)}")

        # Metrics Overview
        sections.append("### Metrics Overview")
        sections.append(f"- **Tokens Used:** {session.tokens_used or 0:,}")
        sections.append(f"- **Files Modified:** {session.files_modified or 0:,}")
        sections.append(f"- **Tasks Completed:** {session.tasks_completed or 0:,}")

        # Session Type
        if session.tasks_completed and session.tasks_completed > 0:
            sections.append(f"- **Productivity:** {session.tasks_completed} tasks completed")

        return "\n".join(sections)

    async def _generate_activity_timeline(self, session: Any) -> str:
        """Generate activity timeline section.

        Args:
            session: Session object

        Returns:
            Activity timeline markdown section
        """
        sections = []
        sections.append("### Activity Timeline")

        # Timeline events
        timeline_events = []

        # Session start
        if session.started_at:
            timeline_events.append(
                f"**{session.started_at.strftime('%H:%M:%S')}** - Session started"
            )

        # Session end (if completed)
        if session.ended_at and session.status == "completed":
            timeline_events.append(
                f"**{session.ended_at.strftime('%H:%M:%S')}** - Session completed"
            )

        # Current time (if active)
        if session.status == "active":
            current_time = datetime.now()
            timeline_events.append(
                f"**{current_time.strftime('%H:%M:%S')}** - Session still active"
            )

        # Add timeline events
        for event in timeline_events:
            sections.append(f"- {event}")

        # Calculate activity rate
        if session.started_at:
            duration_seconds = self._get_duration_seconds(session)
            if duration_seconds > 0:
                ops_per_minute = (session.files_modified or 0) / max(duration_seconds / 60, 1)
                sections.append(f"\n**Activity Rate:** {ops_per_minute:.1f} operations/minute")

        return "\n".join(sections)

    async def _generate_performance_metrics(self, session: Any) -> str:
        """Generate performance metrics section.

        Args:
            session: Session object

        Returns:
            Performance metrics markdown section
        """
        sections = []
        sections.append("### Performance Metrics")

        # Calculate metrics
        duration = self._get_duration_seconds(session)

        if duration > 0:
            # Tokens per minute
            tokens_per_min = (session.tokens_used or 0) / (duration / 60)
            sections.append(f"- **Tokens/Minute:** {tokens_per_min:.1f}")

            # Files per minute
            files_per_min = (session.files_modified or 0) / (duration / 60)
            sections.append(f"- **Files/Minute:** {files_per_min:.1f}")

            # Tasks per minute
            tasks_per_min = (session.tasks_completed or 0) / (duration / 60)
            sections.append(f"- **Tasks/Minute:** {tasks_per_min:.1f}")

        # Efficiency indicators
        sections.append("### Efficiency")

        if session.tasks_completed and session.tasks_completed > 0:
            tokens_per_task = (session.tokens_used or 0) / session.tasks_completed
            sections.append(f"- **Avg Tokens/Task:** {tokens_per_task:.1f}")

        if session.files_modified and session.tokens_used and session.tokens_used > 0:
            tokens_per_file = session.tokens_used / max(session.files_modified, 1)
            sections.append(f"- **Avg Tokens/File:** {tokens_per_file:.1f}")

        # Performance rating
        rating = self._calculate_performance_rating(session, duration)
        rating_emoji = self._get_rating_emoji(rating)
        sections.append(f"- **Performance Rating:** {rating_emoji} {rating}/10")

        return "\n".join(sections)

    async def _generate_key_events(self, session: Any) -> str:
        """Generate key events section.

        Args:
            session: Session object

        Returns:
            Key events markdown section
        """
        sections = []
        sections.append("### Key Events")

        events = []

        # Session milestone events
        if session.started_at:
            events.append({
                "time": session.started_at.strftime("%H:%M:%S"),
                "event": "Session initialized",
                "type": "info"
            })

        # Add task completion events if applicable
        if session.tasks_completed and session.tasks_completed > 0:
            events.append({
                "time": "During session",
                "event": f"Completed {session.tasks_completed} tasks",
                "type": "success"
            })

        if session.files_modified and session.files_modified > 0:
            events.append({
                "time": "During session",
                "event": f"Modified {session.files_modified} files",
                "type": "info"
            })

        if session.ended_at and session.status == "completed":
            events.append({
                "time": session.ended_at.strftime("%H:%M:%S"),
                "event": "Session completed successfully",
                "type": "success"
            })

        # Format events
        for event in events:
            icon = self._get_event_icon(event["type"])
            sections.append(f"- **{event['time']}** {icon} {event['event']}")

        # Add insights if available
        insights = self._generate_session_insights(session)
        if insights:
            sections.append("\n### Insights")
            for insight in insights:
                sections.append(f"- {insight}")

        return "\n".join(sections)

    async def _generate_usage_statistics(self, session: Any) -> str:
        """Generate usage statistics section.

        Args:
            session: Session object

        Returns:
            Usage statistics markdown section
        """
        sections = []
        sections.append("### Usage Statistics")

        # Resource usage breakdown
        sections.append("- **Token Usage:**")
        sections.append(f"  - Total: {session.tokens_used or 0:,} tokens")

        if session.tokens_used and session.tokens_used > 0:
            sections.append(f"  - Average Rate: {session.tokens_used / max(self._get_duration_seconds(session), 1):.1f} tokens/second")

        sections.append("- **File Operations:**")
        sections.append(f"  - Total Modified: {session.files_modified or 0:,} files")

        if session.files_modified and session.files_modified > 0:
            sections.append(f"  - Rate: {session.files_modified / max(self._get_duration_seconds(session), 1):.1f} files/second")

        sections.append("- **Task Management:**")
        sections.append(f"  - Completed: {session.tasks_completed or 0:,} tasks")

        if session.tasks_completed and session.tasks_completed > 0:
            sections.append(f"  - Success Rate: 100% ({session.tasks_completed} completed)")

        # Session metadata
        if session.metadata:
            sections.append("\n### Session Metadata")
            try:
                metadata_dict = json.loads(session.metadata)
                sections.append("```json")
                sections.append(json.dumps(metadata_dict, indent=2))
                sections.append("```")
            except json.JSONDecodeError:
                sections.append(f"- **Raw Metadata:** {session.metadata}")

        return "\n".join(sections)

    def _generate_footer(self) -> str:
        """Generate summary footer section.

        Returns:
            Footer markdown section
        """
        sections = []
        sections.append("---")
        sections.append(f"*Summary generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}*")
        sections.append("*Generated by DevStream Session Management System*")

        return "\n".join(sections)

    def _calculate_duration(self, session: Any) -> str:
        """Calculate and format session duration.

        Args:
            session: Session object

        Returns:
            Formatted duration string
        """
        duration_seconds = self._get_duration_seconds(session)

        if duration_seconds < 60:
            return f"{duration_seconds}s"
        elif duration_seconds < 3600:
            minutes = int(duration_seconds / 60)
            seconds = duration_seconds % 60
            return f"{minutes}m {seconds}s"
        else:
            hours = int(duration_seconds / 3600)
            minutes = int((duration_seconds % 3600) / 60)
            return f"{hours}h {minutes}m"

    def _get_duration_seconds(self, session: Any) -> float:
        """Get session duration in seconds.

        Args:
            session: Session object

        Returns:
            Duration in seconds
        """
        if session.started_at:
            end_time = session.ended_at if session.ended_at else datetime.now()
            return float((end_time - session.started_at).total_seconds())
        return 0.0

    def _calculate_performance_rating(self, session: Any, duration_seconds: float) -> int:
        """Calculate performance rating (1-10).

        Args:
            session: Session object
            duration_seconds: Duration in seconds

        Returns:
            Performance rating (1-10)
        """
        if duration_seconds == 0:
            return 5

        rating = 5  # Base rating

        # Rate based on productivity
        if session.tasks_completed:
            task_rate = session.tasks_completed / max(duration_seconds / 60, 1)
            if task_rate > 2:  # > 2 tasks/minute
                rating += 2
            elif task_rate > 1:  # > 1 task/minute
                rating += 1

        # Rate based on efficiency
        if session.tokens_used and session.tasks_completed:
            tokens_per_task = session.tokens_used / session.tasks_completed
            if tokens_per_task < 1000:  # Efficient token usage
                rating += 1
            elif tokens_per_task > 5000:  # Inefficient token usage
                rating -= 1

        # Rate based on duration
        if duration_seconds < 300:  # < 5 minutes (quick session)
            rating += 1
        elif duration_seconds > 3600:  # > 1 hour (long session)
            rating -= 1

        return max(1, min(10, rating))

    def _get_rating_emoji(self, rating: int) -> str:
        """Get emoji for performance rating.

        Args:
            rating: Performance rating (1-10)

        Returns:
            Rating emoji
        """
        if rating >= 9:
            return "🏆"
        elif rating >= 7:
            return "⭐"
        elif rating >= 5:
            return "✓"
        elif rating >= 3:
            return "⚠️"
        else:
            return "❌"

    def _get_event_icon(self, event_type: str) -> str:
        """Get icon for event type.

        Args:
            event_type: Type of event

        Returns:
            Event icon
        """
        icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌"
        }
        return icons.get(event_type, "📝")

    def _generate_session_insights(self, session: Any) -> List[str]:
        """Generate insights about the session.

        Args:
            session: Session object

        Returns:
            List of insight strings
        """
        insights = []

        # Productivity insights
        if session.tasks_completed and session.tasks_completed > 0:
            if session.tasks_completed >= 10:
                insights.append("High productivity - completed many tasks")
            elif session.tasks_completed >= 5:
                insights.append("Good productivity - several tasks completed")
            else:
                insights.append("Low task completion count")

        # Token usage insights
        if session.tokens_used and session.tokens_used > 0:
            if session.tasks_completed > 0:
                tokens_per_task = session.tokens_used / session.tasks_completed
                if tokens_per_task < 500:
                    insights.append("Efficient token usage")
                elif tokens_per_task > 2000:
                    insights.append("High token consumption per task")

        # File operations insights
        if session.files_modified and session.files_modified > 0:
            if session.tasks_completed == 0:
                insights.append("File modifications without task completions")
            elif session.files_modified > session.tasks_completed * 2:
                insights.append("Many file operations relative to tasks")

        # Duration insights
        duration = self._get_duration_seconds(session)
        if duration < 60:
            insights.append("Quick session - short duration")
        elif duration > 3600:
            insights.append("Extended session - long duration")

        return insights

    async def store_summary_in_memory(self, session_id: str, summary: str) -> bool:
        """Store summary in memory system using Direct DB.

        Args:
            session_id: Session identifier
            summary: Generated summary

        Returns:
            True if stored successfully, False otherwise
        """
        try:
            # Use Direct DB system instead of deprecated MCP
            sys.path.append('.claude/hooks/devstream/utils')
            from direct_client import get_direct_client

            client = get_direct_client()
            result = await client.store_memory(
                content=summary,
                content_type="session_summary",
                keywords=[f"session_{session_id}", "summary", "devstream"],
                session_id=session_id
            )

            self._logger.info(
                "Session summary stored in Direct DB memory",
                extra={
                    "session_id": session_id,
                    "summary_length": len(summary),
                    "memory_id": result.get("memory_id") if result else None
                }
            )

            return result and result.get("success", False)

        except Exception as e:
            self._logger.error(
                "Failed to store summary in Direct DB memory",
                extra={"session_id": session_id, "error": str(e)}
            )
            return False

    async def get_session_summaries(self, session_ids: List[str]) -> Dict[str, str]:
        """Get summaries for multiple sessions (placeholder for MCP integration).

        Args:
            session_ids: List of session identifiers

        Returns:
            Dictionary mapping session IDs to summaries
        """
        summaries = {}

        for session_id in session_ids:
            try:
                # TODO: Retrieve from MCP memory storage
                # For now, generate new summaries
                summary = await self.generate_summary(session_id)
                summaries[session_id] = summary

            except Exception as e:
                self._logger.error(
                    "Failed to get summary for session",
                    extra={"session_id": session_id, "error": str(e)}
                )
                summaries[session_id] = f"# Summary unavailable for {session_id}\n\nError: {str(e)}"

        return summaries