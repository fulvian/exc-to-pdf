#!/usr/bin/env python3
"""
Simplified DevStream SessionEnd Hook - Context7 Implementation

Ends session using simplified SessionManager architecture.
Follows cchooks patterns and Context7 best practices.

Flow:
1. Parse hook input using cchooks create_context()
2. Extract or find active session_id
3. Stop SessionTracker tracking
4. Generate session summary using SessionSummary
5. Mark session as completed using SessionManager.end_session()
6. Store session summary in memory
7. Return success using cchooks output patterns

Context7 Patterns:
- cchooks create_context() for hook input parsing
- SessionManager with aiosqlite async patterns
- SessionTracker with AnyIO task groups (cleanup)
- SessionSummary with structlog context binding
- Non-blocking approach - session ends even if some steps fail
"""

import sys
import json
import asyncio
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / ".claude"))

# Import cchooks for Context7 hook patterns
try:
    from cchooks import create_context
    from cchooks.contexts.base import BaseHookContext
    CCHOOKS_AVAILABLE = True
except ImportError:
    print("Warning: cchooks not available, using fallback mode", file=sys.stderr)
    CCHOOKS_AVAILABLE = False

# Import simplified session management
from hooks.devstream.sessions.session_manager import SessionManager, SessionException
from hooks.devstream.sessions.session_tracker import SessionTracker, TrackingException
from hooks.devstream.sessions.session_summary import SessionSummary, SummaryException

import structlog

logger = structlog.get_logger(__name__)


class SimplifiedSessionEndHook:
    """
    Simplified SessionEnd hook using Context7 patterns.

    Responsibilities:
    - Parse hook input using cchooks create_context()
    - Find active session ID or use provided one
    - Stop tracking with SessionTracker
    - Generate session summary using SessionSummary
    - Mark session as completed using SessionManager
    - Store summary in memory system
    - Provide clean success/failure feedback
    - Non-blocking: session ends even if individual steps fail
    """

    def __init__(self):
        """Initialize the hook with required components."""
        self._logger = structlog.get_logger(__name__).bind(component="SimplifiedSessionEnd")
        self.session_manager: Optional[SessionManager] = None
        self.session_tracker: Optional[SessionTracker] = None
        self.session_summary: Optional[SessionSummary] = None

    async def _initialize_components(self) -> None:
        """Initialize session management components."""
        try:
            # Use DevStream database path
            db_path = project_root / "data" / "devstream.db"

            # Ensure data directory exists
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Initialize SessionManager singleton
            self.session_manager = await SessionManager.get_instance(str(db_path))

            # Initialize SessionTracker
            self.session_tracker = SessionTracker(self.session_manager)

            # Initialize SessionSummary
            self.session_summary = SessionSummary(self.session_manager)

            self._logger.info(
                "Session components initialized",
                extra={"db_path": str(db_path)}
            )

        except Exception as e:
            self._logger.error(
                "Failed to initialize session components",
                extra={"error": str(e)}
            )
            raise SessionException(f"Component initialization failed: {e}") from e

    def _parse_hook_input(self) -> Dict[str, Any]:
        """
        Parse hook input using cchooks or fallback method.

        Returns:
            Parsed hook input data
        """
        if CCHOOKS_AVAILABLE:
            try:
                # Use cchooks create_context() for Context7 pattern
                context = create_context()

                # Extract session info from context
                hook_data = {
                    "session_id": getattr(context, 'session_id', None),
                    "transcript_path": getattr(context, 'transcript_path', None),
                    "cwd": getattr(context, 'cwd', None),
                    "hook_event_name": getattr(context, 'hook_event_name', 'SessionEnd')
                }

                self._logger.debug(
                    "Hook input parsed via cchooks",
                    extra=hook_data
                )

                return hook_data

            except Exception as e:
                self._logger.warning(
                    "cchooks parsing failed, using fallback",
                    extra={"error": str(e)}
                )

        # Fallback: read from stdin
        try:
            input_data = json.load(sys.stdin)
            self._logger.debug(
                "Hook input parsed via fallback",
                extra=input_data
            )
            return input_data

        except json.JSONDecodeError as e:
            self._logger.error(
                "Invalid JSON input",
                extra={"error": str(e)}
            )
            return {}

    async def _get_active_session_id(self, hook_data: Dict[str, Any]) -> Optional[str]:
        """
        Get currently active session ID.

        Args:
            hook_data: Parsed hook input data

        Returns:
            Active session ID or None
        """
        # Try to get from hook data first
        session_id = hook_data.get("session_id")

        if not session_id:
            # Try environment variable
            import os
            session_id = os.environ.get('CLAUDE_SESSION_ID')

        if not session_id:
            # Look for active sessions in database
            try:
                active_sessions = await self.session_manager.list_active_sessions()
                if active_sessions:
                    # Use the most recently started active session
                    session_id = active_sessions[0].id
                    self._logger.debug(
                        "Found active session in database",
                        extra={"session_id": session_id}
                    )
            except Exception as e:
                self._logger.warning(
                    "Failed to query active sessions",
                    extra={"error": str(e)}
                )

        return session_id

    async def _end_session(self, session_id: str) -> Dict[str, Any]:
        """
        End session and generate summary.

        Args:
            session_id: Session identifier

        Returns:
            Session end results
        """
        results = {
            "success": False,
            "session_id": session_id,
            "session_ended": False,
            "tracking_stopped": False,
            "summary_generated": False,
            "summary_stored": False,
            "error": None,
            "warnings": []
        }

        # Get session info before ending
        session = await self.session_manager.get_session(session_id)
        if not session:
            results["error"] = f"Session {session_id} not found"
            return results

        self._logger.info(
            "Ending session",
            extra={
                "session_id": session_id,
                "current_status": session.status,
                "duration": (datetime.now() - session.started_at).total_seconds() if session.started_at else 0
            }
        )

        # Step 1: Stop tracking (non-blocking)
        try:
            await self.session_tracker.stop_tracking(session_id)
            results["tracking_stopped"] = True
            self._logger.debug(
                "Session tracking stopped",
                extra={"session_id": session_id}
            )
        except TrackingException as e:
            warning = f"Failed to stop tracking: {str(e)}"
            results["warnings"].append(warning)
            self._logger.warning(
                "Session tracking stop failed",
                extra={"session_id": session_id, "error": str(e)}
            )
        except Exception as e:
            warning = f"Unexpected tracking error: {str(e)}"
            results["warnings"].append(warning)
            self._logger.error(
                "Unexpected tracking error",
                extra={"session_id": session_id, "error": str(e)}
            )

        # Step 2: Generate summary (non-blocking)
        summary = None
        try:
            if self.session_summary:
                summary = await self.session_summary.generate_summary(session_id)
                results["summary_generated"] = True
                self._logger.info(
                    "Session summary generated",
                    extra={
                        "session_id": session_id,
                        "summary_length": len(summary) if summary else 0
                    }
                )
        except SummaryException as e:
            warning = f"Failed to generate summary: {str(e)}"
            results["warnings"].append(warning)
            self._logger.warning(
                "Session summary generation failed",
                extra={"session_id": session_id, "error": str(e)}
            )
        except Exception as e:
            warning = f"Unexpected summary error: {str(e)}"
            results["warnings"].append(warning)
            self._logger.error(
                "Unexpected summary error",
                extra={"session_id": session_id, "error": str(e)}
            )

        # Step 3: Store summary in memory (non-blocking)
        if summary:
            try:
                if self.session_summary:
                    stored = await self.session_summary.store_summary_in_memory(session_id, summary)
                    if stored:
                        results["summary_stored"] = True
                        self._logger.debug(
                            "Session summary stored in memory",
                            extra={"session_id": session_id}
                        )
            except Exception as e:
                warning = f"Failed to store summary: {str(e)}"
                results["warnings"].append(warning)
                self._logger.warning(
                    "Session summary storage failed",
                    extra={"session_id": session_id, "error": str(e)}
                )

        # Step 4: Mark session as completed (critical step)
        try:
            success = await self.session_manager.end_session(session_id)
            if success:
                results["session_ended"] = True
                self._logger.info(
                    "Session marked as completed",
                    extra={"session_id": session_id}
                )
            else:
                warning = "Session already completed or not found"
                results["warnings"].append(warning)
                self._logger.warning(
                    "Session end had no effect",
                    extra={"session_id": session_id}
                )
        except SessionException as e:
            results["error"] = f"Failed to end session: {str(e)}"
            self._logger.error(
                "Session end failed",
                extra={"session_id": session_id, "error": str(e)}
            )
            return results
        except Exception as e:
            results["error"] = f"Unexpected session end error: {str(e)}"
            self._logger.error(
                "Unexpected session end error",
                extra={"session_id": session_id, "error": str(e)}
            )
            return results

        # If we got here, session was ended successfully
        results["success"] = True

        # Add summary preview to results
        if summary:
            # Take first 200 characters as preview
            preview = summary[:200] + "..." if len(summary) > 200 else summary
            results["summary_preview"] = preview

        return results

    async def run_hook(self) -> Dict[str, Any]:
        """
        Execute the SessionEnd hook.

        Returns:
            Hook execution results
        """
        self._logger.info("Starting simplified SessionEnd hook")

        try:
            # Initialize components
            await self._initialize_components()

            # Parse hook input
            hook_data = self._parse_hook_input()

            # Get active session ID
            session_id = await self._get_active_session_id(hook_data)

            if not session_id:
                self._logger.error("No active session found")
                return {
                    "success": False,
                    "error": "No active session found"
                }

            # Bind session context to structlog
            structlog.contextvars.bind_contextvars(
                session_id=session_id,
                component="SimplifiedSessionEnd",
                operation="session_termination"
            )

            # End session
            results = await self._end_session(session_id)

            if results["success"]:
                self._logger.info(
                    "SessionEnd hook completed successfully",
                    extra={
                        "session_id": session_id,
                        "ended": results.get("session_ended", False),
                        "tracking_stopped": results.get("tracking_stopped", False),
                        "summary_generated": results.get("summary_generated", False),
                        "warnings_count": len(results.get("warnings", []))
                    }
                )
            else:
                self._logger.error(
                    "SessionEnd hook failed",
                    extra={"session_id": session_id, "error": results.get("error")}
                )

            return results

        except Exception as e:
            self._logger.error(
                "SessionEnd hook execution failed",
                extra={"error": str(e)}
            )
            return {
                "success": False,
                "error": str(e)
            }

        finally:
            # Clear context
            structlog.contextvars.clear_contextvars()


async def _get_full_summary(session_id: str) -> Optional[str]:
    """
    Get the full session summary (not just preview).

    Args:
        session_id: Session identifier

    Returns:
        Full summary markdown or None if not available
    """
    try:
        # Try to get the summary from memory if it was stored
        # For now, regenerate the full summary
        from .session_summary import SessionSummary

        # Create a temporary SessionSummary to generate the full summary
        db_path = project_root / "data" / "devstream.db"
        session_manager = await SessionManager.get_instance(str(db_path))
        session_summary = SessionSummary(session_manager)

        return await session_summary.generate_summary(session_id)

    except Exception as e:
        logger.error(f"Failed to get full summary for {session_id}: {e}")
        return None


async def main():
    """Main entry point for the SessionEnd hook."""
    hook = SimplifiedSessionEndHook()
    results = await hook.run_hook()

    # Output results - ALWAYS use readable format to show summary
    if results["success"]:
        print(f"✅ Session ended: {results['session_id']}")

        if results.get("session_ended"):
            print("   📝 Session marked as completed")

        if results.get("tracking_stopped"):
            print("   📊 Tracking stopped")

        if results.get("summary_generated"):
            print("   📄 Summary generated")

        if results.get("summary_stored"):
            print("   💾 Summary stored in memory")

        # Show warnings if any
        if results.get("warnings"):
            print("   ⚠️  Warnings:")
            for warning in results["warnings"]:
                print(f"      - {warning}")

        # Show full summary if available
        if results.get("summary_preview"):
            # Get the full summary instead of just preview
            full_summary = await _get_full_summary(results['session_id'])
            if full_summary:
                print(f"\n📋 SESSION SUMMARY:\n{full_summary}")
            else:
                print(f"\n📋 Summary Preview:\n{results['summary_preview']}")

        sys.exit(0)
    else:
        error_msg = results.get("error", "Unknown error")
        if CCHOOKS_AVAILABLE:
            # Use cchooks error pattern
            print(json.dumps({
                "decision": "block",
                "reason": f"SessionEnd failed: {error_msg}"
            }))
        else:
            print(f"❌ SessionEnd failed: {error_msg}", file=sys.stderr)

        sys.exit(2)


if __name__ == "__main__":
    """Entry point with asyncio loop safety."""
    try:
        # Check if event loop is already running
        loop = asyncio.get_running_loop()

        # Schedule task in existing loop
        task = loop.create_task(main())

    except RuntimeError:
        # No event loop running, create new one
        asyncio.run(main())