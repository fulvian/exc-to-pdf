#!/usr/bin/env python3
"""
Simplified DevStream SessionStart Hook - Context7 Implementation

Initializes session using simplified SessionManager architecture.
Follows cchooks patterns and Context7 best practices.

Flow:
1. Parse hook input using cchooks create_context()
2. Extract or generate session_id from input
3. Create session using SessionManager.create_session()
4. Initialize SessionTracker for real-time progress
5. Store session start event in memory
6. Return success using cchooks output patterns

Context7 Patterns:
- cchooks create_context() for hook input parsing
- SessionManager with aiosqlite async patterns
- SessionTracker with AnyIO task groups
- structlog context binding for automatic session propagation
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


class SimplifiedSessionStartHook:
    """
    Simplified SessionStart hook using Context7 patterns.

    Responsibilities:
    - Parse hook input using cchooks create_context()
    - Initialize session using SessionManager
    - Start tracking with SessionTracker
    - Log session start with structlog context binding
    - Provide clean success/failure feedback
    """

    def __init__(self):
        """Initialize the hook with required components."""
        self._logger = structlog.get_logger(__name__).bind(component="SimplifiedSessionStart")
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
                    "hook_event_name": getattr(context, 'hook_event_name', 'SessionStart')
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

    def _get_session_id(self, hook_data: Dict[str, Any]) -> str:
        """
        Extract or generate session ID.

        Args:
            hook_data: Parsed hook input data

        Returns:
            Session identifier
        """
        # Try to get from hook data
        session_id = hook_data.get("session_id")

        if not session_id:
            # Try environment variable (Claude Code may provide this)
            import os
            session_id = os.environ.get('CLAUDE_SESSION_ID')

        if not session_id:
            # Generate new session ID
            session_id = f"sess-{uuid.uuid4().hex[:16]}"
            self._logger.debug(
                "Generated new session ID",
                extra={"session_id": session_id}
            )

        return session_id

    async def _initialize_session(self, session_id: str) -> Dict[str, Any]:
        """
        Initialize session using simplified architecture.

        Args:
            session_id: Session identifier

        Returns:
            Session initialization results
        """
        results = {
            "success": False,
            "session_id": session_id,
            "session_created": False,
            "tracking_started": False,
            "error": None
        }

        try:
            # Check if session already exists
            existing_session = await self.session_manager.get_session(session_id)

            if existing_session and existing_session.status == "active":
                self._logger.info(
                    "Session already active, resuming",
                    extra={"session_id": session_id}
                )

                # Start tracking for existing session
                await self.session_tracker.start_tracking(session_id)

                results.update({
                    "success": True,
                    "session_resumed": True,
                    "tracking_started": True
                })

                return results

            # Create new session
            session = await self.session_manager.create_session(session_id)

            # Update session with metadata
            await self.session_manager.update_session(
                session_id,
                metadata=json.dumps({
                    "hook_type": "SessionStart",
                    "started_at": datetime.now().isoformat(),
                    "project_root": str(project_root)
                })
            )

            self._logger.info(
                "Created new session",
                extra={
                    "session_id": session.id,
                    "started_at": session.started_at.isoformat()
                }
            )

            # Start tracking the new session
            await self.session_tracker.start_tracking(session_id)

            results.update({
                "success": True,
                "session_created": True,
                "tracking_started": True,
                "session_data": {
                    "id": session.id,
                    "status": session.status,
                    "started_at": session.started_at.isoformat()
                }
            })

            # Store session start event in memory only if session summary is enabled
            # Check environment variable to respect .env.devstream configuration
            import os
            session_summary_enabled = os.environ.get('DEVSTREAM_HOOK_SESSIONSTART', 'true').lower() == 'true'

            if self.session_summary and session_summary_enabled:
                await self.session_summary.store_summary_in_memory(
                    session_id,
                    f"# Session Started\n\nSession ID: {session_id}\nStarted: {session.started_at.isoformat()}\nStatus: {session.status}"
                )

            return results

        except SessionException as e:
            results["error"] = str(e)
            self._logger.error(
                "Session initialization failed",
                extra={"session_id": session_id, "error": str(e)}
            )
            raise

        except TrackingException as e:
            results["error"] = str(e)
            self._logger.error(
                "Session tracking failed",
                extra={"session_id": session_id, "error": str(e)}
            )
            # Continue even if tracking fails

        except Exception as e:
            results["error"] = str(e)
            self._logger.error(
                "Unexpected session initialization error",
                extra={"session_id": session_id, "error": str(e)}
            )
            raise

    async def run_hook(self) -> Dict[str, Any]:
        """
        Execute the SessionStart hook.

        Returns:
            Hook execution results
        """
        self._logger.info("Starting simplified SessionStart hook")

        try:
            # Initialize components
            await self._initialize_components()

            # Parse hook input
            hook_data = self._parse_hook_input()

            # Get session ID
            session_id = self._get_session_id(hook_data)

            # Bind session context to structlog
            structlog.contextvars.bind_contextvars(
                session_id=session_id,
                component="SimplifiedSessionStart",
                operation="session_initialization"
            )

            # Initialize session
            results = await self._initialize_session(session_id)

            if results["success"]:
                self._logger.info(
                    "SessionStart hook completed successfully",
                    extra={
                        "session_id": session_id,
                        "created": results.get("session_created", False),
                        "tracking": results.get("tracking_started", False)
                    }
                )
            else:
                self._logger.error(
                    "SessionStart hook failed",
                    extra={"session_id": session_id, "error": results.get("error")}
                )

            return results

        except Exception as e:
            self._logger.error(
                "SessionStart hook execution failed",
                extra={"error": str(e)}
            )
            return {
                "success": False,
                "error": str(e)
            }

        finally:
            # Clear context
            structlog.contextvars.clear_contextvars()


async def main():
    """Main entry point for the SessionStart hook."""
    hook = SimplifiedSessionStartHook()
    results = await hook.run_hook()

    # Output results based on cchooks patterns
    if results["success"]:
        if CCHOOKS_AVAILABLE:
            # Use cchooks output pattern if available
            print(json.dumps({
                "decision": "allow",
                "reason": f"Session initialized: {results['session_id']}"
            }))
        else:
            # Fallback output
            print(f"✅ Session initialized: {results['session_id']}")
            if results.get("session_created"):
                print("   📝 New session created")
            else:
                print("   🔄 Existing session resumed")
            print("   📊 Tracking started")

        sys.exit(0)
    else:
        error_msg = results.get("error", "Unknown error")
        if CCHOOKS_AVAILABLE:
            # Use cchooks error pattern
            print(json.dumps({
                "decision": "block",
                "reason": f"SessionStart failed: {error_msg}"
            }))
        else:
            print(f"❌ SessionStart failed: {error_msg}", file=sys.stderr)

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