#!/usr/bin/env python3
"""
DevStream SessionStart Hook - Session Initialization

Initializes work session in work_sessions table using WorkSessionManager.
Integrates with Claude Code SessionStart hook system.

Flow:
1. Extract session_id from environment or hook payload
2. Call WorkSessionManager.resume_session() (creates or resumes)
3. Bind session context using structlog (automatic log inheritance)
4. Store initialization event in memory
5. Return success

Context7 Patterns:
- WorkSessionManager uses aiosqlite async patterns
- structlog context binding for automatic session_id in logs
"""

import sys
import os
import asyncio
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Import DevStream utilities
sys.path.append(str(Path(__file__).parent.parent / 'utils'))
from common import DevStreamHookBase, get_project_context
from logger import get_devstream_logger
from session_coordinator import get_session_coordinator

# Import WorkSessionManager and cleanup utilities
sys.path.append(str(Path(__file__).parent))
from work_session_manager import WorkSessionManager
from session_cleanup_utils import SessionCleanupManager


class SessionStartHook:
    """
    SessionStart hook for DevStream session initialization.

    Responsibilities:
    - Initialize or resume work session in work_sessions table
    - Bind session context for automatic log propagation
    - Track session start in memory system
    - Provide session info to other hooks
    """

    def __init__(self):
        self.hook_type = 'session_start'
        self.structured_logger = get_devstream_logger('session_start')
        self.logger = self.structured_logger.logger  # Compatibility
        self.session_manager = WorkSessionManager()

        # Session coordinator for multi-session management
        self.coordinator = get_session_coordinator()

        # Enhanced cleanup manager for zombie session handling
        self.cleanup_manager = SessionCleanupManager(self.coordinator)

    def get_session_id(self) -> str:
        """
        Get session ID from environment or generate new one.

        Returns:
            str: Session identifier
        """
        # Try to get from environment (Claude Code may provide this)
        session_id = os.environ.get('CLAUDE_SESSION_ID')

        if not session_id:
            # Generate new session ID
            session_id = f"sess-{uuid.uuid4().hex[:16]}"
            self.logger.debug(f"Generated new session ID: {session_id}")

        return session_id

    async def initialize_session(self, session_id: str) -> Dict[str, Any]:
        """
        Initialize work session using WorkSessionManager.

        Args:
            session_id: Session identifier

        Returns:
            Dict with session initialization results
        """
        results = {
            "success": False,
            "session_id": session_id,
            "session_created": False,
            "session_resumed": False,
            "error": None
        }

        try:
            # Session ID-based idempotency check (v2 - multi-session safe)
            # Check if session already exists and is active before cleanup
            existing_session = await self.session_manager.get_session(session_id)

            if existing_session and existing_session.status == "active":
                self.logger.info(
                    f"Session {session_id[:12]}... already initialized - idempotent return"
                )

                # Update last_activity_at and return existing session
                await self.session_manager.resume_session(session_id)

                # Bind context for automatic log propagation
                self.session_manager.bind_session_context(
                    session_id=existing_session.id,
                    session_name=existing_session.session_name
                )

                results["success"] = True
                results["session_resumed"] = True
                results["session_data"] = {
                    "id": existing_session.id,
                    "status": existing_session.status,
                    "started_at": existing_session.started_at.isoformat(),
                    "tokens_used": existing_session.tokens_used
                }

                self.logger.debug(f"Idempotent return for active session: {session_id[:12]}...")
                return results

            # Proactive cleanup of zombie sessions before checking limits
            self.logger.info("Performing proactive session cleanup...")
            cleanup_stats = self.cleanup_manager.aggressive_cleanup()

            if cleanup_stats.zombie_sessions_cleaned > 0 or cleanup_stats.stale_sessions_cleaned > 0:
                self.logger.info(
                    f"Proactive cleanup removed {cleanup_stats.zombie_sessions_cleaned} zombie "
                    f"and {cleanup_stats.stale_sessions_cleaned} stale sessions"
                )

            # Validate registry integrity
            if not self.cleanup_manager.validate_and_fix_registry():
                self.logger.warning("Registry validation failed, attempting emergency repair")
                if not self.cleanup_manager.force_cleanup_all_sessions():
                    raise RuntimeError("Failed to repair session registry")

            # Check session limits via coordinator (after cleanup)
            if self.coordinator.is_session_limit_reached():
                # Emergency override if still at limit after cleanup
                if self.cleanup_manager.EMERGENCY_OVERRIDE:
                    self.logger.warning(
                        f"Session limit still reached after cleanup, using emergency override"
                    )
                    # Force cleanup of all sessions as last resort
                    if not self.cleanup_manager.force_cleanup_all_sessions():
                        raise RuntimeError(
                            f"Session limit reached ({self.coordinator.MAX_SESSIONS} sessions) "
                            f"and emergency cleanup failed. "
                            f"Please manually delete {self.coordinator.registry_path}"
                        )
                else:
                    raise RuntimeError(
                        f"Session limit reached ({self.coordinator.MAX_SESSIONS} sessions). "
                        f"Please close an existing session before starting a new one."
                    )

            # Register session with coordinator
            db_path = self.session_manager.db_path
            if not self.coordinator.register_session(session_id, db_path):
                raise RuntimeError("Failed to register session with coordinator")

            self.logger.info(
                f"Session registered with coordinator: {session_id}",
                extra={"active_sessions": self.coordinator.get_session_count()}
            )

            # Resume or create session
            self.logger.info(f"Initializing session: {session_id}")
            session = await self.session_manager.resume_session(session_id)

            # Bind context for automatic log propagation
            self.session_manager.bind_session_context(
                session_id=session.id,
                session_name=session.session_name
            )

            # Determine if created or resumed
            if session.tokens_used == 0:
                results["session_created"] = True
                self.logger.info(f"Created new work session: {session_id}")
            else:
                results["session_resumed"] = True
                self.logger.info(f"Resumed existing work session: {session_id}, tokens_used={session.tokens_used}")

            results["success"] = True
            results["session_data"] = {
                "id": session.id,
                "status": session.status,
                "started_at": session.started_at.isoformat(),
                "tokens_used": session.tokens_used
            }

        except Exception as e:
            results["error"] = str(e)
            self.structured_logger.log_hook_error(e, {
                "session_id": session_id,
                "operation": "initialize_session"
            })

        return results

    async def display_all_pending_summaries(self) -> int:
        """
        Display ALL pending session summaries from session-specific marker files (Phase 4).

        Iterates all marker files in ~/.claude/state/devstream_session_*.txt,
        displays summaries for sessions with summary_displayed=False,
        updates registry, and deletes marker files.

        Returns:
            Number of summaries displayed

        Note:
            Supports multi-session scenarios (Sonnet 4.5 + GLM-4.6 concurrent).
            Thread-safe registry updates via SessionCoordinator.
        """
        import glob
        import time

        state_dir = Path.home() / ".claude" / "state"
        marker_pattern = str(state_dir / "devstream_session_*.txt")

        # Find all session-specific marker files
        marker_files = glob.glob(marker_pattern)

        if not marker_files:
            self.logger.debug("No pending session summaries found")
            return 0

        self.logger.info(f"Found {len(marker_files)} session-specific marker files")

        displayed_count = 0

        for marker_file_path in marker_files:
            try:
                marker_file = Path(marker_file_path)

                # Extract session_id from filename: devstream_session_{session_id}.txt
                filename = marker_file.name
                if not filename.startswith("devstream_session_"):
                    continue

                session_id = filename.replace("devstream_session_", "").replace(".txt", "")

                # Check if summary already displayed in registry
                if not self.coordinator._acquire_lock(timeout=5):
                    self.logger.warning(f"Failed to acquire lock for {session_id}, skipping")
                    continue

                try:
                    sessions = self.coordinator._read_registry()

                    # Check if session exists and summary not displayed
                    if session_id in sessions:
                        session_info = sessions[session_id]
                        if session_info.summary_displayed:
                            self.logger.debug(f"Summary already displayed for {session_id}, skipping")
                            # Delete marker file even if already displayed
                            marker_file.unlink()
                            continue

                    # Read and display summary
                    with open(marker_file, "r") as f:
                        summary = f.read()

                    if summary and len(summary.strip()) > 0:
                        # Display summary to user
                        print("\n" + "=" * 70)
                        print(f"📋 SESSION SUMMARY - {session_id[:12]}...")
                        print("=" * 70)
                        print(summary)
                        print("=" * 70 + "\n")

                        displayed_count += 1
                        self.logger.info(f"Displayed summary for session {session_id}")

                        # Update registry: mark summary as displayed
                        if session_id in sessions:
                            sessions[session_id].summary_displayed = True
                            self.coordinator._write_registry(sessions)
                            self.coordinator._sessions_cache = sessions

                        # Delete marker file after display
                        marker_file.unlink()
                        self.logger.debug(f"Deleted marker file: {marker_file.name}")

                finally:
                    self.coordinator._release_lock()

            except Exception as e:
                self.logger.error(f"Failed to process marker file {marker_file_path}: {e}")
                continue

        if displayed_count > 0:
            self.logger.info(f"Displayed {displayed_count} session summaries")

        return displayed_count

    async def cleanup_old_sessions(self, retention_days: int = 7) -> int:
        """
        Cleanup old sessions and zombie sessions (Phase 4).

        Removes:
        - Sessions with status "ended" older than retention_days
        - Zombie sessions (process PID no longer exists)
        - Associated marker files

        Args:
            retention_days: Retention period for ended sessions (default: 7 days)

        Returns:
            Number of sessions cleaned up

        Note:
            Uses psutil for PID validation (Context7 pattern).
            Thread-safe via SessionCoordinator locking.
        """
        import time
        import psutil

        cleanup_count = 0
        current_time = time.time()
        retention_seconds = retention_days * 24 * 3600

        if not self.coordinator._acquire_lock(timeout=10):
            self.logger.error("Failed to acquire lock for session cleanup")
            return 0

        try:
            sessions = self.coordinator._read_registry()
            sessions_to_remove = []

            for session_id, session_info in sessions.items():
                should_remove = False
                reason = ""

                # Check 1: Zombie sessions (PID doesn't exist)
                if not psutil.pid_exists(session_info.pid):
                    should_remove = True
                    reason = f"zombie (PID {session_info.pid} doesn't exist)"

                # Check 2: Old ended sessions (retention period exceeded)
                elif session_info.status == "ended" and session_info.ended_at:
                    age_seconds = current_time - session_info.ended_at
                    if age_seconds > retention_seconds:
                        should_remove = True
                        age_days = age_seconds / 86400
                        reason = f"expired (ended {age_days:.1f} days ago, retention={retention_days} days)"

                if should_remove:
                    self.logger.info(f"Cleaning up session {session_id}: {reason}")
                    sessions_to_remove.append(session_id)

                    # Delete associated marker file if exists
                    if session_info.marker_file_path:
                        marker_file = Path(session_info.marker_file_path)
                        if marker_file.exists():
                            marker_file.unlink()
                            self.logger.debug(f"Deleted marker file: {marker_file}")

            # Remove sessions from registry
            for session_id in sessions_to_remove:
                del sessions[session_id]
                cleanup_count += 1

            # Write updated registry if changes made
            if cleanup_count > 0:
                self.coordinator._write_registry(sessions)
                self.coordinator._sessions_cache = sessions
                self.logger.info(f"Cleaned up {cleanup_count} sessions")

        finally:
            self.coordinator._release_lock()

        return cleanup_count

    async def migrate_legacy_marker_file(self) -> bool:
        """
        Migrate legacy devstream_last_session.txt to session-specific format (Phase 4).

        If legacy marker file exists:
        1. Read summary content
        2. Create session-specific marker file for a legacy session
        3. Update registry with legacy session info
        4. Delete legacy marker file

        Returns:
            True if migration performed, False if no legacy file

        Note:
            One-time migration for backward compatibility.
            Creates synthetic session ID for legacy summary.
        """
        import time
        import hashlib

        legacy_file = Path.home() / ".claude" / "state" / "devstream_last_session.txt"

        if not legacy_file.exists():
            return False

        try:
            self.logger.info("Found legacy marker file, migrating to session-specific format")

            # Read legacy summary
            with open(legacy_file, "r") as f:
                summary = f.read()

            if not summary or len(summary.strip()) == 0:
                # Empty legacy file, just delete it
                legacy_file.unlink()
                self.logger.debug("Deleted empty legacy marker file")
                return False

            # Generate synthetic session ID for legacy summary
            # Use hash of summary content for deterministic ID
            summary_hash = hashlib.sha256(summary.encode()).hexdigest()[:16]
            legacy_session_id = f"sess-legacy-{summary_hash}"

            # Create session-specific marker file
            marker_file = (
                Path.home() / ".claude" / "state" /
                f"devstream_session_{legacy_session_id}.txt"
            )

            with open(marker_file, "w") as f:
                f.write(summary)

            self.logger.info(f"Created session-specific marker file: {marker_file.name}")

            # Update registry with legacy session info
            if not self.coordinator._acquire_lock(timeout=5):
                self.logger.warning("Failed to acquire lock for legacy migration")
                # Still delete legacy file even if registry update fails
                legacy_file.unlink()
                return True

            try:
                from session_coordinator import SessionInfo

                sessions = self.coordinator._read_registry()

                # Create synthetic SessionInfo for legacy session
                legacy_session_info = SessionInfo(
                    session_id=legacy_session_id,
                    pid=0,  # Unknown PID
                    started_at=time.time() - 86400,  # Assume 1 day ago
                    last_heartbeat=time.time() - 86400,
                    status="ended",
                    ended_at=time.time() - 3600,  # Assume ended 1 hour ago
                    marker_file_path=str(marker_file),
                    compaction_events=[],
                    summary_displayed=False,
                    model_type="unknown",
                    session_name="Legacy Session"
                )

                sessions[legacy_session_id] = legacy_session_info
                self.coordinator._write_registry(sessions)
                self.coordinator._sessions_cache = sessions

                self.logger.info(f"Registered legacy session in registry: {legacy_session_id}")

            finally:
                self.coordinator._release_lock()

            # Delete legacy marker file
            legacy_file.unlink()
            self.logger.info("Deleted legacy marker file")

            return True

        except Exception as e:
            self.logger.error(f"Failed to migrate legacy marker file: {e}")
            return False

    async def display_previous_summary(self) -> None:
        """
        Display previous session summary (Phase 4 - refactored).

        Phase 4 Workflow:
        1. Migrate legacy marker file (if exists)
        2. Cleanup old/zombie sessions
        3. Display ALL pending summaries (session-specific marker files)

        Note:
            Replaces single-summary display with multi-summary support.
            Backward compatible with legacy devstream_last_session.txt.
        """
        # Step 1: Migrate legacy marker file to session-specific format
        legacy_migrated = await self.migrate_legacy_marker_file()
        if legacy_migrated:
            self.logger.info("Legacy marker file migrated to session-specific format")

        # Step 2: Cleanup old and zombie sessions (proactive maintenance)
        cleanup_count = await self.cleanup_old_sessions(retention_days=7)
        if cleanup_count > 0:
            self.logger.info(f"Cleaned up {cleanup_count} old/zombie sessions")

        # Step 3: Display ALL pending summaries
        displayed_count = await self.display_all_pending_summaries()
        if displayed_count > 0:
            self.logger.info(f"Displayed {displayed_count} pending session summaries")
        else:
            self.logger.debug("No pending summaries to display")

    async def run_hook(self, hook_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute SessionStart hook.

        Args:
            hook_data: Optional hook execution data

        Returns:
            Hook execution results
        """
        self.structured_logger.log_hook_start(hook_data or {}, {
            "phase": "session_start"
        })

  # Display previous session summary (if available)
        # TEMPORARILY DISABLED: Commented out to prevent summary display in multi-session scenarios
        # await self.display_previous_summary()
        # TODO: Re-enable when single-session workflow or manual summary retrieval is needed

        # Get session ID
        session_id = self.get_session_id()

        # Initialize session
        results = await self.initialize_session(session_id)

        # Log completion
        if results["success"]:
            self.structured_logger.log_hook_success({
                "session_id": session_id,
                "created": results.get("session_created", False),
                "resumed": results.get("session_resumed", False)
            })
        else:
            self.logger.error(f"SessionStart failed: {results.get('error')}")

        return results


async def main():
    """
    Main entry point for SessionStart hook.

    Called by Claude Code hook system.
    """
    hook = SessionStartHook()
    results = await hook.run_hook()

    # Output results for hook system
    if results["success"]:
        print(f"✅ Session initialized: {results['session_id']}")
        if results.get("session_created"):
            print("   📝 New session created in sessions table")
        elif results.get("session_resumed"):
            print("   🔄 Existing session resumed")
    else:
        print(f"❌ SessionStart failed: {results.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    """
    SessionStart hook entry point with asyncio loop safety.

    Handles two execution contexts:
    1. Claude Code hooks (event loop already running)
    2. Standalone execution (no event loop)

    Fix: Never call run_until_complete() on running loop.
    Reference: https://docs.python.org/3/library/asyncio-task.html#asyncio.get_running_loop

    Exception Handling:
    - CancelledError: Task cancelled during execution (graceful warning)
    - RuntimeError: No loop vs loop closed/thread mismatch (distinguish)
    - Generic Exception: Catch-all with detailed logging + re-raise
    """
    import structlog

    logger = structlog.get_logger()

    try:
        # Attempt to get existing running loop
        loop = asyncio.get_running_loop()

        # CORRECT: Schedule task in existing loop WITHOUT running it
        # The loop is already running, task will execute automatically
        task = loop.create_task(main())

        logger.debug("SessionStart scheduled in existing event loop",
                    loop_id=id(loop), task_repr=str(task))

        # NOTE: Do NOT await or run_until_complete here!
        # The hook framework will handle task completion.

    except RuntimeError as e:
        # Distinguish between "no running loop" vs other RuntimeErrors
        if "no running event loop" in str(e).lower():
            # Expected case: standalone execution without event loop
            logger.debug("SessionStart creating new event loop")
            try:
                asyncio.run(main())
            except asyncio.CancelledError:
                logger.warning("SessionStart task cancelled during execution")
            except Exception as ex:
                logger.error("SessionStart execution failed",
                           error=str(ex), error_type=type(ex).__name__)
                raise
        else:
            # Other RuntimeError: loop closed, thread mismatch, etc.
            logger.error("SessionStart asyncio runtime error",
                        error=str(e), error_type="RuntimeError")
            raise

    except asyncio.CancelledError:
        # Task cancelled in existing loop (non-critical)
        logger.warning("SessionStart task cancelled in existing loop")

    except Exception as e:
        # Catch-all for unexpected errors
        logger.error("SessionStart unexpected error",
                    error=str(e), error_type=type(e).__name__)
        raise