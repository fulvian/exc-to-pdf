#!/usr/bin/env python3
"""
DevStream Work Session Manager - Session Lifecycle Management

Context7-compliant session state management using aiosqlite async patterns
and structlog context binding for automatic session context inheritance.

Research Sources:
- aiosqlite: async with context managers, row_factory, explicit commits
- structlog: bind_contextvars() for thread-local context propagation
"""

import sys
import os
import aiosqlite
import structlog
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

# Import DevStream utilities
sys.path.append(str(Path(__file__).parent.parent / 'utils'))
from common import DevStreamHookBase, get_project_context
from logger import get_devstream_logger


@dataclass
class WorkSession:
    """
    Work session data model.

    Represents a complete work session with all tracking data.
    Simplified to match existing 'sessions' table schema.
    """
    id: str
    tokens_used: int
    status: str
    started_at: datetime
    ended_at: Optional[datetime]
    files_modified: int = 0
    tasks_completed: int = 0
    metadata: Optional[str] = None

    # Additional fields that can be stored in metadata JSON
    @property
    def plan_id(self) -> Optional[str]:
        """Extract plan_id from metadata if available."""
        if self.metadata:
            import json
            try:
                data = json.loads(self.metadata)
                return data.get('plan_id')
            except:
                return None
        return None

    @property
    def session_name(self) -> Optional[str]:
        """Extract session_name from metadata if available."""
        if self.metadata:
            import json
            try:
                data = json.loads(self.metadata)
                return data.get('session_name')
            except:
                return None
        return None

    @property
    def context_summary(self) -> Optional[str]:
        """Extract context_summary from metadata if available."""
        if self.metadata:
            import json
            try:
                data = json.loads(self.metadata)
                return data.get('context_summary')
            except:
                return None
        return None


class WorkSessionManager:
    """
    Work Session Manager for DevStream session lifecycle management.

    Manages sessions table CRUD operations using Context7-validated
    aiosqlite async patterns and structlog context binding.

    Key Features:
    - Async connection management with context managers
    - Session state tracking in sessions table
    - Automatic context binding for log inheritance
    - Graceful error handling with structured logging

    Context7 Patterns Applied:
    - aiosqlite: async with connect(), row_factory, explicit commits
    - structlog: bind_contextvars() for automatic context propagation
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize WorkSessionManager with Context7-compliant dynamic path resolution.

        Args:
            db_path: Path to DevStream database (auto-detected if None)
        """
        self.structured_logger = get_devstream_logger('work_session_manager')
        self.logger = self.structured_logger.logger  # Compatibility

        # Context7 Pattern: Dynamic database path resolution with priority order
        if db_path is None:
            self.db_path = self._get_dynamic_db_path()
        else:
            self.db_path = db_path

        self.logger.info(f"WorkSessionManager initialized with DB: {self.db_path}")

    def _get_dynamic_db_path(self) -> str:
        """
        Context7-compliant dynamic database path resolution.

        Priority Order (Dynaconf-inspired):
        1. DEVSTREAM_DB_PATH environment variable (explicit override)
        2. DEVSTREAM_PROJECT_ROOT + data/devstream.db (multi-project mode)
        3. Current working directory + data/devstream.db (single-project mode)
        4. Fallback to script location (legacy compatibility)

        Returns:
            str: Resolved database path
        """
        import os

        # Priority 1: Explicit database path from environment
        explicit_db_path = os.getenv("DEVSTREAM_DB_PATH")
        if explicit_db_path:
            self.logger.debug(f"Using explicit DB path from environment: {explicit_db_path}")
            return explicit_db_path

        # Priority 2: Multi-project mode with project root
        project_root = os.getenv("DEVSTREAM_PROJECT_ROOT")
        if project_root:
            db_path = os.path.join(project_root, "data", "devstream.db")
            self.logger.debug(f"Using multi-project DB path: {db_path}")
            return db_path

        # Priority 3: Current working directory (single-project mode)
        current_dir = os.getcwd()
        db_path = os.path.join(current_dir, "data", "devstream.db")

        # Check if database exists in current directory
        if os.path.exists(db_path):
            self.logger.debug(f"Using current directory DB path: {db_path}")
            return db_path

        # Priority 4: Fallback to script location (legacy compatibility)
        # Calculate script location relative to sessions directory
        script_dir = Path(__file__).parent.parent.parent.parent.parent
        fallback_path = str(script_dir / 'data' / 'devstream.db')

        self.logger.debug(f"Using fallback DB path: {fallback_path}")
        return fallback_path

    def _get_connection(self) -> aiosqlite.Connection:
        """
        Get database connection with Context7 aiosqlite pattern.

        Uses async context manager pattern from Context7 research:
        - Returns aiosqlite.connect() directly (Connection object with __aenter__/__aexit__)
        - row_factory set after connection established
        - Caller uses: async with manager._get_connection() as db:

        Returns:
            aiosqlite.Connection: Database connection (not awaited - has async context manager protocol)

        Raises:
            aiosqlite.Error: If connection fails
        """
        # Context7 pattern: Return connection object directly, don't await here
        # The Connection object implements __aenter__/__aexit__ for async with
        conn = aiosqlite.connect(self.db_path)
        # Note: row_factory must be set after await in __aenter__
        return conn

    async def get_session(self, session_id: str) -> Optional[WorkSession]:
        """
        Get session by ID from database.

        Args:
            session_id: Session identifier

        Returns:
            WorkSession if found, None otherwise

        Raises:
            aiosqlite.Error: If database query fails
        """
        async with self._get_connection() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT id, tokens_used, status, started_at, ended_at,
                       files_modified, tasks_completed, metadata
                FROM sessions
                WHERE id = ?
                """,
                (session_id,)
            ) as cursor:
                row = await cursor.fetchone()

                if row is None:
                    return None

                return WorkSession(
                    id=row['id'],
                    tokens_used=row['tokens_used'],
                    status=row['status'],
                    started_at=datetime.fromisoformat(row['started_at']),
                    ended_at=datetime.fromisoformat(row['ended_at']) if row['ended_at'] else None,
                    files_modified=row['files_modified'],
                    tasks_completed=row['tasks_completed'],
                    metadata=row['metadata']
                )

    # Session lifecycle methods will be implemented in next tasks
    async def create_session(
        self,
        session_id: str,
        plan_id: Optional[str] = None,
        session_name: Optional[str] = None,
        context_window_size: Optional[int] = None
    ) -> WorkSession:
        """
        Create new work session in database.

        Args:
            session_id: Unique session identifier
            plan_id: Optional intervention plan ID
            session_name: Optional human-readable session name
            context_window_size: Optional context window size

        Returns:
            WorkSession: Created session object

        Raises:
            aiosqlite.Error: If database operation fails
            ValueError: If session_id is invalid
        """
        if not session_id or not session_id.strip():
            raise ValueError("session_id cannot be empty")

        import json

        try:
            # Prepare data
            now = datetime.now().isoformat()

            # Store additional fields in metadata JSON
            metadata = {}
            if plan_id:
                metadata['plan_id'] = plan_id
            if session_name:
                metadata['session_name'] = session_name
            if context_window_size:
                metadata['context_window_size'] = context_window_size

            metadata_json = json.dumps(metadata) if metadata else None

            # Context7 pattern: async with for connection management + explicit commit
            async with self._get_connection() as db:
                await db.execute(
                    """
                    INSERT INTO sessions (
                        id, tokens_used, status, started_at, files_modified,
                        tasks_completed, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        0,  # tokens_used starts at 0
                        'active',
                        now,  # started_at
                        0,  # files_modified
                        0,  # tasks_completed
                        metadata_json
                    )
                )
                await db.commit()  # Explicit commit (Context7 pattern)

            self.logger.info(f"Created work session: {session_id}")

            # Return created session
            return WorkSession(
                id=session_id,
                tokens_used=0,
                status='active',
                started_at=datetime.fromisoformat(now),
                ended_at=None,
                files_modified=0,
                tasks_completed=0,
                metadata=metadata_json
            )

        except aiosqlite.IntegrityError as e:
            self.logger.error(f"Session already exists: {session_id}")
            raise ValueError(f"Session {session_id} already exists") from e
        except Exception as e:
            self.logger.error(f"Failed to create session {session_id}: {e}")
            raise

    async def resume_session(self, session_id: str) -> WorkSession:
        """
        Resume existing session or create new one.

        Logic:
        1. Try to get existing session from database
        2. If found: return existing session (sessions table doesn't have last_activity_at)
        3. If not found: Call create_session() to create new one

        Args:
            session_id: Session identifier to resume

        Returns:
            WorkSession: Resumed or newly created session

        Raises:
            aiosqlite.Error: If database operation fails
        """
        # Try to get existing session
        existing_session = await self.get_session(session_id)

        if existing_session is not None:
            # Session exists - return it
            self.logger.info(f"Resumed existing work session: {session_id}")
            return existing_session
        else:
            # Session doesn't exist - create new one
            self.logger.info(f"Session {session_id} not found, creating new session")
            return await self.create_session(
                session_id=session_id,
                session_name=f"Session {session_id[:8]}"
            )

    async def update_session_progress(
        self,
        session_id: str,
        tokens_delta: int = 0,
        tasks_completed_delta: int = 0,
        files_modified_delta: int = 0
    ) -> bool:
        """
        Update session progress metrics.

        Args:
            session_id: Session to update
            tokens_delta: Token count increment (added to existing tokens_used)
            tasks_completed_delta: Tasks completed increment
            files_modified_delta: Files modified increment

        Returns:
            bool: True if update successful

        Raises:
            aiosqlite.Error: If database operation fails
        """
        # Build UPDATE query dynamically based on what's provided
        updates = []
        params = []

        if tokens_delta != 0:
            updates.append("tokens_used = tokens_used + ?")
            params.append(tokens_delta)

        if tasks_completed_delta != 0:
            updates.append("tasks_completed = tasks_completed + ?")
            params.append(tasks_completed_delta)

        if files_modified_delta != 0:
            updates.append("files_modified = files_modified + ?")
            params.append(files_modified_delta)

        if not updates:
            # Nothing to update
            return True

        # Add session_id for WHERE clause
        params.append(session_id)

        query = f"UPDATE sessions SET {', '.join(updates)} WHERE id = ?"

        async with self._get_connection() as db:
            cursor = await db.execute(query, params)
            await db.commit()

            # Check if any row was updated
            if cursor.rowcount == 0:
                self.logger.warning(f"No session found to update: {session_id}")
                return False

        self.logger.debug(f"Updated session progress: {session_id}, tokens_delta={tokens_delta}")
        return True

    async def end_session(
        self,
        session_id: str,
        context_summary: Optional[str] = None
    ) -> bool:
        """
        End work session and mark as completed.

        Updates status='completed', sets ended_at timestamp,
        and optionally stores context summary in metadata.

        Args:
            session_id: Session to end
            context_summary: Optional session summary

        Returns:
            bool: True if session ended successfully

        Raises:
            aiosqlite.Error: If database operation fails
        """
        now = datetime.now().isoformat()

        # Get existing session to preserve metadata
        existing_session = await self.get_session(session_id)
        if existing_session is None:
            self.logger.warning(f"No session found to end: {session_id}")
            return False

        # Parse existing metadata and add context_summary
        import json
        metadata = {}
        if existing_session.metadata:
            try:
                metadata = json.loads(existing_session.metadata)
            except:
                pass

        if context_summary:
            metadata['context_summary'] = context_summary

        metadata_json = json.dumps(metadata) if metadata else None

        async with self._get_connection() as db:
            cursor = await db.execute(
                """
                UPDATE sessions
                SET status = ?,
                    ended_at = ?,
                    metadata = ?
                WHERE id = ?
                """,
                ('completed', now, metadata_json, session_id)
            )
            await db.commit()

            if cursor.rowcount == 0:
                self.logger.warning(f"No session found to end: {session_id}")
                return False

        self.logger.info(f"Ended work session: {session_id}")
        return True

    def bind_session_context(
        self,
        session_id: str,
        session_name: Optional[str] = None
    ) -> None:
        """
        Bind session context using structlog contextvars.

        Context7 Pattern: structlog.contextvars.bind_contextvars()
        Automatically propagates context to all subsequent log messages
        in the current thread/async context.

        This makes session_id available in ALL log messages without
        explicitly passing it to every log call.

        Args:
            session_id: Session ID to bind
            session_name: Optional session name

        Example:
            manager.bind_session_context("sess-123", "Phase 1")
            # All subsequent logs will include session_id="sess-123"
        """
        context_data = {"session_id": session_id}
        if session_name:
            context_data["session_name"] = session_name

        structlog.contextvars.bind_contextvars(**context_data)
        self.logger.debug(f"Bound session context: {session_id}")

    def clear_session_context(self) -> None:
        """
        Clear session context from structlog contextvars.

        Context7 Pattern: structlog.contextvars.clear_contextvars()
        Removes all bound context variables from current context.

        Should be called at session end to prevent context leakage
        into subsequent sessions.
        """
        structlog.contextvars.clear_contextvars()
        self.logger.debug("Cleared session context")


# Convenience functions for hook integration
async def create_or_resume_session(
    session_id: str,
    plan_id: Optional[str] = None,
    session_name: Optional[str] = None
) -> WorkSession:
    """
    Convenience function to create or resume session.

    Args:
        session_id: Session identifier
        plan_id: Optional plan ID
        session_name: Optional session name

    Returns:
        WorkSession: Created or resumed session
    """
    manager = WorkSessionManager()
    return await manager.resume_session(session_id)


if __name__ == "__main__":
    print("WorkSessionManager - DevStream Session Lifecycle Management")
    print("Context7-compliant implementation using aiosqlite + structlog")