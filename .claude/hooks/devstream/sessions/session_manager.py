"""
Simplified Session Manager using Context7 aiosqlite patterns.

Provides singleton session management with async context managers
for automatic cleanup and proper error handling.
"""

import asyncio
import json
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any

import aiosqlite
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Session:
    """Session data model with essential fields only."""
    id: str
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    status: str  # 'active' or 'completed'
    tokens_used: int = 0
    files_modified: int = 0
    tasks_completed: int = 0
    metadata: Optional[str] = None  # JSON string

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            'id': self.id,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'ended_at': self.ended_at.isoformat() if self.ended_at else None,
            'status': self.status,
            'tokens_used': self.tokens_used,
            'files_modified': self.files_modified,
            'tasks_completed': self.tasks_completed,
            'metadata': self.metadata
        }


class SessionException(Exception):
    """Session manager exception."""
    pass


class SessionManager:
    """Simplified session manager using Context7 aiosqlite patterns."""

    _instances: Dict[str, 'SessionManager'] = {}
    _lock = asyncio.Lock()

    def __init__(self, db_path: str) -> None:
        """Initialize session manager with database path.

        Args:
            db_path: Path to SQLite database

        Raises:
            ValueError: If db_path is empty
        """
        if not db_path:
            raise ValueError("Database path cannot be empty")

        self.db_path = db_path
        self._logger = structlog.get_logger(__name__).bind(db_path=db_path)

    @classmethod
    async def get_instance(cls, db_path: str) -> 'SessionManager':
        """Get singleton instance for database path.

        Args:
            db_path: Database path

        Returns:
            SessionManager instance
        """
        async with cls._lock:
            if db_path not in cls._instances:
                cls._instances[db_path] = cls(db_path)
                await cls._instances[db_path]._ensure_tables()
            return cls._instances[db_path]

    @asynccontextmanager
    async def _get_connection(self) -> Any:
        """Get database connection with proper error handling.

        Yields:
            aiosqlite.Connection: Database connection

        Raises:
            SessionException: If connection fails
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Enable foreign keys
                await db.execute("PRAGMA foreign_keys = ON")
                # Set row factory for dict-like access
                db.row_factory = aiosqlite.Row
                yield db
        except (aiosqlite.Error, sqlite3.Error) as e:
            # Don't wrap IntegrityError - let it bubble up to be handled by specific methods
            if isinstance(e, aiosqlite.IntegrityError):
                raise
            self._logger.error(
                "Database connection failed",
                extra={"error": str(e), "db_path": self.db_path}
            )
            raise SessionException(f"Failed to connect to database: {e}") from e

    async def _ensure_tables(self) -> None:
        """Ensure sessions table exists.

        Raises:
            SessionException: If table creation fails
        """
        try:
            async with self._get_connection() as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        ended_at TIMESTAMP NULL,
                        status TEXT NOT NULL CHECK (status IN ('active', 'completed')),
                        tokens_used INTEGER DEFAULT 0,
                        files_modified INTEGER DEFAULT 0,
                        tasks_completed INTEGER DEFAULT 0,
                        metadata TEXT
                    )
                """)

                # Create indexes for performance
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sessions_status
                    ON sessions(status)
                """)
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sessions_started_at
                    ON sessions(started_at DESC)
                """)

                await db.commit()
                self._logger.info("Sessions table ensured")

        except aiosqlite.Error as e:
            self._logger.error(
                "Table creation failed",
                extra={"error": str(e)}
            )
            raise SessionException(f"Failed to create sessions table: {e}") from e

    async def create_session(self, session_id: str) -> Session:
        """Create new session in database.

        Args:
            session_id: Unique session identifier

        Returns:
            Created session object

        Raises:
            SessionException: If session creation fails
            ValueError: If session_id is empty
        """
        if not session_id:
            raise ValueError("Session ID cannot be empty")

        try:
            async with self._get_connection() as db:
                await db.execute("""
                    INSERT INTO sessions (id, status, started_at)
                    VALUES (?, 'active', CURRENT_TIMESTAMP)
                """, (session_id,))
                await db.commit()

                # Retrieve created session
                session = await self.get_session(session_id)
                if session:
                    self._logger.info(
                        "Session created",
                        extra={"session_id": session_id}
                    )
                    return session
                else:
                    raise SessionException("Failed to retrieve created session")

        except aiosqlite.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                self._logger.error(
                    "Session already exists",
                    extra={"session_id": session_id, "error": str(e)}
                )
                raise SessionException(f"Session {session_id} already exists") from e
            else:
                self._logger.error(
                    "Integrity constraint violation",
                    extra={"session_id": session_id, "error": str(e)}
                )
                raise SessionException(f"Integrity error for session {session_id}: {e}") from e
        except aiosqlite.Error as e:
            self._logger.error(
                "Session creation failed",
                extra={"session_id": session_id, "error": str(e)}
            )
            raise SessionException(f"Failed to create session: {e}") from e

    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session object or None if not found
        """
        if not session_id:
            return None

        try:
            async with self._get_connection() as db:
                async with db.execute(
                    "SELECT * FROM sessions WHERE id = ?",
                    (session_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        return self._row_to_session(row)
                    return None

        except aiosqlite.Error as e:
            self._logger.error(
                "Session retrieval failed",
                extra={"session_id": session_id, "error": str(e)}
            )
            return None

    async def update_session(self, session_id: str, **kwargs: Any) -> bool:
        """Update session metadata.

        Args:
            session_id: Session identifier
            **kwargs: Fields to update (tokens_used, files_modified, tasks_completed, metadata)

        Returns:
            True if update successful, False otherwise
        """
        if not session_id or not kwargs:
            return False

        # Validate allowed fields
        allowed_fields = {
            'tokens_used', 'files_modified', 'tasks_completed', 'metadata'
        }
        invalid_fields = set(kwargs.keys()) - allowed_fields
        if invalid_fields:
            raise ValueError(f"Invalid fields: {invalid_fields}")

        try:
            # Build update query
            set_clauses = []
            params = []
            for field, value in kwargs.items():
                set_clauses.append(f"{field} = ?")
                if field == 'metadata' and value is not None:
                    # Convert metadata dict to JSON string
                    if isinstance(value, dict):
                        value = json.dumps(value)
                params.append(value)

            params.append(session_id)  # WHERE clause

            query = f"UPDATE sessions SET {', '.join(set_clauses)} WHERE id = ?"

            async with self._get_connection() as db:
                cursor = await db.execute(query, params)
                await db.commit()

                success = int(cursor.rowcount) > 0
                if success:
                    self._logger.debug(
                        "Session updated",
                        extra={"session_id": session_id, "fields": list(kwargs.keys())}
                    )
                return success

        except aiosqlite.Error as e:
            self._logger.error(
                "Session update failed",
                extra={"session_id": session_id, "fields": list(kwargs.keys()), "error": str(e)}
            )
            return False

    async def end_session(self, session_id: str) -> bool:
        """Mark session as completed.

        Args:
            session_id: Session identifier

        Returns:
            True if session ended successfully, False otherwise
        """
        if not session_id:
            return False

        try:
            async with self._get_connection() as db:
                cursor = await db.execute("""
                    UPDATE sessions
                    SET status = 'completed', ended_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND status = 'active'
                """, (session_id,))
                await db.commit()

                success = int(cursor.rowcount) > 0
                if success:
                    self._logger.info(
                        "Session ended",
                        extra={"session_id": session_id}
                    )
                return success

        except aiosqlite.Error as e:
            self._logger.error(
                "Session end failed",
                extra={"session_id": session_id, "error": str(e)}
            )
            return False

    async def list_active_sessions(self) -> List[Session]:
        """List all active sessions.

        Returns:
            List of active session objects
        """
        try:
            async with self._get_connection() as db:
                async with db.execute("""
                    SELECT * FROM sessions
                    WHERE status = 'active'
                    ORDER BY started_at DESC
                """) as cursor:
                    return [self._row_to_session(row) async for row in cursor]

        except aiosqlite.Error as e:
            self._logger.error(
                "Active sessions listing failed",
                extra={"error": str(e)}
            )
            return []

    async def cleanup_zombie_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up zombie sessions (active sessions older than max_age_hours).

        Args:
            max_age_hours: Maximum age in hours for active sessions

        Returns:
            Number of sessions cleaned up
        """
        try:
            async with self._get_connection() as db:
                cursor = await db.execute("""
                    UPDATE sessions
                    SET status = 'completed', ended_at = CURRENT_TIMESTAMP
                    WHERE status = 'active'
                    AND started_at < datetime('now', '-{} hours')
                """.format(max_age_hours))
                await db.commit()

                cleaned_count = int(cursor.rowcount)
                if cleaned_count > 0:
                    self._logger.info(
                        "Zombie sessions cleaned up",
                        extra={"count": cleaned_count, "max_age_hours": max_age_hours}
                    )
                return cleaned_count

        except aiosqlite.Error as e:
            self._logger.error(
                "Zombie session cleanup failed",
                extra={"error": str(e), "max_age_hours": max_age_hours}
            )
            return 0

    def _row_to_session(self, row: aiosqlite.Row) -> Session:
        """Convert database row to Session object.

        Args:
            row: Database row

        Returns:
            Session object
        """
        return Session(
            id=row['id'],
            started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
            ended_at=datetime.fromisoformat(row['ended_at']) if row['ended_at'] else None,
            status=row['status'],
            tokens_used=row['tokens_used'] or 0,
            files_modified=row['files_modified'] or 0,
            tasks_completed=row['tasks_completed'] or 0,
            metadata=row['metadata']
        )

    async def close(self) -> None:
        """Close session manager and cleanup resources."""
        # For now, just log closure
        self._logger.info("Session manager closed")