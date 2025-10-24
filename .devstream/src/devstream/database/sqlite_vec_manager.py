"""
SQLite-vec Extension Manager
Context7-validated integration pattern from /asg017/sqlite-vec

Gestisce l'integrazione sqlite-vec con aiosqlite seguendo best practice validate.
"""

import sqlite3
import logging
from typing import Optional, Any

import structlog

try:
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

logger = structlog.get_logger()


class SQLiteVecManager:
    """
    Manager per sqlite-vec extension seguendo Context7 pattern.

    Pattern validato da /asg017/sqlite-vec documentation:
    - Direct connection loading
    - Enable/disable extension loading
    - Version verification
    """

    def __init__(self):
        self.is_available = SQLITE_VEC_AVAILABLE
        if not self.is_available:
            logger.warning("sqlite-vec not available - vector search disabled")

    def load_extension(self, connection: Any) -> bool:
        """
        Load sqlite-vec extension on connection.

        Context7 Pattern:
        ```python
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        ```

        Args:
            connection: SQLite connection (raw sqlite3 or aiosqlite wrapper)

        Returns:
            True if loaded successfully
        """
        if not self.is_available:
            logger.debug("sqlite-vec not available - skipping extension load")
            return False

        try:
            # Handle different connection types
            raw_conn = self._get_raw_connection(connection)

            # Context7 validated pattern
            raw_conn.enable_load_extension(True)
            sqlite_vec.load(raw_conn)
            raw_conn.enable_load_extension(False)

            # Verify loading
            version = self._get_vec_version(raw_conn)
            logger.info("sqlite-vec extension loaded", version=version)
            return True

        except Exception as e:
            logger.error("Failed to load sqlite-vec extension", error=str(e))
            return False

    def _get_raw_connection(self, connection: Any) -> sqlite3.Connection:
        """Extract raw sqlite3.Connection from various wrapper types."""

        # Direct sqlite3 connection
        if isinstance(connection, sqlite3.Connection):
            return connection

        # aiosqlite connection wrapper
        if hasattr(connection, '_conn'):
            return connection._conn

        # SQLAlchemy connection wrapper
        if hasattr(connection, 'connection'):
            return self._get_raw_connection(connection.connection)

        # Try getting underlying connection attribute
        for attr in ['_connection', '_sqlite_connection', 'driver_connection']:
            if hasattr(connection, attr):
                conn = getattr(connection, attr)
                if conn:
                    return self._get_raw_connection(conn)

        # Fallback - assume it's already raw
        return connection

    def _get_vec_version(self, connection: sqlite3.Connection) -> Optional[str]:
        """Get sqlite-vec version for verification."""
        try:
            cursor = connection.execute("SELECT vec_version()")
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception:
            return None

    def create_vec_table(self, connection: Any, table_name: str,
                        vector_column: str, dimension: int) -> bool:
        """
        Create virtual table per vector search.

        Context7 Pattern:
        ```sql
        CREATE VIRTUAL TABLE vec_table USING vec0(
            id TEXT PRIMARY KEY,
            embedding FLOAT[384]
        )
        ```
        """
        if not self.is_available:
            return False

        try:
            raw_conn = self._get_raw_connection(connection)

            sql = f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {table_name}
                USING vec0(
                    memory_id TEXT PRIMARY KEY,
                    {vector_column} FLOAT[{dimension}]
                )
            """

            raw_conn.execute(sql)
            logger.info("Created vec0 virtual table",
                       table=table_name, dimension=dimension)
            return True

        except Exception as e:
            logger.error("Failed to create vec0 table",
                        table=table_name, error=str(e))
            return False

    def create_fts_table(self, connection: Any, table_name: str,
                        content_columns: list[str]) -> bool:
        """Create FTS5 virtual table per keyword search."""
        try:
            raw_conn = self._get_raw_connection(connection)

            columns_spec = ", ".join([
                "memory_id UNINDEXED",
                *content_columns
            ])

            sql = f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {table_name}
                USING fts5(
                    {columns_spec},
                    tokenize=porter
                )
            """

            raw_conn.execute(sql)
            logger.info("Created FTS5 virtual table", table=table_name)
            return True

        except Exception as e:
            logger.error("Failed to create FTS5 table",
                        table=table_name, error=str(e))
            return False


# Global instance
vec_manager = SQLiteVecManager()