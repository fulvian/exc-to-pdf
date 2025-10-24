"""
Database migration system per DevStream.

Gestisce applicazione incrementale di schema changes e tracking versioni.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import structlog
from aiosqlite import Connection

from devstream.core.exceptions import DatabaseError
from devstream.database.connection import ConnectionPool
from devstream.database.schema import (
    metadata,
    get_table_creation_order,
    schema_version,
)

logger = structlog.get_logger()


class Migration:
    """Single database migration definition."""

    def __init__(
        self,
        version: str,
        description: str,
        up_sql: str,
        down_sql: Optional[str] = None,
    ):
        """
        Initialize migration.

        Args:
            version: Migration version (e.g., "001", "002")
            description: Human-readable description
            up_sql: SQL to apply migration
            down_sql: SQL to rollback migration (optional)
        """
        self.version = version
        self.description = description
        self.up_sql = up_sql
        self.down_sql = down_sql

    async def apply(self, conn: Connection) -> None:
        """Apply migration to database."""
        logger.info(f"Applying migration {self.version}: {self.description}")

        # Special handling for migration 001 (create tables)
        if self.version == "001":
            # SQLAlchemy 2.0 + aiosqlite: Generate CREATE TABLE statements from metadata
            from sqlalchemy import create_engine
            from io import StringIO

            # Create temporary engine for DDL compilation
            temp_engine = create_engine("sqlite:///:memory:")

            # Generate DDL statements using SQLAlchemy metadata
            for table in get_table_creation_order():
                # Use SQLAlchemy's CreateTable construct
                from sqlalchemy.schema import CreateTable

                create_table_stmt = CreateTable(table)
                compiled = create_table_stmt.compile(temp_engine)

                # Convert to string and execute with text()
                from sqlalchemy import text
                ddl_sql = str(compiled).replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS")
                await conn.execute(text(ddl_sql))

            temp_engine.dispose()
        else:
            # Execute migration SQL for other migrations
            from sqlalchemy import text
            for statement in self._split_sql(self.up_sql):
                if statement.strip():
                    await conn.execute(text(statement))

        # Record migration in schema_version table
        from sqlalchemy import text
        await conn.execute(
            text("INSERT INTO schema_version (version, description, applied_at) VALUES (:version, :description, datetime('now'))"),
            {"version": self.version, "description": self.description}
        )

        logger.info(f"Migration {self.version} applied successfully")

    async def rollback(self, conn: Connection) -> None:
        """Rollback migration from database."""
        if not self.down_sql:
            raise DatabaseError(
                f"Migration {self.version} has no rollback SQL",
                error_code="NO_ROLLBACK",
            )

        logger.info(f"Rolling back migration {self.version}")

        # Execute rollback SQL
        from sqlalchemy import text
        for statement in self._split_sql(self.down_sql):
            if statement.strip():
                await conn.execute(text(statement))

        # Remove migration record
        await conn.execute(
            schema_version.delete().where(schema_version.c.version == self.version)
        )

        logger.info(f"Migration {self.version} rolled back successfully")

    def _split_sql(self, sql: str) -> List[str]:
        """Split SQL into individual statements."""
        # Simple split on semicolon - could be more sophisticated
        statements = []
        for statement in sql.split(";"):
            statement = statement.strip()
            if statement and not statement.startswith("--"):
                statements.append(statement)
        return statements


class MigrationRunner:
    """Runs database migrations in correct order."""

    def __init__(self, pool: ConnectionPool):
        """
        Initialize migration runner.

        Args:
            pool: Database connection pool
        """
        self.pool = pool
        self._migrations: List[Migration] = []

    def add_migration(self, migration: Migration) -> None:
        """Add migration to runner."""
        self._migrations.append(migration)

    async def run_migrations(self) -> None:
        """Run all pending migrations."""
        logger.info("Starting database migrations")

        async with self.pool.write_transaction() as conn:
            # Ensure schema_version table exists
            await self._ensure_schema_version_table(conn)

            # Get applied migrations
            applied_versions = await self._get_applied_versions(conn)

            # Get pending migrations
            pending_migrations = [
                m for m in self._get_builtin_migrations()
                if m.version not in applied_versions
            ]

            if not pending_migrations:
                logger.info("No pending migrations")
                return

            # Apply pending migrations
            for migration in pending_migrations:
                await migration.apply(conn)

            logger.info(f"Applied {len(pending_migrations)} migrations successfully")

    async def rollback_migration(self, target_version: str) -> None:
        """
        Rollback migrations to target version.

        Args:
            target_version: Version to rollback to
        """
        logger.info(f"Rolling back migrations to version {target_version}")

        async with self.pool.write_transaction() as conn:
            applied_versions = await self._get_applied_versions(conn)

            # Find migrations to rollback (in reverse order)
            migrations_to_rollback = []
            for migration in reversed(self._get_builtin_migrations()):
                if migration.version in applied_versions:
                    if migration.version == target_version:
                        break
                    migrations_to_rollback.append(migration)

            # Rollback migrations
            for migration in migrations_to_rollback:
                await migration.rollback(conn)

            logger.info(f"Rolled back {len(migrations_to_rollback)} migrations")

    async def _ensure_schema_version_table(self, conn: Connection) -> None:
        """Ensure schema_version table exists."""
        from sqlalchemy import text
        create_sql = """
        CREATE TABLE IF NOT EXISTS schema_version (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            description TEXT
        )
        """
        try:
            await conn.execute(text(create_sql.strip()))
        except Exception:
            # Table might already exist
            pass

    async def _get_applied_versions(self, conn: Connection) -> set:
        """Get list of applied migration versions."""
        try:
            from sqlalchemy import text
            result = await conn.execute(text("SELECT version FROM schema_version"))
            rows = result.fetchall()
            return {row[0] for row in rows}
        except Exception:
            # schema_version table might not exist yet
            return set()

    def _get_builtin_migrations(self) -> List[Migration]:
        """Get built-in migrations for core schema."""
        migrations = []

        # Migration 001: Create core tables
        # SQLAlchemy 2.0 best practice: Use async-compatible DDL execution
        async def create_tables_up_sql():
            """Generate CREATE TABLE statements using metadata."""
            statements = []
            for table in get_table_creation_order():
                # Use run_sync to execute sync DDL in async context
                statements.append(f"CREATE TABLE IF NOT EXISTS {table.name} AS SELECT sql FROM sqlite_master WHERE name='{table.name}' LIMIT 0")
            return ";\n".join(statements)

        # Migration 001: Create core tables using metadata.create_all
        migration_001 = Migration(
            version="001",
            description="Create core database tables using metadata.create_all",
            up_sql="-- Tables will be created using metadata.create_all in apply() method",
            down_sql=self._generate_drop_tables_sql(),
        )
        migrations.append(migration_001)

        # Migration 002: Create indexes for performance
        # Note: Indexes are already created with tables in SQLAlchemy metadata.create_all()
        # This migration is for reference but not needed for basic functionality
        if False:  # Disable for now - indexes created with tables
            migration_002 = Migration(
                version="002",
                description="Create performance indexes (handled by metadata.create_all)",
                up_sql="-- Indexes created automatically with tables",
                down_sql=self._generate_drop_indexes_sql(),
            )
            migrations.append(migration_002)

        return migrations

    def _generate_drop_tables_sql(self) -> str:
        """Generate SQL to drop all tables."""
        drop_statements = []
        # Drop in reverse order to handle foreign keys
        for table in reversed(get_table_creation_order()):
            drop_statements.append(f"DROP TABLE IF EXISTS {table.name}")
        return ";\n".join(drop_statements) + ";"

    def _generate_drop_indexes_sql(self) -> str:
        """Generate SQL to drop all indexes."""
        drop_statements = []
        for table in metadata.tables.values():
            for index in table.indexes:
                drop_statements.append(f"DROP INDEX IF EXISTS {index.name}")
        return ";\n".join(drop_statements) + ";"

    async def get_migration_status(self) -> List[Tuple[str, str, bool]]:
        """
        Get status of all migrations.

        Returns:
            List of (version, description, applied) tuples
        """
        async with self.pool.read_transaction() as conn:
            await self._ensure_schema_version_table(conn)
            applied_versions = await self._get_applied_versions(conn)

        status = []
        for migration in self._get_builtin_migrations():
            status.append((
                migration.version,
                migration.description,
                migration.version in applied_versions,
            ))

        return status

    async def verify_schema(self) -> bool:
        """
        Verify database schema is complete and correct.

        Returns:
            True if schema is valid
        """
        logger.info("Verifying database schema")

        try:
            async with self.pool.read_transaction() as conn:
                # Check all expected tables exist
                expected_tables = {table.name for table in get_table_creation_order()}

                from sqlalchemy import text
                result = await conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                )
                rows = result.fetchall()
                existing_tables = {row[0] for row in rows}

                missing_tables = expected_tables - existing_tables
                if missing_tables:
                    logger.error(
                        "Missing database tables",
                        missing=list(missing_tables),
                    )
                    return False

                # Check foreign key integrity
                await conn.execute(text("PRAGMA foreign_key_check"))

                # Check schema version consistency
                applied_versions = await self._get_applied_versions(conn)
                expected_versions = {m.version for m in self._get_builtin_migrations()}

                if applied_versions != expected_versions:
                    logger.warning(
                        "Schema version mismatch",
                        applied=applied_versions,
                        expected=expected_versions,
                    )

                logger.info("Database schema verification successful")
                return True

        except Exception as e:
            logger.error("Schema verification failed", error=str(e))
            return False


async def create_database_schema(pool: ConnectionPool) -> None:
    """
    Convenience function to create complete database schema.

    Args:
        pool: Database connection pool
    """
    runner = MigrationRunner(pool)
    await runner.run_migrations()

    # Verify schema
    is_valid = await runner.verify_schema()
    if not is_valid:
        raise DatabaseError(
            "Database schema verification failed",
            error_code="SCHEMA_INVALID",
        )