"""
Production-ready database migrations using Alembic + SQLAlchemy async patterns.

Context7-validated implementation based on research from:
- /sqlalchemy/alembic patterns for async migrations
- Production deployment best practices
- Transaction safety and rollback patterns
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, Optional, List

import structlog
from alembic import command, config, script
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.runtime import migration
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy import pool, text

from devstream.core.config import get_config
from devstream.core.exceptions import DatabaseError
from devstream.database.connection import ConnectionPool

logger = structlog.get_logger(__name__)


class AlembicMigrationManager:
    """
    Production-ready migration manager using Alembic with async SQLAlchemy.

    Context7-validated patterns:
    - Async migration execution with proper transaction handling
    - Production-safe rollback mechanisms
    - Environment-specific configuration
    - Comprehensive logging and error handling
    """

    def __init__(self, pool: ConnectionPool, environment: str = "production"):
        """
        Initialize Alembic migration manager.

        Args:
            pool: Database connection pool
            environment: Environment name (production, development, testing)
        """
        self.pool = pool
        self.environment = environment
        self.config_obj = self._create_alembic_config()

    def _create_alembic_config(self) -> Config:
        """Create Alembic configuration with Context7-validated patterns."""

        # Get project root and migrations directory
        project_root = Path(__file__).parent.parent.parent.parent
        migrations_dir = project_root / "deployment" / "migrations"
        migrations_dir.mkdir(parents=True, exist_ok=True)

        # Create alembic.ini if it doesn't exist
        alembic_ini = project_root / "alembic.ini"
        if not alembic_ini.exists():
            self._create_alembic_ini(alembic_ini, migrations_dir)

        # Load configuration
        alembic_cfg = Config(str(alembic_ini))

        # Set script location to our migrations directory
        alembic_cfg.set_main_option("script_location", str(migrations_dir))

        # Configure database URL for this environment
        app_config = get_config()
        if hasattr(app_config.database, 'async_url'):
            db_url = app_config.database.async_url
        else:
            # Convert sync SQLite path to async
            db_path = app_config.database.db_path
            db_url = f"sqlite+aiosqlite:///{db_path}"

        alembic_cfg.set_main_option("sqlalchemy.url", db_url)

        return alembic_cfg

    def _create_alembic_ini(self, ini_path: Path, migrations_dir: Path) -> None:
        """Create alembic.ini configuration file."""

        ini_content = f"""# Alembic configuration for DevStream
# Context7-validated production patterns

[alembic]
# Path to migration scripts
script_location = {migrations_dir}

# Template used to generate migration files
file_template = %%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d_%%(rev)s_%%(slug)s

# sys.path path, will be prepended to sys.path if present.
prepend_sys_path = .

# Timezone to use when rendering the date within the migration file
timezone = UTC

# Max length of characters to apply to the "slug" field
truncate_slug_length = 40

# Set to 'true' to run the environment during
# the 'revision' command, regardless of autogenerate
revision_environment = false

# Set to 'true' to allow .pyc and .pyo files without
# a source .py file to be detected as revisions in the
# versions/ directory
sourceless = false

# Version number format or a callable function returning a string
# version_num_format = %04d

# Version path separator; As mentioned above, this is the character used to split
# version_locations. The default within new alembic.ini files is "os", which uses
# os.pathsep. If this key is omitted entirely, it falls back to the legacy
# behavior of splitting on spaces and/or commas.
version_path_separator = :

# Set this to true to search source files recursively
# in each "version_locations" directory
recursive_version_locations = false

# The output encoding used when revision files
# are written from script.py.mako
output_encoding = utf-8

# Database URL - will be set programmatically
sqlalchemy.url =

[post_write_hooks]
# Post-write hooks define scripts or Python functions that are run
# on newly generated revision scripts.

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
"""
        ini_path.write_text(ini_content)
        logger.info("Created alembic.ini configuration", path=str(ini_path))

    async def initialize_migrations(self) -> None:
        """
        Initialize migration environment.

        Context7 pattern: Setup migration directory structure and env.py.
        """
        try:
            # Initialize Alembic environment
            command.init(self.config_obj, str(self.config_obj.get_main_option("script_location")))
            logger.info("Initialized Alembic migration environment")

            # Create custom env.py for async support
            await self._create_async_env_py()

        except Exception as e:
            if "already exists" not in str(e):
                raise DatabaseError(f"Failed to initialize migrations: {e}")
            logger.info("Migration environment already exists")

    async def _create_async_env_py(self) -> None:
        """Create env.py with Context7-validated async patterns."""

        migrations_dir = Path(self.config_obj.get_main_option("script_location"))
        env_py_path = migrations_dir / "env.py"

        env_py_content = '''"""
Alembic environment configuration for DevStream async migrations.

Context7-validated patterns for production async migrations.
"""

import asyncio
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Import your metadata
from devstream.database.schema import metadata

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add your model's MetaData object here for 'autogenerate' support
target_metadata = metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # For SQLite batch operations
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Configure context and run migrations."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,  # Important for SQLite
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in async mode.

    Context7 pattern: Use async engine with proper connection handling.
    """
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
'''

        env_py_path.write_text(env_py_content)
        logger.info("Created async env.py", path=str(env_py_path))

    async def create_migration(
        self,
        message: str,
        autogenerate: bool = True,
        sql_migration: Optional[str] = None
    ) -> str:
        """
        Create new migration revision.

        Args:
            message: Migration description
            autogenerate: Use autogenerate to detect changes
            sql_migration: Manual SQL if not using autogenerate

        Returns:
            Migration revision ID
        """
        try:
            if autogenerate:
                # Context7 pattern: Autogenerate with proper async handling
                revision = command.revision(
                    self.config_obj,
                    message=message,
                    autogenerate=True
                )
            else:
                # Manual migration
                revision = command.revision(
                    self.config_obj,
                    message=message
                )

                if sql_migration:
                    # Add custom SQL to the migration file
                    await self._add_sql_to_migration(revision.revision, sql_migration)

            logger.info(
                "Created migration",
                revision=revision.revision,
                message=message,
                autogenerate=autogenerate
            )
            return revision.revision

        except Exception as e:
            raise DatabaseError(f"Failed to create migration: {e}")

    async def upgrade(self, revision: str = "head") -> None:
        """
        Apply migrations up to specified revision.

        Args:
            revision: Target revision ('head', '+1', or specific revision)
        """
        logger.info("Starting database upgrade", target_revision=revision)

        try:
            # Context7 pattern: Programmatic command execution with async
            await self._run_alembic_command_async(
                lambda cfg: command.upgrade(cfg, revision)
            )
            logger.info("Database upgrade completed successfully")

        except Exception as e:
            logger.error("Database upgrade failed", error=str(e))
            raise DatabaseError(f"Migration upgrade failed: {e}")

    async def downgrade(self, revision: str) -> None:
        """
        Rollback migrations to specified revision.

        Args:
            revision: Target revision ('-1', 'base', or specific revision)
        """
        logger.warning("Starting database downgrade", target_revision=revision)

        try:
            await self._run_alembic_command_async(
                lambda cfg: command.downgrade(cfg, revision)
            )
            logger.info("Database downgrade completed")

        except Exception as e:
            logger.error("Database downgrade failed", error=str(e))
            raise DatabaseError(f"Migration downgrade failed: {e}")

    async def get_current_revision(self) -> Optional[str]:
        """Get current database revision."""
        try:
            async_engine = create_async_engine(
                self.config_obj.get_main_option("sqlalchemy.url")
            )

            async with async_engine.connect() as connection:
                def get_revision(conn):
                    context = MigrationContext.configure(conn)
                    return context.get_current_revision()

                revision = await connection.run_sync(get_revision)

            await async_engine.dispose()
            return revision

        except Exception as e:
            logger.error("Failed to get current revision", error=str(e))
            return None

    async def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get complete migration history."""
        try:
            script_dir = script.ScriptDirectory.from_config(self.config_obj)
            revisions = []

            current_rev = await self.get_current_revision()

            for revision in script_dir.walk_revisions():
                revisions.append({
                    "revision": revision.revision,
                    "description": revision.doc,
                    "branch_labels": revision.branch_labels,
                    "depends_on": revision.depends_on,
                    "is_current": revision.revision == current_rev,
                    "is_applied": revision.revision == current_rev or (
                        current_rev and script_dir.get_revision(current_rev)
                        and revision.revision in [
                            r.revision for r in script_dir.walk_revisions(
                                "base", current_rev
                            )
                        ]
                    )
                })

            return revisions

        except Exception as e:
            logger.error("Failed to get migration history", error=str(e))
            return []

    async def validate_migrations(self) -> bool:
        """
        Validate migration state and database consistency.

        Returns:
            True if migrations are consistent
        """
        try:
            logger.info("Validating migration state")

            # Check if database is up to date
            current_rev = await self.get_current_revision()
            if not current_rev:
                logger.warning("No current revision found")
                return False

            script_dir = script.ScriptDirectory.from_config(self.config_obj)
            head_rev = script_dir.get_current_head()

            if current_rev != head_rev:
                logger.warning(
                    "Database is not up to date",
                    current=current_rev,
                    head=head_rev
                )
                return False

            logger.info("Migration validation successful")
            return True

        except Exception as e:
            logger.error("Migration validation failed", error=str(e))
            return False

    async def _run_alembic_command_async(self, command_func) -> None:
        """
        Run Alembic command in async context.

        Context7 pattern: Execute sync Alembic commands from async context.
        """
        async_engine = create_async_engine(
            self.config_obj.get_main_option("sqlalchemy.url")
        )

        try:
            async with async_engine.begin() as conn:
                def run_command(connection):
                    # Set connection in config attributes for env.py
                    self.config_obj.attributes["connection"] = connection
                    command_func(self.config_obj)

                await conn.run_sync(run_command)

        finally:
            await async_engine.dispose()

    async def _add_sql_to_migration(self, revision: str, sql: str) -> None:
        """Add custom SQL to migration file."""
        migrations_dir = Path(self.config_obj.get_main_option("script_location"))
        versions_dir = migrations_dir / "versions"

        # Find migration file
        migration_file = None
        for file_path in versions_dir.glob("*.py"):
            if revision in file_path.name:
                migration_file = file_path
                break

        if not migration_file:
            raise DatabaseError(f"Migration file not found for revision {revision}")

        # Add SQL to upgrade function
        content = migration_file.read_text()
        upgrade_placeholder = "    pass"

        sql_operations = f'''    # Custom SQL migration
    op.execute("""{sql}""")'''

        content = content.replace(upgrade_placeholder, sql_operations)
        migration_file.write_text(content)

        logger.info("Added custom SQL to migration", revision=revision)


async def run_production_migrations() -> None:
    """
    Production entry point for running migrations.

    Context7-validated pattern for production deployment.
    """
    logger.info("Starting production migration process")

    try:
        # Get production configuration
        config_obj = get_config()

        # Create connection pool
        pool = ConnectionPool(
            db_path=config_obj.database.db_path,
            max_connections=config_obj.database.max_connections
        )
        await pool.initialize()

        try:
            # Create migration manager
            migration_manager = AlembicMigrationManager(pool, environment="production")

            # Initialize if needed
            await migration_manager.initialize_migrations()

            # Run migrations to head
            await migration_manager.upgrade("head")

            # Validate final state
            is_valid = await migration_manager.validate_migrations()
            if not is_valid:
                raise DatabaseError("Migration validation failed after upgrade")

            logger.info("Production migrations completed successfully")

        finally:
            await pool.close()

    except Exception as e:
        logger.error("Production migration failed", error=str(e))
        raise


# CLI entry point for production migrations
async def migrate_cli():
    """CLI entry point for database migrations."""
    import sys

    if len(sys.argv) > 1:
        command_name = sys.argv[1]

        if command_name == "upgrade":
            await run_production_migrations()
        elif command_name == "status":
            # Show migration status
            config_obj = get_config()
            pool = ConnectionPool(db_path=config_obj.database.db_path)
            await pool.initialize()

            try:
                manager = AlembicMigrationManager(pool)
                await manager.initialize_migrations()
                history = await manager.get_migration_history()

                print("Migration Status:")
                for rev in history:
                    status = "✓" if rev["is_applied"] else "✗"
                    current = " (current)" if rev["is_current"] else ""
                    print(f"{status} {rev['revision']}: {rev['description']}{current}")

            finally:
                await pool.close()
        else:
            print("Usage: python -m devstream.database.migrations_alembic [upgrade|status]")
    else:
        await run_production_migrations()


if __name__ == "__main__":
    asyncio.run(migrate_cli())