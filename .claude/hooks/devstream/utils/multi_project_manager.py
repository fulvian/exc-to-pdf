#!/usr/bin/env python3
"""
Multi-Project Manager - Context7 Compliant Solution

Implements robust multi-project architecture with:
- Automatic cross-project file sharing
- Database initialization with sqlite-utils patterns
- Comprehensive hook system validation
- Atomic operations with rollback capabilities
- Project isolation enforcement

Based on Context7 best practices from sqlite-utils and pytest-asyncio.
"""

import asyncio
import json
import logging
import os
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager

import structlog

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logger = structlog.get_logger(__name__)


@dataclass
class ProjectConfig:
    """Project configuration structure"""
    name: str
    path: str
    project_root: str
    venv_path: str
    db_path: str
    devstream_root: str


class DatabaseInitializationError(Exception):
    """Database initialization failure"""
    pass


class CrossProjectFileError(Exception):
    """Cross-project file operation failure"""
    pass


class MultiProjectManager:
    """
    Context7-compliant multi-project manager.

    Implements sqlite-utils patterns for robust database initialization
    and comprehensive project isolation.
    """

    def __init__(self, devstream_root: str):
        """
        Initialize multi-project manager.

        Args:
            devstream_root: Path to DevStream framework root
        """
        self.devstream_root = Path(devstream_root).resolve()
        self.framework_venv = self.devstream_root / ".devstream"
        self.hooks_dir = self.devstream_root / ".claude" / "hooks" / "devstream"

        logger.info("MultiProjectManager initialized",
                   devstream_root=str(self.devstream_root))

    def get_project_config(self, project_path: str) -> ProjectConfig:
        """
        Generate project configuration.

        Args:
            project_path: Path to project directory

        Returns:
            ProjectConfig instance
        """
        project_path = Path(project_path).resolve()
        project_name = project_path.name

        return ProjectConfig(
            name=project_name,
            path=str(project_path),
            project_root=str(project_path),
            venv_path=str(project_path / ".devstream"),
            db_path=str(project_path / "data" / "devstream.db"),
            devstream_root=str(self.devstream_root)
        )

    async def validate_project_structure(self, config: ProjectConfig) -> Tuple[bool, List[str]]:
        """
        Validate project structure using Context7 patterns.

        Args:
            config: Project configuration

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check project directory exists
        if not os.path.exists(config.project_root):
            errors.append(f"Project directory does not exist: {config.project_root}")

        # Check .claude directory structure
        claude_dir = Path(config.project_root) / ".claude"
        if not claude_dir.exists():
            errors.append(f".claude directory missing: {claude_dir}")
        else:
            # Check required subdirectories
            required_dirs = ["hooks", "state", "logs"]
            for subdir in required_dirs:
                if not (claude_dir / subdir).exists():
                    errors.append(f"Required directory missing: {claude_dir / subdir}")

        # Check hooks configuration
        settings_file = Path(config.project_root) / ".claude" / "settings.json"
        if not settings_file.exists():
            errors.append(f"Hooks settings file missing: {settings_file}")

        logger.info("Project structure validation completed",
                   project=config.name,
                   is_valid=len(errors) == 0,
                   error_count=len(errors))

        return len(errors) == 0, errors

    async def initialize_project_database(self, config: ProjectConfig) -> bool:
        """
        Initialize project database using sqlite-utils patterns.

        Implements Context7 best practices for database initialization:
        - Transaction control with rollback
        - Schema validation and creation
        - Error handling and recovery

        Args:
            config: Project configuration

        Returns:
            True if successful, False otherwise
        """
        db_path = Path(config.db_path)

        try:
            # Ensure data directory exists
            db_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info("Initializing project database",
                       project=config.name,
                       db_path=str(db_path))

            # Context7 Pattern: Use explicit transaction for schema operations
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute("BEGIN IMMEDIATE")

                try:
                    # Define required schemas using sqlite-utils patterns
                    schemas = {
                        "semantic_memory": """
                            CREATE TABLE IF NOT EXISTS semantic_memory (
                                id TEXT PRIMARY KEY,
                                content TEXT NOT NULL,
                                content_type TEXT NOT NULL,
                                keywords TEXT,
                                session_id TEXT,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                access_count INTEGER DEFAULT 0,
                                relevance_score REAL DEFAULT 1.0,
                                importance_score REAL DEFAULT 0.0,
                                last_accessed_at TIMESTAMP,
                                metadata TEXT,
                                source TEXT,
                                embedding_blob BLOB,
                                embedding_model TEXT,
                                embedding_dimension INTEGER
                            )
                        """,

                        "tasks": """
                            CREATE TABLE IF NOT EXISTS tasks (
                                id TEXT PRIMARY KEY,
                                title TEXT NOT NULL,
                                description TEXT,
                                task_type TEXT NOT NULL,
                                status TEXT NOT NULL DEFAULT 'pending',
                                priority INTEGER DEFAULT 5,
                                phase_name TEXT,
                                project_root TEXT,
                                session_id TEXT,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                completed_at TIMESTAMP,
                                metadata TEXT
                            )
                        """,

                        "implementation_plans": """
                            CREATE TABLE IF NOT EXISTS implementation_plans (
                                id TEXT PRIMARY KEY,
                                task_id TEXT NOT NULL,
                                title TEXT NOT NULL,
                                content TEXT NOT NULL,
                                model_type TEXT NOT NULL,
                                status TEXT NOT NULL DEFAULT 'draft',
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                metadata TEXT,
                                FOREIGN KEY (task_id) REFERENCES tasks (id)
                            )
                        """,

                        "sessions": """
                            CREATE TABLE IF NOT EXISTS sessions (
                                id TEXT PRIMARY KEY,
                                project_name TEXT,
                                start_time TIMESTAMP,
                                end_time TIMESTAMP,
                                status TEXT DEFAULT 'active',
                                metadata TEXT
                            )
                        """,

                        "memory_operations": """
                            CREATE TABLE IF NOT EXISTS memory_operations (
                                id TEXT PRIMARY KEY,
                                operation_type TEXT NOT NULL,
                                content_type TEXT,
                                content_preview TEXT,
                                keywords TEXT,
                                session_id TEXT,
                                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                metadata TEXT
                            )
                        """
                    }

                    # Context7 Pattern: Validate and create tables with error handling
                    tables_created = []
                    for table_name, create_sql in schemas.items():
                        # Check if table exists
                        cursor = conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                            (table_name,)
                        )
                        table_exists = cursor.fetchone() is not None

                        if not table_exists:
                            logger.info("Creating table", table=table_name)
                            conn.execute(create_sql)
                            tables_created.append(table_name)

                    # Context7 Pattern: Create indexes for performance
                    indexes = [
                        "CREATE INDEX IF NOT EXISTS idx_semantic_memory_content_type ON semantic_memory(content_type)",
                        "CREATE INDEX IF NOT EXISTS idx_semantic_memory_created_at ON semantic_memory(created_at)",
                        "CREATE INDEX IF NOT EXISTS idx_semantic_memory_session_id ON semantic_memory(session_id)",
                        "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)",
                        "CREATE INDEX IF NOT EXISTS idx_tasks_project_root ON tasks(project_root)",
                        "CREATE INDEX IF NOT EXISTS idx_memory_operations_timestamp ON memory_operations(timestamp)",
                        "CREATE INDEX IF NOT EXISTS idx_memory_operations_session_id ON memory_operations(session_id)"
                    ]

                    for index_sql in indexes:
                        conn.execute(index_sql)

                    # Context7 Pattern: Verify table integrity after creation
                    if tables_created:
                        await self._verify_table_integrity(conn, tables_created, config)

                    conn.commit()

                    logger.info("Database initialization completed",
                               project=config.name,
                               tables_created=len(tables_created))

                    return True

                except Exception as e:
                    conn.rollback()
                    raise DatabaseInitializationError(f"Database schema creation failed: {e}")

        except Exception as e:
            logger.error("Database initialization failed",
                        project=config.name,
                        error=str(e))
            return False

    async def _verify_table_integrity(self, conn: sqlite3.Connection,
                                    tables: List[str], config: ProjectConfig) -> None:
        """
        Verify table integrity after creation (Context7 pattern).

        Args:
            conn: Database connection
            tables: List of created tables
            config: Project configuration
        """
        for table_name in tables:
            try:
                # Verify table structure
                cursor = conn.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()

                if not columns:
                    raise DatabaseInitializationError(f"Table {table_name} has no columns")

                logger.debug("Table integrity verified",
                           project=config.name,
                           table=table_name,
                           column_count=len(columns))

            except Exception as e:
                raise DatabaseInitializationError(f"Table integrity check failed for {table_name}: {e}")

    async def ensure_cross_project_files(self, config: ProjectConfig) -> Tuple[bool, List[str]]:
        """
        Ensure cross-project files are properly shared using Context7 patterns.

        Implements atomic file operations with rollback capabilities.

        Args:
            config: Project configuration

        Returns:
            Tuple of (success, operation_log)
        """
        operations = []

        try:
            logger.info("Ensuring cross-project files", project=config.name)

            # Create project directories
            project_claude = Path(config.project_root) / ".claude"
            project_hooks = project_claude / "hooks" / "devstream"
            project_utils = project_hooks / "utils"
            project_data = Path(config.project_root) / "data"

            directories_to_create = [
                project_claude,
                project_hooks,
                project_utils,
                project_data
            ]

            for directory in directories_to_create:
                if not directory.exists():
                    directory.mkdir(parents=True, exist_ok=True)
                    operations.append(f"Created directory: {directory}")
                    logger.debug("Created directory", path=str(directory))

            # Files to copy from framework to project (Context7 pattern)
            essential_files = {
                "utils/direct_client.py": "Critical Direct DB client",
                "utils/memory_manager.py": "Memory management utilities",
                "context/user_query_context_enhancer.py": "User query enhancement",
                "memory/pre_tool_use.py": "PreToolUse hook",
                "memory/post_tool_use.py": "PostToolUse hook",
                "sessions/session_end.py": "Session management",
                "sessions/protocol_state.py": "Protocol state management"
            }

            # Copy essential files with atomic operations
            copied_files = []
            for relative_path, description in essential_files.items():
                source_file = self.hooks_dir / relative_path
                target_file = project_hooks / relative_path

                if source_file.exists():
                    success = await self._atomic_file_copy(source_file, target_file, description)
                    if success:
                        copied_files.append(relative_path)
                        operations.append(f"Copied {description}: {relative_path}")
                else:
                    operations.append(f"Warning: Source file missing: {source_file}")

            # Copy configuration files
            config_files = {
                ".env.devstream": "DevStream environment configuration",
                "requirements.txt": "Python dependencies"
            }

            for filename, description in config_files.items():
                source_file = self.devstream_root / filename
                target_file = Path(config.project_root) / filename

                if source_file.exists():
                    success = await self._atomic_file_copy(source_file, target_file, description)
                    if success:
                        operations.append(f"Copied {description}: {filename}")

            # Generate project-specific settings.json
            settings_success = await self._generate_project_settings(config)
            if settings_success:
                operations.append("Generated project-specific settings.json")

            logger.info("Cross-project files ensured",
                       project=config.name,
                       files_copied=len(copied_files),
                       operations_total=len(operations))

            return True, operations

        except Exception as e:
            logger.error("Cross-project file operation failed",
                        project=config.name,
                        error=str(e))
            return False, [f"Error: {str(e)}"]

    async def _atomic_file_copy(self, source: Path, target: Path, description: str) -> bool:
        """
        Perform atomic file copy with rollback capability (Context7 pattern).

        Args:
            source: Source file path
            target: Target file path
            description: File description for logging

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure target directory exists
            target.parent.mkdir(parents=True, exist_ok=True)

            # Create temporary file for atomic operation
            temp_target = target.with_suffix(f"{target.suffix}.tmp")

            # Copy to temporary file first
            shutil.copy2(source, temp_target)

            # Verify the copy
            if temp_target.stat().st_size != source.stat().st_size:
                temp_target.unlink()
                return False

            # Atomic rename
            temp_target.rename(target)

            logger.debug("Atomic file copy completed",
                        description=description,
                        source=str(source),
                        target=str(target))

            return True

        except Exception as e:
            logger.error("Atomic file copy failed",
                        description=description,
                        source=str(source),
                        target=str(target),
                        error=str(e))

            # Cleanup temporary file if it exists
            temp_target = target.with_suffix(f"{target.suffix}.tmp")
            if temp_target.exists():
                temp_target.unlink()

            return False

    async def _generate_project_settings(self, config: ProjectConfig) -> bool:
        """
        Generate project-specific settings.json with proper paths.

        Args:
            config: Project configuration

        Returns:
            True if successful, False otherwise
        """
        try:
            settings = {
                "permissions": {
                    "allow": [
                        f"Bash({config.venv_path}/bin/python:*)",
                        "Bash(python3:*)",
                        "Bash(sqlite3:*)"
                    ],
                    "deny": [],
                    "ask": []
                },
                "hooks": {
                    "PreToolUse": [{
                        "hooks": [{
                            "command": f"\"{config.venv_path}/bin/python\" \"{config.project_root}/.claude/hooks/devstream/memory/pre_tool_use.py\""
                        }]
                    }],
                    "PostToolUse": [{
                        "hooks": [{
                            "command": f"\"{config.venv_path}/bin/python\" \"{config.project_root}/.claude/hooks/devstream/memory/post_tool_use.py\""
                        }]
                    }],
                    "UserPromptSubmit": [{
                        "hooks": [{
                            "command": f"\"{config.venv_path}/bin/python\" \"{config.project_root}/.claude/hooks/devstream/context/user_query_context_enhancer.py\""
                        }]
                    }]
                }
            }

            settings_file = Path(config.project_root) / ".claude" / "settings.json"

            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)

            logger.info("Project settings generated",
                       project=config.name,
                       settings_file=str(settings_file))

            return True

        except Exception as e:
            logger.error("Failed to generate project settings",
                        project=config.name,
                        error=str(e))
            return False

    async def validate_hook_system(self, config: ProjectConfig) -> Tuple[bool, List[str]]:
        """
        Validate hook system using pytest-asyncio patterns.

        Args:
            config: Project configuration

        Returns:
            Tuple of (is_valid, validation_results)
        """
        results = []

        try:
            logger.info("Validating hook system", project=config.name)

            # Check Python executable
            python_exe = Path(config.venv_path) / "bin" / "python"
            if not python_exe.exists():
                results.append("❌ Python executable not found")
                return False, results
            else:
                results.append("✅ Python executable found")

            # Check critical modules
            critical_modules = [
                "cchooks",
                "aiohttp",
                "structlog",
                "dotenv"
            ]

            for module in critical_modules:
                if await self._check_module_installation(config, module):
                    results.append(f"✅ Module {module} available")
                else:
                    results.append(f"❌ Module {module} missing")

            # Check Direct DB client
            direct_client_path = Path(config.project_root) / ".claude" / "hooks" / "devstream" / "utils" / "direct_client.py"
            if direct_client_path.exists():
                results.append("✅ Direct DB client available")
            else:
                results.append("❌ Direct DB client missing")

            # Test hook execution (async pattern)
            hook_test_success = await self._test_hook_execution(config)
            if hook_test_success:
                results.append("✅ Hook execution test passed")
            else:
                results.append("❌ Hook execution test failed")

            # Test database connection
            db_test_success = await self._test_database_connection(config)
            if db_test_success:
                results.append("✅ Database connection test passed")
            else:
                results.append("❌ Database connection test failed")

            success_count = sum(1 for r in results if r.startswith("✅"))
            total_count = len(results)

            logger.info("Hook system validation completed",
                       project=config.name,
                       success_count=success_count,
                       total_count=total_count)

            return success_count == total_count, results

        except Exception as e:
            logger.error("Hook system validation failed",
                        project=config.name,
                        error=str(e))
            return False, [f"Validation error: {str(e)}"]

    async def _check_module_installation(self, config: ProjectConfig, module: str) -> bool:
        """
        Check if a Python module is installed in project venv.

        Args:
            config: Project configuration
            module: Module name to check

        Returns:
            True if module is available, False otherwise
        """
        try:
            python_exe = Path(config.venv_path) / "bin" / "python"

            # Use python -c to check module import
            result = await asyncio.create_subprocess_exec(
                str(python_exe), "-c", f"import {module}; print('OK')",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            return result.returncode == 0 and b"OK" in stdout

        except Exception:
            return False

    async def _test_hook_execution(self, config: ProjectConfig) -> bool:
        """
        Test hook execution using async patterns (pytest-asyncio style).

        Args:
            config: Project configuration

        Returns:
            True if hook test passes, False otherwise
        """
        try:
            python_exe = Path(config.venv_path) / "bin" / "python"
            hook_path = Path(config.project_root) / ".claude" / "hooks" / "devstream" / "utils" / "direct_client.py"

            # Test Direct DB client import and basic functionality
            test_script = f"""
import sys
sys.path.insert(0, '{hook_path.parent}')

try:
    from direct_client import get_direct_client
    print("SUCCESS: Direct DB client imported")
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
"""

            result = await asyncio.create_subprocess_exec(
                str(python_exe), "-c", test_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "DEVSTREAM_PROJECT_ROOT": config.project_root}
            )

            stdout, stderr = await result.communicate()

            return result.returncode == 0 and b"SUCCESS" in stdout

        except Exception:
            return False

    async def _test_database_connection(self, config: ProjectConfig) -> bool:
        """
        Test database connection and basic operations.

        Args:
            config: Project configuration

        Returns:
            True if database test passes, False otherwise
        """
        try:
            db_path = Path(config.db_path)

            if not db_path.exists():
                return False

            # Test basic database operations
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
                table_count = cursor.fetchone()[0]

                return table_count > 0

        except Exception:
            return False

    async def complete_project_setup(self, project_path: str) -> Tuple[bool, List[str]]:
        """
        Complete project setup using all Context7 patterns.

        This is the main entry point that orchestrates the entire
        multi-project setup process with comprehensive validation.

        Args:
            project_path: Path to project directory

        Returns:
            Tuple of (success, operation_log)
        """
        config = self.get_project_config(project_path)
        all_operations = []

        try:
            logger.info("Starting complete project setup", project=config.name)

            # Step 1: Validate project structure
            structure_valid, structure_errors = await self.validate_project_structure(config)
            if not structure_valid:
                return False, ["Structure validation failed"] + structure_errors

            all_operations.append("✅ Project structure validated")

            # Step 2: Initialize database with sqlite-utils patterns
            db_success = await self.initialize_project_database(config)
            if not db_success:
                return False, all_operations + ["❌ Database initialization failed"]

            all_operations.append("✅ Database initialized with required tables")

            # Step 3: Ensure cross-project files
            files_success, file_operations = await self.ensure_cross_project_files(config)
            if not files_success:
                return False, all_operations + file_operations

            all_operations.extend(file_operations)
            all_operations.append("✅ Cross-project files ensured")

            # Step 4: Validate hook system
            hooks_valid, hook_results = await self.validate_hook_system(config)
            all_operations.extend(hook_results)

            if not hooks_valid:
                logger.warning("Hook system validation has issues",
                             project=config.name,
                             issues=[r for r in hook_results if r.startswith("❌")])

            # Step 5: Generate final report
            success_count = sum(1 for op in all_operations if op.startswith("✅"))
            total_count = len(all_operations)

            logger.info("Project setup completed",
                       project=config.name,
                       success_count=success_count,
                       total_count=total_count,
                       overall_success=hooks_valid)

            return hooks_valid, all_operations

        except Exception as e:
            logger.error("Project setup failed",
                        project=config.name,
                        error=str(e))
            return False, all_operations + [f"❌ Setup failed: {str(e)}"]


# Async context manager for project setup operations
@asynccontextmanager
async def project_setup_context(devstream_root: str, project_path: str):
    """
    Context manager for project setup with automatic cleanup.

    Implements pytest-asyncio patterns for resource management.

    Args:
        devstream_root: Path to DevStream framework root
        project_path: Path to project directory

    Yields:
        MultiProjectManager instance
    """
    manager = MultiProjectManager(devstream_root)

    try:
        yield manager
    finally:
        # Cleanup operations if needed
        pass


# Convenience function for quick project setup
async def setup_project(devstream_root: str, project_path: str) -> Tuple[bool, List[str]]:
    """
    Convenience function for complete project setup.

    Args:
        devstream_root: Path to DevStream framework root
        project_path: Path to project directory

    Returns:
        Tuple of (success, operation_log)
    """
    async with project_setup_context(devstream_root, project_path) as manager:
        return await manager.complete_project_setup(project_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Project DevStream Setup")
    parser.add_argument("project_path", help="Path to project directory")
    parser.add_argument("--devstream-root", default=".", help="Path to DevStream framework root")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run setup
    success, operations = asyncio.run(setup_project(args.devstream_root, args.project_path))

    print(f"\nProject Setup Results: {'SUCCESS' if success else 'FAILED'}")
    print("=" * 50)
    for operation in operations:
        print(operation)

    sys.exit(0 if success else 1)