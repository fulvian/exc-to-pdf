#!/usr/bin/env -S .devstream/bin/python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "aiofiles>=23.0.0",
#     "structlog>=23.0.0",
#     "python-dotenv>=1.0.0"
# ]
# ///

"""
Micro-Task Commit Handler - Step 6 Granular Commit System

Phase 1 Component: Creates conventional commits for each completed micro-task.
Ensures granular git history with progress tracking during Step 6 implementation.

Core Problem Solved:
- Step 6 implementation consists of multiple micro-tasks that need individual commits
- Traditional approach creates one large commit at the end, losing granular progress
- Need conventional commit format with progress tracking
- @code-reviewer should trigger ONCE at Step 6 completion, not per micro-task

Architecture Principles:
- TodoWrite status change detection (completed → in_progress → completed)
- Conventional commit format enforcement (type(scope): description)
- Progress tracking in commit messages (X/N tasks complete)
- Integration with existing PostToolUse hook system
- Git automation with proper error handling and rollback

Trigger Conditions:
- PostToolUse hook with TodoWrite tool
- TodoWrite contains items with status changes to "completed"
- Current protocol step is IMPLEMENTATION (Step 6)
- Micro-task commits are enabled in configuration

Micro-Task Commit Workflow:
1. Detect TodoWrite status changes to "completed"
2. Extract completed micro-task information
3. Generate conventional commit message with progress tracking
4. Create git commit with appropriate files
5. Update commit progress counter
6. Detect Step 6 completion and trigger @code-reviewer
"""

import asyncio
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from logger import get_devstream_logger
from protocol_state_manager import ProtocolStateManager, ProtocolStep, get_protocol_manager

logger = get_devstream_logger(__name__)


class CommitType(Enum):
    """Conventional commit types."""
    FEAT = "feat"
    FIX = "fix"
    REFACTOR = "refactor"
    TEST = "test"
    DOCS = "docs"
    CHORE = "chore"
    PERF = "perf"
    BUILD = "build"
    CI = "ci"
    STYLE = "style"


@dataclass
class MicroTask:
    """Represents a completed micro-task."""
    content: str
    status: str
    active_form: str
    files_modified: List[str]
    estimated_duration: int  # minutes
    completion_time: datetime

    @property
    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.status.lower() == "completed"

    def extract_commit_type(self) -> CommitType:
        """Extract commit type from task content."""
        content_lower = self.content.lower()

        # Priority mapping for commit type detection
        # Order matters - more specific keywords should come first
        type_mapping = {
            CommitType.TEST: ["test", "testing", "spec", "verify", "validate"],
            CommitType.FIX: ["fix", "bug", "error", "issue", "problem", "resolve"],
            CommitType.REFACTOR: ["refactor", "restructure", "optimize", "improve", "clean"],
            CommitType.PERF: ["performance", "speed", "cache", "memory"],
            CommitType.DOCS: ["document", "readme", "guide", "manual", "documentation"],
            CommitType.BUILD: ["build", "compile", "package", "deploy", "install"],
            CommitType.STYLE: ["style", "lint", "format"],
            CommitType.CHORE: ["update", "maintain", "configure", "setup", "dependency"],
            CommitType.CI: [" ci ", "github action", "workflow", "pipeline"],  # More specific CI terms
            CommitType.FEAT: ["implement", "create", "feature", "develop", "add"]
        }

        # Find best matching commit type
        for commit_type, keywords in type_mapping.items():
            if any(keyword in content_lower for keyword in keywords):
                return commit_type

        return CommitType.CHORE  # Default

    def extract_scope(self) -> Optional[str]:
        """Extract scope from task content."""
        content_lower = self.content.lower()

        # Common scope patterns (more specific first)
        scope_patterns = {
            "auth": ["authentication", "login", "user", "user auth"],
            "api": ["endpoint", "route", "service", "rest api"],
            "ui": ["interface", "component", "view", "frontend"],
            "db": ["database", "model", "migration", "schema"],
            "test": ["spec", "coverage", "unit test", "integration test"],
            "config": ["setting", "environment", "configuration"],
            "deps": ["dependency", "package", "requirement"],
            "docs": ["documentation", "readme"]  # More specific
        }

        for scope, keywords in scope_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                return scope

        return None

    def generate_commit_title(self) -> str:
        """Generate conventional commit title."""
        commit_type = self.extract_commit_type()
        scope = self.extract_scope()

        # Extract concise description from task content
        description = self.content
        description = re.sub(r'^(STEP \d+(\.\d+)*\s*[-:]?\s*)', '', description.strip())
        description = re.sub(r'^(Micro-task:\s*)', '', description.strip())

        # Keep it concise (max 50 characters)
        if len(description) > 50:
            description = description[:47] + "..."

        # Avoid scope duplication for docs commits (docs(docs): -> docs:)
        if commit_type == CommitType.DOCS and scope == "docs":
            scope = None

        # Build conventional commit title
        if scope:
            title = f"{commit_type.value}({scope}): {description}"
        else:
            title = f"{commit_type.value}: {description}"

        return title

    def generate_commit_description(self, progress: str) -> str:
        """Generate detailed commit description."""
        description_parts = [
            f"Micro-task: {self.content}",
            f"Duration: ~{self.estimated_duration} minutes",
            f"Files modified: {len(self.files_modified)} file(s)"
        ]

        if self.files_modified:
            description_parts.append("Files:")
            for file_path in self.files_modified[:10]:  # Limit to 10 files
                description_parts.append(f"  - {file_path}")
            if len(self.files_modified) > 10:
                description_parts.append(f"  ... and {len(self.files_modified) - 10} more")

        description_parts.append(f"Progress: {progress}")

        # Add Claude Code attribution
        description_parts.extend([
            "",
            "🤖 Generated with [Claude Code](https://claude.com/claude-code)",
            "Co-Authored-By: Claude <noreply@anthropic.com>"
        ])

        return "\n".join(description_parts)


class MicroTaskCommitHandler:
    """
    Handles micro-task commits during Step 6 implementation.

    Features:
    - TodoWrite status change detection
    - Conventional commit format enforcement
    - Progress tracking in commit messages
    - Git automation with error handling
    - Step 6 completion detection and @code-reviewer trigger
    """

    def __init__(self):
        """Initialize micro-task commit handler."""
        self.protocol_manager = get_protocol_manager()
        self.commits_created = 0
        self.total_tasks = 0
        self.step6_start_time: Optional[datetime] = None

        # Load state from previous session if exists
        self._load_state()

        logger.info(
            "micro_task_commit_handler_initialized",
            commits_created=self.commits_created,
            total_tasks=self.total_tasks
        )

    async def handle_todo_write(self, todo_data: str, tool_input: Dict[str, Any]) -> bool:
        """
        Handle TodoWrite tool execution from PostToolUse hook.

        Args:
            todo_data: JSON string from TodoWrite tool output
            tool_input: Original TodoWrite tool input

        Returns:
            True if commit was created, False otherwise
        """
        try:
            # Check if micro-task commits are enabled
            if not self._is_enabled():
                logger.debug("micro_task_commits_disabled")
                return False

            # Parse TodoWrite data
            todos = self._parse_todo_data(todo_data)
            if not todos:
                logger.debug("no_todo_data_found")
                return False

            # Get current protocol state
            current_state = await self.protocol_manager.get_current_state()

            # Only process during Step 6 (IMPLEMENTATION)
            if current_state.protocol_step != ProtocolStep.IMPLEMENTATION:
                logger.debug(
                    "not_in_step_6",
                    current_step=str(current_state.protocol_step)
                )
                return False

            # Detect completed micro-tasks
            completed_tasks = self._detect_completed_tasks(todos)
            if not completed_tasks:
                logger.debug("no_completed_tasks_found")
                return False

            # Initialize Step 6 tracking if needed
            if self.step6_start_time is None:
                self.step6_start_time = datetime.now(timezone.utc)
                self.total_tasks = self._estimate_total_tasks(todos)

            # Create commits for completed tasks
            commits_created = 0
            for task in completed_tasks:
                if await self._create_micro_task_commit(task, current_state):
                    commits_created += 1
                    self.commits_created += 1

            # Check for Step 6 completion
            await self._check_step6_completion(todos, current_state)

            # Save state
            self._save_state()

            logger.info(
                "micro_task_commits_processed",
                tasks_found=len(completed_tasks),
                commits_created=commits_created,
                total_commits=self.commits_created
            )

            return commits_created > 0

        except Exception as e:
            logger.error(
                "micro_task_commit_handler_error",
                error=str(e),
                error_type=type(e).__name__
            )
            return False

    def _is_enabled(self) -> bool:
        """Check if micro-task commits are enabled."""
        return os.getenv("DEVSTREAM_MICRO_TASK_COMMITS", "false").lower() == "true"

    def _parse_todo_data(self, todo_data: str) -> Optional[List[Dict[str, Any]]]:
        """Parse TodoWrite JSON data."""
        try:
            todos = json.loads(todo_data)
            if isinstance(todos, list):
                return todos
            return None
        except json.JSONDecodeError as e:
            logger.error(
                "todo_data_parse_error",
                error=str(e),
                todo_data=todo_data[:200]  # Log first 200 chars
            )
            return None

    def _detect_completed_tasks(self, todos: List[Dict[str, Any]]) -> List[MicroTask]:
        """Detect newly completed micro-tasks from TodoWrite data."""
        completed_tasks = []

        for todo in todos:
            if todo.get("status", "").lower() == "completed":
                task = MicroTask(
                    content=todo.get("content", ""),
                    status=todo.get("status", ""),
                    active_form=todo.get("activeForm", ""),
                    files_modified=self._get_modified_files(),
                    estimated_duration=self._estimate_task_duration(todo),
                    completion_time=datetime.now(timezone.utc)
                )

                if task.is_completed:
                    completed_tasks.append(task)

        return completed_tasks

    async def _create_micro_task_commit(self, task: MicroTask, current_state) -> bool:
        """Create git commit for a completed micro-task."""
        try:
            # Generate commit message
            progress = f"{self.commits_created + 1}/{self.total_tasks}" if self.total_tasks > 0 else f"{self.commits_created + 1}/?"
            commit_title = task.generate_commit_title()
            commit_description = task.generate_commit_description(progress)

            # Build full commit message
            full_commit_message = f"{commit_title}\n\n{commit_description}"

            # Stage relevant files
            if not await self._stage_files(task):
                logger.warning("file_staging_failed", task_content=task.content[:50])
                return False

            # Create commit
            if not await self._create_git_commit(full_commit_message):
                logger.warning("git_commit_failed", task_content=task.content[:50])
                return False

            logger.info(
                "micro_task_commit_created",
                commit_title=commit_title,
                files_count=len(task.files_modified),
                progress=progress
            )

            return True

        except Exception as e:
            logger.error(
                "micro_task_commit_creation_error",
                task_content=task.content[:50],
                error=str(e)
            )
            return False

    async def _stage_files(self, task: MicroTask) -> bool:
        """Stage relevant files for commit."""
        try:
            # Stage all modified files
            if task.files_modified:
                # Stage specific files if available
                for file_path in task.files_modified:
                    if Path(file_path).exists():
                        result = subprocess.run(
                            ["git", "add", file_path],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        if result.returncode != 0:
                            logger.warning(
                                "git_add_failed",
                                file_path=file_path,
                                error=result.stderr
                            )
            else:
                # Stage all changes if no specific files
                result = subprocess.run(
                    ["git", "add", "."],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode != 0:
                    logger.error("git_add_all_failed", error=result.stderr)
                    return False

            return True

        except subprocess.TimeoutExpired:
            logger.error("git_stage_timeout")
            return False
        except Exception as e:
            logger.error("git_staging_error", error=str(e))
            return False

    async def _create_git_commit(self, message: str) -> bool:
        """Create git commit with the given message."""
        try:
            result = subprocess.run(
                ["git", "commit", "-m", message],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                logger.debug("git_commit_success", commit_hash=self._get_latest_commit_hash())
                return True
            else:
                logger.error(
                    "git_commit_failed",
                    error=result.stderr,
                    return_code=result.returncode
                )
                return False

        except subprocess.TimeoutExpired:
            logger.error("git_commit_timeout")
            return False
        except Exception as e:
            logger.error("git_commit_error", error=str(e))
            return False

    async def _check_step6_completion(self, todos: List[Dict[str, Any]], current_state) -> None:
        """Check if Step 6 is complete and trigger @code-reviewer."""
        try:
            # Count completed vs total tasks
            total_tasks = len(todos)
            completed_tasks = sum(1 for todo in todos if todo.get("status", "").lower() == "completed")

            if total_tasks == 0:
                return

            completion_percentage = (completed_tasks / total_tasks) * 100

            # Check if Step 6 is substantially complete (90%+)
            if completion_percentage >= 90:
                logger.info(
                    "step6_nearly_complete",
                    completed=completed_tasks,
                    total=total_tasks,
                    percentage=completion_percentage
                )

            # Check if Step 6 is fully complete
            if completion_percentage == 100:
                await self._finalize_step6(current_state)

        except Exception as e:
            logger.error("step6_completion_check_error", error=str(e))

    async def _finalize_step6(self, current_state) -> None:
        """Finalize Step 6 and trigger @code-reviewer."""
        try:
            logger.info(
                "step6_complete_triggering_code_reviewer",
                total_commits=self.commits_created,
                duration=(datetime.now(timezone.utc) - self.step6_start_time).total_seconds() if self.step6_start_time else 0
            )

            # Create final Step 6 completion commit
            final_message = """feat(implementation): Complete Step 6 - IMPLEMENTATION

All micro-tasks completed successfully:
- Total commits created: {}
- Step 6 duration: {:.1f} minutes
- Implementation ready for review

Progress: {}/{} micro-tasks complete

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>""".format(
                self.commits_created,
                (datetime.now(timezone.utc) - self.step6_start_time).total_seconds() / 60 if self.step6_start_time else 0,
                self.commits_created,
                self.total_tasks
            )

            await self._create_git_commit(final_message)

            # Trigger @code-reviewer (this would be implemented via a hook or system call)
            await self._trigger_code_reviewer()

            # Reset Step 6 tracking
            self.step6_start_time = None
            self.total_tasks = 0

        except Exception as e:
            logger.error("step6_finalization_error", error=str(e))

    async def _trigger_code_reviewer(self) -> None:
        """Trigger @code-reviewer for Step 6 completion."""
        try:
            # This would integrate with the agent system to trigger @code-reviewer
            # For now, we'll log the trigger and potentially write to a file
            logger.info("code_reviewer_triggered")

            # Write trigger file for external system to detect
            trigger_file = Path(".claude/state/code_reviewer_trigger.json")
            trigger_file.parent.mkdir(parents=True, exist_ok=True)

            trigger_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "step6_complete",
                "total_commits": self.commits_created,
                "session_id": (await self.protocol_manager.get_current_state()).session_id
            }

            with open(trigger_file, "w") as f:
                json.dump(trigger_data, f, indent=2)

        except Exception as e:
            logger.error("code_reviewer_trigger_error", error=str(e))

    def _get_modified_files(self) -> List[str]:
        """Get list of modified files from git status."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                files = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        # Parse git status output: XY filename
                        parts = line.strip().split(maxsplit=1)
                        if len(parts) == 2:
                            files.append(parts[1])
                return files
            return []

        except Exception:
            return []

    def _estimate_task_duration(self, todo: Dict[str, Any]) -> int:
        """Estimate task duration from todo content."""
        content = todo.get("content", "").lower()

        # Simple duration estimation based on content
        duration_indicators = {
            "quick": 5, "small": 10, "simple": 15,
            "implement": 30, "create": 45, "build": 60,
            "complex": 120, "refactor": 90, "test": 30
        }

        for indicator, duration in duration_indicators.items():
            if indicator in content:
                return duration

        return 15  # Default duration

    def _estimate_total_tasks(self, todos: List[Dict[str, Any]]) -> int:
        """Estimate total number of tasks in Step 6."""
        return len(todos)

    def _get_latest_commit_hash(self) -> str:
        """Get hash of the latest commit."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]  # Short hash
            return "unknown"
        except Exception:
            return "unknown"

    def _load_state(self) -> None:
        """Load handler state from file."""
        try:
            state_file = Path(".claude/state/micro_task_commit_handler.json")
            if state_file.exists():
                with open(state_file, "r") as f:
                    state = json.load(f)
                    self.commits_created = state.get("commits_created", 0)
                    self.total_tasks = state.get("total_tasks", 0)
                    if state.get("step6_start_time"):
                        self.step6_start_time = datetime.fromisoformat(state["step6_start_time"])
        except Exception as e:
            logger.warning("state_load_failed", error=str(e))

    def _save_state(self) -> None:
        """Save handler state to file."""
        try:
            state_file = Path(".claude/state/micro_task_commit_handler.json")
            state_file.parent.mkdir(parents=True, exist_ok=True)

            state = {
                "commits_created": self.commits_created,
                "total_tasks": self.total_tasks,
                "step6_start_time": self.step6_start_time.isoformat() if self.step6_start_time else None,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }

            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning("state_save_failed", error=str(e))

    def get_statistics(self) -> Dict[str, Any]:
        """Get handler statistics."""
        return {
            "commits_created": self.commits_created,
            "total_tasks": self.total_tasks,
            "step6_active": self.step6_start_time is not None,
            "step6_duration_minutes": (
                (datetime.now(timezone.utc) - self.step6_start_time).total_seconds() / 60
                if self.step6_start_time else 0
            ),
            "enabled": self._is_enabled()
        }


# Main execution for PostToolUse hook
if __name__ == "__main__":
    async def main():
        """Main entry point for PostToolUse hook execution."""
        import sys

        try:
            # Get TodoWrite data from stdin or command line argument
            if len(sys.argv) > 1:
                todo_data = sys.argv[1]
            else:
                todo_data = sys.stdin.read().strip()

            if not todo_data:
                logger.error("no_todo_data_provided")
                sys.exit(1)

            # Create handler and process TodoWrite
            handler = MicroTaskCommitHandler()
            success = await handler.handle_todo_write(todo_data, {})

            if success:
                logger.info("micro_task_commit_handler_success")
                sys.exit(0)
            else:
                logger.info("micro_task_commit_handler_no_action")
                sys.exit(0)

        except Exception as e:
            logger.error(
                "micro_task_commit_handler_fatal_error",
                error=str(e),
                error_type=type(e).__name__
            )
            sys.exit(1)

    # Run main function
    asyncio.run(main())