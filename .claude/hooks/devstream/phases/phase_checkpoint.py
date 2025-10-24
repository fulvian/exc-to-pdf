#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "aiohttp>=3.8.0",
#     "structlog>=23.0.0",
#     "python-dotenv>=1.0.0",
# ]
# ///

"""
Phase Checkpoint Hook - Protocol v2.2.0 Task 15

Automates checkpoint creation and git commit/push at phase completion.

Features:
1. Detect phase completion via TodoWrite monitoring (all tasks in phase = "completed")
2. Create checkpoint via CheckpointManager with metadata
3. Extract phase info (phase number, description, completed tasks)
4. git add . && git commit -m "PHASE X/Y - <description> COMPLETE"
5. Automatic git push origin <branch> (DEVSTREAM_PHASE_AUTO_PUSH=true)
6. Store phase completion metadata in DevStream memory
7. Log phase checkpoint audit trail

Configuration (.env.devstream):
DEVSTREAM_PHASE_CHECKPOINT_ENABLED=true
DEVSTREAM_PHASE_AUTO_PUSH=true
DEVSTREAM_PHASE_AUTO_COMMIT=true
DEVSTREAM_PHASE_COMMIT_PREFIX="PHASE"

Integration Points:
- PostToolUse hook triggers phase detection after TodoWrite updates
- CheckpointManager provides atomic savepoint functionality
- Git operations use subprocess with proper error handling
- Memory storage for phase completion audit trail

Architecture:
- Tier 1 Monolithic (no agent delegation)
- Graceful degradation if git operations fail
- Error handling + retry logic for auto-push
- Performance: <1s execution time
"""

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'checkpoints'))

from logger import get_devstream_logger
from mcp_client import get_mcp_client
from checkpoint_manager import CheckpointManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.devstream')

logger = get_devstream_logger(__name__)


class PhaseCheckpointManager:
    """
    Manages automated phase checkpoints with git commit/push.

    Detects phase completion from TodoWrite updates and automates:
    - Checkpoint creation
    - Git commit with descriptive message
    - Auto-push to remote (optional)
    - Memory storage for audit trail
    """

    def __init__(self):
        """Initialize phase checkpoint manager."""
        self.checkpoint_manager = CheckpointManager()
        self.memory_client = get_mcp_client()

        # Configuration from environment
        self.enabled = os.getenv('DEVSTREAM_PHASE_CHECKPOINT_ENABLED', 'false').lower() == 'true'
        self.auto_push = os.getenv('DEVSTREAM_PHASE_AUTO_PUSH', 'false').lower() == 'true'
        self.auto_commit = os.getenv('DEVSTREAM_PHASE_AUTO_COMMIT', 'true').lower() == 'true'
        self.commit_prefix = os.getenv('DEVSTREAM_PHASE_COMMIT_PREFIX', 'PHASE')

        # Statistics
        self.checkpoints_created = 0
        self.commits_created = 0
        self.pushes_executed = 0

        logger.logger.info(
            "phase_checkpoint_manager_initialized",
            enabled=self.enabled,
            auto_push=self.auto_push,
            auto_commit=self.auto_commit
        )

    async def detect_phase_completion(
        self,
        todowrite_tasks: List[Dict[str, str]]
    ) -> Optional[Dict[str, Any]]:
        """
        Detect if a phase has been completed from TodoWrite tasks.

        Args:
            todowrite_tasks: Current TodoWrite task list

        Returns:
            Phase info dict if phase complete, None otherwise

        Phase Detection Logic:
        - Group tasks by phase (inferred from task content)
        - Check if all tasks in a phase are "completed"
        - Return first completed phase found
        """
        if not self.enabled:
            return None

        if not todowrite_tasks:
            return None

        # Group tasks by phase (simple heuristic: "Task X" or "Implement Task X")
        phase_tasks = {}

        for task in todowrite_tasks:
            content = task.get("content", "")
            status = task.get("status", "pending")

            # Extract task number (e.g., "Task 13" or "Implement Task 13")
            task_number = self._extract_task_number(content)

            if task_number is not None:
                if task_number not in phase_tasks:
                    phase_tasks[task_number] = []

                phase_tasks[task_number].append({
                    "content": content,
                    "status": status
                })

        # Check for completed phases
        for task_num, tasks in phase_tasks.items():
            all_completed = all(t["status"] == "completed" for t in tasks)

            if all_completed:
                # Phase complete!
                return {
                    "task_number": task_num,
                    "description": tasks[0]["content"],  # Use first task as description
                    "total_tasks": len(tasks),
                    "completed_tasks": [t["content"] for t in tasks]
                }

        return None

    def _extract_task_number(self, content: str) -> Optional[int]:
        """
        Extract task number from task content.

        Args:
            content: Task content string

        Returns:
            Task number or None if not found

        Examples:
            "Task 13: E2E Test Workflow" → 13
            "Implement Task 15: Phase Checkpoint Hook" → 15
            "Complete workflow validation" → None
        """
        import re

        # Pattern: "Task X" or "Implement Task X" where X is a number
        match = re.search(r'[Tt]ask\s+(\d+)', content)

        if match:
            return int(match.group(1))

        return None

    async def create_phase_checkpoint(
        self,
        phase_info: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Create checkpoint for completed phase.

        Args:
            phase_info: Phase information from detect_phase_completion

        Returns:
            Tuple of (success, checkpoint_id)
        """
        try:
            # Get git commit hash
            git_commit = await self._get_git_commit_hash()

            # Get current branch
            git_branch = await self._get_git_branch()

            # Get modified files count
            files_modified = await self._count_modified_files()

            # Create checkpoint context
            context = {
                "phase_number": phase_info["task_number"],
                "phase_description": phase_info["description"],
                "completed_tasks": phase_info["completed_tasks"],
                "git_commit": git_commit,
                "git_branch": git_branch,
                "files_modified": files_modified,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            # Create checkpoint via CheckpointManager
            checkpoint = await self.checkpoint_manager.create_checkpoint(
                checkpoint_type="phase_completion",
                description=f"Phase {phase_info['task_number']} - {phase_info['description']}",
                context=context
            )

            self.checkpoints_created += 1

            logger.logger.info(
                "phase_checkpoint_created",
                checkpoint_id=checkpoint["id"],
                phase_number=phase_info["task_number"],
                tasks_completed=phase_info["total_tasks"]
            )

            return True, str(checkpoint["id"])

        except Exception as e:
            logger.logger.error(
                "phase_checkpoint_creation_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            return False, None

    async def execute_git_commit(
        self,
        phase_info: Dict[str, Any],
        checkpoint_id: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Execute git commit for phase completion.

        Args:
            phase_info: Phase information
            checkpoint_id: Checkpoint ID from create_phase_checkpoint

        Returns:
            Tuple of (success, commit_hash)
        """
        if not self.auto_commit:
            logger.logger.debug("auto_commit_disabled")
            return False, None

        try:
            # Stage all changes
            await self._run_git_command(['git', 'add', '.'])

            # Check if there are changes to commit
            status_output = await self._run_git_command(['git', 'status', '--porcelain'])

            if not status_output.strip():
                logger.logger.info("no_changes_to_commit")
                return False, None

            # Generate commit message
            commit_message = self._generate_commit_message(phase_info, checkpoint_id)

            # Create commit
            await self._run_git_command([
                'git', 'commit', '-m', commit_message
            ])

            # Get new commit hash
            commit_hash = await self._get_git_commit_hash()

            self.commits_created += 1

            logger.logger.info(
                "git_commit_created",
                commit_hash=commit_hash[:8],
                phase_number=phase_info["task_number"]
            )

            return True, commit_hash

        except subprocess.CalledProcessError as e:
            logger.logger.error(
                "git_commit_failed",
                error=str(e),
                stderr=e.stderr if hasattr(e, 'stderr') else None
            )
            return False, None

        except Exception as e:
            logger.logger.error(
                "git_commit_error",
                error=str(e),
                error_type=type(e).__name__
            )
            return False, None

    async def execute_git_push(
        self,
        phase_info: Dict[str, Any]
    ) -> bool:
        """
        Execute git push to remote.

        Args:
            phase_info: Phase information

        Returns:
            True if push successful, False otherwise
        """
        if not self.auto_push:
            logger.logger.debug("auto_push_disabled")
            return False

        try:
            # Get current branch
            branch = await self._get_git_branch()

            if not branch:
                logger.logger.warning("no_git_branch_detected")
                return False

            # Push to origin with retry logic
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    await self._run_git_command([
                        'git', 'push', 'origin', branch
                    ], timeout=30)

                    self.pushes_executed += 1

                    logger.logger.info(
                        "git_push_succeeded",
                        branch=branch,
                        phase_number=phase_info["task_number"],
                        attempt=attempt + 1
                    )

                    return True

                except subprocess.CalledProcessError as e:
                    if attempt < max_retries - 1:
                        logger.logger.warning(
                            "git_push_retry",
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            error=str(e)
                        )
                        await asyncio.sleep(2)  # Wait before retry
                    else:
                        raise

        except subprocess.CalledProcessError as e:
            logger.logger.error(
                "git_push_failed",
                error=str(e),
                stderr=e.stderr if hasattr(e, 'stderr') else None
            )
            return False

        except Exception as e:
            logger.logger.error(
                "git_push_error",
                error=str(e),
                error_type=type(e).__name__
            )
            return False

    async def store_phase_metadata(
        self,
        phase_info: Dict[str, Any],
        checkpoint_id: str,
        commit_hash: Optional[str],
        push_success: bool
    ) -> bool:
        """
        Store phase completion metadata in DevStream memory.

        Args:
            phase_info: Phase information
            checkpoint_id: Checkpoint ID
            commit_hash: Git commit hash (if commit created)
            push_success: Whether push succeeded

        Returns:
            True if storage successful, False otherwise
        """
        try:
            content = (
                f"Phase Checkpoint Created\n"
                f"Phase: Task {phase_info['task_number']}\n"
                f"Description: {phase_info['description']}\n"
                f"Tasks Completed: {phase_info['total_tasks']}\n"
                f"Checkpoint ID: {checkpoint_id}\n"
                f"Git Commit: {commit_hash[:8] if commit_hash else 'N/A'}\n"
                f"Auto-Push: {'SUCCESS' if push_success else 'SKIPPED/FAILED'}\n"
                f"Timestamp: {datetime.now(timezone.utc).isoformat()}"
            )

            keywords = [
                "phase-checkpoint",
                f"task-{phase_info['task_number']}",
                checkpoint_id,
                "automation"
            ]

            await self.memory_client.store_memory(
                content=content,
                content_type="decision",
                keywords=keywords
            )

            logger.logger.debug(
                "phase_metadata_stored",
                checkpoint_id=checkpoint_id,
                phase_number=phase_info["task_number"]
            )

            return True

        except Exception as e:
            logger.logger.error(
                "phase_metadata_storage_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            return False

    def _generate_commit_message(
        self,
        phase_info: Dict[str, Any],
        checkpoint_id: str
    ) -> str:
        """
        Generate git commit message for phase completion.

        Args:
            phase_info: Phase information
            checkpoint_id: Checkpoint ID

        Returns:
            Formatted commit message
        """
        task_list = "\n".join(f"- {task}" for task in phase_info["completed_tasks"])

        message = f"""{self.commit_prefix} {phase_info['task_number']} - {phase_info['description']} COMPLETE

Tasks Completed:
{task_list}

Checkpoint ID: {checkpoint_id}
Files Modified: {phase_info.get('files_modified', 'N/A')}

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
"""

        return message

    async def _get_git_commit_hash(self) -> str:
        """Get current git commit hash."""
        try:
            output = await self._run_git_command(['git', 'rev-parse', 'HEAD'])
            return output.strip()
        except Exception:
            return "unknown"

    async def _get_git_branch(self) -> Optional[str]:
        """Get current git branch."""
        try:
            output = await self._run_git_command(['git', 'branch', '--show-current'])
            return output.strip() if output.strip() else None
        except Exception:
            return None

    async def _count_modified_files(self) -> int:
        """Count modified files in git working directory."""
        try:
            output = await self._run_git_command(['git', 'status', '--porcelain'])
            lines = [line for line in output.strip().split('\n') if line.strip()]
            return len(lines)
        except Exception:
            return 0

    async def _run_git_command(
        self,
        command: List[str],
        timeout: int = 10
    ) -> str:
        """
        Run git command with timeout.

        Args:
            command: Git command as list
            timeout: Timeout in seconds

        Returns:
            Command stdout output

        Raises:
            subprocess.CalledProcessError: If command fails
        """
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.getcwd()
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            if process.returncode != 0:
                error = subprocess.CalledProcessError(
                    process.returncode,
                    command,
                    output=stdout,
                    stderr=stderr
                )
                error.stderr = stderr.decode('utf-8')
                raise error

            return stdout.decode('utf-8')

        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise subprocess.CalledProcessError(-1, command, output=b"", stderr=b"Timeout")

    def get_statistics(self) -> Dict[str, Any]:
        """Get phase checkpoint statistics."""
        return {
            "enabled": self.enabled,
            "auto_push": self.auto_push,
            "auto_commit": self.auto_commit,
            "checkpoints_created": self.checkpoints_created,
            "commits_created": self.commits_created,
            "pushes_executed": self.pushes_executed
        }


# Global instance
_global_manager: Optional[PhaseCheckpointManager] = None


def get_phase_checkpoint_manager() -> PhaseCheckpointManager:
    """Get global phase checkpoint manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = PhaseCheckpointManager()
    return _global_manager


async def handle_todowrite_update(todowrite_tasks: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Handle TodoWrite update and check for phase completion.

    Args:
        todowrite_tasks: Current TodoWrite task list

    Returns:
        Result dict with phase checkpoint info
    """
    manager = get_phase_checkpoint_manager()

    # Detect phase completion
    phase_info = await manager.detect_phase_completion(todowrite_tasks)

    if not phase_info:
        return {"phase_complete": False}

    logger.logger.info(
        "phase_completion_detected",
        task_number=phase_info["task_number"],
        total_tasks=phase_info["total_tasks"]
    )

    # Create checkpoint
    checkpoint_success, checkpoint_id = await manager.create_phase_checkpoint(phase_info)

    if not checkpoint_success:
        logger.logger.warning("checkpoint_creation_failed")
        return {"phase_complete": True, "checkpoint_created": False}

    # Execute git commit
    commit_success, commit_hash = await manager.execute_git_commit(phase_info, checkpoint_id)

    # Execute git push (if enabled)
    push_success = False
    if commit_success:
        push_success = await manager.execute_git_push(phase_info)

    # Store metadata in DevStream memory
    await manager.store_phase_metadata(
        phase_info,
        checkpoint_id,
        commit_hash,
        push_success
    )

    return {
        "phase_complete": True,
        "checkpoint_created": checkpoint_success,
        "checkpoint_id": checkpoint_id,
        "commit_created": commit_success,
        "commit_hash": commit_hash,
        "push_executed": push_success,
        "phase_info": phase_info
    }


# Example usage and testing
if __name__ == "__main__":
    async def test_phase_checkpoint():
        """Test phase checkpoint functionality."""

        # Mock TodoWrite tasks (Task 15 just completed)
        mock_tasks = [
            {"content": "Implement Task 13: E2E Test Workflow", "status": "completed", "activeForm": "..."},
            {"content": "Implement Task 14: Complete Workflow Validation", "status": "completed", "activeForm": "..."},
            {"content": "Implement Task 15: Phase Checkpoint Hook with auto-push", "status": "completed", "activeForm": "..."},
            {"content": "Implement Task 16: Agent Delegation Policy with token optimization", "status": "pending", "activeForm": "..."}
        ]

        print("🧪 Testing Phase Checkpoint Manager")
        print("=" * 70)

        # Test phase detection
        manager = get_phase_checkpoint_manager()
        phase_info = await manager.detect_phase_completion(mock_tasks)

        if phase_info:
            print(f"\n✅ Phase detected: Task {phase_info['task_number']}")
            print(f"   Tasks completed: {phase_info['total_tasks']}")
            print(f"   Description: {phase_info['description']}")
        else:
            print("\n❌ No phase completion detected")

        # Test statistics
        stats = manager.get_statistics()
        print(f"\n📊 Statistics:")
        print(f"   Enabled: {stats['enabled']}")
        print(f"   Auto-push: {stats['auto_push']}")
        print(f"   Checkpoints created: {stats['checkpoints_created']}")

        print("\n🎉 Phase checkpoint test complete")

    # Run test
    asyncio.run(test_phase_checkpoint())
