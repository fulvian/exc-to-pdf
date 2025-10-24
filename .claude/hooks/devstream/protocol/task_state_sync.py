#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "aiofiles>=23.0.0",
#     "structlog>=23.0.0",
#     "python-dotenv>=1.0.0",
#     "cachetools>=5.0.0",
# ]
# ///

"""
Task State Synchronization - Automatic State Tracking for DevStream Protocol

Phase 1 Component: Automatic state synchronization with PostToolUse hook.
Tracks progress and maintains atomic updates with crash recovery.

Architecture Principles:
- Automatic state synchronization via PostToolUse hook integration
- Progress tracking with atomic state updates
- Crash recovery with state restoration capabilities
- Integration with existing DevStream memory and task systems
- Performance optimization with caching and debouncing

Synchronization Points:
1. PostToolUse Hook: Automatic state updates after tool execution
2. Task Progress: Track TodoWrite changes and task completions
3. Memory Integration: Log progress and decisions to DevStream memory
4. Session Management: Handle session start/end and crash recovery
5. Validation Triggers: Trigger step validation when appropriate

State Tracking Model:
- Tool usage patterns (Write, Edit, Read, Bash, etc.)
- Task progress indicators (TodoWrite updates, task completions)
- Context7 usage (research and documentation retrieval)
- Memory storage (decisions, code, learning, context)
- Step completion validation

Performance Optimizations:
- Debounced state updates to reduce I/O
- In-memory caching with TTL
- Batch operations for multiple updates
- Asynchronous processing for non-blocking behavior
"""

import asyncio
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Callable
from dataclasses import dataclass, asdict
import cachetools

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from logger import get_devstream_logger
from protocol_state_manager import ProtocolStateManager, ProtocolStep, get_protocol_manager
from step_validator import StepValidator
from mcp_client import get_mcp_client
from debouncer import AsyncDebouncer

logger = get_devstream_logger(__name__)


class SyncTrigger(Enum):
    """Types of synchronization triggers."""
    TOOL_EXECUTION = "tool_execution"
    TASK_UPDATE = "task_update"
    MEMORY_STORAGE = "memory_storage"
    STEP_VALIDATION = "step_validation"
    SESSION_EVENT = "session_event"
    MANUAL_SYNC = "manual_sync"


@dataclass
class SyncEvent:
    """Synchronization event data."""
    trigger: SyncTrigger
    timestamp: str
    session_id: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProgressMetrics:
    """Progress tracking metrics."""
    tools_used: Dict[str, int]  # Tool usage counts
    files_modified: int
    lines_written: int
    memory_entries: int
    todo_updates: int
    context7_queries: int
    duration_seconds: float
    last_activity: str


class TaskStateSync:
    """
    Automatic state synchronization for DevStream protocol tracking.

    Integrates with PostToolUse hook to automatically track progress,
    update protocol state, and maintain synchronization across crashes.
    """

    def __init__(self, memory_client=None, protocol_manager=None):
        """
        Initialize task state synchronization.

        Args:
            memory_client: Optional MCP memory client
            protocol_manager: Optional protocol state manager
        """
        self.memory_client = memory_client or get_mcp_client()
        self.protocol_manager = protocol_manager or get_protocol_manager()
        self.step_validator = StepValidator()

        # Debouncer for state updates (500ms delay)
        self.state_debouncer = AsyncDebouncer(delay=0.5)

        # In-memory caches
        self.session_metrics: Dict[str, ProgressMetrics] = {}
        self.active_sessions: Set[str] = set()
        self.sync_history: List[SyncEvent] = []

        # Performance metrics
        self.sync_count = 0
        self.last_sync_time = None

        logger.logger.info(
            "task_state_sync_initialized",
            memory_client_available=self.memory_client is not None,
            debouncer_delay_ms=500
        )

    async def synchronize_on_tool_execution(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        execution_result: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> bool:
        """
        Synchronize state after tool execution (PostToolUse hook integration).

        Args:
            tool_name: Name of the executed tool
            tool_input: Tool input parameters
            execution_result: Tool execution result
            session_id: Optional session ID

        Returns:
            True if synchronization successful, False otherwise
        """
        try:
            # Get or initialize session
            if session_id is None:
                current_state = await self.protocol_manager.get_current_state()
                session_id = current_state.session_id

            # Ensure session is tracked
            if session_id not in self.session_metrics:
                await self._initialize_session_metrics(session_id)

            # Update progress metrics
            await self._update_tool_metrics(session_id, tool_name, tool_input, execution_result)

            # Determine if step advancement is needed
            await self._evaluate_step_advancement(session_id, tool_name, tool_input)

            # Debounced state update
            await self.state_debouncer.debounce(
                self._perform_state_sync,
                session_id,
                SyncTrigger.TOOL_EXECUTION,
                {
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "execution_result": execution_result
                }
            )

            self.sync_count += 1
            self.last_sync_time = datetime.now(timezone.utc).isoformat()

            logger.logger.debug(
                "tool_execution_sync_completed",
                tool_name=tool_name,
                session_id=session_id,
                sync_count=self.sync_count
            )

            return True

        except Exception as e:
            logger.logger.error(
                "tool_execution_sync_error",
                tool_name=tool_name,
                session_id=session_id,
                error=str(e),
                error_type=type(e).__name__
            )
            return False

    async def synchronize_on_task_update(
        self,
        task_id: str,
        update_type: str,
        update_data: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> bool:
        """
        Synchronize state on task updates.

        Args:
            task_id: DevStream task ID
            update_type: Type of update (created, updated, completed)
            update_data: Update data
            session_id: Optional session ID

        Returns:
            True if synchronization successful, False otherwise
        """
        try:
            if session_id is None:
                current_state = await self.protocol_manager.get_current_state()
                session_id = current_state.session_id

            # Update task metrics
            metrics = self.session_metrics.get(session_id)
            if metrics:
                if update_type == "todo_update":
                    metrics.todo_updates += 1
                elif update_type == "task_completed":
                    # Trigger step validation for implementation completion
                    await self._validate_implementation_completion(session_id, task_id)

            # Sync state immediately for task updates
            await self._perform_state_sync(
                session_id,
                SyncTrigger.TASK_UPDATE,
                {
                    "task_id": task_id,
                    "update_type": update_type,
                    "update_data": update_data
                }
            )

            logger.logger.debug(
                "task_update_sync_completed",
                task_id=task_id,
                update_type=update_type,
                session_id=session_id
            )

            return True

        except Exception as e:
            logger.logger.error(
                "task_update_sync_error",
                task_id=task_id,
                update_type=update_type,
                error=str(e)
            )
            return False

    async def synchronize_on_memory_storage(
        self,
        content: str,
        content_type: str,
        keywords: List[str],
        session_id: Optional[str] = None
    ) -> bool:
        """
        Synchronize state on memory storage operations.

        Args:
            content: Stored content
            content_type: Type of content stored
            keywords: Content keywords
            session_id: Optional session ID

        Returns:
            True if synchronization successful, False otherwise
        """
        try:
            if session_id is None:
                current_state = await self.protocol_manager.get_current_state()
                session_id = current_state.session_id

            # Update memory metrics
            metrics = self.session_metrics.get(session_id)
            if metrics:
                metrics.memory_entries += 1

                # Check for Context7 usage
                if "context7" in content.lower() or content_type == "context":
                    metrics.context7_queries += 1

            # Sync state with debouncing for memory operations
            await self.state_debouncer.debounce(
                self._perform_state_sync,
                session_id,
                SyncTrigger.MEMORY_STORAGE,
                {
                    "content_type": content_type,
                    "keywords": keywords,
                    "content_length": len(content)
                }
            )

            logger.logger.debug(
                "memory_storage_sync_completed",
                content_type=content_type,
                session_id=session_id,
                keywords_count=len(keywords)
            )

            return True

        except Exception as e:
            logger.logger.error(
                "memory_storage_sync_error",
                content_type=content_type,
                error=str(e)
            )
            return False

    async def perform_crash_recovery(self, session_id: Optional[str] = None) -> bool:
        """
        Perform crash recovery by restoring state from disk.

        Args:
            session_id: Optional session ID to recover

        Returns:
            True if recovery successful, False otherwise
        """
        try:
            logger.logger.info(
                "crash_recovery_started",
                session_id=session_id or "auto-detect"
            )

            # Load current state from disk
            current_state = await self.protocol_manager.get_current_state()

            if session_id and session_id != current_state.session_id:
                # Different session requested, initialize new one
                recovered_state = await self.protocol_manager.initialize_session()
            else:
                recovered_state = current_state

            # Restore session metrics
            await self._restore_session_metrics(recovered_state.session_id)

            # Log recovery to memory
            await self._log_crash_recovery(recovered_state.session_id)

            logger.logger.info(
                "crash_recovery_completed",
                session_id=recovered_state.session_id,
                current_step=str(recovered_state.protocol_step)
            )

            return True

        except Exception as e:
            logger.logger.error(
                "crash_recovery_failed",
                session_id=session_id,
                error=str(e),
                error_type=type(e).__name__
            )
            return False

    async def _initialize_session_metrics(self, session_id: str) -> None:
        """Initialize metrics for a new session."""
        self.session_metrics[session_id] = ProgressMetrics(
            tools_used={},
            files_modified=0,
            lines_written=0,
            memory_entries=0,
            todo_updates=0,
            context7_queries=0,
            duration_seconds=0.0,
            last_activity=datetime.now(timezone.utc).isoformat()
        )
        self.active_sessions.add(session_id)

        logger.logger.debug(
            "session_metrics_initialized",
            session_id=session_id
        )

    async def _update_tool_metrics(
        self,
        session_id: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        execution_result: Dict[str, Any]
    ) -> None:
        """Update metrics based on tool execution."""
        metrics = self.session_metrics.get(session_id)
        if not metrics:
            await self._initialize_session_metrics(session_id)
            metrics = self.session_metrics[session_id]

        # Update tool usage count
        metrics.tools_used[tool_name] = metrics.tools_used.get(tool_name, 0) + 1

        # Update specific metrics based on tool type
        if tool_name in ["Write", "Edit", "MultiEdit"]:
            metrics.files_modified += 1
            # Estimate lines written from content
            content = tool_input.get("content", "") or tool_input.get("new_string", "")
            metrics.lines_written += len(content.splitlines())

        metrics.last_activity = datetime.now(timezone.utc).isoformat()

    async def _evaluate_step_advancement(
        self,
        session_id: str,
        tool_name: str,
        tool_input: Dict[str, Any]
    ) -> None:
        """Evaluate if protocol step should be advanced."""
        try:
            current_state = await self.protocol_manager.get_current_state()

            # Only advance if we're in a valid state
            if current_state.session_id != session_id:
                return

            # Step-specific advancement logic
            if current_state.protocol_step == ProtocolStep.DISCUSSION:
                # Check if discussion is happening (memory storage with decision type)
                if tool_name == "Write" and "decision" in str(tool_input).lower():
                    await self._advance_to_analysis(session_id)

            elif current_state.protocol_step == ProtocolStep.ANALYSIS:
                # Check if analysis is complete (file reading, pattern analysis)
                if tool_name == "Read" or tool_name == "Grep":
                    await self._advance_to_research(session_id)

            elif current_state.protocol_step == ProtocolStep.RESEARCH:
                # Check if research is complete (Context7 usage)
                if "context7" in str(tool_input).lower():
                    await self._advance_to_planning(session_id)

            elif current_state.protocol_step == ProtocolStep.PLANNING:
                # Check if planning is complete (TodoWrite usage)
                if "todowrite" in str(tool_input).lower():
                    await self._advance_to_approval(session_id)

        except Exception as e:
            logger.logger.error(
                "step_advancement_evaluation_error",
                session_id=session_id,
                current_step=str(current_state.protocol_step),
                error=str(e)
            )

    async def _advance_to_analysis(self, session_id: str) -> None:
        """Advance protocol to ANALYSIS step."""
        current_state = await self.protocol_manager.get_current_state()
        if current_state.protocol_step == ProtocolStep.DISCUSSION:
            await self.protocol_manager.advance_step(
                current_state,
                ProtocolStep.ANALYSIS,
                metadata_updates={"step_advanced_at": datetime.now(timezone.utc).isoformat()}
            )
            logger.logger.info("protocol_advanced_to_analysis", session_id=session_id)

    async def _advance_to_research(self, session_id: str) -> None:
        """Advance protocol to RESEARCH step."""
        current_state = await self.protocol_manager.get_current_state()
        if current_state.protocol_step == ProtocolStep.ANALYSIS:
            await self.protocol_manager.advance_step(
                current_state,
                ProtocolStep.RESEARCH,
                metadata_updates={"step_advanced_at": datetime.now(timezone.utc).isoformat()}
            )
            logger.logger.info("protocol_advanced_to_research", session_id=session_id)

    async def _advance_to_planning(self, session_id: str) -> None:
        """Advance protocol to PLANNING step."""
        current_state = await self.protocol_manager.get_current_state()
        if current_state.protocol_step == ProtocolStep.RESEARCH:
            await self.protocol_manager.advance_step(
                current_state,
                ProtocolStep.PLANNING,
                metadata_updates={"step_advanced_at": datetime.now(timezone.utc).isoformat()}
            )
            logger.logger.info("protocol_advanced_to_planning", session_id=session_id)

    async def _advance_to_approval(self, session_id: str) -> None:
        """Advance protocol to APPROVAL step."""
        current_state = await self.protocol_manager.get_current_state()
        if current_state.protocol_step == ProtocolStep.PLANNING:
            await self.protocol_manager.advance_step(
                current_state,
                ProtocolStep.APPROVAL,
                metadata_updates={"step_advanced_at": datetime.now(timezone.utc).isoformat()}
            )
            logger.logger.info("protocol_advanced_to_approval", session_id=session_id)

    async def _validate_implementation_completion(self, session_id: str, task_id: str) -> None:
        """Validate implementation completion and advance to verification."""
        try:
            current_state = await self.protocol_manager.get_current_state()
            if current_state.protocol_step == ProtocolStep.IMPLEMENTATION:
                # Run step validation
                validation_result = await self.step_validator.validate_step(
                    "IMPLEMENTATION",
                    session_id,
                    task_id
                )

                if validation_result.status.value in ["passed", "warning"]:
                    await self.protocol_manager.advance_step(
                        current_state,
                        ProtocolStep.VERIFICATION,
                        metadata_updates={
                            "implementation_validated": True,
                            "validation_result": validation_result.status.value
                        }
                    )
                    logger.logger.info(
                        "protocol_advanced_to_verification",
                        session_id=session_id,
                        validation_status=validation_result.status.value
                    )

        except Exception as e:
            logger.logger.error(
                "implementation_validation_error",
                session_id=session_id,
                task_id=task_id,
                error=str(e)
            )

    async def _perform_state_sync(
        self,
        session_id: str,
        trigger: SyncTrigger,
        data: Dict[str, Any]
    ) -> None:
        """Perform actual state synchronization."""
        try:
            # Create sync event
            sync_event = SyncEvent(
                trigger=trigger,
                timestamp=datetime.now(timezone.utc).isoformat(),
                session_id=session_id,
                data=data,
                metadata={
                    "sync_count": self.sync_count + 1,
                    "active_sessions": len(self.active_sessions)
                }
            )

            # Add to history (keep last 100 events)
            self.sync_history.append(sync_event)
            if len(self.sync_history) > 100:
                self.sync_history = self.sync_history[-100:]

            # Update protocol state metadata
            current_state = await self.protocol_manager.get_current_state()
            if current_state.session_id == session_id:
                metrics = self.session_metrics.get(session_id)
                if metrics:
                    metadata_updates = {
                        "last_sync": sync_event.timestamp,
                        "sync_trigger": trigger.value,
                        "tools_used_count": sum(metrics.tools_used.values()),
                        "files_modified": metrics.files_modified,
                        "memory_entries": metrics.memory_entries
                    }

                    # Update state with new metadata
                    updated_state = current_state.with_updates(metadata_updates=metadata_updates)
                    await self.protocol_manager._save_state(updated_state)

            logger.logger.debug(
                "state_sync_performed",
                session_id=session_id,
                trigger=trigger.value,
                data_keys=list(data.keys())
            )

        except Exception as e:
            logger.logger.error(
                "state_sync_performance_error",
                session_id=session_id,
                trigger=trigger.value,
                error=str(e)
            )

    async def _restore_session_metrics(self, session_id: str) -> None:
        """Restore session metrics from saved state."""
        try:
            current_state = await self.protocol_manager.get_current_state()
            if current_state.session_id == session_id and current_state.metadata:
                # Restore metrics from state metadata
                metadata = current_state.metadata
                self.session_metrics[session_id] = ProgressMetrics(
                    tools_used=metadata.get("tools_used", {}),
                    files_modified=metadata.get("files_modified", 0),
                    lines_written=metadata.get("lines_written", 0),
                    memory_entries=metadata.get("memory_entries", 0),
                    todo_updates=metadata.get("todo_updates", 0),
                    context7_queries=metadata.get("context7_queries", 0),
                    duration_seconds=metadata.get("duration_seconds", 0.0),
                    last_activity=metadata.get("last_activity", datetime.now(timezone.utc).isoformat())
                )
                self.active_sessions.add(session_id)

        except Exception as e:
            logger.logger.error(
                "session_metrics_restore_error",
                session_id=session_id,
                error=str(e)
            )

    async def _log_crash_recovery(self, session_id: str) -> None:
        """Log crash recovery event to memory."""
        try:
            content = (
                f"Crash Recovery Completed\n"
                f"Session ID: {session_id}\n"
                f"Recovery Time: {datetime.now(timezone.utc).isoformat()}\n"
                f"Active Sessions: {len(self.active_sessions)}\n"
                f"Sync History Size: {len(self.sync_history)}"
            )

            keywords = [
                "crash-recovery",
                "session-restore",
                session_id,
                "protocol-state"
            ]

            await self.memory_client.store_memory(
                content=content,
                content_type="context",
                keywords=keywords
            )

        except Exception as e:
            logger.logger.error(
                "crash_recovery_logging_error",
                session_id=session_id,
                error=str(e)
            )

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session summary."""
        metrics = self.session_metrics.get(session_id)
        if not metrics:
            return {"error": "Session not found"}

        # Calculate session duration
        try:
            start_time = datetime.fromisoformat(metrics.last_activity.replace('Z', '+00:00'))
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        except:
            duration = 0

        return {
            "session_id": session_id,
            "duration_seconds": duration,
            "tools_used": metrics.tools_used,
            "total_tool_usage": sum(metrics.tools_used.values()),
            "files_modified": metrics.files_modified,
            "lines_written": metrics.lines_written,
            "memory_entries": metrics.memory_entries,
            "todo_updates": metrics.todo_updates,
            "context7_queries": metrics.context7_queries,
            "last_activity": metrics.last_activity,
            "is_active": session_id in self.active_sessions
        }

    def get_sync_statistics(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        return {
            "total_syncs": self.sync_count,
            "last_sync_time": self.last_sync_time,
            "active_sessions": len(self.active_sessions),
            "total_sessions": len(self.session_metrics),
            "sync_history_size": len(self.sync_history),
            "debouncer_active": self.state_debouncer is not None
        }


# Global instance
_global_sync: Optional[TaskStateSync] = None


def get_task_state_sync() -> TaskStateSync:
    """Get global task state sync instance."""
    global _global_sync
    if _global_sync is None:
        _global_sync = TaskStateSync()
    return _global_sync


# Example usage and testing
if __name__ == "__main__":
    async def test_task_state_sync():
        """Test task state sync functionality."""

        # Mock clients for testing
        class MockMemoryClient:
            async def store_memory(self, content, content_type, keywords):
                pass

        class MockProtocolManager:
            def __init__(self):
                self.state = {
                    "session_id": "test-session",
                    "protocol_step": 1,
                    "metadata": {}
                }

            async def get_current_state(self):
                return type('State', (), self.state)()

            async def advance_step(self, state, step, metadata_updates=None):
                self.state["protocol_step"] = step.value
                if metadata_updates:
                    self.state["metadata"].update(metadata_updates)

            async def _save_state(self, state):
                pass

        sync = TaskStateSync(
            memory_client=MockMemoryClient(),
            protocol_manager=MockProtocolManager()
        )

        # Test 1: Tool execution sync
        print("🧪 Test 1: Tool execution synchronization")
        success = await sync.synchronize_on_tool_execution(
            tool_name="Write",
            tool_input={"file_path": "test.py", "content": "print('hello')\nprint('world')"},
            execution_result={"success": True},
            session_id="test-session"
        )
        assert success == True
        print("✅ Tool execution sync completed")

        # Test 2: Memory storage sync
        print("\n🧪 Test 2: Memory storage synchronization")
        success = await sync.synchronize_on_memory_storage(
            content="Decision made about implementation approach",
            content_type="decision",
            keywords=["decision", "implementation"],
            session_id="test-session"
        )
        assert success == True
        print("✅ Memory storage sync completed")

        # Test 3: Task update sync
        print("\n🧪 Test 3: Task update synchronization")
        success = await sync.synchronize_on_task_update(
            task_id="task-123",
            update_type="todo_update",
            update_data={"status": "in_progress"},
            session_id="test-session"
        )
        assert success == True
        print("✅ Task update sync completed")

        # Test 4: Session summary
        print("\n🧪 Test 4: Session summary")
        summary = sync.get_session_summary("test-session")
        assert summary["session_id"] == "test-session"
        assert summary["files_modified"] == 1
        assert summary["memory_entries"] == 1
        print(f"✅ Session summary: {summary['files_modified']} files, {summary['memory_entries']} memory entries")

        # Test 5: Sync statistics
        print("\n🧪 Test 5: Sync statistics")
        stats = sync.get_sync_statistics()
        assert stats["total_syncs"] >= 3
        assert stats["active_sessions"] == 1
        print(f"✅ Statistics: {stats['total_syncs']} total syncs")

        # Test 6: Crash recovery
        print("\n🧪 Test 6: Crash recovery")
        success = await sync.perform_crash_recovery("test-session")
        assert success == True
        print("✅ Crash recovery completed")

        print("\n🎉 All task state sync tests PASSED")

    # Run tests
    import sys
    asyncio.run(test_task_state_sync())