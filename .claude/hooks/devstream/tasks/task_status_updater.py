#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pydantic>=2.0.0",
#     "python-dotenv>=1.0.0",
#     "aiohttp>=3.8.0",
# ]
# ///

"""
DevStream Task Status Updater - Automated Task Status Management
Context7-compliant task status updates basati su tool usage patterns e session activity.
"""

import json
import sys
import os
import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# Import DevStream utilities
sys.path.append(str(Path(__file__).parent.parent / 'utils'))
from common import DevStreamHookBase, get_project_context
from logger import get_devstream_logger
from mcp_client import get_mcp_client

class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class TaskStatusUpdate:
    """Task status update information."""
    task_id: str
    old_status: str
    new_status: str
    reason: str
    confidence: float
    evidence: List[str]
    timestamp: datetime

class TaskStatusUpdater(DevStreamHookBase):
    """
    Task Status Updater per automated task management durante Claude Code sessions.
    Implementa Context7-validated patterns per task lifecycle automation.
    """

    def __init__(self):
        super().__init__('task_status_updater')
        self.structured_logger = get_devstream_logger('task_status_updater')
        self.mcp_client = get_mcp_client()
        self.start_time = time.time()

        # Status update configuration
        self.update_interval = 300  # 5 minutes
        self.confidence_threshold = 0.7
        self.max_pending_time = 3600  # 1 hour
        self.activity_window = 1800  # 30 minutes

    async def analyze_and_update_tasks(self, trigger_event: Optional[Dict[str, Any]] = None) -> List[TaskStatusUpdate]:
        """
        Analyze current task status e perform updates based on session activity.

        Args:
            trigger_event: Optional triggering event data

        Returns:
            List of task status updates performed
        """
        self.structured_logger.log_hook_start(
            trigger_event or {},
            {"phase": "task_status_analysis"}
        )

        updates_performed = []

        try:
            # Get current active and pending tasks
            current_tasks = await self.get_current_tasks()

            if not current_tasks:
                self.logger.debug("No tasks to analyze for status updates")
                return updates_performed

            # Analyze recent session activity
            session_activity = await self.analyze_recent_activity()

            # Process each task for potential status updates
            for task in current_tasks:
                potential_updates = await self.evaluate_task_status(
                    task,
                    session_activity,
                    trigger_event
                )

                for update in potential_updates:
                    if update.confidence >= self.confidence_threshold:
                        success = await self.apply_task_update(update)
                        if success:
                            updates_performed.append(update)

            # Check for stale tasks that need attention
            stale_updates = await self.check_stale_tasks(current_tasks, session_activity)
            for update in stale_updates:
                success = await self.apply_task_update(update)
                if success:
                    updates_performed.append(update)

            # Log summary of updates
            if updates_performed:
                self.logger.info(f"Applied {len(updates_performed)} task status updates")

                # Store update summary in memory
                await self.store_update_summary(updates_performed)

            return updates_performed

        except Exception as e:
            self.structured_logger.log_hook_error(e, {"trigger_event": trigger_event})
            raise

    async def get_current_tasks(self) -> List[Dict[str, Any]]:
        """
        Get current tasks from DevStream system.

        Returns:
            List of current tasks
        """
        try:
            # Get active and pending tasks
            active_tasks_response = await self.mcp_client.list_tasks(status="active")
            pending_tasks_response = await self.mcp_client.list_tasks(status="pending")

            tasks = []

            # Parse active tasks response
            if active_tasks_response and active_tasks_response.get('content'):
                content = active_tasks_response.get('content', [])
                if content and len(content) > 0:
                    text_content = content[0].get('text', '')
                    active_tasks = self.parse_tasks_from_response(text_content)
                    tasks.extend(active_tasks)

            # Parse pending tasks response
            if pending_tasks_response and pending_tasks_response.get('content'):
                content = pending_tasks_response.get('content', [])
                if content and len(content) > 0:
                    text_content = content[0].get('text', '')
                    pending_tasks = self.parse_tasks_from_response(text_content)
                    tasks.extend(pending_tasks)

            return tasks

        except Exception as e:
            self.logger.warning(f"Failed to get current tasks: {e}")
            return []

    def parse_tasks_from_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse tasks from MCP response text.

        Args:
            response_text: MCP response text

        Returns:
            List of parsed tasks
        """
        tasks = []

        # Simple parsing - in production would have structured MCP response
        if "Task ID:" in response_text:
            # Simulate parsing structured task data
            lines = response_text.split('\n')
            current_task = {}

            for line in lines:
                line = line.strip()
                if line.startswith("Task ID:"):
                    if current_task:
                        tasks.append(current_task)
                    current_task = {"id": line.split(":", 1)[1].strip()}
                elif line.startswith("Title:") and current_task:
                    current_task["title"] = line.split(":", 1)[1].strip()
                elif line.startswith("Status:") and current_task:
                    current_task["status"] = line.split(":", 1)[1].strip()
                elif line.startswith("Priority:") and current_task:
                    try:
                        current_task["priority"] = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        current_task["priority"] = 5
                elif line.startswith("Phase:") and current_task:
                    current_task["phase"] = line.split(":", 1)[1].strip()
                elif line.startswith("Created:") and current_task:
                    current_task["created"] = line.split(":", 1)[1].strip()

            if current_task:
                tasks.append(current_task)

        return tasks

    async def analyze_recent_activity(self) -> Dict[str, Any]:
        """
        Analyze recent session activity per task status evaluation.

        Returns:
            Session activity analysis
        """
        activity_analysis = {
            "tool_usage": [],
            "code_changes": [],
            "success_operations": 0,
            "failed_operations": 0,
            "active_files": [],
            "implementation_patterns": [],
            "recent_memories": [],
            "activity_confidence": 0.0
        }

        try:
            # Search for recent session activity in memory
            recent_query = f"tool result success implementation code"

            recent_memories_response = await self.mcp_client.search_memory(
                query=recent_query,
                limit=10
            )

            if recent_memories_response:
                activity_analysis["recent_memories"] = self.extract_activity_from_memories(
                    recent_memories_response
                )

            # Calculate overall activity confidence
            activity_analysis["activity_confidence"] = self.calculate_activity_confidence(
                activity_analysis
            )

            return activity_analysis

        except Exception as e:
            self.logger.warning(f"Failed to analyze recent activity: {e}")
            return activity_analysis

    def extract_activity_from_memories(self, memories_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract activity indicators from memory search results.

        Args:
            memories_response: MCP search response

        Returns:
            List of activity indicators
        """
        activity_indicators = []

        content = memories_response.get('content', [])
        if content and len(content) > 0:
            text_content = content[0].get('text', '')

            # Look for tool usage patterns
            if 'TOOL:' in text_content or 'Success: True' in text_content:
                activity_indicators.append({
                    "type": "tool_success",
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.8
                })

            # Look for implementation patterns
            if any(keyword in text_content.lower() for keyword in [
                'edit', 'write', 'create', 'implement', 'fix'
            ]):
                activity_indicators.append({
                    "type": "implementation",
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.7
                })

            # Look for completion indicators
            if any(keyword in text_content.lower() for keyword in [
                'completed', 'finished', 'done', 'success'
            ]):
                activity_indicators.append({
                    "type": "completion",
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.9
                })

        return activity_indicators

    def calculate_activity_confidence(self, activity_analysis: Dict[str, Any]) -> float:
        """
        Calculate confidence score for recent activity.

        Args:
            activity_analysis: Activity analysis data

        Returns:
            Confidence score (0.0-1.0)
        """
        confidence = 0.0
        total_weight = 0.0

        # Weight based on recent memories
        memory_count = len(activity_analysis.get("recent_memories", []))
        if memory_count > 0:
            confidence += min(memory_count * 0.1, 0.4)
            total_weight += 0.4

        # Weight based on success/failure ratio
        success_ops = activity_analysis.get("success_operations", 0)
        failed_ops = activity_analysis.get("failed_operations", 0)
        total_ops = success_ops + failed_ops

        if total_ops > 0:
            success_ratio = success_ops / total_ops
            confidence += success_ratio * 0.3
            total_weight += 0.3

        # Weight based on implementation patterns
        impl_patterns = len(activity_analysis.get("implementation_patterns", []))
        if impl_patterns > 0:
            confidence += min(impl_patterns * 0.1, 0.3)
            total_weight += 0.3

        # Normalize by total weight
        if total_weight > 0:
            confidence = confidence / total_weight

        return min(confidence, 1.0)

    async def evaluate_task_status(
        self,
        task: Dict[str, Any],
        session_activity: Dict[str, Any],
        trigger_event: Optional[Dict[str, Any]] = None
    ) -> List[TaskStatusUpdate]:
        """
        Evaluate if a task needs status update.

        Args:
            task: Task information
            session_activity: Recent session activity
            trigger_event: Optional trigger event

        Returns:
            List of potential task updates
        """
        potential_updates = []

        task_id = task.get('id', 'unknown')
        current_status = task.get('status', 'pending')
        task_title = task.get('title', '').lower()

        # Check for completion indicators
        if current_status in ['active', 'pending']:
            completion_update = await self.check_task_completion(
                task, session_activity, trigger_event
            )
            if completion_update:
                potential_updates.append(completion_update)

        # Check for activation indicators
        if current_status == 'pending':
            activation_update = await self.check_task_activation(
                task, session_activity, trigger_event
            )
            if activation_update:
                potential_updates.append(activation_update)

        # Check for failure indicators
        if current_status == 'active':
            failure_update = await self.check_task_failure(
                task, session_activity, trigger_event
            )
            if failure_update:
                potential_updates.append(failure_update)

        return potential_updates

    async def check_task_completion(
        self,
        task: Dict[str, Any],
        session_activity: Dict[str, Any],
        trigger_event: Optional[Dict[str, Any]]
    ) -> Optional[TaskStatusUpdate]:
        """
        Check if task should be marked as completed.

        Args:
            task: Task information
            session_activity: Session activity
            trigger_event: Trigger event

        Returns:
            Task completion update or None
        """
        task_id = task.get('id', 'unknown')
        task_title = task.get('title', '').lower()

        evidence = []
        confidence = 0.0

        # Check for completion patterns in activity
        recent_memories = session_activity.get("recent_memories", [])

        for memory in recent_memories:
            if memory.get("type") == "completion":
                evidence.append("Completion indicators found in recent activity")
                confidence += 0.3
            elif memory.get("type") == "implementation":
                evidence.append("Implementation activity detected")
                confidence += 0.2

        # Check trigger event for completion signals
        if trigger_event:
            tool_name = trigger_event.get('tool_name', '')
            if tool_name == 'TodoWrite':
                # Check if todo was marked as completed
                tool_input = trigger_event.get('tool_input', {})
                todos = tool_input.get('todos', [])

                for todo in todos:
                    if todo.get('status') == 'completed' and task_title in todo.get('content', '').lower():
                        evidence.append(f"Todo marked completed: {todo.get('content')}")
                        confidence += 0.5

        # Task-specific completion logic
        if 'hook' in task_title and 'implementation' in task_title:
            # Check for hook file creation/modification
            if any(mem.get("type") == "implementation" for mem in recent_memories):
                evidence.append("Hook implementation activity detected")
                confidence += 0.3

        # Only suggest completion if confidence is reasonable
        if confidence >= 0.6 and evidence:
            return TaskStatusUpdate(
                task_id=task_id,
                old_status=task.get('status', 'active'),
                new_status=TaskStatus.COMPLETED.value,
                reason=f"Automatic completion based on session activity (confidence: {confidence:.2f})",
                confidence=confidence,
                evidence=evidence,
                timestamp=datetime.now()
            )

        return None

    async def check_task_activation(
        self,
        task: Dict[str, Any],
        session_activity: Dict[str, Any],
        trigger_event: Optional[Dict[str, Any]]
    ) -> Optional[TaskStatusUpdate]:
        """
        Check if pending task should be activated.

        Args:
            task: Task information
            session_activity: Session activity
            trigger_event: Trigger event

        Returns:
            Task activation update or None
        """
        task_id = task.get('id', 'unknown')
        task_title = task.get('title', '').lower()

        evidence = []
        confidence = 0.0

        # Check for work starting on this task
        recent_memories = session_activity.get("recent_memories", [])

        for memory in recent_memories:
            if memory.get("type") == "implementation":
                evidence.append("Implementation work started")
                confidence += 0.4

        # Check trigger event for activation signals
        if trigger_event:
            tool_name = trigger_event.get('tool_name', '')

            if tool_name == 'TodoWrite':
                # Check if todo was marked as in_progress
                tool_input = trigger_event.get('tool_input', {})
                todos = tool_input.get('todos', [])

                for todo in todos:
                    if todo.get('status') == 'in_progress' and task_title in todo.get('content', '').lower():
                        evidence.append(f"Todo marked in progress: {todo.get('content')}")
                        confidence += 0.7

        # Auto-activate high-priority pending tasks with recent activity
        task_priority = task.get('priority', 5)
        if task_priority >= 8 and session_activity.get("activity_confidence", 0.0) > 0.5:
            evidence.append(f"High-priority task with active session (priority: {task_priority})")
            confidence += 0.3

        if confidence >= 0.7 and evidence:
            return TaskStatusUpdate(
                task_id=task_id,
                old_status=task.get('status', 'pending'),
                new_status=TaskStatus.ACTIVE.value,
                reason=f"Automatic activation based on work initiation (confidence: {confidence:.2f})",
                confidence=confidence,
                evidence=evidence,
                timestamp=datetime.now()
            )

        return None

    async def check_task_failure(
        self,
        task: Dict[str, Any],
        session_activity: Dict[str, Any],
        trigger_event: Optional[Dict[str, Any]]
    ) -> Optional[TaskStatusUpdate]:
        """
        Check if active task should be marked as failed.

        Args:
            task: Task information
            session_activity: Session activity
            trigger_event: Trigger event

        Returns:
            Task failure update or None
        """
        task_id = task.get('id', 'unknown')

        evidence = []
        confidence = 0.0

        # Check for failure patterns
        failed_operations = session_activity.get("failed_operations", 0)
        success_operations = session_activity.get("success_operations", 0)

        if failed_operations > success_operations and failed_operations > 3:
            evidence.append(f"High failure rate: {failed_operations} failures vs {success_operations} successes")
            confidence += 0.4

        # Check for explicit failure indicators in trigger event
        if trigger_event:
            tool_name = trigger_event.get('tool_name', '')
            tool_input = trigger_event.get('tool_input', {})

            if 'error' in str(tool_input).lower() or 'failed' in str(tool_input).lower():
                evidence.append("Explicit failure indicators in tool usage")
                confidence += 0.5

        # Only mark as failed with high confidence and significant evidence
        if confidence >= 0.8 and len(evidence) >= 2:
            return TaskStatusUpdate(
                task_id=task_id,
                old_status=task.get('status', 'active'),
                new_status=TaskStatus.FAILED.value,
                reason=f"Automatic failure detection (confidence: {confidence:.2f})",
                confidence=confidence,
                evidence=evidence,
                timestamp=datetime.now()
            )

        return None

    async def check_stale_tasks(
        self,
        current_tasks: List[Dict[str, Any]],
        session_activity: Dict[str, Any]
    ) -> List[TaskStatusUpdate]:
        """
        Check for tasks that have been stale for too long.

        Args:
            current_tasks: Current task list
            session_activity: Session activity analysis

        Returns:
            List of stale task updates
        """
        stale_updates = []
        now = datetime.now()

        for task in current_tasks:
            task_id = task.get('id', 'unknown')
            status = task.get('status', 'pending')
            created_str = task.get('created', '')

            try:
                # Parse creation date (simplified - would need proper parsing)
                if created_str:
                    # Assume ISO format for now
                    created_date = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                    time_since_creation = (now - created_date).total_seconds()

                    # Check for stale pending tasks
                    if status == 'pending' and time_since_creation > self.max_pending_time:
                        # Low activity and old pending task -> skip
                        if session_activity.get("activity_confidence", 0.0) < 0.3:
                            stale_updates.append(TaskStatusUpdate(
                                task_id=task_id,
                                old_status=status,
                                new_status=TaskStatus.SKIPPED.value,
                                reason=f"Stale pending task with low activity (age: {time_since_creation/3600:.1f}h)",
                                confidence=0.8,
                                evidence=[f"Task pending for {time_since_creation/3600:.1f} hours with minimal session activity"],
                                timestamp=now
                            ))

            except (ValueError, TypeError) as e:
                self.logger.warning(f"Failed to parse task creation date: {e}")
                continue

        return stale_updates

    async def apply_task_update(self, update: TaskStatusUpdate) -> bool:
        """
        Apply task status update via MCP.

        Args:
            update: Task status update to apply

        Returns:
            True if update was successful
        """
        try:
            # Create update notes
            notes = f"{update.reason}. Evidence: {'; '.join(update.evidence)}"

            # Update task via MCP
            update_result = await self.mcp_client.update_task(
                task_id=update.task_id,
                status=update.new_status,
                notes=notes
            )

            if update_result:
                self.logger.info(f"Updated task {update.task_id}: {update.old_status} → {update.new_status}")

                # Log the update
                self.structured_logger.log_task_operation(
                    operation="status_update",
                    task_id=update.task_id,
                    task_type="automated_update",
                    status=update.new_status
                )

                return True
            else:
                self.logger.warning(f"Failed to update task {update.task_id}")
                return False

        except Exception as e:
            self.logger.error(f"Error applying task update for {update.task_id}: {e}")
            return False

    async def store_update_summary(self, updates: List[TaskStatusUpdate]) -> None:
        """
        Store summary of task updates in memory.

        Args:
            updates: List of applied updates
        """
        if not updates:
            return

        summary_parts = [
            f"TASK STATUS UPDATES [{datetime.now().isoformat()}]",
            f"Applied {len(updates)} automatic status updates:"
        ]

        for update in updates:
            summary_parts.append(
                f"- {update.task_id}: {update.old_status} → {update.new_status} "
                f"(confidence: {update.confidence:.2f})"
            )

        summary_parts.append(f"Evidence summary: {len([e for u in updates for e in u.evidence])} indicators processed")

        summary_content = '\n'.join(summary_parts)

        # Store in memory
        await self.mcp_client.store_memory(
            content=summary_content,
            content_type="context",
            keywords=["task-updates", "automation", "status-changes", "devstream"]
        )

        # Log memory operation
        self.structured_logger.log_memory_operation(
            operation="store",
            content_type="context",
            content_size=len(summary_content),
            keywords=["task-updates", "automation"]
        )

def create_updater_instance() -> TaskStatusUpdater:
    """Create and return TaskStatusUpdater instance."""
    return TaskStatusUpdater()

async def run_status_update_check(trigger_event: Optional[Dict[str, Any]] = None) -> List[TaskStatusUpdate]:
    """
    Run task status update check.

    Args:
        trigger_event: Optional triggering event

    Returns:
        List of applied updates
    """
    updater = create_updater_instance()
    return await updater.analyze_and_update_tasks(trigger_event)

async def main():
    """Main execution for standalone usage."""
    updater = TaskStatusUpdater()

    try:
        # Read JSON input from stdin if available
        input_data = updater.read_stdin_json() if not sys.stdin.isatty() else {}

        # Run status update analysis
        updates = await updater.analyze_and_update_tasks(input_data)

        # Output results
        if updates:
            print(f"✅ Applied {len(updates)} task status updates:")
            for update in updates:
                print(f"   📋 {update.task_id}: {update.old_status} → {update.new_status}")
                print(f"      Reason: {update.reason}")
                print(f"      Confidence: {update.confidence:.2f}")
        else:
            print("ℹ️  No task status updates needed at this time")

        # Success exit
        updater.success_exit()

    except Exception as e:
        updater.error_exit(f"Task status updater failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())