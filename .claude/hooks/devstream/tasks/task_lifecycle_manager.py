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
DevStream Task Lifecycle Manager - Complete Task Automation
Context7-compliant orchestration di tutti i task management hooks per automation completa.
"""

import json
import sys
import os
import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
from enum import Enum

# Import DevStream utilities
sys.path.append(str(Path(__file__).parent.parent / 'utils'))
from common import DevStreamHookBase, get_project_context
from logger import get_devstream_logger
from mcp_client import get_mcp_client

# Import other task management components
sys.path.append(str(Path(__file__).parent))
from task_status_updater import TaskStatusUpdater, TaskStatusUpdate, run_status_update_check

class TaskLifecycleEvent(Enum):
    """Task lifecycle event types."""
    TASK_CREATED = "task_created"
    TASK_STARTED = "task_started"
    PROGRESS_UPDATE = "progress_update"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    TOOL_EXECUTION = "tool_execution"

@dataclass
class TaskLifecycleState:
    """Current task lifecycle state."""
    active_tasks: Set[str]
    completed_tasks: Set[str]
    failed_tasks: Set[str]
    session_id: str
    last_update: datetime
    automation_enabled: bool

class TaskLifecycleManager(DevStreamHookBase):
    """
    Task Lifecycle Manager per complete automation di task management.
    Coordina tutti gli hook per seamless task lifecycle automation.
    """

    def __init__(self):
        super().__init__('task_lifecycle_manager')
        self.structured_logger = get_devstream_logger('task_lifecycle_manager')
        self.mcp_client = get_mcp_client()
        self.start_time = time.time()

        # Lifecycle management configuration
        self.automation_enabled = True
        self.update_frequency = 300  # 5 minutes
        self.progress_check_interval = 120  # 2 minutes
        self.max_concurrent_tasks = 5

        # State management
        self.current_state = TaskLifecycleState(
            active_tasks=set(),
            completed_tasks=set(),
            failed_tasks=set(),
            session_id="unknown",
            last_update=datetime.now(),
            automation_enabled=True
        )

        # Task status updater
        self.status_updater = TaskStatusUpdater()

    async def handle_lifecycle_event(
        self,
        event_type: TaskLifecycleEvent,
        event_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle task lifecycle event e orchestrate appropriate actions.

        Args:
            event_type: Type of lifecycle event
            event_data: Event data payload

        Returns:
            Event handling results
        """
        self.structured_logger.log_hook_start(event_data, {
            "phase": "lifecycle_event_handling",
            "event_type": event_type.value
        })

        results = {
            "event_type": event_type.value,
            "timestamp": datetime.now().isoformat(),
            "actions_taken": [],
            "status_updates": [],
            "automation_triggered": False
        }

        try:
            # Update current state
            await self.update_lifecycle_state(event_type, event_data)

            # Handle specific event types
            if event_type == TaskLifecycleEvent.SESSION_START:
                results.update(await self.handle_session_start(event_data))

            elif event_type == TaskLifecycleEvent.TOOL_EXECUTION:
                results.update(await self.handle_tool_execution(event_data))

            elif event_type == TaskLifecycleEvent.PROGRESS_UPDATE:
                results.update(await self.handle_progress_update(event_data))

            elif event_type == TaskLifecycleEvent.SESSION_END:
                results.update(await self.handle_session_end(event_data))

            # Run periodic automation checks
            if self.should_run_automation_check():
                automation_results = await self.run_automation_cycle()
                results["status_updates"].extend(automation_results.get("status_updates", []))
                results["automation_triggered"] = True

            # Log results
            self.logger.info(f"Handled {event_type.value}: {len(results['actions_taken'])} actions, "
                           f"{len(results['status_updates'])} updates")

            return results

        except Exception as e:
            self.structured_logger.log_hook_error(e, {
                "event_type": event_type.value,
                "event_data": event_data
            })
            raise

    async def handle_session_start(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle session start event.

        Args:
            event_data: Session start data

        Returns:
            Session start handling results
        """
        results = {"actions_taken": [], "status_updates": []}

        session_id = event_data.get('session_id', 'unknown')
        self.current_state.session_id = session_id

        # Initialize session with current task state
        current_tasks = await self.get_current_tasks_summary()

        if current_tasks:
            # Check for tasks that should be auto-activated
            for task in current_tasks:
                if (task.get('status') == 'pending' and
                    task.get('priority', 5) >= 8 and
                    'hook' in task.get('title', '').lower()):

                    # Auto-activate high-priority hook tasks
                    activation_result = await self.auto_activate_task(task, "Session start with high-priority task")
                    if activation_result:
                        results["status_updates"].append(activation_result)
                        results["actions_taken"].append(f"Auto-activated task: {task['title']}")

            # Store session initialization context
            await self.store_session_context(session_id, current_tasks)
            results["actions_taken"].append("Stored session initialization context")

        return results

    async def handle_tool_execution(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle tool execution event.

        Args:
            event_data: Tool execution data

        Returns:
            Tool execution handling results
        """
        results = {"actions_taken": [], "status_updates": []}

        tool_name = event_data.get('tool_name', 'unknown')
        tool_success = event_data.get('success', False)

        # Track tool execution patterns
        if tool_success and tool_name in ['Edit', 'Write', 'MultiEdit']:
            # Code modification tools indicate active development
            progress_update = await self.update_task_progress_from_tool(event_data)
            if progress_update:
                results["actions_taken"].append(f"Updated task progress from {tool_name}")

        # Check if tool execution indicates task completion
        if tool_name == 'TodoWrite':
            todo_updates = await self.analyze_todo_updates(event_data)
            results["status_updates"].extend(todo_updates)
            if todo_updates:
                results["actions_taken"].append(f"Processed {len(todo_updates)} todo-based status updates")

        # Trigger status update check for significant tools
        if tool_name in ['Edit', 'Write', 'MultiEdit', 'TodoWrite', 'Bash']:
            status_updates = await run_status_update_check(event_data)
            results["status_updates"].extend(status_updates)

        return results

    async def handle_progress_update(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle progress update event.

        Args:
            event_data: Progress update data

        Returns:
            Progress update handling results
        """
        results = {"actions_taken": [], "status_updates": []}

        task_id = event_data.get('task_id')
        progress_score = event_data.get('progress_score', 0.0)

        if task_id and progress_score > 0.8:
            # High progress might indicate near completion
            completion_check = await self.check_near_completion(task_id, progress_score)
            if completion_check:
                results["status_updates"].append(completion_check)
                results["actions_taken"].append(f"Detected near-completion for task {task_id}")

        return results

    async def handle_session_end(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle session end event.

        Args:
            event_data: Session end data

        Returns:
            Session end handling results
        """
        results = {"actions_taken": [], "status_updates": []}

        # Final automation cycle
        automation_results = await self.run_automation_cycle()
        results["status_updates"].extend(automation_results.get("status_updates", []))

        # Store session completion summary
        session_summary = await self.generate_session_summary()
        await self.store_session_completion(session_summary)
        results["actions_taken"].append("Stored session completion summary")

        return results

    async def run_automation_cycle(self) -> Dict[str, Any]:
        """
        Run complete automation cycle.

        Returns:
            Automation cycle results
        """
        results = {"status_updates": [], "actions_taken": []}

        try:
            # Run status update check
            status_updates = await self.status_updater.analyze_and_update_tasks()
            results["status_updates"].extend(status_updates)

            # Check for tasks needing attention
            attention_tasks = await self.identify_tasks_needing_attention()
            for task_id in attention_tasks:
                results["actions_taken"].append(f"Flagged task for attention: {task_id}")

            # Update lifecycle state
            await self.refresh_lifecycle_state()

            self.logger.info(f"Automation cycle completed: {len(results['status_updates'])} updates")

        except Exception as e:
            self.logger.error(f"Automation cycle failed: {e}")

        return results

    async def auto_activate_task(self, task: Dict[str, Any], reason: str) -> Optional[TaskStatusUpdate]:
        """
        Auto-activate a pending task.

        Args:
            task: Task to activate
            reason: Activation reason

        Returns:
            Status update if successful
        """
        task_id = task.get('id', 'unknown')

        try:
            update_result = await self.mcp_client.update_task(
                task_id=task_id,
                status="active",
                notes=f"Auto-activated: {reason}"
            )

            if update_result:
                self.current_state.active_tasks.add(task_id)

                return TaskStatusUpdate(
                    task_id=task_id,
                    old_status="pending",
                    new_status="active",
                    reason=reason,
                    confidence=0.9,
                    evidence=[reason],
                    timestamp=datetime.now()
                )

        except Exception as e:
            self.logger.error(f"Failed to auto-activate task {task_id}: {e}")

        return None

    async def analyze_todo_updates(self, event_data: Dict[str, Any]) -> List[TaskStatusUpdate]:
        """
        Analyze TodoWrite updates for task status changes.

        Args:
            event_data: Tool execution data

        Returns:
            List of inferred status updates
        """
        updates = []

        tool_input = event_data.get('tool_input', {})
        todos = tool_input.get('todos', [])

        for todo in todos:
            status = todo.get('status')
            content = todo.get('content', '').lower()

            # Map todo status to potential task updates
            if status == 'completed' and any(keyword in content for keyword in [
                'hook', 'task', 'implement', 'create', 'develop'
            ]):
                # Look for matching tasks
                matching_tasks = await self.find_tasks_by_keywords(content.split()[:5])
                for task in matching_tasks:
                    if task.get('status') in ['active', 'pending']:
                        updates.append(TaskStatusUpdate(
                            task_id=task['id'],
                            old_status=task['status'],
                            new_status='completed',
                            reason=f"TodoWrite completion: {todo['content'][:100]}",
                            confidence=0.8,
                            evidence=[f"Todo marked completed: {content}"],
                            timestamp=datetime.now()
                        ))

            elif status == 'in_progress' and any(keyword in content for keyword in [
                'hook', 'task', 'implement', 'create', 'develop'
            ]):
                # Activate matching pending tasks
                matching_tasks = await self.find_tasks_by_keywords(content.split()[:5])
                for task in matching_tasks:
                    if task.get('status') == 'pending':
                        updates.append(TaskStatusUpdate(
                            task_id=task['id'],
                            old_status='pending',
                            new_status='active',
                            reason=f"TodoWrite work started: {todo['content'][:100]}",
                            confidence=0.7,
                            evidence=[f"Todo marked in progress: {content}"],
                            timestamp=datetime.now()
                        ))

        return updates

    async def find_tasks_by_keywords(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Find tasks matching keywords.

        Args:
            keywords: Keywords to match

        Returns:
            List of matching tasks
        """
        try:
            tasks_response = await self.mcp_client.list_tasks()

            if not tasks_response or not tasks_response.get('content'):
                return []

            # Parse tasks and match keywords
            content = tasks_response.get('content', [])
            if content and len(content) > 0:
                text_content = content[0].get('text', '')
                all_tasks = self.status_updater.parse_tasks_from_response(text_content)

                matching_tasks = []
                for task in all_tasks:
                    task_title = task.get('title', '').lower()
                    if any(keyword.lower() in task_title for keyword in keywords if len(keyword) > 2):
                        matching_tasks.append(task)

                return matching_tasks

        except Exception as e:
            self.logger.warning(f"Failed to find tasks by keywords: {e}")

        return []

    async def update_task_progress_from_tool(self, event_data: Dict[str, Any]) -> bool:
        """
        Update task progress based on tool execution.

        Args:
            event_data: Tool execution data

        Returns:
            True if progress was updated
        """
        tool_name = event_data.get('tool_name', '')
        tool_input = event_data.get('tool_input', {})

        # Extract relevant information from tool usage
        if tool_name in ['Edit', 'Write', 'MultiEdit']:
            file_path = tool_input.get('file_path', '')

            # Check if this relates to active tasks
            if any(keyword in file_path.lower() for keyword in ['hook', 'devstream', 'task']):
                # This is likely progress on DevStream tasks
                await self.store_progress_indicator(tool_name, file_path)
                return True

        return False

    async def store_progress_indicator(self, tool_name: str, file_path: str) -> None:
        """
        Store progress indicator in memory.

        Args:
            tool_name: Tool used
            file_path: File modified
        """
        progress_content = (
            f"TASK PROGRESS INDICATOR: {tool_name} operation on {file_path}. "
            f"Timestamp: {datetime.now().isoformat()}. "
            f"Session: {self.current_state.session_id}"
        )

        await self.mcp_client.store_memory(
            content=progress_content,
            content_type="context",
            keywords=["task-progress", "development", "automation", file_path.split('/')[-1]]
        )

    async def get_current_tasks_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary of current tasks.

        Returns:
            List of current tasks
        """
        return await self.status_updater.get_current_tasks()

    async def check_near_completion(self, task_id: str, progress_score: float) -> Optional[TaskStatusUpdate]:
        """
        Check if task is near completion.

        Args:
            task_id: Task ID
            progress_score: Current progress score

        Returns:
            Completion update if warranted
        """
        if progress_score >= 0.9:
            # Very high progress score suggests completion
            return TaskStatusUpdate(
                task_id=task_id,
                old_status="active",
                new_status="completed",
                reason=f"High progress score indicates completion ({progress_score:.2f})",
                confidence=progress_score,
                evidence=[f"Progress score: {progress_score:.2f}"],
                timestamp=datetime.now()
            )

        return None

    async def identify_tasks_needing_attention(self) -> List[str]:
        """
        Identify tasks that need attention.

        Returns:
            List of task IDs needing attention
        """
        attention_tasks = []

        # Get current tasks
        current_tasks = await self.get_current_tasks_summary()

        for task in current_tasks:
            task_id = task.get('id', 'unknown')
            status = task.get('status', 'unknown')
            priority = task.get('priority', 5)

            # High priority active tasks with no recent progress
            if status == 'active' and priority >= 8:
                # Check for recent progress
                recent_progress = await self.check_recent_task_progress(task_id)
                if not recent_progress:
                    attention_tasks.append(task_id)

        return attention_tasks

    async def check_recent_task_progress(self, task_id: str) -> bool:
        """
        Check if task has had recent progress.

        Args:
            task_id: Task ID to check

        Returns:
            True if recent progress detected
        """
        # Search for recent progress indicators
        progress_query = f"task progress {task_id[:8]} development"

        recent_memories = await self.mcp_client.search_memory(
            query=progress_query,
            limit=3
        )

        if recent_memories and recent_memories.get('content'):
            # Check if any memories are recent (simplified check)
            content = recent_memories.get('content', [])
            if content and len(content) > 0:
                return True

        return False

    def should_run_automation_check(self) -> bool:
        """
        Check if automation cycle should run.

        Returns:
            True if automation should run
        """
        if not self.automation_enabled:
            return False

        time_since_last_update = (datetime.now() - self.current_state.last_update).total_seconds()
        return time_since_last_update >= self.update_frequency

    async def update_lifecycle_state(
        self,
        event_type: TaskLifecycleEvent,
        event_data: Dict[str, Any]
    ) -> None:
        """
        Update current lifecycle state.

        Args:
            event_type: Event type
            event_data: Event data
        """
        self.current_state.last_update = datetime.now()

        if event_type == TaskLifecycleEvent.SESSION_START:
            self.current_state.session_id = event_data.get('session_id', 'unknown')

    async def refresh_lifecycle_state(self) -> None:
        """Refresh lifecycle state from current tasks."""
        current_tasks = await self.get_current_tasks_summary()

        # Update task sets
        self.current_state.active_tasks = {
            task['id'] for task in current_tasks if task.get('status') == 'active'
        }
        self.current_state.completed_tasks = {
            task['id'] for task in current_tasks if task.get('status') == 'completed'
        }
        self.current_state.failed_tasks = {
            task['id'] for task in current_tasks if task.get('status') == 'failed'
        }

    async def generate_session_summary(self) -> str:
        """
        Generate session summary.

        Returns:
            Session summary text
        """
        summary_parts = [
            f"TASK LIFECYCLE SESSION SUMMARY [{self.current_state.session_id}]",
            f"Duration: {datetime.now().isoformat()}",
            f"Active Tasks: {len(self.current_state.active_tasks)}",
            f"Completed Tasks: {len(self.current_state.completed_tasks)}",
            f"Failed Tasks: {len(self.current_state.failed_tasks)}",
            f"Automation Enabled: {self.automation_enabled}"
        ]

        return '\n'.join(summary_parts)

    async def store_session_context(self, session_id: str, tasks: List[Dict[str, Any]]) -> None:
        """
        Store session initialization context.

        Args:
            session_id: Session ID
            tasks: Current tasks
        """
        context = f"TASK LIFECYCLE SESSION INIT [{session_id}]: {len(tasks)} tasks loaded, automation enabled"

        await self.mcp_client.store_memory(
            content=context,
            content_type="context",
            keywords=["task-lifecycle", "session-init", "automation", session_id[:8]]
        )

    async def store_session_completion(self, summary: str) -> None:
        """
        Store session completion summary.

        Args:
            summary: Session summary
        """
        await self.mcp_client.store_memory(
            content=summary,
            content_type="context",
            keywords=["task-lifecycle", "session-completion", "automation", "summary"]
        )

# Convenience functions for hook integration

async def handle_session_start_event(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle session start lifecycle event."""
    manager = TaskLifecycleManager()
    return await manager.handle_lifecycle_event(TaskLifecycleEvent.SESSION_START, event_data)

async def handle_tool_execution_event(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle tool execution lifecycle event."""
    manager = TaskLifecycleManager()
    return await manager.handle_lifecycle_event(TaskLifecycleEvent.TOOL_EXECUTION, event_data)

async def handle_session_end_event(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle session end lifecycle event."""
    manager = TaskLifecycleManager()
    return await manager.handle_lifecycle_event(TaskLifecycleEvent.SESSION_END, event_data)

async def main():
    """Main execution for testing."""
    manager = TaskLifecycleManager()

    # Test automation cycle
    print("🔄 Running task lifecycle automation cycle...")

    results = await manager.run_automation_cycle()

    print(f"✅ Automation cycle completed:")
    print(f"   📊 Status updates: {len(results.get('status_updates', []))}")
    print(f"   🎯 Actions taken: {len(results.get('actions_taken', []))}")

    for update in results.get('status_updates', [])[:3]:  # Show first 3
        print(f"   📋 {update.task_id}: {update.old_status} → {update.new_status}")

if __name__ == "__main__":
    asyncio.run(main())