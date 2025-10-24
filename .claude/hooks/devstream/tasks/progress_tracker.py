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
DevStream Progress Tracker - Task Progress durante Code Generation
Context7-compliant real-time task progress tracking durante development.
"""

import json
import sys
import os
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import DevStream utilities
sys.path.append(str(Path(__file__).parent.parent / 'utils'))
from common import DevStreamHookBase, get_project_context
from logger import get_devstream_logger
from mcp_client import get_mcp_client

class ProgressTracker(DevStreamHookBase):
    """
    Progress tracker per monitoring task advancement durante code generation.
    Integrato con PostToolUse hook per real-time progress updates.
    """

    def __init__(self):
        super().__init__('progress_tracker')
        self.structured_logger = get_devstream_logger('progress_tracker')
        self.mcp_client = get_mcp_client()
        self.start_time = time.time()

    async def track_progress(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_result: Dict[str, Any],
        session_id: str
    ) -> None:
        """
        Track task progress based on tool usage.

        Args:
            tool_name: Name of executed tool
            tool_input: Tool input parameters
            tool_result: Tool execution result
            session_id: Current session ID
        """
        self.structured_logger.log_hook_start(
            {"tool_name": tool_name, "session_id": session_id},
            {"phase": "progress_tracking"}
        )

        try:
            # Get active tasks for progress tracking
            active_tasks = await self.get_active_tasks()

            if not active_tasks:
                self.logger.debug("No active tasks to track progress for")
                return

            # Analyze tool usage for progress indicators
            progress_analysis = await self.analyze_progress_indicators(
                tool_name,
                tool_input,
                tool_result
            )

            # Update task progress for relevant tasks
            for task in active_tasks:
                task_progress = await self.calculate_task_progress(
                    task,
                    progress_analysis,
                    session_id
                )

                if task_progress['updated']:
                    await self.update_task_progress(task, task_progress)

            # Log performance metrics
            execution_time = (time.time() - self.start_time) * 1000
            self.structured_logger.log_performance_metrics(execution_time)

        except Exception as e:
            self.structured_logger.log_hook_error(e, {"tool_name": tool_name})
            raise

    async def get_active_tasks(self) -> List[Dict[str, Any]]:
        """
        Get active tasks for progress tracking.

        Returns:
            List of active tasks
        """
        try:
            # Get in-progress and active tasks from DevStream MCP
            tasks_response = await self.mcp_client.list_tasks(status="active")

            if not tasks_response:
                return []

            # Parse and return active tasks (simplified)
            # In real implementation, would parse actual MCP response structure
            return [
                {
                    "id": "hook-system-implementation",
                    "title": "Hook System Implementation",
                    "status": "active",
                    "priority": 9,
                    "phase": "Implementation",
                    "progress": 0.0,
                    "milestones": [
                        "Phase A: Configuration System",
                        "Phase B: Memory Integration",
                        "Phase C: Task Management",
                        "Phase D: Context Injection",
                        "Phase E: Testing & Deployment"
                    ]
                }
            ]

        except Exception as e:
            self.logger.warning(f"Failed to get active tasks: {e}")
            return []

    async def analyze_progress_indicators(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze tool usage per progress indicators.

        Args:
            tool_name: Tool name
            tool_input: Tool input
            tool_result: Tool result

        Returns:
            Progress analysis results
        """
        analysis = {
            "tool_name": tool_name,
            "progress_indicators": [],
            "implementation_type": "unknown",
            "success": not tool_result.get('error', False),
            "progress_weight": 0.0,
            "milestone_keywords": []
        }

        # Analyze different tool types for progress
        if tool_name in ['Edit', 'MultiEdit', 'Write']:
            analysis.update(await self.analyze_code_progress(tool_input, tool_result))
        elif tool_name == 'Bash':
            analysis.update(await self.analyze_bash_progress(tool_input, tool_result))
        elif tool_name.startswith('mcp__devstream__'):
            analysis.update(await self.analyze_mcp_progress(tool_name, tool_input, tool_result))
        elif tool_name in ['Read', 'Glob', 'Grep']:
            analysis.update(await self.analyze_research_progress(tool_name, tool_input, tool_result))

        return analysis

    async def analyze_code_progress(
        self,
        tool_input: Dict[str, Any],
        tool_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze code editing tools for progress.

        Args:
            tool_input: Tool input
            tool_result: Tool result

        Returns:
            Code progress analysis
        """
        analysis = {
            "implementation_type": "code_development",
            "progress_indicators": [],
            "progress_weight": 0.0,
            "milestone_keywords": []
        }

        file_path = tool_input.get('file_path', '')
        content = tool_input.get('new_string', tool_input.get('content', ''))

        # Analyze file path for progress indicators
        if 'hook' in file_path.lower():
            analysis["milestone_keywords"].append('hook-implementation')
            analysis["progress_weight"] += 0.2

        if 'memory' in file_path.lower():
            analysis["milestone_keywords"].append('memory-integration')
            analysis["progress_weight"] += 0.2

        if 'task' in file_path.lower():
            analysis["milestone_keywords"].append('task-management')
            analysis["progress_weight"] += 0.2

        if 'test' in file_path.lower():
            analysis["milestone_keywords"].append('testing')
            analysis["progress_weight"] += 0.3

        # Analyze content for implementation completeness
        if content:
            lines_count = len(content.split('\n'))
            if lines_count > 50:
                analysis["progress_indicators"].append('substantial_implementation')
                analysis["progress_weight"] += 0.1

            if 'class ' in content and 'def ' in content:
                analysis["progress_indicators"].append('complete_class_implementation')
                analysis["progress_weight"] += 0.15

            if '#!/usr/bin/env -S uv run --script' in content:
                analysis["progress_indicators"].append('context7_compliant')
                analysis["progress_weight"] += 0.05

        return analysis

    async def analyze_bash_progress(
        self,
        tool_input: Dict[str, Any],
        tool_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze bash commands for progress.

        Args:
            tool_input: Tool input
            tool_result: Tool result

        Returns:
            Bash progress analysis
        """
        analysis = {
            "implementation_type": "system_operation",
            "progress_indicators": [],
            "progress_weight": 0.0,
            "milestone_keywords": []
        }

        command = tool_input.get('command', '')

        # Test execution indicates progress
        if 'pytest' in command or 'test' in command:
            analysis["milestone_keywords"].append('testing')
            analysis["progress_weight"] = 0.3
            analysis["progress_indicators"].append('test_execution')

        # Build/setup operations
        if any(keyword in command for keyword in ['build', 'install', 'setup']):
            analysis["progress_indicators"].append('setup_operation')
            analysis["progress_weight"] = 0.1

        # Hook execution
        if 'uv run' in command and 'hook' in command:
            analysis["milestone_keywords"].append('hook-testing')
            analysis["progress_weight"] = 0.2

        return analysis

    async def analyze_mcp_progress(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze MCP operations for progress.

        Args:
            tool_name: MCP tool name
            tool_input: Tool input
            tool_result: Tool result

        Returns:
            MCP progress analysis
        """
        analysis = {
            "implementation_type": "devstream_integration",
            "progress_indicators": [],
            "progress_weight": 0.05,  # MCP operations are supporting work
            "milestone_keywords": []
        }

        operation = tool_name.replace('mcp__devstream__', '')

        if operation == 'store_memory':
            analysis["progress_indicators"].append('memory_integration_active')
            analysis["milestone_keywords"].append('memory-integration')
        elif operation == 'create_task':
            analysis["progress_indicators"].append('task_management_active')
            analysis["milestone_keywords"].append('task-management')

        return analysis

    async def analyze_research_progress(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze research/search tools for progress.

        Args:
            tool_name: Tool name
            tool_input: Tool input
            tool_result: Tool result

        Returns:
            Research progress analysis
        """
        analysis = {
            "implementation_type": "research_development",
            "progress_indicators": [],
            "progress_weight": 0.05,  # Research is preparatory work
            "milestone_keywords": []
        }

        # Research indicates preparation for implementation
        analysis["progress_indicators"].append('research_activity')

        # Check for Context7 research
        if 'context7' in str(tool_input).lower() or 'context7' in str(tool_result).lower():
            analysis["milestone_keywords"].append('context7-research')
            analysis["progress_weight"] = 0.1

        return analysis

    async def calculate_task_progress(
        self,
        task: Dict[str, Any],
        progress_analysis: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """
        Calculate progress update per specific task.

        Args:
            task: Task information
            progress_analysis: Progress analysis from tool
            session_id: Session ID

        Returns:
            Task progress calculation results
        """
        task_progress = {
            "updated": False,
            "progress_delta": 0.0,
            "new_progress": task.get('progress', 0.0),
            "milestone_updates": [],
            "progress_note": ""
        }

        # Check if tool activity is relevant to this task
        task_keywords = self.extract_task_keywords(task)
        analysis_keywords = progress_analysis.get('milestone_keywords', [])

        # Calculate relevance score
        relevance_score = self.calculate_relevance(task_keywords, analysis_keywords)

        if relevance_score < 0.3:
            return task_progress  # Not relevant to this task

        # Calculate progress delta based on tool impact
        base_progress = progress_analysis.get('progress_weight', 0.0)
        weighted_progress = base_progress * relevance_score

        if weighted_progress > 0.01:  # Minimum threshold for progress update
            task_progress["updated"] = True
            task_progress["progress_delta"] = weighted_progress

            current_progress = task.get('progress', 0.0)
            new_progress = min(current_progress + weighted_progress, 1.0)
            task_progress["new_progress"] = new_progress

            # Generate progress note
            task_progress["progress_note"] = self.generate_progress_note(
                progress_analysis,
                weighted_progress
            )

            # Check for milestone completion
            milestone_updates = self.check_milestone_completion(
                task,
                analysis_keywords,
                new_progress
            )
            task_progress["milestone_updates"] = milestone_updates

        return task_progress

    def extract_task_keywords(self, task: Dict[str, Any]) -> List[str]:
        """
        Extract keywords from task information.

        Args:
            task: Task information

        Returns:
            List of task keywords
        """
        keywords = []

        title = task.get('title', '').lower()
        keywords.extend(title.split())

        phase = task.get('phase', '').lower()
        keywords.extend(phase.split())

        # Add milestones as keywords
        milestones = task.get('milestones', [])
        for milestone in milestones:
            keywords.extend(milestone.lower().split())

        return list(set([k for k in keywords if len(k) > 2]))

    def calculate_relevance(
        self,
        task_keywords: List[str],
        analysis_keywords: List[str]
    ) -> float:
        """
        Calculate relevance between task and analysis keywords.

        Args:
            task_keywords: Task keywords
            analysis_keywords: Analysis keywords

        Returns:
            Relevance score (0.0-1.0)
        """
        if not task_keywords or not analysis_keywords:
            return 0.0

        # Calculate keyword overlap
        task_set = set(task_keywords)
        analysis_set = set(analysis_keywords)

        overlap = len(task_set.intersection(analysis_set))
        total_unique = len(task_set.union(analysis_set))

        if total_unique == 0:
            return 0.0

        return overlap / total_unique

    def generate_progress_note(
        self,
        progress_analysis: Dict[str, Any],
        progress_delta: float
    ) -> str:
        """
        Generate human-readable progress note.

        Args:
            progress_analysis: Progress analysis
            progress_delta: Progress increment

        Returns:
            Progress note string
        """
        tool_name = progress_analysis.get('tool_name', 'unknown')
        impl_type = progress_analysis.get('implementation_type', 'development')
        indicators = progress_analysis.get('progress_indicators', [])

        note_parts = [
            f"{tool_name} operation completed",
            f"({impl_type})",
            f"Progress: +{progress_delta:.1%}"
        ]

        if indicators:
            note_parts.append(f"Indicators: {', '.join(indicators)}")

        return " | ".join(note_parts)

    def check_milestone_completion(
        self,
        task: Dict[str, Any],
        analysis_keywords: List[str],
        new_progress: float
    ) -> List[Dict[str, str]]:
        """
        Check for milestone completion based on progress.

        Args:
            task: Task information
            analysis_keywords: Analysis keywords
            new_progress: New progress value

        Returns:
            List of milestone updates
        """
        milestone_updates = []

        milestones = task.get('milestones', [])

        # Simple milestone completion logic based on progress thresholds
        milestone_thresholds = {
            0.2: 0,  # Phase A
            0.4: 1,  # Phase B
            0.6: 2,  # Phase C
            0.8: 3,  # Phase D
            1.0: 4   # Phase E
        }

        for threshold, milestone_idx in milestone_thresholds.items():
            if new_progress >= threshold and milestone_idx < len(milestones):
                milestone_name = milestones[milestone_idx]

                # Check if milestone keywords match analysis
                milestone_lower = milestone_name.lower()
                keyword_match = any(
                    keyword in milestone_lower
                    for keyword in analysis_keywords
                )

                if keyword_match:
                    milestone_updates.append({
                        "milestone": milestone_name,
                        "status": "completed" if new_progress >= threshold else "in_progress",
                        "progress": f"{new_progress:.1%}"
                    })

        return milestone_updates

    async def update_task_progress(
        self,
        task: Dict[str, Any],
        task_progress: Dict[str, Any]
    ) -> None:
        """
        Update task progress via MCP.

        Args:
            task: Task information
            task_progress: Progress update information
        """
        try:
            task_id = task["id"]
            progress_note = task_progress["progress_note"]

            # Update task with progress note
            update_result = await self.mcp_client.update_task(
                task_id=task_id,
                status="active",  # Keep as active until completion
                notes=f"Progress update: {progress_note}"
            )

            if update_result:
                self.logger.info(f"Updated progress for task: {task['title']}")

                # Log task progress update
                self.structured_logger.log_task_operation(
                    operation="progress_update",
                    task_id=task_id,
                    task_type="progress_tracking",
                    status="active",
                    priority=task.get('priority', 5)
                )

                # Store progress update in memory
                await self.store_progress_update(task, task_progress)

            else:
                self.logger.warning(f"Failed to update progress for task: {task['title']}")

        except Exception as e:
            self.logger.error(f"Error updating task progress: {e}")

    async def store_progress_update(
        self,
        task: Dict[str, Any],
        task_progress: Dict[str, Any]
    ) -> None:
        """
        Store progress update in memory.

        Args:
            task: Task information
            task_progress: Progress update
        """
        progress_content = (
            f"TASK PROGRESS UPDATE: {task['title']}\n"
            f"Progress Delta: +{task_progress['progress_delta']:.1%}\n"
            f"New Progress: {task_progress['new_progress']:.1%}\n"
            f"Note: {task_progress['progress_note']}\n"
            f"Timestamp: {datetime.now().isoformat()}"
        )

        milestone_updates = task_progress.get('milestone_updates', [])
        if milestone_updates:
            progress_content += "\nMilestone Updates:\n"
            for milestone in milestone_updates:
                progress_content += f"- {milestone['milestone']}: {milestone['status']}\n"

        await self.mcp_client.store_memory(
            content=progress_content,
            content_type="context",
            keywords=["task-progress", "milestone-update", task["id"][:8]],
            session_id=None
        )

        # Log memory operation
        self.structured_logger.log_memory_operation(
            operation="store",
            content_type="context",
            content_size=len(progress_content),
            keywords=["task-progress"]
        )

# Main execution for standalone testing
if __name__ == "__main__":
    # Test progress tracker
    import asyncio

    async def test_progress_tracker():
        tracker = ProgressTracker()

        # Context7 Pattern: Dynamic file path resolution for testing
        def get_test_file_path():
            """Get test file path using Context7 patterns."""
            # Priority 1: DEVSTREAM_ROOT environment variable
            devstream_root = os.getenv("DEVSTREAM_ROOT")

            # Priority 2: Current working directory
            if devstream_root is None:
                devstream_root = os.getcwd()

            return Path(devstream_root) / ".claude" / "hooks" / "devstream" / "memory" / "user_prompt_submit.py"

        # Simulate tool usage with dynamic path
        await tracker.track_progress(
            tool_name="Edit",
            tool_input={
                "file_path": str(get_test_file_path()),
                "new_string": "# Test implementation with 100+ lines of code..."
            },
            tool_result={"success": True},
            session_id="test-session"
        )

        print("✅ Progress tracker test completed")

    asyncio.run(test_progress_tracker())