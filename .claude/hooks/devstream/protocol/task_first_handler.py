#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "aiofiles>=23.0.0",
#     "structlog>=23.0.0",
#     "python-dotenv>=1.0.0",
#     "pyinquirer>=1.0.3",
# ]
# ///

"""
Task First Handler - Mandatory STEP 1 Enforcement for DevStream Protocol

Phase 1 Component: Enforces task creation as STEP 1 (before DISCUSSION) with blocking validation.
Ensures Claude Code creates tasks at the beginning of the workflow, not at step 5.

Core Problem Solved:
- Claude Code (especially GLM4.6) ignores CLAUDE.md rules and creates tasks at step 5 instead of step 1
- Task creation MUST happen before DISCUSSION step with blocking validation gates
- Complexity-based triggering analysis to determine when task creation is mandatory

Architecture Principles:
- Complexity-based triggering analysis (duration > 15min, code implementation, architectural decisions)
- MCP integration with devstream_create_task for automatic task creation
- Interactive confirmation requirements using PyInquirer
- Integration with existing hook system (PreToolUse, PostToolUse)
- Graceful degradation when MCP services unavailable

Trigger Conditions (from CLAUDE.md):
1. Estimated task duration > 15 minutes
2. Task requires code implementation (Write/Edit tools)
3. Task requires architectural decisions
4. Task involves multiple files or components
5. Task requires Context7 research

Task Creation Workflow:
1. Analyze task complexity and determine if task creation is mandatory
2. Extract task information from user prompt and tool context
3. Create comprehensive task description with acceptance criteria
4. Present task creation confirmation to user
5. Create task via MCP devstream_create_task
6. Update protocol state with task ID
7. Log task creation to DevStream memory
"""

import asyncio
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from logger import get_devstream_logger
from mcp_client import get_mcp_client
from enforcement_gate import EnforcementGate, EnforcementContext, EnforcementDecision
from protocol_state_manager import ProtocolStateManager, ProtocolStep, get_protocol_manager

# PyInquirer import with graceful degradation
try:
    from PyInquirer import prompt, Separator
    PYINQUIRER_AVAILABLE = True
except ImportError:
    PYINQUIRER_AVAILABLE = False

logger = get_devstream_logger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"      # < 15 min, single file, no architecture
    MODERATE = "moderate"  # 15-60 min, few files, some complexity
    COMPLEX = "complex"    # > 60 min, multiple files, architecture
    CRITICAL = "critical"  # High impact, system-wide changes


@dataclass
class TaskInfo:
    """Extracted task information for creation."""
    title: str
    description: str
    task_type: str  # analysis, coding, documentation, testing, review, research
    priority: int   # 1-10
    estimated_duration: int  # minutes
    complexity: TaskComplexity
    involves_code: bool
    involves_architecture: bool
    requires_context7: bool
    file_count: int
    trigger_reasons: List[str]
    acceptance_criteria: List[str]


class TaskFirstHandler:
    """
    Handles mandatory task creation as STEP 1 of DevStream protocol.

    Enforces task creation before any discussion or analysis begins,
    ensuring proper task lifecycle management from the start.
    """

    def __init__(self, memory_client=None):
        """
        Initialize task first handler.

        Args:
            memory_client: Optional MCP memory client
        """
        self.memory_client = memory_client or get_mcp_client()
        self.protocol_manager = get_protocol_manager()
        self.enforcement_gate = EnforcementGate()
        self.tasks_created = 0

        logger.logger.info(
            "task_first_handler_initialized",
            memory_client_available=self.memory_client is not None,
            pyinquirer_available=PYINQUIRER_AVAILABLE
        )

    async def should_create_task(
        self,
        user_prompt: str,
        tool_name: Optional[str] = None,
        tool_input: Optional[Dict[str, Any]] = None,
        current_step: Optional[ProtocolStep] = None
    ) -> Tuple[bool, TaskInfo]:
        """
        Analyze if task creation should be mandatory.

        Args:
            user_prompt: User's original prompt
            tool_name: Optional tool being used
            tool_input: Optional tool input parameters
            current_step: Current protocol step

        Returns:
            Tuple of (should_create, task_info)
        """
        # Extract task information
        task_info = await self._extract_task_info(user_prompt, tool_name, tool_input)

        # Determine if task creation is mandatory
        should_create = self._evaluate_task_necessity(task_info, current_step)

        logger.logger.info(
            "task_creation_analysis",
            should_create=should_create,
            task_title=task_info.title[:50],
            complexity=task_info.complexity.value,
            duration=task_info.estimated_duration,
            trigger_reasons=len(task_info.trigger_reasons)
        )

        return should_create, task_info

    async def enforce_task_creation(
        self,
        user_prompt: str,
        tool_name: Optional[str] = None,
        tool_input: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Enforce mandatory task creation with user interaction.

        Args:
            user_prompt: User's original prompt
            tool_name: Optional tool being used
            tool_input: Optional tool input parameters
            session_id: Optional session ID

        Returns:
            Tuple of (success, task_id)
        """
        try:
            # Get current protocol state
            current_state = await self.protocol_manager.get_current_state()
            if session_id is None:
                session_id = current_state.session_id

            # Analyze if task creation is needed
            should_create, task_info = await self.should_create_task(
                user_prompt, tool_name, tool_input, current_state.protocol_step
            )

            if not should_create:
                logger.logger.debug(
                    "task_creation_not_required",
                    reason="simple_task",
                    task_title=task_info.title[:50]
                )
                return True, None  # Success, but no task needed

            # Check if task already exists for this session
            if current_state.task_id:
                logger.logger.info(
                    "task_already_exists",
                    task_id=current_state.task_id,
                    session_id=session_id
                )
                return True, current_state.task_id

            # Show enforcement gate for task creation
            enforcement_context = EnforcementContext(
                task_description=f"Create task: {task_info.title}",
                estimated_duration=task_info.estimated_duration,
                complexity_score=self._complexity_to_score(task_info.complexity),
                involves_code=task_info.involves_code,
                involves_architecture=task_info.involves_architecture,
                requires_context7=task_info.requires_context7,
                trigger_reasons=task_info.trigger_reasons,
                session_id=session_id,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

            decision = await self.enforcement_gate.show_enforcement_gate(
                enforcement_context,
                self.memory_client
            )

            if decision == EnforcementDecision.CANCEL:
                logger.logger.info("user_cancelled_task_creation")
                return False, None

            if decision == EnforcementDecision.OVERRIDE:
                logger.logger.info(
                    "user_overrode_task_creation",
                    session_id=session_id,
                    task_title=task_info.title[:50]
                )
                return True, None  # Allow continuation without task

            # User chose PROTOCOL - create task
            task_id = await self._create_task(task_info, session_id)

            if task_id:
                # Update protocol state with task ID
                await self.protocol_manager.advance_step(
                    current_state,
                    ProtocolStep.DISCUSSION,  # Move to discussion step
                    task_id=task_id,
                    metadata_updates={
                        "task_created_at": datetime.now(timezone.utc).isoformat(),
                        "task_title": task_info.title,
                        "task_complexity": task_info.complexity.value
                    }
                )

                # Log task creation to memory
                await self._log_task_creation(task_info, task_id, session_id)

                self.tasks_created += 1

                logger.logger.info(
                    "task_creation_completed",
                    task_id=task_id,
                    task_title=task_info.title[:50],
                    session_id=session_id
                )

                return True, task_id
            else:
                logger.logger.error("task_creation_failed")
                return False, None

        except Exception as e:
            logger.logger.error(
                "task_creation_enforcement_error",
                error=str(e),
                error_type=type(e).__name__
            )
            return False, None

    async def _extract_task_info(
        self,
        user_prompt: str,
        tool_name: Optional[str],
        tool_input: Optional[Dict[str, Any]]
    ) -> TaskInfo:
        """Extract task information from user prompt and context."""

        # Analyze prompt for task characteristics
        prompt_lower = user_prompt.lower()

        # Estimate duration based on prompt complexity
        duration_indicators = {
            "quick fix": 5, "small change": 10, "typo": 5, "minor": 15,
            "implement": 45, "create": 60, "build": 90, "develop": 120,
            "refactor": 180, "migration": 240, "architecture": 300,
            "integration": 150, "testing": 90, "documentation": 60
        }

        estimated_duration = 15  # Default
        for indicator, duration in duration_indicators.items():
            if indicator in prompt_lower:
                estimated_duration = duration
                break

        # Determine complexity
        complexity = TaskComplexity.SIMPLE
        if estimated_duration > 240:
            complexity = TaskComplexity.CRITICAL
        elif estimated_duration > 60:
            complexity = TaskComplexity.COMPLEX
        elif estimated_duration > 15:
            complexity = TaskComplexity.MODERATE

        # Determine task type
        task_type = "analysis"  # Default
        if any(word in prompt_lower for word in ["implement", "code", "develop", "build"]):
            task_type = "coding"
        elif any(word in prompt_lower for word in ["test", "testing", "validate"]):
            task_type = "testing"
        elif any(word in prompt_lower for word in ["document", "documentation", "readme"]):
            task_type = "documentation"
        elif any(word in prompt_lower for word in ["review", "audit", "check"]):
            task_type = "review"
        elif any(word in prompt_lower for word in ["research", "investigate", "explore"]):
            task_type = "research"

        # Extract file count from tool input
        file_count = 1
        if tool_input and "file_path" in tool_input:
            file_count = 1
        elif "multiple files" in prompt_lower or "several files" in prompt_lower:
            file_count = 3

        # Determine flags
        involves_code = tool_name in ["Write", "Edit", "MultiEdit"] or task_type == "coding"
        involves_architecture = any(word in prompt_lower for word in [
            "architecture", "design", "system", "integration", "migration"
        ])
        requires_context7 = any(word in prompt_lower for word in [
            "research", "best practices", "library", "framework", "api"
        ])

        # Generate trigger reasons
        trigger_reasons = []
        if estimated_duration > 15:
            trigger_reasons.append(f"Duration > 15min ({estimated_duration}min)")
        if involves_code:
            trigger_reasons.append("Code implementation required")
        if involves_architecture:
            trigger_reasons.append("Architectural decisions involved")
        if file_count > 1:
            trigger_reasons.append(f"Multiple files ({file_count})")
        if requires_context7:
            trigger_reasons.append("Context7 research required")

        # Create title and description
        title = self._extract_task_title(user_prompt)
        description = user_prompt

        # Determine priority (1-10)
        priority = 5  # Default
        if any(word in prompt_lower for word in ["urgent", "critical", "asap"]):
            priority = 9
        elif any(word in prompt_lower for word in ["important", "priority"]):
            priority = 7
        elif any(word in prompt_lower for word in ["low priority", "later"]):
            priority = 3

        # Generate acceptance criteria
        acceptance_criteria = self._generate_acceptance_criteria(
            task_type, title, prompt_lower
        )

        return TaskInfo(
            title=title,
            description=description,
            task_type=task_type,
            priority=priority,
            estimated_duration=estimated_duration,
            complexity=complexity,
            involves_code=involves_code,
            involves_architecture=involves_architecture,
            requires_context7=requires_context7,
            file_count=file_count,
            trigger_reasons=trigger_reasons,
            acceptance_criteria=acceptance_criteria
        )

    def _extract_task_title(self, user_prompt: str) -> str:
        """Extract concise task title from user prompt."""
        # Remove common prefixes and make title concise
        prompt = user_prompt.strip()

        # Remove common prefixes
        prefixes = [
            "please ", "can you ", "could you ", "i need you to ",
            "help me ", "we need to ", "let's ", "i want to "
        ]

        for prefix in prefixes:
            if prompt.lower().startswith(prefix):
                prompt = prompt[len(prefix):]

        # Capitalize first letter and limit length
        title = prompt.strip().capitalize()

        # Truncate if too long
        if len(title) > 100:
            title = title[:97] + "..."

        return title

    def _generate_acceptance_criteria(
        self,
        task_type: str,
        title: str,
        prompt_lower: str
    ) -> List[str]:
        """Generate appropriate acceptance criteria for the task type."""

        base_criteria = [
            f"Task '{title}' completed successfully",
            "All requirements fulfilled according to specifications"
        ]

        type_specific = {
            "coding": [
                "Code implemented with full type hints and docstrings",
                "All code follows project coding standards",
                "Error handling implemented appropriately",
                "Code tested and validated"
            ],
            "testing": [
                "Comprehensive test suite created",
                "Test coverage achieved (95%+ target)",
                "All test cases pass",
                "Edge cases and error scenarios covered"
            ],
            "documentation": [
                "Documentation is clear and comprehensive",
                "Examples and usage instructions included",
                "Documentation follows project standards",
                "Technical accuracy verified"
            ],
            "research": [
                "Research findings documented and organized",
                "Multiple sources evaluated and compared",
                "Recommendations provided with justification",
                "Relevant best practices identified"
            ],
            "review": [
                "Comprehensive review completed",
                "All issues and improvements identified",
                "Recommendations documented",
                "Review findings communicated clearly"
            ]
        }

        criteria = base_criteria + type_specific.get(task_type, [])

        # Add context-specific criteria
        if "security" in prompt_lower:
            criteria.append("Security considerations addressed")
        if "performance" in prompt_lower:
            criteria.append("Performance requirements met")
        if "api" in prompt_lower:
            criteria.append("API endpoints properly documented and tested")

        return criteria[:6]  # Limit to 6 criteria

    def _evaluate_task_necessity(
        self,
        task_info: TaskInfo,
        current_step: Optional[ProtocolStep]
    ) -> bool:
        """Evaluate if task creation is mandatory based on complexity and context."""

        # If already past DISCUSSION step, don't require new task
        if current_step and current_step.value > ProtocolStep.DISCUSSION.value:
            return False

        # Mandatory conditions (from CLAUDE.md)
        mandatory_conditions = [
            task_info.estimated_duration > 15,
            task_info.involves_code,
            task_info.involves_architecture,
            task_info.file_count > 1,
            task_info.requires_context7,
            task_info.complexity in [TaskComplexity.COMPLEX, TaskComplexity.CRITICAL]
        ]

        return any(mandatory_conditions)

    def _complexity_to_score(self, complexity: TaskComplexity) -> float:
        """Convert complexity enum to score (0.0-1.0)."""
        mapping = {
            TaskComplexity.SIMPLE: 0.2,
            TaskComplexity.MODERATE: 0.5,
            TaskComplexity.COMPLEX: 0.8,
            TaskComplexity.CRITICAL: 1.0
        }
        return mapping.get(complexity, 0.5)

    async def _create_task(self, task_info: TaskInfo, session_id: str) -> Optional[str]:
        """
        Create task via MCP devstream_create_task with circuit breaker pattern.

        Implements graceful fallback with exponential backoff when MCP services
        are unavailable, ensuring session continuity.
        """
        import os
        from pathlib import Path

        # Circuit breaker configuration from environment
        max_retries = int(os.getenv("DEVSTREAM_MCP_CIRCUIT_BREAKER_RETRIES", "3"))
        backoff_factor = float(os.getenv("DEVSTREAM_MCP_CIRCUIT_BREAKER_BACKOFF_FACTOR", "2"))
        initial_delay = float(os.getenv("DEVSTREAM_MCP_CIRCUIT_BREAKER_INITIAL_DELAY", "1"))

        # Fallback log file
        fallback_log = Path.home() / ".claude" / "logs" / "protocol_decisions.jsonl"
        fallback_log.parent.mkdir(parents=True, exist_ok=True)

        attempt = 0
        last_error = None

        while attempt < max_retries:
            attempt += 1
            try:
                # Attempt MCP task creation
                result = await self.memory_client.create_task(
                    title=task_info.title,
                    description=task_info.description,
                    task_type=task_info.task_type,
                    priority=task_info.priority,
                    phase_name="Core Engine & Infrastructure",  # Default phase
                    project="DevStream"
                )

                if result and result.get("task_id"):
                    logger.logger.info(
                        "mcp_task_created",
                        task_id=result["task_id"],
                        title=task_info.title[:50],
                        attempt=attempt
                    )
                    return result["task_id"]
                else:
                    logger.logger.warning(
                        "mcp_task_creation_failed",
                        result=result,
                        attempt=attempt
                    )
                    last_error = f"MCP returned invalid result: {result}"

            except Exception as e:
                last_error = str(e)
                logger.logger.warning(
                    "mcp_task_creation_attempt_failed",
                    attempt=attempt,
                    max_retries=max_retries,
                    error=last_error,
                    error_type=type(e).__name__
                )

            # Exponential backoff: delay = initial_delay * (backoff_factor ^ (attempt-1))
            if attempt < max_retries:
                delay = initial_delay * (backoff_factor ** (attempt - 1))
                logger.logger.info(
                    "mcp_circuit_backoff",
                    attempt=attempt,
                    delay_seconds=delay,
                    next_attempt=attempt + 1
                )
                await asyncio.sleep(delay)

        # All retries failed - implement graceful fallback
        logger.logger.error(
            "mcp_circuit_breaker_tripped",
            max_retries=max_retries,
            last_error=last_error,
            task_title=task_info.title[:50],
            session_id=session_id
        )

        # Fallback to local logging
        await self._fallback_task_logging(task_info, session_id, fallback_log, last_error)

        # Return fallback task ID (timestamp-based)
        fallback_task_id = f"fallback-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

        logger.logger.info(
            "mcp_fallback_task_created",
            fallback_task_id=fallback_task_id,
            task_title=task_info.title[:50],
            degraded_mode=True
        )

        return fallback_task_id

    async def _fallback_task_logging(
        self,
        task_info: TaskInfo,
        session_id: str,
        fallback_log: Path,
        error: str
    ) -> None:
        """
        Log task creation to fallback file when MCP is unavailable.

        Creates structured JSONL log entry for later recovery and audit.
        """
        try:
            import json

            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "task_creation_fallback",
                "session_id": session_id,
                "task_info": {
                    "title": task_info.title,
                    "description": task_info.description,
                    "task_type": task_info.task_type,
                    "priority": task_info.priority,
                    "estimated_duration": task_info.estimated_duration,
                    "complexity": task_info.complexity.value
                },
                "mcp_error": error,
                "fallback_reason": "MCP circuit breaker tripped",
                "degraded_mode": True
            }

            # Ensure directory exists
            fallback_log.parent.mkdir(parents=True, exist_ok=True)

            # Append to fallback log file
            with open(fallback_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            logger.logger.info(
                "fallback_task_logged",
                log_file=str(fallback_log),
                task_title=task_info.title[:50]
            )

        except Exception as e:
            logger.logger.error(
                "fallback_logging_failed",
                error=str(e),
                fallback_log=str(fallback_log)
            )

    async def _log_task_creation(
        self,
        task_info: TaskInfo,
        task_id: str,
        session_id: str
    ) -> None:
        """Log task creation to DevStream memory."""
        try:
            content = (
                f"Task Created (STEP 1): {task_info.title}\n"
                f"Task ID: {task_id}\n"
                f"Session ID: {session_id}\n"
                f"Type: {task_info.task_type} | Priority: {task_info.priority}/10\n"
                f"Duration: {task_info.estimated_duration} minutes | "
                f"Complexity: {task_info.complexity.value}\n"
                f"Description: {task_info.description}\n"
                f"Acceptance Criteria:\n" +
                "\n".join(f"  • {criteria}" for criteria in task_info.acceptance_criteria)
            )

            keywords = [
                "task-created",
                "step-1",
                task_info.task_type,
                session_id,
                task_id
            ]

            await self.memory_client.store_memory(
                content=content,
                content_type="decision",
                keywords=keywords
            )

            logger.logger.debug("task_creation_logged_to_memory", task_id=task_id)

        except Exception as e:
            logger.logger.error(
                "task_creation_logging_failed",
                error=str(e)
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get task first handler statistics."""
        return {
            "tasks_created": self.tasks_created,
            "enforcement_gate_active": True,
            "mcp_integration_available": self.memory_client is not None,
            "pyinquirer_available": PYINQUIRER_AVAILABLE
        }


# Example usage and testing
if __name__ == "__main__":
    async def test_task_first_handler():
        """Test task first handler functionality."""

        # Mock memory client for testing
        class MockMemoryClient:
            async def create_task(self, title, description, task_type, priority, phase_name, project):
                return {"task_id": f"task-{uuid.uuid4().hex[:8]}"}

            async def store_memory(self, content, content_type, keywords):
                pass

        handler = TaskFirstHandler(memory_client=MockMemoryClient())

        # Test 1: Simple task (no enforcement)
        print("🧪 Test 1: Simple task analysis")
        should_create, task_info = await handler.should_create_task(
            "Fix typo in README file",
            tool_name="Edit"
        )
        assert should_create == False
        assert task_info.complexity == TaskComplexity.SIMPLE
        print(f"✅ Should create task: {should_create}")
        print(f"   Task complexity: {task_info.complexity.value}")

        # Test 2: Complex task (enforcement required)
        print("\n🧪 Test 2: Complex task analysis")
        should_create, task_info = await handler.should_create_task(
            "Implement comprehensive user authentication system with JWT tokens",
            tool_name="Write"
        )
        assert should_create == True
        assert task_info.complexity in [TaskComplexity.COMPLEX, TaskComplexity.CRITICAL]
        print(f"✅ Should create task: {should_create}")
        print(f"   Task complexity: {task_info.complexity.value}")
        print(f"   Estimated duration: {task_info.estimated_duration} minutes")
        print(f"   Trigger reasons: {task_info.trigger_reasons}")

        # Test 3: Task information extraction
        print("\n🧪 Test 3: Task information extraction")
        prompt = "Build a REST API for user management with FastAPI"
        should_create, task_info = await handler.should_create_task(prompt)
        assert task_info.task_type == "coding"
        assert "fastapi" in task_info.description.lower()
        assert len(task_info.acceptance_criteria) > 2
        print(f"✅ Task type: {task_info.task_type}")
        print(f"   Title: {task_info.title}")
        print(f"   Acceptance criteria: {len(task_info.acceptance_criteria)} items")

        # Test 4: Statistics
        print("\n🧪 Test 4: Handler statistics")
        stats = handler.get_statistics()
        assert "tasks_created" in stats
        assert "enforcement_gate_active" in stats
        print(f"✅ Statistics: {stats}")

        print("\n🎉 All task first handler tests PASSED")
        print("\n💡 Note: Full enforcement testing requires manual execution")
        print("   Use in real hook execution to see interactive prompts and MCP integration")

    # Run tests
    asyncio.run(test_task_first_handler())