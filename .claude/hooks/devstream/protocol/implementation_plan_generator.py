#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "aiohttp>=3.8.0",
#     "aiofiles>=23.0.0",
#     "structlog>=23.0.0",
#     "python-dotenv>=1.0.0",
#     "pyinquirer>=1.0.3",
# ]
# ///

"""
Implementation Plan Generator - Automatic Step 4 (PLANNING) Automation

DevStream Protocol v2.2.0 - Strategic Choice Gate Integration

Core Features:
1. Automatic plan generation at Step 4 (PLANNING) based on task context
2. Model-specific template selection (GLM-4.6 vs Sonnet 4.5)
3. Strategic Choice Gate for model selection after Step 5 (APPROVAL)
4. Dual storage: DevStream DB + markdown files (docs/development/plan/)
5. GLM-4.6 handoff prompt generation for Sonnet→GLM workflow
6. Context7 research findings integration into plan templates

Architecture Principles:
- Template-based plan generation (research-backed patterns)
- Interactive Strategic Choice Gate using PyInquirer
- MCP integration with devstream_create_implementation_plan
- File naming: piano_[task-slug].md (from CLAUDE.md rules)
- Metadata tracking: complexity, duration, libraries, research findings

Strategic Choice Gate Workflow:
1. User reaches Step 5 (APPROVAL) and approves plan
2. Gate presents two options:
   - Option A: Continue with Sonnet 4.5 (architectural work, complex reasoning)
   - Option B: Handoff to GLM-4.6 (precise execution, cost savings)
3. If Option B selected:
   - Generate GLM handoff prompt with complete context
   - Save plan + handoff to DB and filesystem
   - Display handoff instructions for manual session switch
4. If Option A selected:
   - Save Sonnet plan to DB and filesystem
   - Continue with Sonnet implementation

Template Variables:
{{task_title}}, {{task_id}}, {{phase_name}}, {{priority}}, {{estimated_hours}},
{{component_list}}, {{component_name}}, {{todowrite_tasks}}, {{context7_research}},
{{context7_libraries}}, {{research_findings}}, {{code_example}}, {{use_case}},
{{files_to_modify}}, {{files_to_create}}, {{dependencies}}, {{performance_target}}
"""

import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from logger import get_devstream_logger
from mcp_client import get_mcp_client

# Protocol components
try:
    from protocol_state_manager import ProtocolStateManager, ProtocolStep
    PROTOCOL_AVAILABLE = True
except ImportError:
    PROTOCOL_AVAILABLE = False

# PyInquirer import with graceful degradation
try:
    from PyInquirer import prompt, Separator
    PYINQUIRER_AVAILABLE = True
except ImportError:
    PYINQUIRER_AVAILABLE = False

logger = get_devstream_logger(__name__)


class ModelChoice(Enum):
    """Strategic model choice for implementation."""
    SONNET_45 = "sonnet-4.5"
    GLM_46 = "glm-4.6"


@dataclass
class PlanContext:
    """Context for implementation plan generation."""
    task_id: str
    task_title: str
    task_description: str
    task_type: str
    priority: int
    phase_name: str
    estimated_duration: int  # minutes
    complexity_score: float  # 0.0-1.0

    # Context7 research findings
    context7_libraries: List[str]
    research_findings: str
    code_examples: Dict[str, str]

    # Implementation details
    files_to_modify: List[str]
    files_to_create: List[str]
    dependencies: List[str]
    performance_targets: str

    # TodoWrite tasks
    todowrite_tasks: List[Dict[str, str]]

    # Metadata
    session_id: str
    timestamp: str


class ImplementationPlanGenerator:
    """
    Automatic implementation plan generator for DevStream Protocol v2.2.0.

    Generates model-specific plans with Strategic Choice Gate integration.
    """

    def __init__(self, memory_client=None):
        """
        Initialize plan generator.

        Args:
            memory_client: Optional MCP memory client
        """
        self.memory_client = memory_client or get_mcp_client()
        self.plans_generated = 0

        # Load template paths
        self.templates_dir = Path(__file__).parent.parent.parent.parent.parent / "templates"
        self.sonnet_template_path = self.templates_dir / "implementation-plan-sonnet45.md"
        self.glm_template_path = self.templates_dir / "implementation-plan-glm46.md"
        self.handoff_template_path = self.templates_dir / "handoff-prompt-glm46.md"

        logger.logger.info(
            "plan_generator_initialized",
            templates_dir=str(self.templates_dir),
            memory_client_available=self.memory_client is not None
        )

    async def should_generate_plan(
        self,
        current_step: Optional[ProtocolStep],
        task_id: Optional[str]
    ) -> bool:
        """
        Determine if plan generation should trigger.

        Args:
            current_step: Current protocol step
            task_id: Current task ID

        Returns:
            True if plan should be generated
        """
        # Only generate at Step 4 (PLANNING)
        if not current_step or current_step != ProtocolStep.PLANNING:
            return False

        # Require active task
        if not task_id:
            logger.logger.debug("No active task - plan generation skipped")
            return False

        # Check if plan already exists
        try:
            existing_plan = await self.memory_client.call_tool(
                "devstream_get_implementation_plan",
                {"task_id": task_id}
            )

            if existing_plan:
                logger.logger.info("plan_already_exists", task_id=task_id)
                return False
        except Exception:
            pass  # Plan doesn't exist - proceed with generation

        return True

    async def generate_plan(
        self,
        plan_context: PlanContext,
        model_choice: ModelChoice
    ) -> Tuple[bool, Optional[str]]:
        """
        Generate implementation plan for specified model.

        Args:
            plan_context: Context for plan generation
            model_choice: Target model (Sonnet 4.5 or GLM-4.6)

        Returns:
            Tuple of (success, plan_id)
        """
        try:
            # Load appropriate template
            template_content = await self._load_template(model_choice)

            if not template_content:
                logger.logger.error("template_load_failed", model=model_choice.value)
                return False, None

            # Fill template with context
            plan_content = self._fill_template(template_content, plan_context, model_choice)

            # Generate handoff prompt if GLM-4.6
            handoff_prompt = None
            if model_choice == ModelChoice.GLM_46:
                handoff_prompt = await self._generate_handoff_prompt(plan_context)

            # Generate file path
            plan_file_path = self._generate_plan_file_path(plan_context)

            # Build metadata
            metadata = {
                "complexity": plan_context.complexity_score,
                "estimated_duration": plan_context.estimated_duration,
                "context7_libraries": plan_context.context7_libraries,
                "research_findings": plan_context.research_findings[:500] if plan_context.research_findings else ""
            }

            # Create plan via MCP
            result = await self.memory_client.call_tool(
                "devstream_create_implementation_plan",
                {
                    "task_id": plan_context.task_id,
                    "model_type": model_choice.value,
                    "plan_content": plan_content,
                    "plan_file_path": plan_file_path,
                    "handoff_prompt": handoff_prompt,
                    "metadata": metadata
                }
            )

            if result:
                self.plans_generated += 1

                logger.logger.info(
                    "plan_generated_successfully",
                    task_id=plan_context.task_id,
                    model=model_choice.value,
                    file_path=plan_file_path
                )

                # Log to memory
                await self._log_plan_creation(plan_context, model_choice, plan_file_path)

                return True, plan_file_path
            else:
                logger.logger.error("plan_creation_failed_via_mcp")
                return False, None

        except Exception as e:
            logger.logger.error(
                "plan_generation_error",
                error=str(e),
                error_type=type(e).__name__
            )
            return False, None

    async def show_strategic_choice_gate(
        self,
        plan_context: PlanContext
    ) -> ModelChoice:
        """
        Display Strategic Choice Gate for model selection.

        Args:
            plan_context: Context for decision

        Returns:
            Selected model choice
        """
        logger.logger.info(
            "strategic_choice_gate_shown",
            task_id=plan_context.task_id,
            task_title=plan_context.task_title[:50]
        )

        try:
            if PYINQUIRER_AVAILABLE:
                return await self._interactive_model_selection(plan_context)
            else:
                return await self._non_interactive_model_selection(plan_context)

        except Exception as e:
            logger.logger.error("strategic_choice_gate_error", error=str(e))
            # Default to Sonnet 4.5 on error
            return ModelChoice.SONNET_45

    async def _interactive_model_selection(
        self,
        plan_context: PlanContext
    ) -> ModelChoice:
        """
        Interactive model selection using PyInquirer.

        Args:
            plan_context: Context for decision

        Returns:
            Selected model
        """
        print("\n" + "="*80)
        print("🎯 STRATEGIC CHOICE GATE - Protocol v2.2.0")
        print("="*80)
        print(f"\nTask: {plan_context.task_title}")
        print(f"Estimated Duration: {plan_context.estimated_duration} minutes")
        print(f"Complexity Score: {plan_context.complexity_score:.1f}/1.0")
        print(f"Type: {plan_context.task_type}")
        print("\n" + "-"*80)

        questions = [
            {
                'type': 'list',
                'name': 'model_choice',
                'message': 'Select implementation model:',
                'choices': [
                    {
                        'name': '🧠 Sonnet 4.5 - Architectural, Complex Reasoning, Multi-Agent (RECOMMENDED for complex tasks)',
                        'value': 'sonnet-4.5'
                    },
                    Separator(),
                    {
                        'name': '⚡ GLM-4.6 - Execution-Focused, Cost-Optimized (70% cost savings, best for well-defined tasks)',
                        'value': 'glm-4.6'
                    }
                ],
                'default': 'sonnet-4.5' if plan_context.complexity_score >= 0.7 else 'glm-4.6'
            }
        ]

        # Show model comparison
        print("\n📊 Model Comparison:")
        print("\nSonnet 4.5:")
        print("  ✅ SWE-bench 77.2 (state-of-the-art software engineering)")
        print("  ✅ 30+ hour sustained focus on complex tasks")
        print("  ✅ Native subagent orchestration and delegation")
        print("  ✅ 64K output tokens (rich code + documentation)")
        print("  ✅ Exceptional agentic search and architectural creativity")
        print("\nGLM-4.6:")
        print("  ✅ Tool calling accuracy 90.6% (best-in-class)")
        print("  ✅ 15% fewer tokens (efficient, cost-optimized)")
        print("  ✅ Excellent for standard coding patterns")
        print("  ⚠️  Avoid prolonged reasoning (thinking mode costly)")
        print("  ⚠️  Follow provided patterns exactly")
        print("\n" + "-"*80 + "\n")

        try:
            answers = prompt(questions)
            choice_value = answers.get('model_choice', 'sonnet-4.5')

            logger.logger.info(
                "model_choice_made",
                choice=choice_value,
                task_id=plan_context.task_id
            )

            return ModelChoice(choice_value)

        except Exception as e:
            logger.logger.error("interactive_selection_error", error=str(e))
            return ModelChoice.SONNET_45

    async def _non_interactive_model_selection(
        self,
        plan_context: PlanContext
    ) -> ModelChoice:
        """
        Non-interactive fallback for model selection.

        Args:
            plan_context: Context for decision

        Returns:
            Selected model (defaults based on complexity)
        """
        print("\n" + "="*80)
        print("⚠️  STRATEGIC CHOICE GATE - Non-Interactive Mode")
        print("="*80)
        print("\nPyInquirer not available - using automatic selection")

        # Auto-select based on complexity
        if plan_context.complexity_score >= 0.7:
            choice = ModelChoice.SONNET_45
            print(f"\n✅ Auto-selected: Sonnet 4.5 (high complexity: {plan_context.complexity_score:.1f})")
        else:
            choice = ModelChoice.GLM_46
            print(f"\n✅ Auto-selected: GLM-4.6 (moderate complexity: {plan_context.complexity_score:.1f})")

        print("="*80 + "\n")

        logger.logger.info(
            "auto_model_selection",
            choice=choice.value,
            complexity=plan_context.complexity_score
        )

        return choice

    async def _load_template(self, model_choice: ModelChoice) -> Optional[str]:
        """
        Load template content for specified model.

        Args:
            model_choice: Target model

        Returns:
            Template content or None
        """
        try:
            if model_choice == ModelChoice.SONNET_45:
                template_path = self.sonnet_template_path
            else:
                template_path = self.glm_template_path

            if not template_path.exists():
                logger.logger.error("template_not_found", path=str(template_path))
                return None

            with open(template_path, 'r', encoding='utf-8') as f:
                content = f.read()

            logger.logger.debug("template_loaded", model=model_choice.value, size=len(content))
            return content

        except Exception as e:
            logger.logger.error("template_load_error", error=str(e))
            return None

    def _fill_template(
        self,
        template: str,
        context: PlanContext,
        model_choice: ModelChoice
    ) -> str:
        """
        Fill template with context values.

        Args:
            template: Template content with {{variables}}
            context: Plan context
            model_choice: Target model

        Returns:
            Filled template content
        """
        # Build variable mapping
        variables = {
            "task_title": context.task_title,
            "task_id": context.task_id,
            "phase_name": context.phase_name,
            "priority": str(context.priority),
            "estimated_hours": str(context.estimated_duration // 60),
            "component_list": self._format_component_list(context),
            "todowrite_tasks": self._format_todowrite_tasks(context),
            "context7_research": context.research_findings or "No Context7 research findings available.",
            "context7_libraries": ", ".join(context.context7_libraries) if context.context7_libraries else "None",
            "research_findings": context.research_findings or "Pending research in Step 3 (RESEARCH)",
            "code_example": self._extract_code_example(context),
            "use_case": self._extract_use_case(context),
            "files_to_modify": self._format_file_list(context.files_to_modify),
            "files_to_create": self._format_file_list(context.files_to_create),
            "dependencies": ", ".join(context.dependencies) if context.dependencies else "None",
            "performance_target": context.performance_targets or "Standard performance requirements"
        }

        # Replace all {{variable}} occurrences
        filled = template
        for key, value in variables.items():
            filled = filled.replace(f"{{{{{key}}}}}", str(value))

        return filled

    async def _generate_handoff_prompt(self, context: PlanContext) -> str:
        """
        Generate GLM-4.6 handoff prompt.

        Args:
            context: Plan context

        Returns:
            Handoff prompt content
        """
        try:
            # Load handoff template
            if not self.handoff_template_path.exists():
                logger.logger.error("handoff_template_not_found")
                return self._fallback_handoff_prompt(context)

            with open(self.handoff_template_path, 'r', encoding='utf-8') as f:
                template = f.read()

            # Fill handoff template
            plan_file_path = self._generate_plan_file_path(context)

            handoff_variables = {
                "task_title": context.task_title,
                "task_id": context.task_id,
                "phase_name": context.phase_name,
                "priority": str(context.priority),
                "plan_file_path": plan_file_path,
                "plan_summary": context.task_description[:500],
                "context7_research": context.research_findings or "No research findings",
                "context7_libraries": ", ".join(context.context7_libraries) if context.context7_libraries else "None",
                "research_findings": context.research_findings or "Pending",
                "code_example": self._extract_code_example(context),
                "use_case": self._extract_use_case(context),
                "files_to_modify": self._format_file_list(context.files_to_modify),
                "files_to_create": self._format_file_list(context.files_to_create),
                "dependencies": ", ".join(context.dependencies) if context.dependencies else "None",
                "performance_target": context.performance_targets or "Standard"
            }

            # Replace variables
            filled = template
            for key, value in handoff_variables.items():
                filled = filled.replace(f"{{{{{key}}}}}", str(value))

            return filled

        except Exception as e:
            logger.logger.error("handoff_prompt_generation_error", error=str(e))
            return self._fallback_handoff_prompt(context)

    def _fallback_handoff_prompt(self, context: PlanContext) -> str:
        """Generate fallback handoff prompt if template unavailable."""
        return f"""# Task Handoff: {context.task_title}

**Task ID**: `{context.task_id}`
**Model**: GLM-4.6 (Execution-Focused)

## Context
Sonnet 4.5 has completed Steps 1-5 (DISCUSSION → APPROVAL).
Your role: Precise implementation according to approved plan.

## Implementation Plan
See: `{self._generate_plan_file_path(context)}`

## Mission
Execute micro-tasks one at a time, follow plan exactly, test after each task.

Quality Requirements:
- 95%+ test coverage
- Full type hints + docstrings
- mypy --strict passes
- @code-reviewer validation before commit

**Ready to implement? Start with first TodoWrite task.**
"""

    def _generate_plan_file_path(self, context: PlanContext) -> str:
        """
        Generate plan file path following naming convention.

        Args:
            context: Plan context

        Returns:
            File path string
        """
        # Create slug from task title
        slug = re.sub(r'[^a-z0-9]+', '-', context.task_title.lower())
        slug = slug.strip('-')[:50]  # Limit length

        # Format: docs/development/plan/piano_[task-slug].md
        return f"docs/development/plan/piano_{slug}.md"

    def _format_component_list(self, context: PlanContext) -> str:
        """Format component list for template."""
        if not context.files_to_create:
            return "- Component details to be determined in analysis phase"

        components = []
        for file in context.files_to_create[:5]:  # Limit to 5
            components.append(f"- {file}")

        return "\n".join(components)

    def _format_todowrite_tasks(self, context: PlanContext) -> str:
        """Format TodoWrite tasks for template."""
        if not context.todowrite_tasks:
            return "TodoWrite tasks to be defined in planning phase."

        tasks_formatted = []
        for i, task in enumerate(context.todowrite_tasks, 1):
            title = task.get("content", "Task")
            status = task.get("status", "pending")
            tasks_formatted.append(f"{i}. [{status}] {title}")

        return "\n".join(tasks_formatted)

    def _format_file_list(self, files: List[str]) -> str:
        """Format file list for template."""
        if not files:
            return "- None specified yet"

        return "\n".join(f"- `{file}`" for file in files)

    def _extract_code_example(self, context: PlanContext) -> str:
        """Extract code example from research findings."""
        if not context.code_examples:
            return "# Code examples to be added from Context7 research"

        # Get first code example
        first_key = list(context.code_examples.keys())[0]
        return context.code_examples[first_key]

    def _extract_use_case(self, context: PlanContext) -> str:
        """Extract use case from context."""
        return f"Implement {context.task_title} according to DevStream protocol requirements."

    async def _log_plan_creation(
        self,
        context: PlanContext,
        model_choice: ModelChoice,
        file_path: str
    ) -> None:
        """
        Log plan creation to DevStream memory.

        Args:
            context: Plan context
            model_choice: Selected model
            file_path: Plan file path
        """
        try:
            content = (
                f"Implementation Plan Created (Step 4 PLANNING)\n"
                f"Task: {context.task_title}\n"
                f"Task ID: {context.task_id}\n"
                f"Model: {model_choice.value}\n"
                f"File: {file_path}\n"
                f"Complexity: {context.complexity_score:.1f}/1.0\n"
                f"Duration: {context.estimated_duration} minutes\n"
                f"Session ID: {context.session_id}\n"
                f"Timestamp: {context.timestamp}"
            )

            keywords = [
                "implementation-plan",
                "step-4-planning",
                model_choice.value,
                context.task_id,
                context.session_id
            ]

            await self.memory_client.store_memory(
                content=content,
                content_type="decision",
                keywords=keywords
            )

            logger.logger.debug("plan_creation_logged_to_memory", task_id=context.task_id)

        except Exception as e:
            logger.logger.error("plan_creation_logging_failed", error=str(e))

    def get_statistics(self) -> Dict[str, Any]:
        """Get plan generator statistics."""
        return {
            "plans_generated": self.plans_generated,
            "templates_available": {
                "sonnet_45": self.sonnet_template_path.exists(),
                "glm_46": self.glm_template_path.exists(),
                "handoff": self.handoff_template_path.exists()
            },
            "pyinquirer_available": PYINQUIRER_AVAILABLE
        }


# Global instance
_global_generator: Optional[ImplementationPlanGenerator] = None


def get_plan_generator() -> ImplementationPlanGenerator:
    """Get global plan generator instance."""
    global _global_generator
    if _global_generator is None:
        _global_generator = ImplementationPlanGenerator()
    return _global_generator


# Example usage and testing
if __name__ == "__main__":
    async def test_plan_generator():
        """Test plan generator functionality."""

        # Mock memory client
        class MockMemoryClient:
            async def call_tool(self, tool_name, args):
                if tool_name == "devstream_create_implementation_plan":
                    return {"plan_id": "plan-12345"}
                return None

            async def store_memory(self, content, content_type, keywords):
                pass

        generator = ImplementationPlanGenerator(memory_client=MockMemoryClient())

        # Test 1: Template loading
        print("🧪 Test 1: Template loading")
        sonnet_template = await generator._load_template(ModelChoice.SONNET_45)
        glm_template = await generator._load_template(ModelChoice.GLM_46)
        assert sonnet_template is not None
        assert glm_template is not None
        print(f"✅ Sonnet template: {len(sonnet_template)} chars")
        print(f"✅ GLM template: {len(glm_template)} chars")

        # Test 2: Plan file path generation
        print("\n🧪 Test 2: Plan file path generation")
        test_context = PlanContext(
            task_id="test-123",
            task_title="Implement User Authentication System",
            task_description="Build JWT-based auth",
            task_type="coding",
            priority=8,
            phase_name="Core Engine",
            estimated_duration=120,
            complexity_score=0.8,
            context7_libraries=["fastapi", "pyjwt"],
            research_findings="Use bcrypt for passwords",
            code_examples={},
            files_to_modify=[],
            files_to_create=["src/auth.py"],
            dependencies=["fastapi", "pyjwt"],
            performance_targets="< 100ms response time",
            todowrite_tasks=[],
            session_id="sess-123",
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        file_path = generator._generate_plan_file_path(test_context)
        assert "piano_" in file_path
        assert ".md" in file_path
        print(f"✅ File path: {file_path}")

        # Test 3: Statistics
        print("\n🧪 Test 3: Generator statistics")
        stats = generator.get_statistics()
        assert "plans_generated" in stats
        assert "templates_available" in stats
        print(f"✅ Statistics: {stats}")

        print("\n🎉 All plan generator tests PASSED")
        print("\n💡 Note: Full testing requires manual execution with real MCP")

    # Run tests
    asyncio.run(test_plan_generator())
