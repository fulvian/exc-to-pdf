"""
Ollama AI Planner Implementation

AI-powered task planning using Ollama + embeddinggemma.
Based on Context7 research findings for optimal integration patterns.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..exceptions import DevStreamError
from ..ollama.client import OllamaClient
from ..ollama.models import ChatRequest, ChatMessage, EmbeddingRequest
from ..memory.search import HybridSearchEngine
from .validation import (
    AIResponseValidator, FallbackHandler, ValidationError,
    ValidationSeverity, default_validator, default_fallback_handler
)
from .models import (
    TaskBreakdownRequest,
    PlanGenerationRequest,
    EstimationRequest,
    AITaskSuggestion,
    TaskDependencySuggestion,
    ComplexityEstimation,
    PlanningResult,
    AIPlannerConfig,
    PromptTemplate,
    PlanningMode,
    EstimationApproach,
    ContextSource,
)
from .protocols import (
    BaseAIPlanner,
    BaseTaskBreakdownEngine,
    BaseDependencyAnalyzer,
    BaseComplexityEstimator,
    BaseContextRetriever,
)
from .dependency_analyzer import enhance_dependency_analysis

logger = logging.getLogger(__name__)


# Custom exceptions for AI Planning
class PlanningError(DevStreamError):
    """Base exception for AI planning errors."""
    pass


class ModelNotAvailableError(PlanningError):
    """Raised when AI model is not available."""
    pass


class InvalidPlanningResponseError(PlanningError):
    """Raised when AI response is invalid or unparseable."""
    pass


class ContextRetrievalError(PlanningError):
    """Raised when context retrieval fails."""
    pass


class ValidationError(PlanningError):
    """Raised when planning result validation fails."""
    pass


class CyclicDependencyError(ValidationError):
    """Raised when cyclic dependencies detected."""
    pass


class OllamaContextRetriever(BaseContextRetriever):
    """Enhanced context retriever using memory search engine with intelligent filtering."""

    def __init__(self, search_engine: HybridSearchEngine):
        self.search_engine = search_engine

    async def retrieve_planning_context(
        self,
        query: str,
        max_tokens: int = 2000
    ) -> str:
        """Retrieve and synthesize relevant context for planning from memory."""
        try:
            from ..memory.models import SearchQuery, ContentType

            # Multi-stage context retrieval strategy
            context_sections = []

            # 1. Search for task-related content
            task_context = await self._search_task_content(query, max_tokens // 3)
            if task_context:
                context_sections.append(f"## Task-Related Experience\n{task_context}")

            # 2. Search for planning patterns and methodologies
            planning_query = f"planning methodology approach breakdown {query}"
            planning_context = await self._search_planning_patterns(planning_query, max_tokens // 3)
            if planning_context:
                context_sections.append(f"## Planning Patterns\n{planning_context}")

            # 3. Search for similar project contexts
            project_context = await self._search_project_context(query, max_tokens // 3)
            if project_context:
                context_sections.append(f"## Project Context\n{project_context}")

            # Combine and optimize context
            combined_context = "\n\n".join(context_sections)

            # Apply intelligent filtering and summarization if needed
            if len(combined_context) > max_tokens * 4:  # Rough character-to-token ratio
                combined_context = self._optimize_context(combined_context, max_tokens)

            return combined_context

        except Exception as e:
            logger.error(f"Enhanced context retrieval failed: {e}")
            # Fallback to basic search
            return await self._basic_context_retrieval(query, max_tokens)

    async def _search_task_content(self, query: str, max_tokens: int) -> str:
        """Search for task and implementation related content."""
        from ..memory.models import SearchQuery, ContentType

        search_query = SearchQuery(
            query_text=query,
            max_results=3,
            semantic_weight=0.8,
            keyword_weight=0.2,
            content_types=[ContentType.CODE, ContentType.IMPLEMENTATION]
        )

        results = await self.search_engine.search(search_query)
        return self._format_search_results(results.results, max_tokens, "implementation")

    async def _search_planning_patterns(self, query: str, max_tokens: int) -> str:
        """Search for planning methodologies and patterns."""
        from ..memory.models import SearchQuery, ContentType

        search_query = SearchQuery(
            query_text=query,
            max_results=3,
            semantic_weight=0.7,
            keyword_weight=0.3,
            content_types=[ContentType.DOCUMENTATION, ContentType.PLANNING]
        )

        results = await self.search_engine.search(search_query)
        return self._format_search_results(results.results, max_tokens, "planning")

    async def _search_project_context(self, query: str, max_tokens: int) -> str:
        """Search for project-specific context and lessons learned."""
        from ..memory.models import SearchQuery, ContentType

        search_query = SearchQuery(
            query_text=f"project context lessons learned {query}",
            max_results=2,
            semantic_weight=0.6,
            keyword_weight=0.4,
            content_types=[ContentType.NOTES, ContentType.DOCUMENTATION]
        )

        results = await self.search_engine.search(search_query)
        return self._format_search_results(results.results, max_tokens, "project")

    def _format_search_results(self, results, max_tokens: int, context_type: str) -> str:
        """Format search results with intelligent content extraction."""
        context_parts = []
        current_tokens = 0

        for result in results:
            content = result.memory.content
            estimated_tokens = len(content.split()) * 1.3

            if current_tokens + estimated_tokens > max_tokens:
                # Truncate content to fit remaining space
                remaining_tokens = max_tokens - current_tokens
                remaining_words = int(remaining_tokens / 1.3)
                content = " ".join(content.split()[:remaining_words]) + "..."

            # Extract key information based on context type
            formatted_content = self._extract_key_information(content, context_type)

            context_parts.append(f"- {formatted_content}")
            current_tokens += min(estimated_tokens, max_tokens - current_tokens)

            if current_tokens >= max_tokens:
                break

        return "\n".join(context_parts)

    def _extract_key_information(self, content: str, context_type: str) -> str:
        """Extract key information based on context type."""
        # Implementation patterns for different content types
        if context_type == "implementation":
            # Focus on technical details, patterns, and code structures
            lines = content.split('\n')
            key_lines = [line for line in lines if any(keyword in line.lower()
                        for keyword in ['implement', 'create', 'build', 'pattern', 'approach'])]
            return '\n  '.join(key_lines[:3]) if key_lines else content[:200]

        elif context_type == "planning":
            # Focus on methodology, process, and planning insights
            lines = content.split('\n')
            key_lines = [line for line in lines if any(keyword in line.lower()
                        for keyword in ['plan', 'approach', 'methodology', 'process', 'step', 'phase'])]
            return '\n  '.join(key_lines[:3]) if key_lines else content[:200]

        elif context_type == "project":
            # Focus on context, lessons learned, and project insights
            lines = content.split('\n')
            key_lines = [line for line in lines if any(keyword in line.lower()
                        for keyword in ['context', 'lesson', 'learned', 'issue', 'challenge', 'solution'])]
            return '\n  '.join(key_lines[:3]) if key_lines else content[:200]

        # Default fallback
        return content[:200] + "..." if len(content) > 200 else content

    def _optimize_context(self, context: str, max_tokens: int) -> str:
        """Optimize context by removing redundancy and focusing on key information."""
        lines = context.split('\n')

        # Priority order for different section types
        section_priorities = {
            "## Task-Related Experience": 3,
            "## Planning Patterns": 2,
            "## Project Context": 1
        }

        # Group lines by sections
        sections = {}
        current_section = None

        for line in lines:
            if line.startswith("##"):
                current_section = line
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(line)

        # Build optimized context prioritizing high-value sections
        optimized_lines = []
        current_tokens = 0
        target_tokens = max_tokens * 4  # Character-to-token ratio

        for section, priority in sorted(section_priorities.items(), key=lambda x: x[1], reverse=True):
            if section in sections:
                section_content = [section] + sections[section]
                section_text = '\n'.join(section_content)

                if current_tokens + len(section_text) <= target_tokens:
                    optimized_lines.extend(section_content)
                    current_tokens += len(section_text)
                else:
                    # Include partial section if space allows
                    remaining_space = target_tokens - current_tokens
                    if remaining_space > 100:  # Minimum useful content
                        partial_content = section_text[:remaining_space] + "..."
                        optimized_lines.extend([section, partial_content])
                    break

        return '\n'.join(optimized_lines)

    async def _basic_context_retrieval(self, query: str, max_tokens: int) -> str:
        """Fallback basic context retrieval."""
        try:
            from ..memory.models import SearchQuery

            search_query = SearchQuery(
                query_text=query,
                max_results=5,
                semantic_weight=0.7,
                keyword_weight=0.3
            )

            results = await self.search_engine.search(search_query)

            # Simple content combination
            context_parts = []
            current_tokens = 0

            for result in results.results:
                content = result.memory.content
                estimated_tokens = len(content.split()) * 1.3

                if current_tokens + estimated_tokens > max_tokens:
                    break

                context_parts.append(f"Context: {content}")
                current_tokens += estimated_tokens

            return "\n\n".join(context_parts)

        except Exception as e:
            logger.warning(f"Basic context retrieval failed: {e}")
            return ""

    async def get_similar_tasks(
        self,
        task_description: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get similar tasks from memory."""
        try:
            from ..memory.models import SearchQuery, ContentType

            search_query = SearchQuery(
                query_text=task_description,
                content_types=[ContentType.CODE, ContentType.DOCUMENTATION],
                max_results=limit,
                semantic_weight=0.8,
                keyword_weight=0.2
            )

            results = await self.search_engine.search(search_query)

            similar_tasks = []
            for result in results.results:
                similar_tasks.append({
                    "id": result.memory.id,
                    "content": result.memory.content,
                    "complexity": result.memory.complexity_score,
                    "keywords": result.memory.keywords,
                    "similarity": result.score
                })

            return similar_tasks

        except Exception as e:
            logger.error(f"Similar task retrieval failed: {e}")
            return []  # Non-fatal error


class OllamaTaskBreakdownEngine(BaseTaskBreakdownEngine):
    """Task breakdown engine using Ollama."""

    def __init__(self, ollama_client: OllamaClient, config: AIPlannerConfig):
        self.ollama_client = ollama_client
        self.config = config

    async def breakdown_task(
        self,
        request: TaskBreakdownRequest,
        context: Optional[str] = None
    ) -> List[AITaskSuggestion]:
        """Break down task using AI generation."""
        try:
            prompt = self._build_breakdown_prompt(request, context)

            chat_request = ChatRequest(
                model=self.config.model_name,
                messages=[ChatMessage(role="user", content=prompt)],
                options={
                    "temperature": self.config.temperature,
                    "max_tokens": 2000
                }
            )

            response = await self.ollama_client.chat(chat_request)
            response_text = response.message.content

            # Parse AI response to extract tasks
            tasks = self._parse_breakdown_response(response_text, request)

            logger.info(f"Generated {len(tasks)} task suggestions for: {request.objective}")
            return tasks

        except Exception as e:
            logger.error(f"Task breakdown failed: {e}")
            raise PlanningError(f"Failed to break down task: {e}") from e

    def _build_breakdown_prompt(
        self,
        request: TaskBreakdownRequest,
        context: Optional[str] = None
    ) -> str:
        """Build prompt for task breakdown."""
        base_prompt = f"""
You are an expert software engineering task planner. Break down the following objective into micro-tasks.

OBJECTIVE: {request.objective}

CONSTRAINTS:
- Each task must be 5-{request.max_task_duration_minutes} minutes maximum
- Use action verbs (implement, create, add, test, etc.)
- Maximum {request.max_tasks} tasks total
- Tasks should be specific and actionable

PLANNING MODE: {request.planning_mode.value}
ESTIMATION APPROACH: {request.estimation_approach.value}
"""

        if context:
            base_prompt += f"\n\nCONTEXT:\n{context}\n"

        if request.context:
            base_prompt += f"\n\nADDITIONAL CONTEXT:\n{request.context}\n"

        base_prompt += """
RESPONSE FORMAT (JSON):
{
  "tasks": [
    {
      "title": "Action verb + specific task",
      "description": "Detailed description of what to do",
      "estimated_minutes": integer (5-10),
      "complexity_score": integer (1-10),
      "priority_score": float (0.0-1.0),
      "task_type": "implementation|testing|documentation|analysis|deployment",
      "prerequisites": ["task_title_1", "task_title_2"],
      "reasoning": "Why this task is needed and how it fits"
    }
  ],
  "planning_confidence": float (0.0-1.0),
  "completeness_score": float (0.0-1.0),
  "reasoning": "Overall planning rationale"
}

Generate the task breakdown now:
"""

        return base_prompt

    def _parse_breakdown_response(
        self,
        response_text: str,
        request: TaskBreakdownRequest
    ) -> List[AITaskSuggestion]:
        """Parse AI response into task suggestions with advanced validation."""
        try:
            # Use advanced validation system
            is_valid, parsed_data, error_message = default_validator.validate_and_fix(
                response_text, "task_breakdown"
            )

            if not is_valid:
                logger.warning(f"Validation failed: {error_message}")
                # Attempt fallback generation
                fallback_data = default_fallback_handler.get_fallback_task_breakdown(
                    objective=request.objective,
                    context=request.context,
                    target_count=request.max_tasks
                )
                logger.info("Using fallback task breakdown generation")
                parsed_data = fallback_data

            tasks = []
            for task_data in parsed_data.get("tasks", []):
                try:
                    # Enhanced task creation with validation
                    suggestion = AITaskSuggestion(
                        title=task_data["title"],
                        description=task_data["description"],
                        estimated_minutes=task_data["estimated_minutes"],
                        complexity_score=task_data["complexity_score"],
                        priority_score=task_data.get("priority_score", 5),  # Default priority
                        task_type=task_data.get("task_type", "implementation"),  # Default type
                        prerequisite_tasks=task_data.get("prerequisites", []),
                        reasoning=task_data.get("reasoning", "AI-generated task"),
                        confidence_score=parsed_data.get("confidence_score", 0.8),
                        context_sources=[request.context_source],
                        model_used=self.config.model_name,
                        # Enhanced metadata for validation tracking
                        validation_metadata={
                            "validation_passed": is_valid,
                            "fallback_used": parsed_data.get("fallback_used", False),
                            "validation_timestamp": datetime.now().isoformat()
                        }
                    )
                    tasks.append(suggestion)

                except ValidationError as e:
                    logger.error(f"Task validation failed: {task_data}. Error: {e}")
                    if e.severity in [ValidationSeverity.HIGH, ValidationSeverity.CRITICAL]:
                        raise InvalidPlanningResponseError(f"Critical task validation failure: {e}") from e
                    continue
                except Exception as e:
                    logger.warning(f"Failed to parse task: {task_data}. Error: {e}")
                    continue

            if not tasks:
                logger.error("No valid tasks generated, attempting emergency fallback")
                # Emergency fallback - create basic task structure
                fallback_data = default_fallback_handler.get_fallback_task_breakdown(
                    objective=request.objective,
                    context=request.context,
                    target_count=min(request.max_tasks, 3)  # Conservative fallback
                )

                for task_data in fallback_data["tasks"]:
                    suggestion = AITaskSuggestion(
                        title=task_data["title"],
                        description=task_data["description"],
                        estimated_minutes=task_data["estimated_minutes"],
                        complexity_score=task_data["complexity_score"],
                        priority_score=5,
                        task_type=task_data["task_type"],
                        prerequisite_tasks=[],
                        reasoning="Emergency fallback generation",
                        confidence_score=0.3,  # Low confidence for emergency fallback
                        context_sources=[request.context_source],
                        model_used="fallback_system",
                        validation_metadata={
                            "validation_passed": False,
                            "fallback_used": True,
                            "emergency_fallback": True,
                            "validation_timestamp": datetime.now().isoformat()
                        }
                    )
                    tasks.append(suggestion)

            return tasks

        except ValidationError as e:
            if e.severity == ValidationSeverity.CRITICAL:
                logger.critical(f"Critical validation error: {e}")
                raise InvalidPlanningResponseError(f"Critical validation failure: {e}") from e

            # For non-critical validation errors, log and continue with fallback
            logger.warning(f"Validation warning: {e}")
            return self._emergency_fallback_tasks(request)

        except Exception as e:
            logger.error(f"Unexpected error in task breakdown parsing: {e}")
            # Final fallback mechanism
            return self._emergency_fallback_tasks(request)

    def _emergency_fallback_tasks(self, request: TaskBreakdownRequest) -> List[AITaskSuggestion]:
        """Generate emergency fallback tasks when all else fails."""
        try:
            fallback_data = default_fallback_handler.get_fallback_task_breakdown(
                objective=request.objective,
                context=request.context,
                target_count=3  # Minimal fallback
            )

            tasks = []
            for i, task_data in enumerate(fallback_data["tasks"]):
                suggestion = AITaskSuggestion(
                    title=f"Emergency Task {i+1}: {task_data['title']}",
                    description=task_data["description"],
                    estimated_minutes=task_data["estimated_minutes"],
                    complexity_score=task_data["complexity_score"],
                    priority_score=5,
                    task_type=task_data["task_type"],
                    prerequisite_tasks=[],
                    reasoning="Emergency fallback - system recovery",
                    confidence_score=0.2,
                    context_sources=[request.context_source],
                    model_used="emergency_fallback",
                    validation_metadata={
                        "validation_passed": False,
                        "fallback_used": True,
                        "emergency_fallback": True,
                        "validation_timestamp": datetime.now().isoformat()
                    }
                )
                tasks.append(suggestion)

            logger.warning(f"Emergency fallback generated {len(tasks)} basic tasks")
            return tasks

        except Exception as e:
            logger.critical(f"Emergency fallback failed: {e}")
            raise InvalidPlanningResponseError("Complete system failure - unable to generate any tasks") from e


class OllamaDependencyAnalyzer(BaseDependencyAnalyzer):
    """Dependency analyzer using Ollama."""

    def __init__(self, ollama_client: OllamaClient, config: AIPlannerConfig):
        self.ollama_client = ollama_client
        self.config = config

    async def analyze_dependencies(
        self,
        tasks: List[AITaskSuggestion],
        context: Optional[str] = None
    ) -> List[TaskDependencySuggestion]:
        """Analyze dependencies between tasks."""
        try:
            prompt = self._build_dependency_prompt(tasks, context)

            chat_request = ChatRequest(
                model=self.config.model_name,
                messages=[ChatMessage(role="user", content=prompt)],
                options={
                    "temperature": self.config.temperature,
                    "max_tokens": 1500
                }
            )

            response = await self.ollama_client.chat(chat_request)
            dependencies = self._parse_dependency_response(response.message.content, tasks)

            logger.info(f"Analyzed {len(dependencies)} dependencies for {len(tasks)} tasks")
            return dependencies

        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")
            raise PlanningError(f"Failed to analyze dependencies: {e}") from e

    def _build_dependency_prompt(
        self,
        tasks: List[AITaskSuggestion],
        context: Optional[str] = None
    ) -> str:
        """Build prompt for dependency analysis."""
        task_list = "\n".join([
            f"{i+1}. {task.title}: {task.description}"
            for i, task in enumerate(tasks)
        ])

        prompt = f"""
Analyze dependencies between these tasks. Identify which tasks must be completed before others can start.

TASKS:
{task_list}

DEPENDENCY TYPES:
- blocking: Task B cannot start until Task A is complete
- soft: Task B is easier/better if Task A is done first
- parallel: Tasks can be done simultaneously

RESPONSE FORMAT (JSON):
{{
  "dependencies": [
    {{
      "prerequisite": "Task title (exact match)",
      "dependent": "Task title (exact match)",
      "type": "blocking|soft|parallel_possible",
      "strength": float (0.0-1.0),
      "reasoning": "Why this dependency exists"
    }}
  ],
  "analysis_confidence": float (0.0-1.0)
}}

Analyze dependencies now:
"""

        return prompt

    def _parse_dependency_response(
        self,
        response_text: str,
        tasks: List[AITaskSuggestion]
    ) -> List[TaskDependencySuggestion]:
        """Parse dependency analysis response with advanced validation."""
        try:
            # Use advanced validation system for dependency analysis
            is_valid, parsed_data, error_message = default_validator.validate_and_fix(
                response_text, "dependency_analysis"
            )

            if not is_valid:
                logger.warning(f"Dependency validation failed: {error_message}")
                # Use smart dependency detector for fallback
                from .dependency_analyzer import SmartDependencyDetector
                detector = SmartDependencyDetector()
                implicit_deps = detector.detect_implicit_dependencies(tasks)
                logger.info(f"Using implicit dependency detection, found {len(implicit_deps)} dependencies")
                return implicit_deps

            # Create task title to ID mapping
            task_map = {task.title: task.id for task in tasks}

            dependencies = []
            raw_dependencies = parsed_data.get("dependencies", [])

            for dep_data in raw_dependencies:
                try:
                    # Enhanced dependency parsing with validation
                    prereq_title = dep_data.get("prerequisite_task", dep_data.get("prerequisite", ""))
                    dep_title = dep_data.get("dependent_task", dep_data.get("dependent", ""))

                    # Find matching task IDs
                    prereq_id = task_map.get(prereq_title)
                    dep_id = task_map.get(dep_title)

                    if not prereq_id or not dep_id:
                        logger.warning(f"Task mapping failed: '{prereq_title}' -> {prereq_id}, '{dep_title}' -> {dep_id}")
                        continue

                    if prereq_id == dep_id:
                        logger.warning(f"Self-dependency detected: {prereq_title}")
                        continue

                    # Validate dependency strength
                    strength = dep_data.get("strength", 0.5)
                    if not isinstance(strength, (int, float)) or not (0.0 <= strength <= 1.0):
                        logger.warning(f"Invalid strength {strength}, using default 0.5")
                        strength = 0.5

                    dependency = TaskDependencySuggestion(
                        prerequisite_task_id=prereq_id,
                        dependent_task_id=dep_id,
                        dependency_type=dep_data.get("type", "soft"),
                        strength=strength,
                        reasoning=dep_data.get("reasoning", "AI-generated dependency"),
                        confidence_score=parsed_data.get("analysis_confidence", 0.8),
                        detected_from=["ai_analysis", "validation_system"],
                        validation_metadata={
                            "validation_passed": is_valid,
                            "fallback_used": parsed_data.get("fallback_used", False),
                            "validation_timestamp": datetime.now().isoformat()
                        }
                    )
                    dependencies.append(dependency)

                except Exception as e:
                    logger.warning(f"Failed to parse dependency: {dep_data}. Error: {e}")
                    continue

            # If no valid dependencies found but we have tasks, try implicit detection
            if not dependencies and len(tasks) > 1:
                logger.info("No explicit dependencies found, attempting implicit detection")
                from .dependency_analyzer import SmartDependencyDetector
                detector = SmartDependencyDetector()
                implicit_deps = detector.detect_implicit_dependencies(tasks)
                dependencies.extend(implicit_deps)

            logger.info(f"Parsed {len(dependencies)} dependencies for {len(tasks)} tasks")
            return dependencies

        except ValidationError as e:
            logger.error(f"Dependency validation error: {e}")
            if e.severity in [ValidationSeverity.HIGH, ValidationSeverity.CRITICAL]:
                # For critical errors, still try implicit detection
                logger.warning("Using fallback implicit dependency detection due to critical error")

            return self._fallback_dependency_detection(tasks)

        except Exception as e:
            logger.error(f"Unexpected error in dependency parsing: {e}")
            return self._fallback_dependency_detection(tasks)

    def _fallback_dependency_detection(self, tasks: List[AITaskSuggestion]) -> List[TaskDependencySuggestion]:
        """Fallback dependency detection when AI analysis fails."""
        try:
            from .dependency_analyzer import SmartDependencyDetector
            detector = SmartDependencyDetector()
            implicit_deps = detector.detect_implicit_dependencies(tasks)

            logger.info(f"Fallback dependency detection found {len(implicit_deps)} implicit dependencies")

            # Add metadata to mark these as fallback-generated
            for dep in implicit_deps:
                dep.validation_metadata = {
                    "validation_passed": False,
                    "fallback_used": True,
                    "detection_method": "implicit_pattern_matching",
                    "validation_timestamp": datetime.now().isoformat()
                }

            return implicit_deps

        except Exception as e:
            logger.error(f"Fallback dependency detection failed: {e}")
            return []  # Return empty list as final fallback

    async def validate_dependencies(
        self,
        dependencies: List[TaskDependencySuggestion],
        tasks: List[AITaskSuggestion]
    ) -> Dict[str, Any]:
        """Enhanced dependency validation with cycle detection and optimization."""
        try:
            # Use enhanced dependency analysis for comprehensive validation
            analysis_result = enhance_dependency_analysis(dependencies, tasks)

            logger.info(f"Dependency validation: {analysis_result['validation_passed']}")
            logger.info(f"Cycles detected: {analysis_result['cycles_detected']}")
            logger.info(f"Optimized from {analysis_result['total_dependencies']} to {analysis_result['optimized_dependencies']} dependencies")

            return {
                "valid": analysis_result["validation_passed"],
                "cycles_detected": analysis_result["cycles_detected"],
                "cycles": analysis_result["cycles"],
                "original_count": analysis_result["original_dependencies"],
                "implicit_count": analysis_result["implicit_dependencies"],
                "total_count": analysis_result["total_dependencies"],
                "optimized_count": analysis_result["optimized_dependencies"],
                "analysis": analysis_result["analysis"],
                "optimized_dependencies": analysis_result["optimized_deps"]
            }

        except Exception as e:
            logger.error(f"Enhanced dependency validation failed: {e}")
            # Fallback to basic validation
            return await super().validate_dependencies(dependencies, tasks)


class OllamaComplexityEstimator(BaseComplexityEstimator):
    """Complexity estimator using Ollama."""

    def __init__(self, ollama_client: OllamaClient, config: AIPlannerConfig):
        self.ollama_client = ollama_client
        self.config = config

    async def estimate_task(
        self,
        request: EstimationRequest,
        context: Optional[str] = None
    ) -> ComplexityEstimation:
        """Estimate task complexity and duration."""
        try:
            prompt = self._build_estimation_prompt(request, context)

            chat_request = ChatRequest(
                model=self.config.model_name,
                messages=[ChatMessage(role="user", content=prompt)],
                options={
                    "temperature": self.config.temperature,
                    "max_tokens": 1000
                }
            )

            response = await self.ollama_client.chat(chat_request)
            estimation = self._parse_estimation_response(response.message.content, request)

            logger.info(f"Estimated task '{request.task_title}': {estimation.estimated_minutes}min, complexity {estimation.complexity_score}")
            return estimation

        except Exception as e:
            logger.error(f"Task estimation failed: {e}")
            raise PlanningError(f"Failed to estimate task: {e}") from e

    def _build_estimation_prompt(
        self,
        request: EstimationRequest,
        context: Optional[str] = None
    ) -> str:
        """Build prompt for task estimation."""
        prompt = f"""
You are an expert software development estimator. Analyze this task using proven estimation methodologies.

TASK TO ESTIMATE:
Title: {request.task_title}
Description: {request.task_description}
Approach: {request.approach.value}
"""

        if context:
            prompt += f"\nADDITIONAL CONTEXT: {context}\n"

        if request.project_context:
            prompt += f"\nPROJECT CONTEXT: {request.project_context}\n"

        if request.similar_tasks_context:
            prompt += f"\nSIMILAR TASKS REFERENCE: {request.similar_tasks_context}\n"

        prompt += """
ESTIMATION FRAMEWORK:
Use these research-backed metrics for accurate estimation:

1. COMPLEXITY FACTORS (Lizard-based analysis):
   - Cyclomatic Complexity: Number of decision points and branches
   - Lines of Code: Expected implementation size (NLOC)
   - Token Count: Language complexity and verbosity
   - Parameter Count: Interface complexity
   - Nested Depth: Control structure complexity

2. TIME ESTIMATION FACTORS (SD-Metrics velocity-based):
   - Implementation Time: Core coding and logic
   - Testing Time: Unit tests and validation
   - Integration Time: Component connection and dependencies
   - Documentation Time: Code comments and docs
   - Review/Refactor Time: Code quality improvements

3. RISK FACTORS:
   - Technical Uncertainty: Unknown technologies or patterns
   - Dependency Risk: External components or services
   - Integration Complexity: System interconnections
   - Performance Requirements: Optimization needs
   - Error Handling: Exception and edge case coverage

RESPONSE FORMAT (JSON):
{
  "estimated_minutes": integer (1-60),
  "complexity_score": integer (1-10),
  "uncertainty_factor": float (0.0-1.0),
  "analysis_factors": {
    "cyclomatic_complexity": float (0.0-1.0),
    "implementation_size": float (0.0-1.0),
    "integration_complexity": float (0.0-1.0),
    "testing_overhead": float (0.0-1.0),
    "technical_risk": float (0.0-1.0)
  },
  "risk_factors": ["list of specific risk factors"],
  "assumptions": ["list of key assumptions"],
  "confidence_score": float (0.0-1.0),
  "reasoning": "detailed explanation using the estimation framework"
}

ESTIMATION GUIDELINES:
- Conservative: Add 20-30% buffer, higher complexity scores
- Realistic: Best estimate based on analysis, balanced risk assessment
- Optimistic: Minimal time, assume ideal conditions
- Adaptive: Context-aware estimation based on available information

Provide accurate, evidence-based estimation:
"""

        return prompt

    def _parse_estimation_response(
        self,
        response_text: str,
        request: EstimationRequest
    ) -> ComplexityEstimation:
        """Parse estimation response."""
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            json_text = response_text[json_start:json_end]
            parsed = json.loads(json_text)

            estimation = ComplexityEstimation(
                task_reference=request.task_title,
                estimated_minutes=parsed["estimated_minutes"],
                complexity_score=parsed["complexity_score"],
                uncertainty_factor=parsed["uncertainty_factor"],
                analysis_factors=parsed.get("analysis_factors", {}),
                risk_factors=parsed.get("risk_factors", []),
                assumptions=parsed.get("assumptions", []),
                confidence_score=parsed["confidence_score"],
                estimation_reasoning=parsed["reasoning"],
                approach_used=request.approach,
                model_used=self.config.model_name
            )

            return estimation

        except Exception as e:
            logger.error(f"Failed to parse estimation response: {e}")
            # Return fallback estimation
            return ComplexityEstimation(
                task_reference=request.task_title,
                estimated_minutes=self.config.fallback_task_duration,
                complexity_score=self.config.fallback_complexity,
                uncertainty_factor=0.8,
                confidence_score=0.3,
                estimation_reasoning="Fallback estimation due to parsing error",
                approach_used=request.approach,
                model_used=self.config.model_name
            )

    async def calibrate_estimates(
        self,
        estimations: List[ComplexityEstimation],
        reference_data: Optional[Dict[str, Any]] = None
    ) -> List[ComplexityEstimation]:
        """
        Calibrate estimates using SD-Metrics-based velocity analysis.

        Applies calibration factors based on:
        - Historical velocity data
        - Team performance metrics
        - Task complexity patterns
        - Uncertainty factors
        """
        if not estimations:
            return estimations

        calibrated = []

        # Extract calibration factors from reference data
        velocity_factor = 1.0
        complexity_bias = 0.0
        uncertainty_adjustment = 1.0

        if reference_data:
            # SD-Metrics style velocity analysis
            avg_velocity = reference_data.get("average_velocity_per_day", 1.0)
            team_efficiency = reference_data.get("team_efficiency_factor", 1.0)
            historical_accuracy = reference_data.get("estimation_accuracy", 0.8)

            # Calculate velocity-based time adjustment
            velocity_factor = max(0.5, min(2.0, 1.0 / avg_velocity if avg_velocity > 0 else 1.0))

            # Complexity bias adjustment (teams tend to under/over-estimate)
            complexity_bias = reference_data.get("complexity_bias", 0.0)  # -1.0 to 1.0

            # Uncertainty adjustment based on historical accuracy
            uncertainty_adjustment = max(0.8, min(1.5, 2.0 - historical_accuracy))

            logger.info(f"Calibration factors: velocity={velocity_factor:.2f}, complexity_bias={complexity_bias:.2f}, uncertainty={uncertainty_adjustment:.2f}")

        for estimation in estimations:
            # Apply velocity-based time calibration
            calibrated_time = int(estimation.estimated_minutes * velocity_factor * uncertainty_adjustment)
            calibrated_time = max(1, min(60, calibrated_time))  # Keep within bounds

            # Apply complexity bias calibration
            calibrated_complexity = estimation.complexity_score + complexity_bias
            calibrated_complexity = max(1, min(10, int(calibrated_complexity)))

            # Adjust uncertainty based on confidence and reference data
            calibrated_uncertainty = estimation.uncertainty_factor * uncertainty_adjustment
            calibrated_uncertainty = max(0.0, min(1.0, calibrated_uncertainty))

            # Update confidence based on calibration reliability
            confidence_adjustment = 0.9 if reference_data else 0.7
            calibrated_confidence = estimation.confidence_score * confidence_adjustment

            # Create calibrated estimation
            calibrated_estimation = ComplexityEstimation(
                task_reference=estimation.task_reference,
                estimated_minutes=calibrated_time,
                complexity_score=calibrated_complexity,
                uncertainty_factor=calibrated_uncertainty,
                analysis_factors=estimation.analysis_factors,
                risk_factors=estimation.risk_factors,
                assumptions=estimation.assumptions + [f"Calibrated using velocity factor {velocity_factor:.2f}"],
                confidence_score=calibrated_confidence,
                estimation_reasoning=f"{estimation.estimation_reasoning}\n\nCALIBRATED: Applied velocity factor {velocity_factor:.2f}, complexity bias {complexity_bias:.2f}, uncertainty adjustment {uncertainty_adjustment:.2f}",
                similar_tasks=estimation.similar_tasks,
                reference_data=reference_data or {},
                approach_used=estimation.approach_used,
                model_used=estimation.model_used
            )

            calibrated.append(calibrated_estimation)

        logger.info(f"Calibrated {len(calibrated)} estimations using reference data")
        return calibrated


class OllamaPlanner(BaseAIPlanner):
    """Main Ollama-based AI planner implementation."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        search_engine: Optional[HybridSearchEngine] = None,
        config: Optional[AIPlannerConfig] = None
    ):
        self.config = config or AIPlannerConfig()
        super().__init__(self.config)

        self.ollama_client = ollama_client
        self.search_engine = search_engine

        # Initialize components
        self.context_retriever = OllamaContextRetriever(search_engine) if search_engine else None
        self.breakdown_engine = OllamaTaskBreakdownEngine(ollama_client, self.config)
        self.dependency_analyzer = OllamaDependencyAnalyzer(ollama_client, self.config)
        self.complexity_estimator = OllamaComplexityEstimator(ollama_client, self.config)

    async def generate_task_breakdown(
        self,
        request: TaskBreakdownRequest
    ) -> PlanningResult:
        """Generate complete task breakdown with dependencies."""
        try:
            start_time = datetime.now()

            # Retrieve context if enabled
            context = await self._retrieve_context(request) if request.include_memory_context else None

            # Generate task breakdown
            tasks = await self.breakdown_engine.breakdown_task(request, context)

            # Analyze dependencies
            dependencies = await self.dependency_analyzer.analyze_dependencies(tasks, context)

            # Validate result
            validation = await self._validate_planning_result(tasks, dependencies)

            # Create result
            result = PlanningResult(
                request_id=request.id,
                suggested_tasks=tasks,
                suggested_dependencies=dependencies,
                planning_confidence=validation.get("confidence", 0.8),
                completeness_score=validation.get("completeness", 0.8),
                planning_reasoning=f"Generated {len(tasks)} tasks with {len(dependencies)} dependencies",
                memory_context_used=context[:200] + "..." if context and len(context) > 200 else context,
                provided_context_used=request.context,
                context_sources=[request.context_source],
                generation_duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                model_used=self.config.model_name
            )

            logger.info(f"Generated task breakdown: {len(tasks)} tasks, {len(dependencies)} dependencies")
            return result

        except Exception as e:
            logger.error(f"Task breakdown generation failed: {e}")
            raise PlanningError(f"Failed to generate task breakdown: {e}") from e

    async def generate_plan(
        self,
        request: PlanGenerationRequest
    ) -> PlanningResult:
        """Generate complete intervention plan."""
        # For now, delegate to task breakdown with expanded parameters
        breakdown_request = TaskBreakdownRequest(
            objective=f"{request.title}: {request.description}",
            context=request.provided_context,
            planning_mode=request.planning_mode,
            estimation_approach=request.estimation_approach,
            context_source=request.context_source,
            max_tasks=request.max_phases * request.max_tasks_per_phase,
            include_memory_context=request.include_memory_context,
            memory_context_limit=request.memory_context_limit
        )

        return await self.generate_task_breakdown(breakdown_request)

    async def estimate_complexity(
        self,
        request: EstimationRequest
    ) -> ComplexityEstimation:
        """Estimate task complexity and duration."""
        context = await self._retrieve_context_for_estimation(request) if self.context_retriever else None
        return await self.complexity_estimator.estimate_task(request, context)

    async def health_check(self) -> Dict[str, Any]:
        """Check AI planner health and availability."""
        try:
            # Test basic model availability
            test_request = ChatRequest(
                model=self.config.model_name,
                messages=[ChatMessage(role="user", content="Hello")],
                options={"max_tokens": 10}
            )

            response = await self.ollama_client.chat(test_request)

            return {
                "status": "healthy",
                "model_available": True,
                "model_name": self.config.model_name,
                "embedding_model": self.config.embedding_model,
                "search_engine_available": self.search_engine is not None,
                "last_response": response.message.content[:50],
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "model_available": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _retrieve_context(self, request: TaskBreakdownRequest) -> Optional[str]:
        """Retrieve context for planning."""
        if not self.context_retriever:
            return None

        try:
            return await self.context_retriever.retrieve_planning_context(
                request.objective,
                request.memory_context_limit
            )
        except Exception as e:
            logger.warning(f"Context retrieval failed: {e}")
            return None

    async def _retrieve_context_for_estimation(self, request: EstimationRequest) -> Optional[str]:
        """Retrieve context for estimation."""
        if not self.context_retriever:
            return None

        try:
            similar_tasks = await self.context_retriever.get_similar_tasks(
                request.task_description,
                limit=3
            )

            if similar_tasks:
                context_parts = []
                for task in similar_tasks:
                    context_parts.append(
                        f"Similar task: {task['content'][:200]} "
                        f"(complexity: {task.get('complexity', 'unknown')})"
                    )
                return "\n".join(context_parts)

            return None

        except Exception as e:
            logger.warning(f"Context retrieval for estimation failed: {e}")
            return None

    async def _validate_planning_result(
        self,
        tasks: List[AITaskSuggestion],
        dependencies: List[TaskDependencySuggestion]
    ) -> Dict[str, Any]:
        """Validate planning result quality."""
        # Basic validation
        if not tasks:
            raise ValidationError("No tasks generated")

        # Check task durations
        over_limit = [t for t in tasks if t.estimated_minutes > 10]
        if over_limit:
            logger.warning(f"{len(over_limit)} tasks exceed 10-minute limit")

        # Validate dependencies
        dep_validation = await self.dependency_analyzer.validate_dependencies(dependencies, tasks)

        return {
            "confidence": min(sum(t.confidence_score for t in tasks) / len(tasks), 1.0),
            "completeness": 0.9 if len(tasks) >= 3 else 0.7,
            "dependency_validation": dep_validation,
            "over_limit_tasks": len(over_limit)
        }