"""
AI Planning Protocols and Interfaces

Abstract protocols defining contracts for AI planning components.
Based on Python typing best practices and dependency injection patterns.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol

from .models import (
    TaskBreakdownRequest,
    PlanGenerationRequest,
    EstimationRequest,
    AITaskSuggestion,
    TaskDependencySuggestion,
    ComplexityEstimation,
    PlanningResult,
    AIPlannerConfig,
)


class ValidationError(Exception):
    """Validation error for planning operations."""
    pass


class CyclicDependencyError(ValidationError):
    """Cyclic dependency detected in task graph."""
    pass


class TaskBreakdownEngine(Protocol):
    """Protocol for task breakdown functionality."""

    async def breakdown_task(
        self,
        request: TaskBreakdownRequest,
        context: Optional[str] = None
    ) -> List[AITaskSuggestion]:
        """
        Break down a complex task into micro-tasks.

        Args:
            request: Task breakdown request with parameters
            context: Additional context for breakdown

        Returns:
            List of AI-generated task suggestions

        Raises:
            PlanningError: If breakdown fails
            ModelNotAvailableError: If AI model unavailable
        """
        ...

    async def validate_breakdown(
        self,
        tasks: List[AITaskSuggestion]
    ) -> Dict[str, Any]:
        """
        Validate quality of task breakdown.

        Args:
            tasks: Generated task suggestions

        Returns:
            Validation results with quality metrics

        Raises:
            ValidationError: If breakdown quality insufficient
        """
        ...


class DependencyAnalyzer(Protocol):
    """Protocol for dependency analysis functionality."""

    async def analyze_dependencies(
        self,
        tasks: List[AITaskSuggestion],
        context: Optional[str] = None
    ) -> List[TaskDependencySuggestion]:
        """
        Analyze and suggest dependencies between tasks.

        Args:
            tasks: List of tasks to analyze
            context: Additional context for dependency analysis

        Returns:
            List of suggested dependencies

        Raises:
            PlanningError: If dependency analysis fails
            CyclicDependencyError: If cycles detected
        """
        ...

    async def validate_dependencies(
        self,
        dependencies: List[TaskDependencySuggestion],
        tasks: List[AITaskSuggestion]
    ) -> Dict[str, Any]:
        """
        Validate dependency graph for cycles and consistency.

        Args:
            dependencies: Suggested dependencies
            tasks: Available tasks

        Returns:
            Validation results including cycle detection

        Raises:
            CyclicDependencyError: If cycles found
            ValidationError: If dependencies invalid
        """
        ...


class ComplexityEstimator(Protocol):
    """Protocol for complexity and time estimation."""

    async def estimate_task(
        self,
        request: EstimationRequest,
        context: Optional[str] = None
    ) -> ComplexityEstimation:
        """
        Estimate complexity and duration for a task.

        Args:
            request: Estimation request with task details
            context: Additional context for estimation

        Returns:
            Complexity estimation with reasoning

        Raises:
            PlanningError: If estimation fails
            InsufficientContextError: If context inadequate
        """
        ...

    async def calibrate_estimates(
        self,
        estimations: List[ComplexityEstimation],
        reference_data: Optional[Dict[str, Any]] = None
    ) -> List[ComplexityEstimation]:
        """
        Calibrate estimates based on historical data.

        Args:
            estimations: Initial estimates
            reference_data: Historical performance data

        Returns:
            Calibrated estimates

        Raises:
            CalibrationError: If calibration fails
        """
        ...


class ContextRetriever(Protocol):
    """Protocol for retrieving context for planning."""

    async def retrieve_planning_context(
        self,
        query: str,
        max_tokens: int = 2000
    ) -> str:
        """
        Retrieve relevant context for planning.

        Args:
            query: Query for context retrieval
            max_tokens: Maximum context length

        Returns:
            Relevant context text

        Raises:
            ContextRetrievalError: If context retrieval fails
        """
        ...

    async def get_similar_tasks(
        self,
        task_description: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get similar tasks from memory for reference.

        Args:
            task_description: Description to match against
            limit: Maximum number of similar tasks

        Returns:
            List of similar task metadata

        Raises:
            ContextRetrievalError: If retrieval fails
        """
        ...


class AIPlannerProtocol(Protocol):
    """Main protocol for AI planning functionality."""

    config: AIPlannerConfig

    async def generate_task_breakdown(
        self,
        request: TaskBreakdownRequest
    ) -> PlanningResult:
        """
        Generate complete task breakdown with dependencies.

        Args:
            request: Task breakdown request

        Returns:
            Complete planning result

        Raises:
            PlanningError: If planning fails
            ModelNotAvailableError: If AI model unavailable
            ValidationError: If result validation fails
        """
        ...

    async def generate_plan(
        self,
        request: PlanGenerationRequest
    ) -> PlanningResult:
        """
        Generate complete intervention plan.

        Args:
            request: Plan generation request

        Returns:
            Complete planning result with phases

        Raises:
            PlanningError: If planning fails
            ModelNotAvailableError: If AI model unavailable
        """
        ...

    async def estimate_complexity(
        self,
        request: EstimationRequest
    ) -> ComplexityEstimation:
        """
        Estimate task complexity and duration.

        Args:
            request: Estimation request

        Returns:
            Complexity estimation

        Raises:
            PlanningError: If estimation fails
        """
        ...

    async def health_check(self) -> Dict[str, Any]:
        """
        Check AI planner health and availability.

        Returns:
            Health status with model availability

        Raises:
            HealthCheckError: If health check fails
        """
        ...


class ModelProvider(Protocol):
    """Protocol for AI model providers."""

    async def generate_text(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text using language model.

        Args:
            prompt: Input prompt
            model: Model name
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text

        Raises:
            ModelError: If generation fails
            ModelNotFoundError: If model not available
        """
        ...

    async def generate_embedding(
        self,
        text: str,
        model: str
    ) -> List[float]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed
            model: Embedding model name

        Returns:
            Embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        ...

    async def list_available_models(self) -> List[str]:
        """
        List available models.

        Returns:
            List of model names

        Raises:
            ModelError: If listing fails
        """
        ...


# Abstract base classes for implementation


class BaseAIPlanner(ABC):
    """Abstract base class for AI planners."""

    def __init__(self, config: AIPlannerConfig):
        self.config = config

    @abstractmethod
    async def generate_task_breakdown(
        self,
        request: TaskBreakdownRequest
    ) -> PlanningResult:
        """Generate task breakdown implementation."""
        pass

    @abstractmethod
    async def generate_plan(
        self,
        request: PlanGenerationRequest
    ) -> PlanningResult:
        """Generate plan implementation."""
        pass

    @abstractmethod
    async def estimate_complexity(
        self,
        request: EstimationRequest
    ) -> ComplexityEstimation:
        """Estimate complexity implementation."""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Default health check implementation."""
        return {
            "status": "healthy",
            "config": self.config.model_dump(),
            "timestamp": "2025-09-28T00:00:00Z"
        }


class BaseTaskBreakdownEngine(ABC):
    """Abstract base class for task breakdown engines."""

    @abstractmethod
    async def breakdown_task(
        self,
        request: TaskBreakdownRequest,
        context: Optional[str] = None
    ) -> List[AITaskSuggestion]:
        """Break down task implementation."""
        pass

    async def validate_breakdown(
        self,
        tasks: List[AITaskSuggestion]
    ) -> Dict[str, Any]:
        """Default validation implementation."""
        return {
            "valid": True,
            "task_count": len(tasks),
            "avg_duration": sum(t.estimated_minutes for t in tasks) / len(tasks) if tasks else 0,
            "avg_complexity": sum(t.complexity_score for t in tasks) / len(tasks) if tasks else 0
        }


class BaseDependencyAnalyzer(ABC):
    """Abstract base class for dependency analyzers."""

    @abstractmethod
    async def analyze_dependencies(
        self,
        tasks: List[AITaskSuggestion],
        context: Optional[str] = None
    ) -> List[TaskDependencySuggestion]:
        """Analyze dependencies implementation."""
        pass

    async def validate_dependencies(
        self,
        dependencies: List[TaskDependencySuggestion],
        tasks: List[AITaskSuggestion]
    ) -> Dict[str, Any]:
        """Default dependency validation."""
        task_ids = {task.id for task in tasks}
        valid_deps = []
        invalid_deps = []

        for dep in dependencies:
            if (dep.prerequisite_task_id in task_ids and
                dep.dependent_task_id in task_ids):
                valid_deps.append(dep)
            else:
                invalid_deps.append(dep)

        return {
            "valid": len(invalid_deps) == 0,
            "valid_count": len(valid_deps),
            "invalid_count": len(invalid_deps),
            "cycles_detected": 0  # TODO: Implement cycle detection
        }


class BaseComplexityEstimator(ABC):
    """Abstract base class for complexity estimators."""

    @abstractmethod
    async def estimate_task(
        self,
        request: EstimationRequest,
        context: Optional[str] = None
    ) -> ComplexityEstimation:
        """Estimate task implementation."""
        pass

    async def calibrate_estimates(
        self,
        estimations: List[ComplexityEstimation],
        reference_data: Optional[Dict[str, Any]] = None
    ) -> List[ComplexityEstimation]:
        """Default calibration (no-op)."""
        return estimations


class BaseContextRetriever(ABC):
    """Abstract base class for context retrievers."""

    @abstractmethod
    async def retrieve_planning_context(
        self,
        query: str,
        max_tokens: int = 2000
    ) -> str:
        """Retrieve context implementation."""
        pass

    async def get_similar_tasks(
        self,
        task_description: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Default similar task retrieval (empty)."""
        return []