"""
DevStream AI Planning System

AI-powered task planning and breakdown using Ollama + embeddinggemma.
Provides intelligent task decomposition, dependency analysis, and estimation
with context-aware planning based on semantic memory.

Key Features:
- Intelligent task breakdown using LLM-powered analysis
- Dependency detection and cycle prevention
- Time and complexity estimation
- Memory-informed context injection
- Graceful fallback when AI unavailable

Usage:
    from devstream.planning import OllamaPlanner, TaskBreakdownRequest, PlanningResult

    planner = OllamaPlanner(ollama_client, memory_search)
    request = TaskBreakdownRequest(
        objective="Implement user authentication system",
        context="Web application with JWT tokens"
    )
    result = await planner.generate_task_breakdown(request)
"""

from .models import (
    # Request/Response Models
    TaskBreakdownRequest,
    PlanGenerationRequest,
    EstimationRequest,

    # Result Models
    AITaskSuggestion,
    TaskDependencySuggestion,
    ComplexityEstimation,
    PlanningResult,

    # Configuration
    AIPlannerConfig,
    PromptTemplate,

    # Enums
    PlanningMode,
    EstimationApproach,
    ContextSource,
)

from .protocols import (
    AIPlannerProtocol,
    TaskBreakdownEngine,
    DependencyAnalyzer,
    ComplexityEstimator,
)

from .planner import (
    OllamaPlanner,
    PlanningError,
    ModelNotAvailableError,
    InvalidPlanningResponseError,
    ContextRetrievalError,
)

__version__ = "1.0.0"

__all__ = [
    # Models
    "TaskBreakdownRequest",
    "PlanGenerationRequest",
    "EstimationRequest",
    "AITaskSuggestion",
    "TaskDependencySuggestion",
    "ComplexityEstimation",
    "PlanningResult",
    "AIPlannerConfig",
    "PromptTemplate",
    "PlanningMode",
    "EstimationApproach",
    "ContextSource",

    # Protocols
    "AIPlannerProtocol",
    "TaskBreakdownEngine",
    "DependencyAnalyzer",
    "ComplexityEstimator",

    # Implementation
    "OllamaPlanner",
    "PlanningError",
    "ModelNotAvailableError",
    "InvalidPlanningResponseError",
    "ContextRetrievalError",
]