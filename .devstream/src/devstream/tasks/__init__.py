"""
DevStream Task Management System

A comprehensive task management system with AI-powered planning, dependency resolution,
and memory integration for the DevStream project.

Key Components:
- MicroTask: Granular tasks (max 10 minutes)
- Phase: Logical groupings of related tasks
- InterventionPlan: High-level plans with multiple phases
- TaskDependencyGraph: Dependency management with cycle detection
- TaskEngine: Core task execution engine
- TaskService: High-level service layer with workflow templates
- TaskRepository: Database integration layer

Based on Context7 research:
- Claude Task Master: dependency resolution patterns
- PlanAI: graph-based workflow architecture
- DevStream methodology: micro-task granularity
"""

from .models import (
    # Core Models
    MicroTask,
    Phase,
    InterventionPlan,
    TaskDependencyGraph,

    # Enums
    TaskStatus,
    TaskPriority,
    TaskType,
    TaskComplexity,
)

from .engine import (
    TaskEngine,
    TaskEngineConfig,
    TaskEngineError,
    ValidationError,
    DependencyError,
    ExecutionError,
)

from .repository import (
    TaskRepository,
    TaskRepositoryError,
    TaskNotFoundError,
    DependencyCycleError,
)

from .service import (
    TaskService,
    TaskServiceError,
    WorkflowError,
    PlanningError,
    TaskWorkflowTemplate,
)

__all__ = [
    # Models
    "MicroTask",
    "Phase",
    "InterventionPlan",
    "TaskDependencyGraph",
    "TaskStatus",
    "TaskPriority",
    "TaskType",
    "TaskComplexity",

    # Engine
    "TaskEngine",
    "TaskEngineConfig",
    "TaskEngineError",
    "ValidationError",
    "DependencyError",
    "ExecutionError",

    # Repository
    "TaskRepository",
    "TaskRepositoryError",
    "TaskNotFoundError",
    "DependencyCycleError",

    # Service
    "TaskService",
    "TaskServiceError",
    "WorkflowError",
    "PlanningError",
    "TaskWorkflowTemplate",
]