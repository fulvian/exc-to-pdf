"""
AI Planning Models

Pydantic v2 models for AI-powered task planning and breakdown.
Based on Context7 research findings for Ollama + embeddinggemma integration.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class PlanningMode(str, Enum):
    """Planning approach for task breakdown."""
    DETAILED = "detailed"  # Comprehensive micro-task breakdown
    SUMMARY = "summary"    # High-level phase breakdown
    HYBRID = "hybrid"      # Adaptive based on complexity


class EstimationApproach(str, Enum):
    """Approach for time/complexity estimation."""
    CONSERVATIVE = "conservative"  # Add buffer time, higher complexity
    REALISTIC = "realistic"        # Best estimate based on analysis
    OPTIMISTIC = "optimistic"      # Minimal time, lower complexity
    ADAPTIVE = "adaptive"          # Context-aware estimation


class ContextSource(str, Enum):
    """Source of context for planning."""
    MEMORY = "memory"              # Semantic memory retrieval
    PROVIDED = "provided"          # User-provided context
    MIXED = "mixed"                # Memory + provided context
    MINIMAL = "minimal"            # No additional context


class TaskBreakdownRequest(BaseModel):
    """Request for AI-powered task breakdown."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    objective: str = Field(..., min_length=10, max_length=500)
    context: Optional[str] = Field(None, max_length=2000)
    planning_mode: PlanningMode = Field(default=PlanningMode.DETAILED)
    estimation_approach: EstimationApproach = Field(default=EstimationApproach.REALISTIC)
    context_source: ContextSource = Field(default=ContextSource.MIXED)

    # Constraints
    max_tasks: int = Field(default=15, ge=3, le=50)
    max_task_duration_minutes: int = Field(default=10, ge=5, le=60)
    target_complexity: Optional[int] = Field(None, ge=1, le=10)

    # Context options
    include_memory_context: bool = Field(default=True)
    memory_context_limit: int = Field(default=1000, ge=100, le=5000)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator('objective')
    @classmethod
    def validate_objective_actionable(cls, v: str) -> str:
        """Ensure objective starts with action verb."""
        action_verbs = {
            'implement', 'create', 'build', 'develop', 'design', 'add',
            'remove', 'update', 'fix', 'test', 'deploy', 'configure',
            'setup', 'analyze', 'research', 'investigate', 'review',
            'validate', 'optimize', 'refactor', 'integrate', 'migrate'
        }

        first_word = v.lower().split()[0] if v.split() else ""
        if first_word not in action_verbs:
            raise ValueError(
                f"Objective must start with action verb. Use: {', '.join(sorted(action_verbs))}"
            )
        return v


class PlanGenerationRequest(BaseModel):
    """Request for generating a complete intervention plan."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., min_length=5, max_length=200)
    description: str = Field(..., min_length=20, max_length=1000)
    objectives: List[str] = Field(..., min_items=1, max_items=5)

    # Planning parameters
    planning_mode: PlanningMode = Field(default=PlanningMode.HYBRID)
    estimation_approach: EstimationApproach = Field(default=EstimationApproach.REALISTIC)
    context_source: ContextSource = Field(default=ContextSource.MIXED)

    # Constraints
    max_phases: int = Field(default=5, ge=1, le=10)
    max_tasks_per_phase: int = Field(default=10, ge=3, le=20)
    target_duration_hours: Optional[float] = Field(None, ge=1.0, le=100.0)

    # Context
    provided_context: Optional[str] = Field(None, max_length=3000)
    include_memory_context: bool = Field(default=True)
    memory_context_limit: int = Field(default=2000, ge=500, le=10000)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)


class EstimationRequest(BaseModel):
    """Request for task estimation (complexity, duration)."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_title: str = Field(..., min_length=5, max_length=200)
    task_description: str = Field(..., min_length=10, max_length=1000)

    # Estimation parameters
    approach: EstimationApproach = Field(default=EstimationApproach.REALISTIC)
    context_source: ContextSource = Field(default=ContextSource.MIXED)

    # Context for estimation
    project_context: Optional[str] = Field(None, max_length=1500)
    similar_tasks_context: Optional[str] = Field(None, max_length=1000)

    # Reference for calibration
    reference_tasks: List[Dict[str, Any]] = Field(default_factory=list, max_items=5)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)


class AITaskSuggestion(BaseModel):
    """AI-generated task suggestion with metadata."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., min_length=5, max_length=200)
    description: str = Field(..., min_length=10, max_length=1000)

    # AI-generated estimates
    estimated_minutes: int = Field(..., ge=1, le=60)
    complexity_score: int = Field(..., ge=1, le=10)
    priority_score: float = Field(..., ge=0.0, le=1.0)

    # Task categorization
    task_type: str = Field(...)  # Will map to TaskType enum
    suggested_phase: Optional[str] = Field(None)

    # Dependencies
    prerequisite_tasks: List[str] = Field(default_factory=list)
    dependent_tasks: List[str] = Field(default_factory=list)

    # AI confidence and reasoning
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=20, max_length=500)

    # Context used for generation
    context_sources: List[ContextSource] = Field(default_factory=list)
    memory_references: List[str] = Field(default_factory=list)

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now)
    model_used: Optional[str] = Field(None)

    @field_validator('title')
    @classmethod
    def validate_title_actionable(cls, v: str) -> str:
        """Ensure task title starts with action verb."""
        action_verbs = {
            'implement', 'create', 'add', 'remove', 'update', 'fix',
            'test', 'document', 'refactor', 'deploy', 'configure',
            'setup', 'build', 'analyze', 'research', 'investigate',
            'review', 'validate', 'optimize', 'integrate', 'migrate'
        }

        first_word = v.lower().split()[0] if v.split() else ""
        if first_word not in action_verbs:
            raise ValueError(
                f"Task title must start with action verb. Use: {', '.join(sorted(action_verbs))}"
            )
        return v

    @model_validator(mode='after')
    def validate_task_granularity(self) -> 'AITaskSuggestion':
        """Ensure task meets micro-task criteria."""
        if self.estimated_minutes > 10:
            raise ValueError(
                f"Task duration {self.estimated_minutes} exceeds 10-minute micro-task limit"
            )

        # Check description length suggests appropriate granularity
        if len(self.description) > 800:
            raise ValueError(
                "Task description too long - suggests task should be broken down further"
            )

        return self


class TaskDependencySuggestion(BaseModel):
    """AI-suggested dependency between tasks."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prerequisite_task_id: str = Field(...)
    dependent_task_id: str = Field(...)

    # Dependency metadata
    dependency_type: str = Field(...)  # 'blocking', 'soft', 'parallel_possible'
    strength: float = Field(..., ge=0.0, le=1.0)  # How critical the dependency is

    # AI reasoning
    reasoning: str = Field(..., min_length=10, max_length=300)
    confidence_score: float = Field(..., ge=0.0, le=1.0)

    # Context
    detected_from: List[str] = Field(default_factory=list)  # Sources of detection

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now)

    @model_validator(mode='after')
    def validate_no_self_dependency(self) -> 'TaskDependencySuggestion':
        """Ensure task doesn't depend on itself."""
        if self.prerequisite_task_id == self.dependent_task_id:
            raise ValueError("Task cannot depend on itself")
        return self


class ComplexityEstimation(BaseModel):
    """AI-generated complexity and duration estimation."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_reference: str = Field(...)  # Task ID or description

    # Core estimates
    estimated_minutes: int = Field(..., ge=1, le=60)
    complexity_score: int = Field(..., ge=1, le=10)
    uncertainty_factor: float = Field(..., ge=0.0, le=1.0)

    # Detailed breakdown
    analysis_factors: Dict[str, float] = Field(default_factory=dict)
    risk_factors: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)

    # Confidence and reasoning
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    estimation_reasoning: str = Field(..., min_length=20, max_length=500)

    # Context used
    similar_tasks: List[str] = Field(default_factory=list)
    reference_data: Dict[str, Any] = Field(default_factory=dict)

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now)
    approach_used: EstimationApproach = Field(...)
    model_used: Optional[str] = Field(None)


class PlanningResult(BaseModel):
    """Complete result from AI planning operation."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = Field(...)  # Reference to original request

    # Generated content
    suggested_tasks: List[AITaskSuggestion] = Field(default_factory=list)
    suggested_dependencies: List[TaskDependencySuggestion] = Field(default_factory=list)
    complexity_estimations: List[ComplexityEstimation] = Field(default_factory=list)

    # Planning metadata
    total_estimated_minutes: int = Field(default=0, ge=0)
    average_complexity: float = Field(default=0.0, ge=0.0, le=10.0)
    suggested_phases: List[str] = Field(default_factory=list)

    # AI analysis
    planning_confidence: float = Field(..., ge=0.0, le=1.0)
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    planning_reasoning: str = Field(..., min_length=20, max_length=1000)

    # Context used
    memory_context_used: Optional[str] = Field(None)
    provided_context_used: Optional[str] = Field(None)
    context_sources: List[ContextSource] = Field(default_factory=list)

    # Quality metrics
    task_count: int = Field(default=0, ge=0)
    dependency_count: int = Field(default=0, ge=0)
    cycles_detected: int = Field(default=0, ge=0)

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now)
    generation_duration_ms: Optional[int] = Field(None, ge=0)
    model_used: Optional[str] = Field(None)

    @model_validator(mode='after')
    def validate_planning_consistency(self) -> 'PlanningResult':
        """Ensure planning result is internally consistent."""
        # Update computed fields
        self.task_count = len(self.suggested_tasks)
        self.dependency_count = len(self.suggested_dependencies)

        if self.suggested_tasks:
            self.total_estimated_minutes = sum(task.estimated_minutes for task in self.suggested_tasks)
            self.average_complexity = sum(task.complexity_score for task in self.suggested_tasks) / len(self.suggested_tasks)

        # Validate dependency references
        task_ids = {task.id for task in self.suggested_tasks}
        for dep in self.suggested_dependencies:
            if dep.prerequisite_task_id not in task_ids:
                raise ValueError(f"Dependency references unknown prerequisite task: {dep.prerequisite_task_id}")
            if dep.dependent_task_id not in task_ids:
                raise ValueError(f"Dependency references unknown dependent task: {dep.dependent_task_id}")

        return self


class AIPlannerConfig(BaseModel):
    """Configuration for AI planner."""

    # Model configuration
    model_name: str = Field(default="gemma2")
    embedding_model: str = Field(default="embeddinggemma")

    # Planning parameters
    default_planning_mode: PlanningMode = Field(default=PlanningMode.DETAILED)
    default_estimation_approach: EstimationApproach = Field(default=EstimationApproach.REALISTIC)
    default_context_source: ContextSource = Field(default=ContextSource.MIXED)

    # Context limits
    max_context_tokens: int = Field(default=4000, ge=1000, le=10000)
    memory_context_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    provided_context_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    # Generation parameters
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    max_retries: int = Field(default=3, ge=1, le=5)
    timeout_seconds: int = Field(default=30, ge=10, le=120)

    # Quality thresholds
    min_confidence_score: float = Field(default=0.6, ge=0.0, le=1.0)
    min_completeness_score: float = Field(default=0.7, ge=0.0, le=1.0)

    # Fallback configuration
    enable_fallback: bool = Field(default=True)
    fallback_task_duration: int = Field(default=10, ge=5, le=30)
    fallback_complexity: int = Field(default=5, ge=1, le=10)

    @model_validator(mode='after')
    def validate_context_weights(self) -> 'AIPlannerConfig':
        """Ensure context weights sum to reasonable total."""
        total_weight = self.memory_context_weight + self.provided_context_weight
        if abs(total_weight - 1.0) > 0.1:
            raise ValueError(f"Context weights should sum to ~1.0, got {total_weight}")
        return self


class PromptTemplate(BaseModel):
    """Template for AI prompts."""

    name: str = Field(...)
    description: str = Field(...)
    template: str = Field(..., min_length=50)

    # Template parameters
    required_variables: List[str] = Field(default_factory=list)
    optional_variables: List[str] = Field(default_factory=list)

    # Usage metadata
    planning_mode: Optional[PlanningMode] = Field(None)
    operation_type: str = Field(...)  # 'breakdown', 'estimation', 'dependency'

    # Quality metadata
    success_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    average_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=datetime.now)

    @field_validator('template')
    @classmethod
    def validate_template_format(cls, v: str) -> str:
        """Ensure template contains required placeholders."""
        if '{objective}' not in v and '{task}' not in v:
            raise ValueError("Template must contain either {objective} or {task} placeholder")
        return v