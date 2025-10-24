"""
Task Management Models for DevStream

Foundation models for task management system based on Context7 research:
- Claude Task Master: dependency resolution patterns
- PlanAI: graph-based workflow architecture
- DevStream methodology: micro-task granularity with AI assistance

Core Components:
- InterventionPlan: High-level objective with timeline
- Phase: Logical breakdown of plan
- MicroTask: Granular task (max 10 minutes)
- Dependencies: Graph structure management
- TaskStatus: Lifecycle management
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class TaskStatus(str, Enum):
    """Task lifecycle status based on Claude Task Master patterns"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Task priority levels for execution ordering"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskType(str, Enum):
    """Task type classification aligned with database constraints"""
    ANALYSIS = "analysis"
    CODING = "coding"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    REVIEW = "review"
    RESEARCH = "research"


class TaskComplexity(str, Enum):
    """Task complexity indicators for granularity validation"""
    TRIVIAL = "trivial"      # 1-2 minutes
    SIMPLE = "simple"        # 3-5 minutes
    MODERATE = "moderate"    # 6-8 minutes
    COMPLEX = "complex"      # 9-10 minutes (max allowed)


class MicroTask(BaseModel):
    """
    Granular task unit - max 10 minutes execution time
    Based on DevStream methodology for micro-task management
    """

    # Core Identity
    id: UUID = Field(default_factory=uuid4, description="Unique task identifier")
    title: str = Field(..., min_length=5, max_length=200, description="Clear, actionable task title")
    description: str = Field(..., min_length=10, max_length=1000, description="Detailed task description")

    # Classification
    task_type: TaskType = Field(..., description="Task type for categorization")
    complexity: TaskComplexity = Field(default=TaskComplexity.SIMPLE, description="Complexity indicator")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Execution priority")

    # Status Management
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task status")
    created_at: datetime = Field(default_factory=datetime.now, description="Task creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    started_at: Optional[datetime] = Field(None, description="Task start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Task completion timestamp")

    # Execution Context
    estimated_minutes: int = Field(default=5, ge=1, le=10, description="Estimated execution time in minutes")
    actual_minutes: Optional[int] = Field(None, ge=0, description="Actual execution time in minutes")
    assignee: Optional[str] = Field(None, description="Task assignee (user or agent)")

    # Dependency Management
    depends_on: Set[UUID] = Field(default_factory=set, description="Task dependencies (UUIDs)")
    blocks: Set[UUID] = Field(default_factory=set, description="Tasks blocked by this task")

    # Context and Memory
    keywords: List[str] = Field(default_factory=list, description="Keywords for memory search")
    context_tags: List[str] = Field(default_factory=list, description="Context tags for categorization")
    related_memory_ids: List[str] = Field(default_factory=list, description="Related memory entries")

    # Implementation Details
    implementation_notes: Optional[str] = Field(None, max_length=2000, description="Implementation details")
    acceptance_criteria: List[str] = Field(default_factory=list, description="Task completion criteria")
    output_artifacts: List[str] = Field(default_factory=list, description="Expected output artifacts")

    # AI Integration
    ai_generated: bool = Field(default=False, description="Whether task was AI-generated")
    ai_suggestions: Optional[Dict] = Field(None, description="AI suggestions for task execution")
    validation_passed: bool = Field(default=False, description="Whether task passed granularity validation")

    @field_validator('title')
    @classmethod
    def validate_title_actionable(cls, v):
        """Ensure title is actionable (starts with verb)"""
        action_verbs = {
            'implement', 'create', 'add', 'remove', 'update', 'fix', 'test',
            'document', 'refactor', 'deploy', 'configure', 'setup', 'build',
            'analyze', 'research', 'investigate', 'review', 'validate'
        }
        first_word = v.lower().split()[0] if v.split() else ""
        if first_word not in action_verbs:
            raise ValueError(f"Title must start with action verb. Use: {', '.join(sorted(action_verbs))}")
        return v

    @field_validator('estimated_minutes')
    @classmethod
    def validate_micro_task_constraint(cls, v, info):
        """Enforce micro-task time constraint"""
        if v > 10:
            raise ValueError("MicroTask must be completable in 10 minutes or less")

        # Adjust based on complexity if available
        if info.data and 'complexity' in info.data:
            complexity = info.data['complexity']
            complexity_limits = {
                TaskComplexity.TRIVIAL: 2,
                TaskComplexity.SIMPLE: 5,
                TaskComplexity.MODERATE: 8,
                TaskComplexity.COMPLEX: 10
            }

            if v > complexity_limits[complexity]:
                raise ValueError(f"Estimated time {v}min exceeds {complexity.value} complexity limit of {complexity_limits[complexity]}min")

        return v

    @model_validator(mode='after')
    def validate_status_timestamps(self):
        """Validate timestamp consistency based on status"""
        if self.status == TaskStatus.IN_PROGRESS and not self.started_at:
            self.started_at = datetime.now()

        if self.status == TaskStatus.COMPLETED:
            if not self.started_at:
                self.started_at = self.created_at
            if not self.completed_at:
                self.completed_at = datetime.now()

        return self

    def mark_in_progress(self) -> None:
        """Mark task as in progress"""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()
        self.updated_at = datetime.now()

    def mark_completed(self, actual_minutes: Optional[int] = None) -> None:
        """Mark task as completed"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()
        if actual_minutes is not None:
            self.actual_minutes = actual_minutes

    def mark_blocked(self, reason: Optional[str] = None) -> None:
        """Mark task as blocked"""
        self.status = TaskStatus.BLOCKED
        self.updated_at = datetime.now()
        if reason and self.implementation_notes:
            self.implementation_notes += f"\n[BLOCKED]: {reason}"
        elif reason:
            self.implementation_notes = f"[BLOCKED]: {reason}"

    def add_dependency(self, task_id: UUID) -> None:
        """Add a dependency to this task"""
        self.depends_on.add(task_id)
        self.updated_at = datetime.now()

    def remove_dependency(self, task_id: UUID) -> None:
        """Remove a dependency from this task"""
        self.depends_on.discard(task_id)
        self.updated_at = datetime.now()

    def is_ready_to_start(self, completed_tasks: Set[UUID]) -> bool:
        """Check if all dependencies are completed"""
        return self.depends_on.issubset(completed_tasks)

    def get_execution_time(self) -> Optional[timedelta]:
        """Get actual execution time if task is completed"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
            set: lambda v: list(v)
        }


class Phase(BaseModel):
    """
    Logical grouping of related MicroTasks
    Based on PlanAI workflow patterns for task organization
    """

    # Core Identity
    id: UUID = Field(default_factory=uuid4, description="Unique phase identifier")
    name: str = Field(..., min_length=5, max_length=100, description="Phase name")
    description: str = Field(..., min_length=10, max_length=500, description="Phase description")
    objective: str = Field(..., min_length=10, max_length=300, description="Phase objective")

    # Phase Organization
    order_index: int = Field(..., ge=1, description="Phase execution order")
    tasks: List[UUID] = Field(default_factory=list, description="MicroTask IDs in this phase")

    # Status Management
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Phase overall status")
    created_at: datetime = Field(default_factory=datetime.now, description="Phase creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    started_at: Optional[datetime] = Field(None, description="Phase start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Phase completion timestamp")

    # Estimates and Tracking
    estimated_hours: float = Field(default=1.0, ge=0.1, le=40.0, description="Estimated hours for phase")
    actual_hours: Optional[float] = Field(None, ge=0.0, description="Actual hours spent")

    # Dependencies
    depends_on_phases: Set[UUID] = Field(default_factory=set, description="Phase dependencies")

    # Context
    success_criteria: List[str] = Field(default_factory=list, description="Phase completion criteria")
    deliverables: List[str] = Field(default_factory=list, description="Expected phase deliverables")
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")

    def add_task(self, task_id: UUID) -> None:
        """Add a task to this phase"""
        if task_id not in self.tasks:
            self.tasks.append(task_id)
            self.updated_at = datetime.now()

    def remove_task(self, task_id: UUID) -> None:
        """Remove a task from this phase"""
        if task_id in self.tasks:
            self.tasks.remove(task_id)
            self.updated_at = datetime.now()

    def calculate_progress(self, task_statuses: Dict[UUID, TaskStatus]) -> float:
        """Calculate phase completion percentage"""
        if not self.tasks:
            return 0.0

        completed_count = sum(
            1 for task_id in self.tasks
            if task_statuses.get(task_id) == TaskStatus.COMPLETED
        )
        return (completed_count / len(self.tasks)) * 100.0

    def is_ready_to_start(self, completed_phases: Set[UUID]) -> bool:
        """Check if all phase dependencies are completed"""
        return self.depends_on_phases.issubset(completed_phases)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
            set: lambda v: list(v)
        }


class InterventionPlan(BaseModel):
    """
    High-level plan containing multiple phases
    Based on DevStream intervention planning methodology
    """

    # Core Identity
    id: UUID = Field(default_factory=uuid4, description="Unique plan identifier")
    title: str = Field(..., min_length=10, max_length=200, description="Plan title")
    description: str = Field(..., min_length=20, max_length=2000, description="Detailed plan description")
    objective: str = Field(..., min_length=10, max_length=500, description="Primary plan objective")

    # Plan Organization
    phases: List[UUID] = Field(default_factory=list, description="Phase IDs in execution order")
    total_tasks: int = Field(default=0, ge=0, description="Total number of tasks across all phases")

    # Timeline Management
    created_at: datetime = Field(default_factory=datetime.now, description="Plan creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    started_at: Optional[datetime] = Field(None, description="Plan start timestamp")
    target_completion: Optional[datetime] = Field(None, description="Target completion date")
    completed_at: Optional[datetime] = Field(None, description="Actual completion timestamp")

    # Status and Progress
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Overall plan status")
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Completion percentage")

    # Estimates
    estimated_hours: float = Field(default=8.0, ge=0.5, le=200.0, description="Total estimated hours")
    actual_hours: Optional[float] = Field(None, ge=0.0, description="Actual hours spent")

    # Context and Metadata
    category: str = Field(default="development", description="Plan category")
    tags: List[str] = Field(default_factory=list, description="Plan tags for organization")
    stakeholders: List[str] = Field(default_factory=list, description="Plan stakeholders")

    # AI Integration
    ai_generated: bool = Field(default=False, description="Whether plan was AI-generated")
    ai_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="AI generation confidence")
    human_approved: bool = Field(default=False, description="Whether plan was human-approved")

    # Success Metrics
    success_criteria: List[str] = Field(default_factory=list, description="Plan success criteria")
    key_deliverables: List[str] = Field(default_factory=list, description="Key plan deliverables")
    quality_gates: List[str] = Field(default_factory=list, description="Quality checkpoints")

    def add_phase(self, phase_id: UUID) -> None:
        """Add a phase to this plan"""
        if phase_id not in self.phases:
            self.phases.append(phase_id)
            self.updated_at = datetime.now()

    def remove_phase(self, phase_id: UUID) -> None:
        """Remove a phase from this plan"""
        if phase_id in self.phases:
            self.phases.remove(phase_id)
            self.updated_at = datetime.now()

    def calculate_overall_progress(self, phase_progresses: Dict[UUID, float]) -> float:
        """Calculate overall plan progress from phase progresses"""
        if not self.phases:
            return 0.0

        total_progress = sum(
            phase_progresses.get(phase_id, 0.0) for phase_id in self.phases
        )
        self.progress_percentage = total_progress / len(self.phases)
        return self.progress_percentage

    def get_current_phase(self, phase_statuses: Dict[UUID, TaskStatus]) -> Optional[UUID]:
        """Get the current active phase"""
        for phase_id in self.phases:
            status = phase_statuses.get(phase_id, TaskStatus.PENDING)
            if status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]:
                return phase_id
        return None

    def estimate_completion_date(self, hours_per_day: float = 6.0) -> Optional[datetime]:
        """Estimate completion date based on remaining work"""
        if self.progress_percentage >= 100.0:
            return self.completed_at

        remaining_hours = self.estimated_hours * (1 - self.progress_percentage / 100.0)
        remaining_days = remaining_hours / hours_per_day

        return datetime.now() + timedelta(days=remaining_days)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }


class TaskDependencyGraph(BaseModel):
    """
    Dependency graph management based on Claude Task Master patterns
    Handles cycle detection and topological sorting
    """

    tasks: Dict[UUID, MicroTask] = Field(default_factory=dict, description="Task registry")
    adjacency_list: Dict[UUID, Set[UUID]] = Field(default_factory=dict, description="Dependency graph")
    reverse_adjacency: Dict[UUID, Set[UUID]] = Field(default_factory=dict, description="Reverse dependency graph")

    def add_task(self, task: MicroTask) -> None:
        """Add a task to the dependency graph"""
        self.tasks[task.id] = task
        if task.id not in self.adjacency_list:
            self.adjacency_list[task.id] = set()
        if task.id not in self.reverse_adjacency:
            self.reverse_adjacency[task.id] = set()

        # Update dependencies
        for dep_id in task.depends_on:
            self.add_dependency(dep_id, task.id)

    def add_dependency(self, from_task: UUID, to_task: UUID) -> bool:
        """Add a dependency edge (from_task -> to_task)"""
        # Check for cycle
        if self._would_create_cycle(from_task, to_task):
            return False

        self.adjacency_list.setdefault(from_task, set()).add(to_task)
        self.reverse_adjacency.setdefault(to_task, set()).add(from_task)

        # Update task objects
        if to_task in self.tasks:
            self.tasks[to_task].add_dependency(from_task)

        return True

    def remove_dependency(self, from_task: UUID, to_task: UUID) -> None:
        """Remove a dependency edge"""
        self.adjacency_list.get(from_task, set()).discard(to_task)
        self.reverse_adjacency.get(to_task, set()).discard(from_task)

        # Update task objects
        if to_task in self.tasks:
            self.tasks[to_task].remove_dependency(from_task)

    def _would_create_cycle(self, from_task: UUID, to_task: UUID) -> bool:
        """Check if adding dependency would create a cycle"""
        if from_task == to_task:
            return True

        # DFS to check if to_task can reach from_task
        visited = set()
        stack = [to_task]

        while stack:
            current = stack.pop()
            if current == from_task:
                return True

            if current in visited:
                continue
            visited.add(current)

            stack.extend(self.adjacency_list.get(current, set()))

        return False

    def get_ready_tasks(self, completed_tasks: Set[UUID]) -> List[MicroTask]:
        """Get tasks that are ready to execute (all dependencies completed)"""
        ready_tasks = []

        for task_id, task in self.tasks.items():
            if (task.status == TaskStatus.PENDING and
                task.is_ready_to_start(completed_tasks)):
                ready_tasks.append(task)

        # Sort by priority and creation time
        ready_tasks.sort(key=lambda t: (
            t.priority == TaskPriority.CRITICAL,
            t.priority == TaskPriority.HIGH,
            t.priority == TaskPriority.MEDIUM,
            t.created_at
        ), reverse=True)

        return ready_tasks

    def topological_sort(self) -> List[UUID]:
        """Return topologically sorted task order"""
        in_degree = {task_id: len(deps) for task_id, deps in self.reverse_adjacency.items()}
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for neighbor in self.adjacency_list.get(current, set()):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def get_critical_path(self) -> List[UUID]:
        """Calculate critical path through the dependency graph"""
        # Simplified critical path - tasks with longest dependency chain
        max_depth = {}

        def calculate_depth(task_id: UUID) -> int:
            if task_id in max_depth:
                return max_depth[task_id]

            dependencies = self.reverse_adjacency.get(task_id, set())
            if not dependencies:
                max_depth[task_id] = 0
                return 0

            depth = 1 + max(calculate_depth(dep) for dep in dependencies)
            max_depth[task_id] = depth
            return depth

        for task_id in self.tasks:
            calculate_depth(task_id)

        # Find path with maximum depth
        critical_tasks = sorted(max_depth.items(), key=lambda x: x[1], reverse=True)
        return [task_id for task_id, _ in critical_tasks]

    class Config:
        json_encoders = {
            UUID: lambda v: str(v),
            set: lambda v: list(v)
        }