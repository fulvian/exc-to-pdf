"""
Task Engine Core - DevStream Task Management Engine

A comprehensive task management engine that orchestrates task creation, execution,
and tracking with AI assistance and memory integration.

Key Features:
- Task lifecycle management with validation
- Dependency resolution with cycle detection
- Progress tracking and status management
- Integration with memory system for context
- AI-assisted task planning and breakdown
- Event-driven architecture for extensibility

Based on Context7 research:
- Claude Task Master: dependency resolution patterns
- PlanAI: graph-based workflow orchestration
- DevStream methodology: micro-task granularity
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from uuid import UUID, uuid4

from devstream.tasks.models import (
    InterventionPlan, Phase, MicroTask, TaskDependencyGraph,
    TaskStatus, TaskPriority, TaskType, TaskComplexity
)
from devstream.tasks.repository import TaskRepository, TaskRepositoryError
from devstream.memory.storage import MemoryStorage
from devstream.memory.models import MemoryEntry, ContentType
from devstream.exceptions import DevStreamError


class TaskEngineError(DevStreamError):
    """Task engine specific errors"""
    pass


class ValidationError(TaskEngineError):
    """Task validation errors"""
    pass


class DependencyError(TaskEngineError):
    """Dependency management errors"""
    pass


class ExecutionError(TaskEngineError):
    """Task execution errors"""
    pass


class TaskEngineConfig:
    """Configuration for Task Engine behavior"""

    def __init__(
        self,
        max_concurrent_tasks: int = 3,
        auto_dependency_detection: bool = True,
        strict_granularity_validation: bool = True,
        memory_integration_enabled: bool = True,
        progress_auto_update: bool = True,
        dependency_cycle_detection: bool = True
    ):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.auto_dependency_detection = auto_dependency_detection
        self.strict_granularity_validation = strict_granularity_validation
        self.memory_integration_enabled = memory_integration_enabled
        self.progress_auto_update = progress_auto_update
        self.dependency_cycle_detection = dependency_cycle_detection


class TaskEngine:
    """
    Core Task Management Engine

    Orchestrates all task operations with dependency management,
    progress tracking, and memory integration.
    """

    def __init__(
        self,
        repository: Optional[TaskRepository] = None,
        memory_storage: Optional[MemoryStorage] = None,
        config: Optional[TaskEngineConfig] = None
    ):
        self.repository = repository or TaskRepository()
        self.memory_storage = memory_storage
        self.config = config or TaskEngineConfig()

        # Runtime state
        self._active_tasks: Dict[UUID, MicroTask] = {}
        self._execution_locks: Dict[UUID, asyncio.Lock] = {}
        self._event_handlers: Dict[str, List] = {}

    # ============================================================================
    # INTERVENTION PLAN OPERATIONS
    # ============================================================================

    async def create_intervention_plan(
        self,
        title: str,
        description: str,
        objective: str,
        estimated_hours: float = 8.0,
        category: str = "development",
        tags: Optional[List[str]] = None,
        success_criteria: Optional[List[str]] = None
    ) -> InterventionPlan:
        """Create a new intervention plan with validation"""

        # Create plan model
        plan = InterventionPlan(
            title=title,
            description=description,
            objective=objective,
            estimated_hours=estimated_hours,
            category=category,
            tags=tags or [],
            success_criteria=success_criteria or []
        )

        # Validate plan
        await self._validate_intervention_plan(plan)

        # Save to repository
        saved_plan = await self.repository.create_intervention_plan(plan)

        # Store in memory if enabled
        if self.config.memory_integration_enabled and self.memory_storage:
            await self._store_plan_in_memory(saved_plan)

        # Emit event
        await self._emit_event("plan_created", {"plan": saved_plan})

        return saved_plan

    async def get_intervention_plan(self, plan_id: UUID) -> Optional[InterventionPlan]:
        """Get intervention plan by ID"""
        return await self.repository.get_intervention_plan(plan_id)

    async def update_intervention_plan(self, plan: InterventionPlan) -> InterventionPlan:
        """Update existing intervention plan"""
        # Validate updates
        await self._validate_intervention_plan(plan)

        # Update in repository
        updated_plan = await self.repository.update_intervention_plan(plan)

        # Update progress if auto-update enabled
        if self.config.progress_auto_update:
            await self._update_plan_progress(updated_plan)

        # Emit event
        await self._emit_event("plan_updated", {"plan": updated_plan})

        return updated_plan

    async def delete_intervention_plan(self, plan_id: UUID) -> bool:
        """Delete intervention plan and all associated data"""
        # Get plan for event emission
        plan = await self.repository.get_intervention_plan(plan_id)

        # Delete from repository (cascades to phases and tasks)
        success = await self.repository.delete_intervention_plan(plan_id)

        if success and plan:
            await self._emit_event("plan_deleted", {"plan": plan})

        return success

    # ============================================================================
    # PHASE OPERATIONS
    # ============================================================================

    async def create_phase(
        self,
        plan_id: UUID,
        name: str,
        description: str,
        objective: str,
        order_index: int,
        estimated_hours: float = 2.0,
        depends_on_phases: Optional[Set[UUID]] = None
    ) -> Phase:
        """Create a new phase in a plan"""

        # Create phase model
        phase = Phase(
            name=name,
            description=description,
            objective=objective,
            order_index=order_index,
            estimated_hours=estimated_hours,
            depends_on_phases=depends_on_phases or set()
        )

        # Validate phase
        await self._validate_phase(phase, plan_id)

        # Save to repository
        saved_phase = await self.repository.create_phase(phase, plan_id)

        # Update plan
        plan = await self.repository.get_intervention_plan(plan_id)
        if plan:
            plan.add_phase(saved_phase.id)
            await self.repository.update_intervention_plan(plan)

        # Emit event
        await self._emit_event("phase_created", {"phase": saved_phase, "plan_id": plan_id})

        return saved_phase

    async def get_phases_for_plan(self, plan_id: UUID) -> List[Phase]:
        """Get all phases for a plan, ordered by sequence"""
        return await self.repository.get_phases_for_plan(plan_id)

    async def update_phase(self, phase: Phase) -> Phase:
        """Update existing phase"""
        # Validate updates
        await self._validate_phase_update(phase)

        # Update in repository
        updated_phase = await self.repository.update_phase(phase)

        # Update progress if auto-update enabled
        if self.config.progress_auto_update:
            await self._update_phase_progress(updated_phase)

        # Emit event
        await self._emit_event("phase_updated", {"phase": updated_phase})

        return updated_phase

    # ============================================================================
    # MICRO TASK OPERATIONS
    # ============================================================================

    async def create_micro_task(
        self,
        phase_id: UUID,
        title: str,
        description: str,
        task_type: TaskType,
        complexity: TaskComplexity = TaskComplexity.SIMPLE,
        estimated_minutes: int = 5,
        priority: TaskPriority = TaskPriority.MEDIUM,
        depends_on: Optional[Set[UUID]] = None,
        keywords: Optional[List[str]] = None,
        acceptance_criteria: Optional[List[str]] = None
    ) -> MicroTask:
        """Create a new micro task with full validation"""

        # Create task model
        task = MicroTask(
            title=title,
            description=description,
            task_type=task_type,
            complexity=complexity,
            estimated_minutes=estimated_minutes,
            priority=priority,
            depends_on=depends_on or set(),
            keywords=keywords or [],
            acceptance_criteria=acceptance_criteria or []
        )

        # Validate task
        await self._validate_micro_task(task, phase_id)

        # Auto-detect dependencies if enabled
        if self.config.auto_dependency_detection:
            auto_deps = await self._detect_task_dependencies(task, phase_id)
            task.depends_on.update(auto_deps)

        # Validate dependencies won't create cycles
        if self.config.dependency_cycle_detection:
            await self._validate_task_dependencies(task, phase_id)

        # Save to repository
        saved_task = await self.repository.create_micro_task(task, phase_id)

        # Update phase
        phase = await self.repository.get_phase(phase_id)
        if phase:
            phase.add_task(saved_task.id)
            await self.repository.update_phase(phase)

        # Store in memory if enabled
        if self.config.memory_integration_enabled and self.memory_storage:
            await self._store_task_in_memory(saved_task)

        # Emit event
        await self._emit_event("task_created", {"task": saved_task, "phase_id": phase_id})

        return saved_task

    async def get_micro_task(self, task_id: UUID) -> Optional[MicroTask]:
        """Get micro task by ID"""
        return await self.repository.get_micro_task(task_id)

    async def update_micro_task(self, task: MicroTask) -> MicroTask:
        """Update existing micro task"""
        # Validate updates
        await self._validate_micro_task_update(task)

        # Update in repository
        updated_task = await self.repository.update_micro_task(task)

        # Update memory if enabled
        if self.config.memory_integration_enabled and self.memory_storage:
            await self._update_task_in_memory(updated_task)

        # Emit event
        await self._emit_event("task_updated", {"task": updated_task})

        return updated_task

    async def get_tasks_for_phase(self, phase_id: UUID) -> List[MicroTask]:
        """Get all tasks for a phase"""
        return await self.repository.get_tasks_for_phase(phase_id)

    # ============================================================================
    # TASK EXECUTION OPERATIONS
    # ============================================================================

    async def start_task(self, task_id: UUID, assignee: Optional[str] = None) -> MicroTask:
        """Start task execution with validation"""

        # Get task
        task = await self.repository.get_micro_task(task_id)
        if not task:
            raise ExecutionError(f"Task {task_id} not found")

        # Validate task can be started
        await self._validate_task_can_start(task)

        # Check if task is ready (dependencies completed)
        if not await self._is_task_ready(task):
            raise ExecutionError(f"Task {task_id} has unmet dependencies")

        # Acquire execution lock
        if task_id not in self._execution_locks:
            self._execution_locks[task_id] = asyncio.Lock()

        async with self._execution_locks[task_id]:
            # Update task status
            task.mark_in_progress()
            if assignee:
                task.assignee = assignee

            # Save updates
            updated_task = await self.repository.update_micro_task(task)

            # Add to active tasks
            self._active_tasks[task_id] = updated_task

            # Emit event
            await self._emit_event("task_started", {"task": updated_task})

            return updated_task

    async def complete_task(
        self,
        task_id: UUID,
        actual_minutes: Optional[int] = None,
        output_artifacts: Optional[List[str]] = None,
        completion_notes: Optional[str] = None
    ) -> MicroTask:
        """Complete task execution with validation"""

        # Get task
        task = await self.repository.get_micro_task(task_id)
        if not task:
            raise ExecutionError(f"Task {task_id} not found")

        # Validate task can be completed
        await self._validate_task_can_complete(task)

        async with self._execution_locks.get(task_id, asyncio.Lock()):
            # Update task with completion data
            task.mark_completed(actual_minutes)

            if output_artifacts:
                task.output_artifacts.extend(output_artifacts)

            if completion_notes:
                if task.implementation_notes:
                    task.implementation_notes += f"\n\n[COMPLETION]: {completion_notes}"
                else:
                    task.implementation_notes = f"[COMPLETION]: {completion_notes}"

            # Mark as validated
            task.validation_passed = True

            # Save updates
            updated_task = await self.repository.update_micro_task(task)

            # Remove from active tasks
            self._active_tasks.pop(task_id, None)

            # Update progress if auto-update enabled
            if self.config.progress_auto_update:
                await self._update_dependent_progress(updated_task)

            # Store completion in memory
            if self.config.memory_integration_enabled and self.memory_storage:
                await self._store_task_completion_in_memory(updated_task)

            # Emit event
            await self._emit_event("task_completed", {"task": updated_task})

            return updated_task

    async def block_task(
        self,
        task_id: UUID,
        reason: str,
        blocking_task_id: Optional[UUID] = None
    ) -> MicroTask:
        """Block task execution with reason"""

        task = await self.repository.get_micro_task(task_id)
        if not task:
            raise ExecutionError(f"Task {task_id} not found")

        # Update task status
        task.mark_blocked(reason)

        # Add blocking relationship if specified
        if blocking_task_id:
            task.add_dependency(blocking_task_id)

        # Save updates
        updated_task = await self.repository.update_micro_task(task)

        # Remove from active tasks
        self._active_tasks.pop(task_id, None)

        # Emit event
        await self._emit_event("task_blocked", {
            "task": updated_task,
            "reason": reason,
            "blocking_task_id": blocking_task_id
        })

        return updated_task

    # ============================================================================
    # DEPENDENCY MANAGEMENT
    # ============================================================================

    async def add_task_dependency(self, task_id: UUID, dependency_id: UUID) -> bool:
        """Add dependency between tasks with validation"""

        # Validate both tasks exist
        task = await self.repository.get_micro_task(task_id)
        dep_task = await self.repository.get_micro_task(dependency_id)

        if not task or not dep_task:
            raise DependencyError("One or both tasks not found")

        # Validate dependency won't create cycle
        if not await self.repository.validate_dependencies(task_id, [dependency_id]):
            raise DependencyError("Adding dependency would create cycle")

        # Add dependency
        task.add_dependency(dependency_id)

        # Update in repository
        await self.repository.update_micro_task(task)

        # Emit event
        await self._emit_event("dependency_added", {
            "task_id": task_id,
            "dependency_id": dependency_id
        })

        return True

    async def remove_task_dependency(self, task_id: UUID, dependency_id: UUID) -> bool:
        """Remove dependency between tasks"""

        task = await self.repository.get_micro_task(task_id)
        if not task:
            raise DependencyError(f"Task {task_id} not found")

        # Remove dependency
        task.remove_dependency(dependency_id)

        # Update in repository
        await self.repository.update_micro_task(task)

        # Emit event
        await self._emit_event("dependency_removed", {
            "task_id": task_id,
            "dependency_id": dependency_id
        })

        return True

    async def get_ready_tasks(self, plan_id: UUID) -> List[MicroTask]:
        """Get tasks that are ready to execute"""
        return await self.repository.get_ready_tasks(plan_id)

    async def build_dependency_graph(self, plan_id: UUID) -> TaskDependencyGraph:
        """Build complete dependency graph for a plan"""
        return await self.repository.build_dependency_graph(plan_id)

    # ============================================================================
    # PROGRESS TRACKING
    # ============================================================================

    async def get_plan_progress(self, plan_id: UUID) -> Dict[str, Any]:
        """Get comprehensive progress information for a plan"""

        # Get plan
        plan = await self.repository.get_intervention_plan(plan_id)
        if not plan:
            raise TaskEngineError(f"Plan {plan_id} not found")

        # Get phases
        phases = await self.repository.get_phases_for_plan(plan_id)

        # Get task statistics
        task_stats = await self.repository.get_task_statistics(plan_id)

        # Calculate phase progresses
        phase_progresses = {}
        for phase in phases:
            tasks = await self.repository.get_tasks_for_phase(phase.id)
            if tasks:
                task_statuses = {task.id: task.status for task in tasks}
                progress = phase.calculate_progress(task_statuses)
                phase_progresses[phase.id] = progress

        # Calculate overall progress
        overall_progress = plan.calculate_overall_progress(phase_progresses)

        # Get ready tasks
        ready_tasks = await self.get_ready_tasks(plan_id)

        # Get active tasks
        active_tasks = [task for task in self._active_tasks.values()
                       if any(task.id in phase.tasks for phase in phases)]

        return {
            "plan": plan,
            "phases": phases,
            "overall_progress": overall_progress,
            "phase_progresses": phase_progresses,
            "task_statistics": task_stats,
            "ready_tasks": ready_tasks,
            "active_tasks": active_tasks,
            "total_estimated_hours": sum(phase.estimated_hours for phase in phases),
            "total_actual_hours": sum(phase.actual_hours or 0 for phase in phases)
        }

    async def get_task_execution_metrics(self, plan_id: UUID) -> Dict[str, Any]:
        """Get detailed execution metrics for tasks"""

        # Build dependency graph
        graph = await self.build_dependency_graph(plan_id)

        # Get critical path
        critical_path = graph.get_critical_path()

        # Get topological order
        execution_order = graph.topological_sort()

        # Calculate completion estimates
        estimates = await self._calculate_completion_estimates(plan_id)

        return {
            "critical_path": critical_path,
            "execution_order": execution_order,
            "completion_estimates": estimates,
            "dependency_violations": await self._detect_dependency_violations(plan_id),
            "bottleneck_tasks": await self._identify_bottleneck_tasks(plan_id)
        }

    # ============================================================================
    # SEARCH AND QUERY OPERATIONS
    # ============================================================================

    async def search_tasks(
        self,
        query: str,
        task_types: Optional[List[TaskType]] = None,
        statuses: Optional[List[TaskStatus]] = None,
        plan_id: Optional[UUID] = None,
        limit: int = 50
    ) -> List[MicroTask]:
        """Search tasks with advanced filtering"""
        return await self.repository.search_tasks(
            query=query,
            task_types=task_types,
            statuses=statuses,
            plan_id=plan_id,
            limit=limit
        )

    async def find_related_tasks(
        self,
        task: MicroTask,
        max_results: int = 10
    ) -> List[MicroTask]:
        """Find tasks related to the given task using keywords and memory"""

        related_tasks = []

        # Search by keywords
        if task.keywords:
            for keyword in task.keywords:
                keyword_tasks = await self.repository.search_tasks(
                    query=keyword,
                    limit=max_results
                )
                related_tasks.extend(keyword_tasks)

        # Search by task type
        type_tasks = await self.repository.search_tasks(
            query="",
            task_types=[task.task_type],
            limit=max_results
        )
        related_tasks.extend(type_tasks)

        # Remove duplicates and self
        seen = set()
        unique_tasks = []
        for t in related_tasks:
            if t.id != task.id and t.id not in seen:
                unique_tasks.append(t)
                seen.add(t.id)

        return unique_tasks[:max_results]

    # ============================================================================
    # EVENT SYSTEM
    # ============================================================================

    def register_event_handler(self, event_type: str, handler) -> None:
        """Register event handler for specific event type"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def unregister_event_handler(self, event_type: str, handler) -> None:
        """Unregister event handler"""
        if event_type in self._event_handlers:
            self._event_handlers[event_type].remove(handler)

    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit event to all registered handlers"""
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event_type, data)
                    else:
                        handler(event_type, data)
                except Exception as e:
                    # Log error but don't stop event processing
                    print(f"Error in event handler for {event_type}: {e}")

    # ============================================================================
    # VALIDATION METHODS
    # ============================================================================

    async def _validate_intervention_plan(self, plan: InterventionPlan) -> None:
        """Validate intervention plan"""
        if not plan.title or len(plan.title) < 10:
            raise ValidationError("Plan title must be at least 10 characters")

        if not plan.description or len(plan.description) < 20:
            raise ValidationError("Plan description must be at least 20 characters")

        if plan.estimated_hours <= 0 or plan.estimated_hours > 200:
            raise ValidationError("Plan estimated hours must be between 0 and 200")

    async def _validate_phase(self, phase: Phase, plan_id: UUID) -> None:
        """Validate phase"""
        if not phase.name or len(phase.name) < 5:
            raise ValidationError("Phase name must be at least 5 characters")

        # Check for duplicate order index
        existing_phases = await self.repository.get_phases_for_plan(plan_id)
        for existing_phase in existing_phases:
            if existing_phase.order_index == phase.order_index:
                raise ValidationError(f"Phase order index {phase.order_index} already exists")

    async def _validate_phase_update(self, phase: Phase) -> None:
        """Validate phase update"""
        if not phase.name or len(phase.name) < 5:
            raise ValidationError("Phase name must be at least 5 characters")

    async def _validate_micro_task(self, task: MicroTask, phase_id: UUID) -> None:
        """Validate micro task"""
        if self.config.strict_granularity_validation:
            if task.estimated_minutes > 10:
                raise ValidationError("MicroTask must be completable in 10 minutes or less")

        # Validate acceptance criteria
        if not task.acceptance_criteria:
            raise ValidationError("Task must have at least one acceptance criterion")

    async def _validate_micro_task_update(self, task: MicroTask) -> None:
        """Validate micro task update"""
        await self._validate_micro_task(task, UUID('00000000-0000-0000-0000-000000000000'))  # Dummy phase_id for update

    async def _validate_task_dependencies(self, task: MicroTask, phase_id: UUID) -> None:
        """Validate task dependencies won't create cycles"""
        if task.depends_on:
            dependency_list = list(task.depends_on)
            valid = await self.repository.validate_dependencies(task.id, dependency_list)
            if not valid:
                raise DependencyError("Task dependencies would create cycle")

    async def _validate_task_can_start(self, task: MicroTask) -> None:
        """Validate task can be started"""
        if task.status != TaskStatus.PENDING:
            raise ExecutionError(f"Task {task.id} is not in pending status")

    async def _validate_task_can_complete(self, task: MicroTask) -> None:
        """Validate task can be completed"""
        if task.status != TaskStatus.IN_PROGRESS:
            raise ExecutionError(f"Task {task.id} is not in progress")

    # ============================================================================
    # HELPER METHODS
    # ============================================================================

    async def _is_task_ready(self, task: MicroTask) -> bool:
        """Check if task is ready to execute (dependencies completed)"""
        if not task.depends_on:
            return True

        for dep_id in task.depends_on:
            dep_task = await self.repository.get_micro_task(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False

        return True

    async def _detect_task_dependencies(self, task: MicroTask, phase_id: UUID) -> Set[UUID]:
        """Auto-detect potential dependencies based on task content"""
        dependencies = set()

        # Get existing tasks in phase
        existing_tasks = await self.repository.get_tasks_for_phase(phase_id)

        # Simple keyword-based dependency detection
        task_keywords = set(task.keywords + task.title.lower().split())

        for existing_task in existing_tasks:
            if existing_task.id == task.id:
                continue

            existing_keywords = set(existing_task.keywords + existing_task.title.lower().split())

            # If significant keyword overlap, suggest dependency
            overlap = task_keywords.intersection(existing_keywords)
            if len(overlap) >= 2:  # At least 2 common keywords
                dependencies.add(existing_task.id)

        return dependencies

    async def _update_plan_progress(self, plan: InterventionPlan) -> None:
        """Update plan progress based on phases"""
        phases = await self.repository.get_phases_for_plan(plan.id)

        if not phases:
            return

        phase_progresses = {}
        for phase in phases:
            tasks = await self.repository.get_tasks_for_phase(phase.id)
            if tasks:
                task_statuses = {task.id: task.status for task in tasks}
                progress = phase.calculate_progress(task_statuses)
                phase_progresses[phase.id] = progress

        plan.calculate_overall_progress(phase_progresses)
        await self.repository.update_intervention_plan(plan)

    async def _update_phase_progress(self, phase: Phase) -> None:
        """Update phase progress based on tasks"""
        tasks = await self.repository.get_tasks_for_phase(phase.id)

        if tasks:
            task_statuses = {task.id: task.status for task in tasks}
            progress = phase.calculate_progress(task_statuses)
            # Update phase status based on progress
            if progress == 100.0:
                phase.status = TaskStatus.COMPLETED
                phase.completed_at = datetime.now()
            elif progress > 0.0:
                phase.status = TaskStatus.IN_PROGRESS
                if not phase.started_at:
                    phase.started_at = datetime.now()

            await self.repository.update_phase(phase)

    async def _update_dependent_progress(self, completed_task: MicroTask) -> None:
        """Update progress for dependent entities when task completes"""
        # Get phase and update its progress
        # This would need phase_id to be stored in task or retrieved via query
        pass

    async def _store_plan_in_memory(self, plan: InterventionPlan) -> None:
        """Store plan information in memory system"""
        if not self.memory_storage:
            return

        memory_entry = MemoryEntry(
            id=f"plan_{plan.id}",
            content=f"Intervention Plan: {plan.title}\n\nObjective: {plan.objective}\n\nDescription: {plan.description}",
            content_type=ContentType.DOCUMENTATION,
            keywords=[plan.category] + plan.tags,
            complexity_score=min(int(plan.estimated_hours / 4), 10)
        )

        await self.memory_storage.store_memory(memory_entry)

    async def _store_task_in_memory(self, task: MicroTask) -> None:
        """Store task information in memory system"""
        if not self.memory_storage:
            return

        memory_entry = MemoryEntry(
            id=f"task_{task.id}",
            content=f"Task: {task.title}\n\nDescription: {task.description}\n\nType: {task.task_type.value}",
            content_type=ContentType.CODE if task.task_type == TaskType.IMPLEMENTATION else ContentType.DOCUMENTATION,
            keywords=task.keywords + [task.task_type.value],
            complexity_score={"trivial": 2, "simple": 4, "moderate": 6, "complex": 8}[task.complexity.value]
        )

        await self.memory_storage.store_memory(memory_entry)

    async def _update_task_in_memory(self, task: MicroTask) -> None:
        """Update task information in memory system"""
        await self._store_task_in_memory(task)  # Overwrite existing

    async def _store_task_completion_in_memory(self, task: MicroTask) -> None:
        """Store task completion information in memory"""
        if not self.memory_storage:
            return

        completion_content = f"Completed Task: {task.title}\n\nDescription: {task.description}"
        if task.implementation_notes:
            completion_content += f"\n\nImplementation Notes: {task.implementation_notes}"
        if task.output_artifacts:
            completion_content += f"\n\nOutput Artifacts: {', '.join(task.output_artifacts)}"

        memory_entry = MemoryEntry(
            id=f"task_completion_{task.id}",
            content=completion_content,
            content_type=ContentType.CODE if task.task_type == TaskType.IMPLEMENTATION else ContentType.DOCUMENTATION,
            keywords=task.keywords + [task.task_type.value, "completed"],
            complexity_score={"trivial": 2, "simple": 4, "moderate": 6, "complex": 8}[task.complexity.value]
        )

        await self.memory_storage.store_memory(memory_entry)

    async def _calculate_completion_estimates(self, plan_id: UUID) -> Dict[str, Any]:
        """Calculate completion time estimates"""
        # Get task statistics
        stats = await self.repository.get_task_statistics(plan_id)

        total_tasks = sum(stats.values())
        completed_tasks = stats.get("completed", 0)

        if total_tasks == 0:
            return {"estimated_completion": None, "progress_ratio": 0.0}

        progress_ratio = completed_tasks / total_tasks

        # Simple linear estimate
        plan = await self.repository.get_intervention_plan(plan_id)
        if plan and plan.estimated_hours:
            remaining_hours = plan.estimated_hours * (1 - progress_ratio)
            estimated_completion = datetime.now() + timedelta(hours=remaining_hours)
        else:
            estimated_completion = None

        return {
            "estimated_completion": estimated_completion,
            "progress_ratio": progress_ratio,
            "remaining_tasks": total_tasks - completed_tasks
        }

    async def _detect_dependency_violations(self, plan_id: UUID) -> List[Dict[str, Any]]:
        """Detect tasks that violate dependency constraints"""
        violations = []

        # Get all tasks
        graph = await self.build_dependency_graph(plan_id)

        for task_id, task in graph.tasks.items():
            if task.status == TaskStatus.IN_PROGRESS:
                # Check if all dependencies are completed
                for dep_id in task.depends_on:
                    dep_task = graph.tasks.get(dep_id)
                    if dep_task and dep_task.status != TaskStatus.COMPLETED:
                        violations.append({
                            "task_id": task_id,
                            "task_title": task.title,
                            "violated_dependency_id": dep_id,
                            "violated_dependency_title": dep_task.title,
                            "violation_type": "incomplete_dependency"
                        })

        return violations

    async def _identify_bottleneck_tasks(self, plan_id: UUID) -> List[Dict[str, Any]]:
        """Identify tasks that are bottlenecks in the workflow"""
        bottlenecks = []

        graph = await self.build_dependency_graph(plan_id)

        # Find tasks that many other tasks depend on
        dependency_counts = {}
        for task_id, task in graph.tasks.items():
            for dep_id in task.depends_on:
                dependency_counts[dep_id] = dependency_counts.get(dep_id, 0) + 1

        # Tasks with high dependency count are bottlenecks
        for task_id, count in dependency_counts.items():
            if count >= 3:  # Threshold for bottleneck
                task = graph.tasks.get(task_id)
                if task:
                    bottlenecks.append({
                        "task_id": task_id,
                        "task_title": task.title,
                        "dependent_task_count": count,
                        "status": task.status.value,
                        "is_blocking": task.status != TaskStatus.COMPLETED
                    })

        return sorted(bottlenecks, key=lambda x: x["dependent_task_count"], reverse=True)