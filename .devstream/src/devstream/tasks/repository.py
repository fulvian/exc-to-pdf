"""
Task Repository Layer - SQLAlchemy Integration

Provides async repository pattern for task management models,
integrating Pydantic models with existing SQLAlchemy schema.

Key Features:
- Async CRUD operations for all task entities
- Dependency graph persistence and retrieval
- Type-safe conversion between Pydantic and SQLAlchemy
- Transaction management for complex operations
- Query optimization for large task hierarchies

Based on existing DevStream database schema with extensions for new functionality.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4

from sqlalchemy import select, insert, update, delete, and_, or_, func, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine
from sqlalchemy.exc import IntegrityError

from devstream.database.connection import ConnectionPool
from devstream.database.schema import (
    intervention_plans, phases, micro_tasks, metadata
)
from devstream.tasks.models import (
    InterventionPlan, Phase, MicroTask, TaskDependencyGraph,
    TaskStatus, TaskPriority, TaskType, TaskComplexity
)
from devstream.exceptions import DevStreamError


class TaskRepositoryError(DevStreamError):
    """Task repository specific errors"""
    pass


class TaskNotFoundError(TaskRepositoryError):
    """Task not found in repository"""
    pass


class DependencyCycleError(TaskRepositoryError):
    """Dependency cycle detected"""
    pass


class TaskRepository:
    """
    Async repository for task management operations

    Provides high-level interface for task CRUD operations
    with automatic conversion between Pydantic models and database rows.
    """

    def __init__(self, engine: Optional[AsyncEngine] = None):
        if engine:
            self.engine = engine
        else:
            # Create a default ConnectionPool for testing
            # In production, this would be injected
            pool = ConnectionPool("test.db")
            self.engine = pool.engine

    # ============================================================================
    # INTERVENTION PLAN OPERATIONS
    # ============================================================================

    async def create_intervention_plan(self, plan: InterventionPlan) -> InterventionPlan:
        """Create a new intervention plan"""
        async with self.engine.begin() as conn:
            # Convert Pydantic model to database row
            plan_data = {
                "id": str(plan.id),
                "title": plan.title,
                "description": plan.description,
                "objectives": [plan.objective],  # Map single objective to array
                "technical_specs": {
                    "category": plan.category,
                    "tags": plan.tags,
                    "stakeholders": plan.stakeholders,
                    "ai_generated": plan.ai_generated,
                    "ai_confidence": plan.ai_confidence,
                    "human_approved": plan.human_approved
                },
                "expected_outcome": "; ".join(plan.key_deliverables) if plan.key_deliverables else plan.objective,
                "status": self._map_status_to_db(plan.status),
                "priority": self._map_priority_to_db(plan.priority) if hasattr(plan, 'priority') else 5,
                "estimated_hours": plan.estimated_hours,
                "actual_hours": plan.actual_hours,
                "tags": plan.tags,
                "metadata": {
                    "success_criteria": plan.success_criteria,
                    "quality_gates": plan.quality_gates,
                    "progress_percentage": plan.progress_percentage,
                    "total_tasks": plan.total_tasks
                },
                "created_at": plan.created_at,
                "updated_at": plan.updated_at,
                "completed_at": plan.completed_at
            }

            await conn.execute(insert(intervention_plans).values(**plan_data))
            return plan

    async def get_intervention_plan(self, plan_id: UUID) -> Optional[InterventionPlan]:
        """Get intervention plan by ID"""
        async with self.engine.connect() as conn:
            result = await conn.execute(
                select(intervention_plans).where(intervention_plans.c.id == str(plan_id))
            )
            row = result.fetchone()

            if not row:
                return None

            return self._row_to_intervention_plan(row)

    async def update_intervention_plan(self, plan: InterventionPlan) -> InterventionPlan:
        """Update existing intervention plan"""
        async with self.engine.begin() as conn:
            plan_data = {
                "title": plan.title,
                "description": plan.description,
                "objectives": [plan.objective],
                "status": self._map_status_to_db(plan.status),
                "estimated_hours": plan.estimated_hours,
                "actual_hours": plan.actual_hours,
                "updated_at": datetime.now(),
                "completed_at": plan.completed_at,
                "metadata": {
                    "success_criteria": plan.success_criteria,
                    "quality_gates": plan.quality_gates,
                    "progress_percentage": plan.progress_percentage,
                    "total_tasks": plan.total_tasks
                }
            }

            await conn.execute(
                update(intervention_plans)
                .where(intervention_plans.c.id == str(plan.id))
                .values(**plan_data)
            )

            plan.updated_at = datetime.now()
            return plan

    async def delete_intervention_plan(self, plan_id: UUID) -> bool:
        """Delete intervention plan and all associated phases/tasks"""
        async with self.engine.begin() as conn:
            result = await conn.execute(
                delete(intervention_plans).where(intervention_plans.c.id == str(plan_id))
            )
            return result.rowcount > 0

    async def list_intervention_plans(
        self,
        status: Optional[TaskStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[InterventionPlan]:
        """List intervention plans with optional filtering"""
        async with self.engine.connect() as conn:
            query = select(intervention_plans)

            if status:
                query = query.where(intervention_plans.c.status == self._map_status_to_db(status))

            query = query.order_by(intervention_plans.c.created_at.desc()).limit(limit).offset(offset)

            result = await conn.execute(query)
            rows = result.fetchall()

            return [self._row_to_intervention_plan(row) for row in rows]

    # ============================================================================
    # PHASE OPERATIONS
    # ============================================================================

    async def create_phase(self, phase: Phase, plan_id: UUID) -> Phase:
        """Create a new phase"""
        async with self.engine.begin() as conn:
            phase_data = {
                "id": str(phase.id),
                "plan_id": str(plan_id),
                "name": phase.name,
                "description": phase.description,
                "sequence_order": phase.order_index,
                "is_parallel": False,  # Default to sequential
                "dependencies": [str(dep_id) for dep_id in phase.depends_on_phases],
                "status": self._map_status_to_db(phase.status),
                "estimated_minutes": int(phase.estimated_hours * 60) if phase.estimated_hours else None,
                "actual_minutes": int(phase.actual_hours * 60) if phase.actual_hours else 0,
                "completion_criteria": "; ".join(phase.success_criteria) if phase.success_criteria else None,
                "created_at": phase.created_at,
                "started_at": phase.started_at,
                "completed_at": phase.completed_at
            }

            await conn.execute(insert(phases).values(**phase_data))
            return phase

    async def get_phase(self, phase_id: UUID) -> Optional[Phase]:
        """Get phase by ID"""
        async with self.engine.connect() as conn:
            result = await conn.execute(
                select(phases).where(phases.c.id == str(phase_id))
            )
            row = result.fetchone()

            if not row:
                return None

            return self._row_to_phase(row)

    async def get_phases_for_plan(self, plan_id: UUID) -> List[Phase]:
        """Get all phases for a plan, ordered by sequence"""
        async with self.engine.connect() as conn:
            result = await conn.execute(
                select(phases)
                .where(phases.c.plan_id == str(plan_id))
                .order_by(phases.c.sequence_order)
            )
            rows = result.fetchall()

            return [self._row_to_phase(row) for row in rows]

    async def update_phase(self, phase: Phase) -> Phase:
        """Update existing phase"""
        async with self.engine.begin() as conn:
            phase_data = {
                "name": phase.name,
                "description": phase.description,
                "status": self._map_status_to_db(phase.status),
                "estimated_minutes": int(phase.estimated_hours * 60) if phase.estimated_hours else None,
                "actual_minutes": int(phase.actual_hours * 60) if phase.actual_hours else 0,
                "started_at": phase.started_at,
                "completed_at": phase.completed_at,
                "dependencies": [str(dep_id) for dep_id in phase.depends_on_phases]
            }

            await conn.execute(
                update(phases)
                .where(phases.c.id == str(phase.id))
                .values(**phase_data)
            )

            phase.updated_at = datetime.now()
            return phase

    # ============================================================================
    # MICRO TASK OPERATIONS
    # ============================================================================

    async def create_micro_task(self, task: MicroTask, phase_id: UUID) -> MicroTask:
        """Create a new micro task"""
        async with self.engine.begin() as conn:
            task_data = {
                "id": str(task.id),
                "phase_id": str(phase_id),
                "title": task.title,
                "description": task.description,
                "max_duration_minutes": task.estimated_minutes,
                "max_context_tokens": 256000,  # Default from schema
                "assigned_agent": task.assignee,
                "task_type": self._map_task_type_to_db(task.task_type),
                "status": self._map_status_to_db(task.status),
                "priority": self._map_priority_to_db(task.priority),
                "input_files": [],  # Will be populated later
                "output_files": task.output_artifacts,
                "generated_code": "",  # Will be populated during execution
                "documentation": task.implementation_notes,
                "actual_duration_minutes": task.actual_minutes,
                "context_tokens_used": 0,  # Will be tracked during execution
                "retry_count": 0,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "dependencies": [str(dep_id) for dep_id in task.depends_on],
                "keywords": task.keywords,
                "acceptance_criteria": task.acceptance_criteria,
                "ai_suggestions": task.ai_suggestions,
                "validation_metadata": {
                    "complexity": task.complexity.value,
                    "validation_passed": task.validation_passed,
                    "ai_generated": task.ai_generated,
                    "context_tags": task.context_tags,
                    "related_memory_ids": task.related_memory_ids
                }
            }

            await conn.execute(insert(micro_tasks).values(**task_data))
            return task

    async def get_micro_task(self, task_id: UUID) -> Optional[MicroTask]:
        """Get micro task by ID"""
        async with self.engine.connect() as conn:
            result = await conn.execute(
                select(micro_tasks).where(micro_tasks.c.id == str(task_id))
            )
            row = result.fetchone()

            if not row:
                return None

            return self._row_to_micro_task(row)

    async def get_tasks_for_phase(self, phase_id: UUID) -> List[MicroTask]:
        """Get all tasks for a phase"""
        async with self.engine.connect() as conn:
            result = await conn.execute(
                select(micro_tasks)
                .where(micro_tasks.c.phase_id == str(phase_id))
                .order_by(micro_tasks.c.created_at)
            )
            rows = result.fetchall()

            return [self._row_to_micro_task(row) for row in rows]

    async def update_micro_task(self, task: MicroTask) -> MicroTask:
        """Update existing micro task"""
        async with self.engine.begin() as conn:
            task_data = {
                "title": task.title,
                "description": task.description,
                "status": self._map_status_to_db(task.status),
                "priority": self._map_priority_to_db(task.priority),
                "assigned_agent": task.assignee,
                "actual_duration_minutes": task.actual_minutes,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "documentation": task.implementation_notes,
                "dependencies": [str(dep_id) for dep_id in task.depends_on],
                "keywords": task.keywords,
                "acceptance_criteria": task.acceptance_criteria,
                "validation_metadata": {
                    "complexity": task.complexity.value,
                    "validation_passed": task.validation_passed,
                    "ai_generated": task.ai_generated,
                    "context_tags": task.context_tags,
                    "related_memory_ids": task.related_memory_ids
                }
            }

            await conn.execute(
                update(micro_tasks)
                .where(micro_tasks.c.id == str(task.id))
                .values(**task_data)
            )

            task.updated_at = datetime.now()
            return task

    # ============================================================================
    # DEPENDENCY GRAPH OPERATIONS
    # ============================================================================

    async def build_dependency_graph(self, plan_id: UUID) -> TaskDependencyGraph:
        """Build complete dependency graph for a plan"""
        async with self.engine.connect() as conn:
            # Get all tasks for the plan
            tasks_query = select(micro_tasks).where(
                micro_tasks.c.phase_id.in_(
                    select(phases.c.id).where(phases.c.plan_id == str(plan_id))
                )
            )

            result = await conn.execute(tasks_query)
            rows = result.fetchall()

            # Build dependency graph
            graph = TaskDependencyGraph()

            for row in rows:
                task = self._row_to_micro_task(row)
                graph.add_task(task)

            return graph

    async def get_ready_tasks(self, plan_id: UUID) -> List[MicroTask]:
        """Get tasks that are ready to execute (no pending dependencies)"""
        graph = await self.build_dependency_graph(plan_id)

        # Get completed task IDs
        async with self.engine.connect() as conn:
            completed_query = select(micro_tasks.c.id).where(
                and_(
                    micro_tasks.c.phase_id.in_(
                        select(phases.c.id).where(phases.c.plan_id == str(plan_id))
                    ),
                    micro_tasks.c.status == "completed"
                )
            )

            result = await conn.execute(completed_query)
            completed_task_ids = {UUID(row[0]) for row in result.fetchall()}

        return graph.get_ready_tasks(completed_task_ids)

    async def validate_dependencies(self, task_id: UUID, dependency_ids: List[UUID]) -> bool:
        """Validate that adding dependencies won't create cycles"""
        # Get current task's plan to build graph
        async with self.engine.connect() as conn:
            task_query = select(micro_tasks.c.phase_id).where(micro_tasks.c.id == str(task_id))
            result = await conn.execute(task_query)
            phase_id = result.scalar()

            if not phase_id:
                raise TaskNotFoundError(f"Task {task_id} not found")

            plan_query = select(phases.c.plan_id).where(phases.c.id == phase_id)
            result = await conn.execute(plan_query)
            plan_id = UUID(result.scalar())

        # Build current graph and test dependencies
        graph = await self.build_dependency_graph(plan_id)

        # Test each dependency for cycles
        for dep_id in dependency_ids:
            if graph._would_create_cycle(dep_id, task_id):
                return False

        return True

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
        """Search tasks by title, description, or keywords"""
        async with self.engine.connect() as conn:
            # Build search query
            search_query = select(micro_tasks)
            conditions = []

            # Text search in title and description
            if query:
                search_condition = or_(
                    micro_tasks.c.title.ilike(f"%{query}%"),
                    micro_tasks.c.description.ilike(f"%{query}%"),
                    func.json_extract(micro_tasks.c.keywords, '$').like(f"%{query}%")
                )
                conditions.append(search_condition)

            # Filter by task types
            if task_types:
                db_types = [self._map_task_type_to_db(tt) for tt in task_types]
                conditions.append(micro_tasks.c.task_type.in_(db_types))

            # Filter by statuses
            if statuses:
                db_statuses = [self._map_status_to_db(s) for s in statuses]
                conditions.append(micro_tasks.c.status.in_(db_statuses))

            # Filter by plan
            if plan_id:
                conditions.append(
                    micro_tasks.c.phase_id.in_(
                        select(phases.c.id).where(phases.c.plan_id == str(plan_id))
                    )
                )

            if conditions:
                search_query = search_query.where(and_(*conditions))

            search_query = search_query.order_by(micro_tasks.c.created_at.desc()).limit(limit)

            result = await conn.execute(search_query)
            rows = result.fetchall()

            return [self._row_to_micro_task(row) for row in rows]

    async def get_task_statistics(self, plan_id: UUID) -> Dict[str, int]:
        """Get task statistics for a plan"""
        async with self.engine.connect() as conn:
            stats_query = select(
                micro_tasks.c.status,
                func.count(micro_tasks.c.id).label('count')
            ).where(
                micro_tasks.c.phase_id.in_(
                    select(phases.c.id).where(phases.c.plan_id == str(plan_id))
                )
            ).group_by(micro_tasks.c.status)

            result = await conn.execute(stats_query)
            rows = result.fetchall()

            stats = {row[0]: row[1] for row in rows}

            # Ensure all statuses are represented
            for status in ["pending", "active", "completed", "failed", "skipped"]:
                if status not in stats:
                    stats[status] = 0

            return stats

    # ============================================================================
    # HELPER METHODS
    # ============================================================================

    def _row_to_intervention_plan(self, row) -> InterventionPlan:
        """Convert database row to InterventionPlan model"""
        metadata_obj = row.metadata or {}
        technical_specs = row.technical_specs or {}

        return InterventionPlan(
            id=UUID(row.id),
            title=row.title,
            description=row.description or "",
            objective=row.objectives[0] if row.objectives else row.expected_outcome,
            status=self._map_status_from_db(row.status),
            estimated_hours=row.estimated_hours or 0.0,
            actual_hours=row.actual_hours or 0.0,
            created_at=row.created_at,
            updated_at=row.updated_at,
            completed_at=row.completed_at,
            category=technical_specs.get("category", "development"),
            tags=row.tags or [],
            stakeholders=technical_specs.get("stakeholders", []),
            ai_generated=technical_specs.get("ai_generated", False),
            ai_confidence=technical_specs.get("ai_confidence"),
            human_approved=technical_specs.get("human_approved", False),
            success_criteria=metadata_obj.get("success_criteria", []),
            key_deliverables=[row.expected_outcome] if row.expected_outcome else [],
            quality_gates=metadata_obj.get("quality_gates", []),
            progress_percentage=metadata_obj.get("progress_percentage", 0.0),
            total_tasks=metadata_obj.get("total_tasks", 0)
        )

    def _row_to_phase(self, row) -> Phase:
        """Convert database row to Phase model"""
        return Phase(
            id=UUID(row.id),
            name=row.name,
            description=row.description or "",
            objective=row.completion_criteria or row.name,
            order_index=row.sequence_order,
            status=self._map_status_from_db(row.status),
            estimated_hours=(row.estimated_minutes / 60.0) if row.estimated_minutes else 1.0,
            actual_hours=(row.actual_minutes / 60.0) if row.actual_minutes else 0.0,
            created_at=row.created_at,
            started_at=row.started_at,
            completed_at=row.completed_at,
            depends_on_phases={UUID(dep_id) for dep_id in (row.dependencies or [])}
        )

    def _row_to_micro_task(self, row) -> MicroTask:
        """Convert database row to MicroTask model"""
        validation_metadata = getattr(row, 'validation_metadata', None) or {}

        return MicroTask(
            id=UUID(row.id),
            title=row.title,
            description=row.description,
            task_type=self._map_task_type_from_db(row.task_type),
            complexity=TaskComplexity(validation_metadata.get("complexity", "simple")),
            priority=self._map_priority_from_db(row.priority),
            status=self._map_status_from_db(row.status),
            estimated_minutes=row.max_duration_minutes or 5,
            actual_minutes=row.actual_duration_minutes,
            assignee=row.assigned_agent,
            created_at=row.created_at,
            updated_at=getattr(row, 'updated_at', None),
            started_at=row.started_at,
            completed_at=row.completed_at,
            depends_on={UUID(dep_id) for dep_id in (getattr(row, 'dependencies', None) or [])},
            keywords=getattr(row, 'keywords', None) or [],
            context_tags=validation_metadata.get("context_tags", []),
            related_memory_ids=validation_metadata.get("related_memory_ids", []),
            implementation_notes=row.documentation,
            acceptance_criteria=getattr(row, 'acceptance_criteria', None) or [],
            output_artifacts=row.output_files or [],
            ai_generated=validation_metadata.get("ai_generated", False),
            ai_suggestions=getattr(row, 'ai_suggestions', None),
            validation_passed=validation_metadata.get("validation_passed", False)
        )

    def _map_status_to_db(self, status: TaskStatus) -> str:
        """Map Pydantic TaskStatus to database status"""
        mapping = {
            TaskStatus.PENDING: "pending",
            TaskStatus.IN_PROGRESS: "active",
            TaskStatus.BLOCKED: "blocked",
            TaskStatus.COMPLETED: "completed",
            TaskStatus.CANCELLED: "cancelled"
        }
        return mapping.get(status, "pending")

    def _map_status_from_db(self, db_status: str) -> TaskStatus:
        """Map database status to Pydantic TaskStatus"""
        mapping = {
            "pending": TaskStatus.PENDING,
            "active": TaskStatus.IN_PROGRESS,
            "blocked": TaskStatus.BLOCKED,
            "completed": TaskStatus.COMPLETED,
            "cancelled": TaskStatus.CANCELLED,
            "failed": TaskStatus.CANCELLED,
            "skipped": TaskStatus.CANCELLED,
            "draft": TaskStatus.PENDING,
            "archived": TaskStatus.COMPLETED
        }
        return mapping.get(db_status, TaskStatus.PENDING)

    def _map_priority_to_db(self, priority: TaskPriority) -> int:
        """Map Pydantic TaskPriority to database priority (1-10)"""
        mapping = {
            TaskPriority.LOW: 3,
            TaskPriority.MEDIUM: 5,
            TaskPriority.HIGH: 7,
            TaskPriority.CRITICAL: 10
        }
        return mapping.get(priority, 5)

    def _map_priority_from_db(self, db_priority: int) -> TaskPriority:
        """Map database priority to Pydantic TaskPriority"""
        if db_priority >= 9:
            return TaskPriority.CRITICAL
        elif db_priority >= 7:
            return TaskPriority.HIGH
        elif db_priority >= 4:
            return TaskPriority.MEDIUM
        else:
            return TaskPriority.LOW

    def _map_task_type_to_db(self, task_type: TaskType) -> str:
        """Map Pydantic TaskType to database task_type"""
        mapping = {
            TaskType.RESEARCH: "research",
            TaskType.IMPLEMENTATION: "coding",
            TaskType.TESTING: "testing",
            TaskType.DOCUMENTATION: "documentation",
            TaskType.REFACTORING: "coding",
            TaskType.DEBUGGING: "coding",
            TaskType.INTEGRATION: "coding",
            TaskType.DEPLOYMENT: "coding"
        }
        return mapping.get(task_type, "coding")

    def _map_task_type_from_db(self, db_task_type: str) -> TaskType:
        """Map database task_type to Pydantic TaskType"""
        mapping = {
            "research": TaskType.RESEARCH,
            "coding": TaskType.IMPLEMENTATION,
            "testing": TaskType.TESTING,
            "documentation": TaskType.DOCUMENTATION,
            "analysis": TaskType.RESEARCH,
            "review": TaskType.TESTING
        }
        return mapping.get(db_task_type, TaskType.IMPLEMENTATION)