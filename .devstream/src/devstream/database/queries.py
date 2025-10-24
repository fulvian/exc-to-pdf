"""
Database query operations using async SQLAlchemy Core.

Provides type-safe CRUD operations for all entities with connection pool integration.
Uses prepared statements and batch operations for performance.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog
from sqlalchemy import and_, desc, func, or_, select, update, delete, text
from sqlalchemy.sql import Select

from devstream.core.exceptions import DatabaseError, EntityNotFoundError
from devstream.database.connection import ConnectionPool
from devstream.database.schema import (
    intervention_plans,
    phases,
    micro_tasks,
    semantic_memory,
    agents,
    hooks,
    hook_executions,
    work_sessions,
    context_injections,
    performance_metrics,
    learning_insights,
)

logger = structlog.get_logger()


class BaseQuery:
    """Base class for database queries with common operations."""

    def __init__(self, pool: ConnectionPool):
        """
        Initialize query handler.

        Args:
            pool: Connection pool for database operations
        """
        self.pool = pool

    def generate_id(self) -> str:
        """Generate unique identifier for entities."""
        return uuid4().hex[:32]

    async def execute_read(
        self, query: Select, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute read query and return results as dictionaries.

        Args:
            query: SQLAlchemy select query
            params: Query parameters

        Returns:
            List of result dictionaries
        """
        try:
            async with self.pool.read_transaction() as conn:
                result = await conn.execute(query, params or {})
                rows = result.fetchall()
                # Convert rows to dictionaries
                if rows:
                    columns = result.keys()
                    return [dict(zip(columns, row)) for row in rows]
                return []
        except Exception as e:
            logger.error("Read query failed", error=str(e))
            raise DatabaseError(f"Query failed: {str(e)}", error_code="READ_ERROR")

    async def execute_write(
        self, statement: Any, params: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Execute write statement and return affected rows.

        Args:
            statement: SQLAlchemy insert/update/delete statement
            params: Statement parameters

        Returns:
            Number of affected rows
        """
        try:
            async with self.pool.write_transaction() as conn:
                result = await conn.execute(statement, params or {})
                return result.rowcount
        except Exception as e:
            logger.error("Write statement failed", error=str(e))
            raise DatabaseError(f"Write failed: {str(e)}", error_code="WRITE_ERROR")


class InterventionPlanQueries(BaseQuery):
    """Query operations for intervention plans."""

    async def create(
        self,
        title: str,
        description: str,
        objectives: List[str],
        expected_outcome: str,
        **kwargs: Any,
    ) -> str:
        """
        Create new intervention plan.

        Args:
            title: Plan title
            description: Plan description
            objectives: List of objectives
            expected_outcome: Expected outcome
            **kwargs: Additional fields

        Returns:
            Created plan ID
        """
        plan_id = self.generate_id()

        stmt = intervention_plans.insert().values(
            id=plan_id,
            title=title,
            description=description,
            objectives=json.dumps(objectives),
            expected_outcome=expected_outcome,
            status=kwargs.get("status", "draft"),
            priority=kwargs.get("priority", 5),
            estimated_hours=kwargs.get("estimated_hours"),
            tags=json.dumps(kwargs.get("tags", [])),
            metadata=json.dumps(kwargs.get("metadata", {})),
        )

        await self.execute_write(stmt)
        logger.info("Created intervention plan", plan_id=plan_id, title=title)
        return plan_id

    async def get_by_id(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """
        Get plan by ID.

        Args:
            plan_id: Plan identifier

        Returns:
            Plan data or None if not found
        """
        query = select(intervention_plans).where(intervention_plans.c.id == plan_id)
        results = await self.execute_read(query)

        if results:
            plan = results[0]
            # Parse JSON fields
            plan["objectives"] = json.loads(plan["objectives"])
            plan["tags"] = json.loads(plan["tags"]) if plan["tags"] else []
            plan["metadata"] = json.loads(plan["metadata"]) if plan["metadata"] else {}
            return plan
        return None

    async def list_active(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List active intervention plans.

        Args:
            limit: Maximum results to return

        Returns:
            List of active plans
        """
        query = (
            select(intervention_plans)
            .where(intervention_plans.c.status.in_(["draft", "active"]))
            .order_by(desc(intervention_plans.c.priority))
            .limit(limit)
        )

        results = await self.execute_read(query)

        # Parse JSON fields
        for plan in results:
            plan["objectives"] = json.loads(plan["objectives"])
            plan["tags"] = json.loads(plan["tags"]) if plan["tags"] else []
            plan["metadata"] = json.loads(plan["metadata"]) if plan["metadata"] else {}

        return results

    async def update_status(
        self, plan_id: str, status: str, completed_at: Optional[datetime] = None
    ) -> bool:
        """
        Update plan status.

        Args:
            plan_id: Plan identifier
            status: New status
            completed_at: Completion timestamp if applicable

        Returns:
            True if updated successfully
        """
        stmt = (
            update(intervention_plans)
            .where(intervention_plans.c.id == plan_id)
            .values(
                status=status,
                completed_at=completed_at,
                updated_at=datetime.utcnow(),
            )
        )

        affected = await self.execute_write(stmt)
        return affected > 0


class PhaseQueries(BaseQuery):
    """Query operations for phases."""

    async def create(
        self,
        plan_id: str,
        name: str,
        description: str,
        sequence_order: int,
        **kwargs: Any,
    ) -> str:
        """
        Create new phase.

        Args:
            plan_id: Parent plan ID
            name: Phase name
            description: Phase description
            sequence_order: Sequence order within plan
            **kwargs: Additional fields

        Returns:
            Created phase ID
        """
        phase_id = self.generate_id()

        stmt = phases.insert().values(
            id=phase_id,
            plan_id=plan_id,
            name=name,
            description=description,
            sequence_order=sequence_order,
            is_parallel=kwargs.get("is_parallel", False),
            dependencies=json.dumps(kwargs.get("dependencies", [])),
            status=kwargs.get("status", "pending"),
            estimated_minutes=kwargs.get("estimated_minutes"),
            blocking_reason=kwargs.get("blocking_reason"),
            completion_criteria=kwargs.get("completion_criteria"),
        )

        await self.execute_write(stmt)
        logger.info("Created phase", phase_id=phase_id, name=name, plan_id=plan_id)
        return phase_id

    async def get_by_id(self, phase_id: str) -> Optional[Dict[str, Any]]:
        """
        Get phase by ID.

        Args:
            phase_id: Phase identifier

        Returns:
            Phase data or None if not found
        """
        query = select(phases).where(phases.c.id == phase_id)
        results = await self.execute_read(query)

        if results:
            phase = results[0]
            # Parse JSON fields
            phase["dependencies"] = json.loads(phase["dependencies"]) if phase["dependencies"] else []
            return phase
        return None

    async def list_by_plan(self, plan_id: str) -> List[Dict[str, Any]]:
        """
        List phases for a plan.

        Args:
            plan_id: Plan identifier

        Returns:
            List of phases ordered by sequence
        """
        query = (
            select(phases)
            .where(phases.c.plan_id == plan_id)
            .order_by(phases.c.sequence_order)
        )

        results = await self.execute_read(query)

        # Parse JSON fields
        for phase in results:
            phase["dependencies"] = json.loads(phase["dependencies"]) if phase["dependencies"] else []

        return results

    async def update_status(
        self, phase_id: str, status: str,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None
    ) -> bool:
        """
        Update phase status.

        Args:
            phase_id: Phase identifier
            status: New status
            started_at: Start timestamp
            completed_at: Completion timestamp

        Returns:
            True if updated successfully
        """
        update_values = {"status": status}

        if started_at:
            update_values["started_at"] = started_at
        if completed_at:
            update_values["completed_at"] = completed_at

        stmt = update(phases).where(phases.c.id == phase_id).values(**update_values)

        affected = await self.execute_write(stmt)
        return affected > 0


class MicroTaskQueries(BaseQuery):
    """Query operations for micro tasks."""

    async def create(
        self,
        phase_id: str,
        title: str,
        description: str,
        assigned_agent: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Create new micro task.

        Args:
            phase_id: Parent phase ID
            title: Task title
            description: Task description
            assigned_agent: Agent assignment
            **kwargs: Additional fields

        Returns:
            Created task ID
        """
        task_id = self.generate_id()

        stmt = micro_tasks.insert().values(
            id=task_id,
            phase_id=phase_id,
            title=title,
            description=description,
            assigned_agent=assigned_agent,
            max_duration_minutes=kwargs.get("max_duration_minutes", 10),
            max_context_tokens=kwargs.get("max_context_tokens", 256000),
            task_type=kwargs.get("task_type", "coding"),
            status="pending",
            priority=kwargs.get("priority", 5),
            input_files=json.dumps(kwargs.get("input_files", [])),
            output_files=json.dumps(kwargs.get("output_files", [])),
        )

        await self.execute_write(stmt)
        logger.info(
            "Created micro task",
            task_id=task_id,
            phase_id=phase_id,
            title=title,
        )
        return task_id

    async def get_pending_by_phase(self, phase_id: str) -> List[Dict[str, Any]]:
        """
        Get pending tasks for a phase.

        Args:
            phase_id: Phase identifier

        Returns:
            List of pending tasks
        """
        query = (
            select(micro_tasks)
            .where(
                and_(
                    micro_tasks.c.phase_id == phase_id,
                    micro_tasks.c.status == "pending",
                )
            )
            .order_by(desc(micro_tasks.c.priority))
        )

        results = await self.execute_read(query)

        # Parse JSON fields
        for task in results:
            task["input_files"] = json.loads(task["input_files"]) if task["input_files"] else []
            task["output_files"] = json.loads(task["output_files"]) if task["output_files"] else []

        return results

    async def update_execution(
        self,
        task_id: str,
        status: str,
        duration_minutes: Optional[float] = None,
        tokens_used: Optional[int] = None,
        generated_code: Optional[str] = None,
        error_log: Optional[str] = None,
    ) -> bool:
        """
        Update task execution results.

        Args:
            task_id: Task identifier
            status: New status
            duration_minutes: Actual duration
            tokens_used: Context tokens used
            generated_code: Generated code output
            error_log: Error information

        Returns:
            True if updated successfully
        """
        values = {
            "status": status,
            "actual_duration_minutes": duration_minutes,
            "context_tokens_used": tokens_used,
        }

        if generated_code:
            values["generated_code"] = generated_code
        if error_log:
            values["error_log"] = error_log

        if status == "completed":
            values["completed_at"] = datetime.utcnow()
        elif status == "active":
            values["started_at"] = datetime.utcnow()

        stmt = update(micro_tasks).where(micro_tasks.c.id == task_id).values(**values)

        affected = await self.execute_write(stmt)
        return affected > 0

    async def update_status(self, task_id: str, status: str) -> bool:
        """
        Update task status.

        Args:
            task_id: Task identifier
            status: New status

        Returns:
            True if updated successfully
        """
        values = {"status": status}

        if status == "completed":
            values["completed_at"] = datetime.utcnow()
        elif status == "in_progress":
            # Map in_progress to active (database constraint)
            values["status"] = "active"
            values["started_at"] = datetime.utcnow()
        elif status == "active":
            values["started_at"] = datetime.utcnow()

        stmt = update(micro_tasks).where(micro_tasks.c.id == task_id).values(**values)

        affected = await self.execute_write(stmt)
        return affected > 0

    async def get_active_tasks(self) -> List[Dict[str, Any]]:
        """
        Get all active tasks.

        Returns:
            List of active tasks
        """
        query = (
            select(micro_tasks)
            .where(micro_tasks.c.status.in_(["pending", "in_progress"]))
            .order_by(desc(micro_tasks.c.priority))
        )

        results = await self.execute_read(query)

        # Parse JSON fields
        for task in results:
            task["input_files"] = json.loads(task["input_files"]) if task["input_files"] else []
            task["output_files"] = json.loads(task["output_files"]) if task["output_files"] else []

        return results


class SemanticMemoryQueries(BaseQuery):
    """Query operations for semantic memory."""

    async def store(
        self,
        content: str,
        content_type: str,
        embedding: Optional[List[float]] = None,
        keywords: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Store semantic memory entry.

        Args:
            content: Memory content
            content_type: Type of content
            embedding: Vector embedding
            keywords: Extracted keywords
            **kwargs: Additional metadata

        Returns:
            Created memory ID
        """
        memory_id = self.generate_id()

        stmt = semantic_memory.insert().values(
            id=memory_id,
            content=content,
            content_type=content_type,
            content_format=kwargs.get("content_format", "text"),
            embedding=json.dumps(embedding) if embedding else None,
            keywords=json.dumps(keywords) if keywords else None,
            entities=json.dumps(kwargs.get("entities", [])),
            plan_id=kwargs.get("plan_id"),
            phase_id=kwargs.get("phase_id"),
            task_id=kwargs.get("task_id"),
            complexity_score=kwargs.get("complexity_score"),
            relevance_score=kwargs.get("relevance_score", 1.0),
        )

        await self.execute_write(stmt)
        logger.info("Stored semantic memory", memory_id=memory_id, type=content_type)
        return memory_id

    async def search_by_keywords(
        self, keywords: List[str], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search memories by keywords.

        Args:
            keywords: Keywords to search
            limit: Maximum results

        Returns:
            List of matching memories
        """
        # Build search condition using LIKE for content search
        conditions = []
        for keyword in keywords:
            conditions.append(semantic_memory.c.content.contains(keyword))

        query = (
            select(semantic_memory)
            .where(
                and_(
                    or_(*conditions) if conditions else True,
                    semantic_memory.c.is_archived == False,
                )
            )
            .order_by(desc(semantic_memory.c.relevance_score))
            .limit(limit)
        )

        results = await self.execute_read(query)

        # Parse JSON fields
        for memory in results:
            memory["keywords"] = json.loads(memory["keywords"]) if memory["keywords"] else []
            memory["entities"] = json.loads(memory["entities"]) if memory["entities"] else []
            memory["embedding"] = json.loads(memory["embedding"]) if memory["embedding"] else None

        return results

    async def get_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory data or None if not found
        """
        query = select(semantic_memory).where(semantic_memory.c.id == memory_id)
        results = await self.execute_read(query)

        if results:
            memory = results[0]
            # Parse JSON fields
            memory["keywords"] = json.loads(memory["keywords"]) if memory["keywords"] else []
            memory["entities"] = json.loads(memory["entities"]) if memory["entities"] else []
            memory["embedding"] = json.loads(memory["embedding"]) if memory["embedding"] else None
            return memory
        return None

    async def update_access(self, memory_id: str) -> bool:
        """
        Update memory access statistics.

        Args:
            memory_id: Memory identifier

        Returns:
            True if updated successfully
        """
        stmt = (
            update(semantic_memory)
            .where(semantic_memory.c.id == memory_id)
            .values(
                access_count=semantic_memory.c.access_count + 1,
                last_accessed_at=datetime.utcnow(),
            )
        )

        affected = await self.execute_write(stmt)
        return affected > 0


class WorkSessionQueries(BaseQuery):
    """Query operations for work sessions."""

    async def create(
        self,
        session_name: str,
        plan_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Create new work session.

        Args:
            session_name: Session name
            plan_id: Associated plan ID
            user_id: User identifier
            **kwargs: Additional fields

        Returns:
            Created session ID
        """
        session_id = self.generate_id()

        stmt = work_sessions.insert().values(
            id=session_id,
            session_name=session_name,
            plan_id=plan_id,
            user_id=user_id,
            context_window_size=kwargs.get("context_window_size", 256000),
            status="active",
            active_tasks=json.dumps([]),
            completed_tasks=json.dumps([]),
        )

        await self.execute_write(stmt)
        logger.info("Created work session", session_id=session_id, name=session_name)
        return session_id

    async def get_active(self) -> Optional[Dict[str, Any]]:
        """
        Get current active session.

        Returns:
            Active session data or None
        """
        query = select(work_sessions).where(work_sessions.c.status == "active").limit(1)

        results = await self.execute_read(query)

        if results:
            session = results[0]
            session["active_tasks"] = json.loads(session["active_tasks"])
            session["completed_tasks"] = json.loads(session["completed_tasks"])
            return session
        return None

    async def update_tasks(
        self, session_id: str, task_id: str, completed: bool = False
    ) -> bool:
        """
        Update session task lists.

        Args:
            session_id: Session identifier
            task_id: Task to update
            completed: Whether task is completed

        Returns:
            True if updated successfully
        """
        # Get current session
        query = select(work_sessions).where(work_sessions.c.id == session_id)
        results = await self.execute_read(query)

        if not results:
            return False

        session = results[0]
        active = json.loads(session["active_tasks"])
        completed_list = json.loads(session["completed_tasks"])

        if completed:
            if task_id in active:
                active.remove(task_id)
            if task_id not in completed_list:
                completed_list.append(task_id)
        else:
            if task_id not in active:
                active.append(task_id)

        stmt = (
            update(work_sessions)
            .where(work_sessions.c.id == session_id)
            .values(
                active_tasks=json.dumps(active),
                completed_tasks=json.dumps(completed_list),
                last_activity_at=datetime.utcnow(),
            )
        )

        affected = await self.execute_write(stmt)
        return affected > 0


class QueryManager:
    """Manager for all database queries."""

    def __init__(self, pool: ConnectionPool):
        """
        Initialize query manager.

        Args:
            pool: Connection pool for database operations
        """
        self.pool = pool
        self.plans = InterventionPlanQueries(pool)
        self.phases = PhaseQueries(pool)
        self.tasks = MicroTaskQueries(pool)
        self.memory = SemanticMemoryQueries(pool)
        self.sessions = WorkSessionQueries(pool)

    async def health_check(self) -> bool:
        """
        Perform database health check.

        Returns:
            True if database is healthy
        """
        try:
            async with self.pool.read_transaction() as conn:
                result = await conn.execute(text("SELECT 1"))
                result.fetchone()
                return True
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False