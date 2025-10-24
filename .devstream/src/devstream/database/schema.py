"""
Database schema definition usando SQLAlchemy Core.

Definisce tutte le tabelle, indexes, e constraints per il sistema DevStream.
Usa SQLAlchemy Core (non ORM) per type safety senza overhead.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    TIMESTAMP,
    Boolean,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    CheckConstraint,
    create_engine,
)
from sqlalchemy.sql import func

# Metadata container for all tables
metadata = MetaData()

# ============================================================================
# TASK MANAGEMENT TABLES
# ============================================================================

intervention_plans = Table(
    "intervention_plans",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("title", String(200), nullable=False),
    Column("description", Text),
    Column("objectives", JSON, nullable=False),  # Array of objectives
    Column("technical_specs", JSON),  # Technical specifications
    Column("expected_outcome", Text, nullable=False),
    Column(
        "status",
        String(20),
        CheckConstraint(
            "status IN ('draft', 'active', 'completed', 'archived', 'cancelled')"
        ),
        default="draft",
    ),
    Column("priority", Integer, CheckConstraint("priority BETWEEN 1 AND 10"), default=5),
    Column("estimated_hours", Float),
    Column("actual_hours", Float, default=0.0),
    Column("tags", JSON),  # Array of tags
    Column("metadata", JSON),  # Additional flexible metadata
    Column("created_at", TIMESTAMP, server_default=func.current_timestamp()),
    Column(
        "updated_at",
        TIMESTAMP,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    ),
    Column("completed_at", TIMESTAMP, nullable=True),
)

phases = Table(
    "phases",
    metadata,
    Column("id", String(32), primary_key=True),
    Column(
        "plan_id",
        String(32),
        ForeignKey("intervention_plans.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("name", String(200), nullable=False),
    Column("description", Text),
    Column("sequence_order", Integer, nullable=False),
    Column("is_parallel", Boolean, default=False),
    Column("dependencies", JSON),  # Array of phase_ids
    Column(
        "status",
        String(20),
        CheckConstraint(
            "status IN ('pending', 'active', 'completed', 'blocked', 'skipped')"
        ),
        default="pending",
    ),
    Column("estimated_minutes", Integer),
    Column("actual_minutes", Integer, default=0),
    Column("blocking_reason", Text),
    Column("completion_criteria", Text),
    Column("created_at", TIMESTAMP, server_default=func.current_timestamp()),
    Column("started_at", TIMESTAMP),
    Column("completed_at", TIMESTAMP),
    Index("idx_phases_plan_status", "plan_id", "status"),
    Index("idx_phases_sequence", "plan_id", "sequence_order"),
)

micro_tasks = Table(
    "micro_tasks",
    metadata,
    Column("id", String(32), primary_key=True),
    Column(
        "phase_id",
        String(32),
        ForeignKey("phases.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("title", String(200), nullable=False),
    Column("description", Text, nullable=False),
    Column(
        "max_duration_minutes",
        Integer,
        CheckConstraint("max_duration_minutes <= 10"),
        default=10,
    ),
    Column("max_context_tokens", Integer, default=256000),
    Column("assigned_agent", String(50)),
    Column(
        "task_type",
        String(20),
        CheckConstraint(
            "task_type IN ('analysis', 'coding', 'documentation', 'testing', 'review', 'research')"
        ),
        default="coding",
    ),
    Column(
        "status",
        String(20),
        CheckConstraint(
            "status IN ('pending', 'active', 'completed', 'failed', 'skipped')"
        ),
        default="pending",
    ),
    Column("priority", Integer, CheckConstraint("priority BETWEEN 1 AND 10"), default=5),
    # Input/Output tracking
    Column("input_files", JSON),  # Array of file paths
    Column("output_files", JSON),  # Array of file paths
    Column("generated_code", Text),
    Column("documentation", Text),
    Column("error_log", Text),
    # Execution metadata
    Column("actual_duration_minutes", Float),
    Column("context_tokens_used", Integer),
    Column("retry_count", Integer, default=0),
    Column("parent_task_id", String(32), ForeignKey("micro_tasks.id")),
    # Timestamps
    Column("created_at", TIMESTAMP, server_default=func.current_timestamp()),
    Column("started_at", TIMESTAMP),
    Column("completed_at", TIMESTAMP),
    Column("last_retry_at", TIMESTAMP),
    Index("idx_tasks_phase_status", "phase_id", "status"),
    Index("idx_tasks_agent", "assigned_agent"),
    Index("idx_tasks_type_status", "task_type", "status"),
)

# ============================================================================
# MEMORY SYSTEM TABLES
# ============================================================================

semantic_memory = Table(
    "semantic_memory",
    metadata,
    Column("id", String(32), primary_key=True),
    # Foreign keys (nullable for flexibility)
    Column("plan_id", String(32), ForeignKey("intervention_plans.id", ondelete="CASCADE")),
    Column("phase_id", String(32), ForeignKey("phases.id", ondelete="CASCADE")),
    Column("task_id", String(32), ForeignKey("micro_tasks.id", ondelete="CASCADE")),
    # Content
    Column("content", Text, nullable=False),
    Column(
        "content_type",
        String(20),
        CheckConstraint(
            "content_type IN ('code', 'documentation', 'context', 'output', 'error', 'decision', 'learning')"
        ),
        nullable=False,
    ),
    Column(
        "content_format",
        String(20),
        CheckConstraint("content_format IN ('text', 'markdown', 'code', 'json', 'yaml')"),
        default="text",
    ),
    # Semantic metadata
    Column("keywords", JSON),  # Extracted keywords
    Column("entities", JSON),  # Recognized entities
    Column("sentiment", Float),  # Sentiment score (-1 to 1)
    Column("complexity_score", Integer, CheckConstraint("complexity_score BETWEEN 1 AND 10")),
    # Embedding storage
    Column("embedding", Text),  # Base64 encoded or JSON array
    Column("embedding_model", String(50), default="embeddinggemma"),
    Column("embedding_dimension", Integer, default=384),
    # Context and relations
    Column("context_snapshot", JSON),
    Column("related_memory_ids", JSON),
    # Management metadata
    Column("access_count", Integer, default=0),
    Column("last_accessed_at", TIMESTAMP),
    Column("relevance_score", Float, default=1.0),
    Column("is_archived", Boolean, default=False),
    # Timestamps
    Column("created_at", TIMESTAMP, server_default=func.current_timestamp()),
    Column("updated_at", TIMESTAMP, server_default=func.current_timestamp()),
    Index("idx_memory_task", "task_id"),
    Index("idx_memory_type", "content_type"),
    Index("idx_memory_created", "created_at"),
    Index("idx_memory_relevance", "relevance_score"),
)

# ============================================================================
# AGENT AND HOOK SYSTEM TABLES
# ============================================================================

agents = Table(
    "agents",
    metadata,
    Column("id", String(50), primary_key=True),
    Column("name", String(100), nullable=False),
    Column("role", String(100), nullable=False),
    Column("description", Text),
    Column("capabilities", JSON, nullable=False),  # Array of capabilities
    Column("triggers", JSON),  # Array of trigger events
    Column("config", JSON),  # Agent-specific configuration
    Column("is_active", Boolean, default=True),
    Column("success_rate", Float, default=0.0),
    Column("total_tasks", Integer, default=0),
    Column("successful_tasks", Integer, default=0),
    Column("created_at", TIMESTAMP, server_default=func.current_timestamp()),
    Column("updated_at", TIMESTAMP, server_default=func.current_timestamp()),
)

hooks = Table(
    "hooks",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("name", String(100), nullable=False),
    Column("event_type", String(50), nullable=False),
    Column("trigger_condition", Text),  # Condition expression
    Column("action_type", String(50), nullable=False),
    Column("action_config", JSON),
    Column("is_active", Boolean, default=True),
    Column("execution_order", Integer, default=100),
    Column("created_at", TIMESTAMP, server_default=func.current_timestamp()),
    Index("idx_hooks_event_type", "event_type"),
    Index("idx_hooks_active", "is_active"),
)

hook_executions = Table(
    "hook_executions",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("hook_id", String(32), ForeignKey("hooks.id"), nullable=False),
    Column("event_data", JSON),
    Column("execution_result", JSON),
    Column(
        "status",
        String(20),
        CheckConstraint("status IN ('success', 'failed', 'skipped')"),
        nullable=False,
    ),
    Column("error_message", Text),
    Column("execution_time_ms", Integer),
    Column("created_at", TIMESTAMP, server_default=func.current_timestamp()),
    Index("idx_hook_exec_hook", "hook_id"),
    Index("idx_hook_exec_status", "status"),
)

# ============================================================================
# SESSION AND CONTEXT TABLES
# ============================================================================

work_sessions = Table(
    "work_sessions",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("plan_id", String(32), ForeignKey("intervention_plans.id")),
    Column("user_id", String(100)),
    Column("session_name", String(200)),
    Column("context_window_size", Integer, default=256000),
    Column("tokens_used", Integer, default=0),
    Column(
        "status",
        String(20),
        CheckConstraint("status IN ('active', 'paused', 'completed', 'archived')"),
        default="active",
    ),
    Column("context_summary", Text),
    Column("active_tasks", JSON),  # Array of task_ids
    Column("completed_tasks", JSON),  # Array of task_ids
    Column("started_at", TIMESTAMP, server_default=func.current_timestamp()),
    Column("last_activity_at", TIMESTAMP, server_default=func.current_timestamp()),
    Column("ended_at", TIMESTAMP),
    Index("idx_sessions_status", "status"),
    Index("idx_sessions_plan", "plan_id"),
)

context_injections = Table(
    "context_injections",
    metadata,
    Column("id", String(32), primary_key=True),
    Column(
        "session_id",
        String(32),
        ForeignKey("work_sessions.id"),
        nullable=False,
    ),
    Column("task_id", String(32), ForeignKey("micro_tasks.id")),
    Column("injected_memory_ids", JSON),  # Array of memory_ids
    Column("injection_trigger", String(100)),
    Column("relevance_threshold", Float),
    Column("tokens_injected", Integer),
    Column("effectiveness_score", Float),
    Column("created_at", TIMESTAMP, server_default=func.current_timestamp()),
    Index("idx_injections_session", "session_id"),
    Index("idx_injections_task", "task_id"),
)

# ============================================================================
# ANALYTICS AND METRICS TABLES
# ============================================================================

performance_metrics = Table(
    "performance_metrics",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("metric_type", String(50), nullable=False),
    Column("entity_type", String(50), nullable=False),
    Column("entity_id", String(32), nullable=False),
    Column("metric_value", Float, nullable=False),
    Column("metric_unit", String(20)),
    Column("context", JSON),
    Column("recorded_at", TIMESTAMP, server_default=func.current_timestamp()),
    Index("idx_metrics_type", "metric_type"),
    Index("idx_metrics_entity", "entity_type", "entity_id"),
    Index("idx_metrics_recorded", "recorded_at"),
)

learning_insights = Table(
    "learning_insights",
    metadata,
    Column("id", String(32), primary_key=True),
    Column(
        "insight_type",
        String(20),
        CheckConstraint("insight_type IN ('pattern', 'best_practice', 'anti_pattern')"),
        nullable=False,
    ),
    Column("title", String(200), nullable=False),
    Column("description", Text, nullable=False),
    Column("confidence_score", Float, CheckConstraint("confidence_score BETWEEN 0 AND 1")),
    Column("supporting_evidence", JSON),
    Column("tags", JSON),
    Column("is_validated", Boolean, default=False),
    Column("validation_feedback", Text),
    Column("created_at", TIMESTAMP, server_default=func.current_timestamp()),
    Column("validated_at", TIMESTAMP),
    Index("idx_insights_type", "insight_type"),
    Index("idx_insights_validated", "is_validated"),
)

# ============================================================================
# SCHEMA VERSION TRACKING
# ============================================================================

schema_version = Table(
    "schema_version",
    metadata,
    Column("version", String(20), primary_key=True),
    Column("applied_at", TIMESTAMP, server_default=func.current_timestamp()),
    Column("description", Text),
)


def get_table_creation_order() -> list[Table]:
    """
    Get tables in correct creation order respecting foreign keys.

    Returns:
        Ordered list of tables for creation
    """
    return [
        # Independent tables first
        agents,
        hooks,
        schema_version,
        # Task management hierarchy
        intervention_plans,
        phases,
        micro_tasks,
        # Memory and context
        semantic_memory,
        work_sessions,
        # Dependent tables
        hook_executions,
        context_injections,
        # Analytics
        performance_metrics,
        learning_insights,
    ]


def get_indexes() -> list[Index]:
    """
    Get all indexes defined in schema.

    Returns:
        List of all indexes
    """
    indexes = []
    for table in metadata.tables.values():
        indexes.extend(table.indexes)
    return indexes