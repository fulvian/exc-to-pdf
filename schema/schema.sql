-- ============================================================================
-- DevStream Database Schema
-- SQLite 3 with Extensions: sqlite-vec (vector search)
-- Version: 2.1.0
-- Date: 2025-10-01
--
-- Purpose: DevStream Intervention Planning & Semantic Memory System
--          Combines task lifecycle management, semantic memory with vector
--          embeddings, and context injection for AI-assisted development.
--
-- Key Features:
--   - Intervention Plans & Phases: Hierarchical project structure
--   - Micro Tasks: Atomic work units (max 10 min) with agent assignment
--   - Semantic Memory: Code, documentation, decisions with embeddings
--   - Vector Search: sqlite-vec for semantic similarity search
--   - Full-Text Search: FTS5 for keyword search
--   - Hybrid Search: RRF (Reciprocal Rank Fusion) combining both
--   - Hooks & Agents: Automated workflow triggers
--   - Performance Metrics: Comprehensive tracking
-- ============================================================================

-- ============================================================================
-- SCHEMA VERSION TRACKING
-- Purpose: Track schema migrations and versions
-- Usage: INSERT INTO schema_version (version, description) VALUES ('2.1.0', 'Initial production schema')
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_version (
    version TEXT PRIMARY KEY,              -- Semantic version (e.g., '2.1.0')
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT                       -- Migration description
);

-- ============================================================================
-- INTERVENTION PLANS
-- Purpose: Top-level project/feature planning
-- Relationships: Parent to phases -> micro_tasks
-- Key Columns:
--   - objectives: JSON array of project goals
--   - technical_specs: JSON technical requirements
--   - status: Lifecycle state (draft/active/completed/archived/cancelled)
--   - priority: 1-10 (higher = more important)
-- ============================================================================

CREATE TABLE IF NOT EXISTS intervention_plans (
    id VARCHAR(32) NOT NULL PRIMARY KEY,  -- UUID format (e.g., 'PLAN-001')
    title VARCHAR(200) NOT NULL,          -- Human-readable plan title
    description TEXT,                     -- Detailed plan description
    objectives JSON NOT NULL,             -- JSON array: ["objective1", "objective2"]
    technical_specs JSON,                 -- JSON object: {"framework": "FastAPI", ...}
    expected_outcome TEXT NOT NULL,       -- Success criteria
    status VARCHAR(20) CHECK (status IN ('draft', 'active', 'completed', 'archived', 'cancelled')),
    priority INTEGER CHECK (priority BETWEEN 1 AND 10),
    estimated_hours FLOAT,                -- Initial time estimate
    actual_hours FLOAT,                   -- Actual time spent
    tags JSON,                            -- JSON array: ["backend", "api"]
    metadata JSON,                        -- Additional structured data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP                -- NULL until completed
);

-- ============================================================================
-- PHASES
-- Purpose: Break down intervention plans into logical phases
-- Relationships: Child of intervention_plans, parent to micro_tasks
-- Key Columns:
--   - sequence_order: Execution order within plan
--   - is_parallel: Can execute concurrently with other phases
--   - dependencies: JSON array of phase IDs that must complete first
--   - blocking_reason: Why phase is blocked (if status='blocked')
-- ============================================================================

CREATE TABLE IF NOT EXISTS phases (
    id VARCHAR(32) NOT NULL PRIMARY KEY,
    plan_id VARCHAR(32) NOT NULL,         -- Foreign key to intervention_plans
    name VARCHAR(200) NOT NULL,           -- Phase name (e.g., "Core Engine & Infrastructure")
    description TEXT,                     -- Phase description
    sequence_order INTEGER NOT NULL,      -- Execution order (1, 2, 3, ...)
    is_parallel BOOLEAN,                  -- Can run concurrently with other phases
    dependencies JSON,                    -- JSON array: ["PHASE-001", "PHASE-002"]
    status VARCHAR(20) CHECK (status IN ('pending', 'active', 'completed', 'blocked', 'skipped')),
    estimated_minutes INTEGER,            -- Time estimate for phase
    actual_minutes INTEGER,               -- Actual time spent
    blocking_reason TEXT,                 -- Description of blocker
    completion_criteria TEXT,             -- What defines "done"
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,                 -- When phase started
    completed_at TIMESTAMP,               -- When phase completed
    FOREIGN KEY(plan_id) REFERENCES intervention_plans(id) ON DELETE CASCADE
);

-- ============================================================================
-- MICRO_TASKS
-- Purpose: Atomic work units (max 10 minutes)
-- Relationships: Child of phases, can have parent_task_id for sub-tasks
-- Key Columns:
--   - max_duration_minutes: Hard limit (10 min for micro-tasks)
--   - max_context_tokens: Token budget for task
--   - assigned_agent: Which agent should handle this (e.g., '@python-specialist')
--   - task_type: analysis/coding/documentation/testing/review/research
--   - input_files/output_files: JSON arrays of file paths
--   - generated_code: Code generated by task
--   - retry_count: Number of retry attempts
-- ============================================================================

CREATE TABLE IF NOT EXISTS micro_tasks (
    id VARCHAR(32) NOT NULL PRIMARY KEY,
    phase_id VARCHAR(32) NOT NULL,        -- Foreign key to phases
    title VARCHAR(200) NOT NULL,          -- Task title
    description TEXT NOT NULL,            -- Detailed task description
    max_duration_minutes INTEGER CHECK (max_duration_minutes <= 10),
    max_context_tokens INTEGER,           -- Token budget
    assigned_agent VARCHAR(50),           -- Agent ID (e.g., '@python-specialist')
    task_type VARCHAR(20) CHECK (task_type IN ('analysis', 'coding', 'documentation', 'testing', 'review', 'research')),
    status VARCHAR(20) CHECK (status IN ('pending', 'active', 'completed', 'failed', 'skipped')),
    priority INTEGER CHECK (priority BETWEEN 1 AND 10),
    input_files JSON,                     -- JSON array: ["file1.py", "file2.py"]
    output_files JSON,                    -- JSON array: ["output1.py"]
    generated_code TEXT,                  -- Code generated by task
    documentation TEXT,                   -- Documentation generated
    error_log TEXT,                       -- Error messages if failed
    actual_duration_minutes FLOAT,        -- Actual time spent
    context_tokens_used INTEGER,          -- Actual tokens used
    retry_count INTEGER,                  -- Number of retries
    parent_task_id VARCHAR(32),           -- For sub-tasks
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,                 -- When task started
    completed_at TIMESTAMP,               -- When task completed
    last_retry_at TIMESTAMP,              -- Last retry timestamp
    FOREIGN KEY(phase_id) REFERENCES phases(id) ON DELETE CASCADE,
    FOREIGN KEY(parent_task_id) REFERENCES micro_tasks(id)
);

-- ============================================================================
-- SEMANTIC_MEMORY
-- Purpose: Store all content (code, docs, decisions) with vector embeddings
-- Relationships: Can link to plan_id, phase_id, task_id
-- Key Columns:
--   - content: Full text content
--   - content_type: code/documentation/context/output/error/decision/learning
--   - content_format: text/markdown/code/json/yaml
--   - keywords: JSON array for keyword search
--   - embedding: Vector embedding (768-dim float array as TEXT)
--   - embedding_model: Model used (e.g., 'nomic-embed-text')
--   - context_snapshot: JSON snapshot of execution context
--   - related_memory_ids: JSON array of related memory IDs
--
-- Integration:
--   - Triggers sync to vec_semantic_memory (vector search)
--   - Triggers sync to fts_semantic_memory (keyword search)
-- ============================================================================

CREATE TABLE IF NOT EXISTS semantic_memory (
    id VARCHAR(32) NOT NULL PRIMARY KEY,
    plan_id VARCHAR(32),                  -- Optional: link to intervention plan
    phase_id VARCHAR(32),                 -- Optional: link to phase
    task_id VARCHAR(32),                  -- Optional: link to micro task
    content TEXT NOT NULL,                -- Full content (code, docs, etc.)
    content_type VARCHAR(20) NOT NULL CHECK (content_type IN ('code', 'documentation', 'context', 'output', 'error', 'decision', 'learning')),
    content_format VARCHAR(20) CHECK (content_format IN ('text', 'markdown', 'code', 'json', 'yaml')),
    keywords JSON,                        -- JSON array: ["python", "fastapi", "async"]
    entities JSON,                        -- JSON array: extracted entities
    sentiment FLOAT,                      -- Sentiment score (-1 to 1)
    complexity_score INTEGER CHECK (complexity_score BETWEEN 1 AND 10),
    embedding TEXT,                       -- Vector embedding (768-dim float array serialized as TEXT)
    embedding_model VARCHAR(50),          -- Model name (e.g., 'nomic-embed-text')
    embedding_dimension INTEGER,          -- Dimension count (768 for nomic-embed-text)
    context_snapshot JSON,                -- JSON: execution context at creation time
    related_memory_ids JSON,              -- JSON array: ["MEM-001", "MEM-002"]
    access_count INTEGER,                 -- How many times accessed
    last_accessed_at TIMESTAMP,           -- Last access timestamp
    relevance_score FLOAT,                -- Dynamic relevance score
    is_archived BOOLEAN,                  -- Archived flag
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source TEXT,                          -- Source information
    importance_score REAL,                -- Importance score (0-1)
    metadata TEXT,                        -- Additional metadata
    FOREIGN KEY(plan_id) REFERENCES intervention_plans(id) ON DELETE CASCADE,
    FOREIGN KEY(phase_id) REFERENCES phases(id) ON DELETE CASCADE,
    FOREIGN KEY(task_id) REFERENCES micro_tasks(id) ON DELETE CASCADE
);

-- ============================================================================
-- VECTOR SEARCH TABLE (sqlite-vec Extension)
-- Purpose: Fast semantic similarity search using vector embeddings
-- Schema: VIRTUAL TABLE with vec0 extension
-- Key Columns:
--   - embedding: 768-dimensional float vector
--   - content_type: Partition key for filtered searches
--   - memory_id: Link back to semantic_memory.id
--   - content_preview: First 200 chars for display
--
-- Usage:
--   SELECT memory_id, distance
--   FROM vec_semantic_memory
--   WHERE embedding MATCH <query_vector>
--     AND k = 10
--     AND content_type = 'code'
--   ORDER BY distance;
-- ============================================================================

CREATE VIRTUAL TABLE IF NOT EXISTS vec_semantic_memory USING vec0(
    embedding float[768],                 -- 768-dimensional vector
    content_type TEXT PARTITION KEY,      -- Enables partition filtering
    +memory_id TEXT,                      -- Link to semantic_memory.id
    +content_preview TEXT                 -- First 200 chars
);

-- ============================================================================
-- FULL-TEXT SEARCH TABLE (FTS5 Extension)
-- Purpose: Fast keyword search on semantic_memory content
-- Schema: VIRTUAL TABLE with FTS5 extension
-- Key Columns:
--   - content: Indexed full-text content
--   - content_type: Unindexed (for filtering)
--   - memory_id: Unindexed (link to semantic_memory)
--   - created_at: Unindexed (for sorting)
--
-- Usage:
--   SELECT memory_id, rank
--   FROM fts_semantic_memory
--   WHERE fts_semantic_memory MATCH 'fastapi AND async'
--   ORDER BY rank;
-- ============================================================================

CREATE VIRTUAL TABLE IF NOT EXISTS fts_semantic_memory USING fts5(
    content,                              -- Full-text indexed content
    content_type UNINDEXED,               -- Filter by content type
    memory_id UNINDEXED,                  -- Link to semantic_memory.id
    created_at UNINDEXED,                 -- Timestamp for sorting
    tokenize='unicode61 remove_diacritics 2'  -- Unicode tokenizer with diacritics removal
);

-- ============================================================================
-- AGENTS
-- Purpose: Track available agents and their performance
-- Key Columns:
--   - role: Agent role (e.g., 'Orchestrator', 'Domain Specialist')
--   - capabilities: JSON object describing agent skills
--   - triggers: JSON array of trigger patterns
--   - success_rate: Performance metric (0-1)
-- ============================================================================

CREATE TABLE IF NOT EXISTS agents (
    id VARCHAR(50) NOT NULL PRIMARY KEY,  -- Agent ID (e.g., '@tech-lead')
    name VARCHAR(100) NOT NULL,           -- Human-readable name
    role VARCHAR(100) NOT NULL,           -- Role category
    description TEXT,                     -- Agent description
    capabilities JSON NOT NULL,           -- JSON object: {"languages": ["python"], ...}
    triggers JSON,                        -- JSON array: trigger patterns
    config JSON,                          -- Agent configuration
    is_active BOOLEAN,                    -- Active flag
    success_rate FLOAT,                   -- Success rate (0-1)
    total_tasks INTEGER,                  -- Total tasks assigned
    successful_tasks INTEGER,             -- Successfully completed tasks
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- HOOKS
-- Purpose: Define automated workflow triggers
-- Key Columns:
--   - event_type: Hook trigger event (e.g., 'PreToolUse', 'PostToolUse')
--   - trigger_condition: Condition expression
--   - action_type: Action to perform
--   - action_config: JSON configuration
--   - execution_order: Order of execution (lower = earlier)
-- ============================================================================

CREATE TABLE IF NOT EXISTS hooks (
    id VARCHAR(32) NOT NULL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,           -- Hook name
    event_type VARCHAR(50) NOT NULL,      -- Event trigger type
    trigger_condition TEXT,               -- Condition expression
    action_type VARCHAR(50) NOT NULL,     -- Action type
    action_config JSON,                   -- JSON: action configuration
    is_active BOOLEAN,                    -- Active flag
    execution_order INTEGER,              -- Execution order
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- HOOK_EXECUTIONS
-- Purpose: Track hook execution history
-- Key Columns:
--   - hook_id: Foreign key to hooks
--   - event_data: JSON snapshot of event
--   - execution_result: JSON result data
--   - status: success/failed/skipped
--   - execution_time_ms: Performance metric
-- ============================================================================

CREATE TABLE IF NOT EXISTS hook_executions (
    id VARCHAR(32) NOT NULL PRIMARY KEY,
    hook_id VARCHAR(32) NOT NULL,         -- Foreign key to hooks
    event_data JSON,                      -- JSON: event data
    execution_result JSON,                -- JSON: result data
    status VARCHAR(20) NOT NULL CHECK (status IN ('success', 'failed', 'skipped')),
    error_message TEXT,                   -- Error message if failed
    execution_time_ms INTEGER,            -- Execution time in milliseconds
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(hook_id) REFERENCES hooks(id)
);

-- ============================================================================
-- WORK_SESSIONS
-- Purpose: Track user work sessions for context management
-- Key Columns:
--   - context_window_size: Max context tokens
--   - tokens_used: Current token usage
--   - status: active/paused/completed/archived
--   - active_tasks: JSON array of active task IDs
-- ============================================================================

CREATE TABLE IF NOT EXISTS work_sessions (
    id VARCHAR(32) NOT NULL PRIMARY KEY,
    plan_id VARCHAR(32),                  -- Optional: link to intervention plan
    user_id VARCHAR(100),                 -- User identifier
    session_name VARCHAR(200),            -- Session name
    context_window_size INTEGER,          -- Max context tokens
    tokens_used INTEGER,                  -- Current token usage
    status VARCHAR(20) CHECK (status IN ('active', 'paused', 'completed', 'archived')),
    context_summary TEXT,                 -- Summary of session context
    active_tasks JSON,                    -- JSON array: ["TASK-001", "TASK-002"]
    completed_tasks JSON,                 -- JSON array: completed task IDs
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,                   -- When session ended
    FOREIGN KEY(plan_id) REFERENCES intervention_plans(id)
);

-- ============================================================================
-- CONTEXT_INJECTIONS
-- Purpose: Track context injection events for memory retrieval
-- Key Columns:
--   - injected_memory_ids: JSON array of injected memory IDs
--   - injection_trigger: What triggered injection
--   - relevance_threshold: Minimum relevance score
--   - tokens_injected: Token count
--   - effectiveness_score: How effective was injection (0-1)
-- ============================================================================

CREATE TABLE IF NOT EXISTS context_injections (
    id VARCHAR(32) NOT NULL PRIMARY KEY,
    session_id VARCHAR(32) NOT NULL,      -- Foreign key to work_sessions
    task_id VARCHAR(32),                  -- Optional: link to task
    injected_memory_ids JSON,             -- JSON array: ["MEM-001", "MEM-002"]
    injection_trigger VARCHAR(100),       -- Trigger description
    relevance_threshold FLOAT,            -- Minimum relevance score used
    tokens_injected INTEGER,              -- Number of tokens injected
    effectiveness_score FLOAT,            -- Effectiveness score (0-1)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES work_sessions(id),
    FOREIGN KEY(task_id) REFERENCES micro_tasks(id)
);

-- ============================================================================
-- LEARNING_INSIGHTS
-- Purpose: Track learned patterns and best practices
-- Key Columns:
--   - insight_type: pattern/best_practice/anti_pattern
--   - confidence_score: Confidence in insight (0-1)
--   - supporting_evidence: JSON array of evidence
--   - is_validated: Manual validation flag
-- ============================================================================

CREATE TABLE IF NOT EXISTS learning_insights (
    id VARCHAR(32) NOT NULL PRIMARY KEY,
    insight_type VARCHAR(20) NOT NULL CHECK (insight_type IN ('pattern', 'best_practice', 'anti_pattern')),
    title VARCHAR(200) NOT NULL,          -- Insight title
    description TEXT NOT NULL,            -- Detailed description
    confidence_score FLOAT CHECK (confidence_score BETWEEN 0 AND 1),
    supporting_evidence JSON,             -- JSON array: evidence references
    tags JSON,                            -- JSON array: tags
    is_validated BOOLEAN,                 -- Manual validation flag
    validation_feedback TEXT,             -- Feedback on validation
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    validated_at TIMESTAMP                -- When validated
);

-- ============================================================================
-- PERFORMANCE_METRICS
-- Purpose: Track performance metrics for various entities
-- Key Columns:
--   - metric_type: Type of metric (e.g., 'execution_time', 'token_usage')
--   - entity_type: What is measured (e.g., 'task', 'agent', 'hook')
--   - entity_id: ID of measured entity
--   - metric_value: Numeric value
--   - metric_unit: Unit of measurement (e.g., 'ms', 'tokens')
-- ============================================================================

CREATE TABLE IF NOT EXISTS performance_metrics (
    id VARCHAR(32) NOT NULL PRIMARY KEY,
    metric_type VARCHAR(50) NOT NULL,     -- Metric type
    entity_type VARCHAR(50) NOT NULL,     -- Entity type being measured
    entity_id VARCHAR(32) NOT NULL,       -- Entity ID
    metric_value FLOAT NOT NULL,          -- Metric value
    metric_unit VARCHAR(20),              -- Unit (e.g., 'ms', 'tokens', 'MB')
    context JSON,                         -- JSON: additional context
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- TRIGGERS - Automatic Sync Between Tables
-- Purpose: Keep semantic_memory, vec_semantic_memory, and fts_semantic_memory in sync
-- ============================================================================

-- Trigger: Insert into semantic_memory -> sync to vec & fts
CREATE TRIGGER IF NOT EXISTS sync_insert_memory
  AFTER INSERT ON semantic_memory
  WHEN NEW.embedding IS NOT NULL
  BEGIN
    -- Insert into vec0 table (vector search)
    INSERT INTO vec_semantic_memory(embedding, content_type, memory_id, content_preview)
    VALUES (NEW.embedding, NEW.content_type, NEW.id, substr(NEW.content, 1, 200));

    -- Insert into FTS5 table (keyword search)
    INSERT INTO fts_semantic_memory(rowid, content, content_type, memory_id, created_at)
    VALUES (NEW.rowid, NEW.content, NEW.content_type, NEW.id, NEW.created_at);
  END;

-- Trigger: Update semantic_memory -> sync to vec & fts
CREATE TRIGGER IF NOT EXISTS sync_update_memory
  AFTER UPDATE ON semantic_memory
  WHEN NEW.embedding IS NOT NULL
  BEGIN
    -- Delete old entries
    DELETE FROM vec_semantic_memory WHERE rowid = OLD.rowid;
    DELETE FROM fts_semantic_memory WHERE rowid = OLD.rowid;

    -- Insert updated entries
    INSERT INTO vec_semantic_memory(embedding, content_type, memory_id, content_preview)
    VALUES (NEW.embedding, NEW.content_type, NEW.id, substr(NEW.content, 1, 200));

    INSERT INTO fts_semantic_memory(rowid, content, content_type, memory_id, created_at)
    VALUES (NEW.rowid, NEW.content, NEW.content_type, NEW.id, NEW.created_at);
  END;

-- Trigger: Delete from semantic_memory -> sync to vec & fts
CREATE TRIGGER IF NOT EXISTS sync_delete_memory
  AFTER DELETE ON semantic_memory
  BEGIN
    DELETE FROM vec_semantic_memory WHERE rowid = OLD.rowid;
    DELETE FROM fts_semantic_memory WHERE rowid = OLD.rowid;
  END;

-- ============================================================================
-- INDEXES - Performance Optimization
-- ============================================================================

-- Intervention Plans
CREATE INDEX IF NOT EXISTS idx_intervention_plans_status ON intervention_plans(status);
CREATE INDEX IF NOT EXISTS idx_intervention_plans_priority ON intervention_plans(priority DESC);
CREATE INDEX IF NOT EXISTS idx_intervention_plans_created_at ON intervention_plans(created_at DESC);

-- Phases
CREATE INDEX IF NOT EXISTS idx_phases_plan_id ON phases(plan_id);
CREATE INDEX IF NOT EXISTS idx_phases_status ON phases(status);
CREATE INDEX IF NOT EXISTS idx_phases_sequence_order ON phases(plan_id, sequence_order);

-- Micro Tasks
CREATE INDEX IF NOT EXISTS idx_micro_tasks_phase_id ON micro_tasks(phase_id);
CREATE INDEX IF NOT EXISTS idx_micro_tasks_status ON micro_tasks(status);
CREATE INDEX IF NOT EXISTS idx_micro_tasks_assigned_agent ON micro_tasks(assigned_agent);
CREATE INDEX IF NOT EXISTS idx_micro_tasks_priority ON micro_tasks(priority DESC);
CREATE INDEX IF NOT EXISTS idx_micro_tasks_parent_task_id ON micro_tasks(parent_task_id);

-- Semantic Memory
CREATE INDEX IF NOT EXISTS idx_semantic_memory_content_type ON semantic_memory(content_type);
CREATE INDEX IF NOT EXISTS idx_semantic_memory_plan_id ON semantic_memory(plan_id);
CREATE INDEX IF NOT EXISTS idx_semantic_memory_phase_id ON semantic_memory(phase_id);
CREATE INDEX IF NOT EXISTS idx_semantic_memory_task_id ON semantic_memory(task_id);
CREATE INDEX IF NOT EXISTS idx_semantic_memory_created_at ON semantic_memory(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_semantic_memory_access_count ON semantic_memory(access_count DESC);

-- Work Sessions
CREATE INDEX IF NOT EXISTS idx_work_sessions_plan_id ON work_sessions(plan_id);
CREATE INDEX IF NOT EXISTS idx_work_sessions_status ON work_sessions(status);
CREATE INDEX IF NOT EXISTS idx_work_sessions_started_at ON work_sessions(started_at DESC);

-- Context Injections
CREATE INDEX IF NOT EXISTS idx_context_injections_session_id ON context_injections(session_id);
CREATE INDEX IF NOT EXISTS idx_context_injections_task_id ON context_injections(task_id);
CREATE INDEX IF NOT EXISTS idx_context_injections_created_at ON context_injections(created_at DESC);

-- Hooks
CREATE INDEX IF NOT EXISTS idx_hooks_event_type ON hooks(event_type);
CREATE INDEX IF NOT EXISTS idx_hooks_is_active ON hooks(is_active);
CREATE INDEX IF NOT EXISTS idx_hooks_execution_order ON hooks(execution_order);

-- Hook Executions
CREATE INDEX IF NOT EXISTS idx_hook_executions_hook_id ON hook_executions(hook_id);
CREATE INDEX IF NOT EXISTS idx_hook_executions_status ON hook_executions(status);
CREATE INDEX IF NOT EXISTS idx_hook_executions_created_at ON hook_executions(created_at DESC);

-- Performance Metrics
CREATE INDEX IF NOT EXISTS idx_performance_metrics_entity_type ON performance_metrics(entity_type);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_entity_id ON performance_metrics(entity_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_recorded_at ON performance_metrics(recorded_at DESC);

-- Agents
CREATE INDEX IF NOT EXISTS idx_agents_is_active ON agents(is_active);
CREATE INDEX IF NOT EXISTS idx_agents_success_rate ON agents(success_rate DESC);

-- Learning Insights
CREATE INDEX IF NOT EXISTS idx_learning_insights_insight_type ON learning_insights(insight_type);
CREATE INDEX IF NOT EXISTS idx_learning_insights_is_validated ON learning_insights(is_validated);
CREATE INDEX IF NOT EXISTS idx_learning_insights_confidence_score ON learning_insights(confidence_score DESC);

-- ============================================================================
-- INITIAL DATA - Schema Version
-- ============================================================================

INSERT OR IGNORE INTO schema_version (version, description)
VALUES ('2.1.0', 'Initial DevStream production schema with vector search and full-text search');

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================

-- Usage Notes:
-- 1. This schema requires sqlite-vec extension for vector search
-- 2. FTS5 extension is built-in to modern SQLite
-- 3. To load extensions in Python:
--    conn.enable_load_extension(True)
--    conn.load_extension("vec0")
-- 4. For vector search, embeddings must be 768-dimensional float arrays
-- 5. Hybrid search combines vec_semantic_memory and fts_semantic_memory using RRF
-- 6. All JSON fields should be valid JSON strings
-- 7. Triggers automatically sync semantic_memory changes to vector/FTS tables
