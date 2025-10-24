-- Migration 003: Add active_files column to work_sessions table
-- Purpose: Track files modified during session (Memory Bank activeContext pattern)
-- Context7 Pattern: /movibe/memory-bank-mcp activeContext.md
-- Date: 2025-10-03
-- Task: c5af739922abe80e5d6e755b2bc56f24

-- =============================================================================
-- FORWARD MIGRATION
-- =============================================================================

-- Add active_files JSON column to track files modified during session
ALTER TABLE work_sessions
ADD COLUMN active_files JSON DEFAULT '[]';

-- Verify column added
SELECT
    name,
    type,
    dflt_value,
    CASE WHEN notnull = 0 THEN 'nullable' ELSE 'not null' END as nullable
FROM pragma_table_info('work_sessions')
WHERE name = 'active_files';

-- =============================================================================
-- ROLLBACK (if needed)
-- =============================================================================

-- Uncomment to rollback:
-- ALTER TABLE work_sessions DROP COLUMN active_files;

-- =============================================================================
-- NOTES
-- =============================================================================

-- Column Purpose:
-- - Tracks files actively modified during session (not just created_at time-based)
-- - Enables Memory Bank pattern: session summary shows ACTUAL work done
-- - Fixes empty summary bug (files from hours ago excluded if not tracked)
--
-- Column Type: JSON
-- - Default: '[]' (empty array)
-- - Format: ["path/to/file1.py", "path/to/file2.ts"]
-- - Updated by: PostToolUse hook after Write/Edit/MultiEdit
--
-- Backward Compatibility:
-- - Existing sessions get DEFAULT '[]' automatically
-- - Sessions without tracking fall back to time-based queries
-- - No breaking changes to existing code
