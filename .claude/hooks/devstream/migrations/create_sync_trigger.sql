-- Phase 1: Create sync_embedding_update trigger
-- Purpose: Convert JSON embeddings to BLOB format and sync to vec_semantic_memory
-- Trigger automatically cleans up JSON to prevent duplication

-- Drop existing trigger if it exists
DROP TRIGGER IF EXISTS sync_embedding_update;

-- Create trigger for automatic JSON→BLOB sync
CREATE TRIGGER sync_embedding_update
AFTER UPDATE OF embedding ON semantic_memory
WHEN NEW.embedding IS NOT NULL AND NEW.embedding != ''
BEGIN
    -- Step 1: Convert JSON array → BLOB float32 using vec_f32()
    -- Step 2: Insert/Replace in vec_semantic_memory table
    INSERT OR REPLACE INTO vec_semantic_memory(
        embedding,
        content_type,
        memory_id,
        content_preview
    )
    VALUES (
        vec_f32(NEW.embedding),           -- JSON → BLOB conversion
        NEW.content_type,
        NEW.id,
        substr(NEW.content, 1, 200)       -- First 200 chars preview
    );

    -- Step 3: Cleanup JSON to prevent future duplication
    UPDATE semantic_memory
    SET embedding = NULL
    WHERE id = NEW.id;
END;

-- Verification query to check trigger creation
SELECT name FROM sqlite_master WHERE type='trigger' AND name='sync_embedding_update';