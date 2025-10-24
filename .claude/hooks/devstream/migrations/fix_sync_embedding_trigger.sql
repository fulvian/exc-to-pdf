-- Fix sync_embedding_update trigger to prevent duplicates
-- Root Cause: INSERT OR REPLACE doesn't work on vec0 without unique constraint
-- Solution: DELETE + INSERT pattern (proper UPSERT)

-- Drop old broken trigger
DROP TRIGGER IF EXISTS sync_embedding_update;

-- Create fixed trigger with DELETE + INSERT pattern
CREATE TRIGGER sync_embedding_update
AFTER UPDATE OF embedding ON semantic_memory
WHEN NEW.embedding IS NOT NULL AND NEW.embedding != ''
BEGIN
    -- Step 1: DELETE existing BLOB for this memory_id (prevents duplicates)
    DELETE FROM vec_semantic_memory WHERE memory_id = NEW.id;

    -- Step 2: INSERT new BLOB (JSON → BLOB conversion)
    INSERT INTO vec_semantic_memory(
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
