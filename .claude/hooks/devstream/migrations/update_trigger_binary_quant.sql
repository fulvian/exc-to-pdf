-- Phase 3: Update Trigger for Binary Quantization Support
-- Updates sync_embedding_update trigger to work with dual-column architecture

-- Drop existing trigger
DROP TRIGGER IF EXISTS sync_embedding_update;

-- Create updated trigger with binary quantization support
CREATE TRIGGER sync_embedding_update
AFTER UPDATE OF embedding ON semantic_memory
WHEN NEW.embedding IS NOT NULL AND NEW.embedding != ''
BEGIN
    -- Convert JSON → BLOB and quantize to binary in one operation
    INSERT OR REPLACE INTO vec_semantic_memory(
        embedding_fine,
        embedding_coarse,         -- Auto-quantize using vec_quantize_binary()
        content_type,
        memory_id,
        content_preview
    )
    VALUES (
        vec_f32(NEW.embedding),                                   -- JSON → float32 BLOB
        vec_quantize_binary(vec_f32(NEW.embedding)),             -- Auto-quantize to bit[768]
        NEW.content_type,
        NEW.id,
        substr(NEW.content, 1, 200)
    );

    -- Cleanup JSON to prevent future duplication
    UPDATE semantic_memory SET embedding = NULL WHERE id = NEW.id;
END;

-- Verify trigger creation
SELECT name FROM sqlite_master WHERE type='trigger' AND name='sync_embedding_update';

-- Test trigger with sample data (optional - comment out for production)
/*
-- Test the updated trigger
UPDATE semantic_memory
SET embedding = '[0.1, 0.2, 0.3]'
WHERE id = (SELECT id FROM semantic_memory LIMIT 1);

-- Verify both embeddings were created
SELECT
    memory_id,
    length(embedding_fine) as fine_bytes,
    length(embedding_coarse) as coarse_bytes
FROM vec_semantic_memory
WHERE memory_id = (SELECT id FROM semantic_memory LIMIT 1);
*/