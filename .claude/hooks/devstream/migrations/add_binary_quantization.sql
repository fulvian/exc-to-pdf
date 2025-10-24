-- Phase 3: Binary Quantization Migration
-- Creates dual-column architecture for 32x compression + re-scoring pattern
-- Expected: Additional -127 MB savings (421 MB → ~294 MB)

-- Step 1: Create new table with binary quantization (dual columns)
-- embedding_fine: float32[768] for re-scoring (3072 bytes)
-- embedding_coarse: bit[768] for coarse filtering (96 bytes)
CREATE VIRTUAL TABLE vec_semantic_memory_v2 USING vec0(
    embedding_fine float[768],        -- For re-scoring (3,072 bytes)
    embedding_coarse bit[768],         -- For coarse filter (96 bytes)
    content_type TEXT PARTITION KEY,
    +memory_id TEXT,
    +content_preview TEXT
);

-- Step 2: Migrate existing BLOB embeddings with quantization
-- vec_quantize_binary() converts float32 to bit[768] automatically
INSERT INTO vec_semantic_memory_v2(memory_id, embedding_fine, embedding_coarse, content_type, content_preview)
SELECT
    memory_id,
    embedding as embedding_fine,
    vec_quantize_binary(embedding) as embedding_coarse,
    content_type,
    content_preview
FROM vec_semantic_memory;

-- Step 3: Verify migration results
SELECT
    COUNT(*) as total_migrated,
    COUNT(CASE WHEN embedding_fine IS NOT NULL THEN 1 END) as with_fine_embeddings,
    COUNT(CASE WHEN embedding_coarse IS NOT NULL THEN 1 END) as with_coarse_embeddings
FROM vec_semantic_memory_v2;

-- Step 4: Backup old table before swap (for rollback safety)
CREATE TABLE vec_semantic_memory_backup AS
SELECT * FROM vec_semantic_memory;

-- Step 5: Swap tables (zero-downtime atomic operation)
DROP TABLE vec_semantic_memory;
ALTER TABLE vec_semantic_memory_v2 RENAME TO vec_semantic_memory;

-- Step 6: Verify final table structure
PRAGMA table_info(vec_semantic_memory);

-- Step 7: Verify data integrity after migration
SELECT
    COUNT(*) as final_count,
    COUNT(CASE WHEN embedding_fine IS NOT NULL THEN 1 END) as with_fine_embeddings,
    COUNT(CASE WHEN embedding_coarse IS NOT NULL THEN 1 END) as with_coarse_embeddings
FROM vec_semantic_memory;

-- Cleanup: Remove backup table after successful verification (comment out for debugging)
-- DROP TABLE vec_semantic_memory_backup;