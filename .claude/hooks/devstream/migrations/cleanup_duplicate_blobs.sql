-- Cleanup duplicate BLOB embeddings
-- Strategy: Keep only the LATEST BLOB (highest rowid) for each memory_id

-- Backup count before cleanup
SELECT 'Before cleanup:' as status, COUNT(*) as total_blobs
FROM vec_semantic_memory;

SELECT 'Unique memory_ids:' as status, COUNT(DISTINCT memory_id) as unique_ids
FROM vec_semantic_memory;

-- Delete duplicates (keep only MAX rowid for each memory_id)
DELETE FROM vec_semantic_memory
WHERE rowid NOT IN (
    SELECT MAX(rowid)
    FROM vec_semantic_memory
    GROUP BY memory_id
);

-- Verify cleanup
SELECT 'After cleanup:' as status, COUNT(*) as total_blobs
FROM vec_semantic_memory;

SELECT 'Should match:' as status, COUNT(DISTINCT memory_id) as unique_ids
FROM vec_semantic_memory;
