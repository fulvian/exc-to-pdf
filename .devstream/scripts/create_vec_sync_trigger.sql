-- DevStream Vector Synchronization Trigger
-- Context7 Best Practices Implementation
-- sqlite-vec v0.1.6 compliant

-- Drop existing triggers if they exist (for clean re-creation)
DROP TRIGGER IF EXISTS sync_insert_memory;
DROP TRIGGER IF EXISTS sync_update_memory;
DROP TRIGGER IF EXISTS sync_delete_memory;

-- Enhanced INSERT trigger with vector synchronization
CREATE TRIGGER sync_insert_memory
AFTER INSERT ON semantic_memory
WHEN NEW.embedding IS NOT NULL AND NEW.embedding != ''
BEGIN
    -- FTS5 synchronization (existing functionality)
    INSERT INTO fts_semantic_memory(rowid, content, content_type, memory_id, created_at)
    VALUES (NEW.rowid, NEW.content, NEW.content_type, NEW.id, NEW.created_at);

    -- Vector synchronization (NEW: sync to vec_semantic_memory)
    INSERT INTO vec_semantic_memory(
        embedding,
        content_type,
        memory_id,
        content_preview
    ) VALUES (
        NEW.embedding,
        NEW.content_type,
        NEW.id,
        substr(NEW.content, 1, 200)  -- Content preview for vector search results
    );
END;

-- Enhanced UPDATE trigger with vector synchronization
CREATE TRIGGER sync_update_memory
AFTER UPDATE ON semantic_memory
WHEN NEW.embedding IS NOT NULL AND NEW.embedding != ''
    AND (OLD.embedding IS NULL OR OLD.embedding = '' OR OLD.embedding != NEW.embedding)
BEGIN
    -- Clean up old entries
    DELETE FROM fts_semantic_memory WHERE rowid = OLD.rowid;
    DELETE FROM vec_semantic_memory WHERE memory_id = OLD.id;

    -- FTS5 synchronization (existing functionality)
    INSERT INTO fts_semantic_memory(rowid, content, content_type, memory_id, created_at)
    VALUES (NEW.rowid, NEW.content, NEW.content_type, NEW.id, NEW.created_at);

    -- Vector synchronization (NEW: sync to vec_semantic_memory)
    INSERT INTO vec_semantic_memory(
        embedding,
        content_type,
        memory_id,
        content_preview
    ) VALUES (
        NEW.embedding,
        NEW.content_type,
        NEW.id,
        substr(NEW.content, 1, 200)  -- Content preview for vector search results
    );
END;

-- Enhanced DELETE trigger with vector cleanup
CREATE TRIGGER sync_delete_memory
AFTER DELETE ON semantic_memory
BEGIN
    -- Clean up both indices
    DELETE FROM fts_semantic_memory WHERE rowid = OLD.rowid;
    DELETE FROM vec_semantic_memory WHERE memory_id = OLD.id;
END;

-- Additional trigger for embedding-only updates (Context7 pattern)
CREATE TRIGGER sync_embedding_update
AFTER UPDATE OF embedding ON semantic_memory
WHEN NEW.embedding IS NOT NULL AND NEW.embedding != ''
    AND (OLD.embedding IS NULL OR OLD.embedding = '' OR OLD.embedding != NEW.embedding)
BEGIN
    -- Update vector index when embedding field changes
    DELETE FROM vec_semantic_memory WHERE memory_id = NEW.id;

    INSERT INTO vec_semantic_memory(
        embedding,
        content_type,
        memory_id,
        content_preview
    ) VALUES (
        NEW.embedding,
        NEW.content_type,
        NEW.id,
        substr(NEW.content, 1, 200)
    );
END;