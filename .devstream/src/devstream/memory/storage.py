"""
Memory Storage Layer con sqlite-vec Integration

Async storage layer che integra SQLAlchemy con sqlite-vec
per vector search nativo e FTS5 per keyword search.
"""

import json
import logging
from typing import Any, Optional

import numpy as np
from sqlalchemy import MetaData, Table, Column, String, Text, Float, Integer, Boolean, TIMESTAMP, JSON
from sqlalchemy import select, insert, update, delete, text
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.sql import func

from ..database.connection import ConnectionPool
from ..database.sqlite_vec_manager import vec_manager
from .models import MemoryEntry, ContentType, ContentFormat
from .exceptions import StorageError, VectorSearchError
from .embedding_generator import EmbeddingGenerator, EmbeddingConfig

logger = logging.getLogger(__name__)


class MemoryStorage:
    """
    Async storage layer per memory entries con vector search.

    Integra SQLAlchemy 2.0 async con sqlite-vec per supportare
    sia storage relazionale che vector search operations.
    """

    def __init__(self, connection_pool: ConnectionPool, embedding_config: Optional[EmbeddingConfig] = None):
        """
        Initialize storage con database manager e embedding generator.

        Args:
            connection_pool: Database connection pool instance
            embedding_config: Optional configuration for embedding generation
        """
        self.connection_pool = connection_pool
        self.metadata = MetaData()
        self._init_tables()

        # FASE 2: Initialize embedding generator with Context7 patterns
        self.embedding_generator = EmbeddingGenerator(connection_pool, embedding_config)

    def _init_tables(self) -> None:
        """Initialize memory-specific tables and virtual tables."""
        # Main semantic_memory table già definita in schema.py
        # Virtual tables vengono create via raw SQL in create_virtual_tables()
        # Non possiamo definirle qui con SQLAlchemy Table perché sqlite-vec
        # richiede sintassi specifica non supportata da sqlite_with

        # Le virtual tables sono:
        # - vec_semantic_memory: per vector search usando vec0
        # - fts_semantic_memory: per keyword search usando FTS5
        pass

    async def create_virtual_tables(self) -> None:
        """
        Create virtual tables per vector e FTS search.
        Context7-validated pattern con sqlite-vec manager.

        Da chiamare dopo la creazione del database schema principale.
        """
        try:
            async with self.connection_pool.engine.begin() as conn:
                # Get raw connection per Context7 pattern
                raw_conn = await conn.get_raw_connection()

                # Load sqlite-vec extension using validated pattern
                if not vec_manager.load_extension(raw_conn):
                    logger.warning("sqlite-vec not available - vector search disabled")

                # Create vector search table using manager
                vec_created = vec_manager.create_vec_table(
                    raw_conn,
                    "vec_semantic_memory",
                    "content_embedding",
                    384
                )

                # Create FTS5 table using manager
                fts_created = vec_manager.create_fts_table(
                    raw_conn,
                    "fts_semantic_memory",
                    ["content", "keywords", "entities"]
                )

                # Context7 Pattern: Manual sync instead of triggers
                # sqlite-vec requires BLOB/JSON embeddings, not NULL
                # Virtual tables are populated manually when embeddings are available
                self._vec_table_available = vec_created
                self._fts_table_available = fts_created

                logger.info("Virtual tables setup completed",
                          vec_available=vec_created, fts_available=fts_created)

        except Exception as e:
            logger.error(f"Failed to create virtual tables: {e}")
            raise StorageError(f"Virtual table creation failed: {e}") from e

    async def sync_to_virtual_tables(self, memory: MemoryEntry) -> None:
        """
        Manual sync to virtual tables following Context7 pattern.

        Only syncs to vector table if embedding is available.
        Always syncs to FTS table for text search.
        """
        try:
            async with self.connection_pool.engine.begin() as conn:
                # Sync to FTS table (always available)
                if hasattr(self, '_fts_table_available') and self._fts_table_available:
                    await conn.execute(text("""
                        INSERT OR REPLACE INTO fts_semantic_memory(memory_id, content, keywords, entities)
                        VALUES (:memory_id, :content, :keywords, :entities)
                    """), {
                        'memory_id': memory.id,
                        'content': memory.content,
                        'keywords': json.dumps(memory.keywords),
                        'entities': json.dumps(memory.entities)
                    })

                # Sync to vector table only if embedding available (Context7 pattern)
                if (memory.embedding and
                    hasattr(self, '_vec_table_available') and self._vec_table_available):

                    # Convert embedding to binary format for sqlite-vec
                    import struct
                    embedding_array = np.array(memory.embedding, dtype=np.float32)
                    embedding_binary = embedding_array.tobytes()

                    await conn.execute(text("""
                        INSERT OR REPLACE INTO vec_semantic_memory(memory_id, embedding)
                        VALUES (:memory_id, :embedding)
                    """), {
                        'memory_id': memory.id,
                        'embedding': embedding_binary
                    })

                    logger.info(f"Synced memory to virtual tables", memory_id=memory.id, has_embedding=True)
                else:
                    logger.info(f"Synced memory to FTS only (no embedding)", memory_id=memory.id, has_embedding=False)

        except Exception as e:
            logger.error(f"Failed to sync memory {memory.id} to virtual tables: {e}")
            # Non-blocking: don't raise exception, virtual table sync is optional

    async def _cleanup_virtual_tables(self, memory_id: str) -> None:
        """
        Remove memory from virtual tables following Context7 pattern.
        """
        try:
            async with self.connection_pool.engine.begin() as conn:
                # Cleanup vector table
                if hasattr(self, '_vec_table_available') and self._vec_table_available:
                    await conn.execute(text("""
                        DELETE FROM vec_semantic_memory WHERE memory_id = :memory_id
                    """), {'memory_id': memory_id})

                # Cleanup FTS table
                if hasattr(self, '_fts_table_available') and self._fts_table_available:
                    await conn.execute(text("""
                        DELETE FROM fts_semantic_memory WHERE memory_id = :memory_id
                    """), {'memory_id': memory_id})

                logger.info(f"Cleaned up virtual tables for memory: {memory_id}")

        except Exception as e:
            logger.error(f"Failed to cleanup virtual tables for memory {memory_id}: {e}")
            # Non-blocking: don't raise exception

    async def store_memory(self, memory: MemoryEntry) -> str:
        """
        Store memory entry con embedding.

        Args:
            memory: Memory entry da salvare

        Returns:
            ID del memory entry salvato

        Raises:
            StorageError: Se il salvataggio fallisce
        """
        try:
            # Convert embedding to JSON string per SQLite storage
            embedding_json = None
            if memory.embedding:
                embedding_json = json.dumps(memory.embedding)

            async with self.connection_pool.engine.begin() as conn:
                from ..database.schema import semantic_memory

                stmt = insert(semantic_memory).values(
                    id=memory.id,
                    plan_id=memory.plan_id,
                    phase_id=memory.phase_id,
                    task_id=memory.task_id,
                    content=memory.content,
                    content_type=memory.content_type,
                    content_format=memory.content_format,
                    keywords=json.dumps(memory.keywords),
                    entities=json.dumps(memory.entities),
                    sentiment=memory.sentiment,
                    complexity_score=memory.complexity_score,
                    embedding=embedding_json,
                    embedding_model=memory.embedding_model,
                    embedding_dimension=memory.embedding_dimension,
                    context_snapshot=json.dumps(memory.context_snapshot),
                    related_memory_ids=json.dumps(memory.related_memory_ids),
                    access_count=memory.access_count,
                    last_accessed_at=memory.last_accessed_at,
                    relevance_score=memory.relevance_score,
                    is_archived=memory.is_archived,
                    created_at=memory.created_at,
                    updated_at=memory.updated_at,
                )

                await conn.execute(stmt)
                logger.info(f"Stored memory entry: {memory.id}")

            # Context7 Pattern: Manual sync to virtual tables
            await self.sync_to_virtual_tables(memory)
            return memory.id

        except Exception as e:
            logger.error(f"Failed to store memory {memory.id}: {e}")
            raise StorageError(f"Memory storage failed: {e}") from e

    async def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve memory entry by ID.

        Args:
            memory_id: ID del memory entry

        Returns:
            Memory entry o None se non trovato

        Raises:
            StorageError: Se il retrieval fallisce
        """
        try:
            async with self.connection_pool.engine.connect() as conn:
                from ..database.schema import semantic_memory

                stmt = select(semantic_memory).where(semantic_memory.c.id == memory_id)
                result = await conn.execute(stmt)
                row = result.fetchone()

                if row is None:
                    return None

                return self._row_to_memory_entry(row)

        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            raise StorageError(f"Memory retrieval failed: {e}") from e

    async def update_memory(self, memory: MemoryEntry) -> bool:
        """
        Update existing memory entry.

        Args:
            memory: Updated memory entry

        Returns:
            True se l'update è riuscito

        Raises:
            StorageError: Se l'update fallisce
        """
        try:
            embedding_json = None
            if memory.embedding:
                embedding_json = json.dumps(memory.embedding)

            async with self.connection_pool.engine.begin() as conn:
                from ..database.schema import semantic_memory

                stmt = (
                    update(semantic_memory)
                    .where(semantic_memory.c.id == memory.id)
                    .values(
                        content=memory.content,
                        content_type=memory.content_type,
                        content_format=memory.content_format,
                        keywords=json.dumps(memory.keywords),
                        entities=json.dumps(memory.entities),
                        sentiment=memory.sentiment,
                        complexity_score=memory.complexity_score,
                        embedding=embedding_json,
                        embedding_model=memory.embedding_model,
                        embedding_dimension=memory.embedding_dimension,
                        context_snapshot=json.dumps(memory.context_snapshot),
                        related_memory_ids=json.dumps(memory.related_memory_ids),
                        access_count=memory.access_count,
                        last_accessed_at=memory.last_accessed_at,
                        relevance_score=memory.relevance_score,
                        is_archived=memory.is_archived,
                        updated_at=func.current_timestamp(),
                    )
                )

                result = await conn.execute(stmt)
                success = result.rowcount > 0

                if success:
                    logger.info(f"Updated memory entry: {memory.id}")
                    # Context7 Pattern: Manual sync to virtual tables after update
                    await self.sync_to_virtual_tables(memory)
                else:
                    logger.warning(f"Memory entry not found for update: {memory.id}")

                return success

        except Exception as e:
            logger.error(f"Failed to update memory {memory.id}: {e}")
            raise StorageError(f"Memory update failed: {e}") from e

    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete memory entry.

        Args:
            memory_id: ID del memory entry da eliminare

        Returns:
            True se la deletion è riuscita

        Raises:
            StorageError: Se la deletion fallisce
        """
        try:
            async with self.connection_pool.engine.begin() as conn:
                from ..database.schema import semantic_memory

                stmt = delete(semantic_memory).where(semantic_memory.c.id == memory_id)
                result = await conn.execute(stmt)
                success = result.rowcount > 0

                if success:
                    logger.info(f"Deleted memory entry: {memory_id}")
                    # Context7 Pattern: Manual cleanup from virtual tables
                    await self._cleanup_virtual_tables(memory_id)
                else:
                    logger.warning(f"Memory entry not found for deletion: {memory_id}")

                return success

        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            raise StorageError(f"Memory deletion failed: {e}") from e

    async def search_vectors(self, query_embedding: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        """
        Search memory entries by vector similarity.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of (memory_id, distance) tuples

        Raises:
            VectorSearchError: Se la search fallisce
        """
        try:
            # Convert numpy array to binary format expected by sqlite-vec
            query_array = np.array(query_embedding, dtype=np.float32)
            query_vector = query_array.tobytes()

            async with self.connection_pool.engine.connect() as conn:
                # CRITICAL: Load sqlite-vec extension before vector search
                # Each connection from pool needs extension loaded separately
                raw_conn = await conn.get_raw_connection()
                if not vec_manager.load_extension(raw_conn):
                    logger.warning("sqlite-vec extension not available - vector search may fail")

                result = await conn.execute(
                    text("""
                    SELECT memory_id, distance
                    FROM vec_semantic_memory
                    WHERE embedding MATCH :query_vector
                    ORDER BY distance
                    LIMIT :k
                    """),
                    {"query_vector": query_vector, "k": k}
                )

                return [(row[0], row[1]) for row in result.fetchall()]

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise VectorSearchError(f"Vector search failed: {e}") from e

    async def search_fts(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """
        Search memory entries using FTS5.

        Args:
            query: FTS query string
            k: Number of results to return

        Returns:
            List of (memory_id, rank) tuples

        Raises:
            StorageError: Se la search fallisce
        """
        try:
            async with self.connection_pool.engine.connect() as conn:
                # Load sqlite-vec extension for consistency (includes FTS5)
                # Each connection from pool needs extension loaded separately
                raw_conn = await conn.get_raw_connection()
                if not vec_manager.load_extension(raw_conn):
                    logger.warning("sqlite-vec extension not available - FTS search may fail")

                result = await conn.execute(
                    text("""
                    SELECT memory_id, rank
                    FROM fts_semantic_memory
                    WHERE fts_semantic_memory MATCH :query
                    ORDER BY rank
                    LIMIT :k
                    """),
                    {"query": query, "k": k}
                )

                return [(row[0], row[1]) for row in result.fetchall()]

        except Exception as e:
            logger.error(f"FTS search failed: {e}")
            raise StorageError(f"FTS search failed: {e}") from e

    def _safe_json_loads(self, value):
        """
        Safely load JSON data, handling cases where data is already deserialized.

        Args:
            value: Value to load, can be string, list, dict, or None

        Returns:
            Deserialized Python object or empty default
        """
        if value is None:
            return None
        elif isinstance(value, (list, dict)):
            # Data is already deserialized
            return value
        elif isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                # Return empty default if JSON parsing fails
                return []
        else:
            # Return empty default for unexpected types
            return []

    def _row_to_memory_entry(self, row) -> MemoryEntry:
        """Convert database row to MemoryEntry model."""
        return MemoryEntry(
            id=row.id,
            plan_id=row.plan_id,
            phase_id=row.phase_id,
            task_id=row.task_id,
            content=row.content,
            content_type=ContentType(row.content_type),
            content_format=ContentFormat(row.content_format) if row.content_format is not None else ContentFormat.TEXT,
            keywords=self._safe_json_loads(row.keywords) or [],
            entities=self._safe_json_loads(row.entities) or [],
            sentiment=row.sentiment if row.sentiment is not None else 0.0,
            complexity_score=row.complexity_score if row.complexity_score is not None else 1,
            embedding=self._safe_json_loads(row.embedding),
            embedding_model=row.embedding_model,
            embedding_dimension=row.embedding_dimension,
            context_snapshot=self._safe_json_loads(row.context_snapshot) or {},
            related_memory_ids=self._safe_json_loads(row.related_memory_ids) or [],
            access_count=row.access_count if row.access_count is not None else 0,
            last_accessed_at=row.last_accessed_at,
            relevance_score=row.relevance_score if row.relevance_score is not None else 1.0,
            is_archived=row.is_archived if row.is_archived is not None else False,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    async def store_memories_with_embeddings(self, memory_entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """
        Store multiple memory entries with automatic embedding generation.

        FASE 2: Context7 pattern for batch embedding generation and atomic storage.
        Uses the EmbeddingGenerator for robust Ollama integration with retry logic
        and atomic batch operations.

        Args:
            memory_entries: List of memory entries to store with embeddings

        Returns:
            List of stored memory entries with embeddings populated

        Raises:
            StorageError: If storage operation fails
            EmbeddingGenerationError: If embedding generation fails critically
        """
        if not memory_entries:
            logger.info("No memory entries provided for storage with embeddings")
            return []

        logger.info("Starting batch storage with embedding generation",
                   count=len(memory_entries),
                   model=self.embedding_generator.config.model_name)

        try:
            # Step 1: Ensure model is available
            if not await self.embedding_generator.pull_model_if_needed():
                logger.warning("Embedding model not available, proceeding without embeddings")
                # Store without embeddings rather than failing entirely
                return [await self.store_memory(entry) for entry in memory_entries]

            # Step 2: Generate embeddings and store atomically
            processed_entries = await self.embedding_generator.generate_and_store_embeddings(memory_entries)

            logger.info("Batch storage with embeddings completed",
                       total_stored=len(processed_entries),
                       with_embeddings=sum(1 for e in processed_entries if e.embedding))

            return processed_entries

        except Exception as e:
            logger.error("Failed to store memories with embeddings",
                       count=len(memory_entries),
                       error=str(e))
            raise StorageError(f"Batch storage with embeddings failed: {e}") from e

    async def update_memory_embeddings(self, memory_ids: list[str]) -> list[MemoryEntry]:
        """
        Generate and update embeddings for existing memory entries.

        FASE 2: Batch embedding generation for existing entries without embeddings.

        Args:
            memory_ids: List of memory IDs to update with embeddings

        Returns:
            List of updated memory entries with embeddings

        Raises:
            StorageError: If update operation fails
        """
        if not memory_ids:
            logger.info("No memory IDs provided for embedding update")
            return []

        logger.info("Starting embedding update for existing memories",
                   count=len(memory_ids))

        try:
            # Step 1: Retrieve existing memory entries
            existing_entries = []
            for memory_id in memory_ids:
                entry = await self.get_memory(memory_id)
                if entry:
                    existing_entries.append(entry)
                else:
                    logger.warning("Memory entry not found for embedding update", memory_id=memory_id)

            if not existing_entries:
                logger.info("No existing memory entries found for embedding update")
                return []

            # Step 2: Generate embeddings and update
            updated_entries = await self.embedding_generator.generate_and_store_embeddings(existing_entries)

            logger.info("Embedding update completed",
                   total_requested=len(memory_ids),
                   found_entries=len(existing_entries),
                   updated_entries=len(updated_entries),
                   with_embeddings=sum(1 for e in updated_entries if e.embedding))

            return updated_entries

        except Exception as e:
            logger.error("Failed to update memory embeddings",
                       memory_ids=memory_ids,
                       error=str(e))
            raise StorageError(f"Memory embedding update failed: {e}") from e

    async def get_embedding_generator_status(self) -> dict[str, Any]:
        """
        Get status information about the embedding generator.

        Returns:
            Dictionary with embedding generator status information
        """
        try:
            model_available = await self.embedding_generator.check_model_availability()

            return {
                "model_name": self.embedding_generator.config.model_name,
                "model_available": model_available,
                "batch_size": self.embedding_generator.config.batch_size,
                "max_retries": self.embedding_generator.config.max_retries,
                "base_delay": self.embedding_generator.config.base_delay,
                "timeout": self.embedding_generator.config.timeout,
            }

        except Exception as e:
            logger.error("Failed to get embedding generator status", error=str(e))
            return {
                "error": str(e),
                "model_name": getattr(self.embedding_generator.config, 'model_name', 'unknown'),
                "model_available": False,
            }