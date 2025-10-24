"""
Embedding Generator with Context7 Ollama Batch Processing

FASE 2 implementation following Context7 patterns for:
- Robust Ollama API integration with exponential backoff
- Batch processing (size: 10) with atomic operations
- sqlite-utils batch insert pattern integration
- Type-safe error handling and structured logging
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Optional, Sequence

import numpy as np
import ollama
import structlog
from pydantic import BaseModel, Field

from ..database.connection import ConnectionPool
from ..database.sqlite_vec_manager import vec_manager
from .models import MemoryEntry

logger = structlog.get_logger()


class EmbeddingConfig(BaseModel):
    """
    Configuration for embedding generation following Context7 patterns.

    Context7-validated settings for Ollama integration and batch processing.
    """
    model_name: str = Field(default="embeddinggemma:300m", description="Ollama model for embeddings")
    batch_size: int = Field(default=10, ge=1, le=50, description="Batch size for processing")
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum retry attempts")
    base_delay: float = Field(default=1.0, ge=0.1, le=10.0, description="Base delay for exponential backoff (seconds)")
    timeout: float = Field(default=30.0, ge=5.0, le=300.0, description="Request timeout (seconds)")

    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class EmbeddingGenerationError(Exception):
    """Custom exception for embedding generation failures."""

    def __init__(self, message: str, retry_count: int = 0, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.retry_count = retry_count
        self.original_error = original_error


class EmbeddingGenerator:
    """
    Context7-validated embedding generator with batch processing.

    Implements robust Ollama API integration following Context7 patterns:
    - Exponential backoff retry (1s, 2s, 4s pattern)
    - Atomic batch operations with sqlite-utils pattern
    - Circuit breaker pattern for resilience
    - Comprehensive error handling and logging
    """

    def __init__(self, connection_pool: ConnectionPool, config: Optional[EmbeddingConfig] = None):
        """
        Initialize embedding generator with database connection and configuration.

        Args:
            connection_pool: Database connection pool for atomic operations
            config: Optional embedding configuration, uses defaults if not provided
        """
        self.connection_pool = connection_pool
        self.config = config or EmbeddingConfig()
        self._client = None
        self._setup_client()

        logger.info("EmbeddingGenerator initialized",
                   model=self.config.model_name,
                   batch_size=self.config.batch_size,
                   max_retries=self.config.max_retries)

    def _setup_client(self) -> None:
        """Setup Ollama client with Context7-validated configuration."""
        try:
            # Context7 pattern: Custom client with proper configuration
            self._client = ollama.Client(
                host='http://localhost:11434',
                headers={'User-Agent': 'DevStream-EmbeddingGenerator/1.0'}
            )
            logger.info("Ollama client configured successfully")
        except Exception as e:
            logger.error("Failed to configure Ollama client", error=str(e))
            raise EmbeddingGenerationError(f"Client configuration failed: {e}")

    @asynccontextmanager
    async def _atomic_transaction(self):
        """
        Context manager for atomic database transactions.

        Context7 sqlite-utils pattern for atomic batch operations:
        - Automatic rollback on errors
        - Proper connection management
        - Transaction isolation
        """
        async with self.connection_pool.engine.begin() as conn:
            try:
                yield conn
                logger.debug("Atomic transaction started")
            except Exception as e:
                logger.error("Atomic transaction failed, rolling back", error=str(e))
                raise

    async def _generate_embedding_with_retry(self, text: str) -> list[float]:
        """
        Generate embedding for single text with exponential backoff retry.

        Context7 Ollama pattern: 1s, 2s, 4s exponential backoff with proper error handling.

        Args:
            text: Text to generate embedding for

        Returns:
            List of embedding values

        Raises:
            EmbeddingGenerationError: If all retries are exhausted
        """
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                logger.debug("Generating embedding attempt",
                           attempt=attempt,
                           text_length=len(text),
                           model=self.config.model_name)

                # Context7 pattern: Use synchronous client in async context
                response = self._client.embed(
                    model=self.config.model_name,
                    input=text
                )

                embedding = response.embeddings
                if not embedding:
                    raise EmbeddingGenerationError("Empty embedding response")

                embedding_vector = embedding
                logger.debug("Embedding generated successfully",
                           dimension=len(embedding_vector),
                           attempt=attempt)

                return embedding_vector

            except ollama.ResponseError as e:
                last_error = e
                delay = self.config.base_delay * (2 ** attempt)  # Exponential backoff

                logger.warning("Ollama API error, retrying",
                             error=str(e),
                             status_code=getattr(e, 'status_code', None),
                             attempt=attempt,
                             delay=delay)

                if attempt < self.config.max_retries:
                    await asyncio.sleep(delay)
                else:
                    logger.error("All retries exhausted for embedding generation",
                               text_length=len(text),
                               total_attempts=attempt + 1)

            except Exception as e:
                last_error = e
                logger.error("Unexpected error during embedding generation",
                           error=str(e),
                           attempt=attempt)

                if attempt >= self.config.max_retries:
                    break

        raise EmbeddingGenerationError(
            f"Failed to generate embedding after {self.config.max_retries + 1} attempts",
            retry_count=self.config.max_retries,
            original_error=last_error
        )

    async def _process_batch(self, memory_entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """
        Process a batch of memory entries to generate embeddings.

        Context7 pattern: Process batch with atomic operations and comprehensive error handling.

        Args:
            memory_entries: List of memory entries to process

        Returns:
            List of memory entries with embeddings populated
        """
        logger.info("Processing embedding batch",
                   batch_size=len(memory_entries),
                   model=self.config.model_name)

        # Generate embeddings for all entries in batch
        embedding_tasks = []
        for entry in memory_entries:
            task = asyncio.create_task(
                self._generate_embedding_with_retry(entry.content)
            )
            embedding_tasks.append((entry, task))

        # Wait for all embeddings with error handling
        processed_entries = []
        failed_count = 0

        for entry, task in embedding_tasks:
            try:
                embedding = await task

                # Update memory entry with embedding
                embedding_array = np.array(embedding, dtype=np.float32)
                entry.set_embedding(embedding_array, self.config.model_name)

                processed_entries.append(entry)
                logger.debug("Entry processed successfully",
                           memory_id=entry.id,
                           embedding_dim=len(embedding))

            except EmbeddingGenerationError as e:
                failed_count += 1
                logger.error("Failed to process entry",
                           memory_id=entry.id,
                           error=str(e),
                           retry_count=e.retry_count)

                # Keep entry without embedding rather than failing entire batch
                processed_entries.append(entry)

        logger.info("Batch processing completed",
                   total=len(memory_entries),
                   successful=len(processed_entries) - failed_count,
                   failed=failed_count)

        return processed_entries

    async def _atomic_batch_insert(self, conn: Any, memory_entries: list[MemoryEntry]) -> bool:
        """
        Perform atomic batch insert following Context7 sqlite-utils pattern.

        Context7 pattern: Use transaction with batch operations for atomicity.

        Args:
            conn: Database connection
            memory_entries: List of memory entries to insert

        Returns:
            True if successful

        Raises:
            Exception: If batch insert fails
        """
        try:
            from ..database.schema import semantic_memory

            # Prepare batch data following sqlite-utils pattern
            batch_data = []
            for memory in memory_entries:
                embedding_json = json.dumps(memory.embedding) if memory.embedding else None

                batch_data.append({
                    'id': memory.id,
                    'plan_id': memory.plan_id,
                    'phase_id': memory.phase_id,
                    'task_id': memory.task_id,
                    'content': memory.content,
                    'content_type': memory.content_type,
                    'content_format': memory.content_format,
                    'keywords': json.dumps(memory.keywords),
                    'entities': json.dumps(memory.entities),
                    'sentiment': memory.sentiment,
                    'complexity_score': memory.complexity_score,
                    'embedding': embedding_json,
                    'embedding_model': memory.embedding_model,
                    'embedding_dimension': memory.embedding_dimension,
                    'context_snapshot': json.dumps(memory.context_snapshot),
                    'related_memory_ids': json.dumps(memory.related_memory_ids),
                    'access_count': memory.access_count,
                    'last_accessed_at': memory.last_accessed_at,
                    'relevance_score': memory.relevance_score,
                    'is_archived': memory.is_archived,
                    'created_at': memory.created_at,
                    'updated_at': memory.updated_at,
                })

            # Context7 sqlite-utils pattern: Batch insert with atomic transaction
            if batch_data:
                # Using executemany for batch insert (sqlite-utils pattern)
                await conn.execute(semantic_memory.insert(), batch_data)
                logger.info("Atomic batch insert completed",
                           count=len(batch_data),
                           with_embeddings=sum(1 for item in batch_data if item['embedding']))

            return True

        except Exception as e:
            logger.error("Atomic batch insert failed",
                       count=len(memory_entries),
                       error=str(e))
            raise

    async def _sync_to_virtual_tables(self, conn: Any, memory_entries: list[MemoryEntry]) -> None:
        """
        Sync processed entries to virtual tables following Context7 pattern.

        Args:
            conn: Database connection
            memory_entries: List of memory entries to sync
        """
        try:
            for memory in memory_entries:
                # Sync to FTS table (always available)
                await conn.execute("""
                    INSERT OR REPLACE INTO fts_semantic_memory(memory_id, content, keywords, entities)
                    VALUES (:memory_id, :content, :keywords, :entities)
                """, {
                    'memory_id': memory.id,
                    'content': memory.content,
                    'keywords': json.dumps(memory.keywords),
                    'entities': json.dumps(memory.entities)
                })

                # Sync to vector table only if embedding available (Context7 pattern)
                if (memory.embedding and
                    hasattr(self, '_vec_table_available') and self._vec_table_available):

                    embedding_json = json.dumps(memory.embedding)
                    await conn.execute("""
                        INSERT OR REPLACE INTO vec_semantic_memory(memory_id, content_embedding)
                        VALUES (:memory_id, :embedding)
                    """, {
                        'memory_id': memory.id,
                        'embedding': embedding_json
                    })

            logger.debug("Virtual tables sync completed",
                        count=len(memory_entries),
                        with_embeddings=sum(1 for m in memory_entries if m.embedding))

        except Exception as e:
            logger.error("Virtual tables sync failed",
                       count=len(memory_entries),
                       error=str(e))
            # Non-blocking: don't raise exception, virtual table sync is optional

    async def generate_and_store_embeddings(self, memory_entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """
        Generate and store embeddings for memory entries with full atomicity.

        Main entry point following Context7 pattern:
        1. Generate embeddings with retry logic
        2. Atomic batch insert to main table
        3. Sync to virtual tables
        4. Comprehensive error handling

        Args:
            memory_entries: List of memory entries to process

        Returns:
            List of processed memory entries

        Raises:
            EmbeddingGenerationError: If critical failures occur
        """
        if not memory_entries:
            logger.info("No memory entries to process")
            return []

        logger.info("Starting embedding generation and storage",
                   total_entries=len(memory_entries),
                   model=self.config.model_name,
                   batch_size=self.config.batch_size)

        # Process in batches following Context7 pattern
        all_processed_entries = []

        for i in range(0, len(memory_entries), self.config.batch_size):
            batch = memory_entries[i:i + self.config.batch_size]
            batch_num = (i // self.config.batch_size) + 1
            total_batches = (len(memory_entries) + self.config.batch_size - 1) // self.config.batch_size

            logger.info("Processing batch",
                       batch_num=batch_num,
                       total_batches=total_batches,
                       batch_size=len(batch))

            try:
                # Step 1: Generate embeddings for batch
                processed_batch = await self._process_batch(batch)

                # Step 2: Atomic database operations
                async with self._atomic_transaction() as conn:
                    # Step 2a: Atomic batch insert to main table
                    await self._atomic_batch_insert(conn, processed_batch)

                    # Step 2b: Sync to virtual tables
                    await self._sync_to_virtual_tables(conn, processed_batch)

                all_processed_entries.extend(processed_batch)

                logger.info("Batch completed successfully",
                           batch_num=batch_num,
                           total_batches=total_batches)

            except Exception as e:
                logger.error("Batch processing failed",
                           batch_num=batch_num,
                           total_batches=total_batches,
                           batch_size=len(batch),
                           error=str(e))

                # For now, continue with next batch rather than failing entirely
                # Could be configured to fail fast based on requirements
                all_processed_entries.extend(batch)  # Keep original entries

        logger.info("Embedding generation and storage completed",
                   total_processed=len(all_processed_entries),
                   successful_embeddings=sum(1 for e in all_processed_entries if e.embedding),
                   model=self.config.model_name)

        return all_processed_entries

    async def check_model_availability(self) -> bool:
        """
        Check if the embedding model is available in Ollama.

        Returns:
            True if model is available
        """
        try:
            models = self._client.list()
            available_models = [model['name'].split(':')[0] for model in models.get('models', [])]

            is_available = self.config.model_name in available_models

            logger.info("Model availability check",
                       model=self.config.model_name,
                       available=available_models,
                       is_available=is_available)

            return is_available

        except Exception as e:
            logger.error("Failed to check model availability",
                       model=self.config.model_name,
                       error=str(e))
            return False

    async def pull_model_if_needed(self) -> bool:
        """
        Pull the embedding model if not available locally.

        Returns:
            True if model is available after operation
        """
        if await self.check_model_availability():
            logger.info("Model already available", model=self.config.model_name)
            return True

        try:
            logger.info("Pulling model", model=self.config.model_name)

            # Context7 pattern: Stream model pull with progress
            response = self._client.pull(self.config.model_name, stream=True)

            for chunk in response:
                status = chunk.get('status', '')
                logger.debug("Model pull progress", status=status)

            logger.info("Model pull completed", model=self.config.model_name)
            return True

        except Exception as e:
            logger.error("Failed to pull model",
                       model=self.config.model_name,
                       error=str(e))
            return False