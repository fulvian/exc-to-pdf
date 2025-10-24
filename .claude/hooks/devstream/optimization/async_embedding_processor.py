#!/usr/bin/env .devstream/bin/python
"""
DevStream Async Embedding Batch Processor

Async batch processing system to increase embedding coverage from 0.4% to 80%+
through efficient batch processing with exponential backoff retry strategy.

Uses Context7 research-backed patterns for async batch processing with
semaphore-based concurrency control and intelligent error recovery.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import structlog

logger = structlog.get_logger()


class EmbeddingProcessingError(Exception):
    """Exception raised when embedding batch processing fails completely."""
    pass


class BatchStatus(str, Enum):
    """Status of embedding batch processing."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"


@dataclass
class BatchResult:
    """
    Result of embedding batch processing operation.

    Contains detailed information about batch processing results
    for analysis and optimization.
    """
    batch_id: str
    status: BatchStatus
    memory_ids: List[str]
    success_count: int
    failure_count: int
    processing_time_ms: float
    retry_count: int
    error: Optional[str] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        total = self.success_count + self.failure_count
        return (self.success_count / total * 100.0) if total > 0 else 0.0


class AsyncEmbeddingBatchProcessor:
    """
    Async batch processor for embedding generation with retry logic.

    Implements exponential backoff retry strategy and queue-based processing
    to improve embedding coverage from 0.4% to 80%+ while handling Ollama
    rate limiting and temporary failures gracefully.

    Features:
    - Exponential backoff retry with jitter
    - Semaphore-based concurrency control
    - Queue-based batch processing
    - Rate limiting awareness
    - Comprehensive error handling
    - Performance monitoring
    """

    def __init__(
        self,
        batch_size: int = 20,
        max_retries: int = 3,
        max_concurrent_batches: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter_factor: float = 0.1
    ):
        """
        Initialize async embedding batch processor.

        Args:
            batch_size: Number of items per batch (optimized for Ollama gemma3)
            max_retries: Maximum retry attempts per batch
            max_concurrent_batches: Maximum concurrent batch processing
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay for exponential backoff (seconds)
            jitter_factor: Jitter factor to prevent thundering herd (0.0-1.0)
        """
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.max_concurrent_batches = max_concurrent_batches
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter_factor = jitter_factor

        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent_batches)

        # Ollama client - will be initialized lazily
        self._ollama_client: Optional[Any] = None

        # Performance metrics
        self._total_processed: int = 0
        self._total_successful: int = 0
        self._total_failed: int = 0
        self._total_time_ms: float = 0.0
        self._batch_times: List[float] = []
        self._retry_count: int = 0

        logger.info(
            "AsyncEmbeddingBatchProcessor initialized",
            batch_size=batch_size,
            max_retries=max_retries,
            max_concurrent_batches=max_concurrent_batches,
            base_delay=base_delay,
            max_delay=max_delay,
            jitter_factor=jitter_factor
        )

    def _get_ollama_client(self) -> Any:
        """Get or create Ollama client instance."""
        if self._ollama_client is None:
            try:
                # Import here to avoid circular imports
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
                from ollama_client import OllamaEmbeddingClient

                self._ollama_client = OllamaEmbeddingClient()
                logger.debug("OllamaEmbeddingClient initialized")
            except Exception as e:
                logger.error("Failed to initialize OllamaEmbeddingClient", error=str(e))
                raise EmbeddingProcessingError(f"Cannot initialize Ollama client: {e}") from e

        return self._ollama_client

    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay with jitter.

        Args:
            attempt: Current retry attempt (0-based)

        Returns:
            Delay in seconds
        """
        # Exponential backoff: delay = base_delay * (2 ^ attempt)
        exponential_delay = self.base_delay * (2 ** attempt)

        # Cap at maximum delay
        capped_delay = min(exponential_delay, self.max_delay)

        # Add jitter to prevent thundering herd
        jitter_range = capped_delay * self.jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)

        final_delay = max(0, capped_delay + jitter)

        logger.debug(
            "Calculated retry delay",
            attempt=attempt,
            exponential_delay=exponential_delay,
            jitter=jitter,
            final_delay=final_delay
        )

        return float(final_delay)

    def _split_into_batches(self, memory_ids: List[str], contents: List[str]) -> List[Tuple[List[str], List[str]]]:
        """
        Split memory IDs and contents into batches.

        Args:
            memory_ids: List of memory record IDs
            contents: List of content strings

        Returns:
            List of tuples (batch_memory_ids, batch_contents)
        """
        if len(memory_ids) != len(contents):
            raise ValueError("Memory IDs and contents must have the same length")

        batches = []
        for i in range(0, len(memory_ids), self.batch_size):
            batch_ids = memory_ids[i:i + self.batch_size]
            batch_contents = contents[i:i + self.batch_size]
            batches.append((batch_ids, batch_contents))

        logger.debug(
            "Split into batches",
            total_items=len(memory_ids),
            batch_size=self.batch_size,
            num_batches=len(batches)
        )

        return batches

    async def _process_single_batch(
        self,
        memory_ids: List[str],
        contents: List[str],
        batch_id: str
    ) -> BatchResult:
        """
        Process a single batch of memory items for embedding generation.

        Args:
            memory_ids: List of memory record IDs in this batch
            contents: List of content strings for this batch
            batch_id: Unique identifier for this batch

        Returns:
            BatchResult with processing details
        """
        start_time = time.time()
        retry_count = 0
        last_error = None

        while retry_count <= self.max_retries:
            try:
                # Acquire semaphore for concurrency control
                async with self._semaphore:
                    logger.debug(
                        "Processing batch",
                        batch_id=batch_id,
                        items_count=len(memory_ids),
                        attempt=retry_count + 1
                    )

                    # Get Ollama client
                    ollama_client = self._get_ollama_client()

                    # Process embeddings for this batch
                    success_count = 0
                    failure_count = 0
                    results = {}

                    for memory_id, content in zip(memory_ids, contents):
                        try:
                            # Generate embedding
                            embedding = ollama_client.generate_embedding(content)

                            if embedding:
                                # Store embedding in database
                                success = await self._store_embedding(memory_id, embedding)
                                if success:
                                    success_count += 1
                                    results[memory_id] = True
                                else:
                                    failure_count += 1
                                    results[memory_id] = False
                                    logger.warning(
                                        "Failed to store embedding",
                                        memory_id=memory_id[:8],
                                        batch_id=batch_id
                                    )
                            else:
                                failure_count += 1
                                results[memory_id] = False
                                logger.warning(
                                    "Empty embedding generated",
                                    memory_id=memory_id[:8],
                                    batch_id=batch_id
                                )

                        except Exception as item_error:
                            failure_count += 1
                            results[memory_id] = False
                            logger.error(
                                "Failed to process item in batch",
                                memory_id=memory_id[:8],
                                batch_id=batch_id,
                                error=str(item_error)
                            )

                    processing_time = (time.time() - start_time) * 1000

                    # Update global metrics
                    self._total_processed += len(memory_ids)
                    self._total_successful += success_count
                    self._total_failed += failure_count
                    self._total_time_ms += processing_time
                    self._batch_times.append(processing_time)
                    self._retry_count += retry_count

                    batch_result = BatchResult(
                        batch_id=batch_id,
                        status=BatchStatus.COMPLETED,
                        memory_ids=memory_ids,
                        success_count=success_count,
                        failure_count=failure_count,
                        processing_time_ms=processing_time,
                        retry_count=retry_count
                    )

                    logger.info(
                        "Batch processing completed",
                        batch_id=batch_id,
                        success_count=success_count,
                        failure_count=failure_count,
                        success_rate=batch_result.success_rate,
                        processing_time_ms=processing_time,
                        retry_count=retry_count
                    )

                    return batch_result

            except Exception as e:
                last_error = e
                retry_count += 1

                error_msg = str(e).lower()
                is_retryable = any(keyword in error_msg for keyword in [
                    'connection', 'timeout', 'rate limit', 'temporary',
                    'network', 'unavailable', 'overloaded', '503', '502',
                    'connection reset', 'connection refused'
                ])

                if not is_retryable or retry_count > self.max_retries:
                    # Non-retryable error or max retries exceeded
                    processing_time = (time.time() - start_time) * 1000

                    batch_result = BatchResult(
                        batch_id=batch_id,
                        status=BatchStatus.FAILED,
                        memory_ids=memory_ids,
                        success_count=0,
                        failure_count=len(memory_ids),
                        processing_time_ms=processing_time,
                        retry_count=retry_count,
                        error=str(e)
                    )

                    logger.error(
                        "Batch processing failed permanently",
                        batch_id=batch_id,
                        items_count=len(memory_ids),
                        error=str(e),
                        retry_count=retry_count,
                        processing_time_ms=processing_time
                    )

                    return batch_result

                # Calculate delay and wait before retry
                delay = self._calculate_delay(retry_count - 1)

                logger.warning(
                    "Batch processing failed, retrying",
                    batch_id=batch_id,
                    attempt=retry_count,
                    max_retries=self.max_retries,
                    delay=delay,
                    error=str(e)
                )

                await asyncio.sleep(delay)

        # This should not be reached, but just in case
        processing_time = (time.time() - start_time) * 1000
        error_msg = str(last_error) if last_error else "Unknown error"
        return BatchResult(
            batch_id=batch_id,
            status=BatchStatus.FAILED,
            memory_ids=memory_ids,
            success_count=0,
            failure_count=len(memory_ids),
            processing_time_ms=processing_time,
            retry_count=retry_count,
            error=error_msg
        )

    async def _store_embedding(self, memory_id: str, embedding: List[float]) -> bool:
        """
        Store embedding for a memory record in the database.

        Args:
            memory_id: Memory record ID
            embedding: Embedding vector

        Returns:
            True if successful, False otherwise
        """
        try:
            # Import here to avoid circular imports
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
            from sqlite_vec_helper import get_db_connection_with_vec

            project_root = Path(__file__).parent.parent.parent.parent.parent
            db_path = str(project_root / 'data' / 'devstream.db')

            with get_db_connection_with_vec(db_path) as conn:
                cursor = conn.cursor()

                # Use BLOB storage for efficiency
                import struct
                embedding_blob = struct.pack(f'{len(embedding)}f', *embedding)

                cursor.execute(
                    "UPDATE semantic_memory SET embedding_blob = ?, embedding_model = ?, embedding_dimension = ? WHERE id = ?",
                    (embedding_blob, 'gemma3', len(embedding), memory_id)
                )

                return bool(cursor.rowcount > 0)

        except Exception as e:
            logger.error(
                "Failed to store embedding",
                memory_id=memory_id[:8],
                error=str(e)
            )
            return False

    async def process_embedding_batch(
        self,
        memory_ids: List[str],
        contents: List[str],
        batch_size: Optional[int] = None,
        max_retries: Optional[int] = None
    ) -> Dict[str, bool]:
        """
        Process embeddings in batches to improve coverage and reduce Ollama rate limiting.

        Uses exponential backoff retry strategy and queue-based processing.
        Target: 80%+ embedding coverage vs current 0.4%.

        Args:
            memory_ids: List of memory record IDs to update
            contents: List of content strings to generate embeddings for
            batch_size: Number of items per batch (overrides instance default)
            max_retries: Maximum retry attempts per batch (overrides instance default)

        Returns:
            Dictionary mapping memory_id to success status

        Raises:
            EmbeddingProcessingError: If batch processing fails completely

        Example:
            >>> await process_embedding_batch(["mem1", "mem2"], ["content1", "content2"])
            {"mem1": True, "mem2": True}
        """
        if not memory_ids or not contents:
            logger.warning("Empty input lists provided")
            return {}

        if len(memory_ids) != len(contents):
            raise ValueError("Memory IDs and contents must have the same length")

        # Use provided parameters or instance defaults
        effective_batch_size = batch_size or self.batch_size
        effective_max_retries = max_retries or self.max_retries

        start_time = time.time()

        logger.info(
            "Starting embedding batch processing",
            total_items=len(memory_ids),
            batch_size=effective_batch_size,
            max_retries=effective_max_retries
        )

        try:
            # Split into batches
            batches = self._split_into_batches(memory_ids, contents)

            # Process all batches concurrently
            batch_tasks = []
            for i, (batch_ids, batch_contents) in enumerate(batches):
                batch_id = f"batch_{int(time.time())}_{i}"
                task = self._process_single_batch(batch_ids, batch_contents, batch_id)
                batch_tasks.append(task)

            # Wait for all batches to complete
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Collect results
            results = {}
            total_success = 0
            total_failed = 0

            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(
                        "Batch processing exception",
                        batch_index=i,
                        error=str(result)
                    )
                    # Mark all items in this batch as failed
                    batch_ids, _ = batches[i]
                    for memory_id in batch_ids:
                        results[memory_id] = False
                        total_failed += 1
                else:
                    # BatchResult returned successfully
                    batch_ids, _ = batches[i]
                    # Type guard: result is BatchResult here
                    assert isinstance(result, BatchResult), f"Expected BatchResult, got {type(result)}"
                    for j, memory_id in enumerate(batch_ids):
                        success = j < result.success_count
                        results[memory_id] = success
                        if success:
                            total_success += 1
                        else:
                            total_failed += 1

            total_time = (time.time() - start_time) * 1000
            success_rate = (total_success / len(memory_ids) * 100.0) if memory_ids else 0.0

            logger.info(
                "Embedding batch processing completed",
                total_items=len(memory_ids),
                total_success=total_success,
                total_failed=total_failed,
                success_rate=success_rate,
                total_time_ms=total_time,
                batches_count=len(batches)
            )

            return results

        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            logger.error(
                "Embedding batch processing failed completely",
                total_items=len(memory_ids),
                error=str(e),
                total_time_ms=total_time
            )
            raise EmbeddingProcessingError(f"Batch processing failed: {e}") from e

    async def process_missing_embeddings(
        self,
        limit: int = 1000,
        batch_size: Optional[int] = None
    ) -> Dict[str, bool]:
        """
        Process memory records that are missing embeddings.

        Identifies records without embeddings and processes them in batches.

        Args:
            limit: Maximum number of records to process
            batch_size: Batch size for processing

        Returns:
            Dictionary mapping memory_id to success status
        """
        try:
            # Import here to avoid circular imports
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
            from sqlite_vec_helper import get_db_connection_with_vec

            project_root = Path(__file__).parent.parent.parent.parent.parent
            db_path = str(project_root / 'data' / 'devstream.db')

            with get_db_connection_with_vec(db_path) as conn:
                cursor = conn.cursor()

                # Find records without embeddings
                cursor.execute(
                    """
                    SELECT id, content
                    FROM semantic_memory
                    WHERE (embedding_blob IS NULL OR embedding_blob = '')
                    AND content_type IN ('code', 'documentation', 'decision')
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,)
                )

                records = cursor.fetchall()

            if not records:
                logger.info("No records found missing embeddings")
                return {}

            memory_ids = [record[0] for record in records]
            contents = [record[1] for record in records]

            logger.info(
                "Found records missing embeddings",
                count=len(records),
                limit=limit
            )

            return await self.process_embedding_batch(memory_ids, contents, batch_size)

        except Exception as e:
            logger.error(
                "Failed to process missing embeddings",
                error=str(e)
            )
            raise EmbeddingProcessingError(f"Failed to process missing embeddings: {e}") from e

    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about batch processing performance.

        Returns:
            Dictionary with processing statistics
        """
        if self._total_processed == 0:
            return {
                "total_processed": 0,
                "success_count": 0,
                "error_count": 0,
                "success_rate": 0.0,
                "average_batch_time": 0.0,
                "retry_count": 0,
                "config": {
                    "batch_size": self.batch_size,
                    "max_concurrent_batches": self.max_concurrent_batches,
                    "max_retries": self.max_retries,
                    "base_delay": self.base_delay,
                    "max_delay": self.max_delay
                }
            }

        success_rate = (self._total_successful / self._total_processed * 100.0)
        average_batch_time = (sum(self._batch_times) / len(self._batch_times) if self._batch_times else 0.0)

        return {
            "total_processed": self._total_processed,
            "success_count": self._total_successful,
            "error_count": self._total_failed,
            "success_rate": round(success_rate, 2),
            "average_batch_time": round(average_batch_time, 2),
            "retry_count": self._retry_count,
            "config": {
                "batch_size": self.batch_size,
                "max_concurrent_batches": self.max_concurrent_batches,
                "max_retries": self.max_retries,
                "base_delay": self.base_delay,
                "max_delay": self.max_delay
            }
        }

    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self._total_processed = 0
        self._total_successful = 0
        self._total_failed = 0
        self._total_time_ms = 0.0
        self._batch_times = []
        self._retry_count = 0
        logger.info("Processing statistics reset")


# Global instance for import
_async_embedding_processor = None


def get_async_embedding_processor(
    batch_size: int = 20,
    max_retries: int = 3,
    max_concurrent_batches: int = 3
) -> AsyncEmbeddingBatchProcessor:
    """
    Get or create a global AsyncEmbeddingBatchProcessor instance.

    Args:
        batch_size: Number of items per batch
        max_retries: Maximum retry attempts per batch
        max_concurrent_batches: Maximum concurrent batch processing

    Returns:
        AsyncEmbeddingBatchProcessor instance
    """
    global _async_embedding_processor
    if (_async_embedding_processor is None or  # type: ignore[unreachable]
        _async_embedding_processor.batch_size != batch_size or
        _async_embedding_processor.max_retries != max_retries or
        _async_embedding_processor.max_concurrent_batches != max_concurrent_batches):
        _async_embedding_processor = AsyncEmbeddingBatchProcessor(
            batch_size=batch_size,
            max_retries=max_retries,
            max_concurrent_batches=max_concurrent_batches
        )
    return _async_embedding_processor