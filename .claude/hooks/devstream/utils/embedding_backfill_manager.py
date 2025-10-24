#!/usr/bin/env python3
"""
DevStream Embedding Backfill Manager

Intelligent selective backfill system for missing embeddings in semantic memory.
Uses ContentQualityFilter criteria to prioritize high-value records for embedding generation.

Strategy: Smart Backfill (not full backfill)
- Prioritize high-quality content based on ContentQualityFilter criteria
- Focus on recent and frequently accessed records
- Implement batch processing with error handling and monitoring
- Use AsyncEmbeddingProcessor for efficient batch operations
- Track progress and provide detailed reporting

Target: Selectively backfill ~20-30K high-value records from 115,564 missing embeddings
instead of full backfill which would be inefficient for low-quality content.
"""

import asyncio
import json
import sqlite3
import struct
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging

# Import DevStream utilities
try:
    from .direct_client import get_direct_client
    from .connection_manager import ConnectionManager
    from .logger import get_devstream_logger
except ImportError:
    try:
        from direct_client import get_direct_client
        from connection_manager import ConnectionManager
        from logger import get_devstream_logger
    except ImportError:
        # Fallback logging
        logging.basicConfig(level=logging.INFO)
        def get_devstream_logger(name):
            return logging.getLogger(name)


class EmbeddingBackfillManager:
    """
    Intelligent backfill manager for missing embeddings in semantic memory.

    Uses selective criteria to prioritize high-value records instead of
    processing all missing embeddings indiscriminately.

    Attributes:
        client: DevStream direct client for database operations
        logger: Logger instance for detailed operation tracking
        batch_size: Number of records to process in each batch
        max_batches: Maximum number of batches to process (safety limit)

    Example:
        >>> manager = EmbeddingBackfillManager()
        >>> analysis = await manager.analyze_missing_embeddings()
        >>> print(f"High priority records: {analysis['high_priority_count']}")
        >>> results = await manager.run_selective_backfill()
        >>> print(f"Processed: {results['processed_count']}")
    """

    def __init__(self, batch_size: int = 100, max_batches: int = 300):
        """
        Initialize backfill manager with processing limits.

        Args:
            batch_size: Number of records to process in each batch (default: 100)
            max_batches: Maximum number of batches to process (safety limit, default: 300)
                         This prevents runaway processing of too many records
        """
        self.client = get_direct_client()
        self.logger = get_devstream_logger('embedding_backfill_manager')
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.connection_manager = ConnectionManager.get_instance("data/devstream.db")

    async def analyze_missing_embeddings(self) -> Dict[str, Any]:
        """
        Analyze records missing embeddings and categorize by priority.

        Returns:
            Dictionary with analysis results:
            - total_missing: Total records without embeddings
            - high_priority: Records meeting ContentQualityFilter criteria
            - medium_priority: Records with recent access or high importance
            - low_priority: Old/low-value records
            - analysis_breakdown: Detailed criteria analysis
        """
        start_time = time.time()

        try:
            with self.connection_manager.get_connection() as conn:
                # Get total records missing embeddings
                cursor = conn.execute("""
                    SELECT COUNT(*)
                    FROM semantic_memory
                    WHERE embedding_blob IS NULL
                    AND content IS NOT NULL
                    AND LENGTH(TRIM(content)) > 10
                """)
                total_missing = cursor.fetchone()[0]

                # Analyze by ContentQualityFilter criteria
                # High priority: Code, documentation, decisions, learning
                cursor = conn.execute("""
                    SELECT content_type, COUNT(*)
                    FROM semantic_memory
                    WHERE embedding_blob IS NULL
                    AND content IS NOT NULL
                    AND LENGTH(TRIM(content)) > 10
                    AND content_type IN ('code', 'documentation', 'decision', 'learning')
                    GROUP BY content_type
                """)
                high_priority_by_type = dict(cursor.fetchall())
                high_priority_count = sum(high_priority_by_type.values())

                # Medium priority: Recently accessed or high importance
                seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
                cursor = conn.execute("""
                    SELECT COUNT(*)
                    FROM semantic_memory
                    WHERE embedding_blob IS NULL
                    AND content IS NOT NULL
                    AND LENGTH(TRIM(content)) > 10
                    AND (
                        last_accessed_at > ?
                        OR importance_score >= 0.7
                        OR access_count > 5
                    )
                """, (seven_days_ago,))
                medium_priority_count = cursor.fetchone()[0]

                # Low priority: Everything else
                low_priority_count = total_missing - high_priority_count - medium_priority_count

                # Get sample records for quality assessment
                cursor = conn.execute("""
                    SELECT content_type, LENGTH(content) as content_length,
                           access_count, importance_score, created_at
                    FROM semantic_memory
                    WHERE embedding_blob IS NULL
                    AND content IS NOT NULL
                    AND LENGTH(TRIM(content)) > 10
                    ORDER BY importance_score DESC, access_count DESC
                    LIMIT 10
                """)
                sample_records = [dict(row) for row in cursor.fetchall()]

                duration = (time.time() - start_time) * 1000

                self.logger.logger.info(
                    f"Embedding backfill analysis completed",
                    extra={
                        "total_missing": total_missing,
                        "high_priority": high_priority_count,
                        "medium_priority": medium_priority_count,
                        "low_priority": low_priority_count,
                        "analysis_duration_ms": duration,
                        "operation": "analyze_missing_embeddings"
                    }
                )

                return {
                    "total_missing": total_missing,
                    "high_priority_count": high_priority_count,
                    "medium_priority_count": medium_priority_count,
                    "low_priority_count": low_priority_count,
                    "high_priority_by_type": high_priority_by_type,
                    "sample_records": sample_records,
                    "analysis_duration_ms": duration,
                    "recommended_approach": "selective_backfill",
                    "estimated_processing_time": self._estimate_processing_time(
                        high_priority_count, medium_priority_count
                    )
                }

        except Exception as e:
            self.logger.logger.error(
                f"Embedding backfill analysis failed: {e}",
                extra={"error": str(e), "operation": "analyze_missing_embeddings"}
            )
            raise

    def _estimate_processing_time(self, high_priority: int, medium_priority: int) -> Dict[str, Any]:
        """
        Estimate processing time for backfill operations.

        Args:
            high_priority: Number of high priority records
            medium_priority: Number of medium priority records

        Returns:
            Dictionary with time estimates in minutes
        """
        # Based on empirical data: ~100ms per embedding generation
        # Plus batch overhead and safety factors
        ms_per_embedding = 100
        records_to_process = high_priority + min(medium_priority, 5000)  # Cap medium priority

        total_ms = records_to_process * ms_per_embedding
        # Add 30% overhead for batch processing, error handling, etc.
        total_ms_with_overhead = total_ms * 1.3

        return {
            "records_to_process": records_to_process,
            "estimated_time_minutes": round(total_ms_with_overhead / 60000, 1),
            "estimated_time_hours": round(total_ms_with_overhead / 3600000, 2)
        }

    async def run_selective_backfill(
        self,
        priority_levels: List[str] = ["high", "medium"],
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Run selective backfill for missing embeddings.

        Args:
            priority_levels: Which priority levels to process ("high", "medium", "low")
            dry_run: If True, only analyze without processing

        Returns:
            Dictionary with backfill results:
            - processed_count: Number of records processed
            - success_count: Number of successful embeddings generated
            - error_count: Number of errors encountered
            - batches_processed: Number of batches completed
            - processing_time: Total processing time in milliseconds
        """
        start_time = time.time()

        if dry_run:
            self.logger.logger.info("Running DRY RUN - no actual embedding generation")

        try:
            # Get records to process based on priority
            records_to_process = await self._get_records_for_backfill(priority_levels)

            if not records_to_process:
                return {
                    "processed_count": 0,
                    "success_count": 0,
                    "error_count": 0,
                    "batches_processed": 0,
                    "processing_time_ms": 0,
                    "message": "No records found for backfill"
                }

            # Initialize counters
            processed_count = 0
            success_count = 0
            error_count = 0
            batches_processed = 0

            self.logger.logger.info(
                f"Starting selective backfill for {len(records_to_process)} records",
                extra={
                    "total_records": len(records_to_process),
                    "priority_levels": priority_levels,
                    "dry_run": dry_run,
                    "batch_size": self.batch_size,
                    "operation": "selective_backfill_start"
                }
            )

            # Process in batches
            for i in range(0, len(records_to_process), self.batch_size):
                if batches_processed >= self.max_batches:
                    self.logger.logger.warning(
                        f"Reached maximum batch limit ({self.max_batches}), stopping backfill"
                    )
                    break

                batch = records_to_process[i:i + self.batch_size]
                batch_start = time.time()

                try:
                    if dry_run:
                        # Simulate batch processing
                        batch_results = [{"success": True} for _ in batch]
                    else:
                        # Actual batch processing
                        batch_results = await self._process_embedding_batch(batch)

                    # Update counters
                    for result in batch_results:
                        processed_count += 1
                        if result.get("success", False):
                            success_count += 1
                        else:
                            error_count += 1

                    batches_processed += 1
                    batch_duration = (time.time() - batch_start) * 1000

                    self.logger.logger.info(
                        f"Batch {batches_processed} completed",
                        extra={
                            "batch_number": batches_processed,
                            "batch_size": len(batch),
                            "batch_duration_ms": batch_duration,
                            "cumulative_success_count": success_count,
                            "cumulative_error_count": error_count,
                            "operation": "embedding_batch_completed"
                        }
                    )

                    # Small delay between batches to prevent overwhelming the system
                    if not dry_run:
                        await asyncio.sleep(0.1)

                except Exception as e:
                    self.logger.logger.error(
                        f"Batch {batches_processed} failed: {e}",
                        extra={
                            "batch_number": batches_processed,
                            "error": str(e),
                            "operation": "embedding_batch_failed"
                        }
                    )
                    error_count += len(batch)
                    continue

            total_duration = (time.time() - start_time) * 1000

            results = {
                "processed_count": processed_count,
                "success_count": success_count,
                "error_count": error_count,
                "batches_processed": batches_processed,
                "processing_time_ms": total_duration,
                "success_rate": round((success_count / processed_count * 100) if processed_count > 0 else 0, 2),
                "dry_run": dry_run,
                "priority_levels": priority_levels
            }

            self.logger.logger.info(
                f"Selective backfill completed",
                extra={
                    **results,
                    "operation": "selective_backfill_completed"
                }
            )

            return results

        except Exception as e:
            self.logger.logger.error(
                f"Selective backfill failed: {e}",
                extra={"error": str(e), "operation": "selective_backfill_failed"}
            )
            raise

    async def _get_records_for_backfill(self, priority_levels: List[str]) -> List[Dict[str, Any]]:
        """
        Get records for backfill based on priority levels.

        Args:
            priority_levels: Priority levels to include ("high", "medium", "low")

        Returns:
            List of record dictionaries for processing
        """
        with self.connection_manager.get_connection() as conn:
            # Build WHERE clause based on priority levels
            conditions = [
                "embedding_blob IS NULL",
                "content IS NOT NULL",
                "LENGTH(TRIM(content)) > 10"
            ]

            params = []

            if "high" in priority_levels:
                conditions.append("content_type IN ('code', 'documentation', 'decision', 'learning')")

            if "medium" in priority_levels and "high" not in priority_levels:
                # Only add medium criteria if high priority not included
                seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()
                conditions.append("""
                    (
                        last_accessed_at > ?
                        OR importance_score >= 0.7
                        OR access_count > 5
                    )
                """)
                params.append(seven_days_ago)

            if "low" in priority_levels and "high" not in priority_levels and "medium" not in priority_levels:
                # Only process low priority if neither high nor medium selected
                pass  # Use base conditions only

            # Order by priority criteria
            order_clause = """
                ORDER BY
                    CASE
                        WHEN content_type IN ('code', 'documentation', 'decision', 'learning') THEN 1
                        WHEN (last_accessed_at > datetime('now', '-7 days') OR importance_score >= 0.7 OR access_count > 5) THEN 2
                        ELSE 3
                    END ASC,
                    importance_score DESC,
                    access_count DESC,
                    created_at DESC
            """

            # Limit to prevent runaway processing
            limit = min(self.batch_size * self.max_batches, 30000)  # Max 30K records

            sql = f"""
                SELECT id, content, content_type, keywords,
                       access_count, importance_score, created_at
                FROM semantic_memory
                WHERE {' AND '.join(conditions)}
                {order_clause}
                LIMIT ?
            """

            params.append(limit)

            cursor = conn.execute(sql, params)
            records = [dict(row) for row in cursor.fetchall()]

            self.logger.logger.info(
                f"Retrieved {len(records)} records for backfill",
                extra={
                    "priority_levels": priority_levels,
                    "records_count": len(records),
                    "operation": "get_backfill_records"
                }
            )

            return records

    async def _process_embedding_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of records for embedding generation.

        Args:
            batch: List of record dictionaries to process

        Returns:
            List of result dictionaries for each record
        """
        results = []

        try:
            # Import OllamaEmbeddingClient
            try:
                from .ollama_client import OllamaEmbeddingClient
            except ImportError:
                try:
                    from ollama_client import OllamaEmbeddingClient
                except ImportError:
                    raise ImportError("OllamaEmbeddingClient not available")

            ollama_client = OllamaEmbeddingClient()

            # Process each record in the batch
            for record in batch:
                try:
                    record_id = record['id']
                    content = record['content']

                    # Generate embedding
                    embedding = ollama_client.generate_embedding(content)

                    if embedding and len(embedding) > 0:
                        # Convert to BLOB
                        embedding_blob = struct.pack(f'{len(embedding)}f', *embedding)
                        embedding_dimension = len(embedding)
                        embedding_model = 'embeddinggemma:300m'

                        # Update database with embedding
                        await self._update_record_embedding(
                            record_id, embedding_blob, embedding_model, embedding_dimension
                        )

                        results.append({
                            "record_id": record_id,
                            "success": True,
                            "embedding_dimension": embedding_dimension,
                            "content_type": record['content_type']
                        })
                    else:
                        results.append({
                            "record_id": record_id,
                            "success": False,
                            "error": "Empty embedding generated"
                        })

                except Exception as e:
                    self.logger.logger.warning(
                        f"Failed to process record {record.get('id', 'unknown')}: {e}",
                        extra={
                            "record_id": record.get('id'),
                            "error": str(e),
                            "operation": "process_single_record"
                        }
                    )
                    results.append({
                        "record_id": record.get('id'),
                        "success": False,
                        "error": str(e)
                    })

        except Exception as e:
            self.logger.logger.error(
                f"Batch processing failed: {e}",
                extra={"error": str(e), "operation": "process_embedding_batch"}
            )
            # Return error results for all records in batch
            for record in batch:
                results.append({
                    "record_id": record.get('id'),
                    "success": False,
                    "error": f"Batch processing failed: {str(e)}"
                })

        return results

    async def _update_record_embedding(
        self,
        record_id: str,
        embedding_blob: bytes,
        embedding_model: str,
        embedding_dimension: int
    ) -> None:
        """
        Update a single record with its embedding.

        Args:
            record_id: ID of record to update
            embedding_blob: Binary embedding data
            embedding_model: Model used for embedding generation
            embedding_dimension: Dimension of embedding vector
        """
        with self.connection_manager.get_connection() as conn:
            conn.execute("""
                UPDATE semantic_memory
                SET embedding_blob = ?,
                    embedding_model = ?,
                    embedding_dimension = ?,
                    updated_at = ?
                WHERE id = ?
            """, (embedding_blob, embedding_model, embedding_dimension,
                  datetime.now().isoformat(), record_id))

            # Also update vector table if it exists
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO vec_semantic_memory
                    (memory_id, content_type, embedding)
                    VALUES (?, ?, ?)
                """, (record_id, 'general', embedding_blob))
            except sqlite3.OperationalError:
                # Vector table might not exist or not be accessible
                pass

    async def get_backfill_progress(self) -> Dict[str, Any]:
        """
        Get current progress of embedding backfill.

        Returns:
            Dictionary with progress information:
            - total_records: Total records in semantic_memory
            - records_with_embeddings: Count of records with embeddings
            - records_missing_embeddings: Count without embeddings
            - completion_percentage: Percentage of records with embeddings
        """
        try:
            with self.connection_manager.get_connection() as conn:
                # Get total records
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM semantic_memory
                    WHERE content IS NOT NULL AND LENGTH(TRIM(content)) > 10
                """)
                total_records = cursor.fetchone()[0]

                # Get records with embeddings
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM semantic_memory
                    WHERE embedding_blob IS NOT NULL
                    AND content IS NOT NULL AND LENGTH(TRIM(content)) > 10
                """)
                records_with_embeddings = cursor.fetchone()[0]

                # Calculate missing
                records_missing_embeddings = total_records - records_with_embeddings
                completion_percentage = round((records_with_embeddings / total_records * 100) if total_records > 0 else 0, 2)

                # Get breakdown by content type
                cursor = conn.execute("""
                    SELECT content_type,
                           COUNT(*) as total,
                           COUNT(embedding_blob) as with_embeddings
                    FROM semantic_memory
                    WHERE content IS NOT NULL AND LENGTH(TRIM(content)) > 10
                    GROUP BY content_type
                """)
                by_content_type = [dict(row) for row in cursor.fetchall()]

                return {
                    "total_records": total_records,
                    "records_with_embeddings": records_with_embeddings,
                    "records_missing_embeddings": records_missing_embeddings,
                    "completion_percentage": completion_percentage,
                    "by_content_type": by_content_type,
                    "last_updated": datetime.now().isoformat()
                }

        except Exception as e:
            self.logger.logger.error(f"Failed to get backfill progress: {e}")
            return {"error": str(e)}


# Singleton instance
_backfill_manager = None

def get_backfill_manager() -> EmbeddingBackfillManager:
    """Get singleton backfill manager instance."""
    global _backfill_manager
    if _backfill_manager is None:
        _backfill_manager = EmbeddingBackfillManager()
    return _backfill_manager


# Convenience functions
async def analyze_embedding_backfill() -> Dict[str, Any]:
    """Analyze missing embeddings for backfill planning."""
    manager = get_backfill_manager()
    return await manager.analyze_missing_embeddings()


async def run_selective_backfill(
    priority_levels: List[str] = ["high", "medium"],
    dry_run: bool = False
) -> Dict[str, Any]:
    """Run selective embedding backfill."""
    manager = get_backfill_manager()
    return await manager.run_selective_backfill(priority_levels, dry_run)


async def get_embedding_backfill_progress() -> Dict[str, Any]:
    """Get current embedding backfill progress."""
    manager = get_backfill_manager()
    return await manager.get_backfill_progress()


if __name__ == "__main__":
    async def test_backfill_manager():
        """Test backfill manager functionality."""
        print("🧪 Testing Embedding Backfill Manager...")

        manager = get_backfill_manager()

        # Test analysis
        print("1. Analyzing missing embeddings...")
        analysis = await manager.analyze_missing_embeddings()
        print(f"   Total missing: {analysis['total_missing']}")
        print(f"   High priority: {analysis['high_priority_count']}")
        print(f"   Medium priority: {analysis['medium_priority_count']}")
        print(f"   Estimated time: {analysis['estimated_processing_time']['estimated_time_minutes']} minutes")

        # Test progress
        print("2. Getting current progress...")
        progress = await manager.get_backfill_progress()
        print(f"   Completion: {progress['completion_percentage']}%")
        print(f"   With embeddings: {progress['records_with_embeddings']}")
        print(f"   Missing embeddings: {progress['records_missing_embeddings']}")

        # Test dry run
        print("3. Running dry run backfill...")
        dry_run_results = await manager.run_selective_backfill(["high"], dry_run=True)
        print(f"   Would process: {dry_run_results['processed_count']} records")

        print("🎉 Backfill manager test completed!")

    asyncio.run(test_backfill_manager())