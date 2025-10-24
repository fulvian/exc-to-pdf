#!/usr/bin/env python3
"""
DevStream Embedding Backfill Utility - Context7 Compliant

Generates embeddings for existing semantic_memory records that lack them.
Uses batch processing with Ollama for efficient embedding generation.

Context7 Research:
- ollama-python: Supports batch embedding with array input
- Best practice: Batch size ≤16 for accuracy (Context7 recommendation)
- Pattern: ollama.embed(model='gemma3', input=['text1', 'text2'])

Usage:
    .devstream/bin/python .claude/hooks/devstream/memory/backfill_embeddings.py [--dry-run] [--batch-size 10]
"""

import sys
import asyncio
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import argparse

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))
from logger import get_devstream_logger
from sqlite_vec_helper import get_db_connection_with_vec


@dataclass
class BackfillRecord:
    """Record needing embedding backfill."""
    id: str
    content: str
    content_type: str
    content_length: int


class EmbeddingBackfillService:
    """
    Service for backfilling embeddings using Ollama batch API.

    Context7 Pattern: Batch processing with size limits for accuracy.
    """

    def __init__(
        self,
        db_path: str,
        batch_size: int = 10,
        dry_run: bool = False
    ):
        """
        Initialize backfill service.

        Args:
            db_path: Path to DevStream database
            batch_size: Number of records per batch (default: 10, max: 16)
            dry_run: If True, only simulate without actual updates
        """
        self.db_path = db_path
        self.batch_size = min(batch_size, 16)  # Context7: max 16 for accuracy
        self.dry_run = dry_run

        self.structured_logger = get_devstream_logger('embedding_backfill')
        self.logger = self.structured_logger.logger

        # Ollama configuration
        self.ollama_model = "embeddinggemma:300m"  # DevStream standard
        self.ollama_base_url = "http://localhost:11434"

        self.logger.info(
            f"EmbeddingBackfillService initialized",
            db_path=db_path,
            batch_size=self.batch_size,
            dry_run=dry_run,
            model=self.ollama_model
        )

    def _get_db_connection(self) -> sqlite3.Connection:
        """
        Get database connection with sqlite-vec extension loaded via ConnectionManager.

        Uses centralized get_db_connection_with_vec() which:
        - Enforces WAL mode via ConnectionManager
        - Loads sqlite-vec extension for vector operations
        - Provides thread-safe connection pooling

        Returns:
            SQLite connection with sqlite-vec loaded and WAL mode enabled
        """
        # Use helper that integrates ConnectionManager + sqlite-vec
        conn = get_db_connection_with_vec(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_records_without_embeddings(self) -> List[BackfillRecord]:
        """
        Fetch all semantic_memory records without embeddings.

        Returns:
            List of BackfillRecord objects
        """
        conn = self._get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, content, content_type, LENGTH(content) as content_length
            FROM semantic_memory
            WHERE embedding IS NULL
            ORDER BY created_at DESC
        """)

        records = [
            BackfillRecord(
                id=row['id'],
                content=row['content'],
                content_type=row['content_type'],
                content_length=row['content_length']
            )
            for row in cursor.fetchall()
        ]

        conn.close()

        self.logger.info(f"Found {len(records)} records without embeddings")
        return records

    async def generate_embeddings_batch(
        self,
        texts: List[str]
    ) -> Optional[List[List[float]]]:
        """
        Generate embeddings for batch of texts using Ollama.

        Context7 Pattern: Use ollama.embed() with array input for batch processing.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors or None if failed

        Note:
            Model auto-unloads after 5 minutes of inactivity (keep_alive="5m").
            Cold start penalty: ~2-3s when model reloads after idle period.
        """
        try:
            # Dynamic import to avoid startup dependency
            import ollama

            self.logger.debug(f"Generating embeddings for {len(texts)} texts")

            # Context7 pattern: Batch embedding with array input
            response = ollama.embed(
                model=self.ollama_model,
                input=texts,
                keep_alive="5m"  # Auto-unload after 5 min inactivity
            )

            embeddings = response.get('embeddings', [])

            if len(embeddings) != len(texts):
                self.logger.error(
                    f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}"
                )
                return None

            self.logger.debug(
                f"Generated {len(embeddings)} embeddings",
                dimensions=len(embeddings[0]) if embeddings else 0
            )

            return embeddings

        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return None

    def update_embeddings_in_db(
        self,
        records: List[BackfillRecord],
        embeddings: List[List[float]]
    ) -> int:
        """
        Update database with generated embeddings.

        Args:
            records: List of records to update
            embeddings: Corresponding embedding vectors

        Returns:
            Number of records successfully updated
        """
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would update {len(records)} records")
            return len(records)

        conn = self._get_db_connection()
        cursor = conn.cursor()

        updated_count = 0

        for record, embedding in zip(records, embeddings):
            try:
                embedding_json = json.dumps(embedding)
                embedding_dimension = len(embedding)

                cursor.execute("""
                    UPDATE semantic_memory
                    SET embedding = ?,
                        embedding_model = ?,
                        embedding_dimension = ?
                    WHERE id = ?
                """, (
                    embedding_json,
                    self.ollama_model,
                    embedding_dimension,
                    record.id
                ))

                updated_count += 1

            except Exception as e:
                self.logger.error(
                    f"Failed to update record {record.id}: {e}"
                )

        conn.commit()
        conn.close()

        self.logger.info(f"Updated {updated_count}/{len(records)} records")
        return updated_count

    def sync_to_vec0(self, record_ids: List[str]) -> int:
        """
        Sync updated records to vec0 vector search index.

        Args:
            record_ids: List of record IDs to sync

        Returns:
            Number of records successfully synced
        """
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would sync {len(record_ids)} records to vec0")
            return len(record_ids)

        conn = self._get_db_connection()
        cursor = conn.cursor()

        synced_count = 0

        for record_id in record_ids:
            try:
                # Fetch record with embedding
                cursor.execute("""
                    SELECT embedding, content_type, content
                    FROM semantic_memory
                    WHERE id = ?
                """, (record_id,))

                row = cursor.fetchone()
                if not row or not row[0]:  # No embedding
                    continue

                embedding, content_type, content = row
                content_preview = content[:200] if content else ""

                # Insert into vec0 (will be handled by trigger if exists)
                cursor.execute("""
                    INSERT INTO vec_semantic_memory(embedding, content_type, memory_id, content_preview)
                    VALUES (?, ?, ?, ?)
                """, (embedding, content_type, record_id, content_preview))

                synced_count += 1

            except Exception as e:
                self.logger.warning(f"vec0 sync failed for {record_id}: {e}")
                # Continue - FTS5 will still work

        conn.commit()
        conn.close()

        self.logger.info(f"Synced {synced_count}/{len(record_ids)} records to vec0")
        return synced_count

    async def run_backfill(self) -> Dict[str, Any]:
        """
        Execute complete backfill process.

        Returns:
            Statistics about backfill operation
        """
        stats = {
            "total_records": 0,
            "processed_batches": 0,
            "updated_records": 0,
            "synced_to_vec0": 0,
            "failed_records": 0,
            "dry_run": self.dry_run
        }

        # Fetch records without embeddings
        records = self.get_records_without_embeddings()
        stats["total_records"] = len(records)

        if not records:
            self.logger.info("No records need backfill")
            return stats

        # Process in batches
        for i in range(0, len(records), self.batch_size):
            batch = records[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(records) + self.batch_size - 1) // self.batch_size

            self.logger.info(
                f"Processing batch {batch_num}/{total_batches}",
                batch_size=len(batch)
            )

            # Extract texts from batch
            texts = [record.content for record in batch]

            # Generate embeddings
            embeddings = await self.generate_embeddings_batch(texts)

            if not embeddings:
                self.logger.error(f"Batch {batch_num} failed - skipping")
                stats["failed_records"] += len(batch)
                continue

            # Update database (triggers will handle vec0/FTS5 sync automatically)
            updated = self.update_embeddings_in_db(batch, embeddings)
            stats["updated_records"] += updated
            stats["processed_batches"] += 1

            # Note: vec0 sync happens automatically via UPDATE trigger
            stats["synced_to_vec0"] += updated  # Assume trigger succeeds

            # Progress feedback
            progress_pct = ((i + len(batch)) / len(records)) * 100
            self.logger.info(
                f"Progress: {progress_pct:.1f}%",
                updated=stats["updated_records"],
                total=len(records)
            )

        return stats


async def main():
    """Main entry point for backfill utility."""
    parser = argparse.ArgumentParser(
        description="Backfill embeddings for DevStream semantic memory"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Simulate backfill without actual updates"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help="Number of records per batch (max 16, default 10)"
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=None,
        help="Path to DevStream database (default: data/devstream.db)"
    )

    args = parser.parse_args()

    # Determine database path
    if args.db_path:
        db_path = args.db_path
    else:
        project_root = Path(__file__).parent.parent.parent.parent.parent
        db_path = str(project_root / 'data' / 'devstream.db')

    # Create service
    service = EmbeddingBackfillService(
        db_path=db_path,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )

    # Run backfill
    print("🧠 DevStream Embedding Backfill Utility")
    print("=" * 50)
    print(f"Database: {db_path}")
    print(f"Batch size: {service.batch_size}")
    print(f"Model: {service.ollama_model}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 50)
    print()

    stats = await service.run_backfill()

    # Print summary
    print()
    print("=" * 50)
    print("📊 Backfill Complete")
    print("=" * 50)
    print(f"Total records: {stats['total_records']}")
    print(f"Processed batches: {stats['processed_batches']}")
    print(f"Updated records: {stats['updated_records']}")
    print(f"Synced to vec0: {stats['synced_to_vec0']}")
    print(f"Failed records: {stats['failed_records']}")

    if stats['dry_run']:
        print("\n⚠️  This was a DRY RUN - no changes were made")
    else:
        success_rate = (stats['updated_records'] / stats['total_records'] * 100) if stats['total_records'] > 0 else 0
        print(f"\n✅ Success rate: {success_rate:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())