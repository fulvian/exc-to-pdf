#!/usr/bin/env python3
"""
DevStream Mass Vectorizer - FASE 4 Implementation

Context7-compliant mass vectorization for existing semantic_memory records.
Implements chunking strategy, Ollama batch processing, and CLI interface.

Features:
- Process existing semantic_memory records without embeddings
- 1000-record chunking with progress reporting
- Context7 Ollama batch processing patterns
- CLI interface with --dry-run option
- Memory management for large datasets
- Reuse existing EmbeddingGenerator and Ollama client patterns

Usage:
    .devstream/bin/python scripts/mass_vectorizer.py
    .devstream/bin/python scripts/mass_vectorizer.py --dry-run
    .devstream/bin/python scripts/mass_vectorizer.py --batch-size 500
    .devstream/bin/python scripts/mass_vectorizer.py --limit 5000
"""

import sys
import asyncio
import json
import sqlite3
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Tuple
from datetime import datetime
from dataclasses import dataclass
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TaskProgressColumn
)
from rich.table import Table
from rich.panel import Panel

# Add utils to path for DevStream utilities
sys.path.insert(0, str(Path(__file__).parent.parent / '.claude' / 'hooks' / 'devstream' / 'utils'))

from ollama_client import OllamaEmbeddingClient
from sqlite_vec_helper import get_db_connection_with_vec
from logger import get_devstream_logger


# ============================================================================
# Constants and Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / 'data' / 'devstream.db'

# Default configuration
DEFAULT_BATCH_SIZE = 1000  # Context7 chunking pattern
DEFAULT_LIMIT = 10000      # Maximum records to process
DEFAULT_MODEL = "embeddinggemma:300m"  # Context7 validated model


@dataclass
class VectorizationStats:
    """Statistics for vectorization process."""
    total_records: int = 0
    processed_records: int = 0
    successful_embeddings: int = 0
    failed_embeddings: int = 0
    chunks_processed: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        """Calculate processing duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.processed_records == 0:
            return 0.0
        return (self.successful_embeddings / self.processed_records) * 100.0


class MassVectorizer:
    """
    Context7-compliant mass vectorizer for semantic_memory records.

    Implements FASE 4 of Memory Vector Enhancement plan:
    - Query existing records without embeddings
    - Process in 1000-record chunks with memory management
    - Reuse existing Ollama infrastructure and patterns
    - Provide comprehensive progress tracking and error handling
    """

    def __init__(
        self,
        db_path: Path = DB_PATH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        model: str = DEFAULT_MODEL,
        console: Optional[Console] = None,
        verbose: bool = False
    ):
        """
        Initialize mass vectorizer with configuration.

        Args:
            db_path: Path to DevStream database
            batch_size: Number of records per chunk (Context7: 1000)
            model: Ollama embedding model
            console: Rich console for output
            verbose: Enable verbose logging
        """
        self.db_path = db_path
        self.batch_size = batch_size
        self.model = model
        self.console = console or Console()
        self.verbose = verbose

        # Initialize Context7 components
        self.ollama_client = OllamaEmbeddingClient(model=model)
        self.structured_logger = get_devstream_logger('mass_vectorizer')
        self.logger = self.structured_logger.logger

        # Statistics tracking
        self.stats = VectorizationStats()

        self.logger.info(
            "MassVectorizer initialized",
            db_path=str(db_path),
            batch_size=batch_size,
            model=model,
            verbose=verbose
        )

    def query_missing_embeddings(self, limit: int = DEFAULT_LIMIT) -> List[Dict[str, Any]]:
        """
        Query existing semantic_memory records without embeddings.

        Args:
            limit: Maximum number of records to retrieve

        Returns:
            List of records without embeddings, ordered by timestamp (newest first)

        Raises:
            sqlite3.Error: If database query fails
        """
        query = """
            SELECT id, content, content_type, created_at
            FROM semantic_memory
            WHERE embedding IS NULL OR embedding = ''
            ORDER BY created_at DESC
            LIMIT ?
        """

        try:
            with get_db_connection_with_vec(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (limit,))
                records = []

                for row in cursor.fetchall():
                    records.append({
                        'id': row[0],
                        'content': row[1],
                        'content_type': row[2],
                        'created_at': row[3]
                    })

                self.logger.info(
                    "Queried records without embeddings",
                    count=len(records),
                    limit=limit
                )

                return records

        except sqlite3.Error as e:
            self.logger.error(
                "Failed to query missing embeddings",
                error=str(e),
                query=query
            )
            raise

    def chunk_records(
        self,
        records: List[Dict[str, Any]],
        chunk_size: Optional[int] = None
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Split records into chunks for memory-efficient processing.

        Context7 chunking pattern: 1000-record chunks with generator
        to minimize memory usage for large datasets.

        Args:
            records: List of records to chunk
            chunk_size: Size of each chunk (uses default if None)

        Yields:
            List of records for each chunk
        """
        chunk_size = chunk_size or self.batch_size

        self.logger.debug(
            "Starting chunked processing",
            total_records=len(records),
            chunk_size=chunk_size
        )

        for i in range(0, len(records), chunk_size):
            chunk = records[i:i + chunk_size]
            chunk_num = (i // chunk_size) + 1
            total_chunks = (len(records) + chunk_size - 1) // chunk_size

            self.logger.debug(
                "Generated chunk",
                chunk_num=chunk_num,
                total_chunks=total_chunks,
                chunk_size=len(chunk)
            )

            yield chunk

    async def generate_embeddings_batch(self, records: List[Dict[str, Any]]) -> List[Optional[List[float]]]:
        """
        Generate embeddings for a batch of records using Context7 patterns.

        Reuses existing OllamaEmbeddingClient with error handling and retry logic.
        Processes records individually to handle partial failures gracefully.

        Args:
            records: List of records to generate embeddings for

        Returns:
            List of embedding vectors (None for failed records)
        """
        embeddings: List[Optional[List[float]]] = []

        for i, record in enumerate(records):
            try:
                # Extract content for embedding generation
                content = record['content']
                if not content or not content.strip():
                    self.logger.warning(
                        "Skipping empty content",
                        record_id=record['id']
                    )
                    embeddings.append(None)
                    continue

                # Generate embedding using existing Ollama client
                # Note: OllamaEmbeddingClient uses synchronous calls
                embedding = self.ollama_client.generate_embedding(content)

                if embedding:
                    embeddings.append(embedding)

                    if self.verbose:
                        self.console.print(
                            f"✓ Generated embedding for record {record['id']} "
                            f"(dim: {len(embedding)})",
                            style="green"
                        )
                else:
                    self.logger.warning(
                        "Failed to generate embedding",
                        record_id=record['id'],
                        content_length=len(content)
                    )
                    embeddings.append(None)

            except Exception as e:
                self.logger.error(
                    "Error generating embedding for record",
                    record_id=record['id'],
                    error=str(e),
                    error_type=type(e).__name__
                )
                embeddings.append(None)

        successful_count = sum(1 for e in embeddings if e is not None)
        self.logger.info(
            "Batch embedding generation completed",
            batch_size=len(records),
            successful=successful_count,
            failed=len(records) - successful_count
        )

        return embeddings

    def update_embeddings_batch(
        self,
        records: List[Dict[str, Any]],
        embeddings: List[Optional[List[float]]]
    ) -> int:
        """
        Update records in database with generated embeddings.

        Context7 atomic batch update pattern to prevent partial writes.
        Uses transaction to ensure all-or-nothing updates per chunk.

        Args:
            records: List of records that were processed
            embeddings: List of embedding vectors (None for failed records)

        Returns:
            Number of successfully updated records
        """
        if len(records) != len(embeddings):
            raise ValueError("Records and embeddings lists must have same length")

        updated_count = 0

        try:
            with get_db_connection_with_vec(str(self.db_path)) as conn:
                cursor = conn.cursor()

                # Begin transaction for atomic batch update
                cursor.execute("BEGIN TRANSACTION")

                try:
                    for record, embedding in zip(records, embeddings):
                        if embedding is not None:
                            # Convert embedding to JSON string for storage
                            embedding_json = json.dumps(embedding)

                            # Update record with embedding
                            cursor.execute("""
                                UPDATE semantic_memory
                                SET embedding = ?,
                                    embedding_model = ?,
                                    embedding_dimension = ?,
                                    updated_at = CURRENT_TIMESTAMP
                                WHERE id = ?
                            """, (
                                embedding_json,
                                self.model,
                                len(embedding),
                                record['id']
                            ))

                            updated_count += 1

                            if self.verbose:
                                self.console.print(
                                    f"✓ Updated record {record['id']} with embedding",
                                    style="cyan"
                                )

                    # Commit transaction
                    cursor.execute("COMMIT")

                    self.logger.info(
                        "Batch update completed successfully",
                        updated=updated_count,
                        total=len(records)
                    )

                except Exception as e:
                    # Rollback on error
                    cursor.execute("ROLLBACK")
                    raise e

        except Exception as e:
            self.logger.error(
                "Failed to update embeddings batch",
                error=str(e),
                batch_size=len(records),
                error_type=type(e).__name__
            )
            raise

        return updated_count

    async def process_chunk(self, records: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Process a single chunk of records.

        Args:
            records: Chunk of records to process

        Returns:
            Tuple of (successful_count, failed_count)
        """
        if not records:
            return 0, 0

        # Generate embeddings for chunk
        embeddings = await self.generate_embeddings_batch(records)

        # Update database with generated embeddings
        successful_count = self.update_embeddings_batch(records, embeddings)
        failed_count = len(records) - successful_count

        return successful_count, failed_count

    def dry_run(self, limit: int = DEFAULT_LIMIT) -> None:
        """
        Perform dry run to show what would be processed.

        Args:
            limit: Maximum number of records to analyze
        """
        self.console.print(Panel.fit(
            "[bold blue]MASS VECTORIZER - DRY RUN[/bold blue]\n"
            "Analyzing records without embeddings...",
            title="Dry Run Mode"
        ))

        try:
            # Query records that would be processed
            records = self.query_missing_embeddings(limit=limit)

            # Create summary table
            table = Table(title="Records Requiring Vectorization")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Total records found", str(len(records)))
            table.add_row("Batch size", str(self.batch_size))
            table.add_row("Estimated chunks", str((len(records) + self.batch_size - 1) // self.batch_size))
            table.add_row("Model", self.model)

            # Analyze content types
            content_types = {}
            for record in records:
                content_type = record.get('content_type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1

            table.add_row("\nContent Types:", "")
            for content_type, count in sorted(content_types.items()):
                table.add_row(f"  {content_type}", str(count))

            self.console.print(table)

            # Sample records
            if records and self.verbose:
                self.console.print("\n[bold]Sample Records:[/bold]")
                for i, record in enumerate(records[:3]):
                    content_preview = record['content'][:100] + "..." if len(record['content']) > 100 else record['content']
                    self.console.print(f"\n[yellow]Record {i+1}:[/yellow]")
                    self.console.print(f"  ID: {record['id']}")
                    self.console.print(f"  Type: {record['content_type']}")
                    self.console.print(f"  Content: {content_preview}")

            self.console.print(Panel.fit(
                f"[green]Dry run completed![/green]\n"
                f"Ready to process {len(records)} records in "
                f"{(len(records) + self.batch_size - 1) // self.batch_size} chunks.\n"
                f"Run without --dry-run to execute vectorization.",
                title="Dry Run Summary"
            ))

        except Exception as e:
            self.console.print(f"[red]Dry run failed: {e}[/red]")
            if self.verbose:
                import traceback
                self.console.print(traceback.format_exc())

    async def vectorize_missing_records(self, limit: int = DEFAULT_LIMIT) -> VectorizationStats:
        """
        Main method to vectorize existing records without embeddings.

        Context7 mass vectorization workflow:
        1. Query records without embeddings (newest first)
        2. Process in 1000-record chunks with progress tracking
        3. Generate embeddings using existing Ollama infrastructure
        4. Update database with atomic batch operations
        5. Provide comprehensive statistics and error handling

        Args:
            limit: Maximum number of records to process

        Returns:
            VectorizationStats with processing results
        """
        self.stats.start_time = datetime.now()

        self.console.print(Panel.fit(
            "[bold blue]MASS VECTORIZER - FASE 4[/bold blue]\n"
            "Starting vectorization of existing semantic_memory records...",
            title="Mass Vectorization"
        ))

        try:
            # Step 1: Query records without embeddings
            self.console.print("🔍 Querying records without embeddings...")
            records = self.query_missing_embeddings(limit=limit)
            self.stats.total_records = len(records)

            if not records:
                self.console.print("[green]✓ All records already have embeddings![/green]")
                self.stats.end_time = datetime.now()
                return self.stats

            self.console.print(f"Found {len(records)} records requiring vectorization")

            # Step 2: Process records in chunks with progress tracking
            total_chunks = (len(records) + self.batch_size - 1) // self.batch_size

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=self.console
            ) as progress:

                task = progress.add_task(
                    f"[cyan]Processing {total_chunks} chunks...",
                    total=total_chunks
                )

                # Step 3: Process each chunk
                for chunk_num, chunk in enumerate(self.chunk_records(records), 1):
                    progress.update(
                        task,
                        description=f"[cyan]Processing chunk {chunk_num}/{total_chunks}..."
                    )

                    try:
                        successful, failed = await self.process_chunk(chunk)

                        self.stats.processed_records += len(chunk)
                        self.stats.successful_embeddings += successful
                        self.stats.failed_embeddings += failed
                        self.stats.chunks_processed += 1

                        if self.verbose:
                            self.console.print(
                                f"✓ Chunk {chunk_num} completed: "
                                f"{successful} successful, {failed} failed"
                            )

                    except Exception as e:
                        self.console.print(
                            f"[red]❌ Chunk {chunk_num} failed: {e}[/red]"
                        )
                        self.logger.error(
                            "Chunk processing failed",
                            chunk_num=chunk_num,
                            chunk_size=len(chunk),
                            error=str(e)
                        )

                        # Continue with next chunk rather than failing entirely
                        self.stats.failed_embeddings += len(chunk)
                        self.stats.processed_records += len(chunk)

                    progress.advance(task)

            self.stats.end_time = datetime.now()

            # Step 4: Display final statistics
            self._display_final_stats()

            return self.stats

        except Exception as e:
            self.stats.end_time = datetime.now()
            self.console.print(f"[red]❌ Vectorization failed: {e}[/red]")
            self.logger.error(
                "Mass vectorization failed",
                error=str(e),
                error_type=type(e).__name__,
                duration=self.stats.duration_seconds
            )
            raise

    def _display_final_stats(self) -> None:
        """Display comprehensive final statistics."""
        stats_table = Table(title="Vectorization Results")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        stats_table.add_row("Total records", str(self.stats.total_records))
        stats_table.add_row("Processed records", str(self.stats.processed_records))
        stats_table.add_row("Successful embeddings", str(self.stats.successful_embeddings))
        stats_table.add_row("Failed embeddings", str(self.stats.failed_embeddings))
        stats_table.add_row("Chunks processed", str(self.stats.chunks_processed))
        stats_table.add_row("Success rate", f"{self.stats.success_rate:.1f}%")
        stats_table.add_row("Duration", f"{self.stats.duration_seconds:.1f}s")

        if self.stats.processed_records > 0:
            rate = self.stats.processed_records / self.stats.duration_seconds
            stats_table.add_row("Processing rate", f"{rate:.1f} records/sec")

        self.console.print(stats_table)

        # Status panel
        if self.stats.success_rate >= 90:
            status_style = "green"
            status_msg = "EXCELLENT"
        elif self.stats.success_rate >= 75:
            status_style = "yellow"
            status_msg = "GOOD"
        else:
            status_style = "red"
            status_msg = "NEEDS ATTENTION"

        self.console.print(Panel.fit(
            f"[bold {status_style}]Vectorization {status_msg}[/bold {status_style}]\n"
            f"Successfully processed {self.stats.successful_embeddings}/{self.stats.processed_records} records\n"
            f"Success rate: {self.stats.success_rate:.1f}%",
            title="Final Status"
        ))


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Mass vectorize existing semantic_memory records",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Process up to 10,000 records
  %(prog)s --dry-run                # Show what would be processed
  %(prog)s --batch-size 500         # Use smaller chunks
  %(prog)s --limit 5000             # Process fewer records
  %(prog)s --verbose                # Detailed output
        """
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without executing"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Chunk size for processing (default: {DEFAULT_BATCH_SIZE})"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Maximum records to process (default: {DEFAULT_LIMIT})"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Ollama embedding model (default: {DEFAULT_MODEL})"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser


async def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate arguments
    if args.batch_size <= 0:
        print("Error: --batch-size must be positive", file=sys.stderr)
        sys.exit(1)

    if args.limit <= 0:
        print("Error: --limit must be positive", file=sys.stderr)
        sys.exit(1)

    # Initialize mass vectorizer
    vectorizer = MassVectorizer(
        batch_size=args.batch_size,
        model=args.model,
        verbose=args.verbose
    )

    try:
        if args.dry_run:
            # Perform dry run
            vectorizer.dry_run(limit=args.limit)
        else:
            # Perform actual vectorization
            stats = await vectorizer.vectorize_missing_records(limit=args.limit)

            # Exit with appropriate code based on success rate
            if stats.success_rate >= 90:
                sys.exit(0)  # Success
            elif stats.success_rate >= 75:
                sys.exit(1)  # Warning
            else:
                sys.exit(2)  # Failure

    except KeyboardInterrupt:
        print("\n[yellow]Vectorization interrupted by user[/yellow]")
        sys.exit(130)

    except Exception as e:
        print(f"[red]Vectorization failed: {e}[/red]")
        if args.verbose:
            import traceback
            print(traceback.format_exc())
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())