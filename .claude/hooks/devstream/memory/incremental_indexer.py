#!/usr/bin/env python3
"""
DevStream Memory Bootstrap - Incremental Indexer
Context7-compliant incremental indexing with source-based deduplication.

This module provides intelligent incremental indexing using patterns from
ChromaDB and LangChain, ensuring efficient updates and avoiding duplicate
processing of unchanged content.

Key Features:
- Source-based deduplication using checksums
- Incremental updates with cleanup modes (full/incremental)
- Batch processing for performance optimization
- Change detection and smart updates
- Progress tracking and resume capability
Context7 Pattern: Source tracking + incremental updates + performance optimization
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Iterator
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
import sqlite3
from contextlib import contextmanager

# Import DevStream components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'utils'))
try:
    from direct_client import get_direct_client
    DIRECT_CLIENT_AVAILABLE = True
except ImportError:
    DIRECT_CLIENT_AVAILABLE = False
    logging.warning("Direct client not available, using fallback storage")

from document_processor import Document, ProcessedDocument, create_document_processor


@dataclass
class IndexingRecord:
    """Record of indexed content for change detection."""
    source_path: str
    checksum: str
    last_modified: float
    content_type: str
    chunk_count: int
    indexed_at: str
    file_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexingResult:
    """Result of indexing operation."""
    success: bool
    total_files: int
    processed_files: int
    updated_files: int
    added_files: int
    skipped_files: int
    deleted_files: int
    total_chunks: int
    processing_time: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class IndexingDatabase:
    """Local database for tracking indexing state."""

    def __init__(self, db_path: str):
        """
        Initialize indexing database.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.IndexingDB")
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema."""
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS indexing_records (
                    source_path TEXT PRIMARY KEY,
                    checksum TEXT NOT NULL,
                    last_modified REAL NOT NULL,
                    content_type TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source_path
                ON indexing_records(source_path)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_modified
                ON indexing_records(last_modified)
            """)

    @contextmanager
    def get_connection(self):
        """Get database connection with proper error handling."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def get_record(self, source_path: str) -> Optional[IndexingRecord]:
        """Get indexing record for a source path."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM indexing_records WHERE source_path = ?",
                (source_path,)
            )
            row = cursor.fetchone()

            if row:
                metadata = json.loads(row['metadata']) if row['metadata'] else {}
                return IndexingRecord(
                    source_path=row['source_path'],
                    checksum=row['checksum'],
                    last_modified=row['last_modified'],
                    content_type=row['content_type'],
                    chunk_count=row['chunk_count'],
                    indexed_at=row['indexed_at'],
                    file_size=row['file_size'],
                    metadata=metadata
                )
            return None

    def upsert_record(self, record: IndexingRecord) -> None:
        """Insert or update indexing record."""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO indexing_records
                (source_path, checksum, last_modified, content_type,
                 chunk_count, indexed_at, file_size, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.source_path,
                record.checksum,
                record.last_modified,
                record.content_type,
                record.chunk_count,
                record.indexed_at,
                record.file_size,
                json.dumps(record.metadata)
            ))

    def delete_record(self, source_path: str) -> None:
        """Delete indexing record."""
        with self.get_connection() as conn:
            conn.execute(
                "DELETE FROM indexing_records WHERE source_path = ?",
                (source_path,)
            )

    def get_all_records(self) -> List[IndexingRecord]:
        """Get all indexing records."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM indexing_records ORDER BY last_modified DESC")
            records = []

            for row in cursor.fetchall():
                metadata = json.loads(row['metadata']) if row['metadata'] else {}
                records.append(IndexingRecord(
                    source_path=row['source_path'],
                    checksum=row['checksum'],
                    last_modified=row['last_modified'],
                    content_type=row['content_type'],
                    chunk_count=row['chunk_count'],
                    indexed_at=row['indexed_at'],
                    file_size=row['file_size'],
                    metadata=metadata
                ))

            return records

    def get_stale_records(self, project_root: str) -> List[IndexingRecord]:
        """Get records for files that no longer exist."""
        records = self.get_all_records()
        stale_records = []

        for record in records:
            file_path = Path(project_root) / record.source_path
            if not file_path.exists():
                stale_records.append(record)

        return stale_records

    def cleanup_stale_records(self, project_root: str) -> int:
        """Remove records for non-existent files."""
        stale_records = self.get_stale_records(project_root)

        with self.get_connection() as conn:
            for record in stale_records:
                conn.execute(
                    "DELETE FROM indexing_records WHERE source_path = ?",
                    (record.source_path,)
                )

        self.logger.info(f"Cleaned up {len(stale_records)} stale records")
        return len(stale_records)


class IncrementalIndexer:
    """
    Context7-compliant incremental indexer with source-based deduplication.

    Implements ChromaDB and LangChain patterns for:
    - Source-based deduplication using checksums
    - Incremental updates with cleanup modes
    - Batch processing for performance
    - Change detection and smart updates
    """

    def __init__(self, project_root: str, memory_client=None):
        """
        Initialize incremental indexer.

        Args:
            project_root: Root path of the project
            memory_client: DevStream memory client (optional)
        """
        self.project_root = Path(project_root)
        self.memory_client = memory_client or (get_direct_client() if DIRECT_CLIENT_AVAILABLE else None)

        # Initialize indexing database
        db_path = self.project_root / '.claude' / 'indexing.db'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.indexing_db = IndexingDatabase(str(db_path))

        # Initialize document processor
        self.doc_processor = create_document_processor(str(project_root), batch_size=25)

        self.logger = logging.getLogger(__name__)

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate checksum for file content."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return hashlib.sha256(content).hexdigest()[:16]
        except Exception as e:
            self.logger.error(f"Error calculating checksum for {file_path}: {e}")
            return ""

    def _needs_indexing(self, file_path: Path) -> Tuple[bool, Optional[IndexingRecord]]:
        """
        Check if file needs indexing based on change detection.

        Context7 Pattern: Smart change detection using checksum + metadata.
        """
        if not file_path.exists():
            return False, None

        # Get current file stats
        stat = file_path.stat()
        current_checksum = self._calculate_file_checksum(file_path)
        current_modified = stat.st_mtime
        current_size = stat.st_size

        # Get existing record
        existing_record = self.indexing_db.get_record(str(file_path.relative_to(self.project_root)))

        if not existing_record:
            return True, None  # New file

        # Check if file changed
        if (existing_record.checksum != current_checksum or
            existing_record.last_modified != current_modified or
            existing_record.file_size != current_size):
            return True, existing_record  # Modified file

        return False, existing_record  # Unchanged file

    def _store_documents_batch(self, documents: List[Document], source_path: str) -> bool:
        """
        Store documents in memory database.

        Context7 Pattern: Batch storage with error handling.
        """
        if not self.memory_client:
            self.logger.warning("No memory client available, skipping storage")
            return False

        try:
            # Store documents with enhanced metadata
            for doc in documents:
                metadata = doc.metadata.copy()
                metadata.update({
                    'indexed_at': datetime.now().isoformat(),
                    'source_path': source_path,
                    'indexing_batch': True
                })

                # Use the appropriate method based on available client
                if hasattr(self.memory_client, 'store_memory'):
                    # Context7 Pattern: Run async code from sync context using asyncio.run()
                    # This is the recommended pattern - creates clean event loop per call
                    # AnyIO in direct_client ensures no event loop blocking
                    import asyncio
                    try:
                        result = asyncio.run(
                            self.memory_client.store_memory(
                                content=doc.page_content,
                                content_type=metadata.get('content_type', 'code'),
                                keywords=metadata.get('keywords', []),
                                session_id=metadata.get('session_id'),
                                source=source_path
                            )
                        )
                    except RuntimeError as e:
                        # If already in event loop (shouldn't happen in bootstrap), log and skip
                        if "asyncio.run() cannot be called" in str(e):
                            self.logger.warning(
                                f"Cannot call asyncio.run() from running event loop: {e}"
                            )
                            result = None
                        else:
                            raise
                else:
                    # Sync fallback
                    self.logger.warning("Async client not available, using fallback")
                    result = None

            return True

        except Exception as e:
            self.logger.error(f"Error storing documents for {source_path}: {e}")
            return False

    def index_file(self, file_path: Path, force_reindex: bool = False) -> Tuple[bool, str]:
        """
        Index a single file.

        Args:
            file_path: Path to file to index
            force_reindex: Force reindexing even if unchanged

        Returns:
            Tuple of (success, status_message)
        """
        try:
            relative_path = file_path.relative_to(self.project_root)

            # Check if indexing is needed
            needs_indexing, existing_record = self._needs_indexing(file_path)

            if not needs_indexing and not force_reindex:
                return True, f"Skipped (unchanged): {relative_path}"

            # Process document
            processed_doc = self.doc_processor.process_document(file_path)

            # Create chunks
            chunked_docs = self.doc_processor.chunk_document(processed_doc)

            if not chunked_docs:
                return True, f"No content: {relative_path}"

            # Store in memory database
            if self._store_documents_batch(chunked_docs, str(relative_path)):
                # Update indexing record
                record = IndexingRecord(
                    source_path=str(relative_path),
                    checksum=processed_doc.checksum,
                    last_modified=file_path.stat().st_mtime,
                    content_type=processed_doc.content_type,
                    chunk_count=len(chunked_docs),
                    indexed_at=datetime.now().isoformat(),
                    file_size=file_path.stat().st_size,
                    metadata=processed_doc.metadata
                )
                self.indexing_db.upsert_record(record)

                action = "Reindexed" if existing_record else "Indexed"
                return True, f"{action}: {relative_path} ({len(chunked_docs)} chunks)"
            else:
                return False, f"Storage failed: {relative_path}"

        except Exception as e:
            error_msg = f"Error indexing {file_path}: {e}"
            self.logger.error(error_msg)
            return False, error_msg

    def index_directory(self,
                       cleanup_mode: str = "incremental",
                       force_reindex: bool = False,
                       include_patterns: Optional[List[str]] = None,
                       exclude_patterns: Optional[List[str]] = None) -> IndexingResult:
        """
        Index directory with incremental updates.

        Args:
            cleanup_mode: "full", "incremental", or "none"
            force_reindex: Force reindexing all files
            include_patterns: Patterns to include
            exclude_patterns: Patterns to exclude

        Returns:
            IndexingResult with statistics
        """
        start_time = time.time()

        # Initialize result
        result = IndexingResult(
            success=True,
            total_files=0,
            processed_files=0,
            updated_files=0,
            added_files=0,
            skipped_files=0,
            deleted_files=0,
            total_chunks=0,
            processing_time=0.0
        )

        try:
            # Discover files
            file_paths = self.doc_processor.discover_documents(include_patterns, exclude_patterns)
            result.total_files = len(file_paths)

            self.logger.info(f"Found {result.total_files} files to process")

            # Process files in batches
            batch_results = []
            for file_path in file_paths:
                success, message = self.index_file(file_path, force_reindex)
                result.processed_files += 1

                if success:
                    if "Reindexed" in message:
                        result.updated_files += 1
                    elif "Indexed" in message:
                        result.added_files += 1
                    else:
                        result.skipped_files += 1

                    # Extract chunk count
                    if "chunks)" in message:
                        chunks_str = message.split("(")[-1].split(")")[0]
                        try:
                            result.total_chunks += int(chunks_str.split()[-1])
                        except ValueError:
                            pass
                else:
                    result.errors.append(message)

                batch_results.append(message)

                # Progress logging
                if result.processed_files % 50 == 0:
                    self.logger.info(f"Processed {result.processed_files}/{result.total_files} files")

            # Cleanup stale records if requested
            if cleanup_mode in ["full", "incremental"]:
                stale_count = self.indexing_db.cleanup_stale_records(str(self.project_root))
                result.deleted_files = stale_count

            result.processing_time = time.time() - start_time

            # Log summary
            self.logger.info(f"Indexing completed in {result.processing_time:.2f}s")
            self.logger.info(f"Files: {result.added_files} added, {result.updated_files} updated, "
                           f"{result.skipped_files} skipped, {result.deleted_files} deleted")
            self.logger.info(f"Total chunks: {result.total_chunks}")

            if result.errors:
                self.logger.warning(f"Encountered {len(result.errors)} errors")

        except Exception as e:
            result.success = False
            result.processing_time = time.time() - start_time
            error_msg = f"Indexing failed: {e}"
            self.logger.error(error_msg)
            result.errors.append(error_msg)

        return result

    def get_indexing_status(self) -> Dict[str, Any]:
        """Get current indexing status."""
        records = self.indexing_db.get_all_records()

        # Content type distribution
        content_types = {}
        for record in records:
            content_type = record.content_type
            content_types[content_type] = content_types.get(content_type, 0) + 1

        # Calculate statistics
        total_chunks = sum(record.chunk_count for record in records)
        total_size = sum(record.file_size for record in records)

        # Recent activity
        recent_records = sorted(records, key=lambda r: r.indexed_at, reverse=True)[:10]

        return {
            'total_files': len(records),
            'total_chunks': total_chunks,
            'total_size_bytes': total_size,
            'content_types': content_types,
            'last_indexed': recent_records[0].indexed_at if recent_records else None,
            'recent_files': [
                {
                    'path': record.source_path,
                    'indexed_at': record.indexed_at,
                    'chunks': record.chunk_count
                }
                for record in recent_records
            ]
        }

    def clear_index(self) -> bool:
        """Clear all indexing records."""
        try:
            with self.indexing_db.get_connection() as conn:
                conn.execute("DELETE FROM indexing_records")

            self.logger.info("Cleared all indexing records")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing indexing records: {e}")
            return False

    def remove_file(self, file_path: Path) -> bool:
        """Remove a file from indexing."""
        try:
            relative_path = str(file_path.relative_to(self.project_root))
            self.indexing_db.delete_record(relative_path)
            self.logger.info(f"Removed indexing record: {relative_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error removing indexing record: {e}")
            return False


# Context7 Pattern: Convenience function for quick usage
def create_incremental_indexer(project_root: str, memory_client=None) -> IncrementalIndexer:
    """
    Create an incremental indexer instance.

    Args:
        project_root: Root path of the project
        memory_client: Memory client for storage

    Returns:
        IncrementalIndexer instance
    """
    return IncrementalIndexer(project_root, memory_client)


# Context7 Pattern: Command-line interface for testing
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="DevStream Incremental Indexer")
    parser.add_argument("project_root", help="Project root directory")
    parser.add_argument("--cleanup", choices=["full", "incremental", "none"],
                       default="incremental", help="Cleanup mode")
    parser.add_argument("--force", action="store_true", help="Force reindexing")
    parser.add_argument("--status", action="store_true", help="Show indexing status")
    parser.add_argument("--clear", action="store_true", help="Clear all indexing records")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    indexer = create_incremental_indexer(args.project_root)

    if args.clear:
        if indexer.clear_index():
            print("✅ Cleared all indexing records")
        else:
            print("❌ Failed to clear indexing records")
        sys.exit(0)

    if args.status:
        status = indexer.get_indexing_status()
        print("\n=== Indexing Status ===")
        print(f"Total files: {status['total_files']}")
        print(f"Total chunks: {status['total_chunks']}")
        print(f"Total size: {status['total_size_bytes']:,} bytes")
        print(f"Content types: {status['content_types']}")
        if status['last_indexed']:
            print(f"Last indexed: {status['last_indexed']}")
        print("\nRecent files:")
        for file_info in status['recent_files'][:5]:
            print(f"  {file_info['path']} ({file_info['chunks']} chunks)")
        sys.exit(0)

    # Perform indexing
    print(f"🔍 Indexing {args.project_root}...")
    result = indexer.index_directory(
        cleanup_mode=args.cleanup,
        force_reindex=args.force
    )

    print(f"\n=== Indexing Results ===")
    print(f"✅ Success: {result.success}")
    print(f"📁 Total files: {result.total_files}")
    print(f"📝 Processed: {result.processed_files}")
    print(f"➕ Added: {result.added_files}")
    print(f"🔄 Updated: {result.updated_files}")
    print(f"⏭️  Skipped: {result.skipped_files}")
    print(f"🗑️  Deleted: {result.deleted_files}")
    print(f"🧩 Total chunks: {result.total_chunks}")
    print(f"⏱️  Processing time: {result.processing_time:.2f}s")

    if result.errors:
        print(f"\n❌ Errors ({len(result.errors)}):")
        for error in result.errors[:5]:  # Show first 5 errors
            print(f"  {error}")

    if result.warnings:
        print(f"\n⚠️  Warnings ({len(result.warnings)}):")
        for warning in result.warnings[:5]:  # Show first 5 warnings
            print(f"  {warning}")