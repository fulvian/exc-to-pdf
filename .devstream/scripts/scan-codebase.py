#!/usr/bin/env python3
"""
DevStream Codebase Scanner - Populate Semantic Memory with Embeddings

Scans codebase files (docs, source code, configs) and populates semantic_memory
with content and vector embeddings for semantic search.

Features:
- Multi-format file scanning (*.md, *.py, *.ts, *.tsx, *.rs, *.go, etc.)
- Keyword extraction (AST parsing for code, NLP for docs)
- Ollama embedding generation with batch processing
- Batch database insertion with automatic trigger sync
- Rich progress tracking with ETA
- Graceful error handling with retry logic

Usage:
    .devstream/bin/python scripts/scan-codebase.py
    .devstream/bin/python scripts/scan-codebase.py --docs-only
    .devstream/bin/python scripts/scan-codebase.py --code-only --batch-size 20
    .devstream/bin/python scripts/scan-codebase.py --skip-embeddings --verbose
"""

import sys
import asyncio
import sqlite3
import json
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Third-party imports
import aiohttp
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


# ============================================================================
# Constants and Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / 'data' / 'devstream.db'

# File patterns to scan
CODE_PATTERNS = {
    '*.py': 'python',
    '*.ts': 'typescript',
    '*.tsx': 'typescript',
    '*.js': 'javascript',
    '*.jsx': 'javascript',
    '*.rs': 'rust',
    '*.go': 'go',
    '*.java': 'java',
    '*.cpp': 'cpp',
    '*.c': 'c',
    '*.sh': 'bash',
    '*.sql': 'sql',
}

DOC_PATTERNS = {
    '*.md': 'markdown',
    '*.rst': 'restructuredtext',
    '*.txt': 'text',
}

CONFIG_PATTERNS = {
    '*.json': 'json',
    '*.yaml': 'yaml',
    '*.yml': 'yaml',
    '*.toml': 'toml',
    '*.ini': 'ini',
}

# Directories to scan
SCAN_DIRECTORIES = {
    'docs': ('documentation', DOC_PATTERNS),
    '.claude/agents': ('context', DOC_PATTERNS),
    '.claude/hooks': ('code', CODE_PATTERNS),
    'mcp-devstream-server': ('code', {**CODE_PATTERNS, **CONFIG_PATTERNS}),
    'scripts': ('code', CODE_PATTERNS),
}

# Directories to exclude
EXCLUDED_DIRS = {
    '.git',
    'node_modules',
    '.venv',
    '.devstream',
    '__pycache__',
    'dist',
    'build',
    '.pytest_cache',
    '.mypy_cache',
    'data',  # Don't scan database files
}

# File size limits
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# ============================================================================
# Data Classes
# ============================================================================

class ContentType(Enum):
    """Content type enum for semantic memory."""
    CODE = "code"
    DOCUMENTATION = "documentation"
    CONTEXT = "context"


@dataclass
class ScannedFile:
    """Represents a scanned file with extracted metadata."""
    file_path: str
    content: str
    content_type: ContentType
    keywords: List[str]
    language: Optional[str] = None
    file_hash: Optional[str] = None

    def __post_init__(self):
        """Generate file hash for deduplication."""
        if self.file_hash is None:
            self.file_hash = hashlib.sha256(self.content.encode('utf-8')).hexdigest()[:16]


@dataclass
class MemoryRecord:
    """Represents a semantic memory record ready for insertion."""
    id: str
    content: str
    content_type: str
    keywords: List[str]
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    embedding_dimension: Optional[int] = None


# ============================================================================
# File Scanner
# ============================================================================

class FileScanner:
    """
    Scans codebase directories and extracts file contents.

    Handles file filtering, encoding detection, and content extraction.
    """

    def __init__(
        self,
        console: Console,
        docs_only: bool = False,
        code_only: bool = False,
        verbose: bool = False
    ):
        """
        Initialize file scanner.

        Args:
            console: Rich console for output
            docs_only: Scan only documentation files
            code_only: Scan only code files
            verbose: Enable verbose logging
        """
        self.console = console
        self.docs_only = docs_only
        self.code_only = code_only
        self.verbose = verbose

    def should_scan_file(self, file_path: Path) -> bool:
        """
        Check if file should be scanned based on filters.

        Args:
            file_path: Path to file

        Returns:
            True if file should be scanned, False otherwise
        """
        # Check excluded directories
        if any(excluded in file_path.parts for excluded in EXCLUDED_DIRS):
            return False

        # Check file size
        try:
            if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                if self.verbose:
                    self.console.print(
                        f"⚠️  Skipping large file: {file_path} "
                        f"({file_path.stat().st_size / 1024 / 1024:.1f}MB)",
                        style="yellow"
                    )
                return False
        except OSError:
            return False

        # Apply content type filters
        suffix = file_path.suffix.lower()

        if self.docs_only:
            return suffix in [p.replace('*', '') for p in DOC_PATTERNS.keys()]
        elif self.code_only:
            return suffix in [p.replace('*', '') for p in CODE_PATTERNS.keys()]

        return True

    def read_file_content(self, file_path: Path) -> Optional[str]:
        """
        Read file content with encoding detection.

        Args:
            file_path: Path to file

        Returns:
            File content as string, or None on error
        """
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    if self.verbose:
                        self.console.print(
                            f"✓ Read {file_path} ({len(content)} chars, {encoding})",
                            style="dim"
                        )
                    return content
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                if self.verbose:
                    self.console.print(
                        f"❌ Error reading {file_path}: {e}",
                        style="red"
                    )
                return None

        if self.verbose:
            self.console.print(
                f"❌ Could not decode {file_path} with any encoding",
                style="red"
            )
        return None

    def scan_directory(
        self,
        directory: Path,
        content_type: ContentType,
        patterns: Dict[str, str]
    ) -> List[ScannedFile]:
        """
        Scan directory for files matching patterns.

        Args:
            directory: Directory to scan
            content_type: Content type for scanned files
            patterns: File patterns to match

        Returns:
            List of scanned files
        """
        if not directory.exists():
            if self.verbose:
                self.console.print(
                    f"⚠️  Directory not found: {directory}",
                    style="yellow"
                )
            return []

        scanned_files: List[ScannedFile] = []

        for pattern, language in patterns.items():
            for file_path in directory.rglob(pattern):
                if not self.should_scan_file(file_path):
                    continue

                content = self.read_file_content(file_path)
                if content is None:
                    continue

                # Create ScannedFile (keywords will be extracted later)
                scanned_file = ScannedFile(
                    file_path=str(file_path.relative_to(PROJECT_ROOT)),
                    content=content,
                    content_type=content_type,
                    keywords=[],  # Extracted by KeywordExtractor
                    language=language
                )

                scanned_files.append(scanned_file)

        return scanned_files

    def scan_all(self) -> List[ScannedFile]:
        """
        Scan all configured directories.

        Returns:
            List of all scanned files
        """
        all_files: List[ScannedFile] = []

        for dir_name, (content_type_str, patterns) in SCAN_DIRECTORIES.items():
            directory = PROJECT_ROOT / dir_name
            content_type = ContentType(content_type_str)

            if self.verbose:
                self.console.print(f"\n📁 Scanning {directory}...", style="cyan")

            files = self.scan_directory(directory, content_type, patterns)
            all_files.extend(files)

            if self.verbose:
                self.console.print(
                    f"   Found {len(files)} files",
                    style="green" if files else "dim"
                )

        return all_files


# ============================================================================
# Keyword Extractor
# ============================================================================

class KeywordExtractor:
    """
    Extracts keywords from file content using AST parsing and NLP.

    For code files: Extracts function/class names, imports, identifiers
    For docs: Extracts significant words using frequency analysis
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize keyword extractor.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose

    def extract_python_keywords(self, content: str) -> List[str]:
        """
        Extract keywords from Python code using AST.

        Args:
            content: Python source code

        Returns:
            List of keywords (function names, class names, imports)
        """
        keywords: Set[str] = set()

        try:
            import ast

            tree = ast.parse(content)

            for node in ast.walk(tree):
                # Function definitions
                if isinstance(node, ast.FunctionDef):
                    keywords.add(node.name)
                # Class definitions
                elif isinstance(node, ast.ClassDef):
                    keywords.add(node.name)
                # Import statements
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        keywords.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        keywords.add(node.module.split('.')[0])

        except Exception:
            # Fallback to regex if AST parsing fails
            return self.extract_generic_keywords(content)

        return list(keywords)

    def extract_generic_keywords(self, content: str) -> List[str]:
        """
        Extract keywords using simple regex patterns.

        Args:
            content: File content

        Returns:
            List of keywords extracted via pattern matching
        """
        import re

        keywords: Set[str] = set()

        # Extract CamelCase identifiers
        camel_case = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', content)
        keywords.update(camel_case)

        # Extract snake_case identifiers (function-like)
        snake_case = re.findall(r'\b[a-z_][a-z0-9_]{2,}\b', content)
        keywords.update(snake_case[:10])  # Limit to top 10

        return list(keywords)

    def extract_doc_keywords(self, content: str) -> List[str]:
        """
        Extract keywords from documentation using word frequency.

        Args:
            content: Documentation content

        Returns:
            List of significant keywords
        """
        import re
        from collections import Counter

        # Remove markdown formatting
        content = re.sub(r'[#*`\[\]()]', ' ', content)

        # Extract words (min 4 chars)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())

        # Common stop words to exclude
        stop_words = {
            'this', 'that', 'with', 'from', 'have', 'will', 'your',
            'more', 'when', 'them', 'some', 'what', 'their', 'which',
            'about', 'into', 'than', 'only', 'also', 'other', 'such',
            'should', 'would', 'could', 'these', 'those'
        }

        # Filter and count
        filtered_words = [w for w in words if w not in stop_words]
        word_counts = Counter(filtered_words)

        # Return top 10 most common words
        return [word for word, count in word_counts.most_common(10)]

    def extract_keywords(self, scanned_file: ScannedFile) -> List[str]:
        """
        Extract keywords from scanned file based on language.

        Args:
            scanned_file: Scanned file to extract keywords from

        Returns:
            List of extracted keywords
        """
        keywords: Set[str] = set()

        # Add file name
        file_name = Path(scanned_file.file_path).stem
        keywords.add(file_name)

        # Add parent directory
        parent = Path(scanned_file.file_path).parent.name
        if parent and parent not in ['.', '..']:
            keywords.add(parent)

        # Add language
        if scanned_file.language:
            keywords.add(scanned_file.language)

        # Extract content-based keywords
        if scanned_file.language == 'python':
            keywords.update(self.extract_python_keywords(scanned_file.content))
        elif scanned_file.content_type == ContentType.DOCUMENTATION:
            keywords.update(self.extract_doc_keywords(scanned_file.content))
        else:
            keywords.update(self.extract_generic_keywords(scanned_file.content))

        # Add content type tag
        keywords.add(scanned_file.content_type.value)

        return list(keywords)[:15]  # Limit to 15 keywords


# ============================================================================
# Embedding Generator
# ============================================================================

class EmbeddingGenerator:
    """
    Generates embeddings using Ollama with batch processing and retry logic.

    Features:
    - Batch processing (configurable batch size)
    - Exponential backoff retry (3 attempts)
    - Graceful degradation on Ollama failure
    """

    def __init__(
        self,
        console: Console,
        batch_size: int = 10,
        skip_embeddings: bool = False,
        verbose: bool = False
    ):
        """
        Initialize embedding generator.

        Args:
            console: Rich console for output
            batch_size: Batch size for embedding requests
            skip_embeddings: Skip embedding generation (testing mode)
            verbose: Enable verbose logging
        """
        self.console = console
        self.batch_size = batch_size
        self.skip_embeddings = skip_embeddings
        self.verbose = verbose

        self.ollama_client = OllamaEmbeddingClient()
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0
        }

    async def test_connection(self) -> bool:
        """
        Test Ollama connection before batch processing.

        Returns:
            True if Ollama is available, False otherwise
        """
        if self.skip_embeddings:
            return True

        return self.ollama_client.test_connection()

    async def generate_batch(
        self,
        texts: List[str],
        retry_count: int = 3
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for batch of texts with retry logic.

        Args:
            texts: List of texts to generate embeddings for
            retry_count: Number of retry attempts on failure

        Returns:
            List of embeddings (or None for failures)
        """
        if self.skip_embeddings:
            self.stats['skipped'] += len(texts)
            return [None] * len(texts)

        for attempt in range(retry_count):
            try:
                embeddings = self.ollama_client.generate_embeddings_batch(
                    texts,
                    batch_size=self.batch_size
                )

                # Count successes and failures
                success_count = sum(1 for e in embeddings if e is not None)
                failed_count = len(embeddings) - success_count

                self.stats['total'] += len(texts)
                self.stats['success'] += success_count
                self.stats['failed'] += failed_count

                if self.verbose and failed_count > 0:
                    self.console.print(
                        f"⚠️  {failed_count} embeddings failed in batch",
                        style="yellow"
                    )

                return embeddings

            except Exception as e:
                if attempt < retry_count - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    if self.verbose:
                        self.console.print(
                            f"⚠️  Batch failed (attempt {attempt + 1}/{retry_count}), "
                            f"retrying in {wait_time}s: {e}",
                            style="yellow"
                        )
                    await asyncio.sleep(wait_time)
                else:
                    if self.verbose:
                        self.console.print(
                            f"❌ Batch failed after {retry_count} attempts: {e}",
                            style="red"
                        )
                    self.stats['total'] += len(texts)
                    self.stats['failed'] += len(texts)
                    return [None] * len(texts)

        return [None] * len(texts)

    async def generate_all(
        self,
        scanned_files: List[ScannedFile],
        progress: Progress,
        task_id: Any
    ) -> List[MemoryRecord]:
        """
        Generate embeddings for all scanned files with progress tracking.

        Args:
            scanned_files: List of scanned files
            progress: Rich progress bar
            task_id: Progress task ID

        Returns:
            List of memory records with embeddings
        """
        memory_records: List[MemoryRecord] = []

        # Process in batches
        for i in range(0, len(scanned_files), self.batch_size):
            batch = scanned_files[i:i + self.batch_size]

            # Extract texts for embedding
            texts = [f.content[:5000] for f in batch]  # Limit to 5000 chars

            # Generate embeddings
            embeddings = await self.generate_batch(texts)

            # Create memory records
            for scanned_file, embedding in zip(batch, embeddings):
                memory_id = f"MEM-{scanned_file.file_hash}"

                memory_record = MemoryRecord(
                    id=memory_id,
                    content=scanned_file.content,
                    content_type=scanned_file.content_type.value,
                    keywords=scanned_file.keywords,
                    embedding=embedding,
                    embedding_model="embeddinggemma:300m" if embedding else None,
                    embedding_dimension=len(embedding) if embedding else None
                )

                memory_records.append(memory_record)

            # Update progress
            progress.update(task_id, advance=len(batch))

        return memory_records


# ============================================================================
# Database Inserter
# ============================================================================

class DatabaseInserter:
    """
    Inserts memory records into semantic_memory table with batch processing.

    Triggers automatically sync to vec_semantic_memory and fts_semantic_memory.
    """

    def __init__(self, console: Console, verbose: bool = False):
        """
        Initialize database inserter.

        Args:
            console: Rich console for output
            verbose: Enable verbose logging
        """
        self.console = console
        self.verbose = verbose

        self.stats = {
            'inserted': 0,
            'updated': 0,
            'failed': 0
        }

    def insert_batch(
        self,
        memory_records: List[MemoryRecord]
    ) -> int:
        """
        Insert batch of memory records into database.

        Args:
            memory_records: List of memory records to insert

        Returns:
            Number of records successfully inserted

        Note:
            Database triggers automatically sync to vec_semantic_memory
            and fts_semantic_memory virtual tables.
        """
        try:
            conn = get_db_connection_with_vec(str(DB_PATH))
            cursor = conn.cursor()

            inserted_count = 0

            for record in memory_records:
                try:
                    # Convert embedding to JSON string for storage
                    embedding_json = (
                        json.dumps(record.embedding)
                        if record.embedding
                        else None
                    )

                    # Check if record exists
                    cursor.execute(
                        "SELECT id FROM semantic_memory WHERE id = ?",
                        (record.id,)
                    )
                    exists = cursor.fetchone()

                    if exists:
                        # Update existing record
                        cursor.execute(
                            """
                            UPDATE semantic_memory
                            SET content = ?,
                                content_type = ?,
                                keywords = ?,
                                embedding = ?,
                                embedding_model = ?,
                                embedding_dimension = ?,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                            """,
                            (
                                record.content,
                                record.content_type,
                                json.dumps(record.keywords),
                                embedding_json,
                                record.embedding_model,
                                record.embedding_dimension,
                                record.id
                            )
                        )
                        self.stats['updated'] += 1
                    else:
                        # Insert new record
                        cursor.execute(
                            """
                            INSERT INTO semantic_memory (
                                id, content, content_type, keywords,
                                embedding, embedding_model, embedding_dimension,
                                created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                            """,
                            (
                                record.id,
                                record.content,
                                record.content_type,
                                json.dumps(record.keywords),
                                embedding_json,
                                record.embedding_model,
                                record.embedding_dimension
                            )
                        )
                        self.stats['inserted'] += 1

                    inserted_count += 1

                except Exception as e:
                    if self.verbose:
                        self.console.print(
                            f"❌ Failed to insert {record.id}: {e}",
                            style="red"
                        )
                    self.stats['failed'] += 1

            conn.commit()
            conn.close()

            return inserted_count

        except Exception as e:
            if self.verbose:
                self.console.print(
                    f"❌ Database batch insert failed: {e}",
                    style="red"
                )
            return 0


# ============================================================================
# Main Orchestration
# ============================================================================

async def main(
    docs_only: bool = False,
    code_only: bool = False,
    skip_embeddings: bool = False,
    batch_size: int = 10,
    verbose: bool = False
) -> int:
    """
    Main async orchestration function.

    Args:
        docs_only: Scan only documentation files
        code_only: Scan only code files
        skip_embeddings: Skip Ollama embedding generation
        batch_size: Batch size for embedding requests
        verbose: Enable verbose logging

    Returns:
        Exit code (0=success, 1=partial, 2=failure)
    """
    console = Console()

    # Print header
    console.print(Panel.fit(
        "🔍 DevStream Codebase Scanner\n"
        "Populate Semantic Memory with Embeddings",
        style="bold blue"
    ))

    # Initialize components
    file_scanner = FileScanner(console, docs_only, code_only, verbose)
    keyword_extractor = KeywordExtractor(verbose)
    embedding_generator = EmbeddingGenerator(console, batch_size, skip_embeddings, verbose)
    database_inserter = DatabaseInserter(console, verbose)

    # Step 1: Test Ollama connection
    if not skip_embeddings:
        console.print("\n🤖 Testing Ollama connection...", style="cyan")
        if not await embedding_generator.test_connection():
            console.print(
                "❌ Ollama connection failed. Use --skip-embeddings to continue without embeddings.",
                style="red"
            )
            return 2
        console.print("✅ Ollama connected", style="green")

    # Step 2: Scan files
    console.print("\n📁 Scanning files...", style="cyan")
    scanned_files = file_scanner.scan_all()

    if not scanned_files:
        console.print("❌ No files found to scan", style="red")
        return 2

    console.print(f"✅ Found {len(scanned_files)} files", style="green")

    # Step 3: Extract keywords
    console.print("\n🔑 Extracting keywords...", style="cyan")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            "Extracting keywords...",
            total=len(scanned_files)
        )

        for scanned_file in scanned_files:
            scanned_file.keywords = keyword_extractor.extract_keywords(scanned_file)
            progress.update(task, advance=1)

    console.print("✅ Keywords extracted", style="green")

    # Step 4: Generate embeddings
    console.print("\n🧠 Generating embeddings...", style="cyan")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            "Generating embeddings...",
            total=len(scanned_files)
        )

        memory_records = await embedding_generator.generate_all(
            scanned_files,
            progress,
            task
        )

    if skip_embeddings:
        console.print("⚠️  Embeddings skipped (testing mode)", style="yellow")
    else:
        console.print(
            f"✅ Embeddings generated: "
            f"{embedding_generator.stats['success']}/{embedding_generator.stats['total']}",
            style="green"
        )

    # Step 5: Insert into database
    console.print("\n💾 Inserting into database...", style="cyan")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Inserting records...", total=None)

        inserted_count = database_inserter.insert_batch(memory_records)

        progress.update(task, description=f"✅ Inserted {inserted_count} records")

    console.print(
        f"✅ Database updated: "
        f"{database_inserter.stats['inserted']} inserted, "
        f"{database_inserter.stats['updated']} updated",
        style="green"
    )

    # Step 6: Display summary
    console.print("\n📊 Summary", style="bold cyan")

    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Count", style="white", justify="right")

    summary_table.add_row("Files Scanned", str(len(scanned_files)))
    summary_table.add_row("Memories Created", str(len(memory_records)))
    summary_table.add_row(
        "Embeddings Generated",
        str(embedding_generator.stats['success'])
    )
    summary_table.add_row(
        "Database Inserts",
        str(database_inserter.stats['inserted'])
    )
    summary_table.add_row(
        "Database Updates",
        str(database_inserter.stats['updated'])
    )

    if embedding_generator.stats['failed'] > 0:
        summary_table.add_row(
            "Embedding Failures",
            str(embedding_generator.stats['failed']),
            style="yellow"
        )

    if database_inserter.stats['failed'] > 0:
        summary_table.add_row(
            "Database Failures",
            str(database_inserter.stats['failed']),
            style="red"
        )

    console.print(summary_table)

    # Determine exit code
    if database_inserter.stats['failed'] > 0 or embedding_generator.stats['failed'] > len(scanned_files) * 0.5:
        console.print("\n⚠️  Scan completed with errors", style="yellow")
        return 1
    else:
        console.print("\n✅ Scan completed successfully", style="green")
        return 0


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DevStream Codebase Scanner - Populate Semantic Memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan all files
  .devstream/bin/python scripts/scan-codebase.py

  # Scan only documentation
  .devstream/bin/python scripts/scan-codebase.py --docs-only

  # Scan only code files
  .devstream/bin/python scripts/scan-codebase.py --code-only

  # Skip embeddings (testing)
  .devstream/bin/python scripts/scan-codebase.py --skip-embeddings

  # Custom batch size with verbose output
  .devstream/bin/python scripts/scan-codebase.py --batch-size 20 --verbose
        """
    )

    parser.add_argument(
        '--docs-only',
        action='store_true',
        help='Scan only docs/ directory'
    )

    parser.add_argument(
        '--code-only',
        action='store_true',
        help='Scan only source files (*.py, *.ts, etc.)'
    )

    parser.add_argument(
        '--skip-embeddings',
        action='store_true',
        help='Skip Ollama embedding generation (testing mode)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Batch size for embedding requests (default: 10)'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.docs_only and args.code_only:
        print("❌ Error: --docs-only and --code-only are mutually exclusive", file=sys.stderr)
        sys.exit(2)

    # Run async main
    exit_code = asyncio.run(main(
        docs_only=args.docs_only,
        code_only=args.code_only,
        skip_embeddings=args.skip_embeddings,
        batch_size=args.batch_size,
        verbose=args.verbose
    ))

    sys.exit(exit_code)
