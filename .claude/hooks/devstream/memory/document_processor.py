#!/usr/bin/env python3
"""
DevStream Memory Bootstrap - Document Processor
Context7-compliant document processing using LangChain patterns.

This module provides intelligent document loading, chunking, and processing
capabilities for project memory initialization using best practices from
LangChain, ChromaDB, and modern RAG systems.

Key Features:
- Multi-format document loaders (Python, Markdown, Text, etc.)
- Intelligent chunking with overlap for context preservation
- Metadata extraction and enrichment
- Batch processing for performance optimization
- Source-based deduplication support
Context7 Pattern: Lazy loading + batch processing + metadata enrichment
"""

import os
import ast
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Tuple
from dataclasses import dataclass, field
import logging
import time
import json

# Context7 Pattern: Graceful dependency handling
try:
    from langchain_core.documents import Document
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        Language,
        PythonCodeTextSplitter
    )
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Context7 Fallback: Define minimal Document class
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available, using fallback document processing")

    @dataclass
    class Document:
        page_content: str
        metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedDocument:
    """Enhanced document with processing metadata."""
    content: str
    metadata: Dict[str, Any]
    source_path: str
    content_type: str
    chunk_count: int = 0
    processing_time: float = 0.0
    checksum: str = ""


class DocumentProcessor:
    """
    Context7-compliant document processor for memory bootstrap.

    Implements LangChain patterns for:
    - Multi-format document loading
    - Intelligent chunking with overlap
    - Metadata extraction and enrichment
    - Batch processing for performance
    - Source-based deduplication
    """

    def __init__(self, project_root: str, batch_size: int = 50):
        """
        Initialize document processor.

        Args:
            project_root: Root path of the project to process
            batch_size: Number of documents to process in each batch
        """
        self.project_root = Path(project_root)
        self.batch_size = batch_size

        # Context7 Pattern: Language-specific splitters
        if LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )

            self.code_splitter = PythonCodeTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                length_function=len
            )
        else:
            # Fallback splitter
            self.text_splitter = self._fallback_splitter

        # Supported file types and their processors
        self.file_processors = {
            '.py': self._process_python_file,
            '.md': self._process_markdown_file,
            '.rst': self._process_rst_file,
            '.txt': self._process_text_file,
            '.yml': self._process_yaml_file,
            '.yaml': self._process_yaml_file,
            '.json': self._process_json_file,
            '.toml': self._process_toml_file,
            '.cfg': self._process_config_file,
            '.ini': self._process_config_file,
        }

        # Content type mapping
        self.content_type_mapping = {
            '.py': 'code',
            '.md': 'documentation',
            '.rst': 'documentation',
            '.txt': 'text',
            '.yml': 'configuration',
            '.yaml': 'configuration',
            '.json': 'configuration',
            '.toml': 'configuration',
            '.cfg': 'configuration',
            '.ini': 'configuration',
        }

        self.logger = logging.getLogger(__name__)

    def _fallback_splitter(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """
        Fallback text splitter when LangChain is not available.

        Context7 Pattern: Simple but effective chunking strategy.
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + chunk_size

            # Try to break at word boundary
            if end < text_len:
                last_space = text.rfind(' ', start, end)
                if last_space != -1:
                    end = last_space

            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            start = end - chunk_overlap

            # Avoid infinite loops
            if len(chunks) > 0 and start <= chunks[-1].find(' '):
                start = end

        return chunks

    def _calculate_checksum(self, content: str) -> str:
        """Calculate SHA-256 checksum for content deduplication."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

    def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract rich metadata from file path and content.

        Context7 Pattern: Multi-dimensional metadata for better search.
        """
        relative_path = file_path.relative_to(self.project_root)

        metadata = {
            'source': str(relative_path),
            'filename': file_path.name,
            'extension': file_path.suffix,
            'directory': str(relative_path.parent),
            'size_bytes': file_path.stat().st_size if file_path.exists() else 0,
            'modified_time': file_path.stat().st_mtime if file_path.exists() else 0,
            'project_root': str(self.project_root),
        }

        # Add directory-based context
        if 'src' in str(relative_path):
            metadata['context_type'] = 'source_code'
        elif 'docs' in str(relative_path):
            metadata['context_type'] = 'documentation'
        elif 'test' in str(relative_path):
            metadata['context_type'] = 'test'
        elif 'config' in str(relative_path):
            metadata['context_type'] = 'configuration'
        else:
            metadata['context_type'] = 'general'

        return metadata

    def _process_python_file(self, file_path: Path) -> ProcessedDocument:
        """
        Process Python file with AST analysis.

        Context7 Pattern: Structural analysis for code understanding.
        """
        start_time = time.time()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse AST for structural information
            try:
                tree = ast.parse(content)

                # Extract structural information
                functions = []
                classes = []
                imports = []

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        functions.append({
                            'name': node.name,
                            'line_start': node.lineno,
                            'line_end': getattr(node, 'end_lineno', node.lineno),
                            'args': [arg.arg for arg in node.args.args]
                        })
                    elif isinstance(node, ast.ClassDef):
                        classes.append({
                            'name': node.name,
                            'line_start': node.lineno,
                            'line_end': getattr(node, 'end_lineno', node.lineno),
                            'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                        })
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        if isinstance(node, ast.Import):
                            imports.extend([alias.name for alias in node.names])
                        else:
                            imports.append(f"from {node.module}" if node.module else "from .")

                # Enhanced metadata with AST information
                metadata = self._extract_metadata(file_path)
                metadata.update({
                    'language': 'python',
                    'functions': functions,
                    'classes': classes,
                    'imports': imports,
                    'complexity_score': len(functions) + len(classes) * 2,
                    'content_type': 'code'
                })

            except SyntaxError:
                # Fallback for files with syntax errors
                metadata = self._extract_metadata(file_path)
                metadata.update({
                    'language': 'python',
                    'syntax_error': True,
                    'content_type': 'code'
                })

            # Calculate checksum
            checksum = self._calculate_checksum(content)

            processing_time = time.time() - start_time

            return ProcessedDocument(
                content=content,
                metadata=metadata,
                source_path=str(file_path),
                content_type='code',
                processing_time=processing_time,
                checksum=checksum
            )

        except Exception as e:
            self.logger.error(f"Error processing Python file {file_path}: {e}")
            raise

    def _process_markdown_file(self, file_path: Path) -> ProcessedDocument:
        """
        Process Markdown file with structure extraction.

        Context7 Pattern: Document structure analysis for better chunking.
        """
        start_time = time.time()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract markdown structure
            lines = content.split('\n')
            headers = []
            sections = []
            current_section = None

            for line_num, line in enumerate(lines, 1):
                if line.startswith('#'):
                    level = len(line) - len(line.lstrip('#'))
                    title = line.lstrip('# ').strip()

                    header_info = {
                        'level': level,
                        'title': title,
                        'line': line_num
                    }
                    headers.append(header_info)

                    # Track sections
                    if current_section:
                        current_section['end_line'] = line_num - 1
                        sections.append(current_section)

                    current_section = {
                        'title': title,
                        'level': level,
                        'start_line': line_num,
                        'content': ''
                    }
                elif current_section:
                    current_section['content'] += line + '\n'

            # Add final section
            if current_section:
                sections.append(current_section)

            # Enhanced metadata
            metadata = self._extract_metadata(file_path)
            metadata.update({
                'format': 'markdown',
                'headers': headers,
                'sections': sections,
                'header_count': len(headers),
                'content_type': 'documentation'
            })

            checksum = self._calculate_checksum(content)
            processing_time = time.time() - start_time

            return ProcessedDocument(
                content=content,
                metadata=metadata,
                source_path=str(file_path),
                content_type='documentation',
                processing_time=processing_time,
                checksum=checksum
            )

        except Exception as e:
            self.logger.error(f"Error processing Markdown file {file_path}: {e}")
            raise

    def _process_text_file(self, file_path: Path) -> ProcessedDocument:
        """Process plain text file."""
        start_time = time.time()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            metadata = self._extract_metadata(file_path)
            metadata.update({
                'format': 'text',
                'content_type': 'text'
            })

            checksum = self._calculate_checksum(content)
            processing_time = time.time() - start_time

            return ProcessedDocument(
                content=content,
                metadata=metadata,
                source_path=str(file_path),
                content_type='text',
                processing_time=processing_time,
                checksum=checksum
            )

        except Exception as e:
            self.logger.error(f"Error processing text file {file_path}: {e}")
            raise

    def _process_rst_file(self, file_path: Path) -> ProcessedDocument:
        """Process reStructuredText file."""
        # Similar to markdown but with RST-specific parsing
        return self._process_markdown_file(file_path)

    def _process_yaml_file(self, file_path: Path) -> ProcessedDocument:
        """Process YAML configuration file."""
        return self._process_text_file(file_path)

    def _process_json_file(self, file_path: Path) -> ProcessedDocument:
        """Process JSON configuration file."""
        return self._process_text_file(file_path)

    def _process_toml_file(self, file_path: Path) -> ProcessedDocument:
        """Process TOML configuration file."""
        return self._process_text_file(file_path)

    def _process_config_file(self, file_path: Path) -> ProcessedDocument:
        """Process generic configuration file."""
        return self._process_text_file(file_path)

    def discover_documents(self, include_patterns: Optional[List[str]] = None,
                          exclude_patterns: Optional[List[str]] = None) -> List[Path]:
        """
        Discover documents in project directory.

        Context7 Pattern: Intelligent file discovery with patterns.
        Excludes virtualenvs, build artifacts, and media while preserving hooks.
        """
        if include_patterns is None:
            include_patterns = ['**/*']

        if exclude_patterns is None:
            # Context7 Pattern: Comprehensive exclusion list
            # Based on gitignore_parser best practices (Trust Score 8.9)
            exclude_patterns = [
                # Version control
                '**/.git/**',
                # Python artifacts
                '**/__pycache__/**',
                '**/.pytest_cache/**',
                '**/.tox/**',
                '**/.mypy_cache/**',
                '**/.ruff_cache/**',
                '**/htmlcov/**',
                '**/.eggs/**',
                '**/*.egg-info/**',
                '**/*.pyc',
                '**/*.pyo',
                '**/*.pyd',
                # Build artifacts
                '**/dist/**',
                '**/build/**',
                # Virtual environments (CRITICAL: includes .devstream)
                '**/.venv/**',
                '**/venv/**',
                '**/.devstream/**',
                '**/.reporting/**',
                '**/reporting/**',
                # Node.js
                '**/node_modules/**',
                # Media files
                '**/registrazioni/**',
                # OS artifacts
                '**/.DS_Store',
                '**/.DS_Store/**',
                '**/*.swp',
                '**/*.swo'
            ]

        # Store exclude patterns for use in should_exclude_file
        self.exclude_patterns = exclude_patterns
        discovered_files = []

        for pattern in include_patterns:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file() and file_path.suffix in self.file_processors:
                    # Check exclude patterns using robust method
                    if not self.should_exclude_file(file_path):
                        discovered_files.append(file_path)

        # Sort by modification time (newest first)
        discovered_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        self.logger.info(f"Discovered {len(discovered_files)} files to process")
        return discovered_files

    def should_exclude_file(self, file_path: Path) -> bool:
        """
        Check if file should be excluded using multiple strategies.

        Context7 Pattern: Robust exclusion with path part inspection.
        Strategy based on gitignore_parser and pathspec best practices.

        Performance: O(n) where n=path depth, suitable for large file sets.
        Rationale: Path.parts checking is more reliable than Path.match()
        for directory exclusions (python-pathspec Trust Score 7.1).
        """
        try:
            relative_path = file_path.relative_to(self.project_root)
        except ValueError:
            # File is outside project root
            return True

        relative_str = str(relative_path)

        # Context7 Pattern: Comprehensive directory exclusion list
        # Includes: VCS, build artifacts, virtualenvs, media, caches
        exclude_dirs = {
            '.git', '__pycache__', 'node_modules',
            '.venv', 'venv', '.devstream', '.reporting', 'reporting',
            '.pytest_cache', 'dist', 'build', 'registrazioni',
            '.tox', '.mypy_cache', '.ruff_cache', 'htmlcov',
            '.eggs', '*.egg-info', '.DS_Store'
        }

        # Context7 Pattern: Path part inspection (primary strategy)
        # Most reliable for directory-based exclusions
        for part in relative_path.parts:
            if part in exclude_dirs:
                return True
            # Handle .egg-info and similar patterns
            if part.endswith('.egg-info') or part.endswith('.dist-info'):
                return True

        # Context7 Pattern: File extension exclusion (compiled artifacts)
        exclude_extensions = {'.pyc', '.pyo', '.pyd', '.so', '.dylib', '.swp', '.swo'}
        if relative_path.suffix in exclude_extensions:
            return True

        # Context7 Pattern: Pattern matching (fallback for glob patterns)
        # Only used for patterns not covered by part inspection
        if hasattr(self, 'exclude_patterns'):
            for pattern in self.exclude_patterns:
                try:
                    # Use pathlib.match for gitignore-style patterns
                    if relative_path.match(pattern):
                        return True
                except (ValueError, Exception):
                    # Fallback: string matching for malformed patterns
                    if pattern.strip('*').strip('/') in relative_str:
                        return True

        return False

    def process_document(self, file_path: Path) -> ProcessedDocument:
        """
        Process a single document.

        Context7 Pattern: Type-specific processing with metadata enrichment.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = file_path.suffix.lower()
        processor = self.file_processors.get(file_ext)

        if not processor:
            raise ValueError(f"Unsupported file type: {file_ext}")

        return processor(file_path)

    def process_documents_batch(self, file_paths: List[Path]) -> List[ProcessedDocument]:
        """
        Process a batch of documents.

        Context7 Pattern: Batch processing for performance optimization.
        """
        processed_docs = []

        for file_path in file_paths:
            try:
                doc = self.process_document(file_path)
                processed_docs.append(doc)
            except Exception as e:
                self.logger.warning(f"Failed to process {file_path}: {e}")
                continue

        return processed_docs

    def chunk_document(self, processed_doc: ProcessedDocument) -> List[Document]:
        """
        Split processed document into chunks.

        Context7 Pattern: Intelligent chunking with metadata preservation.
        """
        content_type = processed_doc.content_type

        # Choose appropriate splitter
        if content_type == 'code' and LANGCHAIN_AVAILABLE:
            chunks = self.code_splitter.split_text(processed_doc.content)
        else:
            chunks = self.text_splitter.split_text(processed_doc.content)

        # Create Document objects with enriched metadata
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = processed_doc.metadata.copy()
            chunk_metadata.update({
                'chunk_id': f"{processed_doc.source_path}:{i}",
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_size': len(chunk),
                'document_checksum': processed_doc.checksum
            })

            if LANGCHAIN_AVAILABLE:
                documents.append(Document(page_content=chunk, metadata=chunk_metadata))
            else:
                documents.append(Document(chunk, chunk_metadata))

        processed_doc.chunk_count = len(chunks)
        return documents

    def lazy_process_documents(self, file_paths: List[Path]) -> Iterator[List[Document]]:
        """
        Lazily process documents in batches.

        Context7 Pattern: Memory-efficient lazy processing.
        """
        batch = []

        for file_path in file_paths:
            try:
                # Process single document
                processed_doc = self.process_document(file_path)

                # Chunk the document
                chunked_docs = self.chunk_document(processed_doc)
                batch.extend(chunked_docs)

                # Yield batch when it reaches the target size
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []

            except Exception as e:
                self.logger.warning(f"Failed to process {file_path}: {e}")
                continue

        # Yield remaining documents
        if batch:
            yield batch

    def get_processing_stats(self, processed_docs: List[ProcessedDocument]) -> Dict[str, Any]:
        """
        Get processing statistics.

        Context7 Pattern: Analytics for monitoring and optimization.
        """
        if not processed_docs:
            return {}

        total_files = len(processed_docs)
        total_chunks = sum(doc.chunk_count for doc in processed_docs)
        total_processing_time = sum(doc.processing_time for doc in processed_docs)

        # Content type distribution
        content_types = {}
        for doc in processed_docs:
            content_type = doc.content_type
            content_types[content_type] = content_types.get(content_type, 0) + 1

        # File type distribution
        file_types = {}
        for doc in processed_docs:
            ext = Path(doc.source_path).suffix
            file_types[ext] = file_types.get(ext, 0) + 1

        return {
            'total_files': total_files,
            'total_chunks': total_chunks,
            'total_processing_time': total_processing_time,
            'avg_processing_time': total_processing_time / total_files,
            'content_types': content_types,
            'file_types': file_types,
            'avg_chunks_per_file': total_chunks / total_files if total_files > 0 else 0
        }


# Context7 Pattern: Convenience function for quick usage
def create_document_processor(project_root: str, batch_size: int = 50) -> DocumentProcessor:
    """
    Create a document processor instance.

    Args:
        project_root: Root path of the project
        batch_size: Batch processing size

    Returns:
        DocumentProcessor instance
    """
    return DocumentProcessor(project_root, batch_size)


# Context7 Pattern: Command-line interface for testing
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="DevStream Document Processor")
    parser.add_argument("project_root", help="Project root directory")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch processing size")
    parser.add_argument("--stats", action="store_true", help="Show processing statistics")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    processor = create_document_processor(args.project_root, args.batch_size)

    # Discover documents
    files = processor.discover_documents()
    print(f"Found {len(files)} files to process")

    if args.stats:
        # Process and show statistics
        processed_docs = processor.process_documents_batch(files)
        stats = processor.get_processing_stats(processed_docs)

        print("\n=== Processing Statistics ===")
        print(f"Total files: {stats['total_files']}")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Processing time: {stats['total_processing_time']:.2f}s")
        print(f"Avg time per file: {stats['avg_processing_time']:.2f}s")
        print(f"Avg chunks per file: {stats['avg_chunks_per_file']:.1f}")

        print("\nContent Types:")
        for content_type, count in stats['content_types'].items():
            print(f"  {content_type}: {count}")

        print("\nFile Types:")
        for file_type, count in stats['file_types'].items():
            print(f"  {file_type}: {count}")