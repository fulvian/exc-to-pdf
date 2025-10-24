#!/usr/bin/env python3
"""
DevStream Memory Bootstrap - Main Orchestrator
Context7-compliant memory bootstrap system for project initialization.

This module orchestrates the complete memory bootstrap process, combining
document processing, codebase scanning, and incremental indexing to create
a comprehensive project memory from existing codebases and documentation.

Key Features:
- Complete orchestration of bootstrap workflow
- Multi-modal processing (code + documentation + configuration)
- Context7-compliant incremental updates
- Progress tracking and reporting
- CLI interface with multiple operation modes
- Integration with DevStream startup workflow
Context7 Pattern: Orchestration + Multi-modal processing + CLI integration
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import bootstrap components
try:
    from document_processor import create_document_processor, ProcessedDocument
    from codebase_scanner import create_codebase_scanner, FileAnalysis
    from incremental_indexer import create_incremental_indexer, IndexingResult
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    logging.warning(f"Bootstrap components not available: {e}")

# Import DevStream memory client
try:
    from direct_client import get_direct_client
    MEMORY_CLIENT_AVAILABLE = True
except ImportError:
    MEMORY_CLIENT_AVAILABLE = False
    logging.warning("Memory client not available")


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap process."""
    project_root: str
    mode: str = "full"  # full, code-only, docs-only, incremental
    cleanup_mode: str = "incremental"  # full, incremental, none
    force_reindex: bool = False
    dry_run: bool = False
    batch_size: int = 50
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    verbose: bool = False
    output_format: str = "summary"  # summary, detailed, json


@dataclass
class BootstrapResult:
    """Complete bootstrap operation result."""
    success: bool
    config: BootstrapConfig
    document_result: Optional[IndexingResult] = None
    code_analysis_result: Optional[Dict[str, Any]] = None
    total_processing_time: float = 0.0
    total_files_processed: int = 0
    total_chunks_indexed: int = 0
    errors: List[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class MemoryBootstrap:
    """
    Context7-compliant memory bootstrap orchestrator.

    Implements comprehensive bootstrap workflow:
    - Document processing with LangChain patterns
    - Codebase scanning with AST analysis
    - Incremental indexing with deduplication
    - Progress tracking and reporting
    - CLI integration
    """

    def __init__(self, config: BootstrapConfig):
        """
        Initialize memory bootstrap.

        Args:
            config: Bootstrap configuration
        """
        self.config = config
        self.project_root = Path(config.project_root)

        # Configure logging
        log_level = logging.DEBUG if config.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize components
        if not COMPONENTS_AVAILABLE:
            raise RuntimeError("Bootstrap components not available")

        self.doc_processor = create_document_processor(
            str(self.project_root),
            batch_size=config.batch_size
        )
        self.code_scanner = create_codebase_scanner(str(self.project_root))
        self.memory_client = get_direct_client() if MEMORY_CLIENT_AVAILABLE else None
        self.incremental_indexer = create_incremental_indexer(
            str(self.project_root),
            self.memory_client
        )

        # Validate configuration
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate bootstrap configuration."""
        if not self.project_root.exists():
            raise ValueError(f"Project root does not exist: {self.project_root}")

        if self.config.mode not in ["full", "code-only", "docs-only", "incremental"]:
            raise ValueError(f"Invalid mode: {self.config.mode}")

        if self.config.cleanup_mode not in ["full", "incremental", "none"]:
            raise ValueError(f"Invalid cleanup_mode: {self.config.cleanup_mode}")

        if self.config.output_format not in ["summary", "detailed", "json"]:
            raise ValueError(f"Invalid output_format: {self.config.output_format}")

    def _discover_files(self) -> Tuple[List[Path], List[Path]]:
        """
        Discover files for processing based on mode.

        Context7 Pattern: Mode-based file discovery.
        """
        all_files = self.doc_processor.discover_documents(
            self.config.include_patterns,
            self.config.exclude_patterns
        )

        code_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.rs', '.go', '.java'}
        doc_extensions = {'.md', '.rst', '.txt', '.doc', '.docx'}

        if self.config.mode == "full":
            return all_files, []

        elif self.config.mode == "code-only":
            code_files = [f for f in all_files if f.suffix.lower() in code_extensions]
            return code_files, []

        elif self.config.mode == "docs-only":
            doc_files = [f for f in all_files if f.suffix.lower() in doc_extensions]
            return doc_files, []

        elif self.config.mode == "incremental":
            # For incremental mode, let the indexer decide what needs processing
            return all_files, []

        return [], []

    def _process_documents(self, file_paths: List[Path]) -> IndexingResult:
        """
        Process documents and store in memory.

        Context7 Pattern: Incremental document processing.
        """
        self.logger.info(f"Processing {len(file_paths)} documents...")

        start_time = time.time()

        if self.config.dry_run:
            # Dry run - just count what would be processed
            processed_count = 0
            chunk_count = 0

            for file_path in file_paths:
                try:
                    processed_doc = self.doc_processor.process_document(file_path)
                    chunked_docs = self.doc_processor.chunk_document(processed_doc)
                    processed_count += 1
                    chunk_count += len(chunked_docs)
                except Exception as e:
                    self.logger.warning(f"Dry run - would fail on {file_path}: {e}")

            return IndexingResult(
                success=True,
                total_files=len(file_paths),
                processed_files=processed_count,
                updated_files=0,
                added_files=processed_count,
                skipped_files=len(file_paths) - processed_count,
                deleted_files=0,
                total_chunks=chunk_count,
                processing_time=time.time() - start_time,
                warnings=["Dry run mode - no actual indexing performed"]
            )

        # Actual processing - Context7 Pattern: Explicit file processing with progress tracking
        # ✅ Process the passed file_paths directly (fixes ignored parameter bug)
        processed = 0
        added = 0
        updated = 0
        skipped = 0
        errors = []
        total_chunks = 0

        # Process each file explicitly (Context7: Explicit over Implicit)
        for i, file_path in enumerate(file_paths, 1):
            try:
                success, message = self.incremental_indexer.index_file(
                    file_path,
                    force_reindex=self.config.force_reindex
                )

                # Progress logging (Context7: Observable behavior)
                if i % 10 == 0 or i == len(file_paths):
                    self.logger.info(f"Progress: {i}/{len(file_paths)} files processed")

                processed += 1

                # Parse result message for statistics
                if success:
                    if "Reindexed" in message:
                        updated += 1
                    elif "Indexed" in message:
                        added += 1
                    elif "Skipped" in message:
                        skipped += 1

                    # Extract chunk count if present
                    # Message format: "Indexed: path (5 chunks)" or "Reindexed: path (5 chunks)"
                    if "chunks)" in message:
                        try:
                            chunks_str = message.split("(")[-1].split(")")[0]  # Extract "5 chunks"
                            chunks = int(chunks_str.split()[0])  # Take first word = "5"
                            total_chunks += chunks
                        except (ValueError, IndexError):
                            pass
                else:
                    errors.append(message)

            except Exception as e:
                error_msg = f"Error processing {file_path}: {e}"
                errors.append(error_msg)
                self.logger.error(error_msg)

        processing_time = time.time() - start_time

        # Return comprehensive result
        return IndexingResult(
            success=len(errors) == 0,
            total_files=len(file_paths),
            processed_files=processed,
            updated_files=updated,
            added_files=added,
            skipped_files=skipped,
            deleted_files=0,
            total_chunks=total_chunks,
            processing_time=processing_time,
            errors=errors
        )

    def _analyze_codebase(self, file_paths: List[Path]) -> Dict[str, Any]:
        """
        Perform codebase analysis.

        Context7 Pattern: Comprehensive code analysis with relationships.
        """
        self.logger.info(f"Analyzing codebase for {len(file_paths)} files...")

        start_time = time.time()

        # Filter code files
        code_extensions = {'.py', '.js', '.jsx', '.ts', '.tsx', '.rs', '.go', '.java'}
        code_files = [f for f in file_paths if f.suffix.lower() in code_extensions]

        if not code_files:
            return {
                'files_analyzed': 0,
                'languages': {},
                'total_elements': 0,
                'processing_time': time.time() - start_time,
                'message': 'No code files found for analysis'
            }

        # Perform analysis
        analyses = self.code_scanner.scan_directory(self.project_root)

        # Get comprehensive results
        summary = self.code_scanner.get_scan_summary(analyses)
        xref = self.code_scanner.get_cross_references(analyses)
        dependencies = self.code_scanner.get_dependency_graph(analyses)

        # Store code analysis results in memory if available
        if self.memory_client and not self.config.dry_run:
            try:
                self._store_code_analysis_results(summary, xref, dependencies)
            except Exception as e:
                self.logger.warning(f"Failed to store code analysis results: {e}")

        processing_time = time.time() - start_time

        return {
            'files_analyzed': len(analyses),
            'languages': summary.get('languages', {}),
            'total_elements': summary.get('total_elements', 0),
            'functions': summary.get('total_functions', 0),
            'classes': summary.get('total_classes', 0),
            'imports': summary.get('total_imports', 0),
            'relationships': summary.get('total_relationships', 0),
            'avg_complexity': summary.get('avg_complexity', 0),
            'documentation_coverage': summary.get('documentation_coverage', 0),
            'cross_references_count': len(xref),
            'dependencies_count': len(dependencies),
            'processing_time': processing_time,
            'summary': summary,
            'cross_references_sample': dict(list(xref.items())[:10]),  # First 10 for sample
            'dependencies_sample': dict(list(dependencies.items())[:10])  # First 10 for sample
        }

    def _store_code_analysis_results(self, summary: Dict[str, Any],
                                   xref: Dict[str, List[str]],
                                   dependencies: Dict[str, List[str]]) -> None:
        """Store code analysis results in memory."""
        if not self.memory_client:
            return

        try:
            # Store summary
            summary_content = json.dumps(summary, indent=2, default=str)
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(
                    self.memory_client.store_memory(
                        content=summary_content,
                        content_type="code_analysis",
                        keywords=["codebase", "analysis", "summary"],
                        session_id=None
                    )
                )
            finally:
                loop.close()

            # Store key insights
            insights = [
                f"Codebase contains {summary.get('total_functions', 0)} functions and {summary.get('total_classes', 0)} classes",
                f"Documentation coverage: {summary.get('documentation_coverage', 0):.1%}",
                f"Average complexity: {summary.get('avg_complexity', 0):.1f}",
                f"Languages used: {', '.join(summary.get('languages', {}).keys())}"
            ]

            for insight in insights:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        self.memory_client.store_memory(
                            content=insight,
                            content_type="code_insight",
                            keywords=["codebase", "insight"],
                            session_id=None
                        )
                    )
                finally:
                    loop.close()

        except Exception as e:
            self.logger.error(f"Error storing code analysis results: {e}")

    def run_bootstrap(self) -> BootstrapResult:
        """
        Run complete bootstrap process.

        Context7 Pattern: Orchestrated workflow with error handling.
        """
        start_time = time.time()
        self.logger.info(f"Starting memory bootstrap for {self.project_root}")
        self.logger.info(f"Mode: {self.config.mode}, Cleanup: {self.config.cleanup_mode}")

        result = BootstrapResult(
            success=True,
            config=self.config
        )

        try:
            # Discover files
            all_files, _ = self._discover_files()
            result.total_files_processed = len(all_files)

            if not all_files:
                result.warnings.append("No files found for processing")
                result.total_processing_time = time.time() - start_time
                return result

            # Process documents (primary operation)
            if self.config.mode in ["full", "docs-only", "incremental"]:
                result.document_result = self._process_documents(all_files)
                result.total_chunks_indexed = result.document_result.total_chunks
                result.errors.extend(result.document_result.errors)
                result.warnings.extend(result.document_result.warnings)

            # Analyze codebase (secondary operation)
            if self.config.mode in ["full", "code-only"]:
                result.code_analysis_result = self._analyze_codebase(all_files)

            # Update metadata
            result.metadata = {
                'project_root': str(self.project_root),
                'files_discovered': len(all_files),
                'indexing_status': self.incremental_indexer.get_indexing_status(),
                'memory_client_available': MEMORY_CLIENT_AVAILABLE,
                'bootstrap_timestamp': time.time()
            }

            result.total_processing_time = time.time() - start_time

            self.logger.info(f"Bootstrap completed successfully in {result.total_processing_time:.2f}s")
            self.logger.info(f"Processed {result.total_files_processed} files, "
                           f"indexed {result.total_chunks_indexed} chunks")

        except Exception as e:
            result.success = False
            result.total_processing_time = time.time() - start_time
            error_msg = f"Bootstrap failed: {e}"
            self.logger.error(error_msg)
            result.errors.append(error_msg)

        return result

    def format_output(self, result: BootstrapResult) -> str:
        """
        Format bootstrap result for output.

        Context7 Pattern: Multiple output formats for different use cases.
        """
        if self.config.output_format == "json":
            return json.dumps(asdict(result), indent=2, default=str)

        elif self.config.output_format == "detailed":
            return self._format_detailed_output(result)

        else:  # summary
            return self._format_summary_output(result)

    def _format_summary_output(self, result: BootstrapResult) -> str:
        """Format summary output."""
        lines = []
        lines.append("=== DevStream Memory Bootstrap Results ===")
        lines.append(f"✅ Success: {result.success}")
        lines.append(f"📁 Project: {result.config.project_root}")
        lines.append(f"🔧 Mode: {result.config.mode}")
        lines.append(f"⏱️  Time: {result.total_processing_time:.2f}s")
        lines.append("")

        if result.document_result:
            doc = result.document_result
            lines.append("📄 Document Processing:")
            lines.append(f"  Files processed: {doc.processed_files}/{doc.total_files}")
            lines.append(f"  Added: {doc.added_files}, Updated: {doc.updated_files}, Skipped: {doc.skipped_files}")
            lines.append(f"  Total chunks: {doc.total_chunks}")
            if doc.deleted_files > 0:
                lines.append(f"  Deleted: {doc.deleted_files}")
            lines.append("")

        if result.code_analysis_result:
            analysis = result.code_analysis_result
            lines.append("🔍 Code Analysis:")
            lines.append(f"  Files analyzed: {analysis['files_analyzed']}")
            lines.append(f"  Functions: {analysis['functions']}, Classes: {analysis['classes']}")
            lines.append(f"  Languages: {', '.join(analysis['languages'].keys())}")
            lines.append(f"  Avg complexity: {analysis['avg_complexity']:.1f}")
            lines.append(f"  Documentation coverage: {analysis['documentation_coverage']:.1%}")
            lines.append("")

        if result.errors:
            lines.append(f"❌ Errors ({len(result.errors)}):")
            for error in result.errors[:3]:
                lines.append(f"  {error}")
            if len(result.errors) > 3:
                lines.append(f"  ... and {len(result.errors) - 3} more")
            lines.append("")

        if result.warnings:
            lines.append(f"⚠️  Warnings ({len(result.warnings)}):")
            for warning in result.warnings[:3]:
                lines.append(f"  {warning}")
            if len(result.warnings) > 3:
                lines.append(f"  ... and {len(result.warnings) - 3} more")

        return "\n".join(lines)

    def _format_detailed_output(self, result: BootstrapResult) -> str:
        """Format detailed output."""
        summary = self._format_summary_output(result)
        details = []

        if result.code_analysis_result and result.code_analysis_result.get('summary'):
            summary_data = result.code_analysis_result['summary']
            details.append("\n=== Detailed Code Analysis ===")
            details.append(f"Element types: {summary_data.get('element_types', {})}")
            details.append(f"Relationship types: {summary_data.get('relationship_types', {})}")
            details.append(f"Max complexity: {summary_data.get('max_complexity', 0)}")

        if result.metadata and result.metadata.get('indexing_status'):
            status = result.metadata['indexing_status']
            details.append("\n=== Indexing Status ===")
            details.append(f"Total indexed files: {status['total_files']}")
            details.append(f"Total indexed chunks: {status['total_chunks']}")
            details.append(f"Content types: {status['content_types']}")

        return summary + "\n".join(details)


# Context7 Pattern: CLI interface for easy integration
def create_bootstrap_config(**kwargs) -> BootstrapConfig:
    """Create bootstrap configuration from keyword arguments."""
    return BootstrapConfig(**kwargs)


def main():
    """Command-line interface for memory bootstrap."""
    parser = argparse.ArgumentParser(
        description="DevStream Memory Bootstrap - Initialize project memory from codebase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full bootstrap of current directory
  python memory_bootstrap.py .

  # Code-only analysis
  python memory_bootstrap.py . --mode code-only

  # Incremental update
  python memory_bootstrap.py . --mode incremental

  # Dry run to see what would be processed
  python memory_bootstrap.py . --dry-run

  # Force full reindexing
  python memory_bootstrap.py . --force --cleanup full
        """
    )

    parser.add_argument("project_root", help="Project root directory")
    parser.add_argument("--mode", choices=["full", "code-only", "docs-only", "incremental"],
                       default="full", help="Processing mode (default: full)")
    parser.add_argument("--cleanup", choices=["full", "incremental", "none"],
                       default="incremental", help="Cleanup mode (default: incremental)")
    parser.add_argument("--force", action="store_true", help="Force reindexing of all files")
    parser.add_argument("--dry-run", action="store_true", help="Simulate processing without actual changes")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch processing size (default: 50)")
    parser.add_argument("--output", choices=["summary", "detailed", "json"],
                       default="summary", help="Output format (default: summary)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Create configuration
    config = create_bootstrap_config(
        project_root=args.project_root,
        mode=args.mode,
        cleanup_mode=args.cleanup,
        force_reindex=args.force,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        output_format=args.output,
        verbose=args.verbose
    )

    try:
        # Run bootstrap
        bootstrap = MemoryBootstrap(config)
        result = bootstrap.run_bootstrap()

        # Output results
        output = bootstrap.format_output(result)
        print(output)

        # Exit with appropriate code
        sys.exit(0 if result.success else 1)

    except Exception as e:
        print(f"❌ Fatal error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()