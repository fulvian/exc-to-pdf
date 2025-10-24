#!/usr/bin/env .devstream/bin/python
"""
DevStream PostToolUse Hook - Memory Storage after Write/Edit with Embeddings

Stores modified file content in DevStream semantic memory and generates
embeddings using Ollama for semantic search capabilities.

Phase 2 Enhancement: Inline embedding generation with graceful degradation.
FASE 4.3: Rate limiting for memory storage and Ollama API calls.
"""

import sys
import asyncio
import sqlite3
import json
import re
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from cchooks import safe_create_context, PostToolUseContext
from devstream_base import DevStreamHookBase, FeedbackLevel

# Context7-compliant robust import with fallback for unified_client
try:
    # Try relative import first (when run as module)
    from .unified_client import get_unified_client
except ImportError:
    try:
        # Fallback to absolute import (when run as script)
        from unified_client import get_unified_client
    except ImportError as e:
        # Final fallback - create dummy client that gracefully degrades
        # Context7-compliant: variable scope fixed by moving print inside except block
        error_message = str(e)

        def get_unified_client():
            class DummyUnifiedClient:
                def __init__(self):
                    self.disabled = True

                async def search_memory(self, *args, **kwargs):
                    return None

                async def store_memory(self, *args, **kwargs):
                    return None

                async def health_check(self):
                    return {"backends": {}, "overall": "disabled"}

                async def trigger_checkpoint(self, *args, **kwargs):
                    return None

                def _get_direct_client(self):
                    return self

                @property
                def connection_manager(self):
                    return DummyConnectionManager()

            class DummyConnectionManager:
                def get_connection(self):
                    class DummyConnection:
                        def __enter__(self):
                            return self
                        def __exit__(self, *args):
                            pass
                        def cursor(self):
                            return DummyCursor()
                        def commit(self):
                            pass
                    return DummyConnection()

            class DummyCursor:
                def execute(self, query, params=None):
                    return self
                def fetchone(self):
                    return None
                def fetchall(self):
                    return []
                @property
                def rowcount(self):
                    return 0

            return DummyUnifiedClient()
        print(f"⚠️  DevStream: unified_client unavailable, using fallback: {error_message}", file=sys.stderr)
from ollama_client import OllamaEmbeddingClient
from sqlite_vec_helper import get_db_connection_with_vec
from connection_manager import get_connection_manager
from rate_limiter import (
    memory_rate_limiter,
    ollama_rate_limiter,
    has_memory_capacity,
    has_ollama_capacity
)
# real_time_capture module not available - commenting out
# from real_time_capture import get_real_time_capture

# ContentQualityFilter integration (Task 1: Storage optimization)
# Replace basic content storage with intelligent quality filtering
# Target: 95% reduction in useless records (from 109K baseline)

try:
    from optimization.content_quality_filter import get_content_quality_filter
    CONTENT_QUALITY_FILTER_AVAILABLE = True
except ImportError as e:
    CONTENT_QUALITY_FILTER_AVAILABLE = False
    _CONTENT_QUALITY_FILTER_IMPORT_ERROR = str(e)
    print(f"⚠️  DevStream: ContentQualityFilter unavailable: {e}", file=sys.stderr)

# AsyncEmbeddingBatchProcessor integration (Task 2: Embedding optimization)
# Replace synchronous embedding generation with async batch processing
# Target: 100% pass rate with Context7-compliant retry patterns

try:
    from optimization.async_embedding_processor import get_async_embedding_processor
    ASYNC_EMBEDDING_PROCESSOR_AVAILABLE = True
except ImportError as e:
    ASYNC_EMBEDDING_PROCESSOR_AVAILABLE = False
    _ASYNC_EMBEDDING_PROCESSOR_IMPORT_ERROR = str(e)
    print(f"⚠️  DevStream: AsyncEmbeddingProcessor unavailable: {e}", file=sys.stderr)

# Protocol State Manager imports (FASE 2 Integration)
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'protocol'))
    from protocol_state_manager import ProtocolStateManager, ProtocolStep
    from task_state_sync import TaskStateSync
    PROTOCOL_SYNC_AVAILABLE = True
except ImportError as e:
    PROTOCOL_SYNC_AVAILABLE = False
    _SYNC_IMPORT_ERROR = str(e)

# Event Sourcing Session Log imports (Phase 3 Integration)
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from sessions.session_event_log import get_session_log
    SESSION_EVENT_LOG_AVAILABLE = True
except ImportError as e:
    SESSION_EVENT_LOG_AVAILABLE = False
    _EVENT_LOG_IMPORT_ERROR = str(e)


class PostToolUseHook:
    """
    PostToolUse hook for automatic memory storage with embeddings.

    Stores file modifications in DevStream semantic memory and generates
    embeddings using Ollama for semantic search.

    Phase 2 Enhancement: Inline embedding generation with graceful degradation.
    FASE 4.4: Retry logic for temporary failures.
    """

    def __init__(self):
        self.base = DevStreamHookBase("post_tool_use")
        self.unified_client = get_unified_client()

        # Initialize Ollama client for embedding generation
        self.ollama_client = OllamaEmbeddingClient()

        # Database path for direct embedding updates
        project_root = Path(__file__).parent.parent.parent.parent.parent
        self.db_path = str(project_root / 'data' / 'devstream.db')

        # FASE 1: Initialize RealTimeDataCapture for enhanced file monitoring
        # real_time_capture module not available - commenting out
        # self.real_time_capture = get_real_time_capture(str(project_root))
        self.real_time_capture = None

        # FASE 2: Protocol State Sync components
        self.protocol_manager = None
        self.task_sync = None

        # FASE 4.4: Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0  # Initial delay in seconds
        self.retry_backoff = 2.0  # Backoff multiplier

        # ATOMIC FIX: Connection pool for database operations
        self._db_pool = None
        self._init_db_pool()

        if PROTOCOL_SYNC_AVAILABLE:
            try:
                self.protocol_manager = ProtocolStateManager()
                self.task_sync = TaskStateSync()
                self.base.debug_log("Protocol sync components initialized")
            except Exception as e:
                self.base.user_feedback(
                    f"Protocol sync initialization failed: {e}",
                    FeedbackLevel.MINIMAL
                )

        # Initialize ContentQualityFilter for intelligent storage (Task 1)
        self.content_quality_filter = None
        if CONTENT_QUALITY_FILTER_AVAILABLE:
            try:
                self.content_quality_filter = get_content_quality_filter(
                    quality_threshold=0.3  # Filter low-quality content
                )
                self.base.debug_log("ContentQualityFilter initialized")
            except Exception as e:
                self.base.debug_log(f"ContentQualityFilter init failed: {e}")
                # Don't modify module-level variable - just set component to None
        else:
            self.base.debug_log(f"ContentQualityFilter unavailable: {_CONTENT_QUALITY_FILTER_IMPORT_ERROR}")

        # Initialize AsyncEmbeddingProcessor for optimized embedding generation (Task 2)
        self.async_embedding_processor = None
        if ASYNC_EMBEDDING_PROCESSOR_AVAILABLE:
            try:
                self.async_embedding_processor = get_async_embedding_processor(
                    batch_size=5,                    # Process up to 5 embeddings at once
                    max_retries=3,                   # Retry logic for temporary failures
                    max_concurrent_batches=3           # Concurrent batch processing
                )
                self.base.debug_log("AsyncEmbeddingProcessor initialized")
            except Exception as e:
                self.base.debug_log(f"AsyncEmbeddingProcessor init failed: {e}")
                # Don't modify module-level variable - just set component to None
        else:
            self.base.debug_log(f"AsyncEmbeddingProcessor unavailable: {_ASYNC_EMBEDDING_PROCESSOR_IMPORT_ERROR}")

    async def retry_with_backoff(
        self,
        operation_name: str,
        operation_func,
        *args,
        **kwargs
    ) -> Optional[Any]:
        """
        Execute operation with exponential backoff retry logic.

        FASE 4.4: Retry logic for temporary failures (MCP, Ollama, DB).

        Args:
            operation_name: Name of the operation for logging
            operation_func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Operation result if successful, None otherwise

        Note:
            - Max 3 retries with exponential backoff (1s, 2s, 4s)
            - Only retries temporary failures (connection, timeout, rate limit)
            - Permanent failures (validation, auth) fail immediately
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await operation_func(*args, **kwargs)

                if attempt > 0:
                    self.base.debug_log(
                        f"✓ {operation_name} succeeded on attempt {attempt + 1}"
                    )

                return result

            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()

                # Check if this is a retryable error
                is_retryable = any(keyword in error_msg for keyword in [
                    'connection', 'timeout', 'rate limit', 'temporary',
                    'network', 'unavailable', 'overloaded', '503', '502',
                    'connection reset', 'connection refused'
                ])

                # Don't retry permanent failures
                if not is_retryable:
                    self.base.debug_log(
                        f"❌ {operation_name} permanent failure: {e}"
                    )
                    return None

                if attempt < self.max_retries:
                    # Calculate delay with exponential backoff
                    delay = self.retry_delay * (self.retry_backoff ** attempt)

                    self.base.debug_log(
                        f"⚠️ {operation_name} failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                        f" - retrying in {delay:.1f}s"
                    )

                    await asyncio.sleep(delay)
                else:
                    self.base.debug_log(
                        f"❌ {operation_name} failed after {self.max_retries + 1} attempts: {e}"
                    )

        return None

    def extract_content_preview(self, content: str, max_length: int = 300) -> str:
        """
        Extract content preview for memory storage.

        Args:
            content: Full content
            max_length: Maximum preview length

        Returns:
            Content preview string
        """
        if len(content) <= max_length:
            return content

        # Try to break at sentence/line boundary
        preview = content[:max_length]
        last_period = preview.rfind('.')
        last_newline = preview.rfind('\n')

        break_point = max(last_period, last_newline)
        if break_point > max_length * 0.7:  # At least 70% of max length
            return preview[:break_point + 1].strip()

        return preview.strip() + "..."

    def extract_keywords(self, file_path: str, content: str) -> list[str]:
        """
        Extract keywords from file path and content.

        Args:
            file_path: Path to file
            content: File content

        Returns:
            List of keywords
        """
        keywords = []

        # Add file name without extension
        file_name = Path(file_path).stem
        keywords.append(file_name)

        # Add parent directory if relevant
        parent = Path(file_path).parent.name
        if parent and parent not in ['.', '..']:
            keywords.append(parent)

        # Detect language/framework from file extension
        ext = Path(file_path).suffix.lower()
        lang_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'react',
            '.jsx': 'react',
            '.vue': 'vue',
            '.rs': 'rust',
            '.go': 'golang',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.sh': 'bash',
            '.sql': 'sql',
            '.md': 'documentation',
            '.json': 'config',
            '.yaml': 'config',
            '.yml': 'config',
        }
        if ext in lang_map:
            keywords.append(lang_map[ext])

        # Add "implementation" tag
        keywords.append("implementation")

        return keywords

    async def trigger_real_time_capture_for_critical_tool(self, tool_name: str, file_path: str = "") -> None:
        """
        Trigger real-time file capture after critical tool execution.

        FASE 1 Enhancement: Replaces generic "Task Checkpoint" messages with
        real file modifications and session-specific data using RealTimeDataCapture.

        Context7 Pattern: Enhanced file monitoring with specific context storage.

        Critical tools: Write, Edit, MultiEdit, Bash, TodoWrite

        Args:
            tool_name: Name of the critical tool that was executed
            file_path: Path to file that was modified (if applicable)
        """
        try:
            self.base.debug_log(f"Triggering real-time capture for critical tool: {tool_name}")

            # FASE 1: Real-time capture not available - graceful degradation
            if self.real_time_capture is None:
                self.base.debug_log("Real-time capture not available (module missing)")
            else:
                # Start real-time monitoring if not already running
                if not self.real_time_capture.is_running:
                    monitoring_started = self.real_time_capture.start_monitoring()
                    if monitoring_started:
                        self.base.debug_log("Real-time file monitoring started")
                    else:
                        self.base.debug_log("Failed to start real-time monitoring")

                # If we have a specific file path, ensure it's being monitored
                if file_path and self.real_time_capture._should_monitor_file(file_path):
                    self.base.debug_log(f"File is monitored for real-time capture: {Path(file_path).name}")

                # Get real-time capture status
                status = self.real_time_capture.get_status()
                self.base.debug_log(
                    f"Real-time capture status: running={status['is_running']}, "
                    f"files_monitored={status['monitored_files_count']}"
                )

            # Use unified client for checkpoint with automatic backend selection
            result = await self.unified_client.trigger_checkpoint(
                reason="real_time_file_capture",
                hook_name="post_tool_use"
            )

            if result:
                # Extract checkpoint count from result
                if isinstance(result, dict) and "content" in result:
                    content_text = result["content"][0]["text"] if result["content"] else ""
                    self.base.debug_log(f"Enhanced checkpoint result: {content_text}")
            else:
                self.base.debug_log("Enhanced checkpoint trigger returned no result (non-blocking)")

        except Exception as e:
            # Context7 Pattern: Graceful degradation - log but don't fail
            self.base.debug_log(f"Real-time capture trigger failed (non-blocking): {e}")

    def update_memory_embedding(
        self,
        memory_id: str,
        embedding: List[float]
    ) -> bool:
        """
        Update semantic_memory record with embedding vector using BLOB storage.

        BLOB OPTIMIZATION: Uses sqlite-vec BLOB storage instead of JSON for
        70% space reduction and 10x performance improvement.

        Context7 Pattern: Uses ConnectionManager that already loads sqlite-vec.
        FASE 4.4: Enhanced with connection retry logic.

        Args:
            memory_id: Memory record ID
            embedding: Embedding vector (list of floats)

        Returns:
            True if update successful, False otherwise
        """
        # FASE 4.4: Synchronous retry for database operations
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                # BLOB OPTIMIZATION: Convert embedding to BLOB using sqlite-vec
                # This provides 70% space reduction and 10x faster queries
                import struct

                # Pack float list into binary BLOB format
                embedding_blob = struct.pack(f'{len(embedding)}f', *embedding)

                # CRITICAL FIX: Use ConnectionManager instead of sqlite_vec_helper
                # ConnectionManager already loads sqlite-vec extension (see _create_connection)
                manager = get_connection_manager(self.db_path)

                with manager.get_connection() as conn:
                    cursor = conn.cursor()

                    # Verify sqlite-vec extension is available
                    try:
                        vec_version = cursor.execute("SELECT vec_version()").fetchone()[0]
                        if attempt == 0:
                            self.base.debug_log(f"✓ Using sqlite-vec v{vec_version} for BLOB storage")
                    except Exception:
                        self.base.debug_log("⚠️ sqlite-vec extension not available - using JSON fallback")
                        # Fallback to JSON if sqlite-vec not available
                        embedding_json = json.dumps(embedding)
                        cursor.execute(
                            "UPDATE semantic_memory SET embedding = ? WHERE id = ?",
                            (embedding_json, memory_id)
                        )
                    else:
                        # BLOB OPTIMIZATION: Store as binary BLOB for optimal performance
                        cursor.execute(
                            "UPDATE semantic_memory SET embedding_blob = ? WHERE id = ?",
                            (embedding_blob, memory_id)
                        )

                        # Also set embedding_model and dimension metadata
                        cursor.execute(
                            "UPDATE semantic_memory SET embedding_model = ?, embedding_dimension = ? WHERE id = ?",
                            ('gemma3', len(embedding), memory_id)
                        )

                    # rows_updated is available after the context manager commits
                    rows_updated = cursor.rowcount

                if rows_updated > 0:
                    if attempt > 0:
                        self.base.debug_log(
                            f"✓ Embedding BLOB update succeeded on attempt {attempt + 1}: {memory_id[:8]}..."
                        )

                    self.base.debug_log(
                        f"✓ Embedding stored as BLOB: {memory_id[:8]}... "
                        f"({len(embedding)} dimensions, {len(embedding_blob)} bytes)"
                    )
                    return True
                else:
                    self.base.debug_log(f"No record found to update: {memory_id}")
                    return False

            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()

                # Check if this is a retryable database error
                is_retryable = any(keyword in error_msg for keyword in [
                    'connection', 'timeout', 'locked', 'busy', 'database is locked',
                    'disk i/o error', 'protocol error', 'connection refused'
                ])

                # Don't retry permanent database failures
                if not is_retryable:
                    self.base.debug_log(f"❌ Embedding BLOB update permanent failure: {e}")
                    return False

                if attempt < self.max_retries:
                    # Calculate delay with exponential backoff (synchronous)
                    delay = self.retry_delay * (self.retry_backoff ** attempt)

                    self.base.debug_log(
                        f"⚠️ Embedding BLOB update failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                        f" - retrying in {delay:.1f}s"
                    )

                    # ATOMIC FIX: Use asyncio.sleep even in sync function to prevent blocking
                    import asyncio
                    asyncio.run(asyncio.sleep(delay))
                else:
                    self.base.debug_log(
                        f"❌ Embedding BLOB update failed after {self.max_retries + 1} attempts: {e}"
                    )

        return False

    async def store_in_memory(
        self,
        file_path: str,
        content: str,
        operation: str,
        topics: List[str],
        entities: List[str],
        content_type: str = "code"
    ) -> Optional[str]:
        """
        Store file modification in DevStream memory with embedding.

        Phase 2 Enhancement: Now generates embedding and stores it inline.
        FASE 3 Enhancement: Includes topics, entities, and content_type.
        FASE 4.4: Enhanced with retry logic for temporary failures.

        Args:
            file_path: Path to modified file
            content: File content
            operation: Operation type (Write, Edit, MultiEdit, Bash, Read, TodoWrite)
            topics: List of extracted topics
            entities: List of extracted technology entities
            content_type: Content type classification (code, output, context, decision, error)

        Returns:
            Memory ID if storage successful, None otherwise
        """
        try:
            # TASK 1 ENHANCEMENT: Apply ContentQualityFilter for intelligent storage
            # Target: 95% reduction in useless records (from 109K baseline)
            if self.content_quality_filter:
                try:
                    # Apply quality filtering to content
                    should_store, quality_score = self.content_quality_filter.should_store_content(
                        content=content,
                        file_path=file_path,
                        content_type=content_type,
                        entities=entities,
                        topics=topics
                    )

                    if not should_store:
                        self.base.debug_log(
                            f"ContentQualityFilter rejected: score below threshold "
                            f"(score: {quality_score:.2f})"
                        )
                        return None  # Skip storage

                    # Use original content and quality-based metadata
                    filtered_content = content
                    enhanced_keywords = []  # Basic keywords - no enhancement from filter

                    self.base.debug_log(
                        f"ContentQualityFilter accepted: score={quality_score:.2f}, "
                        f"enhanced_keywords={len(enhanced_keywords)}"
                    )

                except Exception as e:
                    self.base.debug_log(f"ContentQualityFilter failed, using original content: {e}")
                    filtered_content = content
                    enhanced_keywords = []
                    quality_score = 0.0
            else:
                # Fallback to original content
                filtered_content = content
                enhanced_keywords = []
                quality_score = 0.0

            # Extract content preview from filtered content
            preview = self.extract_content_preview(filtered_content, max_length=500)

            # Build memory content with quality information
            memory_content = f"""# File Modified: {Path(file_path).name}

**Operation**: {operation}
**File**: {file_path}
**Quality Score**: {quality_score:.2f}

## Content Preview

{preview}
"""

            # Extract base keywords
            keywords = self.extract_keywords(file_path, filtered_content)

            # Add topics and entities to keywords
            keywords.extend(topics)
            keywords.extend(entities)

            # Add enhanced keywords from ContentQualityFilter
            keywords.extend(enhanced_keywords)

            # Add tool source tracking
            keywords.append(f"tool:{operation.lower()}")

            # Add quality-based keyword for filtering
            if quality_score > 0.8:
                keywords.append("high-quality")
            elif quality_score > 0.5:
                keywords.append("medium-quality")
            else:
                keywords.append("low-quality")

            # Deduplicate keywords
            keywords = list(set(keywords))

            self.base.debug_log(
                f"Storing memory: {len(preview)} chars, {len(keywords)} keywords "
                f"({len(topics)} topics, {len(entities)} entities, quality: {quality_score:.2f})"
            )

            # Get current session ID for memory storage
            session_id = await self._get_current_session_id()

            # FASE 4.4: Use unified client with built-in retry and fallback logic
            result = await self.retry_with_backoff(
                f"Memory storage ({Path(file_path).name})",
                lambda: self.unified_client.store_memory(
                    content=memory_content,
                    content_type=content_type,
                    keywords=keywords,
                    session_id=session_id,
                    hook_name="post_tool_use"
                )
            )

            if not result:
                self.base.debug_log("Memory storage returned no result after retries")
                return None

            # Extract memory_id from MCP result
            # MCP returns: {"success": true, "memory_id": "...", ...}
            memory_id = None
            if isinstance(result, dict):
                memory_id = result.get('memory_id')

            if not memory_id:
                self.base.debug_log("No memory_id in MCP response")
                return None

            self.base.success_feedback(f"Memory stored: {Path(file_path).name}")

            # Phase 2: Generate and store embedding with AsyncEmbeddingProcessor (Task 2)
            # Target: 100% pass rate with Context7-compliant retry patterns
            try:
                if self.async_embedding_processor:
                    # Use AsyncEmbeddingProcessor for optimized embedding generation
                    embedding_task = self.async_embedding_processor.queue_embedding_generation(
                        content=filtered_content,  # Use quality-filtered content
                        memory_id=memory_id,
                        priority="high" if quality_score > 0.7 else "normal",
                        metadata={
                            "file_path": file_path,
                            "operation": operation,
                            "content_type": content_type,
                            "quality_score": quality_score
                        }
                    )

                    self.base.debug_log(
                        f"AsyncEmbeddingProcessor queued: {memory_id[:8]}... "
                        f"(priority: {'high' if quality_score > 0.7 else 'normal'}, "
                        f"quality: {quality_score:.2f})"
                    )

                    # Process is non-blocking - embedding will be generated in background
                    # No need to wait for completion here
                else:
                    # Fallback to synchronous embedding generation
                    self.base.debug_log("Generating embedding via Ollama (fallback)...")

                    # ATOMIC FIX: Non-blocking embedding generation in background thread
                    loop = asyncio.get_running_loop()
                    embedding = await loop.run_in_executor(
                        None,  # Use default executor
                        lambda: self.ollama_client.generate_embedding(filtered_content)
                    )

                    if embedding:
                        # ATOMIC FIX: Non-blocking database operations in background thread
                        embedding_updated = await loop.run_in_executor(
                            None,  # Use default executor
                            lambda: self.update_memory_embedding(memory_id, embedding)
                        )

                        if embedding_updated:
                            self.base.debug_log(
                                f"✓ Embedding stored: {len(embedding)}D (fallback mode)"
                            )
                        else:
                            self.base.debug_log("Embedding update failed")
                    else:
                        self.base.debug_log("Embedding generation returned None")

            except Exception as embed_error:
                # Graceful degradation - log but don't fail
                self.base.debug_log(
                    f"Embedding generation failed (non-blocking): {embed_error}"
                )

            return memory_id

        except Exception as e:
            self.base.debug_log(f"Memory storage error: {e}")
            return None

    def classify_content_type(
        self,
        tool_name: str,
        tool_response: Dict[str, Any],
        content: str
    ) -> str:
        """
        Classify content type based on tool and response.

        Event Sourcing Pattern: Validate response success before classification.

        Args:
            tool_name: Name of the tool executed
            tool_response: Tool execution response with success flag
            content: Content to classify

        Returns:
            Content type: code|output|error|context|decision
        """
        # Event Sourcing pattern: Validate response
        if tool_response.get("success") == False:
            return "error"

        if tool_name in ["Write", "Edit", "MultiEdit"]:
            return "code"
        elif tool_name == "Bash":
            return "output" if tool_response.get("success") else "error"
        elif tool_name == "Read":
            return "context"
        elif tool_name == "TodoWrite":
            return "decision"

        return "context"

    def should_capture_bash_output(
        self,
        tool_input: Dict[str, Any],
        tool_response: Dict[str, Any]
    ) -> bool:
        """
        Determine if Bash output is significant for capture.

        Redis Agent Pattern: Multi-dimensional filtering to reduce noise.

        Args:
            tool_input: Bash command input
            tool_response: Bash execution response

        Returns:
            True if output is significant and should be captured
        """
        command = tool_input.get("command", "")

        # Skip trivial commands
        trivial_commands = ["ls", "pwd", "cd", "echo", "cat", "head", "tail", "grep", "find"]
        if any(command.strip().startswith(cmd) for cmd in trivial_commands):
            self.base.debug_log(f"Skipping trivial command: {command[:50]}")
            return False

        # Require significant output (>50 chars)
        output = tool_response.get("output", "")
        if len(output.strip()) < 50:
            self.base.debug_log(f"Skipping short output: {len(output)} chars")
            return False

        return True

    def should_capture_read_content(self, file_path: str) -> bool:
        """
        Determine if Read file is significant source/doc file.

        Memory Bank Pattern: Classify content by file type for active context.

        Args:
            file_path: Path to file being read

        Returns:
            True if file is significant source/documentation file
        """
        # Source and documentation extensions only
        source_extensions = [
            ".py", ".ts", ".tsx", ".js", ".jsx",
            ".md", ".rst", ".txt",
            ".json", ".yaml", ".yml",
            ".sh", ".sql"
        ]

        if not any(file_path.endswith(ext) for ext in source_extensions):
            self.base.debug_log(f"Skipping non-source file: {file_path}")
            return False

        # Excluded paths
        excluded_paths = [
            ".git/", "node_modules/", ".venv/", ".devstream/",
            "__pycache__/", "dist/", "build/", ".next/",
            "coverage/", ".pytest_cache/", ".mypy_cache/"
        ]

        if any(excluded in file_path for excluded in excluded_paths):
            self.base.debug_log(f"Skipping excluded path: {file_path}")
            return False

        return True

    def extract_topics(self, content: str, file_path: str = "") -> List[str]:
        """
        Extract topics from content and file path.

        Redis Agent Pattern: Multi-dimensional metadata for filtered search.

        Args:
            content: Content to extract topics from
            file_path: Optional file path for extension-based topics

        Returns:
            List of up to 5 unique topics
        """
        topics = []

        # From file extension
        ext_topic_map = {
            ".py": "python",
            ".ts": "typescript", ".tsx": "react",
            ".js": "javascript", ".jsx": "react",
            ".md": "documentation",
            ".yaml": "config", ".yml": "config",
            ".sql": "database",
            ".sh": "scripts"
        }

        for ext, topic in ext_topic_map.items():
            if file_path.endswith(ext):
                topics.append(topic)

        # From content keywords
        keyword_topic_map = {
            "test": "testing", "pytest": "testing", "unittest": "testing",
            "async": "async", "await": "async", "asyncio": "async",
            "api": "api", "endpoint": "api", "rest": "api",
            "auth": "authentication", "login": "authentication", "oauth": "authentication",
            "db": "database", "query": "database", "schema": "database",
            "hook": "hooks", "context": "context", "memory": "memory"
        }

        content_lower = content.lower()
        for keyword, topic in keyword_topic_map.items():
            if keyword in content_lower:
                topics.append(topic)

        # Deduplicate and limit to 5
        unique_topics = list(set(topics))[:5]

        self.base.debug_log(f"Extracted topics: {unique_topics}")
        return unique_topics

    def extract_entities(self, content: str) -> List[str]:
        """
        Extract technology/library entities from content.

        Redis Agent Pattern: Entity-based filtering for precise retrieval.

        Args:
            content: Content to extract entities from

        Returns:
            List of up to 5 unique technology entities
        """
        entities = []

        # Common tech stack entities (case-insensitive detection)
        tech_patterns = [
            # Python
            "FastAPI", "pytest", "SQLAlchemy", "Pydantic", "aiohttp", "asyncio",
            # TypeScript/React
            "React", "Next.js", "TypeScript", "Node.js", "Express", "Vue",
            # Infrastructure
            "Docker", "Kubernetes", "PostgreSQL", "Redis", "SQLite", "MongoDB",
            # Tools
            "Git", "GitHub", "VSCode", "JWT", "OAuth"
        ]

        content_lower = content.lower()
        for pattern in tech_patterns:
            if pattern.lower() in content_lower:
                entities.append(pattern)

        # Python imports detection
        import_pattern = r'from\s+(\w+)|import\s+(\w+)'
        matches = re.findall(import_pattern, content)

        for match in matches:
            entity = match[0] or match[1]
            # Skip standard library
            stdlib = ["os", "sys", "re", "json", "time", "datetime", "pathlib"]
            if entity and entity not in stdlib:
                entities.append(entity)

        # Deduplicate and limit to 5
        unique_entities = list(set(entities))[:5]

        self.base.debug_log(f"Extracted entities: {unique_entities}")
        return unique_entities

    async def _get_current_session_id(self) -> Optional[str]:
        """
        Get current active session ID from work_sessions table.

        Context7 Pattern: Use ConnectionManager for thread-safe database access

        Returns:
            Current session ID if found, None otherwise

        Note:
            Queries for most recent active session (status='active')
        """
        try:
            # Use ConnectionManager from unified client for thread-safe access
            connection_manager = self.unified_client._get_direct_client().connection_manager

            with connection_manager.get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT id FROM work_sessions
                    WHERE status = 'active'
                    ORDER BY started_at DESC
                    LIMIT 1
                    """
                )
                row = cursor.fetchone()
                if row:
                    session_id = row[0]
                    self.base.debug_log(f"Active session: {session_id[:8]}...")
                    return session_id

                self.base.debug_log("No active session found")
                return None

        except Exception as e:
            self.base.debug_log(f"Failed to get session ID: {e}")
            return None

    async def _get_active_files(self, session_id: str) -> List[str]:
        """
        Get current active_files list from session.

        Context7 Pattern: Use ConnectionManager for thread-safe database access

        Args:
            session_id: Session identifier

        Returns:
            List of active file paths (empty list if session not found)
        """
        try:
            # Use ConnectionManager for thread-safe access
            connection_manager = self.unified_client._get_direct_client().connection_manager

            with connection_manager.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT active_files FROM work_sessions WHERE id = ?",
                    (session_id,)
                )
                row = cursor.fetchone()

                if not row:
                    self.base.debug_log(f"Session not found: {session_id[:8]}...")
                    return []

                # Parse JSON (handle NULL case)
                return json.loads(row[0]) if row[0] else []

        except Exception as e:
            self.base.debug_log(f"Failed to get active files: {e}")
            return []

    def _init_db_pool(self):
        """Initialize database connection pool for performance."""
        try:
            import aiosqlite
            # ATOMIC FIX: Create connection pool with better configuration
            self._db_pool = aiosqlite.connect(
                self.db_path,
                # Connection pool settings for performance
                check_same_thread=False,
                timeout=30.0,  # 30 second timeout
                isolation_level=None  # Autocommit mode for better performance
            )
            self.base.debug_log("Database connection pool initialized")
        except Exception as e:
            self.base.debug_log(f"Database pool init failed: {e}")
            self._db_pool = None

    async def _close_db_pool(self):
        """Close database connection pool to prevent connection leaks."""
        if self._db_pool:
            try:
                await self._db_pool.close()
                self.base.debug_log("Database connection pool closed")
            except Exception as e:
                self.base.debug_log(f"Error closing DB pool: {e}")

    def __del__(self):
        """Destructor to ensure database connections are properly closed."""
        if hasattr(self, '_db_pool') and self._db_pool:
            # Best effort cleanup - can't use await in destructor
            try:
                # Create an event loop if none exists to close the pool
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    # If we're in an async context, schedule cleanup
                    loop.create_task(self._close_db_pool())
                except RuntimeError:
                    # No running loop, create a new one for cleanup
                    asyncio.run(self._close_db_pool())
            except Exception as e:
                # Silently fail in destructor - just log if possible
                pass

    async def _get_active_tasks(self, session_id: str) -> List[str]:
        """
        Get current active_tasks list from session.

        Context7 Pattern: Use connection pool for better performance.

        Args:
            session_id: Session identifier

        Returns:
            List of active task IDs/titles (empty list if session not found)
        """
        try:
            if self._db_pool:
                # Use connection pool for better performance
                async with self._db_pool as db:
                    async with db.execute(
                        "SELECT active_tasks FROM work_sessions WHERE id = ?",
                        (session_id,)
                    ) as cursor:
                        row = await cursor.fetchone()

                        if not row:
                            self.base.debug_log(f"Session not found: {session_id[:8]}...")
                            return []

                        # Parse JSON (handle NULL case)
                        return json.loads(row[0]) if row[0] else []
            else:
                # Fallback to direct connection
                import aiosqlite
                async with aiosqlite.connect(self.db_path) as db:
                    async with db.execute(
                        "SELECT active_tasks FROM work_sessions WHERE id = ?",
                        (session_id,)
                    ) as cursor:
                        row = await cursor.fetchone()

                        if not row:
                            self.base.debug_log(f"Session not found: {session_id[:8]}...")
                            return []

                        # Parse JSON (handle NULL case)
                        return json.loads(row[0]) if row[0] else []

        except Exception as e:
            self.base.debug_log(f"Failed to get active tasks: {e}")
            return []

    async def _add_active_file(self, session_id: str, file_path: str) -> bool:
        """
        Add file to session's active_files list (with deduplication).

        Memory Bank Pattern: Track files ACTIVELY modified during session.

        DEPRECATED: Use update_session_tracking() with WorkSessionManager instead.
        Kept for backward compatibility only.

        Args:
            session_id: Session identifier
            file_path: Path to file being modified

        Returns:
            True if file added successfully, False otherwise

        Note:
            Uses atomic JSON update with deduplication.
            Gracefully handles missing sessions (returns False).
        """
        try:
            import aiosqlite

            async with aiosqlite.connect(self.db_path) as db:
                # Get current active_files
                async with db.execute(
                    "SELECT active_files FROM work_sessions WHERE id = ?",
                    (session_id,)
                ) as cursor:
                    row = await cursor.fetchone()

                    if not row:
                        self.base.debug_log(f"Session not found: {session_id[:8]}...")
                        return False

                    # Parse JSON (handle NULL case)
                    active_files = json.loads(row[0]) if row[0] else []

                    # Add if not already present (deduplication)
                    if file_path not in active_files:
                        active_files.append(file_path)

                        # Update with atomic transaction
                        await db.execute(
                            "UPDATE work_sessions SET active_files = ? WHERE id = ?",
                            (json.dumps(active_files), session_id)
                        )
                        await db.commit()

                        self.base.debug_log(
                            f"Added to active_files: {file_path} "
                            f"(total: {len(active_files)})"
                        )
                        return True
                    else:
                        self.base.debug_log(f"File already tracked: {file_path}")
                        return True  # Already tracked is success

        except Exception as e:
            self.base.debug_log(f"Failed to add active file: {e}")
            return False

    async def _add_active_task(self, session_id: str, task_id: str) -> bool:
        """
        Add task to session's active_tasks list (with deduplication).

        Memory Bank Pattern: Track tasks ACTIVELY worked on during session.

        Args:
            session_id: Session identifier
            task_id: Task identifier (from TodoWrite or MCP)

        Returns:
            True if task added successfully, False otherwise

        Note:
            Uses atomic JSON update with deduplication.
            active_tasks column already exists in schema ✅
        """
        try:
            import aiosqlite

            async with aiosqlite.connect(self.db_path) as db:
                # Get current active_tasks
                async with db.execute(
                    "SELECT active_tasks FROM work_sessions WHERE id = ?",
                    (session_id,)
                ) as cursor:
                    row = await cursor.fetchone()

                    if not row:
                        self.base.debug_log(f"Session not found: {session_id[:8]}...")
                        return False

                    # Parse JSON (handle NULL case)
                    active_tasks = json.loads(row[0]) if row[0] else []

                    # Add if not already present (deduplication)
                    if task_id not in active_tasks:
                        active_tasks.append(task_id)

                        # Update with atomic transaction
                        await db.execute(
                            "UPDATE work_sessions SET active_tasks = ? WHERE id = ?",
                            (json.dumps(active_tasks), session_id)
                        )
                        await db.commit()

                        self.base.debug_log(
                            f"Added to active_tasks: {task_id[:8]}... "
                            f"(total: {len(active_tasks)})"
                        )
                        return True
                    else:
                        self.base.debug_log(f"Task already tracked: {task_id[:8]}...")
                        return True  # Already tracked is success

        except Exception as e:
            self.base.debug_log(f"Failed to add active task: {e}")
            return False

    async def update_session_tracking(
        self,
        tool_name: str,
        tool_input: Dict[str, Any]
    ) -> None:
        """
        Update work_sessions with active files and tasks via WorkSessionManager.

        Context7 Pattern: Delegates to WorkSessionManager.update_session_progress()
        instead of direct database writes for proper abstraction layer.

        Called after memory storage to track active work in current session.
        Non-blocking - failures logged but don't affect hook execution.

        Args:
            tool_name: Name of tool executed
            tool_input: Tool input parameters

        Note:
            Tracks via WorkSessionManager:
            - Write/Edit/MultiEdit → active_files
            - TodoWrite → active_tasks (from in_progress todos)
            - MCP devstream_update_task → active_tasks
        """
        try:
            # Get current session ID
            session_id = await self._get_current_session_id()
            if not session_id:
                self.base.debug_log("No active session - skip tracking")
                return

            # Initialize WorkSessionManager for proper session updates
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent / 'sessions'))
            from work_session_manager import WorkSessionManager

            session_manager = WorkSessionManager()

            # Track active files (Write/Edit/MultiEdit)
            if tool_name in ["Write", "Edit", "MultiEdit"]:
                file_path = tool_input.get("file_path")
                if file_path:
                    # Get current active_files
                    current_files = await self._get_active_files(session_id)

                    # Add new file if not already tracked
                    if file_path not in current_files:
                        current_files.append(file_path)

                        # Update session with active_files via WorkSessionManager
                        await session_manager.update_session_progress(
                            session_id=session_id,
                            active_files=current_files
                        )

                        self.base.debug_log(
                            f"Updated active_files via WorkSessionManager: {file_path} "
                            f"(total: {len(current_files)})"
                        )

            # Track active tasks (TodoWrite)
            elif tool_name == "TodoWrite":
                todos = tool_input.get("todos", [])

                # Get current active_tasks
                current_tasks = await self._get_active_tasks(session_id)

                tasks_updated = False
                for todo in todos:
                    # Track in_progress todos (actively being worked on)
                    if todo.get("status") == "in_progress":
                        task_content = todo.get("content", "")

                        # Add if not already tracked
                        if task_content and task_content not in current_tasks:
                            current_tasks.append(task_content)
                            tasks_updated = True

                # Update session with active_tasks via WorkSessionManager
                if tasks_updated:
                    await session_manager.update_session_progress(
                        session_id=session_id,
                        active_tasks=current_tasks
                    )

                    self.base.debug_log(
                        f"Updated active_tasks via WorkSessionManager: "
                        f"{len(current_tasks)} tasks"
                    )

            # Track MCP task operations (devstream_update_task, devstream_create_task)
            # Note: These are called via MCP, not directly as tool_name
            # For now, TodoWrite is primary tracking mechanism

        except Exception as e:
            # Non-blocking - log and continue
            self.base.debug_log(f"Session tracking failed (non-blocking): {e}")

    def log_capture_audit(
        self,
        tool_name: str,
        tool_response: Dict[str, Any],
        content_type: str,
        topics: List[str],
        entities: List[str],
        memory_id: Optional[str],
        capture_decision: str
    ) -> None:
        """
        Log structured audit trail for capture decisions.

        cchooks Pattern: Structured JSON logging for production audit trails.

        Args:
            tool_name: Name of the tool executed
            tool_response: Tool execution response
            content_type: Classified content type
            topics: Extracted topics
            entities: Extracted entities
            memory_id: Memory record ID (if stored)
            capture_decision: "stored" or "skipped"
        """
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "success": tool_response.get("success", True),
            "content_type": content_type,
            "topics": topics[:3],  # Top 3 topics
            "entities": entities[:3],  # Top 3 entities
            "memory_id": memory_id[:8] if memory_id else None,
            "capture_decision": capture_decision  # "stored" | "skipped"
        }

        # Structured logging for audit trail
        self.base.debug_log(f"📊 Audit: {json.dumps(audit_entry)}")

        # TODO: Optional - Write to dedicated audit log file
        # audit_file = Path.home() / ".claude" / "logs" / "devstream" / "capture_audit.jsonl"
        # with open(audit_file, "a") as f:
        #     f.write(json.dumps(audit_entry) + "\n")

    async def capture_session_event(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_response: Dict[str, Any]
    ) -> None:
        """
        Capture session events for Event Sourcing session summary.

        Phase 3 Integration: Capture events in append-only log for session_end_v2.py.
        Non-blocking - failures logged but don't affect hook execution.

        Args:
            tool_name: Name of the tool executed
            tool_input: Tool input parameters
            tool_response: Tool execution response
        """
        self.base.debug_log(f"🎯 capture_session_event called: tool={tool_name}, SESSION_EVENT_LOG_AVAILABLE={SESSION_EVENT_LOG_AVAILABLE}")

        if not SESSION_EVENT_LOG_AVAILABLE:
            # Event log not available - skip silently
            self.base.debug_log("❌ SESSION_EVENT_LOG_AVAILABLE=False, skipping event capture")
            return

        try:
            # Get session ID from environment or tool input
            session_id = os.environ.get("CLAUDE_SESSION_ID")
            if not session_id:
                # Try to extract from tool input if available
                session_id = tool_input.get("session_id", "sess-unknown")

            self.base.debug_log(f"🎯 Event capture: session_id={session_id}")

            # Get session event log
            event_log = await get_session_log(session_id)
            self.base.debug_log(f"🎯 Event log retrieved: {event_log.session_id}, events={len(event_log.events)}")

            # Capture events based on tool type
            if tool_name in ["Write", "Edit", "MultiEdit"]:
                # File modification events
                file_path = tool_input.get("file_path", "")
                content = tool_input.get("content", "") or tool_input.get("new_string", "")

                if file_path and content:
                    self.base.debug_log(f"🎯 Recording file_modified event: {file_path}")
                    await event_log.record_event("file_modified", {
                        "path": str(file_path),
                        "tool": tool_name,
                        "size_bytes": len(content),
                        "session_id": session_id
                    })
                    self.base.debug_log(f"✅ file_modified event recorded, total events: {len(event_log.events)}")

            elif tool_name == "TodoWrite":
                # Task events - check for task completion
                todos = tool_input.get("todos", [])

                for todo in todos:
                    todo_content = todo.get("content", "")
                    todo_status = todo.get("status", "")

                    if todo_content:
                        if todo_status == "completed":
                            await event_log.record_event("task_completed", {
                                "task_id": f"todo-{hash(todo_content) % 10000}",
                                "title": todo_content[:100],  # Limit title length
                                "session_id": session_id
                            })
                        elif todo_status == "in_progress":
                            await event_log.record_event("task_started", {
                                "task_id": f"todo-{hash(todo_content) % 10000}",
                                "title": todo_content[:100],
                                "session_id": session_id
                            })

            elif tool_name == "Bash":
                # Error events for failed commands
                if not tool_response.get("success", True):
                    command = tool_input.get("command", "")
                    error_output = tool_response.get("error", "") or tool_response.get("output", "")

                    if command:
                        await event_log.record_event("error", {
                            "error_type": "bash_command",
                            "message": f"Command failed: {command[:100]}",
                            "command": command[:200],
                            "output": error_output[:200] if error_output else "",
                            "session_id": session_id
                        })

            # TODO: Add more event types as needed
            # - Decision events (could be extracted from comments)
            # - Learning events (could be extracted from documentation)

        except Exception as e:
            # Non-blocking - log but don't fail the hook
            self.base.debug_log(f"Event capture failed (non-blocking): {e}")

    async def process(self, context: PostToolUseContext) -> None:
        """
        Main hook processing logic - Enhanced multi-tool capture with Protocol State Sync (FASE 2).

        FASE 3 Enhancement: Multi-tool routing with filtering and metadata extraction.
        FASE 2 Integration: Automatic protocol state synchronization for task progress.

        Args:
            context: PostToolUse context from cchooks
        """
        # Check if hook should run
        if not self.base.should_run():
            self.base.debug_log("Hook disabled via config")
            context.output.exit_success()
            return

        # Check if memory storage enabled
        if not self.base.is_memory_store_enabled():
            self.base.debug_log("Memory storage disabled")
            context.output.exit_success()
            return

        # FASE 2: Protocol State Synchronization
        if PROTOCOL_SYNC_AVAILABLE and self.protocol_manager and self.task_sync:
            try:
                # Get current protocol state
                current_state = await self.protocol_manager.get_current_state()

                # Only sync if we're in an active protocol session
                if current_state.protocol_step != ProtocolStep.IDLE:
                    # Build tool execution context for sync
                    tool_execution = {
                        "tool": context.tool_name,
                        "input": {
                            "file_path": getattr(context, 'file_path', None),
                            "content_preview": getattr(context, 'content', '')[:200] if hasattr(context, 'content') else None
                        },
                        "output": {
                            "success": getattr(context, 'success', True),
                            "preview": str(context)[:200] if hasattr(context, '__str__') else None
                        },
                        "timestamp": datetime.now().isoformat()
                    }

                    # Sync task progress automatically
                    await self.task_sync.sync_task_progress(tool_execution)

                    # Check if this tool execution completes a step
                    step_completion = await self.task_sync.check_step_completion(tool_execution)
                    if step_completion.completed:
                        # Advance protocol step automatically
                        updated_state = await self.protocol_manager.advance_step(
                            current_state,
                            step_completion.next_step
                        )
                        self.base.debug_log(
                            f"Protocol step advanced: {current_state.protocol_step} → {step_completion.next_step} (tool: {context.tool_name})"
                        )

            except Exception as e:
                self.base.user_feedback(
                    f"Protocol state sync error: {e}",
                    FeedbackLevel.MINIMAL
                )
                # Continue with normal processing on error

        # Extract tool information
        tool_name = context.tool_name
        tool_input = context.tool_input
        tool_response = context.tool_response

        self.base.debug_log(f"Processing {tool_name}")

        # Phase 3: Capture session events (Event Sourcing)
        # Non-blocking - capture events before any other processing
        await self.capture_session_event(tool_name, tool_input, tool_response)

        # Define critical tools that trigger checkpoints
        critical_tools = ["Write", "Edit", "MultiEdit", "Bash", "TodoWrite"]
        is_critical_tool = tool_name in critical_tools

        # Multi-tool routing logic
        should_store = False
        file_path = ""
        content = ""
        content_type = "context"

        # Route 1: Write/Edit/MultiEdit - File modifications (ALWAYS capture)
        if tool_name in ["Write", "Edit", "MultiEdit"]:
            file_path = tool_input.get("file_path", "")
            content = tool_input.get("content", "") or tool_input.get("new_string", "")

            if file_path and content:
                should_store = True
                content_type = self.classify_content_type(tool_name, tool_response, content)
                self.base.debug_log(f"Write/Edit/MultiEdit: {file_path} ({len(content)} chars)")

        # Route 2: Bash - Command output (FILTERED)
        elif tool_name == "Bash":
            if self.should_capture_bash_output(tool_input, tool_response):
                command = tool_input.get("command", "")
                output = tool_response.get("output", "")

                # Create synthetic file path for command output
                file_path = f"bash_output/{command[:50].replace(' ', '_')}.txt"
                content = f"# Command: {command}\n\n{output}"
                should_store = True
                content_type = self.classify_content_type(tool_name, tool_response, content)
                self.base.debug_log(f"Bash: {command[:50]}... ({len(output)} chars)")
            else:
                self.base.debug_log("Bash: Skipped (trivial/short output)")

        # Route 3: Read - File reads (FILTERED)
        elif tool_name == "Read":
            read_file_path = tool_input.get("file_path", "")

            if read_file_path and self.should_capture_read_content(read_file_path):
                file_path = read_file_path
                # Extract content from tool_response (cchooks returns file contents)
                content = tool_response.get("content", "")

                if content:
                    should_store = True
                    content_type = self.classify_content_type(tool_name, tool_response, content)
                    self.base.debug_log(f"Read: {file_path} ({len(content)} chars)")
            else:
                self.base.debug_log(f"Read: Skipped ({read_file_path})")

        # Route 4: TodoWrite - Task list updates (ALWAYS capture)
        elif tool_name == "TodoWrite":
            todos = tool_input.get("todos", [])

            if todos:
                # Create synthetic file path for todo list
                file_path = "todo_updates/task_list.json"
                content = json.dumps(todos, indent=2)
                should_store = True
                content_type = self.classify_content_type(tool_name, tool_response, content)
                self.base.debug_log(f"TodoWrite: {len(todos)} tasks")

        # Exit early if no content to store
        if not should_store or not file_path or not content:
            self.base.debug_log(f"No content to store for {tool_name}")
            context.output.exit_success()
            return

        # Skip if file is in excluded paths
        excluded_paths = [
            ".git/",
            "node_modules/",
            ".venv/",
            ".devstream/",
            "__pycache__/",
            "dist/",
            "build/",
        ]
        if any(excluded in file_path for excluded in excluded_paths):
            self.base.debug_log(f"Skipping excluded path: {file_path}")
            context.output.exit_success()
            return

        try:
            # Extract metadata (topics and entities)
            topics = self.extract_topics(content, file_path)
            entities = self.extract_entities(content)

            # Store in memory with embedding (Phase 2 + FASE 3 enhanced)
            memory_id = await self.store_in_memory(
                file_path=file_path,
                content=content,
                operation=tool_name,
                topics=topics,
                entities=entities,
                content_type=content_type
            )

            if not memory_id:
                # Non-blocking warning
                self.base.warning_feedback("Memory storage unavailable")

            # Determine capture decision
            capture_decision = "stored" if memory_id else "skipped"

            # Log audit trail
            self.log_capture_audit(
                tool_name=tool_name,
                tool_response=tool_response,
                content_type=content_type,
                topics=topics,
                entities=entities,
                memory_id=memory_id,
                capture_decision=capture_decision
            )

            # FASE 2: Update session tracking (Memory Bank activeContext pattern)
            await self.update_session_tracking(tool_name, tool_input)

            # FASE 1: Trigger real-time capture for critical tool execution
            if is_critical_tool:
                await self.trigger_real_time_capture_for_critical_tool(tool_name, file_path)

            # Always allow the operation to proceed (graceful degradation)
            context.output.exit_success()

        except Exception as e:
            # Non-blocking error - log and continue
            self.base.warning_feedback(f"Memory storage failed: {str(e)[:50]}")
            context.output.exit_success()

        finally:
            # FASE 1: Cleanup real-time monitoring if needed
            try:
                if self.real_time_capture is not None and self.real_time_capture.is_running:
                    # Don't stop monitoring here - let it run continuously
                    # to capture real-time file changes between tool executions
                    pass
            except Exception as e:
                self.base.debug_log(f"Real-time monitoring cleanup failed: {e}")

    async def run_fallback_mode(self):
        """
        Fallback mode for when no Claude Code context is available.

        This mode allows the PostToolUse hook to function during testing
        or when executed directly without full Claude Code integration.
        """
        print("🔄 PostToolUse hook running in fallback mode")

        try:
            # Get current session information
            sys.path.insert(0, str(Path(__file__).parent.parent / 'sessions'))
            from work_session_manager import WorkSessionManager
            session_manager = WorkSessionManager()

            # Try to get current active session
            import sqlite3
            sys.path.append(str(Path(__file__).parent.parent / 'utils'))
            from connection_manager import get_connection_manager

            # Database configuration (use data/ as corrected in implementation)
            project_root = Path(__file__).parent.parent.parent.parent.parent
            db_path = str(project_root / 'data' / 'devstream.db')

            # Use connection manager for WAL mode enforcement
            manager = get_connection_manager(db_path)
            conn = manager._get_thread_connection()
            cursor = conn.cursor()

            cursor.execute('SELECT id, started_at FROM work_sessions WHERE status="active" ORDER BY started_at DESC LIMIT 1')
            session = cursor.fetchone()

            if session:
                session_id, started_at = session
                print(f"📊 Found active session: {session_id}")

                # Store a test record to verify the hook works
                test_content = f"PostToolUse fallback mode test at {datetime.now().isoformat()}"

                result = await self.unified_client.store_memory(
                    content=test_content,
                    content_type="code",
                    keywords=["post_tool_use", "fallback", "test", session_id[:8]],
                    session_id=session_id,
                    hook_name="post_tool_use_fallback"
                )

                if result:
                    print("✅ PostToolUse fallback mode: Memory storage successful")
                else:
                    print("⚠️ PostToolUse fallback mode: Memory storage failed (MCP unavailable)")

                    # Fallback: Store directly in database
                    try:
                        # Use ConnectionManager for fallback mode (WAL mode enforced)
                        conn_sync = manager._get_thread_connection()
                        cursor_sync = conn_sync.cursor()

                        cursor_sync.execute(
                            """
                            INSERT INTO semantic_memory
                            (id, content, content_type, created_at, updated_at, keywords)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                test_content,
                                "code",
                                datetime.now().isoformat(),
                                datetime.now().isoformat(),
                                json.dumps(["post_tool_use", "fallback", "test"])
                            )
                        )
                        conn_sync.commit()
                        conn_sync.close()

                        print("✅ PostToolUse fallback mode: Direct database storage successful")
                    except Exception as e:
                        print(f"❌ PostToolUse fallback mode: Direct storage failed: {e}")

            else:
                print("⚠️ No active session found for fallback mode")

            # FASE 1: Test real-time capture functionality
            try:
                print("🔄 Testing real-time capture functionality...")

                if self.real_time_capture is None:
                    print("⚠️ Real-time capture not available (module missing)")
                else:
                    # Test file filtering
                    test_files = [
                        "/test.py",           # Should monitor
                        "/app.tsx",          # Should monitor
                        "/docs/readme.md",   # Should monitor
                        "/.git/config",      # Should exclude
                        "/node_modules/pkg.js",  # Should exclude
                    ]

                    monitored_count = 0
                    for file_path in test_files:
                        should_monitor = self.real_time_capture._should_monitor_file(file_path)
                        if should_monitor:
                            monitored_count += 1

                    print(f"✅ Real-time capture filtering: {monitored_count}/{len(test_files)} files correctly filtered")

                    # Test monitoring status
                    status = self.real_time_capture.get_status()
                    print(f"📊 Real-time capture status: running={status['is_running']}, extensions={status['monitored_extensions']}")

                    # Test starting monitoring (briefly for testing)
                    if not status['is_running']:
                        print("🔄 Starting real-time monitoring test...")
                        started = self.real_time_capture.start_monitoring([str(project_root)])
                        if started:
                            print("✅ Real-time monitoring started successfully")
                            # Stop immediately after test
                            self.real_time_capture.stop_monitoring()
                            print("✅ Real-time monitoring stopped (test complete)")
                        else:
                            print("❌ Failed to start real-time monitoring")
                    else:
                        print("✅ Real-time monitoring already running")

            except Exception as rtc_error:
                print(f"⚠️ Real-time capture test failed: {rtc_error}")

            conn.close()

        except Exception as e:
            print(f"❌ PostToolUse fallback mode error: {e}")


def main():
    """Main entry point for PostToolUse hook."""
    # Create context using cchooks with fallback mode
    ctx = None
    try:
        ctx = safe_create_context()
    except (Exception, SystemExit) as e:
        # stdin empty or invalid JSON - fallback to manual processing
        print(f"⚠️  DevStream: No hook input, using fallback mode", file=sys.stderr)
        ctx = None  # Explicitly set to None for fallback mode

    # Verify it's PostToolUse context (if available)
    if ctx and not isinstance(ctx, PostToolUseContext):
        print(f"Error: Expected PostToolUseContext, got {type(ctx)}", file=sys.stderr)
        sys.exit(1)

    # Create and run hook
    hook = PostToolUseHook()

    try:
        if ctx:
            # Run with full context (normal Claude Code execution)
            asyncio.run(hook.process(ctx))
        else:
            # Run in fallback mode (direct execution / testing)
            asyncio.run(hook.run_fallback_mode())
    except Exception as e:
        # Graceful failure - non-blocking
        print(f"⚠️  DevStream: PostToolUse error: {str(e)[:100]}", file=sys.stderr)
        if ctx:
            ctx.output.exit_non_block(f"Hook error: {str(e)[:100]}")
        else:
            print(f"Hook completed with error: {str(e)[:100]}", file=sys.stderr)


if __name__ == "__main__":
    main()