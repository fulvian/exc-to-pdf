"""
DevStream Memory System - Semantic Memory e Hybrid Search

Sistema integrato per storage, search e retrieval di memoria semantica
con supporto per:
- Vector embeddings con sqlite-vec
- Full-text search con FTS5
- Hybrid search con Reciprocal Rank Fusion
- Context assembly con token budget management

Usage:
    from devstream.memory import MemoryStorage, TextProcessor, HybridSearchEngine, ContextAssembler
    from devstream.memory.models import MemoryEntry, SearchQuery

    # Initialize components
    storage = MemoryStorage(db_manager)
    processor = TextProcessor(ollama_client)
    search_engine = HybridSearchEngine(storage, processor)
    context_assembler = ContextAssembler(search_engine)

    # Process and store memory
    memory = MemoryEntry(id="test", content="Sample content", content_type="code")
    processed_memory = await processor.process_memory_entry(memory)
    await storage.store_memory(processed_memory)

    # Search and assemble context
    query = SearchQuery(query_text="sample search")
    results = await search_engine.search(query)
    context = await context_assembler.assemble_context("query", token_budget=1000)
"""

from .models import (
    MemoryEntry,
    MemoryQueryResult,
    SearchQuery,
    ContextAssemblyResult,
    ContentType,
    ContentFormat,
)
from .storage import MemoryStorage
from .processing import TextProcessor
from .search import HybridSearchEngine
from .context import ContextAssembler
from .embedding_generator import EmbeddingGenerator, EmbeddingConfig, EmbeddingGenerationError
from .quality_evaluator import (
    RAGMetricsEvaluator,
    EvaluationQuery,
    EvaluationDataset,
    EvaluationReport,
    MetricResult,
    MetricType,
)
from .memory_manager import MemoryManager
from .exceptions import (
    MemoryError,
    StorageError,
    SearchError,
    ProcessingError,
    ContextError,
    EmbeddingError,
    VectorSearchError,
    FTSError,
    TokenBudgetError,
)

__version__ = "1.0.0"

__all__ = [
    # Models
    "MemoryEntry",
    "MemoryQueryResult",
    "SearchQuery",
    "ContextAssemblyResult",
    "ContentType",
    "ContentFormat",
    # Core classes
    "MemoryStorage",
    "TextProcessor",
    "HybridSearchEngine",
    "ContextAssembler",
    # FASE 2: Embedding generation
    "EmbeddingGenerator",
    "EmbeddingConfig",
    "EmbeddingGenerationError",
    # RAG Quality Evaluation
    "RAGMetricsEvaluator",
    "EvaluationQuery",
    "EvaluationDataset",
    "EvaluationReport",
    "MetricResult",
    "MetricType",
    # High-level Interface
    "MemoryManager",
    # Exceptions
    "MemoryError",
    "StorageError",
    "SearchError",
    "ProcessingError",
    "ContextError",
    "EmbeddingError",
    "VectorSearchError",
    "FTSError",
    "TokenBudgetError",
]