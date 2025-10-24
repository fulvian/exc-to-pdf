"""
Memory Manager - High-level interface for DevStream Memory System

Provides convenient access to all memory system components including
the new RAG quality evaluation framework.
"""

import logging
from typing import Any, Dict, List, Optional

import structlog

from .storage import MemoryStorage
from .processing import TextProcessor
from .search import HybridSearchEngine
from .context import ContextAssembler
from .embedding_generator import EmbeddingGenerator, EmbeddingConfig
from .quality_evaluator import (
    RAGMetricsEvaluator,
    EvaluationQuery,
    EvaluationDataset,
    EvaluationReport,
    MetricType,
)
from ..database.connection import ConnectionPool

logger = structlog.get_logger()


class MemoryManager:
    """
    High-level manager for DevStream Memory System components.

    Provides unified access to storage, search, context assembly,
    embedding generation, and quality evaluation capabilities.
    """

    def __init__(
        self,
        connection_pool: ConnectionPool,
        embedding_config: Optional[EmbeddingConfig] = None,
        enable_quality_evaluator: bool = True
    ):
        """
        Initialize memory manager with all components.

        Args:
            connection_pool: Database connection pool
            embedding_config: Configuration for embedding generation
            enable_quality_evaluator: Whether to initialize quality evaluator
        """
        self.connection_pool = connection_pool
        self.embedding_config = embedding_config or EmbeddingConfig()

        # Initialize core components
        self.storage = MemoryStorage(connection_pool, self.embedding_config)
        self.embedding_generator = self.storage.embedding_generator
        self.text_processor = TextProcessor(self.embedding_generator._client)
        self.search_engine = HybridSearchEngine(self.storage, self.text_processor)
        self.context_assembler = ContextAssembler(self.search_engine)

        # Initialize quality evaluator if enabled
        self.quality_evaluator = None
        if enable_quality_evaluator:
            try:
                self.quality_evaluator = RAGMetricsEvaluator(
                    storage=self.storage,
                    search_engine=self.search_engine,
                    embedding_generator=self.embedding_generator,
                    embedding_config=self.embedding_config
                )
                logger.info("Quality evaluator initialized successfully")
            except Exception as e:
                logger.warning("Failed to initialize quality evaluator", error=str(e))

        logger.info("MemoryManager initialized",
                   quality_evaluator_enabled=self.quality_evaluator is not None)

    async def initialize(self) -> None:
        """
        Initialize memory system components.

        Creates virtual tables and prepares the system for use.
        """
        try:
            await self.storage.create_virtual_tables()
            logger.info("Memory system initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize memory system", error=str(e))
            raise

    async def store_memory_entry(self, content: str, content_type: str = "context", **kwargs) -> str:
        """
        Store a memory entry with automatic processing.

        Args:
            content: Content to store
            content_type: Type of content
            **kwargs: Additional memory entry fields

        Returns:
            ID of stored memory entry
        """
        from .models import MemoryEntry, ContentType

        import uuid
        from datetime import datetime

        memory_id = kwargs.get('id', str(uuid.uuid4()))

        memory = MemoryEntry(
            id=memory_id,
            content=content,
            content_type=ContentType(content_type),
            **kwargs
        )

        # Store with embedding generation
        processed_entries = await self.storage.store_memories_with_embeddings([memory])
        return processed_entries[0].id if processed_entries else memory_id

    async def search_memories(self, query_text: str, max_results: int = 10, **kwargs) -> List[Any]:
        """
        Search memories using the hybrid search engine.

        Args:
            query_text: Search query
            max_results: Maximum number of results
            **kwargs: Additional search parameters

        Returns:
            List of search results
        """
        from .models import SearchQuery

        query = SearchQuery(
            query_text=query_text,
            max_results=max_results,
            **kwargs
        )

        return await self.search_engine.search(query)

    async def assemble_context(self, query_text: str, token_budget: int = 2000, **kwargs) -> Dict[str, Any]:
        """
        Assemble context for a query.

        Args:
            query_text: Query for context assembly
            token_budget: Maximum tokens for assembled context
            **kwargs: Additional context assembly parameters

        Returns:
            Context assembly result
        """
        return await self.context_assembler.assemble_context(
            query_text, token_budget, **kwargs
        )

    async def evaluate_query_quality(
        self,
        query: str,
        ground_truth_answer: str,
        generated_answer: Optional[str] = None,
        max_contexts: int = 5,
        metrics: Optional[List[MetricType]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate quality of a single query against the memory system.

        Args:
            query: Query to evaluate
            ground_truth_answer: Reference answer
            generated_answer: Generated answer (optional)
            max_contexts: Maximum contexts to retrieve
            metrics: Metrics to evaluate

        Returns:
            Evaluation results for the query
        """
        if not self.quality_evaluator:
            raise RuntimeError("Quality evaluator not initialized")

        # Create evaluation dataset from memory system
        dataset = await self.quality_evaluator.create_evaluation_from_memory_system(
            queries=[query],
            ground_truth_answers=[ground_truth_answer],
            max_contexts_per_query=max_contexts
        )

        # Set generated answer if provided
        if generated_answer:
            dataset.queries[0].generated_answer = generated_answer

        # Evaluate the query
        query_results = await self.quality_evaluator.evaluate_query(dataset.queries[0], metrics)

        return {
            'query': query,
            'query_id': dataset.queries[0].query_id,
            'ground_truth_answer': ground_truth_answer,
            'generated_answer': generated_answer,
            'retrieved_contexts_count': len(dataset.queries[0].retrieved_contexts),
            'metrics': {name: {
                'score': result.score,
                'reasoning': result.reasoning,
                'execution_time_ms': result.execution_time_ms,
                'error': result.error
            } for name, result in query_results.items()}
        }

    async def evaluate_batch_quality(
        self,
        queries: List[str],
        ground_truth_answers: List[str],
        generated_answers: Optional[List[str]] = None,
        max_contexts: int = 5,
        metrics: Optional[List[MetricType]] = None
    ) -> EvaluationReport:
        """
        Evaluate quality of multiple queries against the memory system.

        Args:
            queries: List of queries to evaluate
            ground_truth_answers: List of reference answers
            generated_answers: List of generated answers (optional)
            max_contexts: Maximum contexts to retrieve per query
            metrics: Metrics to evaluate

        Returns:
            Comprehensive evaluation report
        """
        if not self.quality_evaluator:
            raise RuntimeError("Quality evaluator not initialized")

        # Create evaluation dataset from memory system
        dataset = await self.quality_evaluator.create_evaluation_from_memory_system(
            queries=queries,
            ground_truth_answers=ground_truth_answers,
            max_contexts_per_query=max_contexts
        )

        # Set generated answers if provided
        if generated_answers:
            if len(generated_answers) != len(queries):
                raise ValueError("Number of generated answers must match number of queries")

            for i, generated_answer in enumerate(generated_answers):
                dataset.queries[i].generated_answer = generated_answer

        # Evaluate the dataset
        return await self.quality_evaluator.evaluate_dataset(dataset, metrics)

    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the memory system.

        Returns:
            System status information
        """
        status = {
            'storage_initialized': True,
            'embedding_generator': await self.storage.get_embedding_generator_status(),
            'search_engine': True,
            'context_assembler': True,
        }

        if self.quality_evaluator:
            status['quality_evaluator'] = await self.quality_evaluator.get_evaluation_status()
        else:
            status['quality_evaluator'] = {'enabled': False}

        # Get memory statistics
        try:
            # This would require adding a method to get memory count
            # For now, include basic status
            status['memory_statistics'] = {
                'status': 'operational'
            }
        except Exception as e:
            status['memory_statistics'] = {'error': str(e)}

        return status

    async def cleanup(self) -> None:
        """
        Cleanup resources and close connections.
        """
        try:
            if self.connection_pool:
                await self.connection_pool.close()
            logger.info("MemoryManager cleanup completed")
        except Exception as e:
            logger.error("Failed to cleanup MemoryManager", error=str(e))