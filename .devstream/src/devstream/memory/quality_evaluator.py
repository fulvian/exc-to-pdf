"""
RAG Metrics Quality Evaluator for DevStream Memory System

Context7-Ragas inspired implementation for evaluating retrieval-augmented generation
quality with Faithfulness, ContextPrecision, AnswerRelevancy, and ContextRecall metrics.

Uses Ollama embeddinggemma:300m for semantic similarity calculations with async
performance optimization and comprehensive error handling.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import ollama
import structlog
from pydantic import BaseModel, Field

from .models import MemoryEntry, MemoryQueryResult, SearchQuery
from .search import HybridSearchEngine
from .storage import MemoryStorage
from .embedding_generator import EmbeddingGenerator, EmbeddingConfig
from .exceptions import SearchError, StorageError, VectorSearchError

logger = structlog.get_logger()


class MetricType(str, Enum):
    """Types of RAG evaluation metrics."""
    FAITHFULNESS = "faithfulness"
    CONTEXT_PRECISION = "context_precision"
    ANSWER_RELEVANCY = "answer_relevancy"
    CONTEXT_RECALL = "context_recall"


@dataclass
class MetricResult:
    """
    Result of a single metric evaluation.

    Context7-Ragas pattern for structured metric results with detailed
    scoring information and metadata for analysis.
    """
    metric_type: MetricType
    score: float
    reasoning: Optional[str] = None
    execution_time_ms: float = 0.0
    error: Optional[str] = None

    def __post_init__(self):
        """Validate metric result constraints."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Metric score must be between 0 and 1, got {self.score}")


@dataclass
class EvaluationQuery:
    """
    Query for RAG evaluation with ground truth.

    Represents a complete RAG evaluation scenario with query, expected answer,
    and retrieved context for comprehensive quality assessment.
    """
    query_text: str
    ground_truth_answer: str
    retrieved_contexts: List[str]
    generated_answer: Optional[str] = None
    query_id: Optional[str] = None

    def __post_init__(self):
        """Validate evaluation query structure."""
        if not self.query_text.strip():
            raise ValueError("Query text cannot be empty")
        if not self.ground_truth_answer.strip():
            raise ValueError("Ground truth answer cannot be empty")
        if not self.retrieved_contexts:
            raise ValueError("At least one retrieved context is required")


class EvaluationDataset(BaseModel):
    """
    Dataset for RAG metrics evaluation.

    Context7-Ragas pattern for structured evaluation datasets with
    comprehensive validation and metadata support.
    """
    queries: List[EvaluationQuery] = Field(..., description="Evaluation queries")
    name: str = Field(..., description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    created_at: float = Field(default_factory=time.time, description="Creation timestamp")

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class EvaluationReport(BaseModel):
    """
    Comprehensive evaluation report.

    Context7-Ragas inspired report structure with aggregate metrics,
    per-query details, and performance analysis.
    """
    dataset_name: str
    total_queries: int
    successful_evaluations: int

    # Aggregate metrics
    faithfulness_score: float
    context_precision_score: float
    answer_relevancy_score: float
    context_recall_score: float
    overall_score: float

    # Detailed results
    metric_results: List[MetricResult]
    query_results: List[Dict[str, Any]]

    # Performance metrics
    total_execution_time_ms: float
    average_query_time_ms: float

    # Metadata
    embedding_model: str
    evaluation_timestamp: float = Field(default_factory=time.time)

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class RAGMetricsEvaluator:
    """
    RAG metrics evaluator inspired by Context7-Ragas best practices 2025.

    Implements comprehensive evaluation framework for DevStream memory system
    with async performance optimization and Ollama embedding integration.

    Metrics implemented:
    - Faithfulness: Measures factual consistency of generated answer with retrieved context
    - ContextPrecision: Measures signal-to-noise ratio in retrieved context
    - AnswerRelevancy: Measures relevance of generated answer to the original query
    - ContextRecall: Measures coverage of ground truth answer in retrieved context
    """

    def __init__(
        self,
        storage: MemoryStorage,
        search_engine: HybridSearchEngine,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        embedding_config: Optional[EmbeddingConfig] = None
    ):
        """
        Initialize RAG metrics evaluator with required components.

        Args:
            storage: Memory storage instance for retrieval operations
            search_engine: Hybrid search engine for context retrieval
            embedding_generator: Optional embedding generator for semantic similarity
            embedding_config: Configuration for embedding generation
        """
        self.storage = storage
        self.search_engine = search_engine

        # Initialize embedding generator for semantic similarity calculations
        if embedding_generator:
            self.embedding_generator = embedding_generator
        elif embedding_config:
            self.embedding_generator = EmbeddingGenerator(storage.connection_pool, embedding_config)
        else:
            # Use embeddinggemma:300m as default for performance
            default_config = EmbeddingConfig(model_name="embeddinggemma")
            self.embedding_generator = EmbeddingGenerator(storage.connection_pool, default_config)

        # Initialize Ollama client for LLM-based evaluations
        self._llm_client = ollama.Client(host='http://localhost:11434')
        self._llm_model = "phi3.5:3.8b"  # Use available model for LLM evaluations

        logger.info("RAGMetricsEvaluator initialized",
                   embedding_model=self.embedding_generator.config.model_name,
                   llm_model=self._llm_model)

    async def evaluate_faithfulness(
        self,
        generated_answer: str,
        retrieved_contexts: List[str]
    ) -> MetricResult:
        """
        Evaluate Faithfulness metric.

        Measures the factual consistency of the generated answer with the retrieved context.
        Higher scores indicate better factual consistency without contradictions.

        Context7-Ragas pattern: Uses LLM to identify statements in the generated answer
        and verify their presence/support in the retrieved context.

        Args:
            generated_answer: Answer generated by RAG system
            retrieved_contexts: List of retrieved context passages

        Returns:
            Faithfulness metric result with score and reasoning
        """
        start_time = time.time()

        try:
            # Combine retrieved contexts for analysis
            combined_context = "\n\n".join(retrieved_contexts)

            # Context7-Ragas faithfulness evaluation prompt
            evaluation_prompt = f"""
            Evaluate the faithfulness of the given answer based solely on the provided context.

            Context:
            {combined_context}

            Answer:
            {generated_answer}

            Instructions:
            1. Identify all factual statements in the answer
            2. Verify each statement against the context
            3. Check for any contradictions or unsupported claims
            4. Calculate the ratio of supported statements to total statements

            Provide your evaluation in this format:
            Score: [0.0-1.0]
            Reasoning: [Detailed explanation of your evaluation]

            Score guidelines:
            - 1.0: All statements are supported by context, no contradictions
            - 0.8-0.9: Most statements supported, minor unsupported details
            - 0.6-0.7: Some statements supported, notable unsupported claims
            - 0.4-0.5: Few statements supported, many unsupported claims
            - 0.0-0.3: Almost no statements supported or major contradictions
            """

            # Generate evaluation using Ollama
            response = self._llm_client.generate(
                model=self._llm_model,
                prompt=evaluation_prompt
            )

            response_text = response['response']

            # Parse score and reasoning from response
            score = self._parse_score_from_response(response_text)
            reasoning = self._parse_reasoning_from_response(response_text)

            execution_time = (time.time() - start_time) * 1000

            logger.debug("Faithfulness evaluation completed",
                        score=score,
                        execution_time_ms=execution_time)

            return MetricResult(
                metric_type=MetricType.FAITHFULNESS,
                score=score,
                reasoning=reasoning,
                execution_time_ms=execution_time
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error("Faithfulness evaluation failed",
                        error=str(e),
                        execution_time_ms=execution_time)

            return MetricResult(
                metric_type=MetricType.FAITHFULNESS,
                score=0.0,
                execution_time_ms=execution_time,
                error=str(e)
            )

    async def evaluate_context_precision(
        self,
        query: str,
        retrieved_contexts: List[str]
    ) -> MetricResult:
        """
        Evaluate Context Precision metric.

        Measures the signal-to-noise ratio in the retrieved context.
        Higher scores indicate that more relevant contexts appear earlier in the results.

        Context7-Ragas pattern: Uses LLM to judge relevance of each context
        to the original query and calculates precision@k.

        Args:
            query: Original query
            retrieved_contexts: List of retrieved context passages in order

        Returns:
            ContextPrecision metric result with score and reasoning
        """
        start_time = time.time()

        try:
            if not retrieved_contexts:
                return MetricResult(
                    metric_type=MetricType.CONTEXT_PRECISION,
                    score=0.0,
                    reasoning="No contexts provided for evaluation",
                    execution_time_ms=(time.time() - start_time) * 1000
                )

            # Evaluate relevance of each context
            relevance_scores = []

            for i, context in enumerate(retrieved_contexts):
                evaluation_prompt = f"""
                Evaluate the relevance of this context to the query.

                Query:
                {query}

                Context:
                {context}

                Instructions:
                1. Determine if this context contains information relevant to answering the query
                2. Consider whether the context provides useful facts, examples, or explanations
                3. Ignore irrelevant information or noise

                Respond with only: RELEVANT or IRRELEVANT
                """

                response = self._llm_client.generate(
                    model=self._llm_model,
                    prompt=evaluation_prompt
                )

                response_text = response['response'].strip().upper()
                is_relevant = "RELEVANT" in response_text
                relevance_scores.append(1 if is_relevant else 0)

            # Calculate precision@k following Context7-Ragas formula
            precision_at_k_scores = []
            relevant_so_far = 0

            for k, relevance in enumerate(relevance_scores, 1):
                if relevance:
                    relevant_so_far += 1
                precision_at_k = relevant_so_far / k
                precision_at_k_scores.append(precision_at_k)

            # Average precision across all contexts
            final_score = sum(precision_at_k_scores) / len(precision_at_k_scores)

            # Generate reasoning
            relevant_count = sum(relevance_scores)
            reasoning = f"{relevant_count}/{len(retrieved_contexts)} contexts were relevant. " \
                       f"Average precision@k: {final_score:.3f}"

            execution_time = (time.time() - start_time) * 1000

            logger.debug("Context precision evaluation completed",
                        score=final_score,
                        relevant_count=relevant_count,
                        total_contexts=len(retrieved_contexts),
                        execution_time_ms=execution_time)

            return MetricResult(
                metric_type=MetricType.CONTEXT_PRECISION,
                score=final_score,
                reasoning=reasoning,
                execution_time_ms=execution_time
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error("Context precision evaluation failed",
                        error=str(e),
                        execution_time_ms=execution_time)

            return MetricResult(
                metric_type=MetricType.CONTEXT_PRECISION,
                score=0.0,
                execution_time_ms=execution_time,
                error=str(e)
            )

    async def evaluate_answer_relevancy(
        self,
        query: str,
        generated_answer: str
    ) -> MetricResult:
        """
        Evaluate Answer Relevancy metric.

        Measures the relevance of the generated answer to the original query.
        Higher scores indicate that the answer directly addresses the query.

        Context7-Ragas pattern: Uses semantic similarity between query and answer
        with LLM-based relevance assessment.

        Args:
            query: Original query
            generated_answer: Answer generated by RAG system

        Returns:
            AnswerRelevancy metric result with score and reasoning
        """
        start_time = time.time()

        try:
            # Method 1: Semantic similarity using embeddings
            semantic_score = await self._calculate_semantic_similarity(query, generated_answer)

            # Method 2: LLM-based relevance assessment
            evaluation_prompt = f"""
            Evaluate the relevance of this answer to the query.

            Query:
            {query}

            Answer:
            {generated_answer}

            Instructions:
            1. Does the answer directly address the question asked?
            2. Is the answer on-topic or does it deviate?
            3. Does the answer provide useful information for the query?
            4. Is the answer complete enough to satisfy the query?

            Provide your evaluation in this format:
            Score: [0.0-1.0]
            Reasoning: [Brief explanation of relevance assessment]

            Score guidelines:
            - 1.0: Answer perfectly addresses the query
            - 0.8-0.9: Answer addresses the query well with minor gaps
            - 0.6-0.7: Answer partially addresses the query
            - 0.4-0.5: Answer loosely related to the query
            - 0.0-0.3: Answer does not address the query
            """

            response = self._llm_client.generate(
                model=self._llm_model,
                prompt=evaluation_prompt
            )

            response_text = response['response']
            llm_score = self._parse_score_from_response(response_text)
            llm_reasoning = self._parse_reasoning_from_response(response_text)

            # Combine semantic and LLM scores (weighted average)
            # Give more weight to LLM assessment for relevance
            final_score = 0.3 * semantic_score + 0.7 * llm_score

            reasoning = f"LLM assessment: {llm_reasoning}. " \
                       f"Semantic similarity: {semantic_score:.3f}. " \
                       f"Combined score: {final_score:.3f}"

            execution_time = (time.time() - start_time) * 1000

            logger.debug("Answer relevancy evaluation completed",
                        score=final_score,
                        semantic_score=semantic_score,
                        llm_score=llm_score,
                        execution_time_ms=execution_time)

            return MetricResult(
                metric_type=MetricType.ANSWER_RELEVANCY,
                score=final_score,
                reasoning=reasoning,
                execution_time_ms=execution_time
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error("Answer relevancy evaluation failed",
                        error=str(e),
                        execution_time_ms=execution_time)

            return MetricResult(
                metric_type=MetricType.ANSWER_RELEVANCY,
                score=0.0,
                execution_time_ms=execution_time,
                error=str(e)
            )

    async def evaluate_context_recall(
        self,
        ground_truth_answer: str,
        retrieved_contexts: List[str]
    ) -> MetricResult:
        """
        Evaluate Context Recall metric.

        Measures the coverage of the ground truth answer in the retrieved context.
        Higher scores indicate that the retrieved context contains more information
        needed to construct the ground truth answer.

        Context7-Ragas pattern: Uses LLM to identify key claims in ground truth
        and verify their presence in retrieved context.

        Args:
            ground_truth_answer: Reference answer for comparison
            retrieved_contexts: List of retrieved context passages

        Returns:
            ContextRecall metric result with score and reasoning
        """
        start_time = time.time()

        try:
            # Combine retrieved contexts
            combined_context = "\n\n".join(retrieved_contexts)

            # Context7-Ragas context recall evaluation prompt
            evaluation_prompt = f"""
            Evaluate the context recall by checking how much of the ground truth answer
            can be constructed from the retrieved context.

            Ground Truth Answer:
            {ground_truth_answer}

            Retrieved Context:
            {combined_context}

            Instructions:
            1. Identify the key facts, claims, and information in the ground truth answer
            2. For each key point, check if it's present in the retrieved context
            3. Calculate the ratio of covered points to total key points
            4. Consider both explicit mentions and strong implications

            Provide your evaluation in this format:
            Score: [0.0-1.0]
            Reasoning: [Explanation of what was covered and what was missing]

            Score guidelines:
            - 1.0: All key information from ground truth is present in context
            - 0.8-0.9: Most key information present, minor details missing
            - 0.6-0.7: Some key information present, notable gaps
            - 0.4-0.5: Few key points covered, major information missing
            - 0.0-0.3: Almost no key information from ground truth in context
            """

            response = self._llm_client.generate(
                model=self._llm_model,
                prompt=evaluation_prompt
            )

            response_text = response['response']

            # Parse score and reasoning
            score = self._parse_score_from_response(response_text)
            reasoning = self._parse_reasoning_from_response(response_text)

            execution_time = (time.time() - start_time) * 1000

            logger.debug("Context recall evaluation completed",
                        score=score,
                        execution_time_ms=execution_time)

            return MetricResult(
                metric_type=MetricType.CONTEXT_RECALL,
                score=score,
                reasoning=reasoning,
                execution_time_ms=execution_time
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error("Context recall evaluation failed",
                        error=str(e),
                        execution_time_ms=execution_time)

            return MetricResult(
                metric_type=MetricType.CONTEXT_RECALL,
                score=0.0,
                execution_time_ms=execution_time,
                error=str(e)
            )

    async def evaluate_query(
        self,
        evaluation_query: EvaluationQuery,
        metrics: Optional[List[MetricType]] = None
    ) -> Dict[str, MetricResult]:
        """
        Evaluate a single query across specified metrics.

        Args:
            evaluation_query: Query with ground truth and context
            metrics: List of metrics to evaluate (default: all metrics)

        Returns:
            Dictionary mapping metric names to evaluation results
        """
        if metrics is None:
            metrics = list(MetricType)

        logger.info("Starting query evaluation",
                   query_id=evaluation_query.query_id,
                   metrics=[m.value for m in metrics])

        results = {}

        # Evaluate each metric
        for metric_type in metrics:
            try:
                if metric_type == MetricType.FAITHFULNESS:
                    if not evaluation_query.generated_answer:
                        results[metric_type.value] = MetricResult(
                            metric_type=metric_type,
                            score=0.0,
                            reasoning="No generated answer provided for faithfulness evaluation"
                        )
                    else:
                        results[metric_type.value] = await self.evaluate_faithfulness(
                            evaluation_query.generated_answer,
                            evaluation_query.retrieved_contexts
                        )

                elif metric_type == MetricType.CONTEXT_PRECISION:
                    results[metric_type.value] = await self.evaluate_context_precision(
                        evaluation_query.query_text,
                        evaluation_query.retrieved_contexts
                    )

                elif metric_type == MetricType.ANSWER_RELEVANCY:
                    if not evaluation_query.generated_answer:
                        results[metric_type.value] = MetricResult(
                            metric_type=metric_type,
                            score=0.0,
                            reasoning="No generated answer provided for answer relevancy evaluation"
                        )
                    else:
                        results[metric_type.value] = await self.evaluate_answer_relevancy(
                            evaluation_query.query_text,
                            evaluation_query.generated_answer
                        )

                elif metric_type == MetricType.CONTEXT_RECALL:
                    results[metric_type.value] = await self.evaluate_context_recall(
                        evaluation_query.ground_truth_answer,
                        evaluation_query.retrieved_contexts
                    )

            except Exception as e:
                logger.error("Metric evaluation failed",
                           metric_type=metric_type.value,
                           query_id=evaluation_query.query_id,
                           error=str(e))

                results[metric_type.value] = MetricResult(
                    metric_type=metric_type,
                    score=0.0,
                    error=str(e)
                )

        return results

    async def evaluate_dataset(
        self,
        dataset: EvaluationDataset,
        metrics: Optional[List[MetricType]] = None,
        max_concurrent_evaluations: int = 5
    ) -> EvaluationReport:
        """
        Evaluate a complete dataset of queries.

        Args:
            dataset: Dataset of evaluation queries
            metrics: List of metrics to evaluate (default: all metrics)
            max_concurrent_evaluations: Maximum concurrent query evaluations

        Returns:
            Comprehensive evaluation report
        """
        start_time = time.time()

        logger.info("Starting dataset evaluation",
                   dataset_name=dataset.name,
                   total_queries=len(dataset.queries),
                   metrics=metrics)

        if not dataset.queries:
            raise ValueError("Dataset contains no queries to evaluate")

        # Process queries in batches for performance
        all_metric_results = []
        query_results = []
        successful_evaluations = 0

        # Create semaphore to limit concurrent evaluations
        semaphore = asyncio.Semaphore(max_concurrent_evaluations)

        async def evaluate_single_query(eval_query: EvaluationQuery, index: int) -> Tuple[int, Dict[str, MetricResult], bool]:
            async with semaphore:
                try:
                    results = await self.evaluate_query(eval_query, metrics)
                    return index, results, True
                except Exception as e:
                    logger.error("Query evaluation failed",
                               query_index=index,
                               query_id=eval_query.query_id,
                               error=str(e))
                    return index, {}, False

        # Execute all evaluations
        tasks = [
            evaluate_single_query(query, i)
            for i, query in enumerate(dataset.queries)
        ]

        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for task_result in completed_tasks:
            if isinstance(task_result, Exception):
                logger.error("Task failed", error=str(task_result))
                continue

            index, results, success = task_result
            if success and results:
                successful_evaluations += 1

                # Store individual metric results
                for metric_name, metric_result in results.items():
                    all_metric_results.append(metric_result)

                # Store query-level results
                query_results.append({
                    'query_index': index,
                    'query_id': dataset.queries[index].query_id,
                    'metrics': {name: result.score for name, result in results.items()},
                    'execution_times': {name: result.execution_time_ms for name, result in results.items()}
                })

        # Calculate aggregate scores
        aggregate_scores = self._calculate_aggregate_scores(all_metric_results, metrics)

        total_execution_time = (time.time() - start_time) * 1000
        average_query_time = total_execution_time / len(dataset.queries) if dataset.queries else 0

        # Create evaluation report
        report = EvaluationReport(
            dataset_name=dataset.name,
            total_queries=len(dataset.queries),
            successful_evaluations=successful_evaluations,
            faithfulness_score=aggregate_scores.get(MetricType.FAITHFULNESS.value, 0.0),
            context_precision_score=aggregate_scores.get(MetricType.CONTEXT_PRECISION.value, 0.0),
            answer_relevancy_score=aggregate_scores.get(MetricType.ANSWER_RELEVANCY.value, 0.0),
            context_recall_score=aggregate_scores.get(MetricType.CONTEXT_RECALL.value, 0.0),
            overall_score=sum(aggregate_scores.values()) / len(aggregate_scores) if aggregate_scores else 0.0,
            metric_results=all_metric_results,
            query_results=query_results,
            total_execution_time_ms=total_execution_time,
            average_query_time_ms=average_query_time,
            embedding_model=self.embedding_generator.config.model_name
        )

        logger.info("Dataset evaluation completed",
                   dataset_name=dataset.name,
                   overall_score=report.overall_score,
                   success_rate=successful_evaluations / len(dataset.queries),
                   total_time_ms=total_execution_time)

        return report

    async def create_evaluation_from_memory_system(
        self,
        queries: List[str],
        ground_truth_answers: List[str],
        max_contexts_per_query: int = 5
    ) -> EvaluationDataset:
        """
        Create evaluation dataset by retrieving contexts from memory system.

        Args:
            queries: List of queries to evaluate
            ground_truth_answers: List of ground truth answers
            max_contexts_per_query: Maximum contexts to retrieve per query

        Returns:
            Evaluation dataset with retrieved contexts
        """
        if len(queries) != len(ground_truth_answers):
            raise ValueError("Queries and ground truth answers must have same length")

        logger.info("Creating evaluation dataset from memory system",
                   total_queries=len(queries),
                   max_contexts=max_contexts_per_query)

        evaluation_queries = []

        for i, (query, ground_truth) in enumerate(zip(queries, ground_truth_answers)):
            try:
                # Retrieve contexts using the search engine
                search_query = SearchQuery(
                    query_text=query,
                    max_results=max_contexts_per_query,
                    semantic_weight=1.0,
                    keyword_weight=0.5
                )

                search_results = await self.search_engine.search(search_query)

                # Extract context content
                retrieved_contexts = [
                    result.memory_entry.content
                    for result in search_results
                ]

                # Create evaluation query
                eval_query = EvaluationQuery(
                    query_text=query,
                    ground_truth_answer=ground_truth,
                    retrieved_contexts=retrieved_contexts,
                    query_id=f"memory_query_{i}"
                )

                evaluation_queries.append(eval_query)

                logger.debug("Query processed for evaluation",
                           query_index=i,
                           retrieved_contexts=len(retrieved_contexts))

            except Exception as e:
                logger.error("Failed to process query for evaluation",
                           query_index=i,
                           query=query[:50] + "...",
                           error=str(e))

                # Create evaluation query with empty contexts
                eval_query = EvaluationQuery(
                    query_text=query,
                    ground_truth_answer=ground_truth,
                    retrieved_contexts=[],
                    query_id=f"memory_query_{i}_failed"
                )
                evaluation_queries.append(eval_query)

        dataset = EvaluationDataset(
            queries=evaluation_queries,
            name="Memory System Evaluation",
            description=f"Generated from {len(queries)} queries against DevStream memory system"
        )

        logger.info("Evaluation dataset created",
                   dataset_name=dataset.name,
                   total_queries=len(evaluation_queries),
                   avg_contexts_per_query=sum(len(q.retrieved_contexts) for q in evaluation_queries) / len(evaluation_queries))

        return dataset

    async def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using embeddings.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Semantic similarity score between 0 and 1
        """
        try:
            # Generate embeddings for both texts
            embedding1 = await self.embedding_generator._generate_embedding_with_retry(text1)
            embedding2 = await self.embedding_generator._generate_embedding_with_retry(text2)

            # Convert to numpy arrays
            vec1 = np.array(embedding1, dtype=np.float32)
            vec2 = np.array(embedding2, dtype=np.float32)

            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)

            # Ensure score is between 0 and 1
            similarity = max(0.0, min(1.0, similarity))

            return float(similarity)

        except Exception as e:
            logger.error("Failed to calculate semantic similarity", error=str(e))
            return 0.0

    def _parse_score_from_response(self, response_text: str) -> float:
        """
        Parse numeric score from LLM response.

        Args:
            response_text: LLM response text

        Returns:
            Parsed score between 0 and 1
        """
        import re

        # Look for "Score: X.X" pattern
        score_match = re.search(r'Score:\s*([0-9]*\.?[0-9]+)', response_text, re.IGNORECASE)

        if score_match:
            try:
                score = float(score_match.group(1))
                return max(0.0, min(1.0, score))  # Clamp between 0 and 1
            except ValueError:
                pass

        # Fallback: look for any number in the response
        number_match = re.search(r'([0-9]*\.?[0-9]+)', response_text)
        if number_match:
            try:
                score = float(number_match.group(1))
                return max(0.0, min(1.0, score))
            except ValueError:
                pass

        # Default fallback
        logger.warning("Could not parse score from response", response=response_text[:200])
        return 0.5  # Neutral score

    def _parse_reasoning_from_response(self, response_text: str) -> str:
        """
        Parse reasoning from LLM response.

        Args:
            response_text: LLM response text

        Returns:
            Extracted reasoning text
        """
        import re

        # Look for "Reasoning: ..." pattern
        reasoning_match = re.search(r'Reasoning:\s*(.+?)(?:\n\n|\Z)', response_text, re.IGNORECASE | re.DOTALL)

        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
            # Limit reasoning length for storage
            return reasoning[:500] + ("..." if len(reasoning) > 500 else "")

        # Fallback: return a portion of the response
        return response_text[:200] + ("..." if len(response_text) > 200 else "")

    def _calculate_aggregate_scores(
        self,
        metric_results: List[MetricResult],
        metrics: Optional[List[MetricType]]
    ) -> Dict[str, float]:
        """
        Calculate aggregate scores for each metric type.

        Args:
            metric_results: List of individual metric results
            metrics: List of metric types to include

        Returns:
            Dictionary mapping metric names to aggregate scores
        """
        if metrics is None:
            metrics = list(MetricType)

        # Group results by metric type
        results_by_type = {}
        for result in metric_results:
            metric_name = result.metric_type.value
            if metric_name not in results_by_type:
                results_by_type[metric_name] = []
            results_by_type[metric_name].append(result.score)

        # Calculate averages
        aggregate_scores = {}
        for metric_type in metrics:
            metric_name = metric_type.value
            if metric_name in results_by_type and results_by_type[metric_name]:
                aggregate_scores[metric_name] = sum(results_by_type[metric_name]) / len(results_by_type[metric_name])
            else:
                aggregate_scores[metric_name] = 0.0

        return aggregate_scores

    async def get_evaluation_status(self) -> Dict[str, Any]:
        """
        Get status information about the evaluator.

        Returns:
            Dictionary with evaluator status information
        """
        try:
            embedding_status = await self.embedding_generator.check_model_availability()

            return {
                "embedding_model": self.embedding_generator.config.model_name,
                "embedding_available": embedding_status,
                "llm_model": self._llm_model,
                "supported_metrics": [metric.value for metric in MetricType],
                "evaluator_ready": embedding_status,
            }

        except Exception as e:
            logger.error("Failed to get evaluator status", error=str(e))
            return {
                "error": str(e),
                "evaluator_ready": False
            }