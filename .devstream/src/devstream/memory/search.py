"""
Hybrid Search Engine con Reciprocal Rank Fusion (Optimized)

Context7-optimized hybrid search combining vector similarity and keyword search
using enhanced Reciprocal Rank Fusion with optimal weights and scoring.
"""

import asyncio
import logging
import time
from typing import List, Optional, Tuple

import numpy as np

from .models import SearchQuery, MemoryEntry, MemoryQueryResult
from .storage import MemoryStorage
from .processing import TextProcessor
from .exceptions import SearchError, VectorSearchError

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """
    Hybrid search engine con vector e keyword search.

    Combina semantic similarity (embeddings) con keyword matching (FTS5)
    usando Reciprocal Rank Fusion per optimal result ranking.
    """

    def __init__(self, storage: MemoryStorage, processor: TextProcessor):
        """
        Initialize hybrid search engine with Context7-optimized parameters.

        Args:
            storage: Memory storage instance
            processor: Text processor per query embeddings
        """
        self.storage = storage
        self.processor = processor

        # Context7-optimized RRF parameters
        self.rrf_k = 60  # Standard RRF constant
        self.weight_semantic = 1.0  # Semantic search weight
        self.weight_keyword = 1.5   # Increased keyword weight for better relevance

    async def search(self, query: SearchQuery) -> List[MemoryQueryResult]:
        """
        Execute hybrid search con RRF fusion.

        Args:
            query: Search query con parameters

        Returns:
            Ranked list di search results

        Raises:
            SearchError: Se la search fallisce
        """
        try:
            start_time = time.time()

            # Execute search strategies in parallel
            semantic_results, keyword_results = await asyncio.gather(
                self._semantic_search(query),
                self._keyword_search(query),
                return_exceptions=True
            )

            # Handle exceptions
            if isinstance(semantic_results, Exception):
                logger.warning(f"Semantic search failed: {semantic_results}")
                semantic_results = []

            if isinstance(keyword_results, Exception):
                logger.warning(f"Keyword search failed: {keyword_results}")
                keyword_results = []

            # Apply RRF fusion
            fused_results = self._reciprocal_rank_fusion(
                semantic_results, keyword_results, query
            )

            # Apply post-processing filters
            filtered_results = await self._apply_filters(fused_results, query)

            # Limit results
            final_results = filtered_results[:query.max_results]

            search_time = (time.time() - start_time) * 1000
            logger.info(
                f"Hybrid search completed in {search_time:.2f}ms: "
                f"{len(final_results)} results for '{query.query_text}'"
            )

            return final_results

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise SearchError(f"Hybrid search failed: {e}") from e

    async def _semantic_search(self, query: SearchQuery) -> List[Tuple[str, float, int]]:
        """
        Execute semantic search usando vector embeddings.

        Args:
            query: Search query

        Returns:
            List of (memory_id, similarity_score, rank) tuples

        Raises:
            VectorSearchError: Se il vector search fallisce
        """
        try:
            if query.semantic_weight == 0.0:
                return []

            # Generate query embedding
            query_features = await self.processor.process_text(
                query.query_text, include_embedding=True
            )

            if "embedding" not in query_features or not query_features["embedding"]:
                logger.warning("Failed to generate query embedding")
                return []

            query_embedding = np.array(query_features["embedding"], dtype=np.float32)

            # Search vector database
            # Note: Limiting to 2x max_results per better fusion quality
            vector_results = await self.storage.search_vectors(
                query_embedding, k=query.max_results * 2
            )

            # Convert to (memory_id, score, rank) format
            # sqlite-vec returns distance, convert to similarity
            results = []
            for rank, (memory_id, distance) in enumerate(vector_results, 1):
                # Convert distance to similarity score (0-1 range)
                similarity = 1.0 / (1.0 + distance)
                results.append((memory_id, similarity, rank))

            logger.debug(f"Semantic search found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise VectorSearchError(f"Semantic search failed: {e}") from e

    async def _keyword_search(self, query: SearchQuery) -> List[Tuple[str, float, int]]:
        """
        Execute keyword search usando FTS5.

        Args:
            query: Search query

        Returns:
            List of (memory_id, relevance_score, rank) tuples

        Raises:
            SearchError: Se il keyword search fallisce
        """
        try:
            if query.keyword_weight == 0.0:
                return []

            # Prepare FTS query
            fts_query = self._prepare_fts_query(query.query_text)

            # Search FTS database
            fts_results = await self.storage.search_fts(
                fts_query, k=query.max_results * 2
            )

            # Convert to (memory_id, score, rank) format
            # FTS5 returns rank, convert to normalized score
            results = []
            max_rank = max([rank for _, rank in fts_results], default=1.0)

            for rank_pos, (memory_id, fts_rank) in enumerate(fts_results, 1):
                # Normalize FTS rank to 0-1 score (higher rank = lower score)
                normalized_score = 1.0 - (fts_rank / max_rank) if max_rank > 0 else 1.0
                results.append((memory_id, normalized_score, rank_pos))

            logger.debug(f"Keyword search found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            raise SearchError(f"Keyword search failed: {e}") from e

    def _prepare_fts_query(self, query_text: str) -> str:
        """
        Prepare FTS5 query con proper escaping e operators.

        Args:
            query_text: Raw query text

        Returns:
            FTS5-formatted query string
        """
        # Simple preprocessing - could be enhanced
        # Remove special FTS characters and split into terms
        import re

        # Remove special FTS characters
        cleaned = re.sub(r'[^\w\s]', ' ', query_text)
        terms = cleaned.split()

        if not terms:
            return 'content:"" OR keywords:"" OR entities:""'

        # Create phrase query for better matching
        if len(terms) == 1:
            term = terms[0]
            return f'content:"{term}" OR keywords:"{term}" OR entities:"{term}"'

        # Multi-term query with OR operator for broader matching
        # Search across all FTS columns: content, keywords, entities
        term_queries = []
        for term in terms:
            term_query = f'content:"{term}" OR keywords:"{term}" OR entities:"{term}"'
            term_queries.append(f'({term_query})')

        return " AND ".join(term_queries)

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple[str, float, int]],
        keyword_results: List[Tuple[str, float, int]],
        query: SearchQuery
    ) -> List[MemoryQueryResult]:
        """
        Apply Context7-optimized Reciprocal Rank Fusion algorithm.

        Combina semantic e keyword results usando RRF formula ottimizzata:
        RRF_score(d) = Σ(weight_i / (k + rank_i(d)))

        Context7 improvements:
        - Default weights: keyword=1.5, semantic=1.0 (better keyword relevance)
        - RRF k=60 for optimal scoring
        - Score normalization for better interpretability

        Args:
            semantic_results: Vector search results
            keyword_results: FTS search results
            query: Search query con weights

        Returns:
            Fused and ranked results
        """
        # Collect all unique memory IDs
        all_memory_ids = set()
        semantic_dict = {}
        keyword_dict = {}

        # Index semantic results
        for memory_id, score, rank in semantic_results:
            all_memory_ids.add(memory_id)
            semantic_dict[memory_id] = {"score": score, "rank": rank}

        # Index keyword results
        for memory_id, score, rank in keyword_results:
            all_memory_ids.add(memory_id)
            keyword_dict[memory_id] = {"score": score, "rank": rank}

        # Calculate Context7-optimized RRF scores
        fused_scores = []
        for memory_id in all_memory_ids:
            rrf_score = 0.0

            # Semantic contribution (weight=1.0)
            if memory_id in semantic_dict:
                semantic_rank = semantic_dict[memory_id]["rank"]
                semantic_contribution = self.weight_semantic / (self.rrf_k + semantic_rank)
                rrf_score += semantic_contribution

            # Keyword contribution (weight=1.5 for better relevance)
            if memory_id in keyword_dict:
                keyword_rank = keyword_dict[memory_id]["rank"]
                keyword_contribution = self.weight_keyword / (self.rrf_k + keyword_rank)
                rrf_score += keyword_contribution

            # Create result object
            result = MemoryQueryResult(
                memory_entry=MemoryEntry(id=memory_id, content="", content_type="context"),  # Placeholder
                combined_score=rrf_score,
                semantic_score=semantic_dict.get(memory_id, {}).get("score"),
                keyword_score=keyword_dict.get(memory_id, {}).get("score"),
                semantic_rank=semantic_dict.get(memory_id, {}).get("rank"),
                keyword_rank=keyword_dict.get(memory_id, {}).get("rank"),
                final_rank=0,  # Will be set after sorting
                matched_keywords=[],
                matched_entities=[]
            )

            fused_scores.append((rrf_score, memory_id, result))

        # Sort by RRF score (descending)
        fused_scores.sort(key=lambda x: x[0], reverse=True)

        # Context7: Normalize scores to 0-1 range for better interpretability
        if fused_scores:
            max_score = fused_scores[0][0]  # Highest RRF score
            min_score = fused_scores[-1][0]  # Lowest RRF score
            score_range = max_score - min_score

            # Apply normalization
            normalized_scores = []
            for score, memory_id, result in fused_scores:
                if score_range > 0:
                    normalized_score = (score - min_score) / score_range
                else:
                    normalized_score = 1.0 if score > 0 else 0.0

                result.combined_score = normalized_score
                normalized_scores.append((normalized_score, memory_id, result))

            # Re-sort by normalized scores
            normalized_scores.sort(key=lambda x: x[0], reverse=True)
            fused_scores = normalized_scores

        # Set final ranks and return results
        results = []
        for final_rank, (score, memory_id, result) in enumerate(fused_scores, 1):
            result.final_rank = final_rank
            results.append(result)

        logger.debug(f"RRF fusion completed: {len(results)} combined results with normalized scores")
        return results

    async def _apply_filters(self, results: List[MemoryQueryResult], query: SearchQuery) -> List[MemoryQueryResult]:
        """
        Apply post-search filters to results.

        Args:
            results: Search results to filter
            query: Search query con filter parameters

        Returns:
            Filtered results
        """
        filtered_results = []

        for result in results:
            # Load full memory entry for filtering
            try:
                memory = await self.storage.get_memory(result.memory_entry.id)
                if memory is None:
                    logger.warning(f"Memory entry not found: {result.memory_entry.id}")
                    continue

                # Apply filters
                if not self._passes_filters(memory, query):
                    continue

                # Update result con full memory entry
                result.memory_entry = memory

                # Extract matching information
                result.matched_keywords = self._extract_matched_keywords(memory, query)
                result.matched_entities = self._extract_matched_entities(memory, query)

                filtered_results.append(result)

            except Exception as e:
                logger.warning(f"Failed to load memory {result.memory_entry.id}: {e}")
                continue

        return filtered_results

    def _passes_filters(self, memory: MemoryEntry, query: SearchQuery) -> bool:
        """Check if memory entry passes query filters."""
        # Content type filter
        if query.content_types and memory.content_type not in query.content_types:
            return False

        # Relevance filter
        if memory.relevance_score < query.min_relevance:
            return False

        # Date filters
        if query.created_after and memory.created_at < query.created_after:
            return False

        if query.created_before and memory.created_at > query.created_before:
            return False

        # Context filters
        if query.plan_id and memory.plan_id != query.plan_id:
            return False

        if query.phase_id and memory.phase_id != query.phase_id:
            return False

        if query.task_id and memory.task_id != query.task_id:
            return False

        # Archive filter
        if memory.is_archived:
            return False

        return True

    def _extract_matched_keywords(self, memory: MemoryEntry, query: SearchQuery) -> List[str]:
        """Extract keywords that matched in the search."""
        query_words = set(query.query_text.lower().split())
        memory_keywords = set(keyword.lower() for keyword in memory.keywords)
        return list(query_words.intersection(memory_keywords))

    def _extract_matched_entities(self, memory: MemoryEntry, query: SearchQuery) -> List[str]:
        """Extract entities that matched in the search."""
        query_text_lower = query.query_text.lower()
        matched_entities = []

        for entity in memory.entities:
            if isinstance(entity, dict) and "text" in entity:
                entity_text = entity["text"].lower()
                if entity_text in query_text_lower:
                    matched_entities.append(entity["text"])

        return matched_entities

    async def get_similar_memories(self, memory_id: str, k: int = 5) -> List[MemoryQueryResult]:
        """
        Find memories similar to a given memory entry.

        Args:
            memory_id: ID of reference memory
            k: Number of similar memories to return

        Returns:
            List of similar memories

        Raises:
            SearchError: Se la similarity search fallisce
        """
        try:
            # Get reference memory
            reference_memory = await self.storage.get_memory(memory_id)
            if reference_memory is None:
                raise SearchError(f"Reference memory not found: {memory_id}")

            if reference_memory.embedding is None:
                raise SearchError(f"Reference memory has no embedding: {memory_id}")

            # Use reference content as query
            query = SearchQuery(
                query_text=reference_memory.content[:500],  # Truncate for query
                max_results=k,
                semantic_weight=1.0,
                keyword_weight=0.3,  # Lower weight for keyword matching
            )

            # Execute search
            results = await self.search(query)

            # Filter out the reference memory itself
            similar_results = [r for r in results if r.memory_entry.id != memory_id]

            return similar_results[:k]

        except Exception as e:
            logger.error(f"Similar memory search failed: {e}")
            raise SearchError(f"Similar memory search failed: {e}") from e