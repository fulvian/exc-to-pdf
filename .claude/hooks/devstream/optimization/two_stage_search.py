#!/usr/bin/env .devstream/bin/python
"""
DevStream TwoStageSearch System with Binary Quantization

Context7-compliant two-stage search system to achieve <100ms query time vs +500ms baseline
through binary quantization, coarse filtering, and re-ranking optimization.

Based on Context7 research from:
- SQLite-VEC binary quantization and two-stage search patterns
- Re-ranking strategies with candidate filtering
- Performance optimization through multi-stage retrieval
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from collections import defaultdict
import struct

import structlog
import numpy as np

logger = structlog.get_logger()


class QuantizationType(str, Enum):
    """Types of vector quantization for performance optimization."""
    FLOAT32 = "float32"           # Full precision (baseline)
    BINARY = "binary"             # Binary quantization (1-bit)
    INT8 = "int8"                 # 8-bit integer quantization
    FLOAT16 = "float16"           # 16-bit float quantization


class SearchStage(str, Enum):
    """Search stages in the two-stage process."""
    COARSE_FILTER = "coarse_filter"    # Fast candidate filtering
    FINE_RANK = "fine_rank"            # Precise re-ranking
    FINAL_MERGE = "final_merge"        # Result merging and finalization


@dataclass
class QuantizedVector:
    """Quantized vector representation for efficient storage and search."""
    original_vector: List[float]
    quantized_data: bytes
    quantization_type: QuantizationType
    dimension: int
    compression_ratio: float = 0.0

    def __post_init__(self):
        """Calculate compression ratio after initialization."""
        if self.quantization_type == QuantizationType.FLOAT32:
            self.compression_ratio = 1.0
        elif self.quantization_type == QuantizationType.BINARY:
            self.compression_ratio = 32.0  # 32x compression (32 bits -> 1 bit)
        elif self.quantization_type == QuantizationType.FLOAT16:
            self.compression_ratio = 2.0   # 2x compression (32 bits -> 16 bits)
        elif self.quantization_type == QuantizationType.INT8:
            self.compression_ratio = 4.0   # 4x compression (32 bits -> 8 bits)


@dataclass
class SearchCandidate:
    """Search candidate from coarse filtering stage."""
    id: str
    vector: Optional[List[float]] = None
    quantized_vector: Optional[QuantizedVector] = None
    coarse_score: float = 0.0
    fine_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_stage: SearchStage = SearchStage.COARSE_FILTER


@dataclass
class SearchResults:
    """Complete search results with stage information."""
    query: str
    candidates: List[SearchCandidate]
    total_candidates: int
    coarse_time_ms: float
    fine_time_ms: float
    total_time_ms: float
    quantization_used: QuantizationType
    stage_stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def top_results(self) -> List[SearchCandidate]:
        """Get top results sorted by fine score."""
        return sorted(self.candidates, key=lambda x: x.fine_score, reverse=True)

    @property
    def performance_gain(self) -> float:
        """Calculate performance improvement over baseline."""
        baseline_time = 500.0  # 500ms baseline
        return ((baseline_time - self.total_time_ms) / baseline_time) * 100


@dataclass
class SearchMetrics:
    """Performance metrics for search operations."""
    total_searches: int = 0
    avg_coarse_time_ms: float = 0.0
    avg_fine_time_ms: float = 0.0
    avg_total_time_ms: float = 0.0
    avg_candidates_filtered: int = 0
    avg_final_results: int = 0
    cache_hit_rate: float = 0.0
    compression_efficiency: float = 0.0


class TwoStageSearch:
    """
    Advanced two-stage search system with binary quantization.

    Features:
    - Binary quantization for efficient coarse filtering
    - Two-stage search: coarse filtering + fine re-ranking
    - Adaptive candidate selection based on query characteristics
    - Performance optimization through vector compression
    - Context7-compliant search patterns
    - Real-time performance monitoring and optimization

    Target Performance:
    - Query time: <100ms (from 500ms baseline)
    - Compression ratio: 32x with binary quantization
    - Accuracy retention: 85%+ vs full precision
    - Memory efficiency: 90%+ reduction in storage
    """

    def __init__(
        self,
        vector_dimension: int = 384,
        quantization_type: QuantizationType = QuantizationType.BINARY,
        coarse_candidate_limit: int = 100,
        fine_result_limit: int = 20,
        similarity_threshold: float = 0.7,
        enable_adaptive_limits: bool = True
    ):
        """
        Initialize TwoStageSearch system.

        Args:
            vector_dimension: Dimension of vectors to search
            quantization_type: Type of quantization to use
            coarse_candidate_limit: Max candidates from coarse filtering
            fine_result_limit: Max results after fine re-ranking
            similarity_threshold: Minimum similarity threshold
            enable_adaptive_limits: Whether to adapt limits based on query
        """
        self.vector_dimension = vector_dimension
        self.quantization_type = quantization_type
        self.coarse_candidate_limit = coarse_candidate_limit
        self.fine_result_limit = fine_result_limit
        self.similarity_threshold = similarity_threshold
        self.enable_adaptive_limits = enable_adaptive_limits

        # Storage for quantized vectors
        self._vector_store: Dict[str, QuantizedVector] = {}
        self._metadata_store: Dict[str, Dict[str, Any]] = {}

        # Search cache for performance
        self._search_cache: Dict[str, SearchResults] = {}

        # Performance metrics
        self._metrics = SearchMetrics()
        self._search_times: List[float] = []

        # Quantization parameters
        self._quantization_params = self._initialize_quantization_params()

        logger.info(
            "TwoStageSearch initialized",
            vector_dimension=vector_dimension,
            quantization_type=quantization_type.value,
            coarse_limit=coarse_candidate_limit,
            fine_limit=fine_result_limit,
            adaptive_limits=enable_adaptive_limits
        )

    def _initialize_quantization_params(self) -> Dict[str, Any]:
        """Initialize quantization parameters."""
        params = {
            QuantizationType.BINARY: {
                'bits_per_dimension': 1,
                'dtype': np.uint8,
                'threshold': 0.0
            },
            QuantizationType.INT8: {
                'bits_per_dimension': 8,
                'dtype': np.int8,
                'scale_factor': 127.0
            },
            QuantizationType.FLOAT16: {
                'bits_per_dimension': 16,
                'dtype': np.float16,
                'precision': 'half'
            },
            QuantizationType.FLOAT32: {
                'bits_per_dimension': 32,
                'dtype': np.float32,
                'precision': 'full'
            }
        }
        return params[self.quantization_type]

    def quantize_vector(self, vector: List[float]) -> QuantizedVector:
        """
        Quantize a vector using the configured quantization type.

        Args:
            vector: Original float32 vector

        Returns:
            QuantizedVector with compressed representation
        """
        if not vector:
            raise ValueError("Vector cannot be empty")

        start_time = time.time()
        np_vector = np.array(vector, dtype=np.float32)

        if self.quantization_type == QuantizationType.BINARY:
            # Binary quantization: sign-based
            binary_vector = (np_vector >= 0).astype(np.uint8)
            # Pack 8 bits per byte
            packed_bytes = np.packbits(binary_vector)
            quantized_data = packed_bytes.tobytes()

        elif self.quantization_type == QuantizationType.INT8:
            # 8-bit quantization with scaling
            scale_factor = self._quantization_params['scale_factor']
            scaled = np_vector * scale_factor / np.max(np.abs(np_vector))
            quantized_data = scaled.astype(np.int8).tobytes()

        elif self.quantization_type == QuantizationType.FLOAT16:
            # 16-bit float quantization
            quantized_data = np_vector.astype(np.float16).tobytes()

        else:  # FLOAT32
            quantized_data = np_vector.tobytes()

        quantized_vector = QuantizedVector(
            original_vector=vector,
            quantized_data=quantized_data,
            quantization_type=self.quantization_type,
            dimension=len(vector)
        )

        quantization_time = (time.time() - start_time) * 1000
        logger.debug(
            "Vector quantized",
            dimension=len(vector),
            quantization_type=self.quantization_type.value,
            compression_ratio=quantized_vector.compression_ratio,
            quantization_time_ms=quantization_time
        )

        return quantized_vector

    def dequantize_vector(self, quantized_vector: QuantizedVector) -> List[float]:
        """
        Dequantize a vector back to float32 representation.

        Args:
            quantized_vector: QuantizedVector to dequantize

        Returns:
            Original float32 vector
        """
        if quantized_vector.quantization_type == QuantizationType.FLOAT32:
            return list(np.frombuffer(quantized_vector.quantized_data, dtype=np.float32))

        elif quantized_vector.quantization_type == QuantizationType.FLOAT16:
            return list(np.frombuffer(quantized_vector.quantized_data, dtype=np.float16))

        elif quantized_vector.quantization_type == QuantizationType.INT8:
            int8_array = np.frombuffer(quantized_vector.quantized_data, dtype=np.int8)
            scale_factor = 1.0 / self._quantization_params['scale_factor']
            return list(int8_array.astype(np.float32) * scale_factor)

        elif quantized_vector.quantization_type == QuantizationType.BINARY:
            # Unpack bits and convert to -1/+1
            packed_bytes = np.frombuffer(quantized_vector.quantized_data, dtype=np.uint8)
            unpacked_bits = np.unpackbits(packed_bytes)[:quantized_vector.dimension]
            # Convert to -1.0/+1.0
            float_vector = (unpacked_bits.astype(np.float32) * 2.0) - 1.0
            return list(float_vector)

        else:
            return quantized_vector.original_vector.copy()

    def add_vector(self, vector_id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None):
        """
        Add a vector to the search index.

        Args:
            vector_id: Unique identifier for the vector
            vector: Float32 vector to add
            metadata: Optional metadata associated with the vector
        """
        if vector_id in self._vector_store:
            logger.warning("Vector already exists, overwriting", vector_id=vector_id)

        # Quantize the vector
        quantized_vector = self.quantize_vector(vector)

        # Store quantized vector and metadata
        self._vector_store[vector_id] = quantized_vector
        if metadata:
            self._metadata_store[vector_id] = metadata

        logger.debug(
            "Vector added to index",
            vector_id=vector_id,
            dimension=len(vector),
            quantization_type=self.quantization_type.value,
            compression_ratio=quantized_vector.compression_ratio
        )

    def _coarse_filter_search(
        self,
        query_vector: List[float],
        query_quantized: QuantizedVector,
        candidate_limit: int
    ) -> List[SearchCandidate]:
        """
        Perform coarse filtering search using quantized vectors.

        Args:
            query_vector: Original query vector
            query_quantized: Quantized query vector
            candidate_limit: Maximum candidates to return

        Returns:
            List of search candidates from coarse filtering
        """
        start_time = time.time()
        candidates = []

        # Calculate similarity scores using quantized vectors
        for vector_id, stored_quantized in self._vector_store.items():
            # Fast similarity calculation with quantized vectors
            if self.quantization_type == QuantizationType.BINARY:
                # Hamming distance for binary vectors
                similarity = self._binary_similarity(
                    query_quantized.quantized_data,
                    stored_quantized.quantized_data
                )
            else:
                # Dequantize for precise calculation (only for candidates)
                stored_vector = self.dequantize_vector(stored_quantized)
                similarity = self._cosine_similarity(query_vector, stored_vector)

            if similarity >= self.similarity_threshold:
                candidate = SearchCandidate(
                    id=vector_id,
                    quantized_vector=stored_quantized,
                    coarse_score=similarity,
                    metadata=self._metadata_store.get(vector_id, {}),
                    source_stage=SearchStage.COARSE_FILTER
                )
                candidates.append(candidate)

        # Sort by coarse score and limit results
        candidates.sort(key=lambda x: x.coarse_score, reverse=True)
        candidates = candidates[:candidate_limit]

        coarse_time = (time.time() - start_time) * 1000

        logger.debug(
            "Coarse filtering completed",
            candidates_found=len(candidates),
            coarse_time_ms=coarse_time,
            similarity_threshold=self.similarity_threshold
        )

        return candidates

    def _binary_similarity(self, query_bits: bytes, stored_bits: bytes) -> float:
        """
        Calculate similarity between binary quantized vectors using Hamming distance.

        Args:
            query_bits: Quantized query vector bytes
            stored_bits: Quantized stored vector bytes

        Returns:
            Similarity score (0.0-1.0)
        """
        # Unpack bits
        query_array = np.unpackbits(np.frombuffer(query_bits, dtype=np.uint8))
        stored_array = np.unpackbits(np.frombuffer(stored_bits, dtype=np.uint8))

        # Truncate to vector dimension
        query_array = query_array[:self.vector_dimension]
        stored_array = stored_array[:self.vector_dimension]

        # Calculate Hamming distance
        hamming_distance = np.sum(query_array != stored_array)
        max_distance = len(query_array)

        # Convert to similarity (1.0 = identical, 0.0 = completely different)
        similarity = 1.0 - (hamming_distance / max_distance)

        return float(similarity)

    def _cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vector1: First vector
            vector2: Second vector

        Returns:
            Cosine similarity score (0.0-1.0)
        """
        if len(vector1) != len(vector2):
            return 0.0

        np_v1 = np.array(vector1, dtype=np.float32)
        np_v2 = np.array(vector2, dtype=np.float32)

        # Calculate cosine similarity
        dot_product = np.dot(np_v1, np_v2)
        norm_v1 = np.linalg.norm(np_v1)
        norm_v2 = np.linalg.norm(np_v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        similarity = dot_product / (norm_v1 * norm_v2)
        return float(max(0.0, similarity))

    def _fine_rank_search(
        self,
        query_vector: List[float],
        candidates: List[SearchCandidate]
    ) -> List[SearchCandidate]:
        """
        Perform fine re-ranking on candidates from coarse filtering.

        Args:
            query_vector: Original query vector
            candidates: Candidates from coarse filtering stage

        Returns:
            Re-ranked candidates with fine scores
        """
        start_time = time.time()

        # Calculate precise similarity scores
        for candidate in candidates:
            if candidate.quantized_vector:
                # Dequantize for precise calculation
                stored_vector = self.dequantize_vector(candidate.quantized_vector)
                candidate.vector = stored_vector

            # Calculate precise similarity
            if candidate.vector:
                candidate.fine_score = self._cosine_similarity(query_vector, candidate.vector)
            else:
                candidate.fine_score = candidate.coarse_score

        # Sort by fine score
        candidates.sort(key=lambda x: x.fine_score, reverse=True)

        fine_time = (time.time() - start_time) * 1000

        logger.debug(
            "Fine ranking completed",
            candidates_processed=len(candidates),
            fine_time_ms=fine_time
        )

        return candidates

    def _adaptive_limit_selection(
        self,
        query_vector: List[float],
        base_coarse_limit: int,
        base_fine_limit: int
    ) -> Tuple[int, int]:
        """
        Adaptively select search limits based on query characteristics.

        Args:
            query_vector: Query vector to analyze
            base_coarse_limit: Base coarse filtering limit
            base_fine_limit: Base fine ranking limit

        Returns:
            Tuple of (adapted_coarse_limit, adapted_fine_limit)
        """
        if not self.enable_adaptive_limits:
            return base_coarse_limit, base_fine_limit

        # Analyze query vector characteristics
        query_norm = np.linalg.norm(query_vector)
        query_sparsity = np.count_nonzero(query_vector) / len(query_vector)

        # Adapt limits based on query characteristics
        if query_sparsity < 0.3:  # Sparse query
            coarse_limit = int(base_coarse_limit * 1.5)
            fine_limit = int(base_fine_limit * 1.2)
        elif query_norm > 10.0:  # High magnitude query
            coarse_limit = int(base_coarse_limit * 0.8)
            fine_limit = int(base_fine_limit * 0.9)
        else:
            coarse_limit = base_coarse_limit
            fine_limit = base_fine_limit

        # Ensure reasonable bounds
        coarse_limit = max(10, min(coarse_limit, 500))
        fine_limit = max(5, min(fine_limit, 100))

        return coarse_limit, fine_limit

    def search(
        self,
        query_vector: List[float],
        result_limit: Optional[int] = None,
        use_cache: bool = True
    ) -> SearchResults:
        """
        Perform two-stage search with binary quantization.

        Args:
            query_vector: Query vector to search with
            result_limit: Optional override for result limit
            use_cache: Whether to use search cache

        Returns:
            Complete search results with performance metrics
        """
        start_time = time.time()

        # Check cache first
        cache_key = hashlib.md5(str(query_vector).encode()).hexdigest()
        if use_cache and cache_key in self._search_cache:
            cached_results = self._search_cache[cache_key]
            logger.debug("Search cache hit", cache_key=cache_key)
            return cached_results

        # Quantize query vector
        query_quantized = self.quantize_vector(query_vector)

        # Adaptively select limits
        coarse_limit, fine_limit = self._adaptive_limit_selection(
            query_vector,
            self.coarse_candidate_limit,
            result_limit or self.fine_result_limit
        )

        # Stage 1: Coarse filtering
        coarse_start = time.time()
        candidates = self._coarse_filter_search(
            query_vector,
            query_quantized,
            coarse_limit
        )
        coarse_time = (time.time() - coarse_start) * 1000

        # Stage 2: Fine re-ranking
        fine_start = time.time()
        if candidates:
            ranked_candidates = self._fine_rank_search(query_vector, candidates)
            ranked_candidates = ranked_candidates[:fine_limit]
        else:
            ranked_candidates = []
        fine_time = (time.time() - fine_start) * 1000

        # Create results
        total_time = (time.time() - start_time) * 1000
        results = SearchResults(
            query="vector_search",
            candidates=ranked_candidates,
            total_candidates=len(candidates),
            coarse_time_ms=coarse_time,
            fine_time_ms=fine_time,
            total_time_ms=total_time,
            quantization_used=self.quantization_type,
            stage_stats={
                'coarse_limit': coarse_limit,
                'fine_limit': fine_limit,
                'vector_dimension': self.vector_dimension,
                'compression_ratio': query_quantized.compression_ratio
            }
        )

        # Cache results
        if use_cache and len(self._search_cache) < 1000:
            self._search_cache[cache_key] = results

        # Update metrics
        self._update_metrics(results)

        logger.info(
            "Two-stage search completed",
            total_candidates=results.total_candidates,
            final_results=len(results.candidates),
            total_time_ms=total_time,
            performance_gain=results.performance_gain,
            quantization_type=self.quantization_type.value
        )

        return results

    def _update_metrics(self, results: SearchResults):
        """Update performance metrics."""
        self._search_times.append(results.total_time_ms)

        # Keep only recent metrics
        if len(self._search_times) > 1000:
            self._search_times = self._search_times[-500:]

        # Update aggregate metrics
        self._metrics.total_searches += 1
        self._metrics.avg_coarse_time_ms = (
            (self._metrics.avg_coarse_time_ms * (self._metrics.total_searches - 1) + results.coarse_time_ms) /
            self._metrics.total_searches
        )
        self._metrics.avg_fine_time_ms = (
            (self._metrics.avg_fine_time_ms * (self._metrics.total_searches - 1) + results.fine_time_ms) /
            self._metrics.total_searches
        )
        self._metrics.avg_total_time_ms = sum(self._search_times) / len(self._search_times)
        self._metrics.avg_candidates_filtered = (
            (self._metrics.avg_candidates_filtered * (self._metrics.total_searches - 1) + results.total_candidates) /
            self._metrics.total_searches
        )
        self._metrics.avg_final_results = (
            (self._metrics.avg_final_results * (self._metrics.total_searches - 1) + len(results.candidates)) /
            self._metrics.total_searches
        )

        # Calculate cache hit rate
        self._metrics.cache_hit_rate = len(self._search_cache) / max(self._metrics.total_searches, 1) * 100

        # Calculate compression efficiency
        if self._vector_store:
            total_compression = sum(qv.compression_ratio for qv in self._vector_store.values())
            self._metrics.compression_efficiency = total_compression / len(self._vector_store)

    def get_metrics(self) -> SearchMetrics:
        """
        Get current search performance metrics.

        Returns:
            SearchMetrics with current performance data
        """
        return self._metrics

    def reset_metrics(self):
        """Reset performance metrics."""
        self._metrics = SearchMetrics()
        self._search_times = []
        self._search_cache.clear()
        logger.info("TwoStageSearch metrics reset")

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics for the search index.

        Returns:
            Dictionary with storage statistics
        """
        if not self._vector_store:
            return {
                'total_vectors': 0,
                'storage_size_bytes': 0,
                'compression_ratio': 0.0,
                'quantization_type': self.quantization_type.value
            }

        total_size = sum(len(qv.quantized_data) for qv in self._vector_store.values())
        avg_compression = sum(qv.compression_ratio for qv in self._vector_store.values()) / len(self._vector_store)

        return {
            'total_vectors': len(self._vector_store),
            'storage_size_bytes': total_size,
            'avg_compression_ratio': avg_compression,
            'quantization_type': self.quantization_type.value,
            'vector_dimension': self.vector_dimension,
            'metadata_entries': len(self._metadata_store)
        }

    def clear_cache(self):
        """Clear the search cache."""
        self._search_cache.clear()
        logger.info("Search cache cleared")

    def optimize_index(self):
        """
        Optimize the search index for better performance.

        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()

        # Re-quantize all vectors if needed
        requantized_count = 0
        for vector_id, quantized_vector in self._vector_store.items():
            if quantized_vector.quantization_type != self.quantization_type:
                # Re-quantize with current type
                new_quantized = self.quantize_vector(quantized_vector.original_vector)
                self._vector_store[vector_id] = new_quantized
                requantized_count += 1

        # Clear cache to force re-computation
        self.clear_cache()

        optimization_time = (time.time() - start_time) * 1000

        results = {
            'vectors_requantized': requantized_count,
            'optimization_time_ms': optimization_time,
            'total_vectors': len(self._vector_store),
            'quantization_type': self.quantization_type.value,
            'storage_stats': self.get_storage_stats()
        }

        logger.info(
            "Index optimization completed",
            **results
        )

        return results


# Global instance for import
_two_stage_search = None


def get_two_stage_search(
    vector_dimension: int = 384,
    quantization_type: QuantizationType = QuantizationType.BINARY,
    coarse_candidate_limit: int = 100,
    fine_result_limit: int = 20,
    similarity_threshold: float = 0.7,
    enable_adaptive_limits: bool = True
) -> TwoStageSearch:
    """
    Get or create global TwoStageSearch instance.

    Args:
        vector_dimension: Dimension of vectors to search
        quantization_type: Type of quantization to use
        coarse_candidate_limit: Max candidates from coarse filtering
        fine_result_limit: Max results after fine re-ranking
        similarity_threshold: Minimum similarity threshold
        enable_adaptive_limits: Whether to adapt limits based on query

    Returns:
        TwoStageSearch instance
    """
    global _two_stage_search

    if (_two_stage_search is None or
        _two_stage_search.vector_dimension != vector_dimension or
        _two_stage_search.quantization_type != quantization_type):

        _two_stage_search = TwoStageSearch(
            vector_dimension=vector_dimension,
            quantization_type=quantization_type,
            coarse_candidate_limit=coarse_candidate_limit,
            fine_result_limit=fine_result_limit,
            similarity_threshold=similarity_threshold,
            enable_adaptive_limits=enable_adaptive_limits
        )

    return _two_stage_search