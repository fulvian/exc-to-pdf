"""
DevStream optimization components.

This package contains optimization modules for the DevStream memory system:
- ContentQualityFilter: Intelligent content filtering for storage reduction
- AsyncEmbeddingBatchProcessor: Efficient async embedding processing
- SemanticCacheKeys: Semantic-aware cache key system for improved hit rates
"""

__version__ = "1.0.0"
__all__ = [
    "content_quality_filter",
    "async_embedding_processor",
    "semantic_cache_keys",
    "task_aware_query_constructor",
    "two_stage_search"
]