"""
Memory System Custom Exceptions

Hierarchy di eccezioni specifiche per il memory system
per error handling granulare e debugging efficace.
"""

from typing import Any, Optional


class MemoryError(Exception):
    """Base exception per memory system errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class StorageError(MemoryError):
    """Error durante storage/retrieval operations."""
    pass


class SearchError(MemoryError):
    """Error durante search operations."""
    pass


class ProcessingError(MemoryError):
    """Error durante text processing operations."""
    pass


class ContextError(MemoryError):
    """Error durante context assembly operations."""
    pass


class EmbeddingError(ProcessingError):
    """Error durante embedding generation."""
    pass


class VectorSearchError(SearchError):
    """Error durante vector search operations."""
    pass


class FTSError(SearchError):
    """Error durante FTS operations."""
    pass


class TokenBudgetError(ContextError):
    """Error quando il token budget è exceeded."""
    pass