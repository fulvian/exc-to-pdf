"""
Ollama Integration Module for DevStream.

Provides production-ready async client for Ollama with:
- Advanced error handling and retry logic
- Memory management and batch processing
- Fallback mechanisms for graceful degradation
- Context7-validated best practices implementation
"""

from .client import OllamaClient, OllamaConfig
from .exceptions import (
    OllamaError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaRetryExhaustedError,
)
from .models import (
    EmbeddingRequest,
    EmbeddingResponse,
    ChatRequest,
    ChatResponse,
    ModelInfo,
    BatchRequest,
    BatchResponse,
)

__all__ = [
    "OllamaClient",
    "OllamaConfig",
    "OllamaError",
    "OllamaConnectionError",
    "OllamaModelNotFoundError",
    "OllamaTimeoutError",
    "OllamaRetryExhaustedError",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "ChatRequest",
    "ChatResponse",
    "ModelInfo",
    "BatchRequest",
    "BatchResponse",
]