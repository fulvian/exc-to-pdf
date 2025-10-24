"""
Production-ready Ollama async client with Context7-validated best practices.

Features:
- Advanced error handling and retry logic
- Memory management and batch processing
- Fallback mechanisms for graceful degradation
- Comprehensive monitoring and metrics
- Type-safe operations with Pydantic models
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Union, AsyncIterator, Tuple
import httpx
import ollama
import numpy as np

from .config import OllamaConfig
from .exceptions import (
    OllamaError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaServerError,
    OllamaInvalidResponseError,
    map_httpx_error_to_ollama_error,
)
from .models import (
    EmbeddingRequest,
    EmbeddingResponse,
    ChatRequest,
    ChatResponse,
    ModelInfo,
    BatchRequest,
    BatchResponse,
    HealthCheckResponse,
    ModelPullRequest,
    ModelPullProgress,
    ChatMessage,
)
from .retry import RetryHandler, with_retry, create_ollama_giveup_condition

logger = logging.getLogger(__name__)


class OllamaMetrics:
    """Metrics collection for monitoring Ollama operations."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all metrics."""
        self.total_requests: int = 0
        self.successful_requests: int = 0
        self.failed_requests: int = 0
        self.total_response_time: float = 0.0
        self.embedding_requests: int = 0
        self.chat_requests: int = 0
        self.batch_requests: int = 0
        self.models_pulled: int = 0
        self.cache_hits: int = 0
        self.fallback_uses: int = 0
        self.start_time: float = time.time()

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100.0

    @property
    def average_response_time(self) -> float:
        """Calculate average response time in seconds."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_response_time / self.successful_requests

    @property
    def uptime(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self.start_time

    def record_request(self, success: bool, response_time: float, request_type: str = "unknown") -> None:
        """Record a request result."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
            self.total_response_time += response_time
        else:
            self.failed_requests += 1

        # Track by type
        if request_type == "embedding":
            self.embedding_requests += 1
        elif request_type == "chat":
            self.chat_requests += 1
        elif request_type == "batch":
            self.batch_requests += 1

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "average_response_time": self.average_response_time,
            "embedding_requests": self.embedding_requests,
            "chat_requests": self.chat_requests,
            "batch_requests": self.batch_requests,
            "models_pulled": self.models_pulled,
            "cache_hits": self.cache_hits,
            "fallback_uses": self.fallback_uses,
            "uptime": self.uptime,
        }


class ModelCache:
    """LRU cache for model information and availability."""

    def __init__(self, max_size: int = 100) -> None:
        self.max_size = max_size
        self.cache: Dict[str, Tuple[ModelInfo, float]] = {}  # model_name -> (info, timestamp)
        self.access_order: List[str] = []

    def get(self, model_name: str, ttl: float = 300.0) -> Optional[ModelInfo]:
        """Get model info from cache if not expired."""
        if model_name not in self.cache:
            return None

        info, timestamp = self.cache[model_name]
        if time.time() - timestamp > ttl:
            self._remove(model_name)
            return None

        # Update access order (LRU)
        if model_name in self.access_order:
            self.access_order.remove(model_name)
        self.access_order.append(model_name)

        return info

    def put(self, model_name: str, info: ModelInfo) -> None:
        """Add model info to cache."""
        # Remove if exists
        if model_name in self.cache:
            self.access_order.remove(model_name)

        # Evict least recently used if at capacity
        if len(self.cache) >= self.max_size:
            lru_model = self.access_order.pop(0)
            del self.cache[lru_model]

        # Add new entry
        self.cache[model_name] = (info, time.time())
        self.access_order.append(model_name)

    def _remove(self, model_name: str) -> None:
        """Remove model from cache."""
        if model_name in self.cache:
            del self.cache[model_name]
            if model_name in self.access_order:
                self.access_order.remove(model_name)

    def clear(self) -> None:
        """Clear all cached models."""
        self.cache.clear()
        self.access_order.clear()

    def is_available(self, model_name: str) -> Optional[bool]:
        """Check if model is available based on cache."""
        info = self.get(model_name)
        return info is not None


class OllamaClient:
    """
    Production-ready async Ollama client.

    Based on Context7 research of best practices for async HTTP clients
    and AI model integration patterns.
    """

    def __init__(self, config: Optional[OllamaConfig] = None) -> None:
        self.config = config or OllamaConfig()
        self.metrics = OllamaMetrics()
        self.model_cache = ModelCache(max_size=self.config.model_cache_size)

        # Initialize HTTP client with Context7-validated settings
        self._http_client: Optional[httpx.AsyncClient] = None
        self._ollama_client: Optional[ollama.AsyncClient] = None

        # Retry handler with Ollama-specific configuration
        self.retry_handler = RetryHandler(
            config=self.config.retry,
            giveup_condition=create_ollama_giveup_condition(),
        )

        logger.info(f"OllamaClient initialized with host: {self.config.host}")

    async def __aenter__(self) -> "OllamaClient":
        """Async context manager entry."""
        await self._initialize_clients()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def _initialize_clients(self) -> None:
        """Initialize Ollama client with Context7 simplified pattern."""
        if self._ollama_client is not None:
            return  # Already initialized

        # Context7 Pattern: Simple AsyncClient without custom httpx
        # Let ollama handle HTTP client internally
        self._ollama_client = ollama.AsyncClient(
            host=self.config.host
        )

        # Set to None to avoid cleanup issues
        self._http_client = None

        logger.debug("Ollama client initialized with Context7 pattern")

    async def close(self) -> None:
        """Clean up resources."""
        # Context7 pattern - ollama handles its own cleanup
        self._ollama_client = None
        self._http_client = None
        logger.debug("Ollama client closed")

    @property
    def is_initialized(self) -> bool:
        """Check if clients are initialized."""
        return self._ollama_client is not None

    async def _ensure_initialized(self) -> None:
        """Ensure clients are initialized before use."""
        if not self.is_initialized:
            await self._initialize_clients()

    async def health_check(self) -> HealthCheckResponse:
        """
        Perform health check against Ollama server.

        Based on Context7 patterns for service health monitoring.
        """
        await self._ensure_initialized()
        start_time = time.time()

        try:
            # Context7 Pattern: Use ollama client for health check
            models = await self.list_models()
            response_time = (time.time() - start_time) * 1000

            return HealthCheckResponse(
                status="healthy",
                version=None,  # Ollama client doesn't expose version directly
                models_available=len(models),
                response_time_ms=response_time,
                timestamp=time.time(),
                uptime_seconds=self.metrics.uptime,
            )

        except Exception as exc:
            response_time = (time.time() - start_time) * 1000
            logger.warning(f"Health check failed: {exc}")

            return HealthCheckResponse(
                status="unhealthy",
                models_available=0,
                response_time_ms=response_time,
                timestamp=time.time(),
                uptime_seconds=self.metrics.uptime,
            )

    async def list_models(self) -> List[ModelInfo]:
        """
        List available models with caching.

        Uses Context7-validated patterns for data fetching and caching.
        """
        await self._ensure_initialized()

        @with_retry(self.config.retry)
        async def _fetch_models() -> List[ModelInfo]:
            start_time = time.time()
            try:
                response = await self._ollama_client.list()
                models = []

                for model_data in response.get("models", []):
                    model_info = ModelInfo(
                        name=model_data["name"],
                        size=model_data["size"],
                        digest=model_data["digest"],
                        modified_at=model_data["modified_at"],
                        details=model_data.get("details"),
                    )
                    models.append(model_info)
                    # Cache the model info
                    self.model_cache.put(model_info.name, model_info)

                self.metrics.record_request(True, time.time() - start_time, "list_models")
                return models

            except Exception as exc:
                self.metrics.record_request(False, time.time() - start_time, "list_models")
                raise map_httpx_error_to_ollama_error(exc, "list_models", self.config.host)

        return await _fetch_models()

    async def pull_model(
        self, request: ModelPullRequest
    ) -> AsyncGenerator[ModelPullProgress, None]:
        """
        Pull/download a model with progress tracking.

        Implements Context7 streaming patterns for long-running operations.
        """
        await self._ensure_initialized()

        try:
            async for progress in await self._ollama_client.pull(
                model=request.name,
                stream=request.stream,
                insecure=request.insecure,
            ):
                yield ModelPullProgress(
                    status=progress.get("status", "unknown"),
                    digest=progress.get("digest"),
                    total=progress.get("total"),
                    completed=progress.get("completed"),
                )

            self.metrics.models_pulled += 1
            logger.info(f"Successfully pulled model: {request.name}")

        except Exception as exc:
            logger.error(f"Failed to pull model {request.name}: {exc}")
            raise map_httpx_error_to_ollama_error(exc, "pull_model", self.config.host)

    async def generate_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings with retry logic and fallback support.

        Based on Context7 patterns for AI model operations.
        """
        await self._ensure_initialized()

        @with_retry(self.config.retry)
        async def _generate_embedding() -> EmbeddingResponse:
            start_time = time.time()
            try:
                # Check if model is available in cache
                if not self.model_cache.is_available(request.model):
                    # Try to auto-pull if configured
                    if self.config.fallback.auto_pull_missing_models:
                        try:
                            pull_request = ModelPullRequest(name=request.model)
                            async for _ in self.pull_model(pull_request):
                                pass  # Wait for pull to complete
                        except Exception as pull_exc:
                            logger.warning(f"Auto-pull failed for {request.model}: {pull_exc}")

                response = await self._ollama_client.embeddings(
                    model=request.model,
                    prompt=request.prompt,
                    options=request.options,
                    keep_alive=request.keep_alive,
                )

                result = EmbeddingResponse(
                    model=response["model"],
                    embedding=response["embedding"],
                    prompt_eval_count=response.get("prompt_eval_count"),
                    eval_duration=response.get("eval_duration"),
                )

                self.metrics.record_request(True, time.time() - start_time, "embedding")
                return result

            except ollama.ResponseError as exc:
                if exc.status_code == 404:
                    # Model not found - try fallback if configured
                    if self.config.fallback.enable_fallback and self.config.fallback.fallback_on_model_not_found:
                        return await self._try_embedding_fallback(request, start_time)
                    else:
                        raise OllamaModelNotFoundError(
                            model_name=request.model,
                            original_error=exc,
                        )
                else:
                    self.metrics.record_request(False, time.time() - start_time, "embedding")
                    raise map_httpx_error_to_ollama_error(exc, "generate_embedding", self.config.host)

            except Exception as exc:
                self.metrics.record_request(False, time.time() - start_time, "embedding")
                raise map_httpx_error_to_ollama_error(exc, "generate_embedding", self.config.host)

        return await _generate_embedding()

    async def _try_embedding_fallback(
        self, original_request: EmbeddingRequest, start_time: float
    ) -> EmbeddingResponse:
        """Try fallback models for embedding generation."""
        for fallback_model in self.config.fallback.fallback_models:
            if fallback_model == original_request.model:
                continue  # Skip original model

            try:
                logger.info(f"Trying fallback model: {fallback_model}")
                fallback_request = original_request.model_copy()
                fallback_request.model = fallback_model

                response = await self._ollama_client.embeddings(
                    model=fallback_request.model,
                    prompt=fallback_request.prompt,
                    options=fallback_request.options,
                    keep_alive=fallback_request.keep_alive,
                )

                result = EmbeddingResponse(
                    model=response["model"],
                    embedding=response["embedding"],
                    prompt_eval_count=response.get("prompt_eval_count"),
                    eval_duration=response.get("eval_duration"),
                )

                self.metrics.fallback_uses += 1
                self.metrics.record_request(True, time.time() - start_time, "embedding")
                logger.info(f"Fallback successful with model: {fallback_model}")
                return result

            except Exception as fallback_exc:
                logger.warning(f"Fallback model {fallback_model} failed: {fallback_exc}")
                continue

        # All fallbacks failed
        self.metrics.record_request(False, time.time() - start_time, "embedding")
        raise OllamaModelNotFoundError(
            model_name=original_request.model,
            available_models=self.config.fallback.fallback_models,
        )

    async def chat(self, request: ChatRequest) -> Union[ChatResponse, AsyncIterator[ChatResponse]]:
        """
        Generate chat completions with support for streaming.

        Implements Context7 patterns for conversational AI operations.
        """
        await self._ensure_initialized()

        @with_retry(self.config.retry)
        async def _chat() -> Union[ChatResponse, AsyncIterator[ChatResponse]]:
            start_time = time.time()
            try:
                if request.stream:
                    return self._stream_chat(request, start_time)
                else:
                    response = await self._ollama_client.chat(
                        model=request.model,
                        messages=[msg.model_dump() for msg in request.messages],
                        stream=False,
                        format=request.format,
                        options=request.options,
                        keep_alive=request.keep_alive,
                    )

                    result = ChatResponse(
                        model=response["model"],
                        message=ChatMessage(**response["message"]),
                        done=response["done"],
                        created_at=response["created_at"],
                        total_duration=response.get("total_duration"),
                        load_duration=response.get("load_duration"),
                        prompt_eval_count=response.get("prompt_eval_count"),
                        prompt_eval_duration=response.get("prompt_eval_duration"),
                        eval_count=response.get("eval_count"),
                        eval_duration=response.get("eval_duration"),
                    )

                    self.metrics.record_request(True, time.time() - start_time, "chat")
                    return result

            except Exception as exc:
                self.metrics.record_request(False, time.time() - start_time, "chat")
                raise map_httpx_error_to_ollama_error(exc, "chat", self.config.host)

        return await _chat()

    async def _stream_chat(
        self, request: ChatRequest, start_time: float
    ) -> AsyncIterator[ChatResponse]:
        """Handle streaming chat responses."""
        try:
            async for chunk in await self._ollama_client.chat(
                model=request.model,
                messages=[msg.model_dump() for msg in request.messages],
                stream=True,
                format=request.format,
                options=request.options,
                keep_alive=request.keep_alive,
            ):
                yield ChatResponse(
                    model=chunk["model"],
                    message=ChatMessage(**chunk["message"]),
                    done=chunk["done"],
                    created_at=chunk["created_at"],
                    total_duration=chunk.get("total_duration"),
                    load_duration=chunk.get("load_duration"),
                    prompt_eval_count=chunk.get("prompt_eval_count"),
                    prompt_eval_duration=chunk.get("prompt_eval_duration"),
                    eval_count=chunk.get("eval_count"),
                    eval_duration=chunk.get("eval_duration"),
                )

                if chunk["done"]:
                    self.metrics.record_request(True, time.time() - start_time, "chat")
                    break

        except Exception as exc:
            self.metrics.record_request(False, time.time() - start_time, "chat")
            raise map_httpx_error_to_ollama_error(exc, "stream_chat", self.config.host)

    async def process_batch(self, request: BatchRequest) -> BatchResponse:
        """
        Process multiple items in batches with Context7-validated memory management.

        Implements efficient batching patterns for high-throughput operations.
        """
        await self._ensure_initialized()
        start_time = time.time()

        results: List[Union[EmbeddingResponse, ChatResponse]] = []
        errors: List[str] = []
        successful_items = 0
        failed_items = 0

        # Process items in batches
        batch_size = min(request.batch_size, self.config.batch.max_batch_size)
        semaphore = asyncio.Semaphore(self.config.batch.max_concurrent_batches)

        async def process_batch_chunk(batch_items: List[Union[str, ChatMessage]]) -> None:
            nonlocal successful_items, failed_items

            async with semaphore:
                try:
                    if request.operation == "embedding":
                        for item in batch_items:
                            if isinstance(item, str):
                                embed_request = EmbeddingRequest(
                                    model=request.model,
                                    prompt=item,
                                    options=request.options,
                                )
                                try:
                                    result = await self.generate_embedding(embed_request)
                                    results.append(result)
                                    successful_items += 1
                                except Exception as e:
                                    errors.append(f"Item {len(results)}: {str(e)}")
                                    failed_items += 1

                    elif request.operation == "chat":
                        for item in batch_items:
                            if isinstance(item, ChatMessage):
                                chat_request = ChatRequest(
                                    model=request.model,
                                    messages=[item],
                                    options=request.options,
                                )
                                try:
                                    result = await self.chat(chat_request)
                                    if isinstance(result, ChatResponse):
                                        results.append(result)
                                        successful_items += 1
                                    else:
                                        # Handle streaming response by collecting final result
                                        async for final_chunk in result:
                                            if final_chunk.done:
                                                results.append(final_chunk)
                                                successful_items += 1
                                                break
                                except Exception as e:
                                    errors.append(f"Item {len(results)}: {str(e)}")
                                    failed_items += 1

                except Exception as batch_error:
                    logger.error(f"Batch processing error: {batch_error}")
                    errors.append(f"Batch error: {str(batch_error)}")
                    failed_items += len(batch_items)

        # Create batches and process concurrently
        tasks = []
        for i in range(0, len(request.items), batch_size):
            batch_items = request.items[i:i + batch_size]
            task = asyncio.create_task(process_batch_chunk(batch_items))
            tasks.append(task)

        # Wait for all batches to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        total_duration = time.time() - start_time
        self.metrics.record_request(True, total_duration, "batch")

        return BatchResponse(
            model=request.model,
            operation=request.operation,
            results=results,
            total_items=len(request.items),
            successful_items=successful_items,
            failed_items=failed_items,
            total_duration=total_duration,
            errors=errors if errors else None,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current client metrics."""
        return self.metrics.to_dict()

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics.reset()

    async def cleanup_resources(self) -> None:
        """Clean up resources and clear caches."""
        self.model_cache.clear()
        await self.close()
        logger.info("Ollama client resources cleaned up")