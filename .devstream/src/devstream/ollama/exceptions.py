"""
Ollama-specific exceptions for robust error handling.

Based on Context7 best practices for error hierarchy and handling patterns.
"""

from typing import Optional, Any, Dict
import httpx
from devstream.core.exceptions import DevStreamError


class OllamaError(DevStreamError):
    """Base exception for all Ollama-related errors."""

    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Add original_error to context if provided
        if context is None:
            context = {}
        if original_error is not None:
            context["original_error"] = str(original_error)
            context["original_error_type"] = type(original_error).__name__

        super().__init__(message, error_code, context)
        self.original_error = original_error


class OllamaConnectionError(OllamaError):
    """Raised when unable to connect to Ollama server."""

    def __init__(
        self,
        message: str = "Failed to connect to Ollama server",
        host: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        context = {"host": host} if host else None
        super().__init__(
            message=message,
            original_error=original_error,
            error_code="OLLAMA_CONNECTION_ERROR",
            context=context,
        )


class OllamaTimeoutError(OllamaError):
    """Raised when Ollama request times out."""

    def __init__(
        self,
        message: str = "Ollama request timed out",
        timeout_duration: Optional[float] = None,
        operation: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        context = {
            "timeout_duration": timeout_duration,
            "operation": operation,
        }
        super().__init__(
            message=message,
            original_error=original_error,
            error_code="OLLAMA_TIMEOUT_ERROR",
            context={k: v for k, v in context.items() if v is not None},
        )


class OllamaModelNotFoundError(OllamaError):
    """Raised when requested model is not available."""

    def __init__(
        self,
        model_name: str,
        message: Optional[str] = None,
        available_models: Optional[list[str]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        if message is None:
            message = f"Model '{model_name}' not found"

        context = {
            "model_name": model_name,
            "available_models": available_models,
        }
        super().__init__(
            message=message,
            original_error=original_error,
            error_code="OLLAMA_MODEL_NOT_FOUND",
            context={k: v for k, v in context.items() if v is not None},
        )


class OllamaRetryExhaustedError(OllamaError):
    """Raised when all retry attempts have been exhausted."""

    def __init__(
        self,
        operation: str,
        max_retries: int,
        last_error: Optional[Exception] = None,
        retry_history: Optional[list[Exception]] = None,
    ) -> None:
        message = (
            f"Operation '{operation}' failed after {max_retries} retry attempts"
        )
        context = {
            "operation": operation,
            "max_retries": max_retries,
            "retry_count": len(retry_history) if retry_history else 0,
            "last_error_type": type(last_error).__name__ if last_error else None,
        }
        super().__init__(
            message=message,
            original_error=last_error,
            error_code="OLLAMA_RETRY_EXHAUSTED",
            context=context,
        )


class OllamaServerError(OllamaError):
    """Raised when Ollama server returns an error response."""

    def __init__(
        self,
        status_code: int,
        response_text: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        message = f"Ollama server error: HTTP {status_code}"
        if response_text:
            message += f" - {response_text}"

        context = {
            "status_code": status_code,
            "response_text": response_text,
            "headers": headers,
        }
        super().__init__(
            message=message,
            original_error=original_error,
            error_code="OLLAMA_SERVER_ERROR",
            context={k: v for k, v in context.items() if v is not None},
        )


class OllamaInvalidResponseError(OllamaError):
    """Raised when Ollama returns an invalid or unexpected response."""

    def __init__(
        self,
        message: str = "Invalid response from Ollama server",
        response_data: Optional[Any] = None,
        expected_format: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ) -> None:
        context = {
            "response_data_type": type(response_data).__name__ if response_data else None,
            "expected_format": expected_format,
        }
        super().__init__(
            message=message,
            original_error=original_error,
            error_code="OLLAMA_INVALID_RESPONSE",
            context={k: v for k, v in context.items() if v is not None},
        )


def map_httpx_error_to_ollama_error(
    error: httpx.HTTPError,
    operation: str = "unknown",
    host: Optional[str] = None,
) -> OllamaError:
    """
    Map HTTPX errors to appropriate Ollama-specific errors.

    Based on Context7 research of HTTPX error handling patterns.
    """
    if isinstance(error, httpx.ConnectError):
        return OllamaConnectionError(
            message=f"Failed to connect to Ollama server during {operation}",
            host=host,
            original_error=error,
        )

    elif isinstance(error, httpx.TimeoutException):
        timeout_type = "unknown"
        if isinstance(error, httpx.ConnectTimeout):
            timeout_type = "connection"
        elif isinstance(error, httpx.ReadTimeout):
            timeout_type = "read"
        elif isinstance(error, httpx.WriteTimeout):
            timeout_type = "write"
        elif isinstance(error, httpx.PoolTimeout):
            timeout_type = "pool"

        return OllamaTimeoutError(
            message=f"Ollama {timeout_type} timeout during {operation}",
            operation=operation,
            original_error=error,
        )

    elif isinstance(error, httpx.HTTPStatusError):
        return OllamaServerError(
            status_code=error.response.status_code,
            response_text=error.response.text,
            headers=dict(error.response.headers),
            original_error=error,
        )

    else:
        # Generic network or request error
        return OllamaConnectionError(
            message=f"Network error during {operation}: {str(error)}",
            host=host,
            original_error=error,
        )