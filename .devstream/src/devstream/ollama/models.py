"""
Pydantic models for Ollama requests and responses.

Type-safe data structures based on Ollama API specification.
"""

from typing import Any, Dict, List, Optional, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, field_validator
import numpy as np


class ModelInfo(BaseModel):
    """Information about an Ollama model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., description="Model name")
    size: int = Field(..., description="Model size in bytes")
    digest: str = Field(..., description="Model digest/hash")
    modified_at: datetime = Field(..., description="Last modification time")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional model details"
    )

    @field_validator("name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name format."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


class EmbeddingRequest(BaseModel):
    """Request for generating embeddings."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., description="Model to use for embeddings")
    prompt: str = Field(..., description="Text to generate embeddings for")
    options: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional model options"
    )
    keep_alive: Optional[str] = Field(
        default=None, description="How long to keep model loaded"
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Validate prompt is not empty."""
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v.strip()

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate model name."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


class EmbeddingResponse(BaseModel):
    """Response containing embeddings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model used")
    embedding: List[float] = Field(..., description="Generated embedding vector")
    prompt_eval_count: Optional[int] = Field(
        default=None, description="Number of tokens evaluated"
    )
    eval_duration: Optional[int] = Field(
        default=None, description="Evaluation duration in nanoseconds"
    )

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: List[float]) -> List[float]:
        """Validate embedding vector."""
        if not v:
            raise ValueError("Embedding vector cannot be empty")
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Embedding must contain only numbers")
        return v

    def to_numpy(self) -> np.ndarray:
        """Convert embedding to numpy array."""
        return np.array(self.embedding, dtype=np.float32)

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return len(self.embedding)


class ChatMessage(BaseModel):
    """A single chat message."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"] = Field(
        ..., description="Message role"
    )
    content: str = Field(..., description="Message content")
    images: Optional[List[str]] = Field(
        default=None, description="Base64 encoded images"
    )

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate content is not empty."""
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        return v.strip()


class ChatRequest(BaseModel):
    """Request for chat completion."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., description="Model to use for chat")
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    stream: bool = Field(default=False, description="Enable streaming response")
    format: Optional[Literal["json"]] = Field(
        default=None, description="Response format"
    )
    options: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional model options"
    )
    keep_alive: Optional[str] = Field(
        default=None, description="How long to keep model loaded"
    )

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: List[ChatMessage]) -> List[ChatMessage]:
        """Validate messages list."""
        if not v:
            raise ValueError("Messages list cannot be empty")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate model name."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


class ChatResponse(BaseModel):
    """Response from chat completion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model used")
    message: ChatMessage = Field(..., description="Generated message")
    done: bool = Field(..., description="Whether generation is complete")
    created_at: datetime = Field(..., description="Response creation time")
    total_duration: Optional[int] = Field(
        default=None, description="Total duration in nanoseconds"
    )
    load_duration: Optional[int] = Field(
        default=None, description="Model load duration in nanoseconds"
    )
    prompt_eval_count: Optional[int] = Field(
        default=None, description="Number of prompt tokens evaluated"
    )
    prompt_eval_duration: Optional[int] = Field(
        default=None, description="Prompt evaluation duration in nanoseconds"
    )
    eval_count: Optional[int] = Field(
        default=None, description="Number of tokens generated"
    )
    eval_duration: Optional[int] = Field(
        default=None, description="Generation duration in nanoseconds"
    )


class BatchRequest(BaseModel):
    """Request for batch processing multiple items."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(..., description="Model to use")
    items: List[Union[str, ChatMessage]] = Field(
        ..., description="Items to process in batch"
    )
    operation: Literal["embedding", "chat"] = Field(
        ..., description="Type of operation to perform"
    )
    batch_size: int = Field(
        default=10, ge=1, le=100, description="Number of items per batch"
    )
    options: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional options"
    )

    @field_validator("items")
    @classmethod
    def validate_items(cls, v: List[Union[str, ChatMessage]]) -> List[Union[str, ChatMessage]]:
        """Validate items list."""
        if not v:
            raise ValueError("Items list cannot be empty")
        if len(v) > 1000:  # Reasonable limit
            raise ValueError("Too many items in batch (max 1000)")
        return v


class BatchResponse(BaseModel):
    """Response from batch processing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="Model used")
    operation: str = Field(..., description="Operation performed")
    results: List[Union[EmbeddingResponse, ChatResponse]] = Field(
        ..., description="Results for each item"
    )
    total_items: int = Field(..., description="Total number of items processed")
    successful_items: int = Field(..., description="Number of successful items")
    failed_items: int = Field(..., description="Number of failed items")
    total_duration: float = Field(..., description="Total processing duration in seconds")
    errors: Optional[List[str]] = Field(
        default=None, description="Error messages for failed items"
    )

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.successful_items / self.total_items) * 100.0

    @property
    def average_duration_per_item(self) -> float:
        """Calculate average processing duration per item."""
        if self.total_items == 0:
            return 0.0
        return self.total_duration / self.total_items


class HealthCheckResponse(BaseModel):
    """Response from health check endpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["healthy", "unhealthy"] = Field(..., description="Health status")
    version: Optional[str] = Field(default=None, description="Ollama version")
    models_available: int = Field(..., description="Number of available models")
    uptime_seconds: Optional[float] = Field(
        default=None, description="Server uptime in seconds"
    )
    response_time_ms: float = Field(..., description="Response time in milliseconds")
    timestamp: datetime = Field(..., description="Check timestamp")


class ModelPullRequest(BaseModel):
    """Request to pull/download a model."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Model name to pull")
    insecure: bool = Field(default=False, description="Allow insecure connections")
    stream: bool = Field(default=True, description="Stream download progress")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate model name."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


class ModelPullProgress(BaseModel):
    """Progress update during model pull."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str = Field(..., description="Current status")
    digest: Optional[str] = Field(default=None, description="Download digest")
    total: Optional[int] = Field(default=None, description="Total bytes")
    completed: Optional[int] = Field(default=None, description="Completed bytes")

    @property
    def progress_percentage(self) -> Optional[float]:
        """Calculate download progress percentage."""
        if self.total is None or self.completed is None:
            return None
        if self.total == 0:
            return 100.0
        return (self.completed / self.total) * 100.0