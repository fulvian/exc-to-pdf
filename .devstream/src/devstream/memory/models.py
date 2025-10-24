"""
Memory System Pydantic Models

Type-safe models per memory entries, search queries,
e risultati con full validation.
"""

import json
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field, validator


class ContentType(str, Enum):
    """Types di contenuto per memory entries."""
    CODE = "code"
    DOCUMENTATION = "documentation"
    CONTEXT = "context"
    OUTPUT = "output"
    ERROR = "error"
    DECISION = "decision"
    LEARNING = "learning"


class ContentFormat(str, Enum):
    """Formati di contenuto supportati."""
    TEXT = "text"
    MARKDOWN = "markdown"
    CODE = "code"
    JSON = "json"
    YAML = "yaml"


class MemoryEntry(BaseModel):
    """
    Memory entry con embedding e metadata semantici.

    Model core per storage di contenuti con embedding vettoriali
    e metadati per search e retrieval efficace.
    """
    id: str = Field(..., description="Unique identifier per memory entry")

    # Foreign keys (optional per flexibility)
    plan_id: Optional[str] = Field(None, description="Associated intervention plan ID")
    phase_id: Optional[str] = Field(None, description="Associated phase ID")
    task_id: Optional[str] = Field(None, description="Associated task ID")

    # Content
    content: str = Field(..., description="Main content text")
    content_type: ContentType = Field(..., description="Type of content")
    content_format: ContentFormat = Field(ContentFormat.TEXT, description="Content format")

    # Semantic metadata
    keywords: list[str] = Field(default_factory=list, description="Extracted keywords")
    entities: list[dict[str, str]] = Field(default_factory=list, description="Named entities")
    sentiment: float = Field(0.0, ge=-1.0, le=1.0, description="Sentiment score")
    complexity_score: int = Field(1, ge=1, le=10, description="Content complexity")

    # Embedding storage
    embedding: Optional[list[float]] = Field(None, description="Vector embedding")
    embedding_model: Optional[str] = Field(None, description="Model used for embedding")
    embedding_dimension: Optional[int] = Field(None, description="Embedding dimension")

    # Context and relations
    context_snapshot: dict[str, Any] = Field(default_factory=dict, description="Context at creation")
    related_memory_ids: list[str] = Field(default_factory=list, description="Related memory IDs")

    # Management metadata
    access_count: int = Field(0, description="Number of times accessed")
    last_accessed_at: Optional[datetime] = Field(None, description="Last access timestamp")
    relevance_score: float = Field(1.0, ge=0.0, le=1.0, description="Relevance score")
    is_archived: bool = Field(False, description="Archive status")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Update timestamp")

    @validator('embedding')
    def validate_embedding_dimension(cls, v, values):
        """Validate embedding dimension matches declared dimension."""
        if v is not None and values.get('embedding_dimension') is not None:
            embedding_dim = values.get('embedding_dimension')
            if len(v) != embedding_dim:
                raise ValueError(f"Embedding length {len(v)} doesn't match dimension {embedding_dim}")
        return v

    def set_embedding(self, embedding_array: np.ndarray, model_name: str = "embeddinggemma") -> None:
        """Set embedding from numpy array."""
        self.embedding = embedding_array.tolist()
        self.embedding_dimension = len(self.embedding)
        self.embedding_model = model_name

    def get_embedding_array(self) -> Optional[np.ndarray]:
        """Get embedding as numpy array."""
        if self.embedding is None:
            return None
        return np.array(self.embedding, dtype=np.float32)

    class Config:
        """Pydantic config."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            np.ndarray: lambda v: v.tolist(),
        }


class SearchQuery(BaseModel):
    """
    Query model per hybrid search operations.

    Definisce parametri per semantic e keyword search
    con opzioni di filtering e ranking.
    """
    query_text: str = Field(..., description="Query text")
    max_results: int = Field(10, ge=1, le=100, description="Maximum results to return")

    # Search method weights
    semantic_weight: float = Field(1.0, ge=0.0, le=2.0, description="Weight for semantic search")
    keyword_weight: float = Field(1.0, ge=0.0, le=2.0, description="Weight for keyword search")

    # Filtering options
    content_types: Optional[list[ContentType]] = Field(None, description="Filter by content types")
    min_relevance: float = Field(0.0, ge=0.0, le=1.0, description="Minimum relevance threshold")
    created_after: Optional[datetime] = Field(None, description="Filter by creation date")
    created_before: Optional[datetime] = Field(None, description="Filter by creation date")

    # Context filtering
    plan_id: Optional[str] = Field(None, description="Filter by plan ID")
    phase_id: Optional[str] = Field(None, description="Filter by phase ID")
    task_id: Optional[str] = Field(None, description="Filter by task ID")

    # RRF parameters
    rrf_k: int = Field(60, ge=1, le=100, description="RRF constant for rank fusion")

    class Config:
        """Pydantic config."""
        use_enum_values = True


class MemoryQueryResult(BaseModel):
    """
    Result model per search queries.

    Include memory entry, score, e ranking metadata
    per analysis e debugging.
    """
    memory_entry: MemoryEntry = Field(..., description="Memory entry")

    # Scoring information
    combined_score: float = Field(..., description="Final combined relevance score")
    semantic_score: Optional[float] = Field(None, description="Semantic similarity score")
    keyword_score: Optional[float] = Field(None, description="Keyword match score")

    # Ranking information
    semantic_rank: Optional[int] = Field(None, description="Rank in semantic results")
    keyword_rank: Optional[int] = Field(None, description="Rank in keyword results")
    final_rank: int = Field(..., description="Final rank after fusion")

    # Matching details
    matched_keywords: list[str] = Field(default_factory=list, description="Keywords that matched")
    matched_entities: list[str] = Field(default_factory=list, description="Entities that matched")


class ContextAssemblyResult(BaseModel):
    """
    Result model per context assembly operations.

    Include assembled context, token usage,
    e metadata per optimization.
    """
    assembled_context: str = Field(..., description="Final assembled context")
    memory_entries: list[MemoryEntry] = Field(..., description="Memory entries used")

    # Token management
    total_tokens: int = Field(..., description="Total tokens in assembled context")
    tokens_budget: int = Field(..., description="Available token budget")
    tokens_remaining: int = Field(..., description="Remaining tokens after assembly")

    # Assembly metadata
    relevance_threshold: float = Field(..., description="Relevance threshold used")
    truncated: bool = Field(False, description="Whether context was truncated")
    assembly_strategy: str = Field("relevance", description="Strategy used for assembly")

    # Timing
    assembly_time_ms: float = Field(..., description="Time taken for assembly in milliseconds")

    @validator('tokens_remaining')
    def validate_token_budget(cls, v, values):
        """Validate token budget calculations."""
        total = values.get('total_tokens', 0)
        budget = values.get('tokens_budget', 0)
        expected_remaining = budget - total

        if abs(v - expected_remaining) > 1:  # Allow small rounding differences
            raise ValueError(f"Token calculation error: {v} != {expected_remaining}")
        return v