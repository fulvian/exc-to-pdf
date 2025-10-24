"""
Configuration for Ollama client.

Production-ready configuration with Context7-validated defaults.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_retries: int = Field(
        default=3, ge=0, le=10, description="Maximum number of retry attempts"
    )
    base_delay: float = Field(
        default=1.0, ge=0.1, le=60.0, description="Base delay between retries in seconds"
    )
    max_delay: float = Field(
        default=60.0, ge=1.0, le=300.0, description="Maximum delay between retries"
    )
    exponential_base: float = Field(
        default=2.0, ge=1.1, le=10.0, description="Exponential backoff base"
    )
    jitter: bool = Field(
        default=True, description="Add random jitter to prevent thundering herd"
    )
    retry_on_status_codes: List[int] = Field(
        default=[429, 502, 503, 504], description="HTTP status codes to retry on"
    )
    retry_on_timeout: bool = Field(
        default=True, description="Whether to retry on timeout errors"
    )
    retry_on_connection_error: bool = Field(
        default=True, description="Whether to retry on connection errors"
    )

    @field_validator("max_delay")
    @classmethod
    def validate_max_delay_greater_than_base(cls, v: float, info) -> float:
        """Ensure max_delay is greater than base_delay."""
        if hasattr(info, "data") and "base_delay" in info.data:
            base_delay = info.data["base_delay"]
            if v <= base_delay:
                raise ValueError("max_delay must be greater than base_delay")
        return v


class TimeoutConfig(BaseModel):
    """Configuration for timeout behavior."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    connect_timeout: float = Field(
        default=10.0, ge=1.0, le=60.0, description="Connection timeout in seconds"
    )
    read_timeout: float = Field(
        default=120.0, ge=5.0, le=600.0, description="Read timeout in seconds"
    )
    write_timeout: float = Field(
        default=60.0, ge=5.0, le=300.0, description="Write timeout in seconds"
    )
    pool_timeout: float = Field(
        default=10.0, ge=1.0, le=60.0, description="Connection pool timeout in seconds"
    )

    @property
    def total_timeout(self) -> float:
        """Calculate total maximum timeout."""
        return self.connect_timeout + max(self.read_timeout, self.write_timeout)


class BatchConfig(BaseModel):
    """Configuration for batch processing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    default_batch_size: int = Field(
        default=10, ge=1, le=100, description="Default batch size for processing"
    )
    max_batch_size: int = Field(
        default=50, ge=1, le=1000, description="Maximum allowed batch size"
    )
    max_concurrent_batches: int = Field(
        default=3, ge=1, le=10, description="Maximum concurrent batch operations"
    )
    memory_limit_mb: int = Field(
        default=512, ge=64, le=4096, description="Memory limit for batch processing"
    )
    enable_streaming: bool = Field(
        default=True, description="Enable streaming for large batches"
    )

    @field_validator("max_batch_size")
    @classmethod
    def validate_max_batch_size(cls, v: int, info) -> int:
        """Ensure max_batch_size is >= default_batch_size."""
        if hasattr(info, "data") and "default_batch_size" in info.data:
            default_size = info.data["default_batch_size"]
            if v < default_size:
                raise ValueError("max_batch_size must be >= default_batch_size")
        return v


class FallbackConfig(BaseModel):
    """Configuration for fallback mechanisms."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enable_fallback: bool = Field(
        default=True, description="Enable fallback mechanisms"
    )
    fallback_models: List[str] = Field(
        default=["llama3.2", "gemma2"], description="Fallback models in order of preference"
    )
    fallback_on_model_not_found: bool = Field(
        default=True, description="Use fallback when model is not found"
    )
    fallback_on_timeout: bool = Field(
        default=False, description="Use fallback on timeout (dangerous)"
    )
    auto_pull_missing_models: bool = Field(
        default=True, description="Automatically pull missing models"
    )
    max_auto_pull_attempts: int = Field(
        default=1, ge=0, le=3, description="Maximum auto-pull attempts per model"
    )


class PerformanceConfig(BaseModel):
    """Configuration for performance optimization."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    connection_pool_size: int = Field(
        default=10, ge=1, le=100, description="HTTP connection pool size"
    )
    connection_pool_max_keepalive: int = Field(
        default=5, ge=1, le=50, description="Maximum keep-alive connections"
    )
    enable_http2: bool = Field(
        default=True, description="Enable HTTP/2 support"
    )
    enable_compression: bool = Field(
        default=True, description="Enable response compression"
    )
    trust_env: bool = Field(
        default=True, description="Trust environment variables for proxy config"
    )
    verify_ssl: bool = Field(
        default=True, description="Verify SSL certificates"
    )


class OllamaConfig(BaseSettings):
    """
    Main configuration for Ollama client.

    Supports loading from environment variables with OLLAMA_ prefix.
    """

    model_config = ConfigDict(
        env_prefix="OLLAMA_",
        env_file=".env",
        extra="ignore",  # Ignore extra environment variables
        case_sensitive=False,
    )

    # Connection settings
    host: str = Field(
        default="http://localhost:11434",
        description="Ollama server host URL"
    )
    api_version: str = Field(
        default="v1", description="API version to use"
    )

    # Authentication (if needed in future)
    api_key: Optional[str] = Field(
        default=None, description="API key for authentication"
    )

    # Default models
    default_embedding_model: str = Field(
        default="nomic-embed-text", description="Default model for embeddings"
    )
    default_chat_model: str = Field(
        default="llama3.2", description="Default model for chat"
    )

    # Component configurations
    timeout: TimeoutConfig = Field(
        default_factory=TimeoutConfig, description="Timeout configuration"
    )
    retry: RetryConfig = Field(
        default_factory=RetryConfig, description="Retry configuration"
    )
    batch: BatchConfig = Field(
        default_factory=BatchConfig, description="Batch processing configuration"
    )
    fallback: FallbackConfig = Field(
        default_factory=FallbackConfig, description="Fallback configuration"
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig, description="Performance configuration"
    )

    # Health check settings
    health_check_interval: float = Field(
        default=60.0, ge=10.0, le=3600.0, description="Health check interval in seconds"
    )
    health_check_timeout: float = Field(
        default=5.0, ge=1.0, le=30.0, description="Health check timeout in seconds"
    )

    # Logging and monitoring
    enable_metrics: bool = Field(
        default=True, description="Enable performance metrics collection"
    )
    log_requests: bool = Field(
        default=False, description="Log all requests (debug mode)"
    )
    log_responses: bool = Field(
        default=False, description="Log all responses (debug mode)"
    )

    # Model management
    model_cache_size: int = Field(
        default=5, ge=1, le=20, description="Number of models to keep in memory"
    )
    model_keep_alive: str = Field(
        default="5m", description="How long to keep models loaded"
    )

    # Additional custom options for models
    default_model_options: Dict[str, Any] = Field(
        default_factory=dict, description="Default options for all models"
    )

    @field_validator("host")
    @classmethod
    def validate_host_url(cls, v: str) -> str:
        """Validate host URL format."""
        v = v.strip()
        if not v:
            raise ValueError("Host URL cannot be empty")

        if not v.startswith(("http://", "https://")):
            # Default to http if no scheme provided
            v = f"http://{v}"

        # Remove trailing slash
        return v.rstrip("/")

    @field_validator("default_model_options")
    @classmethod
    def validate_model_options(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model options."""
        # Ensure reasonable limits on model options
        if len(v) > 50:
            raise ValueError("Too many default model options (max 50)")
        return v

    @property
    def base_url(self) -> str:
        """Get the base API URL."""
        return f"{self.host}/api/{self.api_version}"

    @property
    def health_url(self) -> str:
        """Get the health check URL."""
        return f"{self.host}/api/version"

    def get_model_options(self, additional_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get merged model options with defaults."""
        options = self.default_model_options.copy()
        if additional_options:
            options.update(additional_options)
        return options

    def to_httpx_timeout(self) -> object:
        """Convert timeout config to HTTPX timeout object."""
        import httpx
        return httpx.Timeout(
            timeout=self.timeout.total_timeout,
            connect=self.timeout.connect_timeout,
            read=self.timeout.read_timeout,
            write=self.timeout.write_timeout,
            pool=self.timeout.pool_timeout,
        )

    def to_httpx_limits(self) -> object:
        """Convert performance config to HTTPX limits object."""
        import httpx
        return httpx.Limits(
            max_connections=self.performance.connection_pool_size,
            max_keepalive_connections=self.performance.connection_pool_max_keepalive,
        )