"""
Configuration management per DevStream basato su Pydantic Settings.

Supporta multiple sorgenti di configurazione:
- Environment variables
- .env files
- YAML configuration files
- CLI arguments override
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """Configurazione database SQLite."""

    # Database path e connessione
    db_path: str = Field(default="./data/devstream.db", description="Path al database SQLite")
    connection_timeout: float = Field(default=30.0, description="Timeout connessione in secondi")
    max_connections: int = Field(default=10, description="Numero massimo connessioni pool")

    # Performance settings
    wal_mode: bool = Field(default=True, description="Enable WAL mode per performance")
    cache_size: int = Field(default=2000, description="Cache size in KB")

    # Vector search settings
    enable_vector_search: bool = Field(default=False, description="Enable sqlite-vss vector search")
    vector_dimension: int = Field(default=384, description="Dimensioni embedding vectors")

    # FTS5 settings
    enable_fts: bool = Field(default=True, description="Enable FTS5 full-text search")
    fts_tokenizer: str = Field(default="unicode61", description="FTS5 tokenizer")

    @field_validator("db_path")
    @classmethod
    def ensure_db_directory(cls, v: str) -> str:
        """Crea directory se non esiste."""
        db_path = Path(v)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return str(db_path)

    model_config = {"env_prefix": "DEVSTREAM_DB_"}


class OllamaConfig(BaseSettings):
    """Configurazione Ollama per embedding generation."""

    # Ollama server settings
    endpoint: str = Field(default="http://localhost:11434", description="Ollama server endpoint")
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Max retry attempts")

    # Model settings
    embedding_model: str = Field(default="nomic-embed-text", description="Embedding model name")
    model_download_timeout: float = Field(default=300.0, description="Model download timeout")

    # Performance settings
    batch_size: int = Field(default=32, description="Batch size for embedding generation")
    max_workers: int = Field(default=4, description="Max parallel workers")
    memory_threshold: float = Field(default=0.8, description="Memory usage threshold (0-1)")

    # Advanced settings
    enable_gpu: bool = Field(default=True, description="Use GPU if available")
    keep_alive: str = Field(default="5m", description="Model keep alive duration")

    model_config = {"env_prefix": "DEVSTREAM_OLLAMA_"}


class MemoryConfig(BaseSettings):
    """Configurazione sistema memoria semantica."""

    # Memory storage settings
    max_content_length: int = Field(default=10000, description="Max content length per memory item")
    max_memories_per_task: int = Field(default=100, description="Max memories per task")

    # Search settings
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold for search")
    max_search_results: int = Field(default=50, description="Max search results returned")

    # Context injection settings
    max_context_tokens: int = Field(default=2000, description="Max tokens for context injection")
    context_relevance_threshold: float = Field(default=0.6, description="Relevance threshold for context")

    # Memory curation settings
    enable_memory_decay: bool = Field(default=True, description="Enable automatic memory relevance decay")
    decay_rate: float = Field(default=0.1, description="Memory decay rate per day")
    min_relevance_score: float = Field(default=0.1, description="Minimum relevance before archival")

    model_config = {"env_prefix": "DEVSTREAM_MEMORY_"}


class TaskConfig(BaseSettings):
    """Configurazione sistema task management."""

    # Task constraints
    max_task_duration_minutes: int = Field(default=10, description="Max duration per micro-task")
    max_task_context_tokens: int = Field(default=256000, description="Max context tokens per task")

    # Plan generation settings
    max_objectives_per_plan: int = Field(default=10, description="Max objectives per intervention plan")
    max_phases_per_plan: int = Field(default=5, description="Max phases per plan")
    max_tasks_per_phase: int = Field(default=10, description="Max micro-tasks per phase")

    # Agent settings
    default_agent: str = Field(default="coder", description="Default agent assignment")
    agent_timeout: float = Field(default=300.0, description="Agent execution timeout")

    # Task tracking
    enable_time_tracking: bool = Field(default=True, description="Enable task time tracking")
    enable_token_tracking: bool = Field(default=True, description="Enable token usage tracking")

    model_config = {"env_prefix": "DEVSTREAM_TASK_"}


class HookConfig(BaseSettings):
    """Configurazione hook system."""

    # Hook execution settings
    enable_hooks: bool = Field(default=True, description="Enable hook system")
    hook_timeout: float = Field(default=30.0, description="Hook execution timeout")
    max_concurrent_hooks: int = Field(default=5, description="Max concurrent hook executions")

    # Task enforcement settings
    force_task_creation: bool = Field(default=True, description="Force task creation on user interaction")
    auto_save_memory: bool = Field(default=True, description="Auto-save task output to memory")
    auto_inject_context: bool = Field(default=True, description="Auto-inject relevant context")

    # Hook storage settings
    hook_log_retention_days: int = Field(default=30, description="Hook execution log retention")

    model_config = {"env_prefix": "DEVSTREAM_HOOK_"}


class LoggingConfig(BaseSettings):
    """Configurazione logging strutturato."""

    # Logging levels
    log_level: str = Field(default="INFO", description="Logging level")
    enable_structured_logging: bool = Field(default=True, description="Enable structured JSON logging")

    # Log destinations
    log_to_file: bool = Field(default=True, description="Log to file")
    log_file_path: str = Field(default="./logs/devstream.log", description="Log file path")
    log_to_console: bool = Field(default=True, description="Log to console")

    # Log rotation
    max_log_size_mb: int = Field(default=100, description="Max log file size in MB")
    log_retention_days: int = Field(default=30, description="Log file retention in days")

    @field_validator("log_file_path")
    @classmethod
    def ensure_log_directory(cls, v: str) -> str:
        """Crea directory log se non esiste."""
        log_path = Path(v)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return str(log_path)

    model_config = {"env_prefix": "DEVSTREAM_LOG_"}


class DevStreamConfig(BaseSettings):
    """Configurazione principale DevStream."""

    # Environment settings
    environment: str = Field(default="development", description="Environment (development/production/testing)")
    debug: bool = Field(default=False, description="Enable debug mode")

    # Feature flags
    enable_vector_search: bool = Field(default=False, description="Enable experimental vector search")
    enable_performance_monitoring: bool = Field(default=True, description="Enable performance monitoring")

    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tasks: TaskConfig = Field(default_factory=TaskConfig)
    hooks: HookConfig = Field(default_factory=HookConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "DevStreamConfig":
        """Carica configurazione da file YAML."""
        import yaml

        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)

    @classmethod
    def from_env(cls, env_file: Optional[Union[str, Path]] = None) -> "DevStreamConfig":
        """Carica configurazione da environment variables e .env file."""
        env_settings = {}

        if env_file:
            env_path = Path(env_file)
            if env_path.exists():
                from dotenv import load_dotenv
                load_dotenv(env_path)
        else:
            # Try to load from default .env file
            default_env = Path(".env")
            if default_env.exists():
                from dotenv import load_dotenv
                load_dotenv(default_env)

        return cls(**env_settings)

    def to_dict(self) -> Dict[str, Any]:
        """Converti configurazione in dictionary."""
        return self.model_dump()

    def validate_dependencies(self) -> List[str]:
        """Valida che le dipendenze necessarie siano disponibili."""
        issues = []

        # Check Ollama connectivity
        try:
            import httpx
            response = httpx.get(f"{self.ollama.endpoint}/api/tags", timeout=5.0)
            if response.status_code != 200:
                issues.append(f"Ollama server not reachable at {self.ollama.endpoint}")
        except Exception as e:
            issues.append(f"Ollama connectivity check failed: {e}")

        # Check vector search dependencies
        if self.enable_vector_search:
            try:
                import sqlite_vss  # type: ignore
            except ImportError:
                issues.append("sqlite-vss not installed but vector search is enabled")

        # Check database path
        try:
            db_path = Path(self.database.db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Database path validation failed: {e}")

        return issues

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "env_nested_delimiter": "__",
        "extra": "ignore"  # Ignore extra environment variables
    }