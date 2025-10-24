"""
Production configuration management for DevStream.

Context7-validated patterns for environment-specific configuration:
- Environment variable-based configuration
- Secure secrets management
- Multi-environment support
- Configuration validation
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass, field

from pydantic import BaseSettings, Field, validator, SecretStr
from pydantic_settings import SettingsConfigDict
import structlog

logger = structlog.get_logger(__name__)


class DatabaseProductionConfig(BaseSettings):
    """Production database configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DEVSTREAM_DB_",
        case_sensitive=False
    )

    # Database paths and connection
    db_path: str = Field(
        default="/opt/devstream/data/production.db",
        description="Production database file path"
    )

    backup_dir: str = Field(
        default="/opt/devstream/backups",
        description="Database backup directory"
    )

    # Connection pool settings
    max_connections: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Maximum database connections"
    )

    connection_timeout: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Connection timeout in seconds"
    )

    # Backup and maintenance
    backup_enabled: bool = Field(
        default=True,
        description="Enable automated backups"
    )

    backup_interval: int = Field(
        default=3600,
        ge=300,
        description="Backup interval in seconds"
    )

    backup_retention_days: int = Field(
        default=30,
        ge=1,
        description="Number of days to retain backups"
    )

    # Performance tuning
    wal_mode: bool = Field(
        default=True,
        description="Enable SQLite WAL mode for better concurrency"
    )

    pragma_settings: Dict[str, Any] = Field(
        default_factory=lambda: {
            "journal_mode": "WAL",
            "synchronous": "NORMAL",
            "cache_size": 10000,
            "temp_store": "memory",
            "mmap_size": 268435456,  # 256MB
        },
        description="SQLite PRAGMA settings for production"
    )

    @property
    def async_url(self) -> str:
        """Get async SQLAlchemy URL."""
        return f"sqlite+aiosqlite:///{self.db_path}"

    @property
    def sync_url(self) -> str:
        """Get sync SQLAlchemy URL."""
        return f"sqlite:///{self.db_path}"


class SecurityProductionConfig(BaseSettings):
    """Production security configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DEVSTREAM_SECURITY_",
        case_sensitive=False
    )

    # Authentication
    secret_key: SecretStr = Field(
        description="Secret key for JWT and encryption (required in production)"
    )

    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )

    jwt_expiration: int = Field(
        default=3600,
        ge=300,
        description="JWT token expiration in seconds"
    )

    # API Security
    api_key_required: bool = Field(
        default=True,
        description="Require API key for access"
    )

    allowed_hosts: List[str] = Field(
        default_factory=lambda: ["api.devstream.local"],
        description="Allowed hostnames for requests"
    )

    cors_enabled: bool = Field(
        default=False,
        description="Enable CORS (should be false in production)"
    )

    cors_origins: List[str] = Field(
        default_factory=list,
        description="Allowed CORS origins"
    )

    # Rate limiting
    rate_limit_enabled: bool = Field(
        default=True,
        description="Enable rate limiting"
    )

    rate_limit_requests: int = Field(
        default=1000,
        ge=1,
        description="Rate limit: requests per hour"
    )

    rate_limit_burst: int = Field(
        default=50,
        ge=1,
        description="Rate limit: burst requests"
    )

    @validator("secret_key")
    def secret_key_must_be_strong(cls, v):
        """Validate secret key strength in production."""
        secret = v.get_secret_value() if hasattr(v, 'get_secret_value') else str(v)
        if len(secret) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v


class LoggingProductionConfig(BaseSettings):
    """Production logging configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DEVSTREAM_LOG_",
        case_sensitive=False
    )

    # Log levels and formatting
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )

    format: Literal["structured", "text", "json"] = Field(
        default="structured",
        description="Log format for production"
    )

    # File logging
    file_enabled: bool = Field(
        default=True,
        description="Enable file logging"
    )

    file_path: str = Field(
        default="/opt/devstream/logs/devstream.log",
        description="Log file path"
    )

    file_max_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        ge=1024 * 1024,
        description="Maximum log file size in bytes"
    )

    file_backup_count: int = Field(
        default=5,
        ge=1,
        description="Number of backup log files to keep"
    )

    # Console logging
    console_enabled: bool = Field(
        default=True,
        description="Enable console logging"
    )

    # Structured logging features
    include_timestamps: bool = Field(
        default=True,
        description="Include timestamps in logs"
    )

    include_caller_info: bool = Field(
        default=False,
        description="Include caller file/line info (performance impact)"
    )

    # Log filtering
    excluded_loggers: List[str] = Field(
        default_factory=lambda: [
            "urllib3.connectionpool",
            "asyncio",
        ],
        description="Logger names to exclude/silence"
    )


class MonitoringProductionConfig(BaseSettings):
    """Production monitoring configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DEVSTREAM_MONITORING_",
        case_sensitive=False
    )

    # Health checks
    health_check_enabled: bool = Field(
        default=True,
        description="Enable health check endpoint"
    )

    health_check_path: str = Field(
        default="/health",
        description="Health check endpoint path"
    )

    health_check_timeout: int = Field(
        default=10,
        ge=1,
        description="Health check timeout in seconds"
    )

    # Metrics collection
    metrics_enabled: bool = Field(
        default=True,
        description="Enable metrics collection"
    )

    metrics_path: str = Field(
        default="/metrics",
        description="Metrics endpoint path"
    )

    # Performance monitoring
    profiling_enabled: bool = Field(
        default=False,
        description="Enable performance profiling (dev only)"
    )

    slow_query_threshold: float = Field(
        default=1.0,
        ge=0.1,
        description="Log queries slower than this (seconds)"
    )

    # Error tracking
    error_tracking_enabled: bool = Field(
        default=True,
        description="Enable error tracking and alerts"
    )

    error_sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Error sampling rate (0.0 to 1.0)"
    )


class PerformanceProductionConfig(BaseSettings):
    """Production performance configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DEVSTREAM_PERF_",
        case_sensitive=False
    )

    # Server configuration
    workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of worker processes"
    )

    max_requests: int = Field(
        default=1000,
        ge=100,
        description="Max requests per worker before restart"
    )

    timeout: int = Field(
        default=120,
        ge=30,
        description="Request timeout in seconds"
    )

    keepalive: int = Field(
        default=5,
        ge=1,
        description="Keep-alive timeout"
    )

    # Memory management
    memory_limit_mb: int = Field(
        default=512,
        ge=128,
        description="Memory limit per worker in MB"
    )

    # Connection limits
    max_concurrent_connections: int = Field(
        default=100,
        ge=10,
        description="Maximum concurrent connections"
    )

    # Caching
    cache_enabled: bool = Field(
        default=True,
        description="Enable response caching"
    )

    cache_ttl: int = Field(
        default=300,
        ge=60,
        description="Cache TTL in seconds"
    )


class ProductionConfig(BaseSettings):
    """
    Master production configuration combining all sub-configurations.

    Context7-validated pattern for comprehensive production setup.
    """

    model_config = SettingsConfigDict(
        env_prefix="DEVSTREAM_",
        case_sensitive=False,
        env_file=".env.production",
        env_file_encoding="utf-8"
    )

    # Environment identification
    environment: Literal["production"] = Field(
        default="production",
        description="Environment name"
    )

    version: str = Field(
        default="1.0.0",
        description="Application version"
    )

    debug: bool = Field(
        default=False,
        description="Debug mode (should be False in production)"
    )

    # Sub-configurations
    database: DatabaseProductionConfig = Field(
        default_factory=DatabaseProductionConfig,
        description="Database configuration"
    )

    security: SecurityProductionConfig = Field(
        default_factory=SecurityProductionConfig,
        description="Security configuration"
    )

    logging: LoggingProductionConfig = Field(
        default_factory=LoggingProductionConfig,
        description="Logging configuration"
    )

    monitoring: MonitoringProductionConfig = Field(
        default_factory=MonitoringProductionConfig,
        description="Monitoring configuration"
    )

    performance: PerformanceProductionConfig = Field(
        default_factory=PerformanceProductionConfig,
        description="Performance configuration"
    )

    # Deployment information
    deployment_timestamp: Optional[str] = Field(
        default=None,
        description="Deployment timestamp"
    )

    git_commit: Optional[str] = Field(
        default=None,
        description="Git commit SHA"
    )

    deployment_id: Optional[str] = Field(
        default=None,
        description="Unique deployment identifier"
    )

    def validate_production_requirements(self) -> List[str]:
        """
        Validate that all production requirements are met.

        Returns:
            List of validation errors
        """
        errors = []

        # Check required production settings
        if self.debug:
            errors.append("Debug mode must be disabled in production")

        if not self.security.secret_key:
            errors.append("Secret key is required in production")

        # Check database configuration
        db_path = Path(self.database.db_path)
        if not db_path.parent.exists():
            errors.append(f"Database directory does not exist: {db_path.parent}")

        # Check backup configuration
        if self.database.backup_enabled:
            backup_dir = Path(self.database.backup_dir)
            if not backup_dir.exists():
                errors.append(f"Backup directory does not exist: {backup_dir}")

        # Check log directory
        if self.logging.file_enabled:
            log_path = Path(self.logging.file_path)
            if not log_path.parent.exists():
                errors.append(f"Log directory does not exist: {log_path.parent}")

        # Validate security settings
        if not self.security.allowed_hosts:
            errors.append("Allowed hosts must be configured")

        if self.security.cors_enabled and self.environment == "production":
            errors.append("CORS should be disabled in production")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding secrets)."""
        config_dict = self.dict()

        # Remove sensitive information
        if "security" in config_dict and "secret_key" in config_dict["security"]:
            config_dict["security"]["secret_key"] = "***REDACTED***"

        return config_dict

    @classmethod
    def load_from_environment(cls, env_file: Optional[str] = None) -> "ProductionConfig":
        """
        Load production configuration from environment.

        Args:
            env_file: Optional environment file path

        Returns:
            Configured ProductionConfig instance
        """
        if env_file:
            return cls(_env_file=env_file)
        return cls()

    def setup_logging(self) -> None:
        """Setup production logging based on configuration."""
        import logging.config
        import structlog

        # Configure standard logging
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "format": "%(message)s"
                },
                "structured": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                }
            },
            "handlers": {},
            "root": {
                "level": self.logging.level,
                "handlers": []
            }
        }

        # Console handler
        if self.logging.console_enabled:
            logging_config["handlers"]["console"] = {
                "class": "logging.StreamHandler",
                "level": self.logging.level,
                "formatter": self.logging.format,
                "stream": "ext://sys.stdout"
            }
            logging_config["root"]["handlers"].append("console")

        # File handler
        if self.logging.file_enabled:
            logging_config["handlers"]["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": self.logging.level,
                "formatter": self.logging.format,
                "filename": self.logging.file_path,
                "maxBytes": self.logging.file_max_size,
                "backupCount": self.logging.file_backup_count
            }
            logging_config["root"]["handlers"].append("file")

        # Apply configuration
        logging.config.dictConfig(logging_config)

        # Configure structlog
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
        ]

        if self.logging.include_timestamps:
            processors.append(structlog.processors.TimeStamper(fmt="ISO"))

        if self.logging.format == "json":
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(
                logging.getLevelName(self.logging.level)
            ),
            logger_factory=structlog.WriteLoggerFactory(),
            cache_logger_on_first_use=True,
        )

        # Silence excluded loggers
        for logger_name in self.logging.excluded_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

        logger.info(
            "Production logging configured",
            level=self.logging.level,
            format=self.logging.format,
            file_enabled=self.logging.file_enabled,
            console_enabled=self.logging.console_enabled
        )


# Global configuration instance
_production_config: Optional[ProductionConfig] = None


def get_production_config() -> ProductionConfig:
    """Get global production configuration instance."""
    global _production_config

    if _production_config is None:
        _production_config = ProductionConfig.load_from_environment()

        # Validate production requirements
        errors = _production_config.validate_production_requirements()
        if errors:
            error_msg = "Production configuration validation failed:\n" + "\n".join(f"- {err}" for err in errors)
            raise ValueError(error_msg)

        # Setup logging
        _production_config.setup_logging()

        logger.info(
            "Production configuration loaded",
            environment=_production_config.environment,
            version=_production_config.version,
            debug=_production_config.debug
        )

    return _production_config


def reset_production_config() -> None:
    """Reset global configuration (for testing)."""
    global _production_config
    _production_config = None


# Configuration factory for different environments
def create_config_for_environment(environment: str) -> ProductionConfig:
    """
    Create configuration for specific environment.

    Args:
        environment: Environment name

    Returns:
        Environment-specific configuration
    """
    env_file_map = {
        "production": ".env.production",
        "staging": ".env.staging",
        "development": ".env.development",
        "testing": ".env.testing"
    }

    env_file = env_file_map.get(environment)
    if env_file and Path(env_file).exists():
        return ProductionConfig.load_from_environment(env_file)

    return ProductionConfig.load_from_environment()