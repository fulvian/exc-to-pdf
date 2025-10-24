from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


class CodexSettings(BaseSettings):
    """Runtime configuration for the Codex adapter layer."""

    model_config = ConfigDict(
        env_prefix="DEVSTREAM_CODEX_",
        env_file=".env.devstream",
        extra="ignore",
        frozen=True,
    )

    enabled: bool = Field(
        default=True,
        description="Enable DevStream Codex adapters when true.",
    )
    protocol_timeout_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=30.0,
        description="Timeout applied when invoking protocol enforcement logic.",
    )
    context_token_budget: int = Field(
        default=2000,
        ge=128,
        le=8192,
        description="Maximum tokens to allocate when assembling Codex context payloads.",
    )
    mcp_timeout_ms: int = Field(
        default=5000,
        ge=1000,
        le=20000,
        description="Timeout in milliseconds for MCP calls issued by Codex adapters.",
    )
    sample_output_path: Optional[Path] = Field(
        default=Path("data/codex_event_samples.jsonl"),
        description="Where to persist captured Codex payload samples (JSONL).",
    )
    config_path: Optional[Path] = Field(
        default=None,
        description="Optional override pointing to a Codex-specific YAML configuration file.",
    )

    @classmethod
    def load(cls, *, override_path: Optional[Path] = None) -> "CodexSettings":
        """Load settings from environment and optional override file."""
        path = override_path or cls().config_path
        if path is None:
            return cls()

        resolved = path.expanduser().resolve()
        data = _load_yaml(resolved)
        return cls(**data)

    def resolve_sample_path(self) -> Optional[Path]:
        """Resolve the sample output path if configured."""
        if self.sample_output_path is None:
            return None
        return self.sample_output_path.expanduser().resolve()


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML configuration from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Codex config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError("Codex config YAML must contain a mapping at the root")

    return data


__all__ = ["CodexSettings"]
