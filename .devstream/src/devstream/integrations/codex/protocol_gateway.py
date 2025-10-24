"""Codex adapter for DevStream protocol enforcement."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Dict, Optional

from .config import CodexSettings
from .logging import get_codex_logger


@dataclass(frozen=True)
class ProtocolAssessment:
    """Result of protocol enforcement analysis."""

    enforce_protocol: bool
    prompt: Optional[str]
    triggers: tuple[str, ...]


class ProtocolGatewayAdapter:
    """Bridge that reuses Claude hook logic for Codex user prompts."""

    def __init__(self, settings: Optional[CodexSettings] = None) -> None:
        self.settings = settings or CodexSettings()
        self.logger = get_codex_logger("protocol_gateway")
        self._hook = self._load_user_prompt_hook()

    async def evaluate_prompt(self, user_input: str) -> ProtocolAssessment:
        """Evaluate a Codex prompt and return enforcement guidance."""
        complexity = self._hook.estimate_task_complexity(user_input)
        triggers = tuple(complexity.get("triggers", []))
        enforce = bool(complexity.get("enforce_protocol"))
        prompt: Optional[str] = None

        if enforce:
            try:
                prompt = await asyncio.wait_for(
                    self._hook.check_protocol_enforcement(user_input),
                    timeout=self.settings.protocol_timeout_seconds,
                )
            except asyncio.TimeoutError:
                self.logger.warning("Protocol enforcement timed out", user_input_preview=user_input[:120])
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.error(
                    "Protocol enforcement failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

        return ProtocolAssessment(
            enforce_protocol=enforce,
            prompt=prompt,
            triggers=triggers,
        )

    def _load_user_prompt_hook(self) -> Any:
        """Dynamically import the Claude UserPromptSubmit hook."""
        repo_root = Path(__file__).resolve().parents[4]
        hook_path = repo_root / ".claude" / "hooks" / "devstream" / "context" / "user_query_context_enhancer.py"

        if not hook_path.exists():
            raise FileNotFoundError(f"User prompt hook not found at {hook_path}")

        spec = spec_from_file_location("devstream_codex.user_prompt_hook", hook_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load hook module from {hook_path}")

        module = module_from_spec(spec)
        spec.loader.exec_module(module)

        hook_cls = getattr(module, "UserPromptSubmitHook", None)
        if hook_cls is None:
            raise AttributeError("UserPromptSubmitHook class not found in hook module")

        return hook_cls()


__all__ = ["ProtocolGatewayAdapter", "ProtocolAssessment"]
