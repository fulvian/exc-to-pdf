"""Session lifecycle adapters for Codex integration."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Dict, Optional

from .config import CodexSettings
from .logging import get_codex_logger


class SessionLifecycleAdapter:
    """Bridge session lifecycle events from Codex into DevStream services."""

    def __init__(self, settings: Optional[CodexSettings] = None) -> None:
        self.settings = settings or CodexSettings()
        self.logger = get_codex_logger("session_lifecycle")
        self._session_start_hook = self._load_class(
            relative_path=("tasks", "session_start.py"),
            class_name="SessionStartHook",
        )()
        task_module = self._load_module(
            relative_path=("tasks", "task_lifecycle_manager.py"),
        )
        self._task_lifecycle_manager = task_module.TaskLifecycleManager()
        self._task_event_enum = task_module.TaskLifecycleEvent
        self._stop_module = self._load_module(
            relative_path=("tasks", "stop.py"),
        )

    async def on_session_start(self, session_id: str, cwd: str) -> None:
        """Handle Codex session start."""
        payload: Dict[str, Any] = {"session_id": session_id, "cwd": cwd}

        await self._session_start_hook.process_session_start(payload)
        await self._task_lifecycle_manager.handle_lifecycle_event(
            self._task_event_enum.SESSION_START,
            payload,
        )

    async def on_tool_execution(
        self,
        session_id: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_result: Dict[str, Any],
    ) -> None:
        """Notify lifecycle manager about Codex tool execution."""
        payload = {
            "session_id": session_id,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_result": tool_result,
        }
        await self._task_lifecycle_manager.handle_lifecycle_event(
            self._task_event_enum.TOOL_EXECUTION,
            payload,
        )

    async def on_session_stop(self, session_id: str) -> Optional[str]:
        """Handle Codex session termination and return summary text."""
        summary: Optional[str] = None
        try:
            summary = await self._stop_module.extract_session_summary()
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error("Session summary generation failed", error=str(exc))

        await self._task_lifecycle_manager.handle_lifecycle_event(
            self._task_event_enum.SESSION_END,
            {"session_id": session_id},
        )

        return summary

    def _load_module(self, *, relative_path: tuple[str, ...]) -> Any:
        repo_root = Path(__file__).resolve().parents[4]
        module_path = repo_root / ".claude" / "hooks" / "devstream"
        for part in relative_path:
            module_path /= part

        if not module_path.exists():
            raise FileNotFoundError(f"Hook module not found: {module_path}")

        spec = spec_from_file_location(
            f"devstream_codex.{module_path.stem}",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module from {module_path}")

        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_class(self, *, relative_path: tuple[str, ...], class_name: str) -> Any:
        module = self._load_module(relative_path=relative_path)
        cls = getattr(module, class_name, None)
        if cls is None:
            raise AttributeError(f"{class_name} not found in module {module}")
        return cls


__all__ = ["SessionLifecycleAdapter"]
