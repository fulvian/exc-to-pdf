"""Context injection and memory capture adapters for Codex."""

from __future__ import annotations

import asyncio
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Dict, Optional

from .config import CodexSettings
from .logging import get_codex_logger


class ContextPipelineAdapter:
    """Adapter that reuses Claude pre/post tool hooks for Codex events."""

    def __init__(self, settings: Optional[CodexSettings] = None) -> None:
        self.settings = settings or CodexSettings()
        self.logger = get_codex_logger("context_pipeline")
        self._pre_hook = self._load_hook(
            relative_path=("memory", "pre_tool_use.py"),
            class_name="PreToolUseHook",
        )
        self._post_hook = self._load_hook(
            relative_path=("memory", "post_tool_use.py"),
            class_name="PostToolUseHook",
        )

    async def build_pre_context(self, file_path: str, content: str) -> Optional[str]:
        """Return enhanced context for a pending tool execution."""
        try:
            return await asyncio.wait_for(
                self._pre_hook.assemble_context(file_path, content),
                timeout=self.settings.protocol_timeout_seconds,
            )
        except asyncio.TimeoutError:
            self.logger.warning("Pre-tool context assembly timed out", file_path=file_path)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error(
                "Pre-tool context assembly failed",
                error=str(exc),
                file_path=file_path,
            )
        return None

    async def capture_post_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_result: Dict[str, Any],
    ) -> Optional[str]:
        """Replica il routing multi-tool dell'hook PostToolUse originale."""

        file_path = ""
        content = ""
        should_store = False

        if tool_name in {"Write", "Edit", "MultiEdit"}:
            file_path = tool_input.get("file_path", "")
            content = tool_input.get("content", "") or tool_input.get("new_string", "")
            should_store = bool(file_path and content)
        elif tool_name == "Bash":
            if self._post_hook.should_capture_bash_output(tool_input, tool_result):
                command = tool_input.get("command", "")
                output = tool_result.get("output", "")
                file_path = f"bash_output/{command[:50].replace(' ', '_')}.txt"
                content = f"# Command: {command}\n\n{output}"
                should_store = bool(content.strip())
        elif tool_name == "Read":
            candidate_path = tool_input.get("file_path", "")
            if candidate_path and self._post_hook.should_capture_read_content(candidate_path):
                file_path = candidate_path
                content = tool_result.get("content", "")
                should_store = bool(content)
        elif tool_name == "TodoWrite":
            todos = tool_input.get("todos", [])
            if todos:
                file_path = "todo_updates/task_list.json"
                content = json.dumps(todos, indent=2)
                should_store = True

        if not should_store:
            return None

        excluded_paths = [
            ".git/",
            "node_modules/",
            ".venv/",
            ".devstream/",
            "__pycache__/",
            "dist/",
            "build/",
        ]
        if any(part in file_path for part in excluded_paths):
            return None

        topics = self._post_hook.extract_topics(content, file_path)
        entities = self._post_hook.extract_entities(content)
        content_type = self._post_hook.classify_content_type(tool_name, tool_result, content)

        memory_id = await self._post_hook.store_in_memory(
            file_path=file_path,
            content=content,
            operation=tool_name,
            topics=topics,
            entities=entities,
            content_type=content_type,
        )

        await self._post_hook.update_session_tracking(tool_name, tool_input)

        try:
            await asyncio.wait_for(
                self._post_hook.trigger_checkpoint_for_critical_tool(tool_name),
                timeout=self.settings.protocol_timeout_seconds,
            )
        except asyncio.TimeoutError:
            self.logger.warning("Checkpoint trigger timed out", tool_name=tool_name)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error(
                "Checkpoint trigger failed",
                error=str(exc),
                tool_name=tool_name,
            )

        try:
            self._post_hook.log_capture_audit(
                tool_name=tool_name,
                tool_response=tool_result,
                content_type=content_type,
                topics=topics,
                entities=entities,
                memory_id=memory_id,
                capture_decision="stored" if memory_id else "skipped",
            )
        except Exception:  # pragma: no cover - audit opzionale
            pass

        return memory_id

    def _load_hook(self, *, relative_path: tuple[str, ...], class_name: str) -> Any:
        """Utility to load hook classes from the Claude hook directory."""
        repo_root = Path(__file__).resolve().parents[4]
        hook_path = repo_root / ".claude" / "hooks" / "devstream"
        for part in relative_path:
            hook_path /= part

        if not hook_path.exists():
            raise FileNotFoundError(f"Hook file not found: {hook_path}")

        spec = spec_from_file_location(f"devstream_codex.{class_name.lower()}", hook_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load hook module from {hook_path}")

        module = module_from_spec(spec)
        spec.loader.exec_module(module)

        hook_cls = getattr(module, class_name, None)
        if hook_cls is None:
            raise AttributeError(f"{class_name} not found in {hook_path}")

        return hook_cls()


__all__ = ["ContextPipelineAdapter"]
