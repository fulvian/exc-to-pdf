"""Event dispatcher che collega gli adapter Codex."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .config import CodexSettings
from .context_pipeline import ContextPipelineAdapter
from .events import CodexEvent, CodexEventType, normalize_event
from .logging import get_codex_logger
from .protocol_gateway import ProtocolGatewayAdapter, ProtocolAssessment
from .session import SessionLifecycleAdapter


class CodexIntegrationRuntime:
    """Gestore runtime che instrada gli eventi Codex verso DevStream."""

    def __init__(
        self,
        settings: Optional[CodexSettings] = None,
        *,
        protocol_adapter: Optional[ProtocolGatewayAdapter] = None,
        context_adapter: Optional[ContextPipelineAdapter] = None,
        session_adapter: Optional[SessionLifecycleAdapter] = None,
        record_samples: bool = True,
    ) -> None:
        self.settings = settings or CodexSettings()
        self.logger = get_codex_logger("runtime")
        self.protocol_adapter = protocol_adapter or ProtocolGatewayAdapter(self.settings)
        self.context_adapter = context_adapter or ContextPipelineAdapter(self.settings)
        self.session_adapter = session_adapter or SessionLifecycleAdapter(self.settings)
        self._record_samples = record_samples
        self._payload_samples: Dict[CodexEventType, list[Dict[str, Any]]] = defaultdict(list)
        self._sample_path: Optional[Path] = self.settings.resolve_sample_path()

    async def handle_event(self, raw_event: Dict[str, Any] | CodexEvent) -> Dict[str, Any]:
        """Gestisce un evento Codex e restituisce l'esito dell'elaborazione."""
        event = raw_event if isinstance(raw_event, CodexEvent) else normalize_event(raw_event)
        self._record_payload_sample(event)

        if event.event_type == CodexEventType.SESSION_START:
            return await self._handle_session_start(event)
        if event.event_type == CodexEventType.USER_PROMPT_SUBMIT:
            return await self._handle_user_prompt(event)
        if event.event_type == CodexEventType.TOOL_PRE_EXECUTE:
            return await self._handle_tool_pre(event)
        if event.event_type == CodexEventType.TOOL_POST_EXECUTE:
            return await self._handle_tool_post(event)
        if event.event_type == CodexEventType.SESSION_STOP:
            return await self._handle_session_stop(event)

        self.logger.debug("Evento non gestito", event_type=event.event_type.value)
        return {"status": "ignored", "event_type": event.event_type.value}

    def get_payload_samples(self) -> Dict[str, list[Dict[str, Any]]]:
        """Restituisce un campione dei payload osservati."""
        return {etype.value: samples[:] for etype, samples in self._payload_samples.items()}

    async def _handle_session_start(self, event: CodexEvent) -> Dict[str, Any]:
        cwd = event.cwd or ""
        await self.session_adapter.on_session_start(event.session_id, cwd)
        return {"status": "session_started", "session_id": event.session_id}

    async def _handle_user_prompt(self, event: CodexEvent) -> Dict[str, Any]:
        user_input = event.payload.get("user_input", "")
        if not user_input:
            self.logger.warning("Prompt senza testo", session_id=event.session_id)
            return {"status": "skipped", "reason": "empty_input"}

        assessment: ProtocolAssessment = await self.protocol_adapter.evaluate_prompt(user_input)
        result = {
            "status": "protocol_enforced" if assessment.enforce_protocol else "protocol_optional",
            "enforce": assessment.enforce_protocol,
            "triggers": list(assessment.triggers),
        }
        if assessment.prompt:
            result["prompt"] = assessment.prompt
        return result

    async def _handle_tool_pre(self, event: CodexEvent) -> Dict[str, Any]:
        tool_input = event.payload.get("tool_input", {})
        file_path = tool_input.get("file_path")
        content = tool_input.get("content") or tool_input.get("new_string")

        if not file_path or not content:
            return {"status": "skipped", "reason": "missing_file_or_content"}

        context_text = await self.context_adapter.build_pre_context(file_path, content)
        return {
            "status": "context_ready" if context_text else "context_missing",
            "context": context_text,
        }

    async def _handle_tool_post(self, event: CodexEvent) -> Dict[str, Any]:
        tool_name = event.payload.get("tool_name", "")
        tool_input = event.payload.get("tool_input", {})
        tool_result = event.payload.get("tool_result", {})

        memory_id = await self.context_adapter.capture_post_tool(
            tool_name,
            tool_input,
            tool_result,
        )

        await self.session_adapter.on_tool_execution(
            event.session_id,
            tool_name,
            tool_input,
            tool_result,
        )

        return {
            "status": "captured" if memory_id else "skipped",
            "memory_id": memory_id,
        }

    async def _handle_session_stop(self, event: CodexEvent) -> Dict[str, Any]:
        summary = await self.session_adapter.on_session_stop(event.session_id)
        return {
            "status": "session_stopped",
            "session_id": event.session_id,
            "summary": summary,
        }

    def _record_payload_sample(self, event: CodexEvent) -> None:
        if not self._record_samples:
            return

        samples = self._payload_samples[event.event_type]
        if len(samples) < 5:
            samples.append(event.payload)

        if self._sample_path is None:
            return

        try:
            self._sample_path.parent.mkdir(parents=True, exist_ok=True)
            with self._sample_path.open("a", encoding="utf-8") as handle:
                json.dump(
                    {
                        "event_type": event.event_type.value,
                        "payload": event.payload,
                    },
                    handle,
                )
                handle.write("\n")
        except Exception as exc:  # pragma: no cover - best effort logging
            self.logger.warning(
                "Unable to persist Codex payload sample",
                error=str(exc),
            )


__all__ = ["CodexIntegrationRuntime"]
