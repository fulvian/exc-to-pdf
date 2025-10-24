"""Event schemas for Codex integration."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ConfigDict


class CodexEventType(str, Enum):
    """Enumerates upstream Codex events we care about."""

    SESSION_START = "session_start"
    USER_PROMPT_SUBMIT = "user_prompt_submit"
    TOOL_PRE_EXECUTE = "tool_pre_execute"
    TOOL_POST_EXECUTE = "tool_post_execute"
    SESSION_STOP = "session_stop"
    HEARTBEAT = "heartbeat"


class CodexEvent(BaseModel):
    """Normalized structure for Codex CLI events."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    event_type: CodexEventType = Field(..., description="Type of Codex event received.")
    session_id: str = Field(..., description="Unique Codex session identifier.")
    cwd: Optional[str] = Field(
        default=None,
        description="Workspace directory associated with the event.",
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw event payload supplied by Codex.",
    )

    @property
    def tool_name(self) -> Optional[str]:
        """Convenience accessor for tool-related events."""
        return self.payload.get("tool_name")

    @property
    def timestamp(self) -> Optional[str]:
        """Return timestamp string if provided by Codex."""
        return self.payload.get("timestamp")


def normalize_event(raw: Dict[str, Any]) -> CodexEvent:
    """Convert raw Codex telemetry into a validated event."""
    if "event_type" not in raw:
        raise ValueError("Codex event payload missing 'event_type'")

    return CodexEvent(
        event_type=CodexEventType(raw["event_type"]),
        session_id=raw.get("session_id", "unknown"),
        cwd=raw.get("cwd"),
        payload=raw,
    )


__all__ = ["CodexEvent", "CodexEventType", "normalize_event"]
