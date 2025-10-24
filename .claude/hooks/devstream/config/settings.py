#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pydantic>=2.0.0",
#     "jsonschema>=4.0.0",
# ]
# ///

"""
DevStream Claude Settings Configuration Generator
Context7-compliant settings.json generation per research patterns.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from pydantic import BaseModel, Field
import jsonschema

class HookCommand(BaseModel):
    """Single hook command configuration."""
    type: str = Field(default="command", description="Hook type (always 'command')")
    command: List[str] = Field(..., description="Command array to execute")
    timeout: int = Field(default=60, description="Timeout in seconds")

class HookGroup(BaseModel):
    """Group of hooks for a specific matcher."""
    matcher: str = Field(default="", description="Regex matcher for hook trigger")
    hooks: List[HookCommand] = Field(..., description="List of hook commands")

class HookConfiguration(BaseModel):
    """Complete hook configuration following Context7 patterns."""
    UserPromptSubmit: List[HookGroup] = Field(default_factory=list)
    PreToolUse: List[HookGroup] = Field(default_factory=list)
    PostToolUse: List[HookGroup] = Field(default_factory=list)
    SessionStart: List[HookGroup] = Field(default_factory=list)
    Stop: List[HookGroup] = Field(default_factory=list)
    Notification: List[HookGroup] = Field(default_factory=list)

class ClaudeSettings(BaseModel):
    """Complete Claude Code settings configuration."""
    hooks: HookConfiguration = Field(default_factory=HookConfiguration)

def generate_devstream_settings() -> Dict[str, Any]:
    """
    Generate DevStream-specific Claude settings following Context7 patterns.

    Returns:
        Complete settings dictionary
    """
    # DevStream hook commands base path
    hooks_base = ".claude/hooks/devstream"

    # Enhanced user context hooks
    enhanced_user_hooks = HookGroup(
        matcher="",  # Apply to all
        hooks=[
            HookCommand(
                command=["uv", "run", "--script", f"{hooks_base}/context/user_query_context_enhancer.py"]
            )
        ]
    )

    # Task management hooks
    task_hooks_start = HookGroup(
        matcher="",  # Apply to all
        hooks=[
            HookCommand(
                command=["uv", "run", "--script", f"{hooks_base}/tasks/session_start.py"]
            )
        ]
    )

    task_hooks_stop = HookGroup(
        matcher="",  # Apply to all
        hooks=[
            HookCommand(
                command=["uv", "run", "--script", f"{hooks_base}/tasks/stop.py"]
            )
        ]
    )

    # MCP DevStream tool hooks (Context7 pattern)
    mcp_devstream_hooks = HookGroup(
        matcher="mcp__devstream__.*",
        hooks=[
            HookCommand(
                command=["uv", "run", "--script", f"{hooks_base}/memory/pre_tool_use.py"]
            )
        ]
    )

    # Context injection hooks
    context_hooks = HookGroup(
        matcher="",  # Apply to all
        hooks=[
            HookCommand(
                command=["uv", "run", "--script", f"{hooks_base}/context/project_context.py"]
            )
        ]
    )

    # Build configuration
    hook_config = HookConfiguration(
        UserPromptSubmit=[enhanced_user_hooks],
        SessionStart=[task_hooks_start, context_hooks],
        Stop=[task_hooks_stop],
        PreToolUse=[mcp_devstream_hooks]
    )

    settings = ClaudeSettings(hooks=hook_config)

    return settings.model_dump(exclude_none=True)

def validate_settings(settings: Dict[str, Any]) -> bool:
    """
    Validate Claude settings against JSON schema.

    Args:
        settings: Settings dictionary to validate

    Returns:
        True if valid, False otherwise
    """
    # Basic schema for Claude Code hooks (Context7-derived)
    schema = {
        "type": "object",
        "properties": {
            "hooks": {
                "type": "object",
                "properties": {
                    "UserPromptSubmit": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "matcher": {"type": "string"},
                                "hooks": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "type": {"type": "string"},
                                            "command": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            },
                                            "timeout": {"type": "integer"}
                                        },
                                        "required": ["type", "command"]
                                    }
                                }
                            },
                            "required": ["hooks"]
                        }
                    },
                    # Similar structure for other hook types
                    "PreToolUse": {"type": "array"},
                    "PostToolUse": {"type": "array"},
                    "SessionStart": {"type": "array"},
                    "Stop": {"type": "array"},
                    "Notification": {"type": "array"}
                }
            }
        },
        "required": ["hooks"]
    }

    try:
        jsonschema.validate(settings, schema)
        return True
    except jsonschema.ValidationError as e:
        print(f"Settings validation failed: {e}", file=sys.stderr)
        return False

def write_settings_file(settings_path: Path) -> None:
    """
    Write DevStream settings to .claude/settings.json

    Args:
        settings_path: Path to settings.json file
    """
    # Generate settings
    settings = generate_devstream_settings()

    # Validate settings
    if not validate_settings(settings):
        print("Failed to generate valid settings", file=sys.stderr)
        sys.exit(1)

    # Ensure directory exists
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    # Write settings file
    with open(settings_path, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2)

    print(f"✅ DevStream Claude settings written to {settings_path}")
    print("📋 Settings include:")
    print("  - UserPromptSubmit: Enhanced context injection + memory storage")
    print("  - SessionStart: Task planning + project context injection")
    print("  - Stop: Task completion automation")
    print("  - PreToolUse: Intelligent context injection for tools")

def main():
    """Main settings generation function."""
    # Default settings path (local to project)
    default_path = Path.cwd() / ".claude" / "settings.json"

    # Get path from args or use default
    if len(sys.argv) > 1:
        settings_path = Path(sys.argv[1])
    else:
        settings_path = default_path

    # Generate and write settings
    write_settings_file(settings_path)

if __name__ == "__main__":
    main()