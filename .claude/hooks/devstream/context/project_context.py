#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "cchooks>=0.1.4",
#     "aiohttp>=3.8.0",
#     "structlog>=23.0.0",
#     "python-dotenv>=1.0.0",
# ]
# ///

"""
DevStream SessionStart Hook - Project Context Initialization
Lightweight project context injection at session start.
"""

import sys
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from cchooks import safe_create_context, SessionStartContext
from devstream_base import DevStreamHookBase


class SessionStartHook:
    """
    SessionStart hook for lightweight project context injection.
    Provides basic project overview at session initialization.
    """

    def __init__(self):
        self.base = DevStreamHookBase("session_start")

    def get_project_info(self) -> Dict[str, Any]:
        """
        Get basic project information.

        Returns:
            Project info dict
        """
        project_root = Path.cwd()

        # Check if this is a DevStream project
        is_devstream = (
            (project_root / ".claude" / "hooks" / "devstream").exists() and
            (project_root / "data" / "devstream.db").exists()
        )

        if not is_devstream:
            return {"is_devstream_project": False}

        return {
            "is_devstream_project": True,
            "project_name": project_root.name,
            "methodology": "Research-Driven Development",
            "features": [
                "Task Management & Tracking",
                "Semantic Memory Storage",
                "Context7 Library Integration",
                "Hook System with cchooks"
            ]
        }

    def format_project_context(self, project_info: Dict[str, Any]) -> str:
        """
        Format project context for display.

        Args:
            project_info: Project information

        Returns:
            Formatted context string
        """
        if not project_info.get("is_devstream_project"):
            return ""

        context_parts = [
            "# 📁 DevStream Project",
            "",
            f"**Project**: {project_info['project_name']}",
            f"**Methodology**: {project_info['methodology']}",
            "",
            "**Key Features**:",
        ]

        for feature in project_info.get("features", []):
            context_parts.append(f"- {feature}")

        context_parts.extend([
            "",
            "💡 *All hooks using cchooks with graceful fallback strategy*",
            ""
        ])

        return "\n".join(context_parts)

    def get_previous_session_summary(self) -> Optional[str]:
        """
        Read previous session summary if available.

        Returns:
            Previous session summary or None
        """
        summary_file = Path.home() / ".claude" / "state" / "devstream_last_session.txt"

        if not summary_file.exists():
            return None

        try:
            summary = summary_file.read_text()
            # Delete file after reading (one-time display)
            summary_file.unlink()
            return summary
        except Exception as e:
            self.base.debug_log(f"Failed to read session summary: {e}")
            return None

    async def process(self, context: SessionStartContext) -> None:
        """
        Main hook processing logic.

        Args:
            context: SessionStart context from cchooks
        """
        # Check if hook should run
        if not self.base.should_run():
            self.base.debug_log("Hook disabled via config")
            context.output.exit_success()
            return

        self.base.debug_log("Session start - initializing project context")

        try:
            # Check for previous session summary FIRST
            prev_summary = self.get_previous_session_summary()

            # Get project information
            project_info = self.get_project_info()

            # Format context
            project_context = self.format_project_context(project_info)

            # If we have a previous session summary, prepend it to the project context
            if prev_summary:
                summary_section = (
                    "\n" + "=" * 60 + "\n"
                    "📋 Previous Session Summary\n"
                    + "=" * 60 + "\n"
                    + prev_summary + "\n"
                    + "=" * 60 + "\n\n"
                )
                project_context = summary_section + project_context

            if project_context:
                # Inject combined context (summary + project info)
                self.base.inject_context(project_context)
                self.base.success_feedback("Project context loaded")
            else:
                self.base.debug_log("Not a DevStream project")

            # Always allow the operation to proceed
            context.output.exit_success()

        except Exception as e:
            # Non-blocking error - log and continue
            self.base.warning_feedback(f"Project context failed: {str(e)[:50]}")
            context.output.exit_success()


def main():
    """Main entry point for SessionStart hook."""
    # Create context using cchooks
    ctx = safe_create_context()

    # Verify it's SessionStart context
    if not isinstance(ctx, SessionStartContext):
        print(f"Error: Expected SessionStartContext, got {type(ctx)}", file=sys.stderr)
        sys.exit(1)

    # Create and run hook
    hook = SessionStartHook()

    try:
        # Run async processing
        asyncio.run(hook.process(ctx))
    except Exception as e:
        # Graceful failure - non-blocking
        print(f"⚠️  DevStream: SessionStart error", file=sys.stderr)
        ctx.output.exit_non_block(f"Hook error: {str(e)[:100]}")


if __name__ == "__main__":
    main()