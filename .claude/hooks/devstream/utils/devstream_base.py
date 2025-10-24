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
DevStream Hook Base Class - Context7 & cchooks Compliant
Provides foundation for all DevStream hooks with graceful fallback.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, Optional
from enum import Enum

# Load environment config
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent.parent / '.env.devstream', override=False)


class FeedbackLevel(Enum):
    """User feedback verbosity levels."""
    SILENT = "silent"      # No user output
    MINIMAL = "minimal"    # Only warnings/errors
    VERBOSE = "verbose"    # Detailed output


class DevStreamHookBase:
    """
    Base class for all DevStream hooks using cchooks.
    Provides graceful fallback, error handling, and user feedback.
    """

    def __init__(self, hook_name: str):
        """
        Initialize DevStream hook base.

        Args:
            hook_name: Name of the hook for logging
        """
        self.hook_name = hook_name
        self.feedback_level = self._get_feedback_level()
        self.fallback_mode = os.getenv('DEVSTREAM_FALLBACK_MODE', 'graceful')
        self.debug = os.getenv('DEVSTREAM_DEBUG', 'false').lower() == 'true'

    def _get_feedback_level(self) -> FeedbackLevel:
        """Get configured feedback level."""
        level = os.getenv('DEVSTREAM_FEEDBACK_LEVEL', 'minimal')
        try:
            return FeedbackLevel(level)
        except ValueError:
            return FeedbackLevel.MINIMAL

    def user_feedback(self, message: str, level: FeedbackLevel = FeedbackLevel.MINIMAL):
        """
        Provide feedback to user based on configured verbosity.

        Args:
            message: Message to display
            level: Minimum feedback level required
        """
        if self.feedback_level.value == "silent":
            return

        if level == FeedbackLevel.MINIMAL or self.feedback_level == FeedbackLevel.VERBOSE:
            print(message, file=sys.stderr)

    def debug_log(self, message: str):
        """Log debug message if debug mode enabled."""
        if self.debug:
            print(f"🔍 DevStream [{self.hook_name}]: {message}", file=sys.stderr)

    def warning_feedback(self, message: str):
        """Show warning to user (non-invasive)."""
        self.user_feedback(f"⚠️  DevStream: {message}", FeedbackLevel.MINIMAL)

    def error_feedback(self, message: str):
        """Show error to user (clear but brief)."""
        self.user_feedback(f"❌ DevStream: {message}", FeedbackLevel.MINIMAL)

    def success_feedback(self, message: str):
        """Show success message (only in verbose mode)."""
        self.user_feedback(f"✅ DevStream: {message}", FeedbackLevel.VERBOSE)

    async def safe_mcp_call(
        self,
        mcp_client: Any,
        tool_name: str,
        params: Dict[str, Any],
        timeout: float = 5.0
    ) -> Optional[Dict[str, Any]]:
        """
        Call MCP tool with graceful fallback and timeout.

        Args:
            mcp_client: MCP client instance
            tool_name: MCP tool name
            params: Tool parameters
            timeout: Timeout in seconds (default: 5.0)

        Returns:
            Tool result or None if failed
        """
        try:
            self.debug_log(f"Calling MCP tool: {tool_name} (timeout: {timeout}s)")
            result = await asyncio.wait_for(
                mcp_client.call_tool(tool_name, params),
                timeout=timeout
            )
            self.debug_log(f"MCP call succeeded: {tool_name}")
            return result
        except asyncio.TimeoutError:
            self.warning_feedback(f"MCP timeout after {timeout}s ({tool_name})")
            return None
        except ConnectionError:
            self.warning_feedback(f"MCP server unavailable ({tool_name})")
            return None
        except TimeoutError:
            self.warning_feedback(f"MCP timeout ({tool_name})")
            return None
        except Exception as e:
            self.debug_log(f"MCP error ({tool_name}): {e}")
            self.warning_feedback(f"MCP call failed ({tool_name})")
            return None

    def should_run(self) -> bool:
        """
        Check if hook should run based on configuration.

        Returns:
            True if hook should execute
        """
        # Global disable
        if os.getenv('DEVSTREAM_HOOKS_ENABLED', 'true').lower() != 'true':
            self.debug_log("DevStream hooks globally disabled")
            return False

        # Per-hook disable
        hook_env = f'DEVSTREAM_HOOK_{self.hook_name.upper().replace("-", "_")}'
        if os.getenv(hook_env, 'true').lower() != 'true':
            self.debug_log(f"Hook {self.hook_name} disabled via env")
            return False

        return True

    def is_memory_store_enabled(self) -> bool:
        """
        Check if memory storage is enabled.

        Returns:
            True if memory storage should be used
        """
        return os.getenv('DEVSTREAM_MEMORY_ENABLED', 'true').lower() == 'true'

    def inject_context(self, context_data: str) -> None:
        """
        Inject additional context into Claude session.
        Uses hookSpecificOutput for UserPromptSubmit.

        Args:
            context_data: Context string to inject
        """
        if not context_data or len(context_data.strip()) == 0:
            return

        # For UserPromptSubmit hooks, use hookSpecificOutput
        # For other hooks, direct stdout
        self.debug_log(f"Injecting context: {len(context_data)} chars")
        print(context_data)


# Export utilities for easy import
def get_project_root() -> Path:
    """Get DevStream project root directory."""
    return Path(__file__).parent.parent.parent.parent


def is_devstream_enabled() -> bool:
    """Quick check if DevStream hooks are enabled."""
    return os.getenv('DEVSTREAM_HOOKS_ENABLED', 'true').lower() == 'true'