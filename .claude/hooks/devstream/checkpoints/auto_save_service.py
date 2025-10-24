"""
Auto-Save Service - Background checkpoint creation.

Features:
- Periodic auto-save (configurable interval)
- Trigger on critical tools (Write, Edit)
- Graceful startup/shutdown
"""

import asyncio
from typing import Optional
from checkpoint_manager import CheckpointManager


class AutoSaveService:
    """
    Background service for automatic checkpoint creation.

    Usage:
        >>> service = AutoSaveService(interval_seconds=60)
        >>> # Service starts automatically
        >>> await service.stop()  # Graceful shutdown
    """

    def __init__(self, interval_seconds: int = 300):
        """
        Initialize auto-save service.

        Args:
            interval_seconds: Auto-save interval (default: 300s = 5 minutes)
        """
        self.interval = interval_seconds
        self.manager = CheckpointManager()
        self._task: Optional[asyncio.Task] = None
        self._running = False

        # Start service automatically
        self._start()

    def _start(self) -> None:
        """Start background auto-save task."""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._auto_save_loop())

    async def _auto_save_loop(self) -> None:
        """Background loop for periodic auto-save."""
        while self._running:
            try:
                await asyncio.sleep(self.interval)

                # Create auto-save checkpoint
                await self.manager.create_checkpoint(
                    checkpoint_type="auto",
                    description="Periodic auto-save",
                    context={"triggered_by": "interval"}
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue
                print(f"Auto-save error: {e}")

    async def stop(self) -> None:
        """Stop auto-save service gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def is_running(self) -> bool:
        """Check if service is running."""
        return self._running


def should_trigger_save(tool_name: str) -> bool:
    """
    Check if tool should trigger immediate checkpoint.

    Args:
        tool_name: Name of tool being executed

    Returns:
        True if tool is critical and should trigger save
    """
    critical_tools = ["Write", "Edit", "MultiEdit", "Delete"]
    return tool_name in critical_tools


# Global service instance (lazy initialization)
_service: Optional[AutoSaveService] = None


def get_auto_save_service() -> AutoSaveService:
    """Get global auto-save service instance."""
    global _service
    if _service is None:
        _service = AutoSaveService()
    return _service
