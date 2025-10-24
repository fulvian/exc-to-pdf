"""
Slash command handlers for checkpoint system.

Commands:
- /save-progress [description] - Create manual checkpoint
"""

from typing import Optional, Dict, Any
from checkpoint_manager import CheckpointManager


async def handle_save_progress(description: Optional[str] = None) -> Dict[str, Any]:
    """
    Handle /save-progress slash command.

    Args:
        description: Optional checkpoint description

    Returns:
        Command execution result

    Example:
        >>> result = await handle_save_progress("Before refactoring")
        >>> print(result["checkpoint_id"])
    """
    manager = CheckpointManager()

    # Create checkpoint
    checkpoint = await manager.create_checkpoint(
        checkpoint_type="manual",
        description=description or "Manual checkpoint via /save-progress",
        context={"triggered_by": "slash_command"}
    )

    # Provide user feedback
    print(f"✅ Checkpoint created: {checkpoint['id']}")
    if description:
        print(f"   Description: {description}")

    return {
        "success": True,
        "checkpoint_id": checkpoint["id"],
        "type": "manual",
        "description": checkpoint["description"]
    }
