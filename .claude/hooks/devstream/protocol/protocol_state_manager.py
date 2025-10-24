#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "aiofiles>=23.0.0",
#     "structlog>=23.0.0",
#     "python-dotenv>=1.0.0",
#     "cryptography>=41.0.0",
#     "cachetools>=5.0.0",
# ]
# ///

"""
Protocol State Manager - DevStream 7-Step Protocol Enforcement

Phase 1 Component: Atomic state persistence with crash recovery.
Ensures task creation happens at STEP 1 (before DISCUSSION) with blocking validation.

Architecture Principles:
- Atomic file operations using write-rename pattern (from atomic_file_writer.py)
- Session ID generation with cryptographic randomness for uniqueness
- State validation with checksums for corruption detection
- Integration patterns compatible with existing hook system
- Restate SDK-inspired durability patterns

State Model:
{
    "session_id": "uuid4",  # Unique session identifier
    "protocol_step": int,   # Current step (1-7, 0=idle)
    "task_id": Optional[str],  # DevStream task ID if created
    "start_time": str,      # ISO 8601 timestamp
    "last_updated": str,    # ISO 8601 timestamp
    "checksum": str,        # SHA256 for integrity validation
    "metadata": dict        # Step-specific data and decisions
}

Integration Points:
- PreToolUse hook: Step validation and enforcement gate
- PostToolUse hook: State synchronization and progress tracking
- MCP tools: Task creation and memory storage
- DevStream memory: Decision logging and audit trail
"""

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from cryptography.fernet import Fernet
import cachetools

# Import existing utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from atomic_file_writer import write_atomic_json, write_atomic
import structlog

logger = structlog.get_logger(__name__)


class ProtocolStep(Enum):
    """DevStream 7-Step Protocol enumeration."""

    IDLE = 0
    DISCUSSION = 1
    ANALYSIS = 2
    RESEARCH = 3
    PLANNING = 4
    APPROVAL = 5
    IMPLEMENTATION = 6
    VERIFICATION = 7

    @classmethod
    def from_int(cls, value: int) -> 'ProtocolStep':
        """Convert integer to ProtocolStep with validation."""
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"Invalid protocol step: {value}. Must be 0-7.")

    def __str__(self) -> str:
        """Human-readable step name."""
        return {
            ProtocolStep.IDLE: "Idle",
            ProtocolStep.DISCUSSION: "Step 1: DISCUSSION",
            ProtocolStep.ANALYSIS: "Step 2: ANALYSIS",
            ProtocolStep.RESEARCH: "Step 3: RESEARCH",
            ProtocolStep.PLANNING: "Step 4: PLANNING",
            ProtocolStep.APPROVAL: "Step 5: APPROVAL",
            ProtocolStep.IMPLEMENTATION: "Step 6: IMPLEMENTATION",
            ProtocolStep.VERIFICATION: "Step 7: VERIFICATION",
        }[self]

    def is_valid_next_step(self, next_step: 'ProtocolStep') -> bool:
        """Check if transition to next_step is valid."""
        if self == ProtocolStep.IDLE:
            return next_step == ProtocolStep.DISCUSSION
        return next_step.value == self.value + 1


@dataclass
class ProtocolState:
    """Immutable protocol state data structure."""

    session_id: str
    protocol_step: ProtocolStep
    task_id: Optional[str] = None
    start_time: Optional[str] = None
    last_updated: Optional[str] = None
    checksum: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize timestamps and validate state."""
        if self.start_time is None:
            self.start_time = datetime.now(timezone.utc).isoformat()

        if self.last_updated is None:
            self.last_updated = self.start_time

        if self.metadata is None:
            self.metadata = {}

        # Validate UUID format
        try:
            uuid.UUID(self.session_id)
        except ValueError:
            raise ValueError(f"Invalid session_id format: {self.session_id}")

        # Calculate checksum if not provided
        if self.checksum is None:
            self.checksum = self.calculate_checksum()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['protocol_step'] = self.protocol_step.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProtocolState':
        """Create from dictionary with validation."""
        # Convert protocol_step from int to enum
        if 'protocol_step' in data:
            data['protocol_step'] = ProtocolStep.from_int(data['protocol_step'])

        return cls(**data)

    def calculate_checksum(self) -> str:
        """Calculate SHA256 checksum for state validation."""
        # Create deterministic representation
        data_for_checksum = {
            'session_id': self.session_id,
            'protocol_step': self.protocol_step.value,
            'task_id': self.task_id,
            'start_time': self.start_time,
            'last_updated': self.last_updated,
            'metadata': self.metadata or {}
        }

        # Sort keys for deterministic ordering
        json_str = json.dumps(data_for_checksum, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode()).hexdigest()

    def is_valid(self) -> bool:
        """Validate state integrity with checksum."""
        if not self.checksum:
            return False

        calculated_checksum = self.calculate_checksum()
        return calculated_checksum == self.checksum

    def with_updates(
        self,
        protocol_step: Optional[ProtocolStep] = None,
        task_id: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None
    ) -> 'ProtocolState':
        """Create new state with updates (immutable pattern)."""
        new_metadata = dict(self.metadata) if self.metadata else {}
        if metadata_updates:
            new_metadata.update(metadata_updates)

        return ProtocolState(
            session_id=self.session_id,
            protocol_step=protocol_step or self.protocol_step,
            task_id=task_id or self.task_id,
            start_time=self.start_time,
            last_updated=datetime.now(timezone.utc).isoformat(),
            checksum=None,  # Will be calculated in __post_init__
            metadata=new_metadata
        )


class ProtocolStateManager:
    """
    Manages DevStream protocol state with atomic persistence and crash recovery.

    Features:
    - Atomic state persistence using write-rename pattern
    - Session ID generation with crash recovery
    - State validation with checksums
    - In-memory caching with TTL for performance
    - Graceful degradation on persistence failures
    """

    def __init__(self, state_file: Optional[Path] = None):
        """
        Initialize protocol state manager.

        Args:
            state_file: Path to state file (default: .claude/state/protocol_state.json)
        """
        self.state_file = state_file or Path(".claude/state/protocol_state.json")
        self.lock_file = self.state_file.with_suffix('.lock')

        # In-memory cache with TTL (5 minutes)
        self._cache = cachetools.TTLCache(maxsize=1, ttl=300)
        self._cache_key = "current_state"

        # Encryption key for sensitive data (optional)
        self._encryption_key = None

        logger.info(
            "protocol_state_manager_initialized",
            state_file=str(self.state_file)
        )

    async def initialize_session(self) -> ProtocolState:
        """
        Initialize new session or recover existing session.

        Returns:
            Current protocol state (new or recovered)

        Raises:
            OSError: If state file operations fail
        """
        try:
            # Try to recover existing state
            existing_state = await self._load_state()
            if existing_state and existing_state.is_valid():
                logger.info(
                    "session_recovered",
                    session_id=existing_state.session_id,
                    current_step=str(existing_state.protocol_step)
                )
                return existing_state

            # Create new session
            session_id = str(uuid.uuid4())
            new_state = ProtocolState(
                session_id=session_id,
                protocol_step=ProtocolStep.IDLE
            )

            # Persist new state
            await self._save_state(new_state)

            logger.info(
                "session_initialized",
                session_id=session_id,
                step=str(new_state.protocol_step)
            )

            return new_state

        except Exception as e:
            logger.error(
                "session_initialization_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            # Return minimal valid state as fallback
            return ProtocolState(
                session_id=str(uuid.uuid4()),
                protocol_step=ProtocolStep.IDLE
            )

    async def get_current_state(self) -> ProtocolState:
        """
        Get current protocol state with caching.

        Returns:
            Current protocol state
        """
        # Check cache first
        if self._cache_key in self._cache:
            cached_state = self._cache[self._cache_key]
            if cached_state.is_valid():
                logger.debug("state_returned_from_cache")
                return cached_state

        # Load from disk
        state = await self._load_state()
        if not state or not state.is_valid():
            logger.warning("invalid_state_detected_reinitializing")
            state = await self.initialize_session()

        # Update cache
        self._cache[self._cache_key] = state

        return state

    async def advance_step(
        self,
        current_state: ProtocolState,
        next_step: ProtocolStep,
        task_id: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> ProtocolState:
        """
        Advance protocol step with validation.

        Args:
            current_state: Current protocol state
            next_step: Next step to advance to
            task_id: Optional DevStream task ID
            metadata_updates: Optional metadata to update
            force: Force transition even if invalid (for recovery)

        Returns:
            Updated protocol state

        Raises:
            ValueError: If step transition is invalid
            OSError: If state persistence fails
        """
        # Validate step transition
        if not force and not current_state.protocol_step.is_valid_next_step(next_step):
            raise ValueError(
                f"Invalid step transition: {current_state.protocol_step} → {next_step}. "
                f"Steps must be followed sequentially."
            )

        # Create updated state
        updated_state = current_state.with_updates(
            protocol_step=next_step,
            task_id=task_id,
            metadata_updates=metadata_updates
        )

        # Persist state
        await self._save_state(updated_state)

        # Update cache
        self._cache[self._cache_key] = updated_state

        logger.info(
            "protocol_step_advanced",
            session_id=updated_state.session_id,
            from_step=str(current_state.protocol_step),
            to_step=str(next_step),
            task_id=task_id
        )

        return updated_state

    async def reset_session(self, session_id: Optional[str] = None) -> ProtocolState:
        """
        Reset protocol session to idle state.

        Args:
            session_id: Optional session ID to reuse (generates new if None)

        Returns:
            New protocol state in IDLE
        """
        new_session_id = session_id or str(uuid.uuid4())
        new_state = ProtocolState(
            session_id=new_session_id,
            protocol_step=ProtocolStep.IDLE
        )

        await self._save_state(new_state)
        self._cache[self._cache_key] = new_state

        logger.info(
            "session_reset",
            session_id=new_session_id,
            previous_session_id=session_id
        )

        return new_state

    async def _load_state(self) -> Optional[ProtocolState]:
        """
        Load state from disk with validation.

        Returns:
            Protocol state if valid, None otherwise
        """
        try:
            if not self.state_file.exists():
                return None

            # Read file contents
            content = self.state_file.read_text(encoding='utf-8')
            if not content.strip():
                return None

            # Parse JSON
            data = json.loads(content)

            # Create state object
            state = ProtocolState.from_dict(data)

            # Validate integrity
            if not state.is_valid():
                logger.warning(
                    "invalid_state_checksum",
                    session_id=state.session_id,
                    stored_checksum=state.checksum,
                    calculated_checksum=state.calculate_checksum()
                )
                return None

            logger.debug(
                "state_loaded_successfully",
                session_id=state.session_id,
                step=str(state.protocol_step)
            )

            return state

        except json.JSONDecodeError as e:
            logger.error(
                "state_json_decode_error",
                file_path=str(self.state_file),
                error=str(e)
            )
            return None
        except Exception as e:
            logger.error(
                "state_load_error",
                file_path=str(self.state_file),
                error=str(e),
                error_type=type(e).__name__
            )
            return None

    async def _save_state(self, state: ProtocolState) -> bool:
        """
        Save state to disk with atomic operations.

        Args:
            state: Protocol state to save

        Returns:
            True if save succeeded, False otherwise
        """
        try:
            # Calculate checksum before saving
            calculated_checksum = state.calculate_checksum()
            state_data = state.to_dict()
            state_data['checksum'] = calculated_checksum

            # Atomic write using existing utility
            success = await write_atomic_json(self.state_file, state_data)

            if success:
                logger.debug(
                    "state_saved_successfully",
                    session_id=state.session_id,
                    step=str(state.protocol_step),
                    checksum=calculated_checksum[:16] + "..."  # Log partial checksum
                )
            else:
                logger.error("state_save_failed")

            return success

        except Exception as e:
            logger.error(
                "state_save_error",
                session_id=state.session_id,
                error=str(e),
                error_type=type(e).__name__
            )
            return False

    def get_session_status(self, state: Optional[ProtocolState] = None) -> Dict[str, Any]:
        """
        Get human-readable session status.

        Args:
            state: Protocol state (loads current if None)

        Returns:
            Dictionary with session status information
        """
        if state is None:
            # Return status without loading (for quick checks)
            return {
                "status": "unknown",
                "message": "Session state not loaded"
            }

        status = {
            "session_id": state.session_id,
            "current_step": str(state.protocol_step),
            "step_number": state.protocol_step.value,
            "task_id": state.task_id,
            "start_time": state.start_time,
            "last_updated": state.last_updated,
            "is_valid": state.is_valid(),
            "metadata_keys": list(state.metadata.keys()) if state.metadata else []
        }

        # Add duration if timestamps available
        if state.start_time and state.last_updated:
            try:
                start = datetime.fromisoformat(state.start_time.replace('Z', '+00:00'))
                updated = datetime.fromisoformat(state.last_updated.replace('Z', '+00:00'))
                duration_seconds = (updated - start).total_seconds()
                status["duration_seconds"] = duration_seconds
                status["duration_formatted"] = f"{duration_seconds:.1f}s"
            except Exception:
                pass  # Timestamp parsing errors are not critical

        return status


# Global instance for module-level usage
_global_manager: Optional[ProtocolStateManager] = None


def get_protocol_manager() -> ProtocolStateManager:
    """Get global protocol manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = ProtocolStateManager()
    return _global_manager


async def initialize_protocol_session() -> ProtocolState:
    """
    Initialize protocol session (convenience function).

    Returns:
        Initial protocol state
    """
    manager = get_protocol_manager()
    return await manager.initialize_session()


# Example usage and testing
if __name__ == "__main__":
    async def test_protocol_manager():
        """Test protocol state manager functionality."""

        # Test 1: Basic state management
        manager = ProtocolStateManager(Path("/tmp/test_protocol_state.json"))

        print("🧪 Test 1: Session initialization")
        state = await manager.initialize_session()
        assert state.protocol_step == ProtocolStep.IDLE
        assert state.session_id is not None
        print(f"✅ Session initialized: {state.session_id}")

        # Test 2: State persistence and recovery
        print("\n🧪 Test 2: State persistence")
        recovered_state = await manager.get_current_state()
        assert recovered_state.session_id == state.session_id
        print("✅ State persistence and recovery working")

        # Test 3: Step advancement
        print("\n🧪 Test 3: Step advancement")
        discussion_state = await manager.advance_step(
            state, ProtocolStep.DISCUSSION
        )
        assert discussion_state.protocol_step == ProtocolStep.DISCUSSION
        print(f"✅ Advanced to: {discussion_state.protocol_step}")

        # Test 4: Invalid transition handling
        print("\n🧪 Test 4: Invalid transition handling")
        try:
            await manager.advance_step(
                discussion_state, ProtocolStep.IMPLEMENTATION  # Skip steps
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"✅ Correctly blocked invalid transition: {e}")

        # Test 5: State validation
        print("\n🧪 Test 5: State validation")
        status = manager.get_session_status(discussion_state)
        assert status["is_valid"] == True
        assert status["current_step"] == "Step 1: DISCUSSION"
        print(f"✅ Status: {status['current_step']} (valid: {status['is_valid']})")

        # Test 6: Session reset
        print("\n🧪 Test 6: Session reset")
        reset_state = await manager.reset_session()
        assert reset_state.protocol_step == ProtocolStep.IDLE
        assert reset_state.session_id != discussion_state.session_id
        print(f"✅ Session reset: {reset_state.session_id}")

        # Cleanup
        if manager.state_file.exists():
            manager.state_file.unlink()

        print("\n🎉 All protocol state manager tests PASSED")

    # Run tests
    asyncio.run(test_protocol_manager())