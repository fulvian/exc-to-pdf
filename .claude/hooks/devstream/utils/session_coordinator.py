#!/usr/bin/env python3
"""
DevStream Session Coordinator - Multi-Session Coordination

Prevents kernel panics by coordinating multiple Claude Code sessions accessing
the same database. Uses PID tracking, heartbeat mechanism, and file locking.

Key Features:
- PID tracking for active sessions
- Atomic file operations for session registry
- Heartbeat mechanism for stale session detection
- File locking with fcntl for critical operations
- Session health monitoring and cleanup
- Configurable session limits

Context7 Research:
- fcntl.flock(): POSIX file locking (exclusive/shared locks)
- psutil: Cross-platform process utilities for PID validation
- Pattern: Lock file + JSON registry for session coordination
"""

import os
import sys
import json
import fcntl
import time
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()

# Import DevStream utilities
sys.path.append(str(Path(__file__).parent))
from path_validator import validate_db_path, PathValidationError


@dataclass
class SessionInfo:
    """
    Session information for tracking (Enhanced for multi-session persistence).

    Attributes:
        session_id: Unique session identifier
        pid: Process ID
        started_at: Session start timestamp
        last_heartbeat: Last heartbeat timestamp
        status: Session status (active, compacted, ended, zombie)
        db_path: Database path for this session
        ended_at: Session end timestamp (None if active)
        marker_file_path: Path to session-specific marker file
        compaction_events: List of compaction events
        summary_displayed: Whether session summary has been displayed
        model_type: AI model type (sonnet-4.5, glm-4.6, unknown)
        session_name: Optional user-friendly session name
    """
    session_id: str
    pid: int
    started_at: float
    last_heartbeat: float
    status: str = "active"
    db_path: Optional[str] = None
    # New fields for multi-session persistence (Phase 1)
    ended_at: Optional[float] = None
    marker_file_path: Optional[str] = None
    compaction_events: List[Dict] = None
    summary_displayed: bool = False
    model_type: str = "unknown"
    session_name: Optional[str] = None

    def __post_init__(self):
        """Initialize mutable default values."""
        if self.compaction_events is None:
            self.compaction_events = []

    def is_stale(self, timeout_seconds: int = 300) -> bool:
        """
        Check if session is stale (no heartbeat for timeout_seconds).

        Args:
            timeout_seconds: Heartbeat timeout (default: 5 minutes)

        Returns:
            True if session is stale
        """
        return (time.time() - self.last_heartbeat) > timeout_seconds

    def is_zombie(self) -> bool:
        """
        Check if session is zombie (process no longer exists).

        Returns:
            True if PID doesn't exist
        """
        try:
            # psutil.pid_exists() checks if PID is valid
            return not psutil.pid_exists(self.pid)
        except Exception:
            return True

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'SessionInfo':
        """Create SessionInfo from dictionary."""
        return cls(**data)


class SessionCoordinator:
    """
    Coordinates multiple DevStream sessions to prevent database conflicts.

    Thread-safe session coordination using:
    - File locking (fcntl) for atomic operations
    - PID tracking with psutil for process validation
    - Heartbeat mechanism for stale session detection
    - Automatic cleanup of zombie/stale sessions

    Usage:
        >>> coordinator = SessionCoordinator.get_instance()
        >>> coordinator.register_session("sess-123")
        >>> # ... do work ...
        >>> coordinator.update_heartbeat("sess-123")
        >>> coordinator.unregister_session("sess-123")
    """

    _instance: Optional['SessionCoordinator'] = None
    _lock_file_handle = None
    _initializing: bool = False

    # Configuration (loaded from .env.devstream)
    MAX_SESSIONS = int(os.getenv('DEVSTREAM_MAX_SESSIONS', '5'))
    HEARTBEAT_TIMEOUT = int(os.getenv('DEVSTREAM_SESSION_HEARTBEAT_TIMEOUT', '300'))
    CLEANUP_INTERVAL = int(os.getenv('DEVSTREAM_SESSION_CLEANUP_INTERVAL', '60'))
    LIMIT_BEHAVIOR = os.getenv('DEVSTREAM_SESSION_LIMIT_BEHAVIOR', 'block')

    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize session coordinator.

        Args:
            registry_path: Path to session registry file
                         (default: ~/.claude/state/session_registry.json)
        """
        # Prevent direct instantiation (use get_instance())
        if not SessionCoordinator._initializing and SessionCoordinator._instance is not None:
            raise RuntimeError("Use SessionCoordinator.get_instance() instead")

        # Session registry path
        if registry_path is None:
            state_dir = Path.home() / '.claude' / 'state'
            state_dir.mkdir(parents=True, exist_ok=True)
            self.registry_path = str(state_dir / 'session_registry.json')
        else:
            self.registry_path = registry_path

        # Lock file for atomic operations
        self.lock_path = self.registry_path + '.lock'

        # In-memory session cache
        self._sessions_cache: Dict[str, SessionInfo] = {}

        # Last cleanup timestamp
        self._last_cleanup = time.time()

        # Thread lock for intra-process synchronization
        # (fcntl.flock only protects inter-process, not inter-thread)
        self._thread_lock = threading.RLock()

        # Logger
        self.logger = logging.getLogger('devstream.session_coordinator')

        # Initialize registry file
        self._init_registry()

        self.logger.info(f"SessionCoordinator initialized: {self.registry_path}")

    @classmethod
    def get_instance(cls, registry_path: Optional[str] = None) -> 'SessionCoordinator':
        """
        Get singleton instance of SessionCoordinator.

        Args:
            registry_path: Registry file path (only used for first init)

        Returns:
            Singleton SessionCoordinator instance
        """
        if cls._instance is None:
            cls._initializing = True
            cls._instance = cls.__new__(cls)
            cls._instance.__init__(registry_path)
            cls._initializing = False
        return cls._instance

    def _acquire_lock(self, timeout: int = 10) -> bool:
        """
        Acquire exclusive lock on registry file.

        Uses threading.RLock() for intra-process (thread) synchronization
        and fcntl.flock() for inter-process synchronization.

        Args:
            timeout: Lock acquisition timeout in seconds

        Returns:
            True if lock acquired, False on timeout

        Raises:
            IOError: If lock file cannot be opened
        """
        # First acquire thread lock (intra-process synchronization)
        if not self._thread_lock.acquire(timeout=timeout):
            self.logger.warning(f"Thread lock acquisition timeout after {timeout}s")
            return False

        start_time = time.time()
        remaining_timeout = timeout

        try:
            # Open lock file (create if doesn't exist)
            if self._lock_file_handle is None or self._lock_file_handle.closed:
                self._lock_file_handle = open(self.lock_path, 'w')

            # Try to acquire file lock with remaining timeout (inter-process synchronization)
            while (time.time() - start_time) < remaining_timeout:
                try:
                    # Non-blocking exclusive lock
                    fcntl.flock(self._lock_file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.logger.debug("Acquired file lock")
                    return True
                except IOError:
                    # Lock held by another process, wait and retry
                    time.sleep(0.1)

            self.logger.warning(f"File lock acquisition timeout after {remaining_timeout}s")
            self._thread_lock.release()  # Release thread lock if file lock failed
            return False
        except Exception as e:
            self.logger.error(f"Error acquiring lock: {e}")
            self._thread_lock.release()  # Release thread lock on error
            raise

    def _release_lock(self) -> None:
        """Release file lock and thread lock."""
        try:
            # First release file lock (inter-process)
            if self._lock_file_handle and not self._lock_file_handle.closed:
                try:
                    fcntl.flock(self._lock_file_handle.fileno(), fcntl.LOCK_UN)
                    self.logger.debug("Released file lock")
                except Exception as e:
                    self.logger.warning(f"Error releasing file lock: {e}")
        finally:
            # Always release thread lock (intra-process)
            try:
                self._thread_lock.release()
                self.logger.debug("Released thread lock")
            except RuntimeError:
                # RLock not held by current thread - can happen in edge cases
                pass

    def _init_registry(self) -> None:
        """
        Initialize session registry file if not exists.

        Also performs automatic schema migration for existing registries.
        """
        if not os.path.exists(self.registry_path):
            # Create empty registry
            try:
                if self._acquire_lock():
                    try:
                        with open(self.registry_path, 'w') as f:
                            json.dump({}, f)
                        self.logger.info("Created session registry")
                    finally:
                        self._release_lock()
            except Exception as e:
                self.logger.error(f"Failed to initialize registry: {e}")
        else:
            # Registry exists - perform automatic schema migration
            try:
                self.migrate_registry_schema()
                self.logger.debug("Registry schema migration check completed")
            except Exception as e:
                self.logger.warning(f"Schema migration failed: {e}")

    def _read_registry(self) -> Dict[str, SessionInfo]:
        """
        Read session registry from file.

        Returns:
            Dictionary of session_id -> SessionInfo

        Note:
            Must be called with lock acquired
        """
        try:
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                return {
                    sid: SessionInfo.from_dict(info)
                    for sid, info in data.items()
                }
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"Error reading registry: {e}")
            return {}

    def _write_registry(self, sessions: Dict[str, SessionInfo]) -> None:
        """
        Write session registry to file.

        Args:
            sessions: Dictionary of session_id -> SessionInfo

        Note:
            Must be called with lock acquired
        """
        try:
            # Write to temp file first (atomic write pattern)
            temp_path = self.registry_path + '.tmp'
            with open(temp_path, 'w') as f:
                data = {sid: info.to_dict() for sid, info in sessions.items()}
                json.dump(data, f, indent=2)
                f.flush()  # Ensure data written to OS
                os.fsync(f.fileno())  # Ensure data written to disk

            # Atomic rename
            os.replace(temp_path, self.registry_path)
        except Exception as e:
            self.logger.error(f"Error writing registry: {e}")
            raise

    def _cleanup_stale_sessions(self) -> int:
        """
        Clean up stale and zombie sessions.

        Returns:
            Number of sessions cleaned up
        """
        # Rate limit cleanup operations
        if (time.time() - self._last_cleanup) < self.CLEANUP_INTERVAL:
            return 0

        cleaned = 0

        if not self._acquire_lock():
            return 0

        try:
            sessions = self._read_registry()

            for session_id, info in list(sessions.items()):
                should_remove = False
                reason = ""

                # Check if zombie (process doesn't exist)
                if info.is_zombie():
                    should_remove = True
                    reason = f"zombie (PID {info.pid} doesn't exist)"

                # Check if stale (no heartbeat)
                elif info.is_stale(self.HEARTBEAT_TIMEOUT):
                    should_remove = True
                    reason = f"stale (no heartbeat for {self.HEARTBEAT_TIMEOUT}s)"

                if should_remove:
                    self.logger.info(f"Cleaning up session {session_id}: {reason}")
                    del sessions[session_id]
                    cleaned += 1

            # Write updated registry
            if cleaned > 0:
                self._write_registry(sessions)
                self._sessions_cache = sessions

            self._last_cleanup = time.time()

        finally:
            self._release_lock()

        return cleaned

    def register_session(
        self,
        session_id: str,
        db_path: Optional[str] = None
    ) -> bool:
        """
        Register new session in coordinator.

        Args:
            session_id: Unique session identifier
            db_path: Database path for this session

        Returns:
            True if registration successful, False if limit exceeded

        Raises:
            RuntimeError: If lock acquisition fails
        """
        # Cleanup stale sessions first
        self._cleanup_stale_sessions()

        if not self._acquire_lock():
            raise RuntimeError("Failed to acquire lock for session registration")

        try:
            sessions = self._read_registry()

            # Check session limit
            active_count = len([s for s in sessions.values() if s.status == "active"])
            if active_count >= self.MAX_SESSIONS:
                # Handle limit behavior based on configuration
                if self.LIMIT_BEHAVIOR == 'block':
                    self.logger.warning(
                        f"Session limit reached (blocking): {active_count}/{self.MAX_SESSIONS}",
                        extra={"behavior": "block", "session_id": session_id}
                    )
                    return False
                elif self.LIMIT_BEHAVIOR == 'warn':
                    self.logger.warning(
                        f"Session limit reached (allowing with warning): {active_count}/{self.MAX_SESSIONS}",
                        extra={"behavior": "warn", "session_id": session_id}
                    )
                    # Continue registration despite limit
                elif self.LIMIT_BEHAVIOR == 'queue':
                    # FUTURE: Implement queuing mechanism
                    self.logger.warning(
                        f"Session limit reached (queuing not yet implemented): {active_count}/{self.MAX_SESSIONS}",
                        extra={"behavior": "queue", "session_id": session_id}
                    )
                    return False

            # Create session info
            current_time = time.time()
            session_info = SessionInfo(
                session_id=session_id,
                pid=os.getpid(),
                started_at=current_time,
                last_heartbeat=current_time,
                status="active",
                db_path=db_path
            )

            # Add to registry
            sessions[session_id] = session_info
            self._write_registry(sessions)

            # Update cache
            self._sessions_cache = sessions

            self.logger.info(
                f"Registered session {session_id} (PID {os.getpid()})",
                extra={"active_sessions": active_count + 1}
            )

            return True

        finally:
            self._release_lock()

    def unregister_session(self, session_id: str) -> bool:
        """
        Unregister session from coordinator.

        Args:
            session_id: Session identifier to unregister

        Returns:
            True if unregistration successful
        """
        if not self._acquire_lock():
            self.logger.error("Failed to acquire lock for session unregistration")
            return False

        try:
            sessions = self._read_registry()

            if session_id in sessions:
                del sessions[session_id]
                self._write_registry(sessions)
                self._sessions_cache = sessions

                self.logger.info(f"Unregistered session {session_id}")
                return True
            else:
                self.logger.warning(f"Session {session_id} not found in registry")
                return False

        finally:
            self._release_lock()

    def update_heartbeat(self, session_id: str) -> bool:
        """
        Update session heartbeat timestamp.

        Args:
            session_id: Session identifier

        Returns:
            True if heartbeat updated successfully
        """
        if not self._acquire_lock(timeout=5):
            # Non-critical operation, can fail
            return False

        try:
            sessions = self._read_registry()

            if session_id in sessions:
                sessions[session_id].last_heartbeat = time.time()
                self._write_registry(sessions)
                self._sessions_cache = sessions
                return True
            else:
                return False

        finally:
            self._release_lock()

    def get_active_sessions(self) -> List[SessionInfo]:
        """
        Get list of active sessions.

        Returns:
            List of active SessionInfo objects
        """
        # Use cache for read-only operations
        if not self._sessions_cache:
            # Cache miss, read from file
            if self._acquire_lock(timeout=2):
                try:
                    self._sessions_cache = self._read_registry()
                finally:
                    self._release_lock()

        return [
            info for info in self._sessions_cache.values()
            if info.status == "active" and not info.is_zombie()
        ]

    def get_session_count(self) -> int:
        """
        Get count of active sessions.

        Returns:
            Number of active sessions
        """
        return len(self.get_active_sessions())

    def is_session_limit_reached(self) -> bool:
        """
        Check if session limit is reached.

        Returns:
            True if at or above limit
        """
        return self.get_session_count() >= self.MAX_SESSIONS

    def get_stats(self) -> Dict:
        """
        Get session coordinator statistics.

        Returns:
            Dictionary with coordinator stats
        """
        active_sessions = self.get_active_sessions()

        return {
            "active_sessions": len(active_sessions),
            "max_sessions": self.MAX_SESSIONS,
            "session_utilization": len(active_sessions) / self.MAX_SESSIONS,
            "registry_path": self.registry_path,
            "heartbeat_timeout": self.HEARTBEAT_TIMEOUT,
            "cleanup_interval": self.CLEANUP_INTERVAL
        }

    def validate_registry_schema(self, sessions: Dict[str, SessionInfo]) -> bool:
        """
        Validate registry schema conforms to enhanced SessionInfo structure.

        Checks that all required fields are present and have correct types.

        Args:
            sessions: Dictionary of session_id -> SessionInfo

        Returns:
            True if valid, False otherwise
        """
        required_fields = {
            'session_id': str,
            'pid': int,
            'started_at': float,
            'last_heartbeat': float,
            'status': str,
        }

        optional_fields = {
            'db_path': (str, type(None)),
            'ended_at': (float, type(None)),
            'marker_file_path': (str, type(None)),
            'compaction_events': list,
            'summary_displayed': bool,
            'model_type': str,
            'session_name': (str, type(None)),
        }

        for session_id, info in sessions.items():
            info_dict = info.to_dict()

            # Check required fields
            for field_name, field_type in required_fields.items():
                if field_name not in info_dict:
                    self.logger.error(
                        f"Validation failed: missing required field '{field_name}' "
                        f"in session {session_id}"
                    )
                    return False

                if not isinstance(info_dict[field_name], field_type):
                    self.logger.error(
                        f"Validation failed: field '{field_name}' has wrong type "
                        f"(expected {field_type}, got {type(info_dict[field_name])}) "
                        f"in session {session_id}"
                    )
                    return False

            # Check optional fields (if present)
            for field_name, field_types in optional_fields.items():
                if field_name in info_dict:
                    if not isinstance(field_types, tuple):
                        field_types = (field_types,)

                    if not isinstance(info_dict[field_name], field_types):
                        self.logger.error(
                            f"Validation failed: field '{field_name}' has wrong type "
                            f"(expected {field_types}, got {type(info_dict[field_name])}) "
                            f"in session {session_id}"
                        )
                        return False

        self.logger.debug(f"Registry schema validation passed for {len(sessions)} sessions")
        return True

    def migrate_registry_schema(self) -> bool:
        """
        Migrate registry to enhanced schema (add missing fields with defaults).

        Adds new fields to existing sessions:
        - ended_at: None (active sessions)
        - marker_file_path: None
        - compaction_events: []
        - summary_displayed: False
        - model_type: "unknown"
        - session_name: None

        Returns:
            True if migration successful, False otherwise
        """
        if not self._acquire_lock():
            self.logger.error("Failed to acquire lock for schema migration")
            return False

        try:
            # Read raw registry data
            with open(self.registry_path, 'r') as f:
                data = json.load(f)

            migrated = False

            for session_id, info_dict in data.items():
                # Check if migration needed
                needs_migration = False

                # Add missing fields with defaults
                if 'ended_at' not in info_dict:
                    info_dict['ended_at'] = None
                    needs_migration = True

                if 'marker_file_path' not in info_dict:
                    info_dict['marker_file_path'] = None
                    needs_migration = True

                if 'compaction_events' not in info_dict:
                    info_dict['compaction_events'] = []
                    needs_migration = True

                if 'summary_displayed' not in info_dict:
                    info_dict['summary_displayed'] = False
                    needs_migration = True

                if 'model_type' not in info_dict:
                    info_dict['model_type'] = "unknown"
                    needs_migration = True

                if 'session_name' not in info_dict:
                    info_dict['session_name'] = None
                    needs_migration = True

                if needs_migration:
                    self.logger.info(f"Migrated session {session_id} to new schema")
                    migrated = True

            if migrated:
                # Write migrated registry (atomic write)
                temp_path = self.registry_path + '.tmp'
                with open(temp_path, 'w') as f:
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())

                os.replace(temp_path, self.registry_path)

                # Reload cache
                self._sessions_cache = self._read_registry()

                self.logger.info("Registry schema migration completed")
            else:
                self.logger.debug("Registry schema already up to date")

            return True

        except Exception as e:
            self.logger.error(f"Schema migration failed: {e}")
            return False

        finally:
            self._release_lock()


# Convenience function for getting coordinator instance
def get_session_coordinator(registry_path: Optional[str] = None) -> SessionCoordinator:
    """
    Get singleton SessionCoordinator instance.

    Args:
        registry_path: Registry file path (optional)

    Returns:
        SessionCoordinator singleton instance
    """
    return SessionCoordinator.get_instance(registry_path)


if __name__ == "__main__":
    # Test session coordinator
    print("DevStream Session Coordinator Test")
    print("=" * 50)

    # Get coordinator instance
    coordinator = SessionCoordinator.get_instance()
    print(f"✅ Coordinator initialized: {coordinator.registry_path}")

    # Register test session
    session_id = f"test-sess-{os.getpid()}"
    success = coordinator.register_session(session_id, "data/devstream.db")
    print(f"✅ Session registered: {success}")

    # Get active sessions
    active = coordinator.get_active_sessions()
    print(f"✅ Active sessions: {len(active)}")

    # Update heartbeat
    coordinator.update_heartbeat(session_id)
    print(f"✅ Heartbeat updated")

    # Get stats
    stats = coordinator.get_stats()
    print(f"✅ Session utilization: {stats['session_utilization']:.1%}")

    # Unregister session
    coordinator.unregister_session(session_id)
    print(f"✅ Session unregistered")

    print("\n🎉 Session Coordinator test completed!")
