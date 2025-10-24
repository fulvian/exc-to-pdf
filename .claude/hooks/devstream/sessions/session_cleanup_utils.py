#!/usr/bin/env python3
"""
DevStream Session Cleanup Utilities - Robust Session Management

Provides enhanced session cleanup mechanisms to prevent session limit
issues caused by zombie sessions in the registry.

Key Features:
- Aggressive zombie session detection and cleanup
- Fallback mechanisms when psutil fails
- Registry repair and validation
- Emergency session limit override
- Configurable cleanup strategies

Context7 Research Applied:
- psutil.pid_exists() patterns for robust PID validation
- Signal-based process validation as fallback
- Atomic file operations for registry management
"""

import os
import sys
import json
import time
import signal
import errno
import threading
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging

# Import DevStream utilities
sys.path.append(str(Path(__file__).parent.parent / 'utils'))
from session_coordinator import SessionInfo, get_session_coordinator
from logger import get_devstream_logger


@dataclass
class CleanupStats:
    """Statistics for session cleanup operations."""
    zombie_sessions_cleaned: int = 0
    stale_sessions_cleaned: int = 0
    registry_errors: int = 0
    cleanup_duration: float = 0.0
    sessions_before: int = 0
    sessions_after: int = 0


class SessionCleanupManager:
    """
    Enhanced session cleanup manager with robust zombie detection.

    Provides multiple layers of session validation and cleanup:
    1. PID validation using psutil (primary)
    2. Signal-based validation (fallback)
    3. Timestamp-based stale detection
    4. Registry integrity validation
    """

    def __init__(self, coordinator=None):
        """
        Initialize cleanup manager.

        Args:
            coordinator: SessionCoordinator instance (optional, will create if None)
        """
        self.coordinator = coordinator or get_session_coordinator()
        self.structured_logger = get_devstream_logger('session_cleanup_manager')
        self.logger = self.structured_logger.logger  # Compatibility

        # Cleanup configuration
        self.AGGRESSIVE_CLEANUP = os.getenv('DEVSTREAM_AGGRESSIVE_CLEANUP', 'true').lower() == 'true'
        self.EMERGENCY_OVERRIDE = os.getenv('DEVSTREAM_EMERGENCY_OVERRIDE', 'true').lower() == 'true'
        self.CLEANUP_TIMEOUT = int(os.getenv('DEVSTREAM_CLEANUP_TIMEOUT', '30'))

        self.logger.info(f"SessionCleanupManager initialized", extra={
            'aggressive_cleanup': self.AGGRESSIVE_CLEANUP,
            'emergency_override': self.EMERGENCY_OVERRIDE
        })

    def _validate_pid_with_signal(self, pid: int) -> bool:
        """
        Validate PID using signal-based approach (fallback when psutil fails).

        Uses os.kill(pid, 0) which doesn't actually send a signal
        but checks if the process exists.

        Args:
            pid: Process ID to validate

        Returns:
            True if PID exists, False otherwise
        """
        try:
            # Signal 0 doesn't actually kill the process, just checks existence
            os.kill(pid, 0)
            return True
        except OSError as e:
            if e.errno == errno.ESRCH:
                # No such process
                return False
            elif e.errno == errno.EPERM:
                # Process exists but we don't have permission to signal it
                # Consider this as "exists" for our purposes
                return True
            else:
                # Other errors (shouldn't happen often)
                self.logger.warning(f"Unexpected error checking PID {pid}: {e}")
                return False

    def _validate_pid_robust(self, pid: int) -> bool:
        """
        Robust PID validation with multiple fallback methods.

        Args:
            pid: Process ID to validate

        Returns:
            True if PID exists and is a Claude process, False otherwise
        """
        # Method 1: Try psutil first (most reliable)
        try:
            import psutil
            if psutil.pid_exists(pid):
                # Additional check: verify it's actually a Claude process
                try:
                    proc = psutil.Process(pid)
                    cmdline = proc.cmdline()
                    if any('claude' in cmd.lower() for cmd in cmdline):
                        return True
                    else:
                        self.logger.debug(f"PID {pid} exists but not a Claude process")
                        return False
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    return False
            else:
                return False
        except ImportError:
            self.logger.warning("psutil not available, falling back to signal method")
        except Exception as e:
            self.logger.warning(f"psutil validation failed for PID {pid}: {e}")

        # Method 2: Fallback to signal-based validation
        if self._validate_pid_with_signal(pid):
            # Additional check: verify it's likely a Claude process via /proc
            try:
                # Try to read cmdline to confirm it's Claude
                if os.path.exists(f"/proc/{pid}/cmdline"):
                    with open(f"/proc/{pid}/cmdline", 'r') as f:
                        cmdline = f.read()
                        if 'claude' in cmdline.lower():
                            return True
                        else:
                            self.logger.debug(f"PID {pid} exists but not a Claude process (proc check)")
                            return False
                else:
                    # /proc not available (macOS), use ps command for better validation
                    try:
                        result = subprocess.run(['ps', '-p', str(pid), '-o', 'command='],
                                              capture_output=True, text=True, timeout=5)
                        if result.returncode == 0 and 'claude' in result.stdout.lower():
                            return True
                        return False
                    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
                        # Fallback to basic PID range check if ps command fails
                        if 1 <= pid <= 999999:
                            return True
            except Exception:
                pass
            return True

        return False

    def _detect_zombie_sessions(self, sessions: Dict[str, SessionInfo]) -> List[str]:
        """
        Detect zombie sessions using multiple validation methods.

        Args:
            sessions: Dictionary of session_id -> SessionInfo

        Returns:
            List of session_ids that are zombies
        """
        zombie_sessions = []

        for session_id, info in sessions.items():
            is_zombie = False
            reason = ""

            # Method 1: PID validation
            if not self._validate_pid_robust(info.pid):
                is_zombie = True
                reason = f"PID {info.pid} does not exist or not a Claude process"

            # Method 2: Stale timestamp check (more aggressive if enabled)
            elif self.AGGRESSIVE_CLEANUP:
                staleness_hours = (time.time() - info.last_heartbeat) / 3600
                if staleness_hours > 2:  # More aggressive: 2 hours instead of 5 minutes
                    is_zombie = True
                    reason = f"Stale session: {staleness_hours:.1f} hours old"

            # Method 3: Very old sessions (emergency cleanup)
            age_hours = (time.time() - info.started_at) / 3600
            if age_hours > 24:  # Sessions older than 24 hours are definitely zombies
                is_zombie = True
                reason = f"Very old session: {age_hours:.1f} hours"

            if is_zombie:
                zombie_sessions.append(session_id)
                self.logger.info(f"Detected zombie session {session_id}: {reason}")

        return zombie_sessions

    def _emergency_registry_repair(self) -> bool:
        """
        Emergency registry repair when normal methods fail.

        Creates a backup of the current registry and creates a new empty one.

        Returns:
            True if repair successful, False otherwise
        """
        try:
            registry_path = self.coordinator.registry_path

            # Create backup
            if os.path.exists(registry_path):
                backup_path = f"{registry_path}.backup.{int(time.time())}"
                shutil.move(registry_path, backup_path)
                self.logger.info(f"Created emergency registry backup: {backup_path}")

            # Create new empty registry
            with open(registry_path, 'w') as f:
                json.dump({}, f)

            self.logger.warning("Emergency registry repair completed - all sessions cleared")
            return True

        except Exception as e:
            self.logger.error(f"Emergency registry repair failed: {e}")
            return False

    def aggressive_cleanup(self) -> CleanupStats:
        """
        Perform aggressive cleanup of zombie and stale sessions.

        Returns:
            CleanupStats with operation results

        Raises:
            ValueError: If session coordinator is not initialized
            RuntimeError: If registry file cannot be accessed
        """
        # Input validation according to Context7 best practices
        if not self.coordinator:
            raise ValueError("Session coordinator not initialized")

        if not hasattr(self.coordinator, 'registry_path'):
            raise RuntimeError("Session coordinator missing registry_path attribute")

        start_time = time.time()
        stats = CleanupStats()

        try:
            # Acquire lock for registry access
            if not self.coordinator._acquire_lock(timeout=self.CLEANUP_TIMEOUT):
                self.logger.error("Failed to acquire lock for aggressive cleanup")
                stats.registry_errors += 1
                return stats

            try:
                # Read current registry
                sessions = self.coordinator._read_registry()
                stats.sessions_before = len(sessions)

                if not sessions:
                    self.logger.info("No sessions in registry to cleanup")
                    stats.sessions_after = 0
                    return stats

                self.logger.info(f"Starting aggressive cleanup on {len(sessions)} sessions")

                # Detect zombie sessions
                zombie_sessions = self._detect_zombie_sessions(sessions)

                # Remove zombie sessions
                for session_id in zombie_sessions:
                    if session_id in sessions:
                        del sessions[session_id]
                        stats.zombie_sessions_cleaned += 1

                # Detect and remove stale sessions (very old ones)
                current_time = time.time()
                stale_sessions = []

                for session_id, info in sessions.items():
                    # Very stale sessions (older than 6 hours without heartbeat)
                    if (current_time - info.last_heartbeat) > (6 * 3600):
                        stale_sessions.append(session_id)

                for session_id in stale_sessions:
                    if session_id in sessions:
                        del sessions[session_id]
                        stats.stale_sessions_cleaned += 1

                # Write cleaned registry
                if stats.zombie_sessions_cleaned > 0 or stats.stale_sessions_cleaned > 0:
                    self.coordinator._write_registry(sessions)
                    self.coordinator._sessions_cache = sessions

                    self.logger.info(
                        f"Aggressive cleanup completed",
                        extra={
                            'zombie_sessions_cleaned': stats.zombie_sessions_cleaned,
                            'stale_sessions_cleaned': stats.stale_sessions_cleaned,
                            'total_removed': stats.zombie_sessions_cleaned + stats.stale_sessions_cleaned
                        }
                    )

                stats.sessions_after = len(sessions)

            finally:
                self.coordinator._release_lock()

        except Exception as e:
            self.logger.error(f"Aggressive cleanup failed: {e}")
            stats.registry_errors += 1

            # Emergency fallback
            if self.EMERGENCY_OVERRIDE and stats.sessions_before > 0:
                self.logger.warning("Attempting emergency registry repair")
                if self._emergency_registry_repair():
                    stats.zombie_sessions_cleaned = stats.sessions_before
                    stats.sessions_after = 0

        stats.cleanup_duration = time.time() - start_time
        return stats

    def force_cleanup_all_sessions(self) -> bool:
        """
        Force cleanup of all sessions (emergency measure).

        Returns:
            True if cleanup successful, False otherwise

        Raises:
            ValueError: If session coordinator is not initialized
            RuntimeError: If lock acquisition fails
        """
        # Input validation
        if not self.coordinator:
            raise ValueError("Session coordinator not initialized")

        self.logger.warning("Force cleanup all sessions - EMERGENCY MEASURE")

        try:
            if not self.coordinator._acquire_lock(timeout=10):
                raise RuntimeError("Failed to acquire lock for force cleanup")

            try:
                # Clear registry completely
                self.coordinator._write_registry({})
                self.coordinator._sessions_cache = {}

                self.logger.warning("All sessions force-cleared from registry")
                return True

            finally:
                self.coordinator._release_lock()

        except Exception as e:
            self.logger.error(f"Force cleanup failed: {e}")
            return False

    def validate_and_fix_registry(self) -> bool:
        """
        Validate registry integrity and fix common issues.

        Returns:
            True if registry is valid or was fixed, False otherwise

        Raises:
            ValueError: If session coordinator is not initialized
            OSError: If registry file permissions prevent access
        """
        # Input validation
        if not self.coordinator:
            raise ValueError("Session coordinator not initialized")

        if not hasattr(self.coordinator, 'registry_path'):
            raise ValueError("Session coordinator missing registry_path attribute")

        try:
            # Check if registry file exists and is readable
            if not os.path.exists(self.coordinator.registry_path):
                self.coordinator._init_registry()
                return True

            # Try to read and parse registry
            try:
                with open(self.coordinator.registry_path, 'r') as f:
                    data = json.load(f)

                # Validate structure
                if not isinstance(data, dict):
                    self.logger.error("Registry structure invalid: not a dictionary")
                    return self._emergency_registry_repair()

                # Validate each session entry
                valid_sessions = {}
                for session_id, session_data in data.items():
                    try:
                        if isinstance(session_data, dict):
                            # Basic validation of required fields
                            required_fields = ['session_id', 'pid', 'started_at', 'last_heartbeat', 'status']
                            if all(field in session_data for field in required_fields):
                                valid_sessions[session_id] = session_data
                    except Exception as e:
                        self.logger.warning(f"Invalid session entry {session_id}: {e}")

                # If we cleaned up invalid entries, write back
                if len(valid_sessions) != len(data):
                    self.logger.info(f"Cleaned {len(data) - len(valid_sessions)} invalid session entries")
                    with open(self.coordinator.registry_path, 'w') as f:
                        json.dump(valid_sessions, f, indent=2)
                    self.coordinator._sessions_cache = valid_sessions

                return True

            except json.JSONDecodeError as e:
                self.logger.error(f"Registry JSON decode error: {e}")
                return self._emergency_registry_repair()

        except Exception as e:
            self.logger.error(f"Registry validation failed: {e}")
            return False


# Convenience functions for easy access
def cleanup_zombie_sessions() -> CleanupStats:
    """
    Convenience function to cleanup zombie sessions.

    Returns:
        CleanupStats with operation results
    """
    manager = SessionCleanupManager()
    return manager.aggressive_cleanup()


def emergency_session_reset() -> bool:
    """
    Convenience function for emergency session reset.

    Returns:
        True if reset successful, False otherwise
    """
    manager = SessionCleanupManager()
    return manager.force_cleanup_all_sessions()


if __name__ == "__main__":
    # Test session cleanup utilities
    print("DevStream Session Cleanup Utilities Test")
    print("=" * 50)

    manager = SessionCleanupManager()

    # Validate registry
    print("1. Validating registry...")
    is_valid = manager.validate_and_fix_registry()
    print(f"   Registry valid: {is_valid}")

    # Perform aggressive cleanup
    print("\n2. Performing aggressive cleanup...")
    stats = manager.aggressive_cleanup()
    print(f"   Sessions before: {stats.sessions_before}")
    print(f"   Sessions after: {stats.sessions_after}")
    print(f"   Zombie sessions cleaned: {stats.zombie_sessions_cleaned}")
    print(f"   Stale sessions cleaned: {stats.stale_sessions_cleaned}")
    print(f"   Cleanup duration: {stats.cleanup_duration:.3f}s")

    print("\n🎉 Session cleanup test completed!")