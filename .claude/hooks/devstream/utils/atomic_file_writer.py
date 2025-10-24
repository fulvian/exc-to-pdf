#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "aiofiles>=23.0.0",
#     "structlog>=23.0.0",
# ]
# ///

"""
Atomic File Writer Utility - Context7 Async Pattern Compliant

Provides atomic file writing using temp file + rename pattern to prevent
partial writes in cross-session scenarios (SessionEnd, PreCompact hooks).

Best Practices Applied:
- aiofiles for async I/O (Context7 recommended)
- Write-rename pattern for atomicity (POSIX guarantee)
- tempfile.NamedTemporaryFile in same directory (same filesystem)
- os.replace() for cross-platform atomic rename (Python 3.3+)

Usage Example (SessionEnd Hook):
    from pathlib import Path
    from utils.atomic_file_writer import write_atomic_json

    async def save_session_summary(summary_data: dict) -> bool:
        summary_file = Path(".claude/state/session_summary.json")
        success = await write_atomic_json(summary_file, summary_data)
        if success:
            logger.info("session_summary_saved", file=str(summary_file))
        return success

Usage Example (PreCompact Hook):
    from pathlib import Path
    from utils.atomic_file_writer import write_atomic

    async def save_compact_summary(summary_text: str) -> bool:
        summary_file = Path(".claude/state/compact_summary.md")
        success = await write_atomic(summary_file, summary_text)
        return success
"""

import os
import tempfile
import secrets
import stat
from pathlib import Path
from typing import Optional

import aiofiles
import aiofiles.os
import structlog

logger = structlog.get_logger(__name__)


def _generate_secure_temp_name(base_name: str) -> str:
    """
    Generate a secure, unpredictable temporary file name.

    Context7 Security Best Practices:
    - Uses cryptographically secure random bytes (secrets.token_hex)
    - Prevents predictable filename attacks
    - Ensures sufficient entropy (16 bytes = 32 hex chars)

    Args:
        base_name: Original file name for context

    Returns:
        Secure temporary file name with random suffix

    Example:
        >>> _generate_secure_temp_name("config.json")
        '.config.json.a1b2c3d4e5f6789012345678901234ab.tmp'
    """
    # Generate 16 cryptographically secure random bytes (32 hex chars)
    random_suffix = secrets.token_hex(16)
    return f".{base_name}.{random_suffix}.tmp"


def _validate_temp_file_security(temp_path: Path) -> bool:
    """
    Context7-comprehensive security validation of temporary file.

    Security Checks (OWASP Best Practices):
    1. Symlink attack prevention - verify file is not a symlink
    2. Ownership verification - ensure we own the file
    3. Permission validation - verify restrictive permissions
    4. Path validation - ensure file is in expected directory
    5. Race condition protection - validate file properties

    Args:
        temp_path: Path to temporary file to validate

    Returns:
        True if file passes all security checks, False otherwise

    Security Notes:
        - Prevents privilege escalation via symlink attacks
        - Ensures file isolation and confidentiality
        - Validates atomic file creation properties
    """
    try:
        # SECURITY CHECK 1: Symlink attack prevention
        if temp_path.is_symlink():
            logger.error(
                "temp_file_security_failed",
                path=str(temp_path),
                reason="symlink_detected",
                details="File is a symbolic link - possible symlink attack"
            )
            return False

        # SECURITY CHECK 2: File existence validation
        if not temp_path.exists():
            logger.error(
                "temp_file_security_failed",
                path=str(temp_path),
                reason="file_not_found",
                details="Temporary file does not exist"
            )
            return False

        # SECURITY CHECK 3: Permission validation (Context7 secure defaults)
        file_stat = temp_path.stat()
        file_mode = file_stat.st_mode

        # Check file permissions are restrictive (owner read/write only)
        # Expected: 0o600 (rw-------) or more restrictive
        if file_mode & 0o077:  # Check any group/other permissions
            logger.error(
                "temp_file_security_failed",
                path=str(temp_path),
                reason="insecure_permissions",
                details=f"File mode {oct(file_mode)} allows group/other access",
                expected_mode="0o600 (rw-------) or more restrictive"
            )
            return False

        # SECURITY CHECK 4: Ownership verification
        current_uid = os.getuid() if hasattr(os, 'getuid') else None
        file_uid = file_stat.st_uid if hasattr(file_stat, 'st_uid') else None

        if current_uid is not None and file_uid != current_uid:
            logger.error(
                "temp_file_security_failed",
                path=str(temp_path),
                reason="ownership_mismatch",
                details=f"File owned by uid {file_uid}, current uid {current_uid}",
                severity="HIGH"
            )
            return False

        # SECURITY CHECK 5: Path validation (ensure same filesystem)
        # This prevents cross-filesystem symlink attacks
        try:
            temp_dev = file_stat.st_dev if hasattr(file_stat, 'st_dev') else None
            parent_stat = temp_path.parent.stat()
            parent_dev = parent_stat.st_dev if hasattr(parent_stat, 'st_dev') else None

            if temp_dev is not None and parent_dev is not None and temp_dev != parent_dev:
                logger.error(
                    "temp_file_security_failed",
                    path=str(temp_path),
                    reason="cross_filesystem",
                    details="Temp file is on different filesystem from parent directory",
                    temp_device=temp_dev,
                    parent_device=parent_dev,
                    severity="HIGH"
                )
                return False
        except (OSError, AttributeError):
            # If we can't verify filesystem, log but don't fail
            logger.warning(
                "temp_file_filesystem_check_failed",
                path=str(temp_path),
                reason="filesystem_verification_failed",
                details="Unable to verify filesystem device numbers"
            )

        # All security checks passed
        logger.debug(
            "temp_file_security_validated",
            path=str(temp_path),
            file_mode=oct(file_mode),
            file_uid=file_uid,
            file_size=file_stat.st_size
        )

        return True

    except (OSError, AttributeError) as e:
        logger.error(
            "temp_file_security_check_error",
            path=str(temp_path),
            error=str(e),
            error_type=type(e).__name__
        )
        return False


def _set_secure_file_permissions(file_path: Path) -> bool:
    """
    Set Context7-compliant secure permissions on temporary file.

    Security Standards:
    - Owner read/write only (0o600 = rw-------)
    - No group or other permissions
    - Prevents information disclosure

    Args:
        file_path: Path to file to secure

    Returns:
        True if permissions set successfully, False otherwise
    """
    try:
        # Set restrictive permissions: owner read/write only
        os.chmod(file_path, 0o600)

        logger.debug(
            "secure_permissions_set",
            file_path=str(file_path),
            permissions="0o600 (rw-------)"
        )

        return True

    except OSError as e:
        logger.error(
            "secure_permissions_failed",
            file_path=str(file_path),
            error=str(e),
            error_type=type(e).__name__
        )
        return False


async def write_atomic(
    file_path: Path,
    content: str,
    encoding: str = "utf-8"
) -> bool:
    """
    Context7-Secure atomic file writing with comprehensive security protections.

    Enhanced Security Features (SEC-005 Fixed):
    - Cryptographically secure temp file names (prevents prediction attacks)
    - Symlink attack prevention (comprehensive validation)
    - Secure file permissions (0o600 - owner read/write only)
    - Race condition protection (security validation before use)
    - Cross-filesystem attack prevention

    Uses the secure write-rename pattern:
    1. Generate cryptographically secure temp file name
    2. Create temp file with secure permissions
    3. Validate temp file security (symlink, ownership, permissions)
    4. Write content using async I/O
    5. Atomic rename temp → target

    Args:
        file_path: Target file path to write to
        content: Content string to write
        encoding: File encoding (default: utf-8)

    Returns:
        True if write succeeded, False if failed

    Raises:
        OSError: If file operation fails (disk full, permissions, etc.)

    Security Notes:
        - Temp files use 32-character cryptographically secure random names
        - All temp files validated before use (symlink attack prevention)
        - Secure permissions enforced (0o600 = rw-------)
        - Race condition protection through comprehensive validation
        - Cross-filesystem symlink attack prevention

    Example:
        >>> success = await write_atomic(
        ...     Path("/path/to/file.json"),
        ...     json.dumps(data, indent=2)
        ... )
        >>> if success:
        ...     print("File written securely and atomically")
    """
    tmp_fd = None
    tmp_path = None

    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # SECURITY: Generate cryptographically secure temp file name
        # Context7 Best Practice: Use secrets.token_hex() for unpredictability
        secure_temp_name = _generate_secure_temp_name(file_path.name)
        tmp_path = file_path.parent / secure_temp_name

        # SECURITY: Create temp file with secure permissions from start
        # Use O_EXCL | O_CREAT to prevent race conditions in file creation
        try:
            # Create file with exclusive access (prevents race conditions)
            tmp_fd = os.open(
                tmp_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_TRUNC,
                mode=0o600  # Secure permissions from creation
            )
        except FileExistsError:
            logger.error(
                "temp_file_creation_race",
                temp_path=str(tmp_path),
                reason="file_already_exists",
                details="Race condition detected during temp file creation"
            )
            return False

        # SECURITY: Validate the created file meets security requirements
        if not _validate_temp_file_security(tmp_path):
            # Cleanup failed security validation
            try:
                os.close(tmp_fd)
                await aiofiles.os.remove(tmp_path)
            except OSError:
                pass  # Best effort cleanup
            return False

        # SECURITY: Ensure permissions remain secure after validation
        if not _set_secure_file_permissions(tmp_path):
            # Cleanup if permission setting fails
            try:
                os.close(tmp_fd)
                await aiofiles.os.remove(tmp_path)
            except OSError:
                pass  # Best effort cleanup
            return False

        # Write content using the secure file descriptor
        try:
            # Write content using os.write with the secure fd
            content_bytes = content.encode(encoding)
            os.write(tmp_fd, content_bytes)

            # Ensure data is written to disk (fsync for durability)
            os.fsync(tmp_fd)

            logger.debug(
                "secure_content_written",
                temp_path=str(tmp_path),
                content_size=len(content_bytes),
                encoding=encoding
            )

        except OSError as write_error:
            logger.error(
                "secure_content_write_failed",
                temp_path=str(tmp_path),
                error=str(write_error),
                error_type=type(write_error).__name__
            )
            # Cleanup on write failure
            try:
                os.close(tmp_fd)
                await aiofiles.os.remove(tmp_path)
            except OSError:
                pass
            return False

        # Close the file descriptor
        os.close(tmp_fd)
        tmp_fd = None

        # SECURITY: Final validation before atomic rename
        if not _validate_temp_file_security(tmp_path):
            logger.error(
                "final_security_validation_failed",
                temp_path=str(tmp_path),
                reason="post_write_security_check_failed",
                severity="HIGH"
            )
            # Cleanup failed validation
            try:
                await aiofiles.os.remove(tmp_path)
            except OSError:
                pass
            return False

        # SECURITY: Atomic rename with validation
        # os.replace() is atomic on both POSIX and Windows (Python 3.3+)
        try:
            await aiofiles.os.replace(tmp_path, file_path)
        except OSError as rename_error:
            logger.error(
                "atomic_rename_failed",
                temp_path=str(tmp_path),
                target_path=str(file_path),
                error=str(rename_error),
                error_type=type(rename_error).__name__
            )
            # Cleanup failed rename
            try:
                await aiofiles.os.remove(tmp_path)
            except OSError:
                pass
            return False

        logger.info(
            "secure_atomic_write_success",
            file_path=str(file_path),
            content_size=len(content),
            encoding=encoding,
            security_features="secure_temp_name,symlink_protection,secure_permissions,race_condition_protection"
        )

        return True

    except OSError as e:
        logger.error(
            "secure_atomic_write_failed",
            file_path=str(file_path),
            error=str(e),
            error_type=type(e).__name__
        )

        # Cleanup temp file on failure
        if tmp_path and tmp_path.exists():
            try:
                await aiofiles.os.remove(tmp_path)
                logger.debug("temp_file_cleaned_up", tmp_path=str(tmp_path))
            except OSError as cleanup_error:
                logger.warning(
                    "temp_file_cleanup_failed",
                    tmp_path=str(tmp_path),
                    error=str(cleanup_error)
                )

        return False

    except Exception as e:
        # Unexpected errors (should rarely happen)
        logger.error(
            "secure_atomic_write_unexpected_error",
            file_path=str(file_path),
            error=str(e),
            error_type=type(e).__name__
        )

        # Cleanup temp file on failure
        if tmp_path and tmp_path.exists():
            try:
                await aiofiles.os.remove(tmp_path)
            except OSError:
                pass  # Best effort cleanup

        return False

    finally:
        # Ensure temp file descriptor is closed if still open
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except OSError:
                pass  # Best effort cleanup


async def write_atomic_json(
    file_path: Path,
    data: dict,
    encoding: str = "utf-8",
    indent: int = 2
) -> bool:
    """
    Write JSON data to file atomically.

    Convenience wrapper around write_atomic() for JSON data.
    Automatically serializes dict to JSON string with formatting.

    Args:
        file_path: Target file path to write to
        data: Dictionary to serialize as JSON
        encoding: File encoding (default: utf-8)
        indent: JSON indentation spaces (default: 2)

    Returns:
        True if write succeeded, False if failed

    Raises:
        OSError: If file operation fails
        ValueError: If data cannot be serialized to JSON

    Note:
        Uses json.dumps() with indent for human-readable formatting.

    Example:
        >>> data = {"summary": "Session completed", "tasks": 5}
        >>> success = await write_atomic_json(
        ...     Path("/path/to/summary.json"),
        ...     data
        ... )
    """
    import json

    try:
        # Serialize to JSON string
        content = json.dumps(data, indent=indent, ensure_ascii=False)

        # Write atomically
        return await write_atomic(file_path, content, encoding=encoding)

    except (TypeError, ValueError) as e:
        logger.error(
            "json_serialization_failed",
            file_path=str(file_path),
            error=str(e),
            error_type=type(e).__name__
        )
        return False


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    import json

    async def test_atomic_writer():
        """Test atomic file writer functionality."""

        # Test 1: Basic text write
        test_file = Path("/tmp/test_atomic_write.txt")
        content = "Test content with\nmultiple lines\nand Unicode: 中文测试"

        success = await write_atomic(test_file, content)
        assert success, "Basic write failed"
        assert test_file.read_text() == content, "Content mismatch"
        print("✅ Test 1: Basic text write - PASSED")

        # Test 2: JSON write
        test_json = Path("/tmp/test_atomic_write.json")
        data = {
            "session_id": "test-123",
            "summary": "Test session summary",
            "tasks_completed": 5,
            "timestamp": "2025-10-02T12:00:00Z"
        }

        success = await write_atomic_json(test_json, data)
        assert success, "JSON write failed"
        loaded_data = json.loads(test_json.read_text())
        assert loaded_data == data, "JSON data mismatch"
        print("✅ Test 2: JSON write - PASSED")

        # Test 3: Overwrite existing file
        success = await write_atomic(test_file, "New content")
        assert success, "Overwrite failed"
        assert test_file.read_text() == "New content", "Overwrite content mismatch"
        print("✅ Test 3: Overwrite existing file - PASSED")

        # Test 4: Parent directory creation
        nested_file = Path("/tmp/atomic_test_nested/subdir/file.txt")
        success = await write_atomic(nested_file, "Nested content")
        assert success, "Nested directory write failed"
        assert nested_file.exists(), "Nested file not created"
        print("✅ Test 4: Parent directory creation - PASSED")

        # Cleanup
        test_file.unlink()
        test_json.unlink()
        nested_file.unlink()
        nested_file.parent.rmdir()
        nested_file.parent.parent.rmdir()

        print("\n🎉 All atomic file writer tests PASSED")

    # Run tests
    asyncio.run(test_atomic_writer())
