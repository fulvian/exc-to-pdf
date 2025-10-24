"""
Enhanced multi-project hook copying module using Copier library.

Context7-compliant implementation for robust DevStream hook copying
with dependency resolution, error handling, and integrity validation.
"""

import asyncio
import hashlib
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import structlog

# Import Copier for template-based copying
try:
    from copier import run_copy
    from copier.errors import CopierError, ExtensionNotFoundError
except ImportError as e:
    raise ImportError("Copier library is required. Install with: pip install copier>=9.0.0") from e

# Import DevStream utilities (optional, only used for potential future extensions)
# try:
#     from .common import DevStreamHookBase
# except ImportError:
#     # Fallback for direct execution
#     sys.path.append(str(Path(__file__).parent))
#     from common import DevStreamHookBase

# Setup structured logging
logger = structlog.get_logger()


class HookCopyError(Exception):
    """Exception raised when hook copying fails."""
    pass


class DependencyError(Exception):
    """Exception raised when required dependencies are missing."""
    pass


class HookValidationError(Exception):
    """Exception raised when hook validation fails."""
    pass


async def copy_devstream_hooks_enhanced(
    source_root: Path,
    target_root: Path,
    required_directories: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Context7-compliant multi-project hook copying using Copier library.

    Provides robust copying of DevStream hooks with dependency resolution,
    error handling, and integrity validation for multi-project deployment.

    Args:
        source_root: Source DevStream root directory (typically devstream/)
        target_root: Target project root where hooks will be copied
        required_directories: List of required directories to copy
                            (defaults to all DevStream directories)

    Returns:
        Dict containing copy results, validation status, and metadata

    Raises:
        HookCopyError: If copying fails validation or encounters errors
        DependencyError: If required dependencies are missing

    Example:
        >>> result = await copy_devstream_hooks_enhanced(
        ...     source_root=Path("/Users/fulvioventura/devstream"),
        ...     target_root=Path("/Users/fulvioventura/reporting")
        ... )
        >>> result["status"]
        'success'
    """
    # Set default required directories if not provided
    if required_directories is None:
        required_directories = [
            "protocol", "agents", "context", "memory", "sessions",
            "utils", "config", "checkpoints", "optimization", "migrations", "tasks"
        ]

    # Validate inputs
    if not source_root.exists():
        raise HookCopyError(f"Source root does not exist: {source_root}")

    if not source_root.is_dir():
        raise HookCopyError(f"Source root is not a directory: {source_root}")

    # Ensure target root exists
    target_root.mkdir(parents=True, exist_ok=True)

    # Validate source contains required DevStream structure
    source_claude = source_root / ".claude"
    if not source_claude.exists():
        raise HookCopyError(f"Source does not contain .claude directory: {source_claude}")

    logger.info(
        "Starting enhanced DevStream hook copying",
        source_root=source_root,
        target_root=target_root,
        required_directories=required_directories
    )

    try:
        # Use Copier for robust template-based copying
        logger.info("Using Copier for hook copying", source=source_claude, target=target_root / ".claude")

        # Run Copier with Context7-compliant settings
        # The run_copy function performs the copying synchronously
        worker = run_copy(
            src_path=str(source_claude),
            dst_path=str(target_root / ".claude"),
            data={
                "project_name": target_root.name,
                "devstream_mode": "multi_project",
                "hook_copy_timestamp": asyncio.get_event_loop().time()
            },
            defaults=True,  # Use defaults for any template variables
            quiet=False,     # Show progress for debugging
            vcs_ref="HEAD",  # Include local changes
            overwrite=True   # Overwrite existing files
        )

        logger.info("Hook copying completed successfully", source=source_root, target=target_root)

        # Post-copy validation
        validation_result = await _validate_hook_copying(
            target_root=target_root,
            required_directories=required_directories
        )

        # Integrity validation using Context7-compliant validator
        integrity_result = await _validate_hook_integrity(
            source_root=source_root,
            target_root=target_root
        )

        return {
            "status": "success",
            "copied_directories": required_directories,
            "source_root": str(source_root),
            "target_root": str(target_root),
            "validation": validation_result,
            "integrity": integrity_result,
            "copier_worker": str(type(worker).__name__)
        }

    except CopierError as e:
        logger.error("Copier operation failed", error=str(e), source=source_root)
        raise HookCopyError(f"Failed to copy hooks with Copier: {e}") from e
    except Exception as e:
        logger.error("Unexpected error during hook copying", error=str(e), source=source_root)
        raise HookCopyError(f"Unexpected error during hook copying: {e}") from e


async def _validate_hook_copying(
    target_root: Path,
    required_directories: List[str]
) -> Dict[str, Any]:
    """
    Validate that hooks were copied correctly.

    Args:
        target_root: Target project root to validate
        required_directories: List of directories that must be present

    Returns:
        Dict containing validation results

    Raises:
        HookValidationError: If validation fails
    """
    logger.info("Starting post-copy validation", target_root=target_root)

    validation_result: Dict[str, Any] = {
        "valid": True,
        "missing_directories": [],
        "missing_files": [],
        "copied_files": []
    }

    target_claude = target_root / ".claude"
    target_hooks = target_claude / "hooks" / "devstream"

    # Check that target structure exists
    if not target_claude.exists():
        validation_result["valid"] = False
        validation_result["missing_directories"].append(".claude")
        raise HookValidationError(f"Target .claude directory not created: {target_claude}")

    if not target_hooks.exists():
        validation_result["valid"] = False
        validation_result["missing_directories"].append(".claude/hooks/devstream")
        raise HookValidationError(f"Target hooks directory not created: {target_hooks}")

    # Check required directories
    for dir_name in required_directories:
        dir_path = target_hooks / dir_name
        if not dir_path.exists():
            validation_result["valid"] = False
            validation_result["missing_directories"].append(dir_name)
            logger.warning("Missing required directory", directory=dir_name, path=dir_path)
        else:
            logger.info("Required directory present", directory=dir_name, path=dir_path)
            # Count files in directory
            files = list(dir_path.rglob("*"))
            validation_result["copied_files"].extend([str(f.relative_to(target_root)) for f in files if f.is_file()])

    if not validation_result["valid"]:
        error_msg = f"Validation failed - missing directories: {validation_result['missing_directories']}"
        logger.error("Hook copying validation failed", missing_directories=validation_result["missing_directories"])
        raise HookValidationError(error_msg)

    logger.info(
        "Hook copying validation passed",
        target_root=target_root,
        copied_files=len(validation_result["copied_files"])
    )

    return validation_result


def calculate_file_checksums(directory: Path) -> Dict[str, str]:
    """
    Calculate SHA256 checksums for all files in a directory.

    Args:
        directory: Directory to scan

    Returns:
        Dict mapping file paths to their SHA256 checksums
    """
    checksums = {}

    for file_path in directory.rglob("*"):
        if file_path.is_file():
            try:
                content = file_path.read_bytes()
                checksum = hashlib.sha256(content).hexdigest()
                relative_path = str(file_path.relative_to(directory))
                checksums[relative_path] = checksum
            except (OSError, PermissionError) as e:
                logger.warning("Failed to calculate checksum", file=file_path, error=str(e))

    return checksums


async def verify_hook_integrity(
    target_root: Path,
    expected_checksums: Dict[str, str]
) -> Dict[str, Any]:
    """
    Verify integrity of copied hooks using checksums.

    Args:
        target_root: Target project root containing copied hooks
        expected_checksums: Expected checksums for files

    Returns:
        Dict containing integrity verification results
    """
    logger.info("Starting integrity verification", target_root=target_root)

    target_hooks = target_root / ".claude" / "hooks" / "devstream"
    actual_checksums = calculate_file_checksums(target_hooks)

    verification_result: Dict[str, Any] = {
        "verified": True,
        "missing_files": [],
        "corrupted_files": [],
        "extra_files": [],
        "total_files": len(actual_checksums)
    }

    # Check for missing files
    for expected_file in expected_checksums:
        if expected_file not in actual_checksums:
            verification_result["verified"] = False
            verification_result["missing_files"].append(expected_file)

    # Check for corrupted files
    for file_path, expected_checksum in expected_checksums.items():
        if file_path in actual_checksums:
            actual_checksum = actual_checksums[file_path]
            if actual_checksum != expected_checksum:
                verification_result["verified"] = False
                verification_result["corrupted_files"].append({
                    "file": file_path,
                    "expected": expected_checksum,
                    "actual": actual_checksum
                })

    # Check for extra files (not necessarily an error)
    for file_path in actual_checksums:
        if file_path not in expected_checksums:
            verification_result["extra_files"].append(file_path)

    if verification_result["verified"]:
        logger.info(
            "Integrity verification passed",
            target_root=target_root,
            files_verified=len(expected_checksums)
        )
    else:
        logger.error(
            "Integrity verification failed",
            target_root=target_root,
            missing_files=len(verification_result["missing_files"]),
            corrupted_files=len(verification_result["corrupted_files"])
        )

    return verification_result


async def _validate_hook_integrity(
    source_root: Path,
    target_root: Path
) -> Dict[str, Any]:
    """
    Validate integrity of copied hooks using Context7-compliant validator.

    Args:
        source_root: Source DevStream root directory
        target_root: Target project root with copied hooks

    Returns:
        Dict containing integrity validation results
    """
    logger.info("Starting integrity validation", source=source_root, target=target_root)

    try:
        # Import integrity validator with fallback
        integrity_validator = _import_integrity_validator()

        if integrity_validator is None:
            # Fallback: Basic integrity checks
            return await _basic_integrity_check(source_root, target_root)

        # Use full integrity validator
        validator = integrity_validator.HookIntegrityValidator(str(source_root))
        integrity_report = await validator.generate_integrity_report(str(target_root))

        logger.info("Integrity validation completed",
                   target=target_root,
                   overall_valid=integrity_report["overall_valid"],
                   total_hooks=integrity_report["summary"]["total_hooks"])

        return integrity_report

    except Exception as e:
        logger.error("Integrity validation failed",
                    source=source_root,
                    target=target_root,
                    error=str(e))

        # Return basic integrity check as fallback
        return await _basic_integrity_check(source_root, target_root)


def _import_integrity_validator():
    """
    Import hook integrity validator with multi-project fallback support.

    Returns:
        HookIntegrityValidator module or None if unavailable
    """
    # Strategy 1: Try relative import
    try:
        from . import hook_integrity_validator
        return hook_integrity_validator
    except ImportError:
        pass

    # Strategy 2: Try absolute import
    try:
        import hook_integrity_validator
        return hook_integrity_validator
    except ImportError:
        pass

    # Strategy 3: Try import from current working directory
    try:
        import sys
        import importlib.util

        # Look for hook_integrity_validator.py in various locations
        search_paths = [
            Path.cwd() / ".claude" / "hooks" / "devstream" / "utils" / "hook_integrity_validator.py",
            Path.cwd() / "hook_integrity_validator.py",
        ]

        for search_path in search_paths:
            if search_path.exists():
                spec = importlib.util.spec_from_file_location("hook_integrity_validator", search_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

    except Exception:
        pass

    # Strategy 4: Return None (will use fallback)
    return None


async def _basic_integrity_check(
    source_root: Path,
    target_root: Path
) -> Dict[str, Any]:
    """
    Basic integrity check when full validator is unavailable.

    Args:
        source_root: Source DevStream root directory
        target_root: Target project root with copied hooks

    Returns:
        Dict containing basic integrity validation results
    """
    logger.info("Performing basic integrity check", source=source_root, target=target_root)

    target_hooks = target_root / ".claude" / "hooks" / "devstream"
    source_hooks = source_root / ".claude" / "hooks" / "devstream"

    results = {
        "project_root": str(target_root),
        "validation_timestamp": datetime.now().isoformat(),
        "overall_valid": True,
        "summary": {
            "total_hooks": 0,
            "valid_hooks": 0,
            "critical_failures": 0,
            "warnings": 0
        },
        "results": [],
        "recommendations": [],
        "validation_method": "basic_fallback"
    }

    if not target_hooks.exists():
        results["overall_valid"] = False
        results["summary"]["critical_failures"] = 1
        results["results"].append({
            "valid": False,
            "hook_name": "hooks_directory",
            "validation_type": "basic_check",
            "message": "Target hooks directory does not exist",
            "details": {"path": str(target_hooks)},
            "timestamp": datetime.now().isoformat(),
            "project_root": str(target_root),
            "severity": "critical"
        })
        results["recommendations"].append("Create target hooks directory structure")
        return results

    # Check essential hook files
    essential_hooks = [
        "memory/pre_tool_use.py",
        "memory/post_tool_use.py",
        "context/user_query_context_enhancer.py",
        "utils/direct_client.py"
    ]

    for hook_path in essential_hooks:
        target_file = target_hooks / hook_path
        source_file = source_hooks / hook_path

        hook_result = {
            "valid": True,
            "hook_name": hook_path,
            "validation_type": "basic_check",
            "message": "Basic check passed",
            "details": {},
            "timestamp": datetime.now().isoformat(),
            "project_root": str(target_root),
            "severity": "info"
        }

        results["summary"]["total_hooks"] += 1

        # Check file existence
        if not target_file.exists():
            hook_result["valid"] = False
            hook_result["message"] = f"Hook file missing: {hook_path}"
            hook_result["severity"] = "critical"
            results["summary"]["critical_failures"] += 1
            results["overall_valid"] = False

        # Check file size if source exists
        elif source_file.exists():
            target_size = target_file.stat().st_size
            source_size = source_file.stat().st_size

            hook_result["details"]["source_size"] = source_size
            hook_result["details"]["target_size"] = target_size
            hook_result["details"]["size_match"] = target_size == source_size

            if target_size != source_size:
                hook_result["valid"] = False
                hook_result["message"] = f"File size mismatch for {hook_path}"
                hook_result["severity"] = "warning"
                results["summary"]["warnings"] += 1

        # Check basic syntax for Python files
        if target_file.exists() and target_file.suffix == '.py':
            try:
                content = target_file.read_text(encoding='utf-8')
                # Basic syntax check - ensure it has valid Python structure
                if 'import' in content or 'def' in content or 'class' in content:
                    hook_result["details"]["basic_syntax"] = "appears_valid"
                else:
                    hook_result["details"]["basic_syntax"] = "minimal_content"
                    hook_result["severity"] = "warning"
                    results["summary"]["warnings"] += 1

            except Exception as e:
                hook_result["valid"] = False
                hook_result["message"] = f"Failed to read hook file: {e}"
                hook_result["severity"] = "critical"
                results["summary"]["critical_failures"] += 1
                results["overall_valid"] = False

        if hook_result["valid"]:
            results["summary"]["valid_hooks"] += 1

        results["results"].append(hook_result)

    # Generate recommendations
    if results["summary"]["critical_failures"] > 0:
        results["recommendations"].append(f"Fix {results['summary']['critical_failures']} critical hook failures")

    if results["summary"]["warnings"] > 0:
        results["recommendations"].append(f"Address {results['summary']['warnings']} warning-level issues")

    if results["overall_valid"]:
        results["recommendations"].append("Basic integrity checks passed - consider running full validation")

    logger.info("Basic integrity check completed",
               target=target_root,
               overall_valid=results["overall_valid"],
               hooks_checked=results["summary"]["total_hooks"])

    return results


# Context7-compliant CLI interface for standalone usage
async def main() -> None:
    """CLI interface for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced DevStream hook copying")
    parser.add_argument("source_root", type=Path, help="Source DevStream root directory")
    parser.add_argument("target_root", type=Path, help="Target project root directory")
    parser.add_argument(
        "--directories",
        nargs="*",
        default=["protocol", "agents", "context", "memory", "sessions", "utils"],
        help="Required directories to copy"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    try:
        result = await copy_devstream_hooks_enhanced(
            source_root=args.source_root,
            target_root=args.target_root,
            required_directories=args.directories
        )

        if result["status"] == "success":
            print(f"✅ Hook copying completed successfully")
            print(f"   Source: {result['source_root']}")
            print(f"   Target: {result['target_root']}")
            print(f"   Directories: {', '.join(result['copied_directories'])}")
            if result["validation"]["copied_files"]:
                print(f"   Files copied: {len(result['validation']['copied_files'])}")
        else:
            print(f"❌ Hook copying failed: {result}")
            sys.exit(1)

    except (HookCopyError, DependencyError, HookValidationError) as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("❌ Operation cancelled by user")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())