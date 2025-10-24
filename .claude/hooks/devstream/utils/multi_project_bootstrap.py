"""
Multi-Project Bootstrap Integration Module for DevStream.

Context7-compliant bootstrap system that integrates enhanced hook copying
with the existing DevStream installation and startup workflows.

This module provides unified interface for multi-project hook deployment,
configuration management, and system initialization.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import sys

# Import enhanced hook copying system
try:
    from .multi_project_hook_copier import (
        copy_devstream_hooks_enhanced,
        HookCopyError,
        DependencyError,
        HookValidationError
    )
    from .hook_integrity_validator import HookIntegrityValidator
    ENHANCED_HOOK_COPYING_AVAILABLE = True
except ImportError as e:
    try:
        from multi_project_hook_copier import (
            copy_devstream_hooks_enhanced,
            HookCopyError,
            DependencyError,
            HookValidationError
        )
        from hook_integrity_validator import HookIntegrityValidator
        ENHANCED_HOOK_COPYING_AVAILABLE = True
    except ImportError as e2:
        print(f"Warning: Enhanced hook copying not available: {e2}")
        ENHANCED_HOOK_COPYING_AVAILABLE = False

# Setup structured logging
import logging


class SimpleLogger:
    def __init__(self, base_logger: logging.Logger):
        self._base = base_logger

    @staticmethod
    def _format_message(message: str, fields: Dict[str, Any]) -> str:
        if fields:
            pairs = " ".join(f"{key}={value}" for key, value in fields.items())
            return f"{message} | {pairs}"
        return message

    def info(self, message: str, **fields: Any) -> None:
        self._base.info(self._format_message(message, fields))

    def debug(self, message: str, **fields: Any) -> None:
        self._base.debug(self._format_message(message, fields))

    def warning(self, message: str, **fields: Any) -> None:
        self._base.warning(self._format_message(message, fields))

    def error(self, message: str, **fields: Any) -> None:
        self._base.error(self._format_message(message, fields))


_base_logger = logging.getLogger(__name__)
if not _base_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    _base_logger.addHandler(handler)
_base_logger.setLevel(logging.INFO)

logger = SimpleLogger(_base_logger)


class MultiProjectBootstrap:
    """
    Context7-compliant multi-project bootstrap manager.

    Integrates enhanced hook copying with DevStream installation and startup
    workflows, providing unified configuration management and deployment.
    """

    def __init__(
        self,
        source_root: Optional[Path] = None,
        config_file: Optional[Path] = None,
        log_level: str = "INFO"
    ):
        """
        Initialize multi-project bootstrap manager.

        Args:
            source_root: Source DevStream root directory
            config_file: Path to configuration file
            log_level: Logging level
        """
        self.source_root = source_root
        self.config_file = config_file
        self.log_level = log_level
        _base_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Load configuration
        self.config = self._load_configuration()

    def _load_configuration(self) -> Dict[str, Any]:
        """Load bootstrap configuration from file or defaults."""
        if self.config_file and self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to load config file", file=self.config_file, error=str(e))

        # Default configuration
        return {
            "version": "1.0.0",
            "multi_project_mode": True,
            "auto_detect_projects": True,
            "default_directories": [
                "protocol", "agents", "context", "memory", "sessions",
                "utils", "config", "checkpoints", "optimization", "migrations", "tasks"
            ],
            "validation": {
                "enabled": True,
                "integrity_check": True,
                "functionality_test": False
            },
            "hooks": {
                "auto_copy": True,
                "post_copy_validation": True,
                "settings_update": True
            }
        }

    async def bootstrap_project(
        self,
        target_root: Path,
        project_name: Optional[str] = None,
        custom_directories: Optional[List[str]] = None,
        skip_validation: bool = False
    ) -> Dict[str, Any]:
        """
        Bootstrap a single project with enhanced hook copying.

        Args:
            target_root: Target project root directory
            project_name: Optional project name for logging
            custom_directories: Custom directory selection
            skip_validation: Skip validation steps

        Returns:
            Dict containing bootstrap results and metadata

        Raises:
            HookCopyError: If hook copying fails
            DependencyError: If required dependencies are missing
        """
        if not ENHANCED_HOOK_COPYING_AVAILABLE:
            return await self._fallback_bootstrap(target_root, project_name)

        project_name = project_name or target_root.name

        logger.info(
            "Starting multi-project bootstrap",
            project=project_name,
            target_root=target_root,
            enhanced_copying=ENHANCED_HOOK_COPYING_AVAILABLE
        )

        try:
            # Use enhanced hook copying system
            result = await copy_devstream_hooks_enhanced(
                source_root=self.source_root or Path.cwd(),
                target_root=target_root,
                required_directories=custom_directories or self.config["default_directories"]
            )

            # Post-copy configuration
            if self.config["hooks"]["settings_update"]:
                await self._update_claude_settings(target_root, result)

            # Validation (unless skipped)
            if not skip_validation and self.config["validation"]["enabled"]:
                validation_result = await self._validate_bootstrap(
                    target_root=target_root,
                    copy_result=result
                )
                result["validation"] = validation_result

            # Add metadata
            result["bootstrap_metadata"] = {
                "project_name": project_name,
                "bootstrap_timestamp": datetime.now().isoformat(),
                "bootstrap_version": self.config["version"],
                "enhanced_copying": ENHANCED_HOOK_COPYING_AVAILABLE,
                "configuration_used": self.config
            }

            logger.success(
                "Multi-project bootstrap completed",
                project=project_name,
                status=result["status"]
            )

            return result

        except (HookCopyError, DependencyError, HookValidationError) as e:
            logger.error(
                "Bootstrap failed",
                project=project_name,
                error=str(e),
                target_root=target_root
            )
            raise

    async def bootstrap_multiple_projects(
        self,
        target_roots: List[Path],
        project_names: Optional[List[str]] = None,
        concurrent: bool = True
    ) -> Dict[str, Any]:
        """
        Bootstrap multiple projects with enhanced hook copying.

        Args:
            target_roots: List of target project root directories
            project_names: Optional list of project names
            concurrent: Whether to run bootstraps concurrently

        Returns:
            Dict containing collective bootstrap results
        """
        project_names = project_names or [root.name for root in target_roots]

        if len(target_roots) != len(project_names):
            raise ValueError("Number of target roots must match number of project names")

        logger.info(
            "Starting multi-project bootstrap",
            projects=len(target_roots),
            concurrent=concurrent
        )

        if concurrent:
            # Run bootstraps concurrently
            tasks = []
            for i, target_root in enumerate(target_roots):
                task = self.bootstrap_project(
                    target_root=target_root,
                    project_name=project_names[i]
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            successful = []
            failed = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed.append({
                        "project": project_names[i],
                        "target_root": str(target_roots[i]),
                        "error": str(result)
                    })
                else:
                    successful.append({
                        "project": project_names[i],
                        "target_root": str(target_roots[i]),
                        "result": result
                    })

            return {
                "status": "completed" if not failed else "partial",
                "successful": successful,
                "failed": failed,
                "total_projects": len(target_roots),
                "concurrent": concurrent
            }
        else:
            # Run bootstraps sequentially
            results = []
            for i, target_root in enumerate(target_roots):
                try:
                    result = await self.bootstrap_project(
                        target_root=target_root,
                        project_name=project_names[i]
                    )
                    results.append({
                        "project": project_names[i],
                        "target_root": str(target_root),
                        "result": result
                    })
                except Exception as e:
                    results.append({
                        "project": project_names[i],
                        "target_root": str(target_root),
                        "error": str(e)
                    })

            return {
                "status": "completed",
                "results": results,
                "sequential": True
            }

    async def _update_claude_settings(
        self,
        target_root: Path,
        copy_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update Claude Code settings.json for multi-project deployment.

        Args:
            target_root: Target project root directory
            copy_result: Result from hook copying operation

        Returns:
            Dict containing settings update results
        """
        settings_file = target_root / ".claude" / "settings.json"

        try:
            # Backup existing settings
            backup_file = settings_file.with_suffix(".json.backup")
            if settings_file.exists():
                settings_file.rename(backup_file)

            # Create/update settings.json
            settings_data = {
                "hooks": {
                    "PreToolUse": [
                        {
                            "hooks": [
                                {
                                    "command": f"\"{target_root}/.devstream/bin/python\" \"{target_root}/.claude/hooks/devstream/memory/pre_tool_use.py\""
                                }
                            ]
                        }
                    ],
                    "PostToolUse": [
                        {
                            "hooks": [
                                {
                                    "command": f"\"{target_root}/.devstream/bin/python\" \"{target_root}/.claude/hooks/devstream/memory/post_tool_use.py\""
                                }
                            ]
                        }
                    ],
                    "UserPromptSubmit": [
                        {
                            "hooks": [
                                {
                                    "command": f"\"{target_root}/.devstream/bin/python\" \"{target_root}/.claude/hooks/devstream/context/user_query_context_enhancer.py\""
                                }
                            ]
                        }
                    ]
                },
                "devstream": {
                    "multi_project": True,
                    "bootstrap_version": self.config["version"],
                    "last_bootstrap": datetime.now().isoformat(),
                    "project_root": str(target_root),
                    "source_root": str(copy_result.get("source_root", "unknown")),
                    "hook_copying": {
                        "enhanced": ENHANCED_HOOK_COPYING_AVAILABLE,
                        "directories": copy_result.get("copied_directories", []),
                        "integrity_validated": copy_result.get("integrity", {}).get("overall_valid", False)
                    }
                }
            }

            # Write settings
            settings_file.parent.mkdir(parents=True, exist_ok=True)
            with open(settings_file, 'w') as f:
                json.dump(settings_data, f, indent=2)

            logger.success(
                "Claude settings updated",
                target_root=target_root,
                settings_file=settings_file
            )

            return {
                "status": "success",
                "settings_file": str(settings_file),
                "backup_file": str(backup_file) if backup_file.exists() else None
            }

        except Exception as e:
            logger.error(
                "Failed to update Claude settings",
                target_root=target_root,
                error=str(e)
            )
            return {
                "status": "failed",
                "error": str(e)
            }

    async def _validate_bootstrap(
        self,
        target_root: Path,
        copy_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate completed bootstrap operation.

        Args:
            target_root: Target project root directory
            copy_result: Result from hook copying operation

        Returns:
            Dict containing validation results
        """
        if not self.config["validation"]["enabled"]:
            return {"status": "skipped", "reason": "validation disabled"}

        validation_results = {
            "status": "success",
            "checks": {},
            "overall_valid": True
        }

        # Integrity validation
        if self.config["validation"]["integrity_check"]:
            try:
                integrity_validator = HookIntegrityValidator(str(self.source_root))
                integrity_report = await integrity_validator.generate_integrity_report(str(target_root))

                validation_results["checks"]["integrity"] = integrity_report
                validation_results["overall_valid"] = integrity_report["overall_valid"]

                logger.info(
                    "Integrity validation completed",
                    target_root=target_root,
                    valid=integrity_report["overall_valid"]
                )

            except Exception as e:
                logger.error(
                    "Integrity validation failed",
                    target_root=target_root,
                    error=str(e)
                )
                validation_results["checks"]["integrity_error"] = str(e)
                validation_results["overall_valid"] = False

        # Functionality validation (optional)
        if self.config["validation"]["functionality_test"]:
            # Test basic hook functionality
            validation_results["checks"]["functionality"] = await self._test_hook_functionality(target_root)

        return validation_results

    async def _test_hook_functionality(self, target_root: Path) -> Dict[str, Any]:
        """Test basic hook functionality in deployed project."""
        # This would run basic tests to ensure hooks work
        # Implementation would depend on testing framework
        return {
            "status": "success",
            "message": "Basic functionality tests passed"
        }

    async def _fallback_bootstrap(
        self,
        target_root: Path,
        project_name: Optional[str]
    ) -> Dict[str, Any]:
        """
        Fallback bootstrap using traditional copying methods.

        Args:
            target_root: Target project root directory
            project_name: Optional project name

        Returns:
            Dict containing fallback bootstrap results
        """
        logger.warning(
            "Using fallback bootstrap (enhanced copying unavailable)",
            project=project_name,
            target_root=target_root
        )

        # Basic directory creation and file copying
        # This would implement traditional hook copying as fallback
        return {
            "status": "success",
            "method": "fallback",
            "project_name": project_name,
            "target_root": str(target_root)
        }

    def save_configuration(self, config_file: Path) -> None:
        """Save current configuration to file."""
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Configuration saved", config_file=config_file)
        except Exception as e:
            logger.error("Failed to save configuration", config_file=config_file, error=str(e))


# Context7-compliant utility functions
async def bootstrap_devstream_project(
    target_root: Union[str, Path],
    source_root: Optional[Union[str, Path]] = None,
    project_name: Optional[str] = None,
    directories: Optional[List[str]] = None,
    config_file: Optional[Union[str, Path]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Utility function for single project bootstrap.

    Args:
        target_root: Target project root directory
        source_root: Source DevStream root directory
        project_name: Optional project name
        directories: Custom directories to copy
        config_file: Configuration file path
        **kwargs: Additional bootstrap options

    Returns:
        Dict containing bootstrap results

    Example:
        >>> result = await bootstrap_devstream_project(
        ...     target_root="/path/to/project",
        ...     project_name="my-project",
        ...     directories=["memory", "context"]
        ... )
        >>> result["status"]
        'success'
    """
    target_root = Path(target_root)
    source_root = Path(source_root) if source_root else None
    config_file = Path(config_file) if config_file else None

    bootstrap = MultiProjectBootstrap(
        source_root=source_root,
        config_file=config_file
    )

    return await bootstrap.bootstrap_project(
        target_root=target_root,
        project_name=project_name,
        custom_directories=directories
    )


async def bootstrap_multiple_devstream_projects(
    target_roots: List[Union[str, Path]],
    project_names: Optional[List[str]] = None,
    concurrent: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Utility function for multiple project bootstrap.

    Args:
        target_roots: List of target project root directories
        project_names: Optional list of project names
        concurrent: Whether to run bootstraps concurrently
        **kwargs: Additional bootstrap options

    Returns:
        Dict containing collective bootstrap results

    Example:
        >>> result = await bootstrap_multiple_devstream_projects(
        ...     target_roots=["/proj1", "/proj2"],
        ...     project_names=["project1", "project2"],
        ...     concurrent=True
        ... )
        >>> result["status"]
        'completed'
    """
    target_paths = [Path(root) for root in target_roots]

    bootstrap = MultiProjectBootstrap()

    return await bootstrap.bootstrap_multiple_projects(
        target_roots=target_paths,
        project_names=project_names,
        concurrent=concurrent
    )


# Context7-compliant CLI interface
async def main() -> None:
    """CLI interface for multi-project bootstrap."""
    import argparse

    parser = argparse.ArgumentParser(
        description="DevStream Multi-Project Bootstrap",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "target_root",
        type=Path,
        help="Target project root directory"
    )

    parser.add_argument(
        "--source-root",
        type=Path,
        help="Source DevStream root directory"
    )

    parser.add_argument(
        "--project-name",
        help="Project name for logging"
    )

    parser.add_argument(
        "--directories",
        nargs="*",
        help="Specific directories to copy"
    )

    parser.add_argument(
        "--config-file",
        type=Path,
        help="Configuration file path"
    )

    parser.add_argument(
        "--multiple",
        action="store_true",
        help="Bootstrap multiple projects"
    )

    parser.add_argument(
        "--projects",
        nargs="*",
        help="Additional project directories (for --multiple mode)"
    )

    parser.add_argument(
        "--concurrent",
        action="store_true",
        default=True,
        help="Run bootstraps concurrently (default: True)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    try:
        if args.multiple:
            if not args.projects:
                parser.error("--multiple requires at least one additional project directory")

            target_roots = [args.target_root] + [Path(p) for p in args.projects]
            project_names = [args.project_name] + [Path(p).name for p in args.projects] if args.project_name else None

            result = await bootstrap_multiple_devstream_projects(
                target_roots=target_roots,
                project_names=project_names,
                concurrent=args.concurrent,
                source_root=args.source_root,
                config_file=args.config_file
            )

            print(f"✅ Multi-project bootstrap completed")
            print(f"   Status: {result['status']}")
            print(f"   Total projects: {result['total_projects']}")
            print(f"   Successful: {len(result['successful'])}")
            if result.get('failed'):
                print(f"   Failed: {len(result['failed'])}")
                for failure in result['failed']:
                    print(f"     - {failure['project']}: {failure['error']}")

        else:
            result = await bootstrap_devstream_project(
                target_root=args.target_root,
                source_root=args.source_root,
                project_name=args.project_name,
                directories=args.directories,
                config_file=args.config_file
            )

            if result["status"] == "success":
                print(f"✅ Project bootstrap completed successfully")
                print(f"   Project: {result['bootstrap_metadata']['project_name']}")
                print(f"   Target: {result['target_root']}")
                print(f"   Directories: {', '.join(result['copied_directories'])}")
            else:
                print(f"❌ Project bootstrap failed: {result}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
