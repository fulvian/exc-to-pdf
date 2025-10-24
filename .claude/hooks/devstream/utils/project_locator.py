#!/usr/bin/env python3
"""
DevStream Project Locator - Context7-compliant Facade Pattern

Provides centralized project location and path resolution services.
Implements Facade pattern to simplify access to project-related utilities.
Follows Dependency Injection pattern for testability and loose coupling.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ProjectInfo:
    """Project information container."""
    root_path: Path
    project_type: str
    is_devstream_framework: bool
    config_files: list[str]
    venv_paths: Dict[str, Optional[Path]]


class ProjectLocator:
    """
    Facade for project location and path resolution services.

    Context7 Pattern: Facade + Dependency Injection
    Provides simplified interface to complex path resolution logic.
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize project locator.

        Args:
            base_path: Base path to start searching from (defaults to current file)
        """
        self.base_path = base_path or Path(__file__).parent.parent.parent.parent
        self._cache: Dict[str, ProjectInfo] = {}

    def get_project_root(self, from_path: Optional[Path] = None) -> Path:
        """
        Get DevStream project root directory with caching.

        Args:
            from_path: Path to start searching from

        Returns:
            Path to project root directory
        """
        search_path = from_path or self.base_path
        cache_key = str(search_path)

        if cache_key in self._cache:
            return self._cache[cache_key].root_path

        # Search for project markers
        current = Path(search_path).resolve()

        while current != current.parent:
            if self._is_devstream_root(current):
                project_info = ProjectInfo(
                    root_path=current,
                    project_type="devstream",
                    is_devstream_framework=True,
                    config_files=self._find_config_files(current),
                    venv_paths=self._find_venv_paths(current)
                )
                self._cache[cache_key] = project_info
                return current

            current = current.parent

        # Fallback to base_path if no markers found
        fallback_info = ProjectInfo(
            root_path=self.base_path,
            project_type="unknown",
            is_devstream_framework=False,
            config_files=[],
            venv_paths={}
        )
        self._cache[cache_key] = fallback_info
        return self.base_path

    def _is_devstream_root(self, path: Path) -> bool:
        """
        Check if path is DevStream project root.

        Args:
            path: Path to check

        Returns:
            True if DevStream project root
        """
        markers = [
            ".claude",
            "CLAUDE.md",
            "start-devstream.sh",
            ".env.devstream"
        ]

        return any((path / marker).exists() for marker in markers)

    def _find_config_files(self, path: Path) -> list[str]:
        """Find DevStream configuration files."""
        config_files = []
        for file in ["CLAUDE.md", ".env.devstream", "PROJECT_STRUCTURE.md"]:
            if (path / file).exists():
                config_files.append(file)
        return config_files

    def _find_venv_paths(self, path: Path) -> Dict[str, Optional[Path]]:
        """Find virtual environment paths."""
        return {
            "framework": path / ".devstream" if (path / ".devstream").exists() else None,
            "project": path / ".venv" if (path / ".venv").exists() else None
        }

    def get_project_info(self, from_path: Optional[Path] = None) -> ProjectInfo:
        """
        Get comprehensive project information.

        Args:
            from_path: Path to start searching from

        Returns:
            Complete project information
        """
        root = self.get_project_root(from_path)
        cache_key = str(from_path or self.base_path)

        if cache_key not in self._cache:
            # This shouldn't happen if get_project_root was called first
            self.get_project_root(from_path)

        return self._cache[cache_key]

    def resolve_relative_path(self, relative_path: str, from_path: Optional[Path] = None) -> Path:
        """
        Resolve relative path from project root.

        Args:
            relative_path: Relative path from project root
            from_path: Path to resolve from

        Returns:
            Absolute path
        """
        project_root = self.get_project_root(from_path)
        return project_root / relative_path

    def get_database_path(self, from_path: Optional[Path] = None) -> Path:
        """
        Get DevStream database path.

        Args:
            from_path: Path to resolve from

        Returns:
            Database path
        """
        project_root = self.get_project_root(from_path)
        return project_root / "data" / "devstream.db"

    def is_devstream_available(self) -> bool:
        """
        Check if DevStream framework is available.

        Returns:
            True if DevStream framework is available
        """
        try:
            project_info = self.get_project_info()
            return project_info.is_devstream_framework
        except Exception:
            return False

    def clear_cache(self):
        """Clear internal cache."""
        self._cache.clear()


# Global instance following Singleton pattern (Context7 best practice)
_global_locator: Optional[ProjectLocator] = None


def get_project_locator(base_path: Optional[Path] = None) -> ProjectLocator:
    """
    Get global project locator instance (Dependency Injection).

    Args:
        base_path: Base path for locator (only used on first call)

    Returns:
        ProjectLocator instance
    """
    global _global_locator
    if _global_locator is None:
        _global_locator = ProjectLocator(base_path)
    return _global_locator


# Convenience functions for backward compatibility
def get_project_root(from_path: Optional[Path] = None) -> Path:
    """
    Get DevStream project root directory.

    Args:
        from_path: Path to start searching from

    Returns:
        Path to project root directory
    """
    locator = get_project_locator()
    return locator.get_project_root(from_path)


def get_database_path(from_path: Optional[Path] = None) -> Path:
    """
    Get DevStream database path.

    Args:
        from_path: Path to resolve from

    Returns:
        Database path
    """
    locator = get_project_locator()
    return locator.get_database_path(from_path)


def is_devstream_available() -> bool:
    """
    Check if DevStream framework is available.

    Returns:
        True if DevStream framework is available
    """
    locator = get_project_locator()
    return locator.is_devstream_available()