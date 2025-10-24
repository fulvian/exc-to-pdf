#!/usr/bin/env python3
"""
DevStream Project Configuration Manager
Manages project-specific settings including provider choice
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def get_devstream_root() -> Path:
    """
    Get DevStream root using Context7 patterns.

    Priority 1: DEVSTREAM_ROOT environment variable
    Priority 2: Script location detection (two levels up from scripts/)
    Priority 3: Current working directory
    Priority 4: Common installation locations

    Returns:
        Path to DevStream root directory

    Raises:
        RuntimeError: If DevStream root cannot be determined
    """
    # Priority 1: DEVSTREAM_ROOT environment variable
    devstream_root = os.getenv("DEVSTREAM_ROOT")
    if devstream_root and Path(devstream_root).exists():
        return Path(devstream_root).absolute()

    # Priority 2: Script location detection
    script_file = Path(__file__)
    if script_file.exists():
        # Two levels up from scripts/ directory
        potential_root = script_file.parent.parent
        if (potential_root / "start-devstream.sh").exists():
            return potential_root.absolute()

    # Priority 3: Current working directory
    cwd = Path.cwd()
    if (cwd / "start-devstream.sh").exists():
        return cwd.absolute()

    # Priority 4: Common installation locations
    possible_locations = [
        Path.home() / ".devstream",
        Path.home() / "devstream",
        Path("/opt/devstream"),
        Path.cwd(),
    ]

    for location in possible_locations:
        if location.exists() and (location / "start-devstream.sh").exists():
            return location.absolute()

    # If nothing found, raise an informative error
    raise RuntimeError(
        "DevStream installation not found! Please set DEVSTREAM_ROOT environment variable "
        "or ensure DevStream is properly installed with start-devstream.sh"
    )


class DevStreamConfig:
    """Manages DevStream project configuration."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).absolute()
        self.devstream_dir = self.project_root / ".devstream"
        self.config_file = self.devstream_dir / "config.json"
        self.workspace_file = self.devstream_dir / "workspace.json"

    def ensure_devstream_dir(self) -> bool:
        """Ensure .devstream directory exists."""
        if not self.devstream_dir.exists():
            self.devstream_dir.mkdir(parents=True, exist_ok=True)
            return True
        return False

    def get_config(self) -> Dict[str, Any]:
        """Get project configuration."""
        if not self.config_file.exists():
            return self.get_default_config()

        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        """Get default project configuration."""
        return {
            "project": {
                "name": self.project_root.name,
                "provider": "anthropic",  # Default provider
                "auto_start": True,
                "preferred_tools": ["claude-code", "devstream"]
            },
            "providers": {
                "anthropic": {
                    "enabled": True,
                    "model": "claude-3-5-sonnet-20241022",
                    "auth_method": "oauth"
                },
                "z.ai": {
                    "enabled": True,
                    "model": "glm-4.6",
                    "api_key_env": "ZAI_API_KEY"
                }
            },
            "devstream": {
                "memory_enabled": True,
                "context7_enabled": True,
                "auto_scan": True,
                "database_path": str(self.devstream_dir / "db" / "devstream.db")
            },
            "workspace": {
                "auto_save": True,
                "session_timeout": 3600,
                "max_context_tokens": 8000
            }
        }

    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save project configuration."""
        try:
            self.ensure_devstream_dir()
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except (IOError, json.JSONDecodeError):
            return False

    def get_provider(self) -> str:
        """Get current provider for this project."""
        config = self.get_config()
        return config.get("project", {}).get("provider", "anthropic")

    def set_provider(self, provider: str) -> bool:
        """Set provider for this project."""
        if provider not in ["anthropic", "z.ai"]:
            return False

        config = self.get_config()
        config["project"]["provider"] = provider
        return self.save_config(config)

    def get_workspace_config(self) -> Dict[str, Any]:
        """Get workspace configuration."""
        if not self.workspace_file.exists():
            return {}

        try:
            with open(self.workspace_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def update_workspace_config(self, updates: Dict[str, Any]) -> bool:
        """Update workspace configuration."""
        workspace_config = self.get_workspace_config()
        workspace_config.update(updates)

        try:
            with open(self.workspace_file, 'w') as f:
                json.dump(workspace_config, f, indent=2)
            return True
        except (IOError, json.JSONDecodeError):
            return False

    def get_startup_command(self) -> str:
        """Get appropriate startup command based on provider."""
        provider = self.get_provider()

        try:
            devstream_root = get_devstream_root()
        except RuntimeError:
            # Fallback to environment variable or default for backward compatibility
            devstream_root = Path(os.environ.get("DEVSTREAM_ROOT", "/devstream"))

        if provider == "z.ai":
            return f"{devstream_root}/start-devstream.sh start z.ai"
        else:
            return f"{devstream_root}/start-devstream.sh start anthropic"

    def show_config(self) -> None:
        """Display current configuration."""
        config = self.get_config()
        provider = config.get("project", {}).get("provider", "anthropic")

        print(f"📋 Project Configuration: {self.project_root.name}")
        print("=" * 50)
        print(f"Provider: {provider}")
        print(f"Project Name: {config.get('project', {}).get('name', 'Unknown')}")
        print(f"Auto Start: {config.get('project', {}).get('auto_start', True)}")
        print(f"Memory Enabled: {config.get('devstream', {}).get('memory_enabled', True)}")
        print(f"Context7 Enabled: {config.get('devstream', {}).get('context7_enabled', True)}")
        print("")


def main():
    """Command line interface."""
    if len(sys.argv) < 2:
        print("Usage: devstream-config.py <command> [args]")
        print("")
        print("Commands:")
        print("  show                    Show current configuration")
        print("  set-provider <provider> Set provider (anthropic|z.ai)")
        print("  get-provider            Get current provider")
        print("  init                    Initialize default config")
        print("  startup                 Get startup command")
        print("")
        sys.exit(1)

    command = sys.argv[1]
    project_root = sys.argv[2] if len(sys.argv) > 2 else "."

    config_manager = DevStreamConfig(project_root)

    if command == "show":
        config_manager.show_config()

    elif command == "get-provider":
        provider = config_manager.get_provider()
        print(provider)

    elif command == "set-provider":
        if len(sys.argv) < 3:
            print("Error: Provider required")
            sys.exit(1)
        provider = sys.argv[2]
        if config_manager.set_provider(provider):
            print(f"✅ Provider set to: {provider}")
        else:
            print(f"❌ Failed to set provider: {provider}")
            sys.exit(1)

    elif command == "init":
        default_config = config_manager.get_default_config()
        if config_manager.save_config(default_config):
            print("✅ Default configuration initialized")
        else:
            print("❌ Failed to initialize configuration")
            sys.exit(1)

    elif command == "startup":
        print(config_manager.get_startup_command())

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()