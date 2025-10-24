#!/usr/bin/env python3
"""
DevStream Configuration Validator
Validates complete system configuration before first use
Version: 0.1.0-beta

Validates:
- Python venv and dependencies
- Hook files and executability
- settings.json configuration
- Database and MCP server
- Environment variables
"""

import json
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any


class Colors:
    """ANSI color codes."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'


class ConfigValidator:
    """Validates DevStream system configuration."""

    def __init__(self):
        """Initialize validator."""
        self.project_root = self._detect_project_root()
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.checks_passed = 0
        self.checks_total = 0

    def _detect_project_root(self) -> Path:
        """Detect project root."""
        current = Path(__file__).resolve().parent.parent
        for _ in range(3):
            if (current / '.claude').exists():
                return current
            current = current.parent
        raise FileNotFoundError("Project root not found")

    def log_check(self, name: str) -> None:
        """Log check start."""
        print(f"\n{Colors.BLUE}► Checking:{Colors.NC} {name}")

    def log_ok(self, message: str) -> None:
        """Log success."""
        print(f"  {Colors.GREEN}✓{Colors.NC} {message}")
        self.checks_passed += 1

    def log_fail(self, message: str) -> None:
        """Log failure."""
        print(f"  {Colors.RED}✗{Colors.NC} {message}")
        self.errors.append(message)

    def log_warn(self, message: str) -> None:
        """Log warning."""
        print(f"  {Colors.YELLOW}⚠{Colors.NC} {message}")
        self.warnings.append(message)

    def check_python_venv(self) -> bool:
        """Validate Python virtual environment."""
        self.log_check("Python Virtual Environment")
        self.checks_total += 3

        venv_dir = self.project_root / '.devstream'
        python_bin = venv_dir / 'bin' / 'python'

        # Check venv exists
        if not venv_dir.exists():
            self.log_fail(f"Venv not found: {venv_dir}")
            return False
        self.log_ok("Venv directory exists")

        # Check Python executable
        if not python_bin.exists():
            self.log_fail(f"Python not found: {python_bin}")
            return False
        self.log_ok("Python executable found")

        # Check Python version
        try:
            result = subprocess.run(
                [str(python_bin), '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            version = result.stdout.strip()
            if '3.11' in version:
                self.log_ok(f"Python version: {version}")
                return True
            else:
                self.log_warn(f"Python version: {version} (expected 3.11+)")
                return True
        except Exception as e:
            self.log_fail(f"Python check failed: {e}")
            return False

    def check_python_dependencies(self) -> bool:
        """Validate Python dependencies."""
        self.log_check("Python Dependencies")

        python_bin = self.project_root / '.devstream' / 'bin' / 'python'
        required_packages = [
            ('cchooks', 'cchooks'),
            ('aiohttp', 'aiohttp'),
            ('structlog', 'structlog'),
            ('python-dotenv', 'dotenv'),
        ]

        self.checks_total += len(required_packages)
        all_ok = True

        for package_name, import_name in required_packages:
            try:
                result = subprocess.run(
                    [str(python_bin), '-c', f'import {import_name}'],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    self.log_ok(f"{package_name} installed")
                else:
                    self.log_fail(f"{package_name} NOT installed")
                    all_ok = False
            except Exception as e:
                self.log_fail(f"{package_name} check failed: {e}")
                all_ok = False

        return all_ok

    def check_hook_files(self) -> bool:
        """Validate hook files."""
        self.log_check("Hook Files")

        hooks = [
            '.claude/hooks/devstream/memory/pre_tool_use.py',
            '.claude/hooks/devstream/memory/post_tool_use.py',
            '.claude/hooks/devstream/context/user_query_context_enhancer.py',
            '.claude/hooks/devstream/context/session_start.py',
        ]

        self.checks_total += len(hooks) * 2
        all_ok = True

        for hook in hooks:
            hook_path = self.project_root / hook
            if hook_path.exists():
                self.log_ok(f"{hook} exists")
                if hook_path.stat().st_mode & 0o111:
                    self.log_ok(f"{hook} is executable")
                else:
                    self.log_warn(f"{hook} NOT executable (fixable)")
            else:
                self.log_fail(f"{hook} NOT found")
                all_ok = False

        return all_ok

    def check_settings_json(self) -> bool:
        """Validate settings.json configuration."""
        self.log_check("Claude Code settings.json")
        self.checks_total += 5

        settings_file = Path.home() / '.claude' / 'settings.json'

        # Check file exists
        if not settings_file.exists():
            self.log_fail(f"settings.json NOT found: {settings_file}")
            return False
        self.log_ok("settings.json exists")

        # Read and parse
        try:
            with open(settings_file) as f:
                settings = json.load(f)
            self.log_ok("settings.json is valid JSON")
        except json.JSONDecodeError as e:
            self.log_fail(f"settings.json is invalid JSON: {e}")
            return False

        # Check hooks section
        if 'hooks' not in settings:
            self.log_fail("No 'hooks' section in settings.json")
            return False
        self.log_ok("'hooks' section found")

        # Check hook types
        hooks_section = settings['hooks']
        required_hooks = ['PreToolUse', 'PostToolUse', 'UserPromptSubmit', 'SessionStart']
        found_hooks = sum(1 for h in required_hooks if h in hooks_section)

        if found_hooks == len(required_hooks):
            self.log_ok(f"All {len(required_hooks)} hook types configured")
        elif found_hooks > 0:
            self.log_warn(f"Only {found_hooks}/{len(required_hooks)} hook types configured")
        else:
            self.log_fail("No DevStream hooks configured in settings.json")
            return False

        # Validate paths point to this project
        project_str = str(self.project_root)
        has_correct_paths = False

        for hook_type in required_hooks:
            if hook_type in hooks_section:
                hook_config = json.dumps(hooks_section[hook_type])
                if project_str in hook_config:
                    has_correct_paths = True
                    break

        if has_correct_paths:
            self.log_ok("Hook paths point to this project")
            return True
        else:
            self.log_warn("Hook paths may not point to this project")
            return True

    def check_database(self) -> bool:
        """Validate database configuration."""
        self.log_check("DevStream Database")
        self.checks_total += 2

        db_file = self.project_root / 'data' / 'devstream.db'

        if db_file.exists():
            self.log_ok(f"Database exists: {db_file}")
            self.log_ok(f"Database size: {db_file.stat().st_size} bytes")
            return True
        else:
            self.log_warn("Database not yet created (will be created on first use)")
            return True

    def check_env_file(self) -> bool:
        """Validate environment configuration."""
        self.log_check("Environment Configuration")
        self.checks_total += 2

        env_file = self.project_root / '.env.devstream'

        if not env_file.exists():
            self.log_warn(".env.devstream not found (optional)")
            return True

        self.log_ok(".env.devstream exists")

        # Check critical variables
        with open(env_file) as f:
            content = f.read()

        if 'DEVSTREAM_HOOKS_ENABLED=true' in content:
            self.log_ok("DEVSTREAM_HOOKS_ENABLED=true")
            return True
        else:
            self.log_warn("DEVSTREAM_HOOKS_ENABLED not set to true")
            return True

    def check_mcp_server(self) -> bool:
        """Validate MCP server."""
        self.log_check("MCP Server")
        self.checks_total += 2

        mcp_dir = self.project_root / 'mcp-devstream-server'

        if not mcp_dir.exists():
            self.log_warn("MCP server not found (optional component)")
            return True

        self.log_ok("MCP server directory exists")

        dist_file = mcp_dir / 'dist' / 'index.js'
        if dist_file.exists():
            self.log_ok("MCP server built (dist/index.js exists)")
            return True
        else:
            self.log_warn("MCP server not built (run: cd mcp-devstream-server && npm run build)")
            return True

    def print_summary(self) -> None:
        """Print validation summary."""
        print(f"\n{'=' * 70}")
        print("  Validation Summary")
        print(f"{'=' * 70}\n")

        print(f"Checks Passed:  {self.checks_passed}/{self.checks_total}")
        print(f"Errors:         {len(self.errors)}")
        print(f"Warnings:       {len(self.warnings)}")

        if self.errors:
            print(f"\n{Colors.RED}✗ ERRORS ({len(self.errors)}){Colors.NC}")
            for error in self.errors:
                print(f"  • {error}")

        if self.warnings:
            print(f"\n{Colors.YELLOW}⚠ WARNINGS ({len(self.warnings)}){Colors.NC}")
            for warning in self.warnings:
                print(f"  • {warning}")

        print(f"\n{'=' * 70}")

        if self.errors:
            print(f"{Colors.RED}✗ VALIDATION FAILED{Colors.NC}")
            print("\nPlease fix errors before using DevStream.")
            print("Run: ./install.sh")
        elif self.warnings:
            print(f"{Colors.YELLOW}⚠ VALIDATION PASSED WITH WARNINGS{Colors.NC}")
            print("\nDevStream should work, but check warnings.")
        else:
            print(f"{Colors.GREEN}✓ ALL CHECKS PASSED{Colors.NC}")
            print("\nDevStream is ready to use!")

        print(f"\n{'=' * 70}\n")

    def run(self) -> int:
        """
        Run validation.

        Returns:
            int: Exit code (0 = success, 1 = errors found)
        """
        print(f"\n{'=' * 70}")
        print("  DevStream Configuration Validator")
        print(f"  Project: {self.project_root}")
        print(f"{'=' * 70}")

        # Run all checks
        self.check_python_venv()
        self.check_python_dependencies()
        self.check_hook_files()
        self.check_settings_json()
        self.check_database()
        self.check_env_file()
        self.check_mcp_server()

        # Print summary
        self.print_summary()

        return 0 if not self.errors else 1


def main() -> int:
    """Main entry point."""
    try:
        validator = ConfigValidator()
        return validator.run()
    except Exception as e:
        print(f"{Colors.RED}[✗]{Colors.NC} Fatal error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
