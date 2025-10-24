"""
Hook Integrity Validator - Context7 Compliant Implementation.

This module provides comprehensive integrity validation for DevStream hooks
copied to multi-project environments, ensuring functionality, completeness,
and proper configuration.

Context7-compliant patterns from pytest-asyncio and sqlite-utils implementations.
"""

import asyncio
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of integrity validation with detailed information."""
    valid: bool
    hook_name: str
    validation_type: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    project_root: str
    severity: str  # 'critical', 'warning', 'info'


@dataclass
class HookMetadata:
    """Metadata for a DevStream hook."""
    name: str
    path: str
    expected_size: Optional[int] = None
    expected_checksum: Optional[str] = None
    required_dependencies: List[str] = None
    validation_functions: List[str] = None

    def __post_init__(self):
        if self.required_dependencies is None:
            self.required_dependencies = []
        if self.validation_functions is None:
            self.validation_functions = []


class HookIntegrityError(Exception):
    """Exception raised when hook integrity validation fails."""
    pass


class HookIntegrityValidator:
    """
    Context7-compliant hook integrity validator.

    Implements comprehensive validation using Context7 patterns from
    pytest-asyncio and sqlite-utils for robust multi-project deployment.
    """

    def __init__(self, devstream_root: str):
        """
        Initialize hook integrity validator.

        Args:
            devstream_root: Path to DevStream framework root
        """
        self.devstream_root = Path(devstream_root).resolve()
        self.framework_hooks = self.devstream_root / ".claude" / "hooks" / "devstream"
        self.validation_cache: Dict[str, ValidationResult] = {}

        logger.info("HookIntegrityValidator initialized",
                    devstream_root=str(self.devstream_root))

    async def validate_project_hooks(
        self,
        project_root: str,
        hooks_to_validate: Optional[List[str]] = None
    ) -> Tuple[bool, List[ValidationResult]]:
        """
        Validate all hooks in a project using Context7 patterns.

        Args:
            project_root: Path to project root containing copied hooks
            hooks_to_validate: Specific hooks to validate (None for all)

        Returns:
            Tuple of (overall_valid, validation_results)
        """
        project_path = Path(project_root).resolve()
        project_hooks = project_path / ".claude" / "hooks" / "devstream"

        logger.info("Starting comprehensive hook validation",
                    project_root=str(project_root),
                    hooks_to_validate=hooks_to_validate)

        if not project_hooks.exists():
            error_result = ValidationResult(
                valid=False,
                hook_name="project_hooks",
                validation_type="directory_structure",
                message="Project hooks directory does not exist",
                details={"expected_path": str(project_hooks)},
                timestamp=datetime.now(),
                project_root=str(project_root),
                severity="critical"
            )
            return False, [error_result]

        # Determine hooks to validate
        if hooks_to_validate is None:
            hooks_to_validate = self._discover_project_hooks(project_hooks)

        validation_results = []

        # Context7 Pattern: Execute validations concurrently with error isolation
        validation_tasks = []
        for hook_name in hooks_to_validate:
            task = self._validate_single_hook_safe(project_root, hook_name)
            validation_tasks.append(task)

        # Wait for all validations with timeout (Context7 pattern)
        results = await asyncio.gather(*validation_tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error("Hook validation failed with exception", error=str(result))
                error_result = ValidationResult(
                    valid=False,
                    hook_name="unknown",
                    validation_type="exception",
                    message=f"Validation failed: {str(result)}",
                    details={"exception_type": type(result).__name__},
                    timestamp=datetime.now(),
                    project_root=str(project_root),
                    severity="critical"
                )
                validation_results.append(error_result)
            else:
                validation_results.append(result)

        # Calculate overall validity
        critical_failures = [r for r in validation_results if r.severity == "critical" and not r.valid]
        overall_valid = len(critical_failures) == 0

        # Log validation summary
        valid_count = sum(1 for r in validation_results if r.valid)
        total_count = len(validation_results)
        critical_count = len(critical_failures)
        warning_count = sum(1 for r in validation_results if r.severity == "warning" and not r.valid)

        logger.info("Hook validation completed",
                    project_root=str(project_root),
                    total_hooks=total_count,
                    valid_hooks=valid_count,
                    critical_failures=critical_count,
                    warnings=warning_count,
                    overall_valid=overall_valid)

        return overall_valid, validation_results

    async def _validate_single_hook_safe(
        self,
        project_root: str,
        hook_name: str
    ) -> ValidationResult:
        """
        Safely validate a single hook with error isolation (Context7 pattern).

        Args:
            project_root: Project root path
            hook_name: Name of hook to validate

        Returns:
            ValidationResult with comprehensive details
        """
        try:
            return await self._validate_single_hook(project_root, hook_name)
        except Exception as e:
            logger.error("Hook validation failed",
                        hook_name=hook_name,
                        project_root=project_root,
                        error=str(e))
            return ValidationResult(
                valid=False,
                hook_name=hook_name,
                validation_type="safe_validation",
                message=f"Validation failed: {str(e)}",
                details={"exception_type": type(e).__name__},
                timestamp=datetime.now(),
                project_root=project_root,
                severity="critical"
            )

    async def _validate_single_hook(
        self,
        project_root: str,
        hook_name: str
    ) -> ValidationResult:
        """
        Validate a single hook using comprehensive checks.

        Args:
            project_root: Project root path
            hook_name: Name of hook to validate

        Returns:
            ValidationResult with detailed information
        """
        project_path = Path(project_root).resolve()
        hook_path = project_path / ".claude" / "hooks" / "devstream" / hook_name

        # Context7 Pattern: Perform multiple validation layers
        validations = [
            ("file_existence", self._validate_file_existence),
            ("file_permissions", self._validate_file_permissions),
            ("syntax_validity", self._validate_syntax),
            ("dependencies", self._validate_dependencies),
            ("functionality", self._validate_functionality),
            ("integration", self._validate_integration)
        ]

        all_details = {}
        overall_valid = True
        messages = []
        severity = "info"

        for validation_name, validation_func in validations:
            try:
                valid, details = await validation_func(project_root, hook_name)
                all_details[validation_name] = details

                if not valid:
                    overall_valid = False
                    if validation_name in ["syntax_validity", "functionality"]:
                        severity = "critical"
                    elif severity != "critical":
                        severity = "warning"
                    messages.append(f"{validation_name}: {details.get('message', 'Failed')}")

            except Exception as e:
                logger.warning("Validation step failed",
                             validation=validation_name,
                             hook_name=hook_name,
                             error=str(e))
                all_details[validation_name] = {
                    "valid": False,
                    "error": str(e)
                }
                overall_valid = False
                severity = "critical"
                messages.append(f"{validation_name}: Exception - {str(e)}")

        return ValidationResult(
            valid=overall_valid,
            hook_name=hook_name,
            validation_type="comprehensive",
            message="; ".join(messages) if messages else "All validations passed",
            details=all_details,
            timestamp=datetime.now(),
            project_root=str(project_root),
            severity=severity
        )

    async def _validate_file_existence(
        self,
        project_root: str,
        hook_name: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate that hook file exists and is accessible."""
        project_path = Path(project_root).resolve()
        hook_path = project_path / ".claude" / "hooks" / "devstream" / hook_name

        details = {
            "hook_path": str(hook_path),
            "exists": hook_path.exists(),
            "is_file": hook_path.is_file() if hook_path.exists() else False,
            "is_readable": False
        }

        if hook_path.exists() and hook_path.is_file():
            try:
                details["is_readable"] = os.access(hook_path, os.R_OK)
                details["file_size"] = hook_path.stat().st_size
                details["last_modified"] = datetime.fromtimestamp(hook_path.stat().st_mtime).isoformat()
                return True, details
            except OSError as e:
                details["error"] = str(e)
                return False, details
        else:
            return False, details

    async def _validate_file_permissions(
        self,
        project_root: str,
        hook_name: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate that hook file has correct permissions."""
        project_path = Path(project_root).resolve()
        hook_path = project_path / ".claude" / "hooks" / "devstream" / hook_name

        details = {
            "hook_path": str(hook_path),
            "executable": False,
            "readable": False,
            "writable": False
        }

        if not hook_path.exists():
            return False, {"error": "Hook file does not exist"}

        try:
            stat_info = hook_path.stat()
            details["permissions_octal"] = oct(stat_info.st_mode)[-3:]
            details["executable"] = os.access(hook_path, os.X_OK)
            details["readable"] = os.access(hook_path, os.R_OK)
            details["writable"] = os.access(hook_path, os.W_OK)

            # Python scripts should be executable but this is not strictly required
            if hook_path.suffix == '.py':
                details["python_executable"] = True
                return details["readable"], details
            else:
                return details["readable"], details

        except OSError as e:
            details["error"] = str(e)
            return False, details

    async def _validate_syntax(
        self,
        project_root: str,
        hook_name: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate Python syntax for hook files."""
        project_path = Path(project_root).resolve()
        hook_path = project_path / ".claude" / "hooks" / "devstream" / hook_name

        details = {
            "hook_path": str(hook_path),
            "is_python": hook_path.suffix == '.py',
            "syntax_valid": False
        }

        if not hook_path.exists():
            return False, {"error": "Hook file does not exist"}

        if not details["is_python"]:
            # For non-Python files, just check they're not empty
            try:
                content = hook_path.read_text(encoding='utf-8')
                details["non_empty"] = len(content.strip()) > 0
                return details["non_empty"], details
            except Exception as e:
                details["error"] = str(e)
                return False, details

        # Python syntax validation using python -m py_compile
        try:
            # Get the project's Python executable
            project_venv = project_path / ".devstream" / "bin" / "python"
            python_exe = project_venv if project_venv.exists() else sys.executable

            # Use py_compile for syntax checking
            result = await asyncio.create_subprocess_exec(
                str(python_exe), "-m", "py_compile", str(hook_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await result.communicate()

            details["compile_returncode"] = result.returncode
            details["compile_stdout"] = stdout.decode('utf-8', errors='ignore')
            details["compile_stderr"] = stderr.decode('utf-8', errors='ignore')
            details["syntax_valid"] = result.returncode == 0

            if result.returncode == 0:
                return True, details
            else:
                details["error"] = "Syntax error detected"
                return False, details

        except Exception as e:
            details["error"] = str(e)
            return False, details

    async def _validate_dependencies(
        self,
        project_root: str,
        hook_name: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate that hook dependencies are available."""
        project_path = Path(project_root).resolve()
        hook_path = project_path / ".claude" / "hooks" / "devstream" / hook_name

        details = {
            "hook_path": str(hook_path),
            "dependencies_checked": [],
            "missing_dependencies": [],
            "available_dependencies": []
        }

        if not hook_path.exists():
            return False, {"error": "Hook file does not exist"}

        # Extract imports from Python file
        if hook_path.suffix == '.py':
            try:
                content = hook_path.read_text(encoding='utf-8')
                imports = self._extract_python_imports(content)
                details["extracted_imports"] = imports

                # Check critical DevStream dependencies
                critical_deps = ['structlog', 'pathlib', 'asyncio']
                optional_deps = ['cchooks', 'aiohttp', 'dotenv']

                project_venv = project_path / ".devstream" / "bin" / "python"
                python_exe = project_venv if project_venv.exists() else sys.executable

                for dep in critical_deps + optional_deps:
                    details["dependencies_checked"].append(dep)

                    # Check if dependency is available
                    check_script = f"import {dep}; print('OK')"
                    result = await asyncio.create_subprocess_exec(
                        str(python_exe), "-c", check_script,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )

                    stdout, stderr = await result.communicate()
                    dep_available = result.returncode == 0 and b"OK" in stdout

                    if dep_available:
                        details["available_dependencies"].append(dep)
                    else:
                        details["missing_dependencies"].append(dep)

                # Critical dependencies must be available
                missing_critical = [dep for dep in details["missing_dependencies"] if dep in critical_deps]
                if missing_critical:
                    details["error"] = f"Missing critical dependencies: {missing_critical}"
                    return False, details
                else:
                    return True, details

            except Exception as e:
                details["error"] = str(e)
                return False, details
        else:
            # Non-Python files - assume valid if they exist
            return True, details

    async def _validate_functionality(
        self,
        project_root: str,
        hook_name: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate basic functionality of the hook."""
        project_path = Path(project_root).resolve()
        hook_path = project_path / ".claude" / "hooks" / "devstream" / hook_name

        details = {
            "hook_path": str(hook_path),
            "functionality_test": "not_applicable",
            "test_result": False
        }

        if not hook_path.exists() or hook_path.suffix != '.py':
            return True, details  # Non-Python files or missing files are handled elsewhere

        try:
            # Basic functionality test - try to import the module
            project_venv = project_path / ".devstream" / "bin" / "python"
            python_exe = project_venv if project_venv.exists() else sys.executable

            # Add hook directory to Python path
            hook_dir = str(hook_path.parent)
            test_script = f"""
import sys
sys.path.insert(0, '{hook_dir}')

try:
    import {hook_path.stem}
    print('SUCCESS: Module imported successfully')
except ImportError as e:
    print(f'IMPORT_ERROR: {{e}}')
except Exception as e:
    print(f'RUNTIME_ERROR: {{e}}')
"""

            result = await asyncio.create_subprocess_exec(
                str(python_exe), "-c", test_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONPATH": hook_dir}
            )

            stdout, stderr = await result.communicate()
            output = stdout.decode('utf-8', errors='ignore')

            details["test_output"] = output
            details["test_stderr"] = stderr.decode('utf-8', errors='ignore')
            details["functionality_test"] = "module_import"

            if "SUCCESS" in output:
                details["test_result"] = True
                return True, details
            else:
                details["test_result"] = False
                details["error"] = "Module import failed"
                return False, details

        except Exception as e:
            details["error"] = str(e)
            return False, details

    async def _validate_integration(
        self,
        project_root: str,
        hook_name: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate hook integration with DevStream system."""
        project_path = Path(project_root).resolve()
        hook_path = project_path / ".claude" / "hooks" / "devstream" / hook_name

        details = {
            "hook_path": str(hook_path),
            "integration_test": "path_validation",
            "project_structure_valid": False,
            "settings_compatible": False
        }

        # Check project structure
        claude_dir = project_path / ".claude"
        hooks_dir = claude_dir / "hooks"
        devstream_hooks = hooks_dir / "devstream"

        details["project_structure_valid"] = all([
            claude_dir.exists(),
            hooks_dir.exists(),
            devstream_hooks.exists(),
            hook_path.exists()
        ])

        # Check settings.json if hook might be referenced there
        settings_file = claude_dir / "settings.json"
        details["settings_exists"] = settings_file.exists()

        if settings_file.exists():
            try:
                settings_content = settings_file.read_text(encoding='utf-8')
                settings = json.loads(settings_content)

                # Check if hook is referenced in settings
                hook_referenced = hook_name in json.dumps(settings)
                details["hook_referenced"] = hook_referenced
                details["settings_compatible"] = True

            except Exception as e:
                details["settings_error"] = str(e)
                details["settings_compatible"] = False
        else:
            details["settings_compatible"] = True  # No settings to check

        # Integration is valid if basic structure is correct
        integration_valid = details["project_structure_valid"]

        return integration_valid, details

    def _extract_python_imports(self, content: str) -> List[str]:
        """Extract import statements from Python code."""
        imports = []
        lines = content.split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('import '):
                module = line.replace('import ', '').split(' as ')[0].split(',')[0].strip()
                imports.append(module)
            elif line.startswith('from '):
                parts = line.split(' import ')
                if len(parts) >= 2:
                    module = parts[0].replace('from ', '').strip()
                    imports.append(module)

        return imports

    def _discover_project_hooks(self, hooks_dir: Path) -> List[str]:
        """Discover all hooks in the project hooks directory."""
        hooks = []

        if not hooks_dir.exists():
            return hooks

        # Find all Python files in hooks directory and subdirectories
        for hook_file in hooks_dir.rglob('*.py'):
            if hook_file.is_file():
                # Get relative path from hooks_dir
                relative_path = hook_file.relative_to(hooks_dir)
                hooks.append(str(relative_path))

        # Also include any executable files
        for hook_file in hooks_dir.rglob('*'):
            if hook_file.is_file() and os.access(hook_file, os.X_OK):
                if hook_file.suffix != '.py':  # Exclude Python files already included
                    relative_path = hook_file.relative_to(hooks_dir)
                    hooks.append(str(relative_path))

        return sorted(hooks)

    async def generate_integrity_report(
        self,
        project_root: str,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive integrity validation report.

        Args:
            project_root: Path to project root
            output_file: Optional file to save report

        Returns:
            Dictionary containing full integrity report
        """
        overall_valid, validation_results = await self.validate_project_hooks(project_root)

        report = {
            "project_root": project_root,
            "validation_timestamp": datetime.now().isoformat(),
            "overall_valid": overall_valid,
            "summary": {
                "total_hooks": len(validation_results),
                "valid_hooks": sum(1 for r in validation_results if r.valid),
                "critical_failures": sum(1 for r in validation_results if r.severity == "critical" and not r.valid),
                "warnings": sum(1 for r in validation_results if r.severity == "warning" and not r.valid)
            },
            "results": [asdict(result) for result in validation_results],
            "recommendations": self._generate_recommendations(validation_results)
        }

        # Save report to file if specified
        if output_file:
            try:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, default=str)

                logger.info("Integrity report saved", output_file=output_file)

            except Exception as e:
                logger.error("Failed to save integrity report", output_file=output_file, error=str(e))

        return report

    def _generate_recommendations(self, validation_results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        critical_failures = [r for r in validation_results if r.severity == "critical" and not r.valid]
        warnings = [r for r in validation_results if r.severity == "warning" and not r.valid]

        if critical_failures:
            recommendations.append("CRITICAL: Fix critical hook failures before proceeding with multi-project deployment")
            recommendations.append(f"  - {len(critical_failures)} hooks have critical issues")

        if warnings:
            recommendations.append(f"WARNING: {len(warnings)} hooks have warning-level issues")

        # Specific recommendations based on common failure patterns
        for result in validation_results:
            if not result.valid:
                if "syntax" in result.message.lower():
                    recommendations.append(f"Fix syntax errors in {result.hook_name}")
                elif "dependencies" in result.message.lower():
                    recommendations.append(f"Install missing dependencies for {result.hook_name}")
                elif "permissions" in result.message.lower():
                    recommendations.append(f"Fix file permissions for {result.hook_name}")

        if not critical_failures and not warnings:
            recommendations.append("All hooks validated successfully - project is ready for multi-project deployment")

        return recommendations


# Convenience function for quick validation
async def validate_project_hooks(project_root: str, devstream_root: str = ".") -> Dict[str, Any]:
    """
    Convenience function for quick hook integrity validation.

    Args:
        project_root: Path to project root to validate
        devstream_root: Path to DevStream framework root

    Returns:
        Dictionary containing validation results
    """
    validator = HookIntegrityValidator(devstream_root)
    return await validator.generate_integrity_report(project_root)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate DevStream hook integrity")
    parser.add_argument("project_root", help="Path to project root")
    parser.add_argument("--devstream-root", default=".", help="Path to DevStream framework root")
    parser.add_argument("--output", help="Output file for integrity report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run validation
    asyncio.run(
        validate_project_hooks(
            project_root=args.project_root,
            devstream_root=args.devstream_root
        )
    )