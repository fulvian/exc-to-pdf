"""
Standalone AI Planning Testing Framework

Self-contained testing framework that doesn't require external dependencies.
For validating core planning logic and validation systems.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .models import TaskBreakdownRequest, AITaskSuggestion
from .validation import AIResponseValidator, FallbackHandler, default_validator, default_fallback_handler

logger = logging.getLogger(__name__)


class TestSeverity(str, Enum):
    """Test severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SimpleTestResult:
    """Simple test result without external dependencies."""
    test_name: str
    passed: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None


class StandaloneTestFramework:
    """
    Standalone testing framework for AI planning validation.

    Tests core validation logic without requiring Ollama or other external services.
    """

    def __init__(self):
        self.validator = AIResponseValidator(strict_mode=False)
        self.fallback_handler = FallbackHandler()
        self.test_results: List[SimpleTestResult] = []

    def run_validation_tests(self) -> Dict[str, Any]:
        """Run standalone validation tests."""
        logger.info("Starting standalone validation tests")
        start_time = datetime.now()

        results = []

        # Test JSON validation
        results.append(self._test_json_validation())

        # Test task breakdown validation
        results.append(self._test_task_breakdown_validation())

        # Test dependency validation
        results.append(self._test_dependency_validation())

        # Test fallback mechanisms
        results.append(self._test_fallback_mechanisms())

        # Test error handling
        results.append(self._test_error_handling())

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        return self._generate_report(results, execution_time)

    def _test_json_validation(self) -> SimpleTestResult:
        """Test JSON validation functionality."""
        start_time = datetime.now()

        try:
            # Test valid JSON
            valid_json = '''
            {
                "tasks": [
                    {
                        "title": "Test task",
                        "description": "Test description",
                        "estimated_minutes": 5,
                        "complexity_score": 3,
                        "priority_score": 5,
                        "task_type": "implementation"
                    }
                ],
                "confidence_score": 0.8
            }
            '''

            result = self.validator.validate_json_response(valid_json)
            assert result.is_valid, f"Valid JSON failed validation: {result.error_message}"

            # Test invalid JSON
            invalid_json = "This is not valid JSON"
            result = self.validator.validate_json_response(invalid_json)
            assert not result.is_valid, "Invalid JSON passed validation"

            execution_time = (datetime.now() - start_time).total_seconds()
            return SimpleTestResult(
                test_name="json_validation",
                passed=True,
                execution_time=execution_time,
                metadata={"assertions_passed": 2}
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return SimpleTestResult(
                test_name="json_validation",
                passed=False,
                error_message=str(e),
                execution_time=execution_time
            )

    def _test_task_breakdown_validation(self) -> SimpleTestResult:
        """Test task breakdown validation."""
        start_time = datetime.now()

        try:
            # Valid task breakdown
            valid_data = {
                "estimated_minutes": 10,
                "complexity_score": 5,
                "confidence_score": 0.8
            }

            result = self.validator.validate_task_breakdown_response(valid_data)
            assert result.is_valid, f"Valid task breakdown failed: {result.error_message}"

            # Invalid task breakdown
            invalid_data = {
                "estimated_minutes": 100,  # Too high
                "complexity_score": 15,    # Too high
                "confidence_score": 1.5    # Too high
            }

            result = self.validator.validate_task_breakdown_response(invalid_data)
            assert not result.is_valid, "Invalid task breakdown passed validation"

            execution_time = (datetime.now() - start_time).total_seconds()
            return SimpleTestResult(
                test_name="task_breakdown_validation",
                passed=True,
                execution_time=execution_time,
                metadata={"assertions_passed": 2}
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return SimpleTestResult(
                test_name="task_breakdown_validation",
                passed=False,
                error_message=str(e),
                execution_time=execution_time
            )

    def _test_dependency_validation(self) -> SimpleTestResult:
        """Test dependency validation."""
        start_time = datetime.now()

        try:
            # Valid dependencies
            valid_data = {
                "dependencies": [
                    {
                        "prerequisite_task": "Task A",
                        "dependent_task": "Task B",
                        "type": "blocking",
                        "reasoning": "B depends on A"
                    }
                ]
            }

            result = self.validator.validate_dependency_response(valid_data)
            assert result.is_valid, f"Valid dependencies failed: {result.error_message}"

            # Invalid dependencies
            invalid_data = {
                "dependencies": "not a list"
            }

            result = self.validator.validate_dependency_response(invalid_data)
            assert not result.is_valid, "Invalid dependencies passed validation"

            execution_time = (datetime.now() - start_time).total_seconds()
            return SimpleTestResult(
                test_name="dependency_validation",
                passed=True,
                execution_time=execution_time,
                metadata={"assertions_passed": 2}
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return SimpleTestResult(
                test_name="dependency_validation",
                passed=False,
                error_message=str(e),
                execution_time=execution_time
            )

    def _test_fallback_mechanisms(self) -> SimpleTestResult:
        """Test fallback mechanisms."""
        start_time = datetime.now()

        try:
            # Test task breakdown fallback
            fallback_result = self.fallback_handler.get_fallback_task_breakdown(
                objective="Implement user authentication",
                context="Web application",
                target_count=3
            )

            assert "tasks" in fallback_result, "Fallback missing tasks"
            assert len(fallback_result["tasks"]) >= 3, "Fallback insufficient tasks"
            assert fallback_result["fallback_used"] == True, "Fallback not marked as used"

            # Test complexity estimation fallback
            complexity_result = self.fallback_handler.get_fallback_complexity_estimation(
                task_title="Complex algorithm implementation",
                task_description="Implement advanced sorting algorithm with optimization"
            )

            assert "estimated_minutes" in complexity_result, "Complexity fallback missing estimation"
            assert "complexity_score" in complexity_result, "Complexity fallback missing score"
            assert complexity_result["fallback_used"] == True, "Complexity fallback not marked"

            execution_time = (datetime.now() - start_time).total_seconds()
            return SimpleTestResult(
                test_name="fallback_mechanisms",
                passed=True,
                execution_time=execution_time,
                metadata={"assertions_passed": 5}
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return SimpleTestResult(
                test_name="fallback_mechanisms",
                passed=False,
                error_message=str(e),
                execution_time=execution_time
            )

    def _test_error_handling(self) -> SimpleTestResult:
        """Test error handling."""
        start_time = datetime.now()

        try:
            # Test validation and fix with invalid input
            invalid_response = "This is completely invalid response text"

            is_valid, parsed_data, error_message = self.validator.validate_and_fix(
                invalid_response, "task_breakdown"
            )

            # Should handle gracefully
            assert isinstance(is_valid, bool), "validate_and_fix should return boolean"
            assert isinstance(parsed_data, dict), "validate_and_fix should return dict"

            if not is_valid:
                assert error_message is not None, "Error message should be provided when invalid"

            execution_time = (datetime.now() - start_time).total_seconds()
            return SimpleTestResult(
                test_name="error_handling",
                passed=True,
                execution_time=execution_time,
                metadata={"assertions_passed": 3}
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return SimpleTestResult(
                test_name="error_handling",
                passed=False,
                error_message=str(e),
                execution_time=execution_time
            )

    def _generate_report(self, results: List[SimpleTestResult], total_time: float) -> Dict[str, Any]:
        """Generate test report."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests

        failed_results = [r for r in results if not r.passed]

        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "total_execution_time": total_time,
                "average_test_time": sum(r.execution_time for r in results) / total_tests if results else 0
            },
            "failures": [
                {
                    "test": r.test_name,
                    "error": r.error_message,
                    "execution_time": r.execution_time
                }
                for r in failed_results
            ],
            "detailed_results": [
                {
                    "test": r.test_name,
                    "passed": r.passed,
                    "error": r.error_message,
                    "execution_time": r.execution_time,
                    "metadata": r.metadata
                }
                for r in results
            ],
            "validation_stats": self.validator.get_validation_stats(),
            "fallback_stats": self.fallback_handler.get_fallback_stats(),
            "timestamp": datetime.now().isoformat()
        }

        return report


def run_standalone_tests() -> Dict[str, Any]:
    """Run standalone tests and return report."""
    framework = StandaloneTestFramework()
    return framework.run_validation_tests()


# Export for testing
__all__ = [
    "StandaloneTestFramework",
    "SimpleTestResult",
    "TestSeverity",
    "run_standalone_tests"
]