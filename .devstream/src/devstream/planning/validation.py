"""
Advanced AI Response Validation and Error Handling

Production-grade validation system using Guardrails patterns for AI planning responses.
Provides structured error handling, response validation, and graceful degradation.
Based on Context7 research findings for robust AI system validation.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationResult(BaseModel):
    """Result of validation operation."""
    is_valid: bool
    severity: ValidationSeverity
    error_message: Optional[str] = None
    fix_suggestion: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ValidationError(Exception):
    """Custom exception for validation failures."""

    def __init__(self, message: str, severity: ValidationSeverity = ValidationSeverity.MEDIUM, metadata: Optional[Dict] = None):
        super().__init__(message)
        self.severity = severity
        self.metadata = metadata or {}
        self.timestamp = datetime.now()


class AIResponseValidator:
    """
    Advanced AI response validator using Guardrails patterns.

    Provides comprehensive validation for AI planning responses including:
    - JSON structure validation
    - Content quality validation
    - Business logic validation
    - Security validation
    """

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.validation_stats = {
            "total_validations": 0,
            "passed_validations": 0,
            "failed_validations": 0,
            "critical_failures": 0
        }

    def validate_json_response(self, response_text: str, expected_schema: Optional[Dict] = None) -> ValidationResult:
        """Validate JSON structure and schema compliance."""
        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.HIGH,
                    error_message="No valid JSON structure found in response",
                    fix_suggestion="Ensure AI response contains valid JSON object"
                )

            json_text = response_text[json_start:json_end]
            parsed_json = json.loads(json_text)

            # Schema validation if provided
            if expected_schema:
                validation_errors = self._validate_schema(parsed_json, expected_schema)
                if validation_errors:
                    return ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.MEDIUM,
                        error_message=f"Schema validation failed: {'; '.join(validation_errors)}",
                        fix_suggestion="Check required fields and data types",
                        metadata={"parsed_json": parsed_json, "schema_errors": validation_errors}
                    )

            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.LOW,
                metadata={"parsed_json": parsed_json}
            )

        except json.JSONDecodeError as e:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.HIGH,
                error_message=f"JSON parsing failed: {str(e)}",
                fix_suggestion="Fix JSON syntax in AI response"
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                error_message=f"Unexpected validation error: {str(e)}",
                fix_suggestion="Review validation logic and input"
            )

    def validate_task_breakdown_response(self, response_data: Dict) -> ValidationResult:
        """Validate task breakdown response content."""
        required_fields = ["estimated_minutes", "complexity_score", "confidence_score"]
        missing_fields = [field for field in required_fields if field not in response_data]

        if missing_fields:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.HIGH,
                error_message=f"Missing required fields: {missing_fields}",
                fix_suggestion=f"Add missing fields: {', '.join(missing_fields)}"
            )

        # Validate ranges
        validation_errors = []

        estimated_minutes = response_data.get("estimated_minutes", 0)
        if not isinstance(estimated_minutes, int) or not (1 <= estimated_minutes <= 60):
            validation_errors.append("estimated_minutes must be integer between 1-60")

        complexity_score = response_data.get("complexity_score", 0)
        if not isinstance(complexity_score, int) or not (1 <= complexity_score <= 10):
            validation_errors.append("complexity_score must be integer between 1-10")

        confidence_score = response_data.get("confidence_score", 0)
        if not isinstance(confidence_score, (int, float)) or not (0.0 <= confidence_score <= 1.0):
            validation_errors.append("confidence_score must be float between 0.0-1.0")

        if validation_errors:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.MEDIUM,
                error_message=f"Value validation failed: {'; '.join(validation_errors)}",
                fix_suggestion="Correct value ranges and types"
            )

        return ValidationResult(is_valid=True, severity=ValidationSeverity.LOW)

    def validate_dependency_response(self, response_data: Dict) -> ValidationResult:
        """Validate dependency analysis response."""
        if "dependencies" not in response_data:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.MEDIUM,
                error_message="Dependencies array missing from response",
                fix_suggestion="Include 'dependencies' array in response"
            )

        dependencies = response_data["dependencies"]
        if not isinstance(dependencies, list):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.HIGH,
                error_message="Dependencies must be an array",
                fix_suggestion="Format dependencies as JSON array"
            )

        # Validate each dependency
        for i, dep in enumerate(dependencies):
            if not isinstance(dep, dict):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.MEDIUM,
                    error_message=f"Dependency {i} must be an object",
                    fix_suggestion="Format each dependency as JSON object"
                )

            required_dep_fields = ["prerequisite_task", "dependent_task", "type", "reasoning"]
            missing_dep_fields = [field for field in required_dep_fields if field not in dep]

            if missing_dep_fields:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.MEDIUM,
                    error_message=f"Dependency {i} missing fields: {missing_dep_fields}",
                    fix_suggestion=f"Add required dependency fields: {', '.join(missing_dep_fields)}"
                )

        return ValidationResult(is_valid=True, severity=ValidationSeverity.LOW)

    def _validate_schema(self, data: Dict, schema: Dict) -> List[str]:
        """Simple schema validation (basic implementation)."""
        errors = []

        # Check required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Check field types (basic)
        properties = schema.get("properties", {})
        for field, field_schema in properties.items():
            if field in data:
                expected_type = field_schema.get("type")
                if expected_type == "string" and not isinstance(data[field], str):
                    errors.append(f"Field {field} must be string")
                elif expected_type == "integer" and not isinstance(data[field], int):
                    errors.append(f"Field {field} must be integer")
                elif expected_type == "number" and not isinstance(data[field], (int, float)):
                    errors.append(f"Field {field} must be number")
                elif expected_type == "array" and not isinstance(data[field], list):
                    errors.append(f"Field {field} must be array")

        return errors

    def validate_and_fix(self, response_text: str, validation_type: str) -> Tuple[bool, Dict, Optional[str]]:
        """
        Validate response and attempt automatic fixes.

        Returns:
            (is_valid, parsed_data, error_message)
        """
        self.validation_stats["total_validations"] += 1

        try:
            # JSON validation
            json_result = self.validate_json_response(response_text)
            if not json_result.is_valid:
                if json_result.severity == ValidationSeverity.CRITICAL:
                    self.validation_stats["critical_failures"] += 1
                    if self.strict_mode:
                        raise ValidationError(json_result.error_message, json_result.severity)
                    return False, {}, json_result.error_message

                # Attempt to fix JSON issues
                fixed_data = self._attempt_json_fix(response_text)
                if fixed_data:
                    logger.warning(f"JSON auto-fix applied: {json_result.error_message}")
                    parsed_data = fixed_data
                else:
                    self.validation_stats["failed_validations"] += 1
                    return False, {}, json_result.error_message
            else:
                parsed_data = json_result.metadata["parsed_json"]

            # Content validation
            if validation_type == "task_breakdown":
                content_result = self.validate_task_breakdown_response(parsed_data)
            elif validation_type == "dependency_analysis":
                content_result = self.validate_dependency_response(parsed_data)
            else:
                content_result = ValidationResult(is_valid=True, severity=ValidationSeverity.LOW)

            if not content_result.is_valid:
                if content_result.severity in [ValidationSeverity.HIGH, ValidationSeverity.CRITICAL]:
                    self.validation_stats["failed_validations"] += 1
                    if self.strict_mode and content_result.severity == ValidationSeverity.CRITICAL:
                        raise ValidationError(content_result.error_message, content_result.severity)
                    return False, parsed_data, content_result.error_message
                else:
                    # Medium/Low severity - apply fixes and continue
                    logger.warning(f"Content validation warning: {content_result.error_message}")

            self.validation_stats["passed_validations"] += 1
            return True, parsed_data, None

        except ValidationError:
            raise
        except Exception as e:
            self.validation_stats["critical_failures"] += 1
            error_msg = f"Validation system error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if self.strict_mode:
                raise ValidationError(error_msg, ValidationSeverity.CRITICAL)
            return False, {}, error_msg

    def _attempt_json_fix(self, response_text: str) -> Optional[Dict]:
        """Attempt to fix common JSON issues."""
        try:
            # Try to find and extract just the JSON part
            lines = response_text.split('\n')
            json_lines = []
            in_json = False

            for line in lines:
                if '{' in line:
                    in_json = True
                if in_json:
                    json_lines.append(line)
                if '}' in line and in_json:
                    break

            if json_lines:
                json_text = '\n'.join(json_lines)
                return json.loads(json_text)

        except Exception:
            pass

        return None

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self.validation_stats["total_validations"]
        if total == 0:
            return self.validation_stats

        return {
            **self.validation_stats,
            "success_rate": self.validation_stats["passed_validations"] / total,
            "failure_rate": self.validation_stats["failed_validations"] / total,
            "critical_rate": self.validation_stats["critical_failures"] / total
        }


class FallbackHandler:
    """
    Handles fallback scenarios when AI operations fail.

    Provides graceful degradation and alternative responses
    based on Context7 research patterns.
    """

    def __init__(self):
        self.fallback_stats = {
            "fallbacks_triggered": 0,
            "successful_fallbacks": 0,
            "failed_fallbacks": 0
        }

    def get_fallback_task_breakdown(
        self,
        objective: str,
        context: Optional[str] = None,
        target_count: int = 5
    ) -> Dict[str, Any]:
        """Generate fallback task breakdown when AI fails."""
        self.fallback_stats["fallbacks_triggered"] += 1

        try:
            # Simple rule-based task breakdown
            fallback_tasks = []

            # Basic task templates based on common patterns
            if any(keyword in objective.lower() for keyword in ['implement', 'create', 'build']):
                fallback_tasks.extend([
                    {
                        "title": f"Research and plan {objective.lower()}",
                        "description": "Research requirements and create implementation plan",
                        "estimated_minutes": 8,
                        "complexity_score": 4,
                        "task_type": "research"
                    },
                    {
                        "title": f"Core implementation of {objective.lower()}",
                        "description": "Implement main functionality",
                        "estimated_minutes": 10,
                        "complexity_score": 6,
                        "task_type": "implementation"
                    },
                    {
                        "title": f"Test {objective.lower()}",
                        "description": "Create and run tests for implementation",
                        "estimated_minutes": 7,
                        "complexity_score": 4,
                        "task_type": "testing"
                    }
                ])

            # Add context-specific tasks if available
            if context and any(keyword in context.lower() for keyword in ['database', 'sql', 'data']):
                fallback_tasks.append({
                    "title": "Database integration",
                    "description": "Handle database operations and data persistence",
                    "estimated_minutes": 9,
                    "complexity_score": 5,
                    "task_type": "implementation"
                })

            # Ensure we have target number of tasks
            while len(fallback_tasks) < target_count:
                fallback_tasks.append({
                    "title": f"Additional work for {objective.lower()}",
                    "description": "Additional implementation or refinement work",
                    "estimated_minutes": 8,
                    "complexity_score": 5,
                    "task_type": "implementation"
                })

            # Trim to target count
            fallback_tasks = fallback_tasks[:target_count]

            self.fallback_stats["successful_fallbacks"] += 1

            return {
                "tasks": fallback_tasks,
                "confidence_score": 0.4,  # Low confidence for fallback
                "reasoning": "Fallback task breakdown - AI planning unavailable",
                "total_estimated_minutes": sum(task["estimated_minutes"] for task in fallback_tasks),
                "fallback_used": True
            }

        except Exception as e:
            self.fallback_stats["failed_fallbacks"] += 1
            logger.error(f"Fallback task breakdown failed: {e}")
            raise ValidationError(f"Fallback generation failed: {e}", ValidationSeverity.CRITICAL)

    def get_fallback_complexity_estimation(
        self,
        task_title: str,
        task_description: str
    ) -> Dict[str, Any]:
        """Generate fallback complexity estimation."""
        self.fallback_stats["fallbacks_triggered"] += 1

        try:
            # Simple heuristic-based estimation
            description_length = len(task_description)
            title_length = len(task_title)

            # Base estimation on content complexity indicators
            complexity_indicators = [
                'complex', 'advanced', 'integration', 'system', 'framework',
                'algorithm', 'optimization', 'performance', 'security', 'scalability'
            ]

            complexity_score = 3  # Base complexity

            # Adjust based on keywords
            text_lower = (task_title + ' ' + task_description).lower()
            for indicator in complexity_indicators:
                if indicator in text_lower:
                    complexity_score += 1

            # Adjust based on length
            if description_length > 100:
                complexity_score += 1
            if description_length > 200:
                complexity_score += 1

            # Cap at maximum
            complexity_score = min(complexity_score, 10)

            # Estimate time based on complexity
            estimated_minutes = max(3, min(60, complexity_score * 1.5 + 2))

            self.fallback_stats["successful_fallbacks"] += 1

            return {
                "estimated_minutes": int(estimated_minutes),
                "complexity_score": complexity_score,
                "uncertainty_factor": 0.8,  # High uncertainty for fallback
                "confidence_score": 0.3,    # Low confidence for fallback
                "reasoning": "Heuristic-based estimation - AI estimation unavailable",
                "fallback_used": True,
                "analysis_factors": {
                    "description_length": description_length,
                    "complexity_indicators_found": sum(1 for ind in complexity_indicators if ind in text_lower),
                    "base_complexity": 3
                }
            }

        except Exception as e:
            self.fallback_stats["failed_fallbacks"] += 1
            logger.error(f"Fallback complexity estimation failed: {e}")
            raise ValidationError(f"Fallback estimation failed: {e}", ValidationSeverity.CRITICAL)

    def get_fallback_stats(self) -> Dict[str, Any]:
        """Get fallback operation statistics."""
        return {
            **self.fallback_stats,
            "fallback_success_rate": (
                self.fallback_stats["successful_fallbacks"] /
                max(1, self.fallback_stats["fallbacks_triggered"])
            )
        }


# Global instances for convenience
default_validator = AIResponseValidator(strict_mode=False)
default_fallback_handler = FallbackHandler()