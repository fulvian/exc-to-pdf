"""
Data validation pipeline for Excel processing operations.

This module provides comprehensive data validation functionality including
data type validation, constraint checking, and data quality assessment.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime, date

from .exceptions import DataExtractionException
from .config.excel_config import ExcelConfig, DEFAULT_CONFIG

# Configure logging
logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataType(Enum):
    """Supported data types for validation."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"


@dataclass
class ValidationIssue:
    """
    Information about a validation issue.

    Contains details about validation problems found during data processing.
    """

    level: ValidationLevel
    code: str
    message: str
    row: Optional[int] = None
    column: Optional[str] = None
    value: Optional[Any] = None
    expected_type: Optional[DataType] = None
    constraint: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """
    Results of data validation operations.

    Contains validation summary, issues found, and statistics about the data.
    """

    is_valid: bool
    total_rows: int
    valid_rows: int
    total_cells: int
    valid_cells: int
    issues: List[ValidationIssue]
    data_types: Dict[str, DataType]
    statistics: Dict[str, Any]
    confidence_score: float  # 0.0 to 1.0


@dataclass
class ValidationRule:
    """
    Validation rule configuration.

    Defines how to validate a specific column or data field.
    """

    name: str
    data_type: DataType
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None  # Regex pattern
    allowed_values: Optional[Set[Any]] = None
    custom_validator: Optional[str] = None  # Custom validation function name


class DataValidator:
    """
    Comprehensive data validation pipeline for Excel data.

    Provides validation of data types, constraints, and quality assessment
    for extracted Excel data.
    """

    def __init__(self, config: Optional[ExcelConfig] = None):
        """
        Initialize DataValidator with configuration.

        Args:
            config: Configuration object for validation settings
        """
        self.config = config or DEFAULT_CONFIG

        # Initialize built-in validators
        self._type_validators = {
            DataType.STRING: self._validate_string,
            DataType.INTEGER: self._validate_integer,
            DataType.FLOAT: self._validate_float,
            DataType.BOOLEAN: self._validate_boolean,
            DataType.DATE: self._validate_date,
            DataType.DATETIME: self._validate_datetime,
            DataType.EMAIL: self._validate_email,
            DataType.PHONE: self._validate_phone,
            DataType.URL: self._validate_url,
            DataType.CURRENCY: self._validate_currency,
            DataType.PERCENTAGE: self._validate_percentage,
        }

        logger.debug("DataValidator initialized")

    def validate_table_data(
        self,
        data: List[List[Any]],
        headers: List[str],
        rules: Optional[List[ValidationRule]] = None,
    ) -> ValidationResult:
        """
        Validate table data using provided rules.

        Args:
            data: Table data as list of rows (excluding headers)
            headers: Column headers
            rules: Validation rules for each column

        Returns:
            ValidationResult with validation details

        Raises:
            DataExtractionException: If validation fails critically
        """
        try:
            # Handle edge cases
            if data is None:
                data = []
            if headers is None:
                headers = []

            logger.debug(
                f"Starting validation for table with {len(data)} rows and {len(headers)} columns"
            )

            issues: List[ValidationIssue] = []
            valid_rows = 0
            valid_cells = 0
            total_cells = len(data) * len(headers) if data and headers else 0
            data_types: Dict[str, DataType] = {}
            statistics: Dict[str, Any] = {}

            # Auto-detect data types if rules not provided
            if rules is None:
                rules = self._auto_detect_validation_rules(data, headers)

            # Validate each row
            for row_idx, row in enumerate(data, 1):  # 1-based indexing
                row_valid = True
                for col_idx, (header, value) in enumerate(zip(headers, row)):
                    cell_valid = True
                    rule = rules[col_idx] if col_idx < len(rules) else None

                    if rule:
                        cell_issues = self._validate_cell(value, rule, row_idx, header)
                        issues.extend(cell_issues)
                        if cell_issues:
                            cell_valid = False
                            row_valid = False

                    if cell_valid:
                        valid_cells += 1

                if row_valid:
                    valid_rows += 1

            # Calculate statistics
            data_types = {rule.name: rule.data_type for rule in rules if rule}
            statistics = self._calculate_statistics(data, headers, rules, issues)

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                valid_cells, total_cells, issues
            )

            # Determine overall validity
            # Empty data is valid if there are no error/critical issues
            is_valid = not any(
                issue.level in [ValidationLevel.ERROR, ValidationLevel.CRITICAL]
                for issue in issues
            ) and (
                confidence_score >= 0.8 or total_cells == 0
            )  # Allow empty data

            result = ValidationResult(
                is_valid=is_valid,
                total_rows=len(data),
                valid_rows=valid_rows,
                total_cells=total_cells,
                valid_cells=valid_cells,
                issues=issues,
                data_types=data_types,
                statistics=statistics,
                confidence_score=confidence_score,
            )

            logger.info(
                f"Validation completed: {valid_rows}/{len(data)} rows valid, "
                f"confidence: {confidence_score:.2f}"
            )
            return result

        except Exception as e:
            error_msg = f"Data validation failed: {e}"
            logger.error(error_msg, extra={"error": str(e)})
            raise DataExtractionException(
                error_msg,
                context={
                    "rows": len(data) if data else 0,
                    "columns": len(headers) if headers else 0,
                },
            ) from e

    def _auto_detect_validation_rules(
        self, data: List[List[Any]], headers: List[str]
    ) -> List[ValidationRule]:
        """
        Automatically detect validation rules based on data patterns.

        Args:
            data: Table data
            headers: Column headers

        Returns:
            List of detected validation rules
        """
        if not data or not headers:
            return []

        rules = []
        for col_idx, header in enumerate(headers):
            # Sample values from this column
            column_values = [
                row[col_idx]
                for row in data
                if col_idx < len(row) and row[col_idx] is not None
            ]

            if not column_values:
                # Empty column - default to string type
                rules.append(
                    ValidationRule(
                        name=header, data_type=DataType.STRING, required=False
                    )
                )
                continue

            # Detect data type
            detected_type = self._detect_column_type(column_values)

            # Create rule
            rule = ValidationRule(
                name=header,
                data_type=detected_type,
                required=len(column_values)
                > len(data) * 0.8,  # Required if >80% filled
            )

            # Add constraints based on data
            if detected_type in [DataType.STRING]:
                lengths = [len(str(val)) for val in column_values]
                rule.min_length = min(lengths) if lengths else None
                rule.max_length = max(lengths) if lengths else None

            elif detected_type in [DataType.INTEGER, DataType.FLOAT]:
                numeric_values = [
                    float(val) for val in column_values if self._is_numeric(val)
                ]
                if numeric_values:
                    rule.min_value = min(numeric_values)
                    rule.max_value = max(numeric_values)

            rules.append(rule)

        logger.debug(f"Auto-detected {len(rules)} validation rules")
        return rules

    def _detect_column_type(self, values: List[Any]) -> DataType:
        """
        Detect the most appropriate data type for a column.

        Args:
            values: List of values from the column

        Returns:
            Detected DataType
        """
        if not values:
            return DataType.STRING

        type_scores = {dtype: 0 for dtype in DataType}

        for value in values:
            for dtype in DataType:
                if self._type_validators[dtype](value, check_required=False):
                    type_scores[dtype] += 1

        # Get the type with the highest score
        best_type = max(type_scores.keys(), key=lambda k: type_scores[k])

        # Special pattern detection
        if best_type == DataType.STRING:
            str_values = [str(v) for v in values if v is not None]
            if all(self._is_email(v) for v in str_values if v):
                return DataType.EMAIL
            elif all(self._is_phone(v) for v in str_values if v):
                return DataType.PHONE
            elif all(self._is_url(v) for v in str_values if v):
                return DataType.URL
            elif all(self._is_currency(v) for v in str_values if v):
                return DataType.CURRENCY
            elif all(self._is_percentage(v) for v in str_values if v):
                return DataType.PERCENTAGE

        return best_type

    def _validate_cell(
        self, value: Any, rule: ValidationRule, row: int, column: str
    ) -> List[ValidationIssue]:
        """
        Validate a single cell value against a rule.

        Args:
            value: Cell value to validate
            rule: Validation rule to apply
            row: Row number (1-based)
            column: Column name

        Returns:
            List of validation issues found
        """
        issues = []

        # Check required constraint
        if rule.required and value is None:
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    code="REQUIRED_VALUE_MISSING",
                    message=f"Required value is missing",
                    row=row,
                    column=column,
                    value=value,
                    expected_type=rule.data_type,
                )
            )
            return issues

        # If value is None and not required, skip other validations
        if value is None:
            return issues

        # Validate data type
        if not self._type_validators[rule.data_type](value):
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.ERROR,
                    code="INVALID_DATA_TYPE",
                    message=f"Value is not a valid {rule.data_type.value}",
                    row=row,
                    column=column,
                    value=value,
                    expected_type=rule.data_type,
                )
            )
            return issues

        # Validate constraints
        str_value = str(value)

        # Length constraints
        if rule.min_length is not None and len(str_value) < rule.min_length:
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.WARNING,
                    code="VALUE_TOO_SHORT",
                    message=f"Value is shorter than minimum length {rule.min_length}",
                    row=row,
                    column=column,
                    value=value,
                    constraint=f"min_length={rule.min_length}",
                )
            )

        if rule.max_length is not None and len(str_value) > rule.max_length:
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.WARNING,
                    code="VALUE_TOO_LONG",
                    message=f"Value exceeds maximum length {rule.max_length}",
                    row=row,
                    column=column,
                    value=value,
                    constraint=f"max_length={rule.max_length}",
                )
            )

        # Numeric constraints
        if rule.min_value is not None or rule.max_value is not None:
            try:
                numeric_value = float(
                    str_value.replace(",", "").replace("$", "").replace("%", "")
                )
                if rule.min_value is not None and numeric_value < rule.min_value:
                    issues.append(
                        ValidationIssue(
                            level=ValidationLevel.WARNING,
                            code="VALUE_TOO_SMALL",
                            message=f"Value is less than minimum {rule.min_value}",
                            row=row,
                            column=column,
                            value=value,
                            constraint=f"min_value={rule.min_value}",
                        )
                    )

                if rule.max_value is not None and numeric_value > rule.max_value:
                    issues.append(
                        ValidationIssue(
                            level=ValidationLevel.WARNING,
                            code="VALUE_TOO_LARGE",
                            message=f"Value exceeds maximum {rule.max_value}",
                            row=row,
                            column=column,
                            value=value,
                            constraint=f"max_value={rule.max_value}",
                        )
                    )
            except (ValueError, TypeError):
                pass  # Skip numeric validation if conversion fails

        # Pattern constraint
        if rule.pattern:
            if not re.match(rule.pattern, str_value):
                issues.append(
                    ValidationIssue(
                        level=ValidationLevel.WARNING,
                        code="PATTERN_MISMATCH",
                        message=f"Value does not match required pattern",
                        row=row,
                        column=column,
                        value=value,
                        constraint=f"pattern={rule.pattern}",
                    )
                )

        # Allowed values constraint
        if rule.allowed_values and value not in rule.allowed_values:
            issues.append(
                ValidationIssue(
                    level=ValidationLevel.WARNING,
                    code="VALUE_NOT_ALLOWED",
                    message=f"Value is not in the allowed set",
                    row=row,
                    column=column,
                    value=value,
                    constraint=f"allowed_values={list(rule.allowed_values)}",
                )
            )

        return issues

    def _calculate_statistics(
        self,
        data: List[List[Any]],
        headers: List[str],
        rules: List[ValidationRule],
        issues: List[ValidationIssue],
    ) -> Dict[str, Any]:
        """
        Calculate statistics about the data and validation results.

        Args:
            data: Table data
            headers: Column headers
            rules: Validation rules
            issues: Validation issues found

        Returns:
            Statistics dictionary
        """
        stats: Dict[str, Any] = {
            "row_count": len(data),
            "column_count": len(headers),
            "empty_cells": 0,
            "data_type_distribution": {},
            "issue_distribution": {},
            "completeness_by_column": {},
        }

        # Count issues by level
        issue_dist = stats["issue_distribution"]
        for issue in issues:
            level = issue.level.value
            issue_dist[level] = issue_dist.get(level, 0) + 1

        # Calculate column completeness
        completeness_by_col = stats["completeness_by_column"]
        data_type_dist = stats["data_type_distribution"]

        for col_idx, header in enumerate(headers):
            non_empty_count = sum(
                1 for row in data if col_idx < len(row) and row[col_idx] is not None
            )
            completeness = non_empty_count / len(data) if data else 0
            completeness_by_col[header] = completeness

            if col_idx < len(rules):
                dtype = rules[col_idx].data_type.value
                data_type_dist[dtype] = data_type_dist.get(dtype, 0) + 1

        # Count empty cells
        for row in data:
            for value in row:
                if value is None:
                    stats["empty_cells"] += 1

        return stats

    def _calculate_confidence_score(
        self, valid_cells: int, total_cells: int, issues: List[ValidationIssue]
    ) -> float:
        """
        Calculate overall confidence score for the data.

        Args:
            valid_cells: Number of valid cells
            total_cells: Total number of cells
            issues: Validation issues found

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if total_cells == 0:
            return 0.0

        # Base score from cell validity
        base_score = valid_cells / total_cells

        # Penalty for issues
        error_penalty = (
            sum(1 for issue in issues if issue.level == ValidationLevel.ERROR)
            / total_cells
        )
        warning_penalty = sum(
            1 for issue in issues if issue.level == ValidationLevel.WARNING
        ) / (total_cells * 2)

        # Calculate final score
        confidence_score = max(0.0, base_score - error_penalty - warning_penalty)
        return min(1.0, confidence_score)

    # Built-in type validators
    def _validate_string(self, value: Any, check_required: bool = True) -> bool:
        """Validate string type."""
        if value is None:
            return not check_required
        return isinstance(value, str)

    def _validate_integer(self, value: Any, check_required: bool = True) -> bool:
        """Validate integer type."""
        if value is None:
            return not check_required
        try:
            int(str(value).replace(",", ""))
            return True
        except (ValueError, TypeError):
            return False

    def _validate_float(self, value: Any, check_required: bool = True) -> bool:
        """Validate float type."""
        if value is None:
            return not check_required
        try:
            float(str(value).replace(",", ""))
            return True
        except (ValueError, TypeError):
            return False

    def _validate_boolean(self, value: Any, check_required: bool = True) -> bool:
        """Validate boolean type."""
        if value is None:
            return not check_required
        if isinstance(value, bool):
            return True
        if isinstance(value, str):
            return value.lower() in ["true", "false", "yes", "no", "1", "0"]
        return isinstance(value, (int, float)) and value in [0, 1]

    def _validate_date(self, value: Any, check_required: bool = True) -> bool:
        """Validate date type."""
        if value is None:
            return not check_required
        if isinstance(value, (date, datetime)):
            return True
        if isinstance(value, str):
            try:
                # Try common date formats
                for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]:
                    try:
                        datetime.strptime(value, fmt)
                        return True
                    except ValueError:
                        continue
                return False
            except Exception:
                return False
        return False

    def _validate_datetime(self, value: Any, check_required: bool = True) -> bool:
        """Validate datetime type."""
        if value is None:
            return not check_required
        if isinstance(value, datetime):
            return True
        if isinstance(value, str):
            try:
                # Try common datetime formats
                for fmt in [
                    "%Y-%m-%d %H:%M:%S",
                    "%m/%d/%Y %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%S",
                ]:
                    try:
                        datetime.strptime(value, fmt)
                        return True
                    except ValueError:
                        continue
                return False
            except Exception:
                return False
        return False

    def _validate_email(self, value: Any, check_required: bool = True) -> bool:
        """Validate email type."""
        if value is None:
            return not check_required
        if not isinstance(value, str):
            return False
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(email_pattern, value))

    def _validate_phone(self, value: Any, check_required: bool = True) -> bool:
        """Validate phone number type."""
        if value is None:
            return not check_required
        if not isinstance(value, str):
            return False
        # Remove common formatting characters
        phone = re.sub(r"[\s\-\(\)\+]", "", value)
        phone_pattern = r"^\+?[1-9]\d{6,14}$"  # E.164 format
        return bool(re.match(phone_pattern, phone))

    def _validate_url(self, value: Any, check_required: bool = True) -> bool:
        """Validate URL type."""
        if value is None:
            return not check_required
        if not isinstance(value, str):
            return False
        url_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
        return bool(re.match(url_pattern, value))

    def _validate_currency(self, value: Any, check_required: bool = True) -> bool:
        """Validate currency type."""
        if value is None:
            return not check_required
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            # Remove currency symbols and commas
            clean_value = re.sub(r"[$€£¥\s,]", "", value)
            try:
                float(clean_value)
                return True
            except ValueError:
                return False
        return False

    def _validate_percentage(self, value: Any, check_required: bool = True) -> bool:
        """Validate percentage type."""
        if value is None:
            return not check_required
        if isinstance(value, (int, float)):
            return 0 <= value <= 100
        if isinstance(value, str):
            # Remove % sign
            clean_value = value.replace("%", "").strip()
            try:
                num = float(clean_value)
                return 0 <= num <= 100
            except ValueError:
                return False
        return False

    # Helper methods for pattern detection
    def _is_email(self, value: str) -> bool:
        """Check if value is an email."""
        return self._validate_email(value, check_required=False)

    def _is_phone(self, value: str) -> bool:
        """Check if value is a phone number."""
        return self._validate_phone(value, check_required=False)

    def _is_url(self, value: str) -> bool:
        """Check if value is a URL."""
        return self._validate_url(value, check_required=False)

    def _is_currency(self, value: str) -> bool:
        """Check if value is currency."""
        return self._validate_currency(value, check_required=False)

    def _is_percentage(self, value: str) -> bool:
        """Check if value is percentage."""
        return self._validate_percentage(value, check_required=False)

    def _is_numeric(self, value: Any) -> bool:
        """Check if value can be converted to a number."""
        try:
            float(str(value).replace(",", ""))
            return True
        except (ValueError, TypeError):
            return False
