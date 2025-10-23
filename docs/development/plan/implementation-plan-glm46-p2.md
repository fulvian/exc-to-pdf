# Implementation Plan: Excel Processing Engine - Core Development

**FOR MODEL**: GLM-4.6 (Tool-Focused, Execution-Optimized)
**Task ID**: `1384fd02-4ae9-4570-9662-27a61462db7e`
**Phase**: implementation
**Priority**: 2/10
**Estimated Duration**: 8 hours

---

## 🎯 EXECUTION PROFILE FOR GLM-4.6

You are an **expert coding agent** specialized in **precise execution** of well-defined tasks.

**YOUR STRENGTHS** (leverage these):
- ✅ Tool calling accuracy 90.6% (best-in-class)
- ✅ Efficient token usage (15% fewer than alternatives)
- ✅ Standard coding patterns excellence
- ✅ Integration with Claude Code ecosystem

**YOUR CONSTRAINTS** (respect these):
- ⚠️ AVOID prolonged reasoning (thinking mode costly - 18K tokens)
- ⚠️ FOCUS on execution over exploration
- ⚠️ FOLLOW provided patterns exactly (framework knowledge gaps)
- ⚠️ CHECK syntax precision (13% error rate - mitigate with type hints)
- ⚠️ COMPLETE micro-tasks fully (no early quit - acceptance criteria mandatory)

---

## 📋 MICRO-TASK BREAKDOWN

```python
TodoWrite([
    {"content": "P2.1 Create ExcelReader core class structure", "status": "pending", "activeForm": "Creating ExcelReader core class structure"},
    {"content": "P2.2 Implement multi-sheet discovery functionality", "status": "pending", "activeForm": "Implementing multi-sheet discovery functionality"},
    {"content": "P2.3 Add read-only mode and memory management", "status": "pending", "activeForm": "Adding read-only mode and memory management"},
    {"content": "P2.4 Create TableDetector hybrid detection system", "status": "pending", "activeForm": "Creating TableDetector hybrid detection system"},
    {"content": "P2.5 Implement DataValidator pipeline", "status": "pending", "activeForm": "Implementing DataValidator pipeline"},
    {"content": "P2.6 Create configuration and error handling", "status": "pending", "activeForm": "Creating configuration and error handling"},
    {"content": "P2.7 Write comprehensive tests", "status": "pending", "activeForm": "Writing comprehensive tests"}
])
```

### Task 1: Create ExcelReader core class structure (Duration: 60 min)

**File**: `src/excel_processor.py` (Lines: 1-100)

**ACTION**: Create the main ExcelReader class with initialization and basic methods

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
class ExcelReader:
    """
    Excel file processing engine with hybrid table detection and multi-sheet support.

    Provides comprehensive Excel file analysis capabilities including sheet discovery,
    table detection, and data extraction optimized for PDF generation.

    Attributes:
        file_path: Path to the Excel file
        config: Configuration object for processing options
        workbook: OpenPyXL workbook instance
        _is_read_only: Flag indicating read-only mode usage
    """

    def __init__(
        self,
        file_path: str,
        config: Optional["ExcelConfig"] = None
    ) -> None:
        """
        Initialize ExcelReader with file path and optional configuration.

        Args:
            file_path: Path to the Excel file (.xlsx format)
            config: Optional configuration object for processing settings

        Raises:
            FileNotFoundError: If the specified file does not exist
            InvalidFileException: If the file is not a valid Excel file

        Example:
            >>> reader = ExcelReader("data.xlsx")
            >>> sheets = reader.discover_sheets()
        """
```

**PATTERN REFERENCE**: See `src/templates/pdf_template.py:1` for similar implementation

**ERROR HANDLING** (USE THIS PATTERN):
```python
try:
    # Implementation
    result = operation()
except SpecificException as e:
    logger.error(
        "Operation failed",
        extra={"context": value, "error": str(e)}
    )
    raise CustomException("User-friendly message") from e
```

**TOOL USAGE**:
1. **Tool**: `mcp__devstream__devstream_search_memory`
   **When**: Before implementing, search for existing patterns
   **Example**:
   ```python
   mcp__devstream__devstream_search_memory(
       query="Excel file processing openpyxl patterns",
       content_type="code",
       limit=5
   )
   ```

2. **Tool**: `mcp__context7__resolve-library-id` + `get-library-docs`
   **When**: Unknown library/pattern encountered
   **Example**:
   ```python
   # Step 1: Resolve
   library_id = mcp__context7__resolve-library-id(libraryName="openpyxl")
   # Step 2: Get docs
   docs = mcp__context7__get-library-docs(
       context7CompatibleLibraryID=library_id,
       topic="workbook initialization read_only mode",
       tokens=3000
   )
   ```

**TEST FILE**: `tests/unit/test_excel_processor.py::test_excel_reader_init`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] Function signature matches exactly
- [ ] Full type hints present
- [ ] Docstring complete with example
- [ ] Error handling implemented
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

**COMPLETION COMMAND**:
```bash
# Run after implementation
.devstream/bin/python -m pytest tests/unit/test_excel_processor.py::test_excel_reader_init -v
.devstream/bin/python -m mypy src/excel_processor.py --strict
```

### Task 2: Implement multi-sheet discovery functionality (Duration: 45 min)

**File**: `src/excel_processor.py` (Lines: 101-150)

**ACTION**: Add methods for discovering and accessing worksheets

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
def discover_sheets(self) -> List[str]:
    """
    Discover all worksheets in the Excel file.

    Returns:
        List of worksheet names in order of appearance

    Raises:
        WorkbookException: If unable to access workbook sheets

    Example:
        >>> reader = ExcelReader("data.xlsx")
        >>> sheets = reader.discover_sheets()
        >>> print(sheets)  # ['Sheet1', 'Sheet2', 'Data']
    """

def extract_sheet_data(self, sheet_name: str) -> "SheetData":
    """
    Extract all data and table information from a specific worksheet.

    Args:
        sheet_name: Name of the worksheet to process

    Returns:
        SheetData object containing tables, metadata, and raw data

    Raises:
        WorksheetNotFoundException: If sheet_name does not exist
        DataExtractionException: If unable to extract sheet data

    Example:
        >>> reader = ExcelReader("data.xlsx")
        >>> sheet_data = reader.extract_sheet_data("Sheet1")
        >>> print(f"Found {len(sheet_data.tables)} tables")
    """
```

**PATTERN REFERENCE**: See `src/templates/pdf_template.py:50` for similar implementation

**ERROR HANDLING** (USE THIS PATTERN):
```python
try:
    # Implementation
    result = operation()
except SpecificException as e:
    logger.error(
        "Operation failed",
        extra={"context": value, "error": str(e)}
    )
    raise CustomException("User-friendly message") from e
```

**TOOL USAGE**:
1. **Tool**: `mcp__devstream__devstream_search_memory`
   **When**: Before implementing, search for existing patterns
   **Example**:
   ```python
   mcp__devstream__devstream_search_memory(
       query="openpyxl worksheet iteration sheet names",
       content_type="code",
       limit=5
   )
   ```

2. **Tool**: `mcp__context7__resolve-library-id` + `get-library-docs`
   **When**: Unknown library/pattern encountered
   **Example**:
   ```python
   # Step 1: Resolve
   library_id = mcp__context7__resolve-library-id(libraryName="openpyxl")
   # Step 2: Get docs
   docs = mcp__context7__get-library-docs(
       context7CompatibleLibraryID=library_id,
       topic="worksheet access sheet names iteration",
       tokens=3000
   )
   ```

**TEST FILE**: `tests/unit/test_excel_processor.py::test_discover_sheets`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] Function signature matches exactly
- [ ] Full type hints present
- [ ] Docstring complete with example
- [ ] Error handling implemented
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

**COMPLETION COMMAND**:
```bash
# Run after implementation
.devstream/bin/python -m pytest tests/unit/test_excel_processor.py::test_discover_sheets -v
.devstream/bin/python -m mypy src/excel_processor.py --strict
```

### Task 3: Add read-only mode and memory management (Duration: 45 min)

**File**: `src/excel_processor.py` (Lines: 151-200)

**ACTION**: Implement memory-efficient processing for large files

**FUNCTION SIGNATURE** (USE EXACTLY):
```python
def _initialize_workbook(self, read_only: bool = True) -> None:
    """
    Initialize the workbook with optimal memory settings.

    Args:
        read_only: Whether to open workbook in read-only mode for memory efficiency

    Raises:
        WorkbookInitializationException: If unable to initialize workbook

    Example:
        >>> reader = ExcelReader("large_file.xlsx")
        >>> # Uses read-only mode by default for memory efficiency
    """

def close(self) -> None:
    """
    Close the workbook and release resources.

    Important for memory management, especially in read-only mode.

    Example:
        >>> reader = ExcelReader("data.xlsx")
        >>> # ... process data ...
        >>> reader.close()  # Release resources
    """
```

**PATTERN REFERENCE**: See `src/templates/pdf_template.py:100` for similar implementation

**ERROR HANDLING** (USE THIS PATTERN):
```python
try:
    # Implementation
    result = operation()
except SpecificException as e:
    logger.error(
        "Operation failed",
        extra={"context": value, "error": str(e)}
    )
    raise CustomException("User-friendly message") from e
```

**TOOL USAGE**:
1. **Tool**: `mcp__devstream__devstream_search_memory`
   **When**: Before implementing, search for existing patterns
   **Example**:
   ```python
   mcp__devstream__devstream_search_memory(
       query="openpyxl read_only mode memory management",
       content_type="code",
       limit=5
   )
   ```

2. **Tool**: `mcp__context7__resolve-library-id` + `get-library-docs`
   **When**: Unknown library/pattern encountered
   **Example**:
   ```python
   # Step 1: Resolve
   library_id = mcp__context7__resolve-library-id(libraryName="openpyxl")
   # Step 2: Get docs
   docs = mcp__context7__get-library-docs(
       context7CompatibleLibraryID=library_id,
       topic="read_only mode large file memory optimization",
       tokens=3000
   )
   ```

**TEST FILE**: `tests/unit/test_excel_processor.py::test_read_only_mode`

**ACCEPTANCE CRITERIA** (CHECK ALL BEFORE MARKING COMPLETE):
- [ ] Function signature matches exactly
- [ ] Full type hints present
- [ ] Docstring complete with example
- [ ] Error handling implemented
- [ ] Test written and passing
- [ ] mypy --strict passes (zero errors)

**COMPLETION COMMAND**:
```bash
# Run after implementation
.devstream/bin/python -m pytest tests/unit/test_excel_processor.py::test_read_only_mode -v
.devstream/bin/python -m mypy src/excel_processor.py --strict
```

---

## 🔍 CONTEXT7 RESEARCH FINDINGS (Pre-Researched)

### OpenPyXL Read-Only Mode Pattern
```python
from openpyxl import load_workbook

# Load large file in read-only mode
wb = load_workbook(filename='large_file.xlsx', read_only=True)
ws = wb['big_data']

# Iterate through rows efficiently
for row in ws.rows:
    for cell in row:
        print(cell.value)

# IMPORTANT: Close the workbook after reading
wb.close()
```
**When to use**: Large files (>10MB) where memory efficiency is critical

### Multi-Sheet Processing Pattern
```python
# Get all worksheet names
sheet_names = wb.sheetnames

# Iterate through worksheets
for sheet in wb:
    print(sheet.title)

# Access worksheet by name
ws = wb["Sheet1"]
```
**When to use**: Processing workbooks with multiple worksheets

### Table Detection Hybrid Approach
**Primary**: openpyxl formal tables (ws.tables.values())
**Secondary**: pandas DataFrame inference (pd.read_excel with chunksize)
**Fallback**: Grid pattern detection (contiguous data blocks)

---

## 🚨 CRITICAL CONSTRAINTS (DO NOT VIOLATE)

**FORBIDDEN ACTIONS**:
- ❌ **NO** feature removal to "fix" problems
- ❌ **NO** workarounds instead of proper solutions
- ❌ **NO** simplifications that reduce functionality
- ❌ **NO** skipping error handling
- ❌ **NO** marking task complete with failing tests

**REQUIRED ACTIONS**:
- ✅ **YES** use Context7 for unknowns (tools provided above)
- ✅ **YES** maintain ALL existing functionality
- ✅ **YES** follow exact error handling pattern
- ✅ **YES** full docstrings + type hints EVERY function
- ✅ **YES** check acceptance criteria per micro-task

---

## ✅ QUALITY GATES (MANDATORY BEFORE COMPLETION)

### 1. Test Coverage
```bash
.devstream/bin/python -m pytest tests/ -v \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=html

# REQUIREMENT: ≥ 95% coverage for NEW code
```

### 2. Type Safety
```bash
.devstream/bin/python -m mypy src/excel_processor.py src/table_detector.py src/data_validator.py --strict

# REQUIREMENT: Zero errors
```

### 3. Performance Benchmark
```bash
# Test with large file (>10MB)
time .devstream/bin/python -c "
from src.excel_processor import ExcelReader
reader = ExcelReader('tests/fixtures/large_test_file.xlsx')
sheets = reader.discover_sheets()
reader.close()
"

# TARGET: <10 seconds for 10MB file
```

---

## 📝 COMMIT MESSAGE TEMPLATE

```
feat(excel): implement core Excel processing engine

Add ExcelReader class with multi-sheet support, read-only mode, and memory management
for efficient processing of Excel files in preparation for PDF generation.

Implementation Details:
- ExcelReader class with hybrid openpyxl + pandas approach
- Multi-sheet discovery and data extraction functionality
- Read-only mode for memory-efficient large file processing
- Comprehensive error handling and resource management

Quality Validation:
- ✅ Tests: 15 tests passing, 98% coverage
- ✅ Type safety: mypy --strict passed
- ✅ Performance: <8s for 10MB file

Task ID: 1384fd02-4ae9-4570-9662-27a61462db7e

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## 📊 SUCCESS METRICS

- **Completion**: 100% of micro-tasks with acceptance criteria met
- **Test Coverage**: ≥ 95% for new code
- **Type Safety**: Zero mypy errors
- **Performance**: Meets/exceeds <10s per 10MB file
- **Code Review**: @code-reviewer validation passed

---

**READY TO START?**
1. Mark first TodoWrite task as "in_progress"
2. Search DevStream memory for context
3. Implement according to specification
4. Run tests + type check
5. Mark "completed" when all acceptance criteria met
6. Proceed to next micro-task

**REMEMBER**: Execute, don't explore. Follow patterns, don't invent. Complete tasks, don't quit early. 🚀