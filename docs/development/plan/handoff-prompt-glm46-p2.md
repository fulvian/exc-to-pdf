# 🚀 DevStream Task Handoff: Excel Processing Engine - Core Development

**FROM**: Claude Sonnet 4.5 (Strategic Planning Complete)
**TO**: GLM-4.6 (Implementation Execution)

---

## 📊 TASK CONTEXT

**Task ID**: `1384fd02-4ae9-4570-9662-27a61462db7e`
**Phase**: implementation
**Priority**: 2/10
**Status**: Steps 1-5 COMPLETED by Sonnet 4.5 → Steps 6-7 DELEGATED to you

**Your Role**: You are an **expert execution-focused coding agent**. Sonnet 4.5 has completed all strategic planning. Your job is **precise implementation** according to the approved plan.

---

## ✅ WORK COMPLETED (Steps 1-5)

- ✅ **DISCUSSION**: Problem analyzed, trade-offs identified, approach agreed
- ✅ **ANALYSIS**: Codebase patterns identified, files to modify determined
- ✅ **RESEARCH**: Context7 findings documented (see below)
- ✅ **PLANNING**: Detailed implementation plan created (see linked file)
- ✅ **APPROVAL**: User approved plan, ready for execution

---

## 📋 YOUR IMPLEMENTATION PLAN

**COMPLETE PLAN**: `/Users/fulvioventura/exc-to-pdf/implementation-plan-glm46-p2.md`

**READ THE PLAN FIRST** using:
```bash
cat /Users/fulvioventura/exc-to-pdf/implementation-plan-glm46-p2.md
```

**Plan Summary** (excerpt):
Implement ExcelReader class with multi-sheet discovery, read-only memory management, and hybrid table detection for Excel to PDF conversion pipeline.

---

## 🎯 YOUR MISSION (Steps 6-7)

### Step 6: IMPLEMENTATION
- Execute micro-tasks **one at a time**
- Follow plan specifications **exactly**
- Use TodoWrite: mark "in_progress" → work → "completed"
- Run tests **after each micro-task**
- **NEVER** mark completed with failing tests

### Step 7: VERIFICATION
- **95%+ test coverage** for all new code
- **mypy --strict** zero errors
- **Performance validation** (<10s per 10MB file)
- **@code-reviewer** validation (automatic on commit)

---

## 🔧 DEVSTREAM PROTOCOL COMPLIANCE (MANDATORY)

**CRITICAL RULES** (from @CLAUDE.md):

### Python Environment
```bash
# ALWAYS use .devstream venv
.devstream/bin/python script.py       # ✅ CORRECT
.devstream/bin/python -m pytest       # ✅ CORRECT
python script.py                       # ❌ FORBIDDEN
```

### TodoWrite Workflow
1. Mark first task "in_progress"
2. Implement according to plan
3. Run tests
4. Mark "completed" ONLY when:
   - Tests pass 100%
   - Type check passes
   - Acceptance criteria met
5. Proceed to next task

### Context7 Usage
```python
# When you encounter unknowns
library_id = mcp__context7__resolve-library-id(libraryName="openpyxl")
docs = mcp__context7__get-library-docs(
    context7CompatibleLibraryID=library_id,
    topic="read_only mode large file processing",
    tokens=3000
)
```

### Memory Search
```python
# Before implementing, search for existing patterns
mcp__devstream__devstream_search_memory(
    query="Excel processing openpyxl patterns",
    content_type="code",
    limit=5
)
```

---

## 📚 CONTEXT7 RESEARCH (Pre-Completed by Sonnet)

### Libraries Researched:
- **openpyxl**: Excel file reading and table detection (Trust Score: 7.5)
- **pandas**: Data processing and chunking (Trust Score: 9.2)
- **reportlab**: PDF generation for later phases (Trust Score: 7.5)

### Key Findings:
- Use openpyxl read-only mode for memory efficiency with large files
- Hybrid table detection: formal tables → pandas inference → grid patterns
- Chunking pattern for unlimited file size processing
- Context manager pattern for resource management

### Pattern Examples:
```python
# Read-only mode for large files
wb = load_workbook(filename='large_file.xlsx', read_only=True)
# Process data
wb.close()  # CRITICAL: release resources

# Multi-sheet discovery
sheet_names = wb.sheetnames
for sheet in wb:
    print(sheet.title)

# Hybrid table detection
formal_tables = ws.tables.values()
pandas_tables = pd.read_excel(chunksize=1000)
```

**When to use**: Large files, multi-sheet workbooks, memory-constrained environments

---

## 🏗️ TECHNICAL SPECIFICATIONS

**Files to Modify**:
- `src/excel_processor.py` (new file)
- `tests/unit/test_excel_processor.py` (new file)

**New Files to Create**:
- `src/table_detector.py` (later phase)
- `src/data_validator.py` (later phase)
- `src/exceptions.py` (custom exceptions)
- `src/config/excel_config.py` (configuration)

**Dependencies** (already in requirements.txt):
- openpyxl>=3.1.0
- pandas>=2.0.0
- pytest>=7.0.0

---

## 🚨 CRITICAL CONSTRAINTS (DO NOT VIOLATE)

**FORBIDDEN ACTIONS**:
- ❌ **NO** removal of features (find proper solution instead)
- ❌ **NO** workarounds (implement correctly using Context7)
- ❌ **NO** simplifications that reduce functionality
- ❌ **NO** skipping tests or type hints
- ❌ **NO** early quit on complex tasks (complete fully)

**REQUIRED ACTIONS**:
- ✅ **YES** use `.devstream/bin/python` for ALL commands
- ✅ **YES** follow TodoWrite plan strictly
- ✅ **YES** use Context7 for unknowns (tools provided)
- ✅ **YES** maintain ALL existing functionality
- ✅ **YES** full type hints + docstrings EVERY function
- ✅ **YES** tests for EVERY feature (95%+ coverage)

---

## ✅ QUALITY GATES (Check Before Completion)

### 1. Environment Verification
```bash
# Verify venv and Python version
.devstream/bin/python --version  # Must be 3.11.x
.devstream/bin/python -m pip list | grep -E "(openpyxl|pandas|pytest)"
```

### 2. Implementation
Follow plan in `/Users/fulvioventura/exc-to-pdf/implementation-plan-glm46-p2.md`

### 3. Testing
```bash
# After EVERY micro-task
.devstream/bin/python -m pytest tests/unit/test_excel_processor.py -v
.devstream/bin/python -m mypy src/excel_processor.py --strict

# Before completion (ALL tests)
.devstream/bin/python -m pytest tests/ -v \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=html

# REQUIREMENT: ≥95% coverage, 100% pass rate
```

### 4. Commit (if all tests pass)
```bash
git add src/excel_processor.py tests/unit/test_excel_processor.py
git commit -m "$(cat <<'EOF'
feat(excel): implement core Excel processing engine

Add ExcelReader class with multi-sheet support, read-only mode, and memory management
for efficient processing of Excel files in preparation for PDF generation.

Task ID: 1384fd02-4ae9-4570-9662-27a61462db7e

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

**Note**: @code-reviewer validation automatic on commit

---

## 🔍 DEVSTREAM MEMORY ACCESS

Search for relevant context anytime:
```python
mcp__devstream__devstream_search_memory(
    query="Excel processing engine implementation",
    content_type="code",
    limit=10
)
```

---

## 📊 SUCCESS CRITERIA

- [ ] All TodoWrite tasks completed
- [ ] Tests pass 100%
- [ ] Coverage ≥ 95%
- [ ] mypy --strict passes (zero errors)
- [ ] Performance meets target: <10s per 10MB file
- [ ] @code-reviewer validation passed
- [ ] All acceptance criteria met

---

## 🚀 EXECUTION CHECKLIST

1. [ ] **READ** the complete plan: `cat /Users/fulvioventura/exc-to-pdf/implementation-plan-glm46-p2.md`
2. [ ] **VERIFY** environment: `.devstream/bin/python --version`
3. [ ] **SEARCH** DevStream memory for context
4. [ ] **START** first TodoWrite task (mark "in_progress")
5. [ ] **IMPLEMENT** according to plan specifications
6. [ ] **TEST** after each micro-task
7. [ ] **COMPLETE** task when all criteria met
8. [ ] **REPEAT** steps 4-7 for remaining tasks
9. [ ] **VALIDATE** complete implementation (all quality gates)
10. [ ] **COMMIT** if all tests pass

---

**READY TO IMPLEMENT?**

Start with the first TodoWrite task. Execute precisely. Test thoroughly. Complete fully. 🚀

**Remember**: You are GLM-4.6 - your strength is **precise execution** of well-defined tasks. The strategic thinking is done. Now execute flawlessly. 💪

---

## 🎯 YOUR FIRST ACTION

1. **READ** the implementation plan:
   ```bash
   cat /Users/fulvioventura/exc-to-pdf/implementation-plan-glm46-p2.md
   ```

2. **SETUP** your first TodoWrite task:
   ```python
   TodoWrite([
       {"content": "P2.1 Create ExcelReader core class structure", "status": "in_progress", "activeForm": "Creating ExcelReader core class structure"},
       # ... other tasks
   ])
   ```

3. **SEARCH** for existing patterns:
   ```python
   mcp__devstream__devstream_search_memory(
       query="Excel file processing openpyxl patterns",
       content_type="code",
       limit=5
   )
   ```

4. **IMPLEMENT** the first micro-task following the exact specification in the plan.

**GLM-4.6, you are cleared for implementation! Execute with precision! 🚀**