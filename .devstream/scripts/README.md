# DevStream Scripts

Utility scripts for DevStream installation, verification, and maintenance.

## verify-install.py

Comprehensive validation script for DevStream installation completeness.

### Purpose

Verifies that all DevStream components are properly installed and configured:

- **Python Environment**: Virtual environment (.devstream), Python 3.11+, critical packages
- **Database**: Schema integrity, tables, triggers, indexes, virtual tables
- **MCP Server**: Build artifacts, Node.js dependencies, configuration
- **Hook Configuration**: settings.json configuration, hook script files
- **Optional Components**: Ollama, Git repository, environment file

### Usage

```bash
# Basic verification
.devstream/bin/python scripts/verify-install.py

# Verbose output with detailed checks
.devstream/bin/python scripts/verify-install.py --verbose

# JSON output (for automation/CI)
.devstream/bin/python scripts/verify-install.py --json

# Skip optional checks (Ollama, Git, etc.)
.devstream/bin/python scripts/verify-install.py --skip-optional

# Attempt auto-fix (not yet implemented)
.devstream/bin/python scripts/verify-install.py --fix
```

### Exit Codes

- **0**: All checks passed (installation complete)
- **1**: Warnings present (installation may work with limitations)
- **2**: Critical failures (installation incomplete)

### Output Formats

**Rich Terminal Output** (default):
- Colored table with check results
- Detailed failure/warning sections
- Summary statistics
- Actionable remediation instructions

**JSON Output** (--json flag):
```json
{
  "summary": {
    "PASS": 14,
    "WARNING": 3,
    "FAIL": 4,
    "SKIP": 0
  },
  "results": [
    {
      "category": "Python Environment",
      "name": "Virtual Environment",
      "status": "PASS",
      "message": "Virtual environment exists",
      "details": "Location: /path/to/.devstream/bin/python",
      "remediation": null
    }
  ]
}
```

### Check Categories

#### Python Environment (6 checks)
- Virtual environment existence (.devstream/bin/python)
- Python version (must be 3.11.x)
- Critical packages:
  - cchooks >= 0.1.4
  - aiohttp >= 3.8.0
  - structlog >= 23.0.0
  - python-dotenv >= 1.0.0

#### Database (6 checks)
- Database file existence and permissions
- Core tables (14 expected)
- Triggers (3 expected: sync_insert_memory, sync_update_memory, sync_delete_memory)
- Indexes (37 expected)
- Virtual tables (vec_semantic_memory, fts_semantic_memory)
- Schema version (v2.1.0)

#### MCP Server (4 checks)
- MCP directory existence
- Build artifacts (dist/index.js)
- Node.js dependencies (node_modules/)
- Package configuration (package.json)

#### Hook Configuration (2 checks)
- Settings file (~/.claude/settings.json)
- Hook configuration:
  - PreToolUse hook
  - PostToolUse hook
  - UserPromptSubmit hook
  - Correct Python interpreter (.devstream/bin/python)
  - Script file existence

#### Optional Components (3 checks)
- Ollama service (http://localhost:11434)
- nomic-embed-text model
- Git repository initialization
- .env.devstream file

### Dependencies

**Required**:
- Python 3.11+ (from .devstream venv)
- rich (for terminal output)
- python-dotenv (for environment loading)

**Optional**:
- sqlite3 module (standard library)
- json module (standard library)
- urllib module (standard library)

### Troubleshooting

**Script fails to run**:
```bash
# Ensure venv exists
python3.11 -m venv .devstream

# Install dependencies
.devstream/bin/python -m pip install rich python-dotenv

# Make script executable
chmod +x scripts/verify-install.py
```

**Database checks fail**:
```bash
# Recreate database from schema
sqlite3 data/devstream.db < schema/schema.sql
```

**MCP server checks fail**:
```bash
# Rebuild MCP server
cd mcp-devstream-server
npm install
npm run build
```

**Hook checks fail**:
```bash
# Verify settings.json exists
ls -la ~/.claude/settings.json

# Check hook scripts exist
ls -la .claude/hooks/devstream/memory/*.py
ls -la .claude/hooks/devstream/context/*.py
```

### Integration with CI/CD

Example GitHub Actions workflow:

```yaml
- name: Verify DevStream Installation
  run: |
    .devstream/bin/python scripts/verify-install.py --json > verification.json
    cat verification.json
    exit_code=$?
    if [ $exit_code -eq 2 ]; then
      echo "❌ Critical failures detected"
      exit 1
    elif [ $exit_code -eq 1 ]; then
      echo "⚠️  Warnings present"
    else
      echo "✅ All checks passed"
    fi
```

### Future Enhancements

- **Auto-fix mode**: Automatically remediate common issues
- **Performance benchmarks**: Test query performance, embedding generation speed
- **Hook execution test**: Verify hooks execute without errors
- **MCP server connectivity**: Test MCP server responds to requests
- **Security audit**: Check file permissions, sensitive data exposure

### Related Documentation

- [DevStream Installation Guide](../docs/installation.md)
- [Database Schema](../schema/schema.sql)
- [MCP Server Documentation](../mcp-devstream-server/README.md)
- [Hook System Guide](../docs/hooks.md)

---

## setup-db.py

**Purpose**: Initialize SQLite database from schema.sql with comprehensive validation.

### Features

- **Schema Loading**: Reads and executes `schema/schema.sql`
- **sqlite-vec Extension**: Automatic detection and loading (optional)
- **Comprehensive Verification**: 12 core tables, 2 virtual tables, 3 triggers, 37 indexes
- **Error Handling**: Graceful handling of missing extensions, rollback on failure
- **CLI Options**: Force overwrite, verbose output, custom paths

### Usage

```bash
# Basic setup (default paths)
.devstream/bin/python scripts/setup-db.py

# Force overwrite without confirmation
.devstream/bin/python scripts/setup-db.py --force --verbose

# Custom paths
.devstream/bin/python scripts/setup-db.py \
  --schema-file /path/to/custom.sql \
  --db-path /path/to/custom.db \
  --force
```

### CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--force` | Force overwrite existing database | `False` |
| `--verbose`, `-v` | Detailed validation report | `False` |
| `--schema-file` | Path to schema.sql | `schema/schema.sql` |
| `--db-path` | Path to database file | `data/devstream.db` |
| `--json` | JSON logging output | `False` |

### Exit Codes

- `0`: Success (database created and validated)
- `1`: Error (schema load/execution failed)

### Expected Database Structure

**Core Tables (12)**: intervention_plans, phases, micro_tasks, semantic_memory, agents, hooks, work_sessions, context_injections, hook_executions, learning_insights, performance_metrics, schema_version

**Virtual Tables (2)**: vec_semantic_memory (vec0, optional), fts_semantic_memory (FTS5, required)

**Triggers (3)**: sync_insert_memory, sync_update_memory, sync_delete_memory

**Indexes (37)**: Performance optimization across all tables

### Type Safety

✅ Passes `mypy --strict` validation with full type hints and docstrings.

### Testing

```bash
.devstream/bin/python -m pytest tests/test_setup_db.py -v
```

### Related Files

- **Schema**: `schema/schema.sql`
- **Tests**: `tests/test_setup_db.py`
- **Database**: `data/devstream.db`
