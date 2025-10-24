#!/bin/bash
# DevStream Hook System Validation Script
# Run this after session restart to validate all hooks

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="$PROJECT_ROOT/.devstream"
LOG_DIR="$HOME/.claude/logs/devstream"

echo "🔍 DevStream Hook System Validation"
echo "===================================="
echo ""

# Check 1: Virtual Environment
echo "✅ Check 1: Virtual Environment"
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ FAIL: Venv not found at $VENV_PATH"
    exit 1
fi
echo "   ✓ Venv exists: $VENV_PATH"

# Check 2: Python Version
echo ""
echo "✅ Check 2: Python Version"
PYTHON_VERSION=$("$VENV_PATH/bin/python" --version 2>&1)
if [[ ! "$PYTHON_VERSION" =~ "Python 3.11" ]]; then
    echo "❌ FAIL: Expected Python 3.11, got $PYTHON_VERSION"
    exit 1
fi
echo "   ✓ $PYTHON_VERSION"

# Check 3: Critical Dependencies
echo ""
echo "✅ Check 3: Critical Dependencies"
REQUIRED_DEPS=("cchooks" "aiohttp" "structlog" "python-dotenv")
for dep in "${REQUIRED_DEPS[@]}"; do
    if ! "$VENV_PATH/bin/python" -m pip list 2>/dev/null | grep -q "$dep"; then
        echo "   ❌ FAIL: Missing dependency: $dep"
        echo "   Run: $VENV_PATH/bin/python -m pip install \"$dep>=0.1.4\""
        exit 1
    fi
    VERSION=$("$VENV_PATH/bin/python" -m pip show "$dep" 2>/dev/null | grep Version | cut -d' ' -f2)
    echo "   ✓ $dep ($VERSION)"
done

# Check 4: Hook Scripts
echo ""
echo "✅ Check 4: Hook Scripts Executable"
HOOK_SCRIPTS=(
    ".claude/hooks/devstream/context/project_context.py"
    ".claude/hooks/devstream/context/user_query_context_enhancer.py"
    ".claude/hooks/devstream/memory/pre_tool_use.py"
    ".claude/hooks/devstream/memory/post_tool_use.py"
    ".claude/hooks/devstream/tasks/stop.py"
)
for script in "${HOOK_SCRIPTS[@]}"; do
    SCRIPT_PATH="$PROJECT_ROOT/$script"
    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "   ❌ FAIL: Script not found: $script"
        exit 1
    fi
    if [ ! -x "$SCRIPT_PATH" ]; then
        echo "   ⚠️  WARN: Script not executable: $script"
        chmod +x "$SCRIPT_PATH"
        echo "   ✓ Made executable: $script"
    else
        echo "   ✓ $(basename "$script")"
    fi
done

# Check 5: Settings Configuration
echo ""
echo "✅ Check 5: Settings Configuration"
SETTINGS_FILE="$PROJECT_ROOT/.claude/settings.json"
if [ ! -f "$SETTINGS_FILE" ]; then
    echo "   ❌ FAIL: Settings file not found"
    exit 1
fi

# Verify settings use venv python
if grep -q "uv run --script" "$SETTINGS_FILE"; then
    echo "   ❌ FAIL: Settings still using 'uv run --script'"
    echo "   Update .claude/settings.json to use .devstream/bin/python"
    exit 1
fi

if grep -q ".devstream/bin/python" "$SETTINGS_FILE"; then
    echo "   ✓ Settings using .devstream/bin/python"
else
    echo "   ⚠️  WARN: Could not verify python path in settings"
fi

# Check hooks configured
REQUIRED_HOOKS=("SessionStart" "UserPromptSubmit" "PreToolUse" "PostToolUse" "SessionEnd")
for hook in "${REQUIRED_HOOKS[@]}"; do
    if grep -q "\"$hook\"" "$SETTINGS_FILE"; then
        echo "   ✓ Hook configured: $hook"
    else
        echo "   ❌ FAIL: Hook not configured: $hook"
        exit 1
    fi
done

# Check 6: Log Directory
echo ""
echo "✅ Check 6: Log Directory"
if [ ! -d "$LOG_DIR" ]; then
    echo "   ⚠️  WARN: Log directory not found, will be created on first run"
    echo "   Expected: $LOG_DIR"
else
    echo "   ✓ Log directory exists"

    # Show recent log files
    echo ""
    echo "   Recent log activity:"
    ls -lt "$LOG_DIR"/*.log 2>/dev/null | head -5 | while read line; do
        echo "     $line"
    done
fi

# Check 7: Database
echo ""
echo "✅ Check 7: DevStream Database"
DB_PATH="$PROJECT_ROOT/data/devstream.db"
if [ ! -f "$DB_PATH" ]; then
    echo "   ❌ FAIL: Database not found at $DB_PATH"
    exit 1
fi
DB_SIZE=$(du -h "$DB_PATH" | cut -f1)
echo "   ✓ Database exists ($DB_SIZE)"

# Summary
echo ""
echo "===================================="
echo "🎉 All Checks Passed!"
echo ""
echo "📋 Next Steps:"
echo "1. Restart Claude Code session"
echo "2. Run validation prompts:"
echo "   - 'Create a file called test_validation.py'"
echo "   - 'How do I use React Query for data fetching?'"
echo "3. Check logs after each test:"
echo "   tail -20 ~/.claude/logs/devstream/pre_tool_use.log"
echo "   tail -20 ~/.claude/logs/devstream/post_tool_use.log"
echo "4. Verify memory storage:"
echo "   sqlite3 $DB_PATH \"SELECT * FROM semantic_memory ORDER BY created_at DESC LIMIT 3;\""
echo ""
echo "🚀 Hook System V2.0 Ready for Testing!"