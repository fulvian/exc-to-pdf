#!/usr/bin/env bash
# DevStream Hook Configuration Script
# Automatically configures ~/.claude/settings.json with absolute paths
# Version: 0.1.0-beta

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLAUDE_CONFIG_DIR="$HOME/.claude"
SETTINGS_FILE="$CLAUDE_CONFIG_DIR/settings.json"

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }

print_header() {
    echo ""
    echo "======================================================"
    echo "  DevStream Hook Configuration"
    echo "  Automatic settings.json Setup"
    echo "======================================================"
    echo ""
}

backup_settings() {
    if [ -f "$SETTINGS_FILE" ]; then
        BACKUP_FILE="$SETTINGS_FILE.backup.$(date +%Y%m%d_%H%M%S)"
        cp "$SETTINGS_FILE" "$BACKUP_FILE"
        log_warning "Existing settings.json backed up to:"
        echo "           $BACKUP_FILE"
        return 0
    fi
    return 1
}

create_settings() {
    log_info "Creating settings.json with hook configuration..."

    # Ensure directory exists
    mkdir -p "$CLAUDE_CONFIG_DIR"

    # Create settings.json with ABSOLUTE paths
    cat > "$SETTINGS_FILE" <<EOF
{
  "hooks": {
    "PreToolUse": [
      {
        "hooks": [
          {
            "command": "\"$PROJECT_ROOT/.devstream/bin/python\" \"$PROJECT_ROOT/.claude/hooks/devstream/memory/pre_tool_use.py\""
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "hooks": [
          {
            "command": "\"$PROJECT_ROOT/.devstream/bin/python\" \"$PROJECT_ROOT/.claude/hooks/devstream/memory/post_tool_use.py\""
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "command": "\"$PROJECT_ROOT/.devstream/bin/python\" \"$PROJECT_ROOT/.claude/hooks/devstream/context/user_query_context_enhancer.py\""
          }
        ]
      }
    ],
    "SessionStart": [
      {
        "hooks": [
          {
            "command": "\"$PROJECT_ROOT/.devstream/bin/python\" \"$PROJECT_ROOT/.claude/hooks/devstream/context/session_start.py\""
          }
        ]
      }
    ]
  }
}
EOF

    log_success "settings.json created at: $SETTINGS_FILE"
}

verify_hooks() {
    log_info "Verifying hook files exist..."

    HOOKS=(
        ".claude/hooks/devstream/memory/pre_tool_use.py"
        ".claude/hooks/devstream/memory/post_tool_use.py"
        ".claude/hooks/devstream/context/user_query_context_enhancer.py"
        ".claude/hooks/devstream/context/session_start.py"
    )

    ALL_OK=true
    for hook in "${HOOKS[@]}"; do
        if [ -f "$PROJECT_ROOT/$hook" ]; then
            chmod +x "$PROJECT_ROOT/$hook"
            log_success "Hook verified: $hook"
        else
            log_error "Hook missing: $hook"
            ALL_OK=false
        fi
    done

    if [ "$ALL_OK" = false ]; then
        log_error "Some hooks are missing. Installation may be incomplete."
        exit 1
    fi
}

verify_venv() {
    log_info "Verifying Python virtual environment..."

    if [ ! -d "$PROJECT_ROOT/.devstream" ]; then
        log_error "Virtual environment not found at $PROJECT_ROOT/.devstream"
        log_error "Please run ./install.sh first"
        exit 1
    fi

    if [ ! -f "$PROJECT_ROOT/.devstream/bin/python" ]; then
        log_error "Python interpreter not found in venv"
        exit 1
    fi

    PYTHON_VERSION=$("$PROJECT_ROOT/.devstream/bin/python" --version 2>&1)
    log_success "Python venv found: $PYTHON_VERSION"
}

print_next_steps() {
    echo ""
    echo "======================================================"
    echo "  ✅ Hook Configuration Complete!"
    echo "======================================================"
    echo ""
    echo "📋 Configuration Details:"
    echo "   → Settings file: $SETTINGS_FILE"
    echo "   → Project root:  $PROJECT_ROOT"
    echo "   → Python venv:   $PROJECT_ROOT/.devstream"
    echo ""
    echo "🔄 CRITICAL NEXT STEP:"
    echo ""
    echo "   ⚠️  RESTART Claude Code now!"
    echo ""
    echo "   Why? Claude Code only loads settings.json at startup."
    echo "   Your hooks will NOT work until you restart."
    echo ""
    echo "📖 After Restart:"
    echo "   → Run: ./scripts/verify-install.py"
    echo "   → Check hook logs in ~/.claude/logs/devstream/"
    echo ""
    echo "======================================================"
    echo ""
}

main() {
    print_header

    log_info "Project root: $PROJECT_ROOT"
    echo ""

    verify_venv
    verify_hooks
    backup_settings
    create_settings

    print_next_steps

    log_success "Hook configuration completed successfully!"
}

main "$@"
