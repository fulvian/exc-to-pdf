#!/bin/bash

# DevStream Claude Code Launcher for Z.AI GLM-4.6
# Modular script to launch Claude Code with GLM-4.6 model via Z.AI API
# Follows Context7 best practices for environment variable management

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_status() {
  echo -e "${GREEN}[STATUS]${NC} $1"
}

print_info() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
  echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Function to find DevStream project root using upward search
find_devstream_project_root() {
    local start_dir="${1:-$(pwd)}"
    local current_dir="$start_dir"
    local max_depth=20
    local depth=0

    # Upward search for .env.devstream marker
    while [ "$current_dir" != "/" ] && [ $depth -lt $max_depth ]; do
        if [ -f "$current_dir/.env.devstream" ]; then
            # Validate complete DevStream installation
            if [ -d "$current_dir/.claude/hooks/devstream" ] && \
               [ -d "$current_dir/.devstream" ]; then
                echo "$current_dir"
                return 0
            fi
        fi

        current_dir="$(dirname "$current_dir")"
        depth=$((depth + 1))
    done

    return 1  # Not found
}

# Function to validate DevStream project installation
validate_devstream_project() {
    local project_root="$1"
    local validation_failed=0

    print_status "Validating DevStream project at: $project_root"

    # Check required directories
    if [ ! -d "$project_root/.claude" ]; then
        print_error "Missing: .claude directory"
        validation_failed=1
    fi

    if [ ! -d "$project_root/.claude/hooks/devstream" ]; then
        print_error "Missing: .claude/hooks/devstream directory"
        validation_failed=1
    fi

    if [ ! -d "$project_root/.devstream" ]; then
        print_error "Missing: .devstream virtual environment"
        validation_failed=1
    fi

    # Check required files
    if [ ! -f "$project_root/.env.devstream" ]; then
        print_error "Missing: .env.devstream configuration"
        validation_failed=1
    fi

    if [ ! -f "$project_root/data/devstream.db" ]; then
        print_warning "Missing: data/devstream.db (will be created)"
    fi

    # Check Python version in venv
    if [ -x "$project_root/.devstream/bin/python" ]; then
        local python_version=$("$project_root/.devstream/bin/python" --version 2>&1 | awk '{print $2}')
        if [[ ! "$python_version" =~ ^3\.11\. ]]; then
            print_error "Wrong Python version: $python_version (expected 3.11.x)"
            validation_failed=1
        fi
    else
        print_error "Python interpreter not found in .devstream/bin/"
        validation_failed=1
    fi

    if [ $validation_failed -eq 1 ]; then
        print_error ""
        print_error "DevStream installation incomplete or corrupted"
        print_error "Run: $DEVSTREAM_ROOT/scripts/install-devstream.sh"
        return 1
    fi

    print_status "✅ DevStream project validation passed"
    return 0
}

# Function to display project information
print_project_info() {
    local project_root="$1"
    local project_name=$(basename "$project_root")
    local db_size="N/A"

    if [ -f "$project_root/data/devstream.db" ]; then
        db_size=$(du -h "$project_root/data/devstream.db" | awk '{print $1}')
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🚀 DevStream Project Detected"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "   Project Name:  $project_name"
    echo "   Location:      $project_root"
    echo "   Database:      data/devstream.db ($db_size)"
    echo "   Python Venv:   .devstream/"
    echo "   Python:        $($project_root/.devstream/bin/python --version 2>&1)"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# Get framework location (where DevStream is installed)
DEVSTREAM_SCRIPT_DIR="$SCRIPT_DIR"
DEVSTREAM_ROOT="$(dirname "$SCRIPT_DIR")"

# Priority 1: Explicit project root (manual override)
if [ -n "${DEVSTREAM_PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT="$DEVSTREAM_PROJECT_ROOT"
    print_info "Multi-project mode (explicit): $PROJECT_ROOT"

# Priority 2: Auto-detect from current working directory
elif PROJECT_ROOT=$(find_devstream_project_root "$(pwd)"); then
    export DEVSTREAM_PROJECT_ROOT="$PROJECT_ROOT"
    print_info "Multi-project mode (auto-detected): $PROJECT_ROOT"

# Priority 3: Error - not in DevStream project
else
    print_error "❌ No DevStream project found"
    print_error "   Searched from: $(pwd)"
    print_error "   Looking for:   .env.devstream marker file"
    print_error ""
    print_error "Solutions:"
    print_error "   1. cd to DevStream project directory first"
    print_error "   2. Run install-devstream.sh in current directory"
    print_error "   3. Set DEVSTREAM_PROJECT_ROOT=/path/to/project"
    exit 1
fi

print_info "DevStream installation: $DEVSTREAM_SCRIPT_DIR"

# Function to validate required environment variables
validate_environment() {
  print_status "Validating Z.AI environment..."

  # Load environment variables from DevStream installation root (single source of truth)
  if [ ! -f "$DEVSTREAM_ROOT/.env.devstream" ]; then
    print_error ".env.devstream file not found in DevStream installation: $DEVSTREAM_ROOT/.env.devstream"
    exit 1
  fi

  # Load environment variables from DevStream installation
  set -a
  source "$DEVSTREAM_ROOT/.env.devstream"
  set +a

  # Validate Z.AI API key
  if [ -z "${ZAI_API_KEY:-}" ]; then
    print_error "ZAI_API_KEY not configured in .env"
    print_error "Get your API key from: https://z.ai/manage-apikey/apikey-list"
    exit 1
  fi

  print_info "✅ Z.AI API key configured"
  print_info "✅ Environment validation complete"
}

# Function to setup Z.AI environment variables
setup_zai_environment() {
  print_status "Setting up Z.AI environment for Claude Code..."

  # Export Z.AI specific environment variables
  export ANTHROPIC_BASE_URL="https://api.z.ai/api/anthropic"
  export ANTHROPIC_AUTH_TOKEN="$ZAI_API_KEY"

  print_info "📡 Base URL: $ANTHROPIC_BASE_URL"
  print_info "🔑 Auth Token: ${ZAI_API_KEY:0:10}..."
  print_info "🤖 Target Model: GLM-4.6"
}

# Function to launch Claude Code with GLM-4.6
launch_claude_code() {
  print_status "🚀 Launching Claude Code with GLM-4.6..."

  # Change to project directory
  cd "$PROJECT_ROOT"

  # CONTEXT7 BEST PRACTICE: Use DevStream virtual environment for multi-project support
  # This ensures Claude Code always has access to DevStream modules regardless of project location
  # IMPORTANT: Set environment variables BEFORE configuring Claude Code settings
  export VIRTUAL_ENV="$DEVSTREAM_ROOT/.devstream"
  export PATH="$VIRTUAL_ENV/bin:$PATH"
  export PYTHONPATH="$DEVSTREAM_ROOT/.claude/hooks/devstream:$PYTHONPATH"

  # CRITICAL: Project-specific database path for multi-project isolation
  export DEVSTREAM_DB_PATH="$PROJECT_ROOT/data/devstream.db"

  # Configure Claude Code settings for z.ai (required for model switching)
  configure_claude_settings

  print_info "Working directory: $(pwd)"
  print_info "🤖 Model: GLM-4.6 (via z.ai API)"
  print_info "📡 API: $ANTHROPIC_BASE_URL"
  print_info "🐍 Virtual Environment: $VIRTUAL_ENV"
  print_info "🔧 Python Path: DevStream modules available"
  echo ""

  # Execute Claude Code with DevStream environment (settings will handle model selection)
  # CONTEXT7 BEST PRACTICE: Use environment block to ensure all variables are available to Claude Code
  {
    export VIRTUAL_ENV PATH PYTHONPATH ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN
    exec claude
  }
}

# Function to configure Claude Code settings for z.ai
configure_claude_settings() {
  local settings_file="$HOME/.claude/settings.json"

  print_status "⚙️ Configuring Claude Code settings for GLM-4.6..."

  # Backup existing settings
  if [ -f "$settings_file" ]; then
    local backup_file="$settings_file.backup-zai-$(date +%Y%m%d_%H%M%S)"
    if cp "$settings_file" "$backup_file" >/dev/null 2>&1 || true; then
      if [ -f "$backup_file" ]; then
        print_info "✅ Backed up existing settings"
      else
        print_warning "⚠️  Unable to backup existing settings (continuing without backup)"
      fi
    fi
  fi

  # Use Python for JSON manipulation (following Context7 best practices)
  # NOTE: Always use DevStream installation Python for cross-project compatibility
  "$DEVSTREAM_ROOT/.devstream/bin/python" << EOF
import json
import os

settings_file = "$settings_file"

# Read existing settings
existing_data = {}
if os.path.exists(settings_file):
    try:
        with open(settings_file, 'r') as f:
            existing_data = json.load(f)
    except (json.JSONDecodeError, IOError):
        print("Warning: Could not parse existing settings")

# Preserve important data
preserved_data = {
    "hooks": existing_data.get("hooks", {}),
    "mcpServers": existing_data.get("mcpServers", {}),
    "alwaysThinkingEnabled": existing_data.get("alwaysThinkingEnabled", False)
}

# Create z.ai configuration
zai_config = {
    **preserved_data,
    "model": "glm-4.6",
    "env": {
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-4.6",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-4.6",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-4.5-air",
        "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
        "ANTHROPIC_AUTH_TOKEN": os.environ.get("ZAI_API_KEY", ""),
        "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "PATH": os.environ.get("PATH", ""),
        "DEVSTREAM_DB_PATH": os.environ.get("DEVSTREAM_DB_PATH", "")
    }
}

# Write settings
with open(settings_file, 'w') as f:
    json.dump(zai_config, f, indent=2)

print("✅ Claude Code configured for GLM-4.6")
EOF

  if [ $? -eq 0 ]; then
    print_info "✅ Claude Code settings updated for GLM-4.6"
  else
    print_error "❌ Failed to update Claude Code settings"
    exit 1
  fi
}

# Main execution function
main() {
  echo ""
  print_status "🤖 DevStream Z.AI GLM-4.6 Launcher"
  print_status "==================================="
  echo ""

  # Execute setup steps
  validate_environment
  setup_zai_environment
  launch_claude_code
}

# Initialization logic (moved here to ensure functions are defined first)
validate_devstream_project "$PROJECT_ROOT"
print_project_info "$PROJECT_ROOT"

# Handle script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  main "$@"
fi
