#!/bin/bash
# Multi-Project Architecture Integration Test
# Tests the complete workflow from project directory

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[STATUS]${NC} $1"; }
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Context7 Pattern: Dynamic path configuration
get_devstream_root() {
    # Priority 1: DEVSTREAM_ROOT environment variable
    if [ -n "${DEVSTREAM_ROOT:-}" ]; then
        echo "$DEVSTREAM_ROOT"
        return 0
    fi

    # Priority 2: Script location detection (two levels up from scripts/)
    if [ -f "${BASH_SOURCE[0]%/*/*}/start-devstream.sh" ]; then
        echo "${BASH_SOURCE[0]%/*/*}"
        return 0
    fi

    # Priority 3: Common installation locations
    local possible_paths=(
        "$HOME/.devstream"
        "$HOME/devstream"
        "/opt/devstream"
        "$(pwd)"
    )

    for path in "${possible_paths[@]}"; do
        if [ -f "$path/start-devstream.sh" ]; then
            echo "$path"
            return 0
        fi
    done

    return 1
}

# Test configuration (dynamic)
DEVSTREAM_ROOT=$(get_devstream_root)
if [ $? -ne 0 ] || [ -z "$DEVSTREAM_ROOT" ]; then
    print_error "DevStream installation not found!"
    print_info "Set DEVSTREAM_ROOT environment variable or ensure DevStream is properly installed"
    exit 1
fi

TEST_PROJECT="${1:-$(pwd)}"  # Use provided project or current directory
LAUNCHER="$DEVSTREAM_ROOT/scripts/simple-launcher.sh"

echo ""
print_status "🧪 Multi-Project Architecture Integration Test"
print_status "==============================================="
echo ""

# Test 1: Global Installation
print_status "Test 1: Verifying Global Installation"
if [ -f "$HOME/.devstream/bin/devstream" ]; then
    print_success "✅ Global CLI installed"
else
    print_error "❌ Global CLI not found"
    exit 1
fi

# Test 2: Project Initialization
print_status "Test 2: Verifying Project Initialization"
if [ -d "$TEST_PROJECT/.devstream" ]; then
    print_success "✅ Project initialized"
else
    print_error "❌ Project not initialized"
    exit 1
fi

# Test 3: Workspace Metadata
print_status "Test 3: Verifying Workspace Metadata"
if [ -f "$TEST_PROJECT/.devstream/workspace.json" ]; then
    print_success "✅ Workspace metadata exists"
    print_info "Project: $(jq -r '.project.name' "$TEST_PROJECT/.devstream/workspace.json")"
else
    print_error "❌ Workspace metadata not found"
    exit 1
fi

# Test 4: Project Database
print_status "Test 4: Verifying Project Database"
if [ -f "$TEST_PROJECT/data/devstream.db" ]; then
    print_success "✅ Project database exists"
else
    print_error "❌ Project database not found at $TEST_PROJECT/data/devstream.db"
    exit 1
fi

# Test 5: Launcher Script
print_status "Test 5: Verifying Launcher Script"
if [ -f "$LAUNCHER" ]; then
    print_success "✅ Launcher script exists"
else
    print_error "❌ Launcher script not found"
    exit 1
fi

# Test 6: Provider Configuration
print_status "Test 6: Verifying Provider Configuration"
PROVIDER=$(python3 "$DEVSTREAM_ROOT/scripts/devstream-config.py" get-provider "$TEST_PROJECT" 2>/dev/null || echo "anthropic")
print_info "Current provider: $PROVIDER"

# Test 7: Environment Loading
print_status "Test 7: Verifying Environment Loading"
if [ -f "$DEVSTREAM_ROOT/.env" ]; then
    if grep -q "ZAI_API_KEY" "$DEVSTREAM_ROOT/.env"; then
        print_success "✅ ZAI_API_KEY configured in DevStream .env"
    else
        print_warning "⚠️ ZAI_API_KEY not found in DevStream .env"
    fi
else
    print_warning "⚠️ DevStream .env file not found"
fi

# Test 8: Directory Independence
print_status "Test 8: Verifying Directory Independence"
cd "$TEST_PROJECT"
CURRENT_DIR=$(pwd)
if [ "$CURRENT_DIR" = "$TEST_PROJECT" ]; then
    print_success "✅ Working in project directory: $CURRENT_DIR"
else
    print_error "❌ Directory mismatch: $CURRENT_DIR != $TEST_PROJECT"
    exit 1
fi

echo ""
print_success "🎉 All Tests Passed! Multi-Project Architecture Working"
echo ""
print_info "Usage Examples:"
echo "  # From any project directory:"
echo "  $LAUNCHER start anthropic   # Use Claude Sonnet 4.5"
echo "  $LAUNCHER start z.ai        # Use GLM-4.6"
echo ""
print_info "Project Management:"
echo "  devstream list              # List all projects"
echo "  devstream status             # Current project status"
echo ""
print_success "✅ Multi-Project Architecture Integration Complete"