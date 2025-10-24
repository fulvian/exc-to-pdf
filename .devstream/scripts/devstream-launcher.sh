#!/bin/bash
# DevStream Universal Project Launcher
# Launch DevStream from ANY project directory

set -uo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_status() { echo -e "${GREEN}[STATUS]${NC} $1"; }
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Context7 Pattern: Get DevStream root using environment detection
get_devstream_root() {
    # Priority 1: DEVSTREAM_ROOT environment variable
    if [ -n "${DEVSTREAM_ROOT:-}" ]; then
        echo "$DEVSTREAM_ROOT"
        return 0
    fi

    # Priority 2: Script location detection (two levels up from scripts/)
    if [ -n "${BASH_SOURCE[0]}" ]; then
        local script_dir="${BASH_SOURCE[0]%/*}"
        local parent_dir="${script_dir%/*}"
        if [ -f "$parent_dir/start-devstream.sh" ]; then
            echo "$parent_dir"
            return 0
        fi
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

    # Priority 4: Search in user directory
    if command -v find >/dev/null 2>&1; then
        local found_path
        found_path=$(find "$HOME" -name 'start-devstream.sh' -type f 2>/dev/null | head -1 | xargs dirname 2>/dev/null || echo "")
        if [ -n "$found_path" ] && [ -f "$found_path/start-devstream.sh" ]; then
            echo "$found_path"
            return 0
        fi
    fi

    # Priority 5: Current working directory
    if [ -f "$(pwd)/start-devstream.sh" ]; then
        echo "$(pwd)"
        return 0
    fi

    return 1
}

# Auto-detect DevStream installation using Context7 patterns
find_devstream_installation() {
    local devstream_root
    devstream_root=$(get_devstream_root)

    if [ $? -eq 0 ] && [ -n "$devstream_root" ]; then
        echo "$devstream_root"
        return 0
    fi

    print_error "DevStream installation not found!"
    print_info "Please install DevStream first or set DEVSTREAM_ROOT environment variable"
    print_info "  export DEVSTREAM_ROOT=/path/to/devstream"
    return 1
}

# Get current project directory
get_project_root() {
    local current_dir="$(pwd)"

    # If we're already in a DevStream project, use current directory
    if [ -d "$current_dir/.devstream" ]; then
        echo "$current_dir"
        return 0
    fi

    # If we're in a subdirectory of a DevStream project, find the root
    local search_dir="$current_dir"
    while [ "$search_dir" != "/" ]; do
        if [ -d "$search_dir/.devstream" ]; then
            echo "$search_dir"
            return 0
        fi
        search_dir="$(dirname "$search_dir")"
    done

    # No DevStream project found in current directory tree
    echo "$current_dir"
    return 0
}

# Main function
main() {
    local command="${1:-start}"
    local provider="${2:-anthropic}"

    echo ""
    print_status "🚀 DevStream Universal Launcher"
    print_status "=============================="
    echo ""

    # Get current directory
    local current_dir="$(pwd)"
    print_info "Current directory: $current_dir"

    # Find project root
    local project_root="$(get_project_root)"
    if [ "$project_root" != "$current_dir" ]; then
        print_info "Found DevStream project root: $project_root"
    fi

    # Find DevStream installation
    local devstream_root
    devstream_root="$(find_devstream_installation)"

    if [ $? -ne 0 ]; then
        exit 1
    fi

    print_info "DevStream installation: $devstream_root"
    print_info "Project root: $project_root"
    echo ""

    # Build full command
    local full_cmd="cd \"$project_root\" && \"$devstream_root/start-devstream.sh\" $command $provider"

    print_status "🔄 Launching DevStream..."
    print_info "Command: $full_cmd"
    echo ""

    # Execute command with error handling
    if eval "$full_cmd"; then
        print_status "✅ DevStream launched successfully"
    else
        local exit_code=$?
        print_error "❌ DevStream launch failed (exit code: $exit_code)"
        exit $exit_code
    fi
}

# Parse arguments and run main
main "$@"