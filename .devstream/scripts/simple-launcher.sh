#!/bin/bash
# Simple DevStream Launcher - No blocking issues
# Launch DevStream from ANY project directory

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

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

    # Priority 4: Current working directory
    if [ -f "$(pwd)/start-devstream.sh" ]; then
        echo "$(pwd)"
        return 0
    fi

    return 1
}

# Function to detect project directory using multiple methods
detect_project_directory() {
    local detected_dir=""

    # Method 1: Use current working directory (most reliable for launcher)
    if [[ -n "${PWD}" ]]; then
        detected_dir="$PWD"
    else
        # Method 2: Check if BASH_SOURCE[0] is available
        if [[ -n "${BASH_SOURCE[0]}" ]]; then
            # Get the directory where the script was called from
            detected_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
        else
            # Method 3: Check if script was called with full path
            if [[ "$0" == /* ]]; then
                detected_dir="$(dirname "$0")"
            fi
        fi
    fi

    # Method 4: Look for workspace.json in parent directories (DevStream project marker)
    if [[ -n "$detected_dir" ]]; then
        local search_dir="$detected_dir"
        while [[ "$search_dir" != "/" ]]; do
            if [[ -f "$search_dir/.devstream/workspace.json" ]]; then
                detected_dir="$search_dir"
                break
            fi
            search_dir="$(dirname "$search_dir")"
        done
    fi

    # Method 5: Check for data/devstream.db (project database marker)
    if [[ -n "$detected_dir" ]]; then
        local search_dir="$detected_dir"
        while [[ "$search_dir" != "/" ]]; do
            if [[ -f "$search_dir/data/devstream.db" ]]; then
                detected_dir="$search_dir"
                break
            fi
            search_dir="$(dirname "$search_dir")"
        done
    fi

    # Method 6: Fallback to current working directory
    if [[ -z "$detected_dir" || "$detected_dir" == "." ]]; then
        detected_dir="$(pwd)"
    fi

    # Resolve to absolute path
    detected_dir="$(cd "$detected_dir" && pwd)"

    echo "$detected_dir"
}

# Main function
main() {
    local command="${1:-start}"
    local provider="${2:-anthropic}"

    echo ""
    print_status "🚀 Simple DevStream Launcher"
    print_status "==============================="
    echo ""

    # Detect project directory using multiple methods
    local project_dir="$(detect_project_directory)"
    print_info "Detected project directory: $project_dir"

    # Verify it's a valid DevStream project
    if [[ ! -f "$project_dir/.devstream/workspace.json" ]] && [[ ! -f "$project_dir/data/devstream.db" ]]; then
        print_warning "Directory doesn't appear to be a DevStream project"
        print_info "Initializing DevStream in current directory..."

        # Initialize DevStream if it's not a project yet
        cd "$project_dir"
        local devstream_root
        devstream_root=$(get_devstream_root)
        if [ $? -eq 0 ] && [ -n "$devstream_root" ]; then
            local init_script="$devstream_root/scripts/devstream-init.py"
            if [[ -f "$init_script" ]]; then
                python3 "$init_script" "$project_dir"
            else
                print_error "DevStream initialization script not found at: $init_script"
                exit 1
            fi
        else
            print_error "DevStream installation not found!"
            print_info "Set DEVSTREAM_ROOT environment variable or ensure DevStream is properly installed"
            exit 1
        fi
    fi

    # Find DevStream installation using Context7 patterns
    local devstream_root
    devstream_root=$(get_devstream_root)
    if [ $? -ne 0 ] || [ -z "$devstream_root" ]; then
        print_error "DevStream installation not found!"
        print_info "Set DEVSTREAM_ROOT environment variable or ensure DevStream is properly installed"
        exit 1
    fi

    if [ ! -f "$devstream_root/start-devstream.sh" ]; then
        print_error "DevStream installation not found at: $devstream_root"
        print_info "Expected start-devstream.sh at: $devstream_root/start-devstream.sh"
        exit 1
    fi

    print_info "DevStream installation: $devstream_root"
    print_info "Provider: $provider"
    echo ""

    print_status "🔄 Launching DevStream..."
    print_info "Project root: $project_dir"
    echo ""

    # Set environment variables and execute
    export DEVSTREAM_PROJECT_ROOT="$project_dir"
    cd "$project_dir"

    # Execute DevStream with project directory context
    exec "$devstream_root/start-devstream.sh" "$command" "$provider"
}

# Parse arguments and run main
main "$@"