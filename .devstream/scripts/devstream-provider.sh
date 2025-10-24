#!/bin/bash
# DevStream Provider Management CLI
# Manage AI providers for DevStream projects

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Context7 Pattern: Get DevStream root using environment detection
get_devstream_root() {
    # Priority 1: DEVSTREAM_ROOT environment variable or function parameter
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

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${1:-$(pwd)}"
DEVSTREAM_ROOT="${2:-$(get_devstream_root)}"  # Use dynamic detection if not provided

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Show usage
show_usage() {
    echo "DevStream Provider Management CLI"
    echo "================================"
    echo ""
    echo "Usage: $0 [PROJECT_DIR] [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  get                 Get current provider"
    echo "  set <provider>      Set provider (anthropic|z.ai)"
    echo "  show                Show full configuration"
    echo "  list                List available providers"
    echo "  test <provider>     Test provider connection"
    echo "  init                Initialize provider config"
    echo "  help                Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                           # Show current provider"
    echo "  $0 get                       # Get current provider"
    echo "  $0 set anthropic             # Set to Anthropic"
    echo "  $0 set z.ai                  # Set to z.ai"
    echo "  $0 /path/to/project set z.ai # Set provider for specific project"
    echo ""
}

# Check if project is valid
check_project() {
    # Validate DevStream installation first
    if [ ! -f "$DEVSTREAM_ROOT/start-devstream.sh" ]; then
        print_error "DevStream installation not found at: $DEVSTREAM_ROOT"
        print_info "Set DEVSTREAM_ROOT environment variable or ensure DevStream is properly installed"
        exit 1
    fi

    if [ ! -d "$PROJECT_ROOT" ]; then
        print_error "Project directory does not exist: $PROJECT_ROOT"
        exit 1
    fi

    if [ ! -d "$PROJECT_ROOT/.devstream" ]; then
        print_warning "DevStream not initialized in project: $PROJECT_ROOT"
        print_status "Initializing DevStream..."
        local init_script="$DEVSTREAM_ROOT/scripts/devstream-init.py"
        if [ -f "$init_script" ]; then
            "$init_script" "$PROJECT_ROOT" --force
        else
            print_error "DevStream initialization script not found: $init_script"
            exit 1
        fi
    fi
}

# Get current provider
get_provider() {
    local config_script="$DEVSTREAM_ROOT/scripts/devstream-config.py"

    if [ -f "$config_script" ]; then
        local provider=$(python3 "$config_script" get-provider "$PROJECT_ROOT" 2>/dev/null || echo "anthropic")
        echo "$provider"
    else
        echo "anthropic"
    fi
}

# Set provider
set_provider() {
    local provider="$1"
    local config_script="$DEVSTREAM_ROOT/scripts/devstream-config.py"

    if [ -z "$provider" ]; then
        print_error "Provider required"
        return 1
    fi

    if [ "$provider" != "anthropic" ] && [ "$provider" != "z.ai" ]; then
        print_error "Invalid provider: $provider"
        print_status "Available providers: anthropic, z.ai"
        return 1
    fi

    if [ -f "$config_script" ]; then
        if python3 "$config_script" set-provider "$provider" "$PROJECT_ROOT"; then
            print_success "Provider set to: $provider"
            return 0
        else
            print_error "Failed to set provider"
            return 1
        fi
    else
        print_error "Configuration script not found"
        return 1
    fi
}

# Show full configuration
show_config() {
    local config_script="$DEVSTREAM_ROOT/scripts/devstream-config.py"

    echo ""
    print_status "Provider Configuration for: $(basename "$PROJECT_ROOT")"
    echo "=" * 60

    if [ -f "$config_script" ]; then
        python3 "$config_script" show "$PROJECT_ROOT"
    else
        print_warning "Configuration script not found"
        echo "Default provider: anthropic"
    fi

    # Show environment variables
    echo ""
    print_status "Environment Variables:"
    echo "  ZAI_API_KEY: ${ZAI_API_KEY:+[SET]}${ZAI_API_KEY:-[NOT SET]}"
    echo "  CLAUDE_API_KEY: ${CLAUDE_API_KEY:+[SET]}${CLAUDE_API_KEY:-[NOT SET]}"
    echo ""
}

# List available providers
list_providers() {
    echo ""
    print_status "Available AI Providers:"
    echo ""
    echo "1) 🤖 Anthropic Claude"
    echo "   Model: Claude 3.5 Sonnet (20241022)"
    echo "   Auth: OAuth via Claude Code"
    echo "   Use: Best for complex reasoning, architecture, planning"
    echo "   Cost: Higher"
    echo ""
    echo "2) 🧠 z.ai GLM-4.6"
    echo "   Model: GLM-4.6"
    echo "   Auth: API Key (ZAI_API_KEY)"
    echo "   Use: Fast implementation, code generation, testing"
    echo "   Cost: Lower"
    echo ""
    echo "Current provider: $(get_provider)"
    echo ""
}

# Test provider connection
test_provider() {
    local provider="$1"

    if [ -z "$provider" ]; then
        provider=$(get_provider)
    fi

    print_status "Testing provider: $provider"
    echo ""

    case $provider in
        "anthropic")
            if command -v claude &> /dev/null; then
                print_success "Claude Code CLI found"
                print_status "Testing connection..."
                if claude --version &> /dev/null; then
                    print_success "Anthropic provider is working"
                else
                    print_warning "Claude Code may need login"
                    print_status "Run: claude login"
                fi
            else
                print_error "Claude Code CLI not found"
                print_status "Install with: npm install -g @anthropic-ai/claude-code"
            fi
            ;;
        "z.ai")
            if [ -n "$ZAI_API_KEY" ]; then
                print_success "ZAI_API_KEY found"
                print_status "z.ai provider should work"
            else
                print_error "ZAI_API_KEY not found"
                print_status "Set with: export ZAI_API_KEY='your-key'"
            fi
            ;;
        *)
            print_error "Unknown provider: $provider"
            ;;
    esac

    echo ""
}

# Initialize provider configuration
init_config() {
    local config_script="$DEVSTREAM_ROOT/scripts/devstream-config.py"

    print_status "Initializing provider configuration..."

    check_project

    if [ -f "$config_script" ]; then
        if python3 "$config_script" init "$PROJECT_ROOT"; then
            print_success "Configuration initialized"
            show_config
        else
            print_error "Failed to initialize configuration"
        fi
    else
        print_error "Configuration script not found"
    fi
}

# Main execution
main() {
    # Parse command line arguments
    if [ $# -eq 0 ]; then
        show_usage
        exit 0
    fi

    # Handle help
    if [ "$1" = "--help" ] || [ "$1" = "help" ]; then
        show_usage
        exit 0
    fi

    # Check if first argument is a directory
    if [ -d "$1" ] && [ ! "$1" = "get" ] && [ ! "$1" = "set" ] && [ ! "$1" = "show" ] && [ ! "$1" = "list" ] && [ ! "$1" = "test" ] && [ ! "$1" = "init" ]; then
        PROJECT_ROOT="$1"
        shift
    fi

    # Get command
    local command="${1:-get}"
    shift

    # Execute command
    case $command in
        "get")
            check_project
            get_provider
            ;;
        "set")
            check_project
            set_provider "$1"
            ;;
        "show")
            check_project
            show_config
            ;;
        "list")
            list_providers
            ;;
        "test")
            check_project
            test_provider "$1"
            ;;
        "init")
            init_config
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"