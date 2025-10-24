#!/bin/bash
# Universal Project DevStream Startup Script
# Start Claude Code + DevStream session for ANY project

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
    echo "Usage: $0 [PROJECT_DIR] [DEVSTREAM_ROOT]"
    echo ""
    echo "Arguments:"
    echo "  PROJECT_DIR     Project directory (default: current directory)"
    echo "  DEVSTREAM_ROOT  DevStream installation directory (default: auto-detect)"
    echo ""
    echo "Environment Variables:"
    echo "  DEVSTREAM_ROOT  DevStream installation directory (overrides auto-detection)"
    echo ""
    echo "Examples:"
    echo "  $0                           # Current directory, auto-detect DevStream"
    echo "  $0 /path/to/project          # Specific project, auto-detect DevStream"
    echo "  $0 /path/to/project /path/to/devstream  # Custom paths"
    echo "  DEVSTREAM_ROOT=/path/to/devstream $0     # Using environment variable"
    echo ""
}

# Header
show_header() {
    echo ""
    echo -e "${GREEN}"
    echo "🚀 Universal DevStream Project Launcher"
    echo "===================================="
    echo -e "${NC}"

    # Validate DevStream installation
    if [ ! -f "$DEVSTREAM_ROOT/start-devstream.sh" ]; then
        print_error "DevStream installation not found at: $DEVSTREAM_ROOT"
        print_info "Set DEVSTREAM_ROOT environment variable or ensure DevStream is properly installed"
        echo ""
        print_info "Auto-detection attempted:"
        print_info "  - DEVSTREAM_ROOT environment variable: ${DEVSTREAM_ROOT:-[not set]}"
        print_info "  - Script location detection: ${BASH_SOURCE[0]%/*/*}"
        print_info "  - Common locations: \$HOME/.devstream, \$HOME/devstream, /opt/devstream"
        echo ""
        exit 1
    fi

    print_status "Project: $PROJECT_ROOT"
    print_status "DevStream: $DEVSTREAM_ROOT"

    # Show project name if detectable
    if [ -d "$PROJECT_ROOT/.devstream" ]; then
        PROJECT_NAME=$(basename "$PROJECT_ROOT")
        print_status "Project Name: $PROJECT_NAME ✅"
    else
        print_warning "DevStream not yet initialized for this project"
    fi
    echo ""
}

# Check if project directory exists
check_project() {
    if [ ! -d "$PROJECT_ROOT" ]; then
        print_error "Project directory does not exist: $PROJECT_ROOT"
        exit 1
    fi
}

# Check and install DevStream if needed
check_devstream() {
    print_status "Checking DevStream installation..."

    # Check if DevStream is initialized in project
    if [ ! -d "$PROJECT_ROOT/.devstream" ]; then
        print_warning "DevStream not initialized in this project"
        echo ""
        echo "Options:"
        echo "1) Auto-install DevStream for this project"
        echo "2) Exit and install manually"
        echo ""
        read -p "Choose (1-2): " choice

        case $choice in
            1)
                auto_install_devstream
                ;;
            *)
                print_status "Exiting. Install DevStream manually first."
                exit 0
                ;;
        esac
    else
        print_success "DevStream already initialized"
    fi

    # Check CLI tools
    if [ -f "$HOME/.devstream/bin/devstream" ]; then
        print_success "DevStream CLI available"
    else
        print_warning "DevStream CLI not found in global installation"
        print_status "Consider running global installation first"
    fi
}

# Auto-install DevStream for current project
auto_install_devstream() {
    print_status "Auto-installing DevStream for project..."

    local install_script="$DEVSTREAM_ROOT/scripts/install-devstream-automatic.sh"

    if [ ! -f "$install_script" ]; then
        print_error "Installation script not found: $install_script"
        exit 1
    fi

    # Run automatic installation
    if bash "$install_script" "$PROJECT_ROOT"; then
        print_success "DevStream installed successfully"
    else
        print_error "DevStream installation failed"
        exit 1
    fi
}

# Show comprehensive project status
show_project_status() {
    print_status "Project Status:"
    echo ""

    # Show basic project info
    echo "📁 Directory: $PROJECT_ROOT"
    echo "📂 Project Name: $(basename "$PROJECT_ROOT")"

    # Show DevStream status if available
    if [ -f "$HOME/.devstream/bin/devstream" ] && [ -d "$PROJECT_ROOT/.devstream" ]; then
        echo ""
        cd "$PROJECT_ROOT"
        "$HOME/.devstream/bin/devstream" status
    else
        print_warning "DevStream CLI not available for detailed status"
    fi

    # Show project structure
    echo ""
    print_status "Project Structure:"
    if command -v tree &> /dev/null; then
        tree -L 2 "$PROJECT_ROOT" 2>/dev/null || ls -la "$PROJECT_ROOT"
    else
        ls -la "$PROJECT_ROOT"
    fi

    echo ""
}

# Get current provider for project
get_project_provider() {
    local config_script="$DEVSTREAM_ROOT/scripts/devstream-config.py"

    if [ -f "$config_script" ]; then
        python3 "$config_script" get-provider "$PROJECT_ROOT" 2>/dev/null || echo "anthropic"
    else
        echo "anthropic"
    fi
}

# Set provider for project
set_project_provider() {
    local provider="$1"
    local config_script="$DEVSTREAM_ROOT/scripts/devstream-config.py"

    if [ -f "$config_script" ]; then
        if python3 "$config_script" set-provider "$provider" "$PROJECT_ROOT"; then
            print_success "Provider set to: $provider"
            return 0
        fi
    fi

    print_error "Failed to set provider"
    return 1
}

# Show provider selection menu
show_provider_selection() {
    echo ""
    print_status "Select AI Provider for this project:"
    echo ""
    echo "1) 🤖 Anthropic Claude (Sonnet 4.5)"
    echo "   • Best for complex reasoning and architecture"
    echo "   • OAuth authentication via Claude Code"
    echo "   • Higher context limits"
    echo ""
    echo "2) 🧠 z.ai GLM-4.6"
    echo "   • Fast and cost-effective"
    echo "   • Good for implementation tasks"
    echo "   • API key authentication"
    echo ""
    echo "3) 📊 Show current provider settings"
    echo "4) ⚙️  Configure provider settings"
    echo ""
    read -p "Choose provider (1-4): " choice

    case $choice in
        1)
            set_project_provider "anthropic"
            start_claude_session "anthropic"
            ;;
        2)
            set_project_provider "z.ai"
            start_claude_session "z.ai"
            ;;
        3)
            show_current_provider
            show_provider_selection
            ;;
        4)
            configure_provider_settings
            ;;
        *)
            print_warning "Invalid choice. Using default (Anthropic)..."
            start_claude_session "anthropic"
            ;;
    esac
}

# Show current provider configuration
show_current_provider() {
    local config_script="$DEVSTREAM_ROOT/scripts/devstream-config.py"

    echo ""
    print_status "Current Provider Configuration:"
    echo ""

    if [ -f "$config_script" ]; then
        python3 "$config_script" show "$PROJECT_ROOT"
    else
        print_warning "Configuration script not found"
        echo "Default provider: anthropic"
    fi
    echo ""
}

# Configure provider settings
configure_provider_settings() {
    echo ""
    print_status "Provider Configuration:"
    echo ""
    echo "1) Set Anthropic as default"
    echo "2) Set z.ai as default"
    echo "3) Show API key setup instructions"
    echo "4) Back to provider selection"
    echo ""
    read -p "Choose (1-4): " choice

    case $choice in
        1)
            set_project_provider "anthropic"
            show_current_provider
            configure_provider_settings
            ;;
        2)
            set_project_provider "z.ai"
            show_current_provider
            configure_provider_settings
            ;;
        3)
            show_api_setup_instructions
            configure_provider_settings
            ;;
        4)
            show_provider_selection
            ;;
        *)
            print_warning "Invalid choice"
            configure_provider_settings
            ;;
    esac
}

# Show API setup instructions
show_api_setup_instructions() {
    echo ""
    print_status "API Setup Instructions:"
    echo ""
    echo "🤖 Anthropic Claude:"
    echo "   1. Install Claude Code: npm install -g @anthropic-ai/claude-code"
    echo "   2. Login: claude login"
    echo "   3. Visit: https://claude.ai/"
    echo ""
    echo "🧠 z.ai GLM-4.6:"
    echo "   1. Get API key from z.ai platform"
    echo "   2. Set environment variable: export ZAI_API_KEY='your-key'"
    echo "   3. Add to ~/.bashrc or ~/.zshrc for persistence"
    echo ""
    read -p "Press Enter to continue..."
}

# Load DevStream environment variables
load_devstream_env() {
    local devstream_env="$DEVSTREAM_ROOT/.env"

    if [ -f "$devstream_env" ]; then
        print_status "Loading DevStream environment variables..."
        set -a
        source "$devstream_env"
        set +a
        print_success "Environment variables loaded"
    else
        print_warning "DevStream .env file not found: $devstream_env"
    fi
}

# Start Claude Code session with specific provider
start_claude_session() {
    local provider="${1:-$(get_project_provider)}"

    print_status "Starting Claude Code + DevStream session..."
    print_status "Provider: $provider"
    echo ""

    # Load DevStream environment variables first
    load_devstream_env

    # Set environment variables
    export PROJECT_ROOT="$PROJECT_ROOT"
    export DEVSTREAM_PROJECT="$(basename "$PROJECT_ROOT")"
    export DEVSTREAM_PROVIDER="$provider"

    # Change to project directory
    cd "$PROJECT_ROOT"

    # Provider-specific setup
    case $provider in
        "z.ai")
            if [ -n "$ZAI_API_KEY" ]; then
                print_success "ZAI_API_KEY loaded from DevStream environment"
            else
                print_error "ZAI_API_KEY not found in DevStream .env"
                print_status "Add to $DEVSTREAM_ROOT/.env: ZAI_API_KEY='your-key'"
                echo ""
                return 1
            fi
            print_status "Launching with z.ai GLM-4.6 provider..."
            ;;
        *)
            print_status "Launching with Anthropic Claude provider..."
            ;;
    esac

    # Show session info
    echo ""
    echo -e "${YELLOW}💡 Claude Code + DevStream Session Started${NC}"
    echo "   Project: $(basename "$PROJECT_ROOT")"
    echo "   Provider: $provider"
    echo "   Database: $PROJECT_ROOT/.devstream/db/devstream.db"
    echo "   Environment: DevStream .env loaded"
    echo ""
    echo -e "${YELLOW}🚀 Ready to code!${NC}"
    echo ""

    # Start Claude Code (no --project option, just change directory)
    claude
}

# Start DevStream standalone
start_devstream_standalone() {
    print_status "Starting DevStream standalone..."
    echo ""

    cd "$PROJECT_ROOT"
    "$DEVSTREAM_ROOT/start-devstream.sh" start
}

# Show startup options
show_startup_menu() {
    echo -e "${BLUE}How would you like to start?${NC}"
    echo ""
    echo "1) 🤖 Claude Code + DevStream (choose provider)"
    echo "2) 🔧 DevStream standalone"
    echo "3) 📊 Just show project status"
    echo "4) 🛠️  Initialize/reinstall DevStream"
    echo "5) ⚙️  Provider configuration"
    echo "6) ❓ Show help and alternatives"
    echo ""
    read -p "Choose (1-6): " choice

    case $choice in
        1)
            show_provider_selection
            ;;
        2)
            start_devstream_standalone
            ;;
        3)
            show_project_status
            ;;
        4)
            auto_install_devstream
            ;;
        5)
            configure_provider_settings
            show_startup_menu
            ;;
        6)
            show_help_alternatives
            ;;
        *)
            print_warning "Invalid choice. Starting provider selection..."
            show_provider_selection
            ;;
    esac
}

# Show help and alternatives
show_help_alternatives() {
    echo ""
    print_status "Alternative startup methods:"
    echo ""
    echo "1) 🔄 Direct DevStream start:"
    echo "   cd $PROJECT_ROOT"
    echo "   $DEVSTREAM_ROOT/start-devstream.sh start"
    echo ""
    echo "2) 💻 Manual Claude Code:"
    echo "   cd $PROJECT_ROOT"
    echo "   claude"
    echo ""
    echo "3) 🛠️  DevStream CLI commands:"
    if [ -f "$HOME/.devstream/bin/devstream" ]; then
        echo "   $HOME/.devstream/bin/devstream status"
        echo "   $HOME/.devstream/bin/devstream list"
        echo "   $HOME/.devstream/bin/devstream detect"
    else
        echo "   (Install DevStream CLI first)"
    fi
    echo ""
    echo "4) 📦 Installation commands:"
    echo "   bash $DEVSTREAM_ROOT/scripts/install-devstream-automatic.sh $PROJECT_ROOT"
    echo ""
    echo "5) 🧪 Testing:"
    echo "   cd $PROJECT_ROOT && $HOME/.devstream/bin/devstream-init.py . --force"
    echo ""
}

# Main execution
main() {
    # Parse command line arguments
    case "$1" in
        -h|--help)
            show_usage
            exit 0
            ;;
        --version)
            echo "Universal DevStream Project Launcher v2.2.0"
            exit 0
            ;;
    esac

    show_header
    check_project
    check_devstream
    show_project_status
    show_startup_menu
}

# Run main function with all arguments
main "$@"