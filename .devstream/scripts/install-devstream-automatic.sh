#!/bin/bash
# DevStream Automatic Installation Script
# One-command installation for any project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${1:-$(pwd)}"
DEVSTREAM_HOME="${2:-$HOME/.devstream}"
FORCE_INSTALL="${3:-false}"

# Print functions
print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Header
echo -e "${BLUE}"
echo "=================================================================="
echo "🚀 DevStream Automatic Installation Script v2.2.0"
echo "=================================================================="
echo -e "${NC}"

print_status "Installation Configuration:"
print_status "   Project Root: $PROJECT_ROOT"
print_status "   DevStream Home: $DEVSTREAM_HOME"
print_status "   Force Install: $FORCE_INSTALL"
echo ""

# Helper functions
check_prerequisites() {
    print_status "Checking prerequisites..."

    # Check Python 3.11+
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found"
        exit 1
    fi

    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $(echo "$python_version >= 3.11" | bc -l) -eq 0 ]]; then
        print_error "Python 3.11+ required, found: $python_version"
        exit 1
    fi
    print_success "Python $python_version found"

    # Check Git
    if ! command -v git &> /dev/null; then
        print_error "Git not found"
        exit 1
    fi
    print_success "Git found"

    print_success "All prerequisites met"
}

install_devstream_globally() {
    print_status "Installing DevStream globally..."

    # Create global directories
    mkdir -p "$DEVSTREAM_HOME"/{bin,data,config,hooks,lib,share,templates,logs}

    # Copy CLI tools
    cp "$SCRIPT_DIR/devstream" "$DEVSTREAM_HOME/bin/"
    cp "$SCRIPT_DIR/devstream-init.py" "$DEVSTREAM_HOME/bin/"
    cp "$SCRIPT_DIR/scan-codebase.py" "$DEVSTREAM_HOME/bin/"
    cp "$SCRIPT_DIR/setup-db.py" "$DEVSTREAM_HOME/bin/"

    # Make executables
    chmod +x "$DEVSTREAM_HOME/bin"/*

    # Copy hooks
    if [ -d "$SCRIPT_DIR/../.claude" ]; then
        cp -r "$SCRIPT_DIR/../.claude" "$DEVSTREAM_HOME/"
    fi

    # Create project registry
    mkdir -p "$DEVSTREAM_HOME/data"
    echo '{"version":"1.0.0","created":"","last_updated":"","projects":[],"statistics":{"total_projects":0,"active_projects":0}}' > "$DEVSTREAM_HOME/data/registry.json"

    # Update PATH in shell profile
    update_shell_profile

    print_success "DevStream installed globally"
}

update_shell_profile() {
    local shell_rc=""
    case "$SHELL" in
        */bash) shell_rc="$HOME/.bashrc" ;;
        */zsh) shell_rc="$HOME/.zshrc" ;;
        */fish) shell_rc="$HOME/.config/fish/config.fish" ;;
    esac

    if [ -n "$shell_rc" ] && [ -f "$shell_rc" ]; then
        if ! grep -q "DEVSTREAM_HOME" "$shell_rc"; then
            echo "" >> "$shell_rc"
            echo "# DevStream CLI" >> "$shell_rc"
            echo "export DEVSTREAM_HOME=\"$DEVSTREAM_HOME\"" >> "$shell_rc"
            echo 'export PATH="$DEVSTREAM_HOME/bin:$PATH"' >> "$shell_rc"
            print_status "Updated $shell_rc with DevStream PATH"
        fi
    fi
}

initialize_project() {
    print_status "Initializing DevStream for project: $PROJECT_ROOT"

    # Check if project exists
    if [ ! -d "$PROJECT_ROOT" ]; then
        print_error "Project directory does not exist: $PROJECT_ROOT"
        exit 1
    fi

    # Check if already initialized
    if [ -d "$PROJECT_ROOT/.devstream" ] && [ "$FORCE_INSTALL" != "true" ]; then
        print_warning "DevStream already initialized in $PROJECT_ROOT"
        print_status "Use --force to reinitialize"
        return 0
    fi

    # Initialize project
    cd "$PROJECT_ROOT"
    "$DEVSTREAM_HOME/bin/devstream-init.py" . --verbose

    if [ $? -eq 0 ]; then
        print_success "Project initialized successfully"

        # Register project globally
        "$DEVSTREAM_HOME/bin/devstream" register .

        if [ $? -eq 0 ]; then
            print_success "Project registered globally"
        else
            print_warning "Project registration failed (non-critical)"
        fi
    else
        print_error "Project initialization failed"
        exit 1
    fi
}

verify_installation() {
    print_status "Verifying installation..."

    # Check global installation
    if [ ! -f "$DEVSTREAM_HOME/bin/devstream" ]; then
        print_error "Global CLI tool not found"
        return 1
    fi

    # Check project installation
    if [ ! -d "$PROJECT_ROOT/.devstream" ]; then
        print_error "Project DevStream directory not found"
        return 1
    fi

    # Test CLI commands
    cd "$PROJECT_ROOT"
    if "$DEVSTREAM_HOME/bin/devstream" detect > /dev/null 2>&1; then
        print_success "CLI detection working"
    else
        print_warning "CLI detection may have issues"
    fi

    print_success "Installation verification completed"
}

show_next_steps() {
    echo ""
    print_success "🎉 DevStream installation completed successfully!"
    echo ""
    echo -e "${GREEN}Next Steps:${NC}"
    echo "1. Restart your terminal or run: source ~/.bashrc (or ~/.zshrc)"
    echo "2. Navigate to your project: cd $PROJECT_ROOT"
    echo "3. Start DevStream: ./start-devstream.sh start"
    echo ""
    echo -e "${BLUE}Useful Commands:${NC}"
    echo "  devstream status     - Show project status"
    echo "  devstream list       - List all projects"
    echo "  devstream detect     - Detect current project"
    echo ""
    echo -e "${YELLOW}Project Database:${NC}"
    echo "  $PROJECT_ROOT/.devstream/db/devstream.db"
}

# Main installation flow
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --project-dir)
                PROJECT_ROOT="$2"
                shift 2
                ;;
            --devstream-home)
                DEVSTREAM_HOME="$2"
                shift 2
                ;;
            --force)
                FORCE_INSTALL="true"
                shift
                ;;
            --help)
                echo "Usage: $0 [PROJECT_DIR] [DEVSTREAM_HOME] [--force]"
                echo ""
                echo "Arguments:"
                echo "  PROJECT_DIR     Project directory (default: current directory)"
                echo "  DEVSTREAM_HOME  DevStream home directory (default: ~/.devstream)"
                echo "  --force         Force reinstallation if already exists"
                echo ""
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Execute installation steps
    print_status "Starting DevStream automatic installation..."

    check_prerequisites

    if [ ! -d "$DEVSTREAM_HOME/bin" ] || [ "$FORCE_INSTALL" = "true" ]; then
        install_devstream_globally
    else
        print_status "DevStream already installed globally"
    fi

    initialize_project
    verify_installation
    show_next_steps
}

# Run main function
main "$@"