#!/bin/bash

#==============================================================================
# DevStream Global Installation Script
# Version: 2.2.0
# Description: Install DevStream globally with ~/.devstream/ structure
#==============================================================================

set -e  # Exit on error (disable with --no-exit)
set -u  # Exit on undefined variable

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVSTREAM_ROOT="${SCRIPT_DIR}/.."
DEVSTREAM_HOME="${DEVSTREAM_HOME:-$HOME/.devstream}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Flags
VERBOSE=false
DRY_RUN=false
NO_EXIT=false
SKIP_OPTIONAL=false

#------------------------------------------------------------------------------
# Helper Functions
#------------------------------------------------------------------------------

print_header() {
    echo ""
    echo -e "${BLUE}===================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}===================================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[VERBOSE]${NC} $1"
    fi
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

check_exit_code() {
    local exit_code=$1
    local success_msg=$2
    local error_msg=$3

    if [ $exit_code -eq 0 ]; then
        print_success "$success_msg"
        return 0
    else
        print_error "$error_msg (exit code: $exit_code)"
        if [ "$NO_EXIT" = false ]; then
            exit $exit_code
        fi
        return $exit_code
    fi
}

prompt_continue() {
    local message=$1
    if [ "$SKIP_OPTIONAL" = true ]; then
        print_warning "Skipping: $message"
        return 1
    fi

    echo -e "${YELLOW}?${NC} $message"
    read -p "Continue? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

#------------------------------------------------------------------------------
# Parse Arguments
#------------------------------------------------------------------------------

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --verbose|-v)
                VERBOSE=true
                print_info "Verbose mode enabled"
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                print_warning "Dry-run mode: No changes will be made"
                shift
                ;;
            --no-exit)
                NO_EXIT=true
                print_warning "No-exit mode: Script will continue on errors"
                shift
                ;;
            --skip-optional)
                SKIP_OPTIONAL=true
                print_info "Skipping optional steps"
                shift
                ;;
            --help|-h)
                cat << EOF
DevStream Global Installation Script

Usage: $0 [OPTIONS]

Options:
  --verbose, -v       Enable verbose output
  --dry-run          Show what would be done without making changes
  --no-exit          Continue on errors instead of exiting
  --skip-optional    Skip optional steps without prompting
  --help, -h         Show this help message

Examples:
  $0                              # Standard global installation
  $0 --verbose                    # Installation with detailed output
  $0 --dry-run                    # Test installation without changes

EOF
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

#------------------------------------------------------------------------------
# Main Installation Function
#------------------------------------------------------------------------------

install_devstream_global() {
    """Install DevStream globally with all required components."""

    # Args: None (uses environment variables)
    # Returns: 0 on success, 1 on failure

    # Creates:
    # - ~/.devstream/bin/ (CLI tools)
    # - ~/.devstream/hooks/ (Hook system)
    # - ~/.devstream/templates/ (Project templates)
    # - ~/.devstream/config/ (Global configuration)
    # - ~/.devstream/data/registry.json (Project registry)
}

# Step 1: Prerequisites Check
check_prerequisites() {
    print_header "Step 1: Checking Prerequisites"

    local has_errors=false

    # Check Python 3.11+
    print_info "Checking Python 3.11+..."
    if ! command_exists python3.11; then
        print_error "Python 3.11+ not found"
        echo ""
        echo "Installation instructions:"
        echo "  macOS:   brew install python@3.11"
        echo "  Ubuntu:  sudo apt install python3.11 python3.11-venv"
        echo "  Fedora:  sudo dnf install python3.11"
        echo ""
        has_errors=true
    else
        local python_version=$(python3.11 --version 2>&1 | awk '{print $2}')
        print_success "Python 3.11 found: $python_version"
        print_verbose "Python path: $(which python3.11)"
    fi

    # Check Node.js 16+
    print_info "Checking Node.js 16+..."
    if command_exists node; then
        local node_version=$(node --version)
        local node_major=$(echo "$node_version" | sed 's/v\([0-9]*\).*/\1/')
        if [ "$node_major" -ge 16 ]; then
            print_success "Node.js found: $node_version"
            print_verbose "Node path: $(which node)"
        else
            print_error "Node.js version must be 16+, found: $node_version"
            echo ""
            echo "Installation instructions:"
            echo "  macOS:   brew install node"
            echo "  Ubuntu:  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt install nodejs"
            echo "  Fedora:  sudo dnf install nodejs"
            echo ""
            has_errors=true
        fi
    else
        print_error "Node.js not found"
        echo ""
        echo "Installation instructions:"
        echo "  macOS:   brew install node"
        echo "  Ubuntu:  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt install nodejs"
        echo "  Fedora:  sudo dnf install nodejs"
        echo ""
        has_errors=true
    fi

    # Check Git
    print_info "Checking Git..."
    if command_exists git; then
        local git_version=$(git --version | awk '{print $3}')
        print_success "Git found: $git_version"
        print_verbose "Git path: $(which git)"
    else
        print_error "Git not found"
        echo ""
        echo "Installation instructions:"
        echo "  macOS:   brew install git"
        echo "  Ubuntu:  sudo apt install git"
        echo "  Fedora:  sudo dnf install git"
        echo ""
        has_errors=true
    fi

    if [ "$has_errors" = true ]; then
        print_error "Prerequisites check failed"
        if [ "$NO_EXIT" = false ]; then
            exit 1
        fi
        return 1
    fi

    print_success "All prerequisites met"
}

# Step 2: Create Global Directory Structure
create_global_structure() {
    print_header "Step 2: Creating Global Directory Structure"

    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY-RUN] Would create global DevStream structure at: $DEVSTREAM_HOME"
        return 0
    fi

    # Define directories to create
    local dirs=(
        "bin"
        "hooks"
        "templates"
        "config"
        "data"
        "logs"
        "lib"
        "share"
    )

    print_info "Creating global DevStream directories..."

    for dir in "${dirs[@]}"; do
        local target_dir="$DEVSTREAM_HOME/$dir"
        if ! mkdir -p "$target_dir"; then
            print_error "Failed to create directory: $target_dir"
            return 1
        fi
        print_verbose "Created: $target_dir"
    done

    # Set permissions
    chmod 755 "$DEVSTREAM_HOME"
    print_verbose "Set permissions: 755 for $DEVSTREAM_HOME"

    print_success "Global directory structure created"
}

# Step 3: Install CLI Tools
install_cli_tools() {
    print_header "Step 3: Installing CLI Tools"

    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY-RUN] Would install CLI tools to: $DEVSTREAM_HOME/bin"
        return 0
    fi

    # Install main devstream CLI
    local devstream_cli_source="$DEVSTREAM_ROOT/scripts/devstream"
    if [ -f "$devstream_cli_source" ]; then
        cp "$devstream_cli_source" "$DEVSTREAM_HOME/bin/"
        chmod +x "$DEVSTREAM_HOME/bin/devstream"
        print_success "Installed: devstream CLI tool"
    else
        print_warning "devstream CLI not found at: $devstream_cli_source"
    fi

    # Install devstream-init
    local devstream_init_source="$DEVSTREAM_ROOT/scripts/devstream-init.py"
    if [ -f "$devstream_init_source" ]; then
        cp "$devstream_init_source" "$DEVSTREAM_HOME/bin/"
        chmod +x "$DEVSTREAM_HOME/bin/devstream-init.py"
        print_success "Installed: devstream-init.py"
    else
        print_warning "devstream-init.py not found at: $devstream_init_source"
    fi

    # Install other useful scripts
    local scripts_to_install=(
        "start-devstream.sh"
        "scan-codebase.py"
        "setup-db.py"
        "post-install.py"
    )

    for script in "${scripts_to_install[@]}"; do
        local source_script="$DEVSTREAM_ROOT/scripts/$script"
        if [ -f "$source_script" ]; then
            cp "$source_script" "$DEVSTREAM_HOME/bin/"
            chmod +x "$DEVSTREAM_HOME/bin/$script"
            print_verbose "Installed: $script"
        fi
    done

    print_success "CLI tools installation completed"
}

# Step 4: Install Hook System
install_hook_system() {
    print_header "Step 4: Installing Hook System"

    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY-RUN] Would install hook system to: $DEVSTREAM_HOME/hooks"
        return 0
    fi

    local hooks_source="$DEVSTREAM_ROOT/.claude/hooks"
    if [ -d "$hooks_source" ]; then
        # Copy all hooks
        cp -r "$hooks_source/"* "$DEVSTREAM_HOME/hooks/"
        print_success "Hook system installed"

        # Update hook paths to use global installation
        print_info "Updating hook paths for global installation..."
        find "$DEVSTREAM_HOME/hooks" -name "*.py" -type f -exec sed -i.bak \
            "s|$DEVSTREAM_ROOT/.devstream/bin/python|$DEVSTREAM_HOME/bin/python|g" {} \;

        # Remove backup files
        find "$DEVSTREAM_HOME/hooks" -name "*.py.bak" -delete

        print_success "Hook paths updated for global installation"
    else
        print_warning "Hook system not found at: $hooks_source"
    fi
}

# Step 5: Install Project Templates
install_templates() {
    print_header "Step 5: Installing Project Templates"

    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY-RUN] Would install project templates to: $DEVSTREAM_HOME/templates"
        return 0
    fi

    # Create basic project templates
    local template_types=(
        "python"
        "typescript"
        "go"
        "rust"
        "generic"
    )

    for template in "${template_types[@]}"; do
        local template_dir="$DEVSTREAM_HOME/templates/$template"
        mkdir -p "$template_dir"

        # Create basic template structure
        cat > "$template_dir/template.json" << EOF
{
  "name": "$template",
  "description": "Basic $template project template",
  "version": "1.0.0",
  "directories": ["src", "tests", "docs"],
  "files": {
    "README.md": "# ${template^} Project\\n\\nGenerated by DevStream\\n",
    ".gitignore": "*.log\\n__pycache__\\nnode_modules\\n.env\\n",
    ".devstream": "DevStream project marker\\n"
  },
  "scan_patterns": ["*.py", "*.ts", "*.go", "*.rs"],
  "exclude_patterns": [".git", "node_modules", "__pycache__", "dist", "build"]
}
EOF

        print_verbose "Created template: $template"
    done

    print_success "Project templates installed"
}

# Step 6: Create Global Configuration
create_global_config() {
    print_header "Step 6: Creating Global Configuration"

    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY-RUN] Would create global configuration in: $DEVSTREAM_HOME/config"
        return 0
    fi

    # Create main configuration file
    cat > "$DEVSTREAM_HOME/config/devstream.json" << EOF
{
  "version": "2.2.0",
  "installation_type": "global",
  "home_directory": "$DEVSTREAM_HOME",
  "default_template": "generic",
  "auto_scan": true,
  "registry_path": "$DEVSTREAM_HOME/data/registry.json",
  "log_directory": "$DEVSTREAM_HOME/logs",
  "hook_directory": "$DEVSTREAM_HOME/hooks",
  "template_directory": "$DEVSTREAM_HOME/templates"
}
EOF

    # Create default settings
    cat > "$DEVSTREAM_HOME/config/settings.json" << EOF
{
  "default_python_version": "3.11",
  "default_node_version": "20",
  "auto_update": false,
  "telemetry": false,
  "debug_mode": false,
  "max_projects": 50,
  "cache_size_mb": 1024
}
EOF

    print_success "Global configuration created"
}

# Step 7: Initialize Project Registry
initialize_registry() {
    print_header "Step 7: Initializing Project Registry"

    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY-RUN] Would initialize project registry"
        return 0
    fi

    # Create registry file
    cat > "$DEVSTREAM_HOME/data/registry.json" << EOF
{
  "version": "1.0.0",
  "created": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "last_updated": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "projects": [],
  "statistics": {
    "total_projects": 0,
    "active_projects": 0,
    "last_scan": null
  }
}
EOF

    # Set secure permissions
    chmod 600 "$DEVSTREAM_HOME/data/registry.json"
    print_verbose "Set permissions: 600 for registry.json"

    print_success "Project registry initialized"
}

# Step 8: Setup PATH Environment
setup_path_environment() {
    print_header "Step 8: Setting up PATH Environment"

    local shell_rc=""
    local path_export="export PATH=\"\$PATH:$DEVSTREAM_HOME/bin\""

    # Detect shell and appropriate rc file
    if [ -n "${ZSH_VERSION:-}" ]; then
        shell_rc="$HOME/.zshrc"
    elif [ -n "${BASH_VERSION:-}" ]; then
        shell_rc="$HOME/.bashrc"
        # Also check for .bash_profile
        if [ -f "$HOME/.bash_profile" ]; then
            shell_rc="$HOME/.bash_profile"
        fi
    else
        print_warning "Could not detect shell, manual PATH setup required"
        print_info "Add this to your shell configuration:"
        print_info "  $path_export"
        return 0
    fi

    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY-RUN] Would add PATH to: $shell_rc"
        return 0
    fi

    # Check if PATH already contains DevStream
    if echo "$PATH" | grep -q "$DEVSTREAM_HOME/bin"; then
        print_success "DevStream already in PATH"
        return 0
    fi

    # Add to shell rc file
    if [ -f "$shell_rc" ]; then
        if ! grep -q "devstream/bin" "$shell_rc"; then
            echo "" >> "$shell_rc"
            echo "# DevStream Global Installation" >> "$shell_rc"
            echo "$path_export" >> "$shell_rc"
            print_success "Added DevStream to PATH in $shell_rc"
        else
            print_info "DevStream PATH already configured in $shell_rc"
        fi
    else
        print_warning "Shell configuration file not found: $shell_rc"
        print_info "Add this manually to your shell configuration:"
        print_info "  $path_export"
    fi

    print_info "To use DevStream immediately, run:"
    print_info "  export PATH=\"\$PATH:$DEVSTREAM_HOME/bin\""
    print_info "Or restart your shell to load the new PATH"
}

# Step 9: Post-Installation Verification
verify_installation() {
    print_header "Step 9: Post-Installation Verification"

    local validation_passed=true
    local checks=(
        "Global Directory:$DEVSTREAM_HOME:test -d"
        "CLI Tools Directory:$DEVSTREAM_HOME/bin:test -d"
        "Hooks Directory:$DEVSTREAM_HOME/hooks:test -d"
        "Templates Directory:$DEVSTREAM_HOME/templates:test -d"
        "Config Directory:$DEVSTREAM_HOME/config:test -d"
        "Data Directory:$DEVSTREAM_HOME/data:test -d"
        "Registry File:$DEVSTREAM_HOME/data/registry.json:test -f"
        "DevStream Config:$DEVSTREAM_HOME/config/devstream.json:test -f"
        "Settings File:$DEVSTREAM_HOME/config/settings.json:test -f"
    )

    print_info "Running installation verification checks..."
    echo ""

    for check in "${checks[@]}"; do
        IFS=':' read -r name path test_cmd <<< "$check"

        case $test_cmd in
            "test -f")
                if [ -f "$path" ]; then
                    print_success "$name: ✓"
                else
                    print_error "$name: ✗ ($path not found)"
                    validation_passed=false
                fi
                ;;
            "test -d")
                if [ -d "$path" ]; then
                    print_success "$name: ✓"
                else
                    print_error "$name: ✗ ($path not found)"
                    validation_passed=false
                fi
                ;;
        esac
    done

    echo ""
    if [ "$validation_passed" = true ]; then
        print_success "✅ All verification checks passed!"
    else
        print_error "❌ Some verification checks failed"
        if [ "$NO_EXIT" = false ]; then
            exit 1
        fi
    fi

    # Test CLI tool if available
    if [ -f "$DEVSTREAM_HOME/bin/devstream" ]; then
        print_info "Testing devstream CLI..."
        if "$DEVSTREAM_HOME/bin/devstream" --help >/dev/null 2>&1; then
            print_success "devstream CLI is functional"
        else
            print_warning "devstream CLI may have issues"
        fi
    fi
}

# Step 10: Final Instructions
final_instructions() {
    print_header "Global Installation Complete!"

    echo ""
    echo -e "${GREEN}✓ DevStream has been successfully installed globally${NC}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Installation Summary"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Global Home:      $DEVSTREAM_HOME"
    echo "CLI Tools:        $DEVSTREAM_HOME/bin"
    echo "Hook System:      $DEVSTREAM_HOME/hooks"
    echo "Templates:        $DEVSTREAM_HOME/templates"
    echo "Configuration:    $DEVSTREAM_HOME/config"
    echo "Registry:         $DEVSTREAM_HOME/data/registry.json"
    echo ""

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Next Steps"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "1. 🔄 RESTART your shell or run:"
    echo "   export PATH=\"\$PATH:$DEVSTREAM_HOME/bin\""
    echo ""
    echo "2. 🚀 Initialize a new project:"
    echo "   cd /path/to/your/project"
    echo "   devstream init"
    echo ""
    echo "3. 📋 List your projects:"
    echo "   devstream list"
    echo ""
    echo "4. 🔍 Check installation:"
    echo "   devstream --version"
    echo ""

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Available Commands"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  devstream init              Initialize a new DevStream project"
    echo "  devstream list              List all registered projects"
    echo "  devstream scan              Scan current project for codebase"
    echo "  devstream status            Show DevStream status"
    echo "  devstream --help            Show all available commands"
    echo ""

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Configuration Files"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  Global Config:     $DEVSTREAM_HOME/config/devstream.json"
    echo "  Settings:          $DEVSTREAM_HOME/config/settings.json"
    echo "  Project Registry:  $DEVSTREAM_HOME/data/registry.json"
    echo ""

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Troubleshooting"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "• PATH issues: Restart your shell or manually add to PATH"
    echo "• Permission errors: Check file permissions in $DEVSTREAM_HOME"
    echo "• Hook errors: Check logs in $DEVSTREAM_HOME/logs"
    echo "• CLI issues: Verify $DEVSTREAM_HOME/bin/devstream is executable"
    echo ""
}

#------------------------------------------------------------------------------
# Main Execution
#------------------------------------------------------------------------------

main() {
    print_header "DevStream Global Installation Script v2.2.0"

    # Parse arguments
    parse_args "$@"

    # Show configuration
    print_info "Installation Configuration:"
    print_info "  Global Home: $DEVSTREAM_HOME"
    print_info "  DevStream Root: $DEVSTREAM_ROOT"
    echo ""

    # Run installation steps
    check_prerequisites
    create_global_structure
    install_cli_tools
    install_hook_system
    install_templates
    create_global_config
    initialize_registry
    setup_path_environment
    verify_installation

    final_instructions

    echo ""
    print_success "Global installation completed successfully!"
    print_info "Remember to RESTART your shell to use DevStream commands!"
    echo ""
}

# Run main function with all arguments
main "$@"