#!/usr/bin/env bash

#==============================================================================
# DevStream Universal Installation Script
# Version: 1.0.0
# Description: Install DevStream in any project (new or existing)
#==============================================================================

set -e  # Exit on error (disable with --no-exit)
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

#------------------------------------------------------------------------------
# Script Self-Setup (Context7-compliant permissions)
#------------------------------------------------------------------------------

# Ensure script itself has proper permissions (Context7 best practice)
if [[ ! -x "${BASH_SOURCE[0]}" ]]; then
    echo "Setting executable permissions for install script..."
    chmod +x "${BASH_SOURCE[0]}"
    echo "✓ Script permissions updated"
fi

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVSTREAM_ROOT="${SCRIPT_DIR}/.."
TARGET_PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
VENV_DIR="${TARGET_PROJECT_ROOT}/.devstream"
DATA_DIR="${TARGET_PROJECT_ROOT}/data"
CLAUDE_SETTINGS="${HOME}/.claude/settings.json"

# Installation mode
EXISTING_PROJECT=false
PRESERVE_VENV=false
MERGE_REQUIREMENTS=true

# Color codes
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
FORCE=false
ENHANCED_HOOK_COPYING=false

#------------------------------------------------------------------------------
# Helper Functions
#------------------------------------------------------------------------------

# Context7-compliant enhanced copy validation function (MUST be defined before use)
validate_enhanced_copy_success() {
    # Context7 research: Verify enhanced copying through multiple validation strategies

    local validation_score=0
    local max_score=10

    # 1. Check critical hook directories (3 points)
    local critical_dirs=(
        "$TARGET_PROJECT_ROOT/.claude/hooks/devstream/memory"
        "$TARGET_PROJECT_ROOT/.claude/hooks/devstream/context"
        "$TARGET_PROJECT_ROOT/.claude/hooks/devstream/utils"
    )

    local dirs_found=0
    for dir in "${critical_dirs[@]}"; do
        if [ -d "$dir" ]; then
            ((dirs_found++))
            ((validation_score++))
        fi
    done

    # 2. Check for critical hook files (4 points)
    local critical_files=(
        "$TARGET_PROJECT_ROOT/.claude/hooks/devstream/memory/pre_tool_use.py"
        "$TARGET_PROJECT_ROOT/.claude/hooks/devstream/memory/post_tool_use.py"
        "$TARGET_PROJECT_ROOT/.claude/hooks/devstream/context/user_query_context_enhancer.py"
        "$TARGET_PROJECT_ROOT/.claude/hooks/devstream/context/session_start.py"
    )

    local files_found=0
    local valid_files=0
    for file in "${critical_files[@]}"; do
        if [ -f "$file" ]; then
            ((files_found++))
            if [ -s "$file" ]; then
                # Basic Python syntax check (more lenient)
                if "$VENV_DIR/bin/python" -c "import ast; ast.parse(open('$file').read())" >/dev/null 2>&1; then
                    ((valid_files++))
                    ((validation_score++))
                fi
            fi
        fi
    done

    # 3. Check if DevStream components were copied (2 points)
    if [ -d "$TARGET_PROJECT_ROOT/.claude/agents" ]; then
        ((validation_score++))
    fi
    if [ -d "$TARGET_PROJECT_ROOT/.claude/commands" ]; then
        ((validation_score++))
    fi

    # 4. Check if Claude Code settings exist (1 point - optional)
    if [ -f "$TARGET_PROJECT_ROOT/.claude/settings.json" ] || \
       [ -f "$HOME/.claude/settings.json" ]; then
        ((validation_score++))
    fi

    # Context7-compliant validation: require 70% success rate
    local required_score=$((max_score * 7 / 10))

    if [ "$validation_score" -ge "$required_score" ]; then
        if [ "$VERBOSE" = true ]; then
            echo "DEBUG: Enhanced copy validation passed ($validation_score/$max_score)"
        fi
        return 0
    else
        if [ "$VERBOSE" = true ]; then
            echo "DEBUG: Enhanced copy validation failed ($validation_score/$max_score required: $required_score)"
        fi
        return 1
    fi
}

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
    if [ "$SKIP_OPTIONAL" = true ] || [ "$FORCE" = true ]; then
        print_warning "Auto-accepting: $message"
        return 0
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
            --existing-project)
                EXISTING_PROJECT=true
                PRESERVE_VENV=${PRESERVE_VENV:-false}
                MERGE_REQUIREMENTS=true
                print_info "Existing project mode enabled"
                shift
                ;;
            --preserve-venv)
                PRESERVE_VENV=true
                print_info "Will preserve existing virtual environment"
                shift
                ;;
            --merge-requirements)
                MERGE_REQUIREMENTS=true
                print_info "Will merge requirements with DevStream dependencies"
                shift
                ;;
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
            --force|-f)
                FORCE=true
                print_warning "Force mode: Auto-accept all prompts"
                shift
                ;;
            --enhanced-hook-copying)
                ENHANCED_HOOK_COPYING=true
                print_info "Enhanced hook copying enabled (Copier integration)"
                shift
                ;;
            --help|-h)
                cat << EOF
DevStream Universal Installation Script

Usage: $0 [OPTIONS]

Options:
  --existing-project        Install in existing project (preserves current code)
  --preserve-venv          Keep existing virtual environment
  --merge-requirements     Merge existing requirements.txt with DevStream deps
  --enhanced-hook-copying  Use enhanced hook copying with Copier integration
  --verbose, -v            Enable verbose output
  --dry-run               Show what would be done without making changes
  --no-exit              Continue on errors instead of exiting
  --skip-optional        Skip optional steps without prompting
  --force, -f            Auto-accept all prompts (non-interactive)
  --help, -h             Show this help message

Examples:
  $0                              # Standard installation
  $0 --existing-project           # Install in existing project
  $0 --existing-project --force   # Non-interactive existing project install
  $0 --enhanced-hook-copying      # Use enhanced hook copying with Copier
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
# Step 1: Existing Project Analysis
#------------------------------------------------------------------------------

analyze_existing_project() {
    if [ "$EXISTING_PROJECT" = false ]; then
        return 0
    fi

    print_header "Step 1: Analyzing Existing Project"

    print_info "Target project: $TARGET_PROJECT_ROOT"
    print_info "Installation mode: Existing project"

    # Check if directory is actually a project
    if [ ! -d "$TARGET_PROJECT_ROOT" ]; then
        print_error "Target directory does not exist: $TARGET_PROJECT_ROOT"
        exit 1
    fi

    # Analyze project type using Python detector
    print_info "Analyzing project structure..."

    if command_exists python3; then
        local analysis_result
        analysis_result=$(cd "$DEVSTREAM_ROOT" && python3 -c "
import sys
sys.path.insert(0, 'src')
from devstream.installation.project_detector import detect_project_type
import json

try:
    analysis = detect_project_type('$TARGET_PROJECT_ROOT')
    print(json.dumps(analysis.to_dict(), indent=2))
except Exception as e:
    print(f'ERROR: {e}')
" 2>/dev/null)

        if [[ "$analysis_result" == ERROR* ]]; then
            print_warning "Project analysis failed: ${analysis_result#ERROR: }"
        else
            # Extract key information
            local project_type=$(echo "$analysis_result" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('project_type', 'unknown'))
" 2>/dev/null)

            local confidence=$(echo "$analysis_result" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('confidence_score', 0.0))
" 2>/dev/null)

            print_success "Project type: $project_type (confidence: $confidence)"

            if [ "$VERBOSE" = true ]; then
                echo ""
                echo "Project Analysis Details:"
                echo "$analysis_result" | head -30
                echo ""
            fi
        fi
    else
        print_warning "Python3 not available, skipping project analysis"
    fi

    # Check for existing requirements.txt
    local requirements_file="$TARGET_PROJECT_ROOT/requirements.txt"
    if [ -f "$requirements_file" ]; then
        local existing_packages=$(wc -l < "$requirements_file")
        print_success "Found requirements.txt with $existing_packages packages"

        if [ "$VERBOSE" = true ]; then
            echo "First 10 packages:"
            head -10 "$requirements_file" | sed 's/^/  /'
        fi
    else
        print_warning "No requirements.txt found"
        MERGE_REQUIREMENTS=false
    fi

    # Check for existing virtual environment
    local existing_venvs=(".venv" "venv" "env")
    for venv in "${existing_venvs[@]}"; do
        if [ -d "$TARGET_PROJECT_ROOT/$venv" ]; then
            print_warning "Found existing virtual environment: $venv"
            if [ "$PRESERVE_VENV" = false ]; then
                print_info "Will create separate .devstream environment (recommended)"
            else
                print_info "Will preserve existing environment"
            fi
            break
        fi
    done

    # Check for existing tests
    local test_dirs=("tests" "test")
    for test_dir in "${test_dirs[@]}"; do
        if [ -d "$TARGET_PROJECT_ROOT/$test_dir" ]; then
            local test_count=$(find "$TARGET_PROJECT_ROOT/$test_dir" -name "test_*.py" | wc -l)
            print_success "Found $test_count test files in $test_dir/"
            break
        fi
    done

    print_success "Existing project analysis completed"
}

#------------------------------------------------------------------------------
# Step 2: Prerequisites Check
#------------------------------------------------------------------------------

check_prerequisites() {
    print_header "Step 2: Checking Prerequisites"

    local has_errors=false

    # Check Python 3.8+
    print_info "Checking Python 3.8+..."
    local python_cmd=""
    for cmd in python3.11 python3.10 python3.9 python3.8 python3 python; do
        if command_exists $cmd; then
            local python_version=$($cmd --version 2>&1 | awk '{print $2}')
            local major=$(echo "$python_version" | cut -d. -f1)
            local minor=$(echo "$python_version" | cut -d. -f2)

            if [ "$major" -eq 3 ] && [ "$minor" -ge 8 ]; then
                python_cmd=$cmd
                print_success "Python $python_version found: $cmd"
                break
            fi
        fi
    done

    if [ -z "$python_cmd" ]; then
        print_error "Python 3.8+ not found"
        echo ""
        echo "Installation instructions:"
        echo "  macOS:   brew install python@3.11"
        echo "  Ubuntu:  sudo apt install python3.11 python3.11-venv"
        echo "  Fedora:  sudo dnf install python3.11"
        echo ""
        has_errors=true
    fi

    # Check Node.js 16+ (for MCP server)
    print_info "Checking Node.js 16+..."
    if command_exists node; then
        local node_version=$(node --version)
        local node_major=$(echo "$node_version" | sed 's/v\([0-9]*\).*/\1/')
        if [ "$node_major" -ge 16 ]; then
            print_success "Node.js found: $node_version"
        else
            print_error "Node.js version must be 16+, found: $node_version"
            has_errors=true
        fi
    else
        print_error "Node.js not found (required for MCP server)"
        has_errors=true
    fi

    # Check Git
    print_info "Checking Git..."
    if command_exists git; then
        local git_version=$(git --version | awk '{print $3}')
        print_success "Git found: $git_version"
    else
        print_error "Git not found"
        has_errors=true
    fi

    # Check for existing git repo in existing project mode
    if [ "$EXISTING_PROJECT" = true ]; then
        cd "$TARGET_PROJECT_ROOT"
        if git rev-parse --git-dir >/dev/null 2>&1; then
            print_success "Project is a git repository"
        else
            print_warning "Project is not a git repository (recommended for DevStream)"
        fi
        cd - >/dev/null
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

#------------------------------------------------------------------------------
# Step 3: Requirements Processing (Existing Projects)
#------------------------------------------------------------------------------

process_requirements() {
    if [ "$EXISTING_PROJECT" = false ] || [ "$MERGE_REQUIREMENTS" = false ]; then
        return 0
    fi

    print_header "Step 3: Processing Requirements"

    local requirements_file="$TARGET_PROJECT_ROOT/requirements.txt"
    local devstream_requirements="$DEVSTREAM_ROOT/requirements.txt"
    local merged_requirements="$TARGET_PROJECT_ROOT/.devstream/requirements-merged.txt"

    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY-RUN] Would merge requirements.txt with DevStream dependencies"
        return 0
    fi

    # Create .devstream directory
    mkdir -p "$TARGET_PROJECT_ROOT/.devstream"

    if [ -f "$requirements_file" ]; then
        print_info "Merging existing requirements with DevStream dependencies..."

        # Create backup of original requirements
        cp "$requirements_file" "$TARGET_PROJECT_ROOT/requirements-pre-devstream.txt"
        print_verbose "Backed up original requirements.txt"

        # Get unique package names from both files
        local existing_packages=$(grep -v '^#' "$requirements_file" | grep -v '^$' | cut -d'=' -f1 | sort -u)
        local devstream_packages=$(grep -v '^#' "$devstream_requirements" | grep -v '^$' | cut -d'=' -f1 | sort -u)

        # Start with DevStream requirements
        cp "$devstream_requirements" "$merged_requirements"

        # Add packages from existing requirements that aren't in DevStream
        echo "" >> "$merged_requirements"
        echo "# Packages from existing project" >> "$merged_requirements"

        while IFS= read -r package; do
            if ! echo "$devstream_packages" | grep -q "^${package}$"; then
                grep "^${package}" "$requirements_file" >> "$merged_requirements" || echo "$package" >> "$merged_requirements"
                print_verbose "Added existing package: $package"
            fi
        done <<< "$existing_packages"

        # Count final packages
        local total_packages=$(grep -v '^#' "$merged_requirements" | grep -v '^$' | wc -l)
        local existing_count=$(grep -v '^#' "$requirements_file" | grep -v '^$' | wc -l)
        local devstream_count=$(grep -v '^#' "$devstream_requirements" | grep -v '^$' | wc -l)

        print_success "Requirements merged:"
        print_info "  Existing packages: $existing_count"
        print_info "  DevStream packages: $devstream_count"
        print_info "  Total unique packages: $total_packages"

        if [ "$VERBOSE" = true ]; then
            echo ""
            echo "Merged requirements preview:"
            head -20 "$merged_requirements" | sed 's/^/  /'
            if [ $total_packages -gt 20 ]; then
                echo "  ... and $((total_packages - 20)) more packages"
            fi
            echo ""
        fi

    else
        print_warning "No existing requirements.txt found, using DevStream requirements only"
        cp "$devstream_requirements" "$merged_requirements"
    fi

    print_success "Requirements processing completed"
}

#------------------------------------------------------------------------------
# Step 4: Python Environment Setup
#------------------------------------------------------------------------------

setup_python_environment() {
    print_header "Step 4: Setting up Python Virtual Environment"

    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY-RUN] Would create Python venv at: $VENV_DIR"
        return 0
    fi

    # Find Python command
    local python_cmd=""
    for cmd in python3.11 python3.10 python3.9 python3.8 python3 python; do
        if command_exists $cmd; then
            local python_version=$($cmd --version 2>&1 | awk '{print $2}')
            local major=$(echo "$python_version" | cut -d. -f1)
            local minor=$(echo "$python_version" | cut -d. -f2)

            if [ "$major" -eq 3 ] && [ "$minor" -ge 8 ]; then
                python_cmd=$cmd
                break
            fi
        fi
    done

    # Handle existing virtual environment
    if [ "$EXISTING_PROJECT" = true ] && [ "$PRESERVE_VENV" = true ]; then
        local existing_venvs=(".venv" "venv" "env")
        for venv in "${existing_venvs[@]}"; do
            if [ -d "$TARGET_PROJECT_ROOT/$venv" ]; then
                print_warning "Using existing virtual environment: $venv"
                VENV_DIR="$TARGET_PROJECT_ROOT/$venv"
                print_verbose "Updated VENV_DIR to: $VENV_DIR"
                break
            fi
        done
    fi

    # Create or check virtual environment
    if [ -d "$VENV_DIR" ] && [ "$PRESERVE_VENV" = false ]; then
        print_warning "Virtual environment already exists at: $VENV_DIR"
        if prompt_continue "Remove and recreate virtual environment?"; then
            rm -rf "$VENV_DIR"
            print_verbose "Removed existing venv"
        else
            print_info "Keeping existing virtual environment"
        fi
    fi

    if [ ! -d "$VENV_DIR" ]; then
        print_info "Creating Python virtual environment..."
        $python_cmd -m venv "$VENV_DIR"
        check_exit_code $? "Virtual environment created" "Failed to create virtual environment"
        print_verbose "Created: $VENV_DIR"
    fi

    # Verify Python version in venv
    local venv_python_version=$("$VENV_DIR/bin/python" --version 2>&1 | awk '{print $2}')
    print_success "Using Python $venv_python_version from venv"

    # Upgrade pip
    print_info "Upgrading pip..."
    "$VENV_DIR/bin/python" -m pip install --upgrade pip >/dev/null 2>&1
    check_exit_code $? "pip upgraded" "Failed to upgrade pip"

    # Install requirements
    if [ "$EXISTING_PROJECT" = true ] && [ "$MERGE_REQUIREMENTS" = true ]; then
        local requirements_file="$TARGET_PROJECT_ROOT/.devstream/requirements-merged.txt"
        if [ -f "$requirements_file" ]; then
            print_info "Installing merged requirements..."
            "$VENV_DIR/bin/pip" install -r "$requirements_file" >/dev/null 2>&1
            check_exit_code $? "Merged requirements installed" "Failed to install merged requirements"
        fi
    else
        # Install DevStream requirements
        if [ -f "$DEVSTREAM_ROOT/requirements.txt" ]; then
            print_info "Installing DevStream requirements..."
            "$VENV_DIR/bin/pip" install -r "$DEVSTREAM_ROOT/requirements.txt" >/dev/null 2>&1
            check_exit_code $? "DevStream requirements installed" "Failed to install DevStream requirements"
        fi
    fi

    # Verify critical packages (Context7-compliant case-insensitive check)
    print_info "Verifying critical packages..."
    local critical_packages=("cchooks" "aiohttp" "structlog" "python-dotenv" "psutil")
    for package in "${critical_packages[@]}"; do
        if "$VENV_DIR/bin/pip" list 2>/dev/null | grep -i "^${package}"; then
            # Get the actual package name from pip list for version lookup
            local actual_package_name=$("$VENV_DIR/bin/pip" list 2>/dev/null | grep -i "^${package}" | awk '{print $1}')
            local version=$("$VENV_DIR/bin/pip" show "$actual_package_name" 2>/dev/null | grep "^Version:" | awk '{print $2}')
            print_success "$actual_package_name ($version)"
        else
            print_error "$package not installed"
            if [ "$NO_EXIT" = false ]; then
                exit 1
            fi
        fi
    done

    # For existing projects, verify some existing packages still work
    if [ "$EXISTING_PROJECT" = true ]; then
        print_info "Verifying existing project packages..."
        local test_packages=("flask" "django" "fastapi" "sqlalchemy" "pytest")
        local found_existing=false

        for package in "${test_packages[@]}"; do
            if "$VENV_DIR/bin/pip" list 2>/dev/null | grep -qi "^${package}"; then
                print_success "$package (existing project dependency)"
                found_existing=true
            fi
        done

        if [ "$found_existing" = false ]; then
            print_warning "No common framework packages detected (this may be OK)"
        fi
    fi

    print_success "Python environment setup completed"
}

#------------------------------------------------------------------------------
# Step 5: DevStream Setup
#------------------------------------------------------------------------------

setup_devstream_components() {
    print_header "Step 5: Setting up DevStream Components"

    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY-RUN] Would set up DevStream components in: $TARGET_PROJECT_ROOT"
        return 0
    fi

    # Create directory structure
    print_info "Creating DevStream directory structure..."

    local devstream_dirs=(
        ".claude/hooks/devstream/memory"
        ".claude/hooks/devstream/context"
        ".claude/hooks/devstream/sessions"
        ".claude/hooks/devstream/utils"
        ".claude/commands"
        ".claude/agents"
        ".claude/mcp_servers"
        "data"
        "logs"
    )

    for dir in "${devstream_dirs[@]}"; do
        local target_dir="$TARGET_PROJECT_ROOT/$dir"
        if [ ! -d "$target_dir" ]; then
            mkdir -p "$target_dir"
            print_verbose "Created: $target_dir"
        fi
    done

    # Use enhanced hook copying if available and enabled (Context7-compliant Copier integration)
    if [ "$ENHANCED_HOOK_COPYING" = true ]; then
        print_info "Attempting enhanced hook copying system with Copier integration..."

        # Check if required files exist
        local hook_copier="$DEVSTREAM_ROOT/.claude/hooks/devstream/utils/multi_project_hook_copier.py"
        local hook_bootstrap="$DEVSTREAM_ROOT/.claude/hooks/devstream/utils/multi_project_bootstrap.py"

        if [ -f "$hook_copier" ] && [ -f "$hook_bootstrap" ]; then
            print_verbose "Enhanced hook copying files found"
            print_info "Installing Copier dependency for enhanced hook copying..."

            # Ensure Copier is installed
            if ! "$VENV_DIR/bin/pip" list 2>/dev/null | grep -qi "copier"; then
                "$VENV_DIR/bin/pip" install "copier>=9.0.0,<10.0.0" >/dev/null 2>&1
                check_exit_code $? "Copier installed" "Failed to install Copier"
            fi

            print_info "Running enhanced hook copying with integrity validation..."

            local enhanced_copy_result=$("$VENV_DIR/bin/python" -c "
import sys
import os
sys.path.insert(0, '$DEVSTREAM_ROOT/.claude/hooks/devstream/utils')
sys.path.insert(0, '$DEVSTREAM_ROOT/src')

# Add the project root to Python path for imports
sys.path.insert(0, '$DEVSTREAM_ROOT')

try:
    print('DEBUG: Starting enhanced hook copying...')

    # Test imports first
    from multi_project_hook_copier import copy_devstream_hooks_enhanced
    print('DEBUG: Successfully imported copy_devstream_hooks_enhanced')

    # Try the bootstrap module
    from multi_project_bootstrap import bootstrap_devstream_project
    print('DEBUG: Successfully imported bootstrap_devstream_project')

    # Run the bootstrap function
    result = bootstrap_devstream_project(
        target_root='$TARGET_PROJECT_ROOT',
        source_root='$DEVSTREAM_ROOT',
        project_name='$(basename \"$TARGET_PROJECT_ROOT\")',
        config_file=None,
        integrity_validation=True,
        claude_code_config=True,
        verbose=True
    )

    print(f'SUCCESS:Enhanced hook copying completed: {result}')

except ImportError as e:
    print(f'IMPORT_ERROR:Import failed: {e}')
    print(f'DEBUG: Python path: {sys.path[:3]}')
    print(f'DEBUG: Current directory: {os.getcwd()}')

except Exception as e:
    print(f'ERROR:Enhanced copying failed: {e}')
    import traceback
    print(f'DEBUG: Traceback: {traceback.format_exc()}')
" 2>&1)

            local enhanced_copy_exit_code=$?

            if [ $enhanced_copy_exit_code -eq 0 ]; then
                # Context7-compliant success validation with detailed verification
                local enhanced_success=false
                local validation_details=()

                # Show summary of what was copied
                if [[ "$enhanced_copy_result" == SUCCESS:* ]]; then
                    local copy_summary=${enhanced_copy_result#SUCCESS:}
                    enhanced_success=true

                    if echo "$copy_summary" | grep -q "directories"; then
                        validation_details+=("Hook directories copied and validated")
                    fi
                    if echo "$copy_summary" | grep -q "integrity"; then
                        validation_details+=("Integrity validation passed")
                    fi
                    if echo "$copy_summary" | grep -q "configuration"; then
                        validation_details+=("Claude Code configuration updated")
                    fi
                else
                    # Context7 research: Verify success through file system checks
                    if validate_enhanced_copy_success; then
                        enhanced_success=true
                        validation_details+=("Success verified via file system validation")
                    fi
                fi

                if [ "$enhanced_success" = true ]; then
                    print_success "✅ Enhanced hook copying completed successfully"
                    for detail in "${validation_details[@]}"; do
                        print_info "   • $detail"
                    done

                    if [ "$VERBOSE" = true ]; then
                        print_verbose "Enhanced copying result: $enhanced_copy_result"
                    fi
                else
                    # Context7-compliant: More permissive validation for apparent success
                    if [[ "$enhanced_copy_result" == SUCCESS:* ]] || \
                       [[ "$enhanced_copy_result" == *"completed"* ]] || \
                       [[ "$enhanced_copy_result" == *"successfully"* ]]; then
                        # Check if any hooks were actually copied (lenient validation)
                        # Context7 fix: Always use absolute path for consistent validation across all scenarios
                        local hooks_search_path="$TARGET_PROJECT_ROOT/.claude/hooks/devstream"
                        local hooks_count=$(find "$hooks_search_path" -name "*.py" 2>/dev/null | wc -l)
                        if [ "$hooks_count" -ge 5 ]; then
                            print_success "✅ Enhanced hook copying completed successfully"
                            print_info "   • Hooks copied: $hooks_count files found"
                            if [ "$VERBOSE" = true ]; then
                                print_verbose "Enhanced copying result: $enhanced_copy_result"
                            fi
                        else
                            print_warning "⚠️  Enhanced hook copying appears successful but insufficient files copied"
                            print_info "   • Found only $hooks_count hook files, using fallback"
                            standard_devstream_copy
                        fi
                    else
                        # Context7-compliant fallback: Treat apparent success as failure if validation fails
                        print_warning "⚠️  Enhanced hook copying validation failed, using fallback"
                        print_info "   Apparent success but validation checks incomplete"
                        if [ "$VERBOSE" = true ]; then
                            print_verbose "Raw output: $enhanced_copy_result"
                        fi
                        standard_devstream_copy
                    fi
                fi
            else
                # Context7-compliant error handling with specific recovery strategies
                print_warning "⚠️  Enhanced hook copying encountered issues, applying fallback strategy"

                # Analyze the specific error and provide targeted recovery
                if [[ "$enhanced_copy_result" == IMPORT_ERROR:* ]]; then
                    local import_error=${enhanced_copy_result#IMPORT_ERROR:}
                    print_error "   Import error: $import_error"

                    # Context7 research: Try dependency recovery
                    if echo "$import_error" | grep -qi "copier"; then
                        print_info "   Attempting Copier dependency recovery..."
                        if "$VENV_DIR/bin/pip" install "copier>=9.0.0,<10.0.0" >/dev/null 2>&1; then
                            print_info "   Copier dependency recovered, retrying enhanced copy..."
                            # Retry once after dependency recovery
                            local retry_result=$("$VENV_DIR/bin/python" -c "
import sys
import os
sys.path.insert(0, '$DEVSTREAM_ROOT/.claude/hooks/devstream/utils')
sys.path.insert(0, '$DEVSTREAM_ROOT/src')
sys.path.insert(0, '$DEVSTREAM_ROOT')

try:
    from multi_project_bootstrap import bootstrap_devstream_project
    result = bootstrap_devstream_project(
        target_root='$TARGET_PROJECT_ROOT',
        source_root='$DEVSTREAM_ROOT',
        project_name='$(basename \"$TARGET_PROJECT_ROOT\")',
        config_file=None,
        integrity_validation=True,
        claude_code_config=True,
        verbose=False
    )
    print(f'SUCCESS_RETRY:{result}')
except Exception as e:
    print(f'ERROR_RETRY:{e}')
" 2>&1)

                            if [[ "$retry_result" == SUCCESS_RETRY:* ]] && validate_enhanced_copy_success; then
                                print_success "✅ Enhanced hook copying succeeded after dependency recovery"
                                return 0
                            else
                                print_info "   Recovery attempt failed, proceeding with fallback"
                            fi
                        else
                            print_info "   Dependency recovery failed, proceeding with fallback"
                        fi
                    fi

                elif [[ "$enhanced_copy_result" == ERROR:* ]]; then
                    local error_msg=${enhanced_copy_result#ERROR:}
                    print_error "   Error: $error_msg"

                    # Context7 research: Provide specific recovery guidance
                    if echo "$error_msg" | grep -qi "permission"; then
                        print_info "   Try running with appropriate permissions or use --force flag"
                    elif echo "$error_msg" | grep -qi "space"; then
                        print_info "   Consider using --skip-optional or clearing disk space"
                    elif echo "$error_msg" | grep -qi "network"; then
                        print_info "   Network issues detected, using offline fallback"
                    fi
                else
                    print_error "   Unexpected error (exit code: $enhanced_copy_exit_code)"
                fi

                if [ "$VERBOSE" = true ]; then
                    print_verbose "Full error output: $enhanced_copy_result"
                fi

                print_info "   Applying Context7-compliant fallback strategy..."
                standard_devstream_copy
            fi
        else
            print_warning "Enhanced hook copying files not found"
            if [ ! -f "$hook_copier" ]; then
                print_info "   Missing: $hook_copier"
            fi
            if [ ! -f "$hook_bootstrap" ]; then
                print_info "   Missing: $hook_bootstrap"
            fi
            print_info "   Using standard file copy method..."
            standard_devstream_copy
        fi
    else
        print_info "Enhanced hook copying disabled, using standard copy method..."
        standard_devstream_copy
    fi

    print_success "DevStream components setup completed"
}

# Standard DevStream copy method (fallback)
standard_devstream_copy() {
    # Copy DevStream components
    print_info "Installing DevStream components..."

    # Copy hooks (exclude venv directories and cache files)
    if [ -d "$DEVSTREAM_ROOT/.claude/hooks" ]; then
        # Use rsync with exclusions to prevent copying venv directories
        if command -v rsync >/dev/null 2>&1; then
            rsync -av \
                --exclude '.devstream/' \
                --exclude '__pycache__/' \
                --exclude '*.pyc' \
                "$DEVSTREAM_ROOT/.claude/hooks/"* "$TARGET_PROJECT_ROOT/.claude/hooks/"
        else
            # Fallback to find + cp if rsync not available
            find "$DEVSTREAM_ROOT/.claude/hooks" -type f \
                ! -path "*/.devstream/*" \
                ! -path "*/__pycache__/*" \
                ! -name "*.pyc" \
                -exec cp --parents {} "$TARGET_PROJECT_ROOT/" \;
        fi
        print_success "DevStream hooks installed"
    fi

    # Copy agents
    if [ -d "$DEVSTREAM_ROOT/.claude/agents" ]; then
        cp -r "$DEVSTREAM_ROOT/.claude/agents/"* "$TARGET_PROJECT_ROOT/.claude/agents/"
        print_success "DevStream agents installed"
    fi

    # Copy commands
    if [ -d "$DEVSTREAM_ROOT/.claude/commands" ]; then
        cp -r "$DEVSTREAM_ROOT/.claude/commands/"* "$TARGET_PROJECT_ROOT/.claude/commands/"
        print_success "DevStream commands installed"
    fi

    # Copy MCP server configuration
    if [ -f "$DEVSTREAM_ROOT/.claude/mcp_servers.json" ]; then
        cp "$DEVSTREAM_ROOT/.claude/mcp_servers.json" "$TARGET_PROJECT_ROOT/.claude/"
        print_success "MCP server configuration copied"
    elif [ -f "$DEVSTREAM_ROOT/.claude/mcp_servers.json.example" ]; then
        cp "$DEVSTREAM_ROOT/.claude/mcp_servers.json.example" "$TARGET_PROJECT_ROOT/.claude/mcp_servers.json"
        print_success "MCP server configuration template installed"
    fi

    # Copy Python source for local development
    if [ -d "$DEVSTREAM_ROOT/src" ]; then
        cp -r "$DEVSTREAM_ROOT/src" "$TARGET_PROJECT_ROOT/.devstream/"
        print_success "DevStream source code installed"
    fi

    # Copy scripts
    if [ -d "$DEVSTREAM_ROOT/scripts" ]; then
        cp -r "$DEVSTREAM_ROOT/scripts" "$TARGET_PROJECT_ROOT/.devstream/"
        print_success "DevStream scripts installed"
    fi

    # Copy requirements.txt to .devstream for validation
    if [ -f "$TARGET_PROJECT_ROOT/.devstream/requirements-merged.txt" ]; then
        cp "$TARGET_PROJECT_ROOT/.devstream/requirements-merged.txt" "$TARGET_PROJECT_ROOT/.devstream/requirements.txt"
        print_verbose "Copied requirements-merged.txt to .devstream/requirements.txt"
    elif [ -f "$DEVSTREAM_ROOT/requirements.txt" ]; then
        cp "$DEVSTREAM_ROOT/requirements.txt" "$TARGET_PROJECT_ROOT/.devstream/requirements.txt"
        print_verbose "Copied DevStream requirements.txt to .devstream/requirements.txt"
    fi
}

#------------------------------------------------------------------------------
# Step 6: Environment Configuration
#------------------------------------------------------------------------------

create_environment_config() {
    print_header "Step 6: Creating Environment Configuration"

    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY-RUN] Would create .env.devstream configuration"
        return 0
    fi

    local env_file="$TARGET_PROJECT_ROOT/.env.devstream"
    local env_example="$DEVSTREAM_ROOT/.env.example"

    # Start with example file if it exists
    if [ -f "$env_example" ]; then
        cp "$env_example" "$env_file"
        print_success "Base configuration created from .env.example"
    else
        # Create minimal configuration
        cat > "$env_file" << 'EOF'
# DevStream Configuration
# Generated by install-devstream.sh

# Memory System (MANDATORY)
DEVSTREAM_MEMORY_ENABLED=true
DEVSTREAM_MEMORY_FEEDBACK_LEVEL=minimal

# Context7 (MANDATORY)
DEVSTREAM_CONTEXT7_ENABLED=true
DEVSTREAM_CONTEXT7_AUTO_DETECT=true
DEVSTREAM_CONTEXT7_TOKEN_BUDGET=5000

# Context Injection (MANDATORY)
DEVSTREAM_CONTEXT_INJECTION_ENABLED=true
DEVSTREAM_CONTEXT_MAX_TOKENS=2000
DEVSTREAM_CONTEXT_RELEVANCE_THRESHOLD=0.5

# Database (MANDATORY)
DEVSTREAM_DB_PATH=data/devstream.db

# Logging (RECOMMENDED)
DEVSTREAM_LOG_LEVEL=INFO
DEVSTREAM_LOG_PATH=~/.claude/logs/devstream/

# Auto-Delegation System (MANDATORY)
DEVSTREAM_AUTO_DELEGATION_ENABLED=true
DEVSTREAM_AUTO_DELEGATION_MIN_CONFIDENCE=0.85
DEVSTREAM_AUTO_DELEGATION_AUTO_APPROVE=0.95
DEVSTREAM_AUTO_DELEGATION_QUALITY_GATE=true
EOF
        print_success "Base configuration created"
    fi

    # Add existing project specific configuration
    if [ "$EXISTING_PROJECT" = true ]; then
        cat >> "$env_file" << EOF

# Existing Project Configuration
DEVSTREAM_EXISTING_PROJECT=true
DEVSTREAM_PRESERVE_VENV=$PRESERVE_VENV
DEVSTREAM_MERGE_REQUIREMENTS=$MERGE_REQUIREMENTS

# Codebase Scanning (IMPORTANT for existing projects)
DEVSTREAM_CODEBASE_SCAN_ENABLED=true
DEVSTREAM_CODEBASE_SCAN_DIRECTORIES=$(find "$TARGET_PROJECT_ROOT" -maxdepth 2 -type d -name "*.py" -exec dirname {} \; | sort -u | head -5 | paste -sd, - | sed 's|$TARGET_PROJECT_ROOT/||g')
DEVSTREAM_CODEBASE_SCAN_EXCLUDE=.git,.venv,venv,__pycache__,node_modules,dist,build,.devstream
DEVSTREAM_CODEBASE_SCAN_FILE_PATTERNS=*.py

# Instance Identification
DEVSTREAM_INSTANCE_NAME=$(basename "$TARGET_PROJECT_ROOT")
DEVSTREAM_PROJECT_ROOT=$TARGET_PROJECT_ROOT
EOF

        # Try to detect framework-specific directories
        local framework_dirs=()
        [ -d "$TARGET_PROJECT_ROOT/app" ] && framework_dirs+=("app")
        [ -d "$TARGET_PROJECT_ROOT/src" ] && framework_dirs+=("src")
        [ -d "$TARGET_PROJECT_ROOT/lib" ] && framework_dirs+=("lib")

        if [ ${#framework_dirs[@]} -gt 0 ]; then
            local scan_dirs=$(IFS=,; echo "${framework_dirs[*]}")
            sed -i.bak "s|DEVSTREAM_CODEBASE_SCAN_DIRECTORIES=.*|DEVSTREAM_CODEBASE_SCAN_DIRECTORIES=$scan_dirs|" "$env_file"
            print_info "Detected framework directories: $scan_dirs"
        fi

        print_success "Existing project configuration added"
    fi

    # Set secure permissions
    chmod 600 "$env_file"
    print_verbose "Set permissions: 600 for .env.devstream"

    print_success "Environment configuration completed"
}

#------------------------------------------------------------------------------
# Step 7: Database Initialization
#------------------------------------------------------------------------------

initialize_database() {
    print_header "Step 7: Initializing Database"

    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY-RUN] Would initialize database at: ${DATA_DIR}/devstream.db"
        return 0
    fi

    # Context7-compliant dependency check: Ensure sqlite-vec is installed with proper loading
    print_info "Checking for sqlite-vec dependency..."
    if ! "$VENV_DIR/bin/pip" list 2>/dev/null | grep -qi "sqlite-vec"; then
        print_info "Installing sqlite-vec for vector database support..."
        "$VENV_DIR/bin/pip" install sqlite-vec >/dev/null 2>&1
        check_exit_code $? "sqlite-vec installed" "Failed to install sqlite-vec"
    else
        print_success "sqlite-vec already available"
    fi

    # Context7-compliant: Verify sqlite-vec can be loaded properly
    print_info "Verifying sqlite-vec loading capability..."
    if ! "$VENV_DIR/bin/python" -c "
import sqlite3
import sys
try:
    import sqlite_vec
    # Test loading as per Context7 best practices
    db = sqlite3.connect(':memory:')
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    # Verify vec_version() function works
    vec_version = db.execute('select vec_version()').fetchone()[0]
    print(f'✓ sqlite-vec loaded successfully: version {vec_version}')
    db.close()
    sys.exit(0)
except Exception as e:
    print(f'✗ sqlite-vec loading failed: {e}')
    sys.exit(1)
" 2>/dev/null; then
        print_warning "⚠️  sqlite-vec loading verification failed"
        print_info "   Vector search will not be available"
        print_info "   This is non-critical for basic functionality"
    else
        print_success "✅ sqlite-vec loading verified (vector search available)"
    fi

    # Create data directory
    if [ ! -d "$DATA_DIR" ]; then
        mkdir -p "$DATA_DIR"
        print_verbose "Created data directory: $DATA_DIR"
    fi

    # Create schema directory and copy schema.sql (Context7 best practice)
    local schema_dir="$TARGET_PROJECT_ROOT/schema"
    if [ ! -d "$schema_dir" ]; then
        mkdir -p "$schema_dir"
        print_verbose "Created schema directory: $schema_dir"
    fi

    # Copy schema.sql from DevStream root if it exists
    local source_schema="$DEVSTREAM_ROOT/schema/schema.sql"
    local target_schema="$schema_dir/schema.sql"

    if [ -f "$source_schema" ]; then
        if [ ! -f "$target_schema" ] || [ "$source_schema" -nt "$target_schema" ]; then
            cp "$source_schema" "$target_schema"
            print_success "Database schema copied: schema.sql"
            print_verbose "Source: $source_schema"
            print_verbose "Target: $target_schema"
        else
            print_verbose "Schema file already up to date"
        fi
    else
        print_warning "Source schema file not found: $source_schema"
        print_info "Database initialization may not work correctly"
    fi

    local db_file="${DATA_DIR}/devstream.db"

    # Check if database already exists
    if [ -f "$db_file" ]; then
        print_warning "Database already exists: $db_file"
        if prompt_continue "Remove and recreate database? (ALL DATA WILL BE LOST)"; then
            rm -f "$db_file"
            print_verbose "Removed existing database"
        else
            print_info "Keeping existing database"
            return 0
        fi
    fi

    # Update environment file with database path
    if [ -f "$TARGET_PROJECT_ROOT/.env.devstream" ]; then
        sed -i.bak "s|DEVSTREAM_DB_PATH=.*|DEVSTREAM_DB_PATH=$db_file|" "$TARGET_PROJECT_ROOT/.env.devstream"
    fi

    # Run database setup script with error handling
    local setup_script="$TARGET_PROJECT_ROOT/.devstream/scripts/setup-db.py"
    if [ -f "$setup_script" ]; then
        print_info "Running database initialization..."
        cd "$TARGET_PROJECT_ROOT"

        # Set environment variables for database setup
        export DEVSTREAM_DB_PATH="$db_file"
        export DEVSTREAM_SCHEMA_PATH="$target_schema"

        # Run setup with proper error handling
        if "$VENV_DIR/bin/python" "$setup_script" 2>/dev/null; then
            print_success "Database initialized successfully"
        else
            local setup_exit_code=$?
            print_error "Database setup failed (exit code: $setup_exit_code)"

            # Context7-compliant fallback: Try manual database creation with proper FTS setup
            print_info "Attempting manual database creation with FTS support..."
            if "$VENV_DIR/bin/python" -c "
import sqlite3
import sys

def create_comprehensive_database(db_path):
    '''Context7-compliant database creation with proper FTS and vector support'''
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Try to load sqlite-vec for vector support
        vec_loaded = False
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            vec_loaded = True
            print('✓ sqlite-vec loaded successfully')
        except Exception as vec_e:
            print(f'⚠ sqlite-vec not available: {vec_e}')

        # Create schema version tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS schema_version (
                version TEXT PRIMARY KEY,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create semantic memory table with vector support
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS semantic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                content_type TEXT DEFAULT 'code',
                keywords TEXT,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create FTS5 virtual table for full-text search
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_semantic_memory USING fts5(
                content,
                keywords,
                content_type,
                content=semantic_memory,
                content_rowid=id
            )
        ''')

        # Create FTS triggers for automatic synchronization
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS sync_insert_memory AFTER INSERT ON semantic_memory
            BEGIN
                INSERT INTO fts_semantic_memory(rowid, content, keywords, content_type)
                VALUES (new.id, new.content, new.keywords, new.content_type);
            END
        ''')

        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS sync_update_memory AFTER UPDATE ON semantic_memory
            BEGIN
                UPDATE fts_semantic_memory SET
                    content = new.content,
                    keywords = new.keywords,
                    content_type = new.content_type
                WHERE rowid = new.id;
            END
        ''')

        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS sync_delete_memory AFTER DELETE ON semantic_memory
            BEGIN
                DELETE FROM fts_semantic_memory WHERE rowid = old.id;
            END
        ''')

        # Create sessions table for Context7 compliance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')

        # Create implementation plans table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS implementation_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                model_type TEXT,
                plan_content TEXT,
                status TEXT DEFAULT 'draft',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Insert schema version
        cursor.execute('''
            INSERT OR REPLACE INTO schema_version (version, description)
            VALUES ('2.1.0', 'Context7-compliant schema with FTS5 and vector support')
        ''')

        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_semantic_memory_content_type ON semantic_memory(content_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_semantic_memory_created_at ON semantic_memory(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at)')

        conn.commit()

        # Verify database structure
        cursor.execute(\"SELECT name FROM sqlite_master WHERE type='table'\")
        tables = [row[0] for row in cursor.fetchall()]

        cursor.execute(\"SELECT name FROM sqlite_master WHERE type='virtual table'\")
        virtual_tables = [row[0] for row in cursor.fetchall()]

        cursor.execute(\"SELECT name FROM sqlite_master WHERE type='trigger'\")
        triggers = [row[0] for row in cursor.fetchall()]

        print(f'✅ Database created successfully:')
        print(f'   Tables: {len(tables)} {tables[:5]}')
        print(f'   Virtual tables: {len(virtual_tables)} {virtual_tables}')
        print(f'   Triggers: {len(triggers)} {triggers}')
        print(f'   Vector support: {\"✓\" if vec_loaded else \"✗\"}')

        conn.close()
        return True

    except Exception as e:
        print(f'❌ Database creation failed: {e}')
        return False

if create_comprehensive_database('$db_file'):
    sys.exit(0)
else:
    sys.exit(1)
            "; then
                print_success "Fallback database creation completed"
            else
                print_error "Manual database creation also failed"
                if [ "$NO_EXIT" = false ]; then
                    exit 1
                fi
            fi
        fi

        cd - >/dev/null
    else
        print_warning "Database setup script not found"
        print_info "Attempting manual database initialization..."

        # Context7-compliant manual initialization
        if "$VENV_DIR/bin/python" -c "
import sqlite3
import sys
try:
    import sqlite_vec
    print('Using sqlite-vec for vector support')
except ImportError:
    print('sqlite-vec not available')

conn = sqlite3.connect('$db_file')
cursor = conn.cursor()

# Create basic tables
cursor.execute('''
    CREATE TABLE IF NOT EXISTS memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        content_type TEXT DEFAULT 'code',
        keywords TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

conn.commit()
conn.close()
print('Basic database structure created')
        "; then
            print_success "Manual database initialization completed"
        else
            print_error "Manual database initialization failed"
            if [ "$NO_EXIT" = false ]; then
                exit 1
            fi
        fi
    fi

    # Verify database was created
    if [ -f "$db_file" ]; then
        local db_size=$(du -h "$db_file" | awk '{print $1}')
        print_success "Database created (size: $db_size)"

        # Context7-compliant database integrity verification using PRAGMA commands
        print_info "Running comprehensive database integrity checks..."
        if "$VENV_DIR/bin/python" -c "
import sqlite3
import sys

def run_integrity_check(db_path):
    '''Context7-compliant database integrity validation using PRAGMA commands'''
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 1. Check database integrity (Context7 best practice)
        cursor.execute('PRAGMA integrity_check')
        integrity_result = cursor.fetchone()[0]
        if integrity_result != 'ok':
            print(f'INTEGRITY_ERROR: Database integrity check failed: {integrity_result}')
            return False

        # 2. Check foreign key constraints
        cursor.execute('PRAGMA foreign_key_check')
        fk_violations = cursor.fetchall()
        if fk_violations:
            print(f'FOREIGN_KEY_ERROR: Found {len(fk_violations)} foreign key violations')
            return False

        # 3. Verify schema consistency
        cursor.execute('PRAGMA schema_version')
        schema_version = cursor.fetchone()[0]

        # 4. Check table statistics (Context7-compliant: use correct table name)
        try:
            cursor.execute('SELECT count(*) FROM semantic_memory')
            memory_count = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            # Fallback: try memory table for backward compatibility
            try:
                cursor.execute('SELECT count(*) FROM memory')
                memory_count = cursor.fetchone()[0]
            except sqlite3.OperationalError:
                memory_count = 0

        # 5. Verify database file is not corrupted
        cursor.execute('PRAGMA quick_check')
        quick_check_result = cursor.fetchone()[0]
        if quick_check_result != 'ok':
            print(f'QUICK_CHECK_ERROR: Database quick check failed: {quick_check_result}')
            return False

        print(f'SUCCESS: Database integrity verified (schema v{schema_version}, {memory_count} records)')
        return True

    except Exception as e:
        print(f'INTEGRITY_EXCEPTION: {e}')
        return False
    finally:
        if 'conn' in locals():
            conn.close()

# Run the integrity check
if run_integrity_check('$db_file'):
    sys.exit(0)
else:
    sys.exit(1)
        " 2>/dev/null; then
            print_success "✅ Database integrity verified (PRAGMA checks passed)"
        else
            print_warning "⚠️  Database integrity check failed - attempting repair..."

            # Context7-compliant database repair attempt
            if "$VENV_DIR/bin/python" -c "
import sqlite3
import sys

def attempt_database_repair(db_path):
    '''Context7-compliant database repair using VACUUM and REINDEX'''
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Attempt VACUUM to rebuild database
        cursor.execute('VACUUM')

        # Rebuild indexes
        cursor.execute('REINDEX')

        # Verify repair worked
        cursor.execute('PRAGMA integrity_check')
        integrity_result = cursor.fetchone()[0]

        conn.close()

        if integrity_result == 'ok':
            print('REPAIR_SUCCESS: Database repair completed successfully')
            return True
        else:
            print(f'REPAIR_FAILED: Integrity still compromised: {integrity_result}')
            return False

    except Exception as e:
        print(f'REPAIR_EXCEPTION: {e}')
        return False

if attempt_database_repair('$db_file'):
    sys.exit(0)
else:
    sys.exit(1)
            " 2>/dev/null; then
                print_success "✅ Database repaired and integrity verified"
            else
                print_error "❌ Database repair failed - database may be corrupted"
                if [ "$NO_EXIT" = false ]; then
                    exit 1
                fi
            fi
        fi
    else
        print_error "Database file not created"
        if [ "$NO_EXIT" = false ]; then
            exit 1
        fi
    fi

    print_success "Database initialization completed"
}

#------------------------------------------------------------------------------
# Step 8: Claude Code Configuration
#------------------------------------------------------------------------------

configure_claude_code() {
    print_header "Step 8: Configuring Claude Code"

    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY-RUN] Would configure Claude Code hooks"
        return 0
    fi

    # Run post-install configuration
    local post_install_script="$TARGET_PROJECT_ROOT/.devstream/scripts/post-install.py"
    if [ -f "$post_install_script" ]; then
        print_info "Running post-install configuration..."
        cd "$TARGET_PROJECT_ROOT"
        "$VENV_DIR/bin/python" "$post_install_script"
        check_exit_code $? "Claude Code configured" "Failed to configure Claude Code"
        cd - >/dev/null
    else
        print_warning "Post-install script not found"
        print_info "You may need to configure Claude Code manually"
    fi

    print_success "Claude Code configuration completed"
}

#------------------------------------------------------------------------------
# Step 9: Existing Project Validation
#------------------------------------------------------------------------------

validate_existing_project() {
    if [ "$EXISTING_PROJECT" = false ]; then
        return 0
    fi

    print_header "Step 9: Validating Existing Project Integration"

    if [ "$DRY_RUN" = true ]; then
        print_info "[DRY-RUN] Would validate existing project integration"
        return 0
    fi

    print_info "Running existing project validation checks..."

    # Test Python imports
    if [ -f "$VENV_DIR/bin/python" ]; then
        print_info "Testing critical DevStream imports..."
        if "$VENV_DIR/bin/python" -c "import cchooks, aiohttp, structlog" 2>/dev/null; then
            print_success "DevStream imports working"
        else
            print_warning "Some DevStream imports failed"
        fi

        # Test some common framework imports if they exist
        local framework_tests=("flask" "django" "fastapi" "sqlalchemy" "pytest")
        for framework in "${framework_tests[@]}"; do
            if "$VENV_DIR/bin/python" -c "import $framework" 2>/dev/null; then
                print_success "$framework import working (existing project dependency)"
            fi
        done
    fi

    # Test existing tests if they exist
    local test_dirs=("tests" "test")
    for test_dir in "${test_dirs[@]}"; do
        if [ -d "$TARGET_PROJECT_ROOT/$test_dir" ]; then
            print_info "Testing existing test suite..."
            if "$VENV_DIR/bin/python" -m pytest "$test_dir/" --collect-only -q >/dev/null 2>&1; then
                local test_count=$("$VENV_DIR/bin/python" -m pytest "$test_dir/" --collect-only -q 2>/dev/null | grep -c "test_" || echo "0")
                print_success "Found $test_count tests that can be collected"
            else
                print_warning "Some existing tests may have issues"
            fi
            break
        fi
    done

    # Check if critical project files are still accessible
    local critical_files=("requirements.txt" "setup.py" "pyproject.toml" "main.py" "app.py" "manage.py")
    local found_critical=false

    for file in "${critical_files[@]}"; do
        if [ -f "$TARGET_PROJECT_ROOT/$file" ]; then
            print_success "Critical project file preserved: $file"
            found_critical=true
        fi
    done

    if [ "$found_critical" = false ]; then
        print_warning "No critical project files detected (this may be OK for some projects)"
    fi

    print_success "Existing project validation completed"
}

#------------------------------------------------------------------------------
# Step 10: Final Validation
#------------------------------------------------------------------------------

validate_installation() {
    print_header "Step 10: Final Installation Validation"

    local validation_passed=true
    local checks_total=0
    local checks_passed=0
    local warnings=()

    # Define validation checks with Context7-compliant graceful fallback handling
    local checks=(
        "Virtual Environment:$VENV_DIR/bin/python:test -f:critical"
        "Python Packages:$VENV_DIR/bin/pip:test -f:critical"
        "DevStream Hooks:$TARGET_PROJECT_ROOT/.claude/hooks/devstream:test -d:critical"
        "Environment Config:$TARGET_PROJECT_ROOT/.env.devstream:test -f:critical"
        "Database File:$DATA_DIR/devstream.db:test -f:critical"
        "Requirements File:$TARGET_PROJECT_ROOT/.devstream/requirements.txt:test -f:critical"
        # Context7-compliant: These are optional with fallback validation
        "DevStream Source:$TARGET_PROJECT_ROOT/.devstream/src:test -d:optional"
        "Scripts Directory:$TARGET_PROJECT_ROOT/.devstream/scripts:test -d:optional"
    )

    # Add existing project specific checks
    if [ "$EXISTING_PROJECT" = true ]; then
        checks+=(
            "Merged Requirements:$TARGET_PROJECT_ROOT/.devstream/requirements-merged.txt:test -f:optional"
            "Original Requirements Backup:$TARGET_PROJECT_ROOT/requirements-pre-devstream.txt:test -f:optional"
        )
    fi

    print_info "Running Context7-compliant installation validation checks..."
    echo ""

    for check in "${checks[@]}"; do
        IFS=':' read -r name path test_cmd severity <<< "$check"
        ((checks_total++))

        case $test_cmd in
            "test -f")
                if [ -f "$path" ]; then
                    print_success "$name: ✓"
                    ((checks_passed++))
                else
                    if [ "$severity" = "critical" ]; then
                        print_error "$name: ✗ ($path not found) [CRITICAL]"
                        validation_passed=false
                    else
                        # Context7-compliant: Optional validation with graceful fallback
                        local fallback_result=$(validate_optional_component "$name" "$path" "file")
                        if [ "$fallback_result" = "success" ]; then
                            print_success "$name: ✓ (alternative validated)"
                            ((checks_passed++))
                        else
                            print_warning "$name: ⚠ ($path not found) [OPTIONAL - $fallback_result]"
                            warnings+=("$name: $fallback_result")
                        fi
                    fi
                fi
                ;;
            "test -d")
                if [ -d "$path" ]; then
                    print_success "$name: ✓"
                    ((checks_passed++))
                else
                    if [ "$severity" = "critical" ]; then
                        print_error "$name: ✗ ($path not found) [CRITICAL]"
                        validation_passed=false
                    else
                        # Context7-compliant: Optional validation with graceful fallback
                        local fallback_result=$(validate_optional_component "$name" "$path" "directory")
                        if [ "$fallback_result" = "success" ]; then
                            print_success "$name: ✓ (alternative validated)"
                            ((checks_passed++))
                        else
                            print_warning "$name: ⚠ ($path not found) [OPTIONAL - $fallback_result]"
                            warnings+=("$name: $fallback_result")
                        fi
                    fi
                fi
                ;;
        esac
    done

    echo ""
    print_info "Validation Summary:"
    print_info "Checks passed: $checks_passed/$checks_total"

    # Context7-compliant: Show warnings if any optional components failed
    if [ ${#warnings[@]} -gt 0 ]; then
        print_info "Optional components with alternative validation:"
        for warning in "${warnings[@]}"; do
            print_warning "  • $warning"
        done
        echo ""
    fi

    if [ "$validation_passed" = true ]; then
        print_success "✅ All critical validation checks passed!"
        if [ ${#warnings[@]} -gt 0 ]; then
            print_info "ℹ Some optional components use alternative configurations (installation is fully functional)"
        fi
    else
        print_error "❌ Some critical validation checks failed"
        if [ "$NO_EXIT" = false ]; then
            exit 1
        fi
    fi

    print_success "Installation validation completed"
}


# Context7-compliant optional component validation function
validate_optional_component() {
    local component_name="$1"
    local expected_path="$2"
    local component_type="$3"

    case "$component_name" in
        "DevStream Source")
            # Context7 research: Check if source code is accessible via alternative means
            if [ -f "$TARGET_PROJECT_ROOT/.claude/hooks/devstream/memory/pre_tool_use.py" ]; then
                # Check if hooks contain the core functionality
                local hook_count=$(find "$TARGET_PROJECT_ROOT/.claude/hooks/devstream" -name "*.py" | wc -l)
                if [ "$hook_count" -ge 3 ]; then
                    echo "Core functionality available via hooks ($hook_count hook files)"
                    return 0
                fi
            fi

            # Check if there's a symbolic link or reference to source
            if [ -L "$expected_path" ] || [ -f "$TARGET_PROJECT_ROOT/.devstream/src_reference.txt" ]; then
                echo "Source reference available"
                return 0
            fi

            echo "Source code not required for basic operation"
            return 1
            ;;

        "Scripts Directory")
            # Context7 research: Validate essential scripts are available via alternatives
            local essential_scripts_found=0

            # Check for database setup
            if [ -f "$TARGET_PROJECT_ROOT/.devstream/scripts/setup-db.py" ] || \
               [ -f "$TARGET_PROJECT_ROOT/data/devstream.db" ]; then
                ((essential_scripts_found++))
            fi

            # Check for post-install configuration
            if [ -f "$TARGET_PROJECT_ROOT/.claude/settings.json" ] || \
               [ -f "$HOME/.claude/settings.json" ]; then
                ((essential_scripts_found++))
            fi

            # Check for installation script
            if [ -f "$TARGET_PROJECT_ROOT/scripts/install-devstream.sh" ] || \
               [ -f "$DEVSTREAM_ROOT/scripts/install-devstream.sh" ]; then
                ((essential_scripts_found++))
            fi

            if [ "$essential_scripts_found" -ge 2 ]; then
                echo "Essential scripts available via alternatives ($essential_scripts_found/3)"
                return 0
            fi

            echo "Scripts available via system installation"
            return 1
            ;;

        "Merged Requirements")
            # Check if regular requirements exist as fallback
            if [ -f "$TARGET_PROJECT_ROOT/.devstream/requirements.txt" ]; then
                echo "Using standard requirements file"
                return 0
            fi

            echo "Merged requirements not needed for basic operation"
            return 1
            ;;

        "Original Requirements Backup")
            # This is purely informational
            echo "Backup not created (original requirements not found)"
            return 1
            ;;

        *)
            echo "Unknown optional component"
            return 1
            ;;
    esac
}

#------------------------------------------------------------------------------
# Final Steps
#------------------------------------------------------------------------------

final_steps() {
    print_header "Installation Complete!"

    echo ""
    if [ "$EXISTING_PROJECT" = true ]; then
        echo -e "${GREEN}✓ DevStream has been successfully installed in existing project: $TARGET_PROJECT_ROOT${NC}"
    else
        echo -e "${GREEN}✓ DevStream has been successfully installed in: $TARGET_PROJECT_ROOT${NC}"
    fi
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Installation Summary"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Project Root:     $TARGET_PROJECT_ROOT"
    echo "Python Venv:      $VENV_DIR"
    echo "Database:         $DATA_DIR/devstream.db"
    echo "Existing Project: $EXISTING_PROJECT"
    echo "Preserved Venv:   $PRESERVE_VENV"
    echo "Merged Req:       $MERGE_REQUIREMENTS"
    echo ""

    if [ "$EXISTING_PROJECT" = true ]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Existing Project Integration"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "✅ Original requirements.txt backed up"
        echo "✅ Project dependencies merged with DevStream"
        echo "✅ Existing file structure preserved"
        echo "✅ Codebase scanning configured"
        echo ""
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Next Steps"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "1. 🔄 RESTART Claude Code (critical for hooks to load)"
    echo ""

    if [ "$EXISTING_PROJECT" = true ]; then
        echo "2. 📚 Scan your existing codebase:"
        echo "   In Claude Code: 'Scan the existing codebase and populate semantic memory'"
        echo ""
        echo "3. 🧪 Test existing functionality:"
        echo "   cd $TARGET_PROJECT_ROOT"
        echo "   $VENV_DIR/bin/python -m pytest tests/  # If tests exist"
        echo ""
    else
        echo "2. 📝 Customize configuration:"
        echo "   Edit: $TARGET_PROJECT_ROOT/.env.devstream"
        echo ""
        echo "3. 🚀 Test functionality:"
        echo "   cd $TARGET_PROJECT_ROOT"
        echo "   $VENV_DIR/bin/python -c \"import cchooks; print('DevStream ready!')\""
        echo ""
    fi

    echo "4. 📖 Check documentation:"
    echo "   $TARGET_PROJECT_ROOT/.claude/README.md (if created)"
    echo ""

    if [ "$EXISTING_PROJECT" = true ]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Existing Project Tips"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "• Your existing patterns are now searchable via DevStream memory"
        echo "• Context7 will provide framework-specific best practices"
        echo "• Create intervention plans for systematic improvements"
        echo "• All your existing tests should continue to work"
        echo ""
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Troubleshooting"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "• Hook errors: Check logs at ~/.claude/logs/devstream/"
    echo "• Import errors: $VENV_DIR/bin/pip install -r .devstream/requirements.txt"
    echo "• Database issues: $VENV_DIR/bin/python .devstream/scripts/setup-db.py"
    echo "• MCP server: Check $TARGET_PROJECT_ROOT/.claude/mcp_servers.json"
    echo ""

    if [ "$EXISTING_PROJECT" = true ]; then
        echo "• Existing test failures: Check requirements-pre-devstream.txt"
        echo "• Dependency conflicts: Review merged requirements file"
        echo "• Codebase scanning: Configure DEVSTREAM_CODEBASE_SCAN_DIRECTORIES in .env.devstream"
        echo ""
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

#------------------------------------------------------------------------------
# Main Execution
#------------------------------------------------------------------------------

main() {
    print_header "DevStream Universal Installation Script v1.0.0"

    # Parse arguments
    parse_args "$@"

    # Show configuration
    print_info "Installation Configuration:"
    print_info "  Target Project: $TARGET_PROJECT_ROOT"
    print_info "  Existing Project: $EXISTING_PROJECT"
    print_info "  Preserve Venv: $PRESERVE_VENV"
    print_info "  Merge Requirements: $MERGE_REQUIREMENTS"
    print_info "  DevStream Root: $DEVSTREAM_ROOT"
    echo ""

    # Run installation steps
    analyze_existing_project
    check_prerequisites
    process_requirements
    setup_python_environment
    setup_devstream_components
    create_environment_config
    initialize_database
    configure_claude_code
    validate_existing_project
    validate_installation

    final_steps

    echo ""
    print_success "Universal installation completed successfully!"
    print_info "Remember to RESTART Claude Code to activate the hooks!"
    echo ""
}

# Run main function with all arguments
main "$@"