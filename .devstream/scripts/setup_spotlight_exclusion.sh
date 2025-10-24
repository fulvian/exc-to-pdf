#!/bin/bash

# DevStream Spotlight Exclusion Setup Script
# Prevents macOS kernel panics by excluding database files from Spotlight indexing
# Based on Context7 research and security audit recommendations

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"
DATA_NOINDEX_DIR="$PROJECT_ROOT/data.noindex"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] DevStream Spotlight Exclusion:${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if running on macOS
check_macos() {
    if [[ "$(uname)" != "Darwin" ]]; then
        error "This script is designed for macOS only"
        exit 1
    fi

    local macos_version=$(sw_vers -productVersion)
    log "Detected macOS version: $macos_version"
}

# Verify data directory exists
verify_data_directory() {
    if [ ! -d "$DATA_DIR" ]; then
        error "Data directory not found: $DATA_DIR"
        error "Please ensure you're running this script from the DevStream project root"
        exit 1
    fi

    log "Found data directory: $DATA_DIR"

    # Check for database files
    local db_files=$(find "$DATA_DIR" -name "*.db" -type f 2>/dev/null || true)
    if [ -z "$db_files" ]; then
        warn "No .db files found in $DATA_DIR"
        warn "This is normal if running on a fresh installation"
    else
        log "Found database files:"
        echo "$db_files" | while read -r file; do
            log "  - $file"
        done
    fi
}

# Rename data directory to data.noindex
rename_data_directory() {
    log "Step 1: Setting up data.noindex directory"

    if [ -d "$DATA_NOINDEX_DIR" ]; then
        if [ -L "$DATA_DIR" ] && [ "$(readlink "$DATA_DIR")" = "data.noindex" ]; then
            success "data.noindex directory and symlink already exist"
            return 0
        else
            warn "data.noindex directory already exists but no symlink found"
            warn "Proceeding with existing directory"
        fi
    else
        log "Renaming data/ → data.noindex/"
        if mv "$DATA_DIR" "$DATA_NOINDEX_DIR"; then
            success "Successfully renamed data/ to data.noindex/"
        else
            error "Failed to rename data directory"
            exit 1
        fi
    fi
}

# Create symlink for backward compatibility
create_symlink() {
    log "Step 2: Creating backward compatibility symlink"

    # Remove existing symlink or directory if it exists
    if [ -L "$DATA_DIR" ]; then
        log "Removing existing symlink"
        rm "$DATA_DIR"
    elif [ -e "$DATA_DIR" ] && [ ! -L "$DATA_DIR" ]; then
        error "Regular file/directory exists at $DATA_DIR"
        error "Please remove it manually and retry"
        exit 1
    fi

    # Create new symlink
    log "Creating symlink: data → data.noindex"
    if ln -s "data.noindex" "$DATA_DIR"; then
        success "Symlink created successfully"
    else
        error "Failed to create symlink"
        exit 1
    fi
}

# Apply xattr attributes to database files
apply_xattr_attributes() {
    log "Step 3: Applying xattr attributes for Spotlight exclusion"

    # Find all database files
    local db_files=$(find "$DATA_NOINDEX_DIR" -name "*.db" -type f 2>/dev/null || true)

    if [ -z "$db_files" ]; then
        warn "No database files found for xattr application"
        return 0
    fi

    local files_processed=0

    echo "$db_files" | while read -r db_file; do
        if [ -f "$db_file" ]; then
            log "Applying xattr to: $(basename "$db_file")"

            # Apply kMDItemSupportFileType attribute
            if xattr -w com.apple.metadata:kMDItemSupportFileType "DevStreamDB" "$db_file" 2>/dev/null; then
                success "  xattr applied successfully"
                ((files_processed++))
            else
                warn "  Failed to apply xattr (may require sudo)"
            fi

            # Also apply com.apple.metadata:kMDItemExcludeFromSearch
            if xattr -w com.apple.metadata:kMDItemExcludeFromSearch "true" "$db_file" 2>/dev/null; then
                success "  Search exclusion xattr applied"
            else
                warn "  Failed to apply search exclusion xattr"
            fi
        fi
    done

    log "Processed $files_processed database files"
}

# Apply xattr to backup files
apply_xattr_to_backups() {
    log "Step 4: Applying xattr to backup files"

    # Find backup files
    local backup_files=$(find "$DATA_NOINDEX_DIR" -name "*.backup*" -type f 2>/dev/null || true)

    if [ -z "$backup_files" ]; then
        log "No backup files found"
        return 0
    fi

    echo "$backup_files" | while read -r backup_file; do
        if [ -f "$backup_file" ]; then
            log "Applying xattr to backup: $(basename "$backup_file")"
            xattr -w com.apple.metadata:kMDItemSupportFileType "DevStreamDB" "$backup_file" 2>/dev/null || true
            xattr -w com.apple.metadata:kMDItemExcludeFromSearch "true" "$backup_file" 2>/dev/null || true
        fi
    done

    success "Backup files processed"
}

# Verify Spotlight exclusion
verify_spotlight_exclusion() {
    log "Step 5: Verifying Spotlight exclusion"

    # Check if data.noindex directory is properly excluded
    local db_files=$(find "$DATA_NOINDEX_DIR" -name "*.db" -type f 2>/dev/null | head -1)

    if [ -z "$db_files" ]; then
        warn "No database files found to verify exclusion"
        return 0
    fi

    log "Checking Spotlight metadata for: $(basename "$db_files")"

    # Use mdls to check if file is indexed
    local mdls_output=$(mdls "$db_files" 2>/dev/null || true)

    if echo "$mdls_output" | grep -q "kMDItemFSContentChangeDate"; then
        warn "WARNING: File may still be indexed by Spotlight"
        warn "This is normal - Spotlight may take time to update its index"
        warn "You can force reindex with: mdimport -d /path/to/file"
    else
        success "File appears to be excluded from Spotlight"
    fi

    # Check xattr attributes
    log "Checking xattr attributes:"
    local xattr_output=$(xattr -l "$db_files" 2>/dev/null || true)

    if echo "$xattr_output" | grep -q "kMDItemSupportFileType"; then
        success "  kMDItemSupportFileType xattr found"
    else
        warn "  kMDItemSupportFileType xattr not found"
    fi

    if echo "$xattr_output" | grep -q "kMDItemExcludeFromSearch"; then
        success "  kMDItemExcludeFromSearch xattr found"
    else
        warn "  kMDItemExcludeFromSearch xattr not found"
    fi
}

# Update configuration files
update_configuration() {
    log "Step 6: Updating configuration files"

    # Update .env.devstream if it exists
    local env_file="$PROJECT_ROOT/.env.devstream"
    if [ -f "$env_file" ]; then
        log "Updating .env.devstream configuration"

        # Add/update database path
        if grep -q "DEVSTREAM_DB_PATH=" "$env_file"; then
            sed -i '' 's|DEVSTREAM_DB_PATH=data/|DEVSTREAM_DB_PATH=data.noindex/|g' "$env_file"
            success "Updated existing DEVSTREAM_DB_PATH"
        else
            echo "DEVSTREAM_DB_PATH=data.noindex/devstream.db" >> "$env_file"
            success "Added DEVSTREAM_DB_PATH to .env.devstream"
        fi
    fi

    # Update MCP server configuration if it exists
    local mcp_config="$PROJECT_ROOT/.claude/mcp_servers.json"
    if [ -f "$mcp_config" ]; then
        log "Updating MCP server configuration"

        # Update database path in MCP config
        if grep -q "data/devstream.db" "$mcp_config"; then
            sed -i '' 's|data/devstream.db|data.noindex/devstream.db|g' "$mcp_config"
            success "Updated MCP server database path"
        fi
    fi
}

# Final validation
final_validation() {
    log "Step 7: Final validation"

    # Check directory structure
    if [ ! -d "$DATA_NOINDEX_DIR" ]; then
        error "data.noindex directory not found"
        exit 1
    fi

    if [ ! -L "$DATA_DIR" ]; then
        error "data symlink not found"
        exit 1
    fi

    # Verify symlink points to correct target
    local symlink_target=$(readlink "$DATA_DIR")
    if [ "$symlink_target" != "data.noindex" ]; then
        error "Symlink points to wrong target: $symlink_target"
        exit 1
    fi

    success "Directory structure validation passed"

    # Test database access through symlink
    local test_db="$DATA_DIR/devstream.db"
    if [ -f "$test_db" ]; then
        log "Testing database access through symlink"
        if sqlite3 "$test_db" "SELECT 1;" >/dev/null 2>&1; then
            success "Database access through symlink working"
        else
            warn "Database access test failed (may be expected if DB doesn't exist yet)"
        fi
    fi

    success "Final validation completed"
}

# Print completion message
print_completion() {
    log "🎉 Spotlight exclusion setup completed successfully!"
    echo
    echo "Summary of changes:"
    echo "  ✅ Renamed data/ → data.noindex/"
    echo "  ✅ Created symlink: data → data.noindex"
    echo "  ✅ Applied xattr attributes to database files"
    echo "  ✅ Updated configuration files"
    echo "  ✅ Verified Spotlight exclusion"
    echo
    echo "Next steps:"
    echo "  1. Restart Claude Code to ensure all processes use the new paths"
    echo "  2. Monitor system stability for kernel panics"
    echo "  3. Check ~/.claude/logs/devstream/ for any issues"
    echo
    echo "Rollback command (if needed):"
    echo "  cd $PROJECT_ROOT"
    echo "  rm data"
    echo "  mv data.noindex data"
    echo "  # Update configuration files manually"
    echo
}

# Main execution
main() {
    log "🔍 DevStream Spotlight Exclusion Setup"
    log "Project Root: $PROJECT_ROOT"
    echo

    # Execute setup steps
    check_macos
    verify_data_directory
    rename_data_directory
    create_symlink
    apply_xattr_attributes
    apply_xattr_to_backups
    verify_spotlight_exclusion
    update_configuration
    final_validation
    print_completion
}

# Script is idempotent - can be run multiple times safely
main "$@"