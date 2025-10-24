#!/bin/bash

# DevStream MCP Reconnection Helper
#
# Purpose: Force Claude Code to reload .mcp.json configuration
# Use case: When MCP tools become unavailable after /compact or session issues
#
# Research-backed pattern:
# - Claude Code watches .mcp.json for file modifications
# - Touching the file triggers file watcher reload
# - This is a fallback when server stays alive but client disconnects
#
# Source: DevStream Context7 research (2025-10-02)

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_success() {
  echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
  echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
  echo -e "${RED}❌${NC} $1"
}

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo ""
echo "🔄 DevStream MCP Reconnection Helper"
echo "======================================"
echo ""

# Step 1: Verify .mcp.json exists
if [ ! -f "$PROJECT_ROOT/.mcp.json" ]; then
  print_error ".mcp.json not found in project root: $PROJECT_ROOT"
  echo ""
  echo "Expected location: $PROJECT_ROOT/.mcp.json"
  echo ""
  exit 1
fi

print_success "Found .mcp.json in project root"

# Step 2: Touch file to trigger reload
touch "$PROJECT_ROOT/.mcp.json"
print_success "Triggered .mcp.json file watcher (touch)"

# Step 3: Verify MCP server is running
echo ""
echo "🔍 Checking MCP server status..."

# Check for DevStream MCP process (stdio-based, no HTTP)
if pgrep -f "devstream-mcp-server" >/dev/null 2>&1; then
  MCP_PID=$(pgrep -f "devstream-mcp-server" | head -1)
  print_success "MCP server is running (stdio process PID: $MCP_PID)"

  # Verify database connectivity via process check
  if ps -p $MCP_PID -o command= | grep -q "devstream.db"; then
    print_success "MCP server database connection verified"
  else
    print_warning "MCP server running but database path unclear"
  fi
else
  print_error "MCP server is NOT running"
  echo ""
  echo "To start the server, run:"
  echo "  ./start-devstream.sh"
  echo ""
  exit 1
fi

# Step 4: Instructions for user
echo ""
echo "📋 Next Steps:"
echo "=============="
echo ""
echo "1. Check if Claude Code detected the reload:"
echo "   - Look for MCP reconnection messages in Claude Code"
echo ""
echo "2. If tools are still unavailable:"
echo "   ${YELLOW}Restart Claude Code${NC} (close and reopen)"
echo ""
echo "3. Verify tools after restart:"
echo "   - Try: list available MCP tools"
echo "   - Test: mcp__devstream__devstream_list_tasks"
echo ""

print_success "Reconnection trigger complete!"
echo ""
