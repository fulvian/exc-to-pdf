#!/bin/bash
# Integration test runner for DevStream multi-project architecture

set -e

echo "🧪 DevStream Multi-Project Integration Tests"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set up test environment
export DEVSTREAM_HOME="$HOME/.devstream-test-$$"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Clean up function
cleanup() {
    if [ -d "$DEVSTREAM_HOME" ]; then
        echo "🧹 Cleaning up test environment: $DEVSTREAM_HOME"
        rm -rf "$DEVSTREAM_HOME"
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Ensure test dependencies are available
echo "📦 Checking test dependencies..."

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo "⚠️  pytest not found, using python -m unittest instead"
    TEST_RUNNER="python3 -m unittest"
else
    TEST_RUNNER="pytest"
fi

# Create test environment
echo "🏗️  Setting up test environment..."
mkdir -p "$DEVSTREAM_HOME/data"
mkdir -p "$DEVSTREAM_HOME/logs"

# Run integration tests
echo "🚀 Running integration tests..."
cd "$PROJECT_ROOT"

# Execute tests
if [ "$TEST_RUNNER" = "pytest" ]; then
    $TEST_RUNNER tests/integration/test_accountabilly_mult_project.py -v \
        --tb=short \
        --color=yes \
        --durations=10
else
    $TEST_RUNNER tests.integration.test_accountabilly_mult_project -v
fi

# Check test results
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All integration tests passed!"
    echo ""
    echo "🎯 Multi-Project Architecture Verification Complete:"
    echo "  ✅ Global Installation System"
    echo "  ✅ Project Detection System"
    echo "  ✅ Project Initialization with Codebase Scanning"
    echo "  ✅ Startup Script Multi-Project Support"
    echo "  ✅ Project Isolation"
    echo "  ✅ Database Migration Compatibility"
    echo "  ✅ Error Handling and Recovery"
    echo "  ✅ Parallel Session Support"
else
    echo ""
    echo "❌ Some integration tests failed!"
    echo "Please check the output above for details."
    exit 1
fi

echo ""
echo "🏆 Integration testing completed successfully!"