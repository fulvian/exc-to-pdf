#!/usr/bin/env bash
###############################################################################
# DevStream Production Monitoring Script (macOS)
# Task: d744c555 - Production Monitoring for macOS Crash Fix Deployment
#
# PURPOSE: Lightweight monitoring for hook execution, rate limiting,
#          performance metrics, and crash detection using existing logs.
#
# MONITORING SCOPE:
# 1. Hook execution health (error rates, failures)
# 2. Rate limiter effectiveness (memory/Ollama throttling)
# 3. Performance metrics (memory usage, execution time)
# 4. Crash detection (kernel panic indicators, memory pressure)
# 5. MCP server health (process status, reconnections)
#
# USAGE:
#   ./scripts/monitor-devstream.sh                    # Run once
#   ./scripts/monitor-devstream.sh --watch            # Continuous monitoring (5s refresh)
#   ./scripts/monitor-devstream.sh --alerts           # Show alerts only
#   ./scripts/monitor-devstream.sh --export-metrics   # Export JSON metrics
#
# DEPENDENCIES: None (pure bash + standard macOS tools)
###############################################################################

set -euo pipefail

# Configuration
LOG_DIR="${HOME}/.claude/logs/devstream"
ALERT_THRESHOLD_HOOK_ERROR_RATE=5  # Percent
ALERT_THRESHOLD_MEMORY_GB=8         # Gigabytes
ALERT_THRESHOLD_RATE_LIMIT_THROTTLE=30  # Percent
WATCH_MODE=false
ALERTS_ONLY=false
EXPORT_METRICS=false

# Colors for output
RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RESET='\033[0m'

###############################################################################
# Parse Arguments
###############################################################################

while [[ $# -gt 0 ]]; do
    case "$1" in
        --watch)
            WATCH_MODE=true
            shift
            ;;
        --alerts)
            ALERTS_ONLY=true
            shift
            ;;
        --export-metrics)
            EXPORT_METRICS=true
            shift
            ;;
        --help)
            echo "DevStream Production Monitoring"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --watch            Continuous monitoring (5s refresh)"
            echo "  --alerts           Show alerts only"
            echo "  --export-metrics   Export metrics as JSON"
            echo "  --help             Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

###############################################################################
# Helper Functions
###############################################################################

# Print colored status
status_ok() {
    echo -e "${GREEN}✅ $1${RESET}"
}

status_warning() {
    echo -e "${YELLOW}⚠️  $1${RESET}"
}

status_error() {
    echo -e "${RED}❌ $1${RESET}"
}

status_info() {
    echo -e "${BLUE}ℹ️  $1${RESET}"
}

# Get timestamp
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

###############################################################################
# Monitoring Functions
###############################################################################

# 1. Hook Execution Monitoring
monitor_hooks() {
    local hook_errors=0
    local hook_total=0
    local hook_warnings=0

    # Count errors/warnings in hook logs (last 100 lines)
    for hook_log in "${LOG_DIR}"/*.jsonl "${LOG_DIR}"/*.log; do
        if [[ -f "$hook_log" ]]; then
            hook_total=$((hook_total + $(tail -100 "$hook_log" 2>/dev/null | wc -l)))
            hook_errors=$((hook_errors + $(tail -100 "$hook_log" 2>/dev/null | grep -c "ERROR" || true)))
            hook_warnings=$((hook_warnings + $(tail -100 "$hook_log" 2>/dev/null | grep -c "WARNING" || true)))
        fi
    done

    # Calculate error rate
    local error_rate=0
    if [[ $hook_total -gt 0 ]]; then
        error_rate=$(awk "BEGIN {printf \"%.2f\", ($hook_errors / $hook_total) * 100}")
    fi

    # Status
    if [[ $(echo "$error_rate >= $ALERT_THRESHOLD_HOOK_ERROR_RATE" | bc -l) -eq 1 ]]; then
        status_error "Hook Error Rate: ${error_rate}% (${hook_errors}/${hook_total}) - THRESHOLD EXCEEDED (>${ALERT_THRESHOLD_HOOK_ERROR_RATE}%)"
    elif [[ $hook_warnings -gt 0 ]]; then
        status_warning "Hook Warnings: ${hook_warnings} warnings detected in recent logs"
    else
        status_ok "Hook Execution: ${error_rate}% error rate (${hook_errors}/${hook_total})"
    fi

    # Export metrics
    if [[ "$EXPORT_METRICS" == "true" ]]; then
        echo "  \"hook_execution\": {\"total\": $hook_total, \"errors\": $hook_errors, \"warnings\": $hook_warnings, \"error_rate\": $error_rate}," >> /tmp/devstream_metrics.json
    fi
}

# 2. Rate Limiter Monitoring
monitor_rate_limiters() {
    local memory_throttled=0
    local memory_total=0
    local ollama_throttled=0
    local ollama_total=0

    # Parse rate limiter stats from logs (simplified - production would use structured logs)
    # For now, check if rate limiters are active by checking recent operations
    local recent_memory_ops=$(tail -200 "${LOG_DIR}"/mcp_client.jsonl 2>/dev/null | grep -c "store_memory\|search_memory" || echo 0)
    recent_memory_ops=$(echo "$recent_memory_ops" | tr -d ' ' | tr -d '\n')
    local recent_ollama_ops=$(tail -200 "${LOG_DIR}"/ollama_client.jsonl 2>/dev/null | grep -c "generate_embedding" || echo 0)
    recent_ollama_ops=$(echo "$recent_ollama_ops" | tr -d ' ' | tr -d '\n')

    # Status (conservative - assumes rate limiters are working if we see operations)
    if [[ $recent_memory_ops -gt 50 ]]; then
        status_warning "Memory Operations: ${recent_memory_ops} ops in recent logs (high volume - rate limiter active)"
    elif [[ $recent_memory_ops -gt 0 ]]; then
        status_ok "Memory Operations: ${recent_memory_ops} ops in recent logs"
    else
        status_info "Memory Operations: No recent operations detected"
    fi

    if [[ $recent_ollama_ops -gt 20 ]]; then
        status_warning "Ollama Operations: ${recent_ollama_ops} ops in recent logs (high volume - rate limiter active)"
    elif [[ $recent_ollama_ops -gt 0 ]]; then
        status_ok "Ollama Operations: ${recent_ollama_ops} ops in recent logs"
    else
        status_info "Ollama Operations: No recent operations detected"
    fi

    # Export metrics
    if [[ "$EXPORT_METRICS" == "true" ]]; then
        echo "  \"rate_limiters\": {\"memory_ops\": $recent_memory_ops, \"ollama_ops\": $recent_ollama_ops}," >> /tmp/devstream_metrics.json
    fi
}

# 3. Performance Monitoring
monitor_performance() {
    # Process memory usage (find DevStream-related processes)
    local total_memory_mb=0
    local process_count=0

    while IFS= read -r line; do
        # Extract RSS (in bytes on macOS) and convert to MB
        local rss=$(echo "$line" | awk '{print $6}')
        local memory_mb=$(awk "BEGIN {printf \"%.0f\", $rss / 1024}")
        total_memory_mb=$((total_memory_mb + memory_mb))
        process_count=$((process_count + 1))
    done < <(ps aux | grep -E "(node.*devstream|python.*devstream|ollama)" | grep -v grep)

    local total_memory_gb=$(awk "BEGIN {printf \"%.2f\", $total_memory_mb / 1024}")

    # Status
    if [[ $(echo "$total_memory_gb >= $ALERT_THRESHOLD_MEMORY_GB" | bc -l) -eq 1 ]]; then
        status_error "Memory Usage: ${total_memory_gb} GB (${process_count} processes) - THRESHOLD EXCEEDED (>${ALERT_THRESHOLD_MEMORY_GB} GB)"
    elif [[ $process_count -eq 0 ]]; then
        status_warning "Memory Usage: No DevStream processes detected"
    else
        status_ok "Memory Usage: ${total_memory_gb} GB (${process_count} processes)"
    fi

    # Export metrics
    if [[ "$EXPORT_METRICS" == "true" ]]; then
        echo "  \"performance\": {\"memory_gb\": $total_memory_gb, \"process_count\": $process_count}," >> /tmp/devstream_metrics.json
    fi
}

# 4. Crash Detection
monitor_crashes() {
    local kernel_panics=0
    local memory_pressure_warnings=0

    # Check system logs for kernel panics (last 1 hour)
    if command -v log &> /dev/null; then
        kernel_panics=$(log show --predicate 'eventMessage contains "panic"' --last 1h 2>/dev/null | grep -c "panic" || echo 0)
        memory_pressure_warnings=$(log show --predicate 'eventMessage contains "memory pressure"' --last 1h 2>/dev/null | grep -c "memory pressure" || echo 0)
    fi

    # Check DevStream logs for crash indicators
    local hook_crashes=$(tail -500 "${LOG_DIR}"/*.log 2>/dev/null | grep -c "Traceback\|MemoryError\|SIGKILL\|SIGTERM" || echo 0)

    # Status
    if [[ $kernel_panics -gt 0 ]]; then
        status_error "Crash Detection: ${kernel_panics} kernel panics detected in last hour"
    elif [[ $memory_pressure_warnings -gt 5 ]]; then
        status_warning "Crash Detection: ${memory_pressure_warnings} memory pressure warnings in last hour"
    elif [[ $hook_crashes -gt 0 ]]; then
        status_warning "Crash Detection: ${hook_crashes} hook crashes detected in recent logs"
    else
        status_ok "Crash Detection: No crashes detected"
    fi

    # Export metrics
    if [[ "$EXPORT_METRICS" == "true" ]]; then
        echo "  \"crash_detection\": {\"kernel_panics\": $kernel_panics, \"memory_pressure_warnings\": $memory_pressure_warnings, \"hook_crashes\": $hook_crashes}," >> /tmp/devstream_metrics.json
    fi
}

# 5. MCP Server Health
monitor_mcp_server() {
    local mcp_processes=$(ps aux | grep -E "node.*mcp-devstream-server" | grep -v grep | wc -l | tr -d ' ' | tr -d '\n')
    local mcp_errors
    mcp_errors=$(tail -100 "${LOG_DIR}"/mcp-server.log 2>/dev/null | grep -c "ERROR" || echo "0")
    mcp_errors=$(echo "$mcp_errors" | tr -d ' ' | tr -d '\n' | head -1)
    local mcp_last_heartbeat="N/A"

    # Check for recent heartbeat (if MCP server logs heartbeats)
    if [[ -f "${LOG_DIR}/mcp-server.log" ]]; then
        mcp_last_heartbeat=$(tail -50 "${LOG_DIR}/mcp-server.log" 2>/dev/null | grep "heartbeat\|uptime" | tail -1 | awk '{print $1, $2}' || echo "N/A")
    fi

    # Status
    if [[ $mcp_processes -eq 0 ]]; then
        status_error "MCP Server: No running processes detected"
    elif [[ $mcp_processes -gt 2 ]]; then
        status_warning "MCP Server: ${mcp_processes} processes running (possible orphans)"
    elif [[ $mcp_errors -gt 5 ]]; then
        status_warning "MCP Server: ${mcp_errors} errors in recent logs"
    else
        status_ok "MCP Server: ${mcp_processes} process(es) running, ${mcp_errors} recent errors"
    fi

    if [[ "$mcp_last_heartbeat" != "N/A" ]]; then
        status_info "MCP Server Last Heartbeat: ${mcp_last_heartbeat}"
    fi

    # Export metrics
    if [[ "$EXPORT_METRICS" == "true" ]]; then
        echo "  \"mcp_server\": {\"processes\": $mcp_processes, \"errors\": $mcp_errors, \"last_heartbeat\": \"$mcp_last_heartbeat\"}," >> /tmp/devstream_metrics.json
    fi
}

###############################################################################
# Main Monitoring Loop
###############################################################################

run_monitoring() {
    clear
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "DevStream Production Monitoring - $(timestamp)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Initialize metrics export
    if [[ "$EXPORT_METRICS" == "true" ]]; then
        echo "{" > /tmp/devstream_metrics.json
        echo "  \"timestamp\": \"$(timestamp)\"," >> /tmp/devstream_metrics.json
    fi

    # Run all monitors
    if [[ "$ALERTS_ONLY" == "false" ]]; then
        echo "📊 Hook Execution Health:"
        monitor_hooks
        echo ""

        echo "⚙️  Rate Limiter Status:"
        monitor_rate_limiters
        echo ""

        echo "💾 Performance Metrics:"
        monitor_performance
        echo ""

        echo "🚨 Crash Detection:"
        monitor_crashes
        echo ""

        echo "🔌 MCP Server Health:"
        monitor_mcp_server
        echo ""
    else
        # Alerts-only mode: only show warnings/errors
        monitor_hooks 2>&1 | grep -E "⚠️|❌" || true
        monitor_rate_limiters 2>&1 | grep -E "⚠️|❌" || true
        monitor_performance 2>&1 | grep -E "⚠️|❌" || true
        monitor_crashes 2>&1 | grep -E "⚠️|❌" || true
        monitor_mcp_server 2>&1 | grep -E "⚠️|❌" || true
    fi

    # Finalize metrics export
    if [[ "$EXPORT_METRICS" == "true" ]]; then
        echo "  \"monitoring_completed\": true" >> /tmp/devstream_metrics.json
        echo "}" >> /tmp/devstream_metrics.json
        status_ok "Metrics exported to /tmp/devstream_metrics.json"
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ "$WATCH_MODE" == "true" ]]; then
        echo "Refreshing in 5 seconds... (Ctrl+C to stop)"
        sleep 5
    fi
}

###############################################################################
# Entry Point
###############################################################################

# Verify log directory exists
if [[ ! -d "$LOG_DIR" ]]; then
    status_error "Log directory not found: $LOG_DIR"
    exit 1
fi

# Run monitoring
if [[ "$WATCH_MODE" == "true" ]]; then
    # Continuous monitoring
    while true; do
        run_monitoring
    done
else
    # Single run
    run_monitoring
fi
