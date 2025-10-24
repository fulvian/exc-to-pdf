# DevStream Monitoring System

**Version**: 1.0.0 | **Status**: ✅ Production Ready | **Auto-Start**: Integrated in `start-devstream.sh`

---

## Overview

Comprehensive monitoring system that prevents MCP zombie processes, detects server health issues, and alerts on database staleness.

**Key Features**:
- ✅ Automatic zombie cleanup (before every tool execution + periodic background)
- ✅ MCP server health monitoring (30s ping interval)
- ✅ Database staleness detection (10-minute threshold)
- ✅ Fully integrated with `start-devstream.sh` (no manual setup)

---

## Quick Start

### Automatic (Recommended)

All monitoring starts automatically when you launch DevStream:

```bash
# Start DevStream with monitoring
./start-devstream.sh start

# Check status
./start-devstream.sh status

# Stop everything
./start-devstream.sh stop
```

**That's it!** Monitoring runs in background automatically.

---

## What Gets Monitored

### 1. MCP Process Monitor
- **Interval**: 60 seconds
- **Purpose**: Detect and kill zombie MCP processes
- **Threshold**: Max 2 instances allowed
- **Action**: Automatically kills oldest processes
- **Log**: `~/.claude/logs/devstream/process_monitor_daemon.log`
- **PID**: `.devstream/process_monitor.pid`

### 2. MCP Health Check
- **Interval**: 30 seconds
- **Purpose**: Ping MCP server for responsiveness
- **Timeout**: 10 seconds
- **States**: Healthy (<5s), Degraded (1-2 failures), Critical (3+ failures)
- **Log**: `~/.claude/logs/devstream/health_check_daemon.log`
- **PID**: `.devstream/health_check.pid`

### 3. Database Update Monitor
- **Interval**: 60 seconds
- **Purpose**: Track database write staleness
- **Threshold**: 600 seconds (10 minutes)
- **Action**: Alert if no writes for >10min
- **Log**: `~/.claude/logs/devstream/db_monitor_daemon.log`
- **PID**: `.devstream/db_monitor.pid`

---

## Manual Control (Advanced)

### Single-Check Mode

Run monitors once and exit (useful for debugging):

```bash
# Check for zombie processes
.devstream/bin/python .claude/hooks/devstream/monitoring/mcp_process_monitor.py --once

# Ping MCP server health
.devstream/bin/python .claude/hooks/devstream/monitoring/mcp_health_check.py --once --timeout 10

# Check database staleness
.devstream/bin/python .claude/hooks/devstream/monitoring/database_update_monitor.py --once --threshold 600
```

### Daemon Mode (Manual)

Run monitors continuously in foreground (for development):

```bash
# Process monitor (60s interval)
.devstream/bin/python .claude/hooks/devstream/monitoring/mcp_process_monitor.py

# Health check (custom 15s interval)
.devstream/bin/python .claude/hooks/devstream/monitoring/mcp_health_check.py --interval 15 --timeout 5

# Database monitor (custom 30s interval, 5min threshold)
.devstream/bin/python .claude/hooks/devstream/monitoring/database_update_monitor.py --interval 30 --threshold 300
```

---

## Integration Points

### 1. PreToolUse Hook (Automatic)
**File**: `.claude/hooks/devstream/memory/pre_tool_use.py` (line 727-743)

Runs **before every Write/Edit operation**:
- Checks MCP process count
- Kills zombies if >2 instances detected
- Non-blocking (failures don't stop tool execution)

### 2. Start Script (Automatic)
**File**: `start-devstream.sh` (`start_mcp_server()` function)

Runs **before MCP server launch**:
- Pre-launch zombie cleanup
- Ensures clean start with max 2 instances

### 3. Background Daemons (Automatic)
**File**: `start-devstream.sh` (`start_monitors()` function)

Launched **automatically with `./start-devstream.sh start`**:
- 3 monitoring processes in background
- PID files for easy management
- Stopped automatically with `./start-devstream.sh stop`

---

## Troubleshooting

### Monitors Not Running

**Check status**:
```bash
./start-devstream.sh status
```

**Expected output**:
```
📊 Monitoring Daemon Status
==============================

✅ Process Monitor: Running (PID: 12345)
✅ Health Check: Running (PID: 12346)
✅ Database Monitor: Running (PID: 12347)

✅ All 3 monitoring daemons operational
```

**If not running, restart**:
```bash
./start-devstream.sh restart
```

---

### Zombie Processes Accumulating

**Symptoms**: `./start-devstream.sh status` shows >2 MCP processes

**Quick fix**:
```bash
# Manual cleanup
.devstream/bin/python .claude/hooks/devstream/monitoring/mcp_cleanup_hook.py

# Or restart everything
./start-devstream.sh restart
```

**Root cause**: Monitors may have stopped. Check logs:
```bash
tail -20 ~/.claude/logs/devstream/process_monitor_daemon.log
```

---

### Database Not Updating

**Symptoms**: Database monitor shows staleness >600s

**Diagnostic**:
```bash
# Check last update time
.devstream/bin/python .claude/hooks/devstream/monitoring/database_update_monitor.py --once

# Check MCP health
.devstream/bin/python .claude/hooks/devstream/monitoring/mcp_health_check.py --once
```

**Common causes**:
1. MCP server timeout (health check will show degraded/critical)
2. Zombie process lock contention (process monitor will show >2 instances)
3. Hook execution failure (check `~/.claude/logs/devstream/post_tool_use.jsonl`)

**Resolution**: Restart DevStream to clear all locks and zombies:
```bash
./start-devstream.sh restart
```

---

### High CPU Usage

**Symptoms**: Python processes consuming >10% CPU

**Check intervals**:
```bash
# View monitor PIDs and CPU usage
ps aux | grep -E "(process_monitor|health_check|db_monitor)" | grep -v grep
```

**Solution**: Increase intervals (edit `start-devstream.sh`):
```bash
# In start_monitors() function, change:
--interval 30   # Health check: 30s → 60s
--interval 60   # Other monitors: 60s → 120s
```

---

## Log Files

All monitors write structured JSON logs:

```
~/.claude/logs/devstream/
├── process_monitor_daemon.log      # Zombie detection + cleanup
├── health_check_daemon.log         # MCP ping latency + status
└── db_monitor_daemon.log           # Database staleness alerts
```

**View logs**:
```bash
# Real-time monitoring
tail -f ~/.claude/logs/devstream/process_monitor_daemon.log

# Last 20 lines
tail -20 ~/.claude/logs/devstream/health_check_daemon.log

# Search for errors
grep -E "error|critical" ~/.claude/logs/devstream/*.log
```

---

## Performance Impact

**Background Monitors** (3 daemons):
- CPU: <1% combined average
- Memory: ~150MB total (50MB each)
- Disk I/O: Minimal (append-only logs)

**PreToolUse Hook**:
- Overhead: <100ms per tool execution
- Frequency: Every Write/Edit operation
- Non-blocking: Always allows tool execution

**Total Impact**: Negligible (<1% system resources)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ LAYER 1: PreToolUse Hook (Proactive Cleanup)           │
│ • Runs before EVERY Write/Edit                          │
│ • Kills zombies immediately (<100ms)                    │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ LAYER 2: Process Monitor (Background - 60s)            │
│ • Detects zombie accumulation                           │
│ • Automatic cleanup when >2 instances                   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ LAYER 3: Health Check (Background - 30s)               │
│ • Pings MCP server                                       │
│ • Tracks healthy/degraded/critical states               │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ LAYER 4: Database Monitor (Background - 60s)           │
│ • Checks last write timestamp                           │
│ • Alerts when >10min stale                              │
└─────────────────────────────────────────────────────────┘
```

---

## Files

### Monitor Scripts
- `mcp_process_monitor.py` - Zombie detection and cleanup
- `mcp_cleanup_hook.py` - Lightweight cleanup for PreToolUse
- `mcp_health_check.py` - MCP server health monitoring
- `database_update_monitor.py` - Database staleness detection

### Integration
- `../memory/pre_tool_use.py` - PreToolUse hook with cleanup (line 727-743)
- `../../../../start-devstream.sh` - Launch script with monitor integration

### Documentation
- `mcp-zombie-prevention-system.md` - Complete system documentation
- `README.md` - This file

---

**Last Updated**: 2025-10-12
**Status**: ✅ Production Ready - Fully Automatic
**Maintainer**: DevStream Core Team
