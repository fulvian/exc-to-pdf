---
description: Trigger immediate checkpoint for all active DevStream tasks (manual progress save)
allowed-tools: mcp__devstream__devstream_trigger_checkpoint
---

Call the `mcp__devstream__devstream_trigger_checkpoint` tool with the following parameters:
- `reason`: "manual"

Then report the result to the user, showing:
- Number of active tasks saved
- Or "No active tasks found" if applicable
- Or error message if checkpoint failed

## Context

This command triggers a manual checkpoint save for all active DevStream tasks. Checkpoints store task state in semantic memory for progress tracking, recovery, and historical analysis.

**Checkpoint Triggers:**
- **Automatic**: Every 5 minutes (auto_save)
- **Critical Tools**: After Write/Edit/Bash/TodoWrite (tool_trigger)
- **Manual**: This command (manual)
- **Shutdown**: Session end (shutdown)

## What Happens

1. **Query Active Tasks**: Retrieves all tasks with status = "active"
2. **Create Checkpoints**: Stores checkpoint entry in semantic_memory for each task
3. **Return Feedback**: Shows number of tasks saved or "No active tasks" message

## Checkpoint Data

Each checkpoint stores:
- Task ID, title, phase, status, priority
- Elapsed time (if task started)
- Timestamp and reason
- Structured JSON in context_snapshot

## Use Cases

- **Before Context Switch**: Save progress before switching to a different task
- **After Major Progress**: Checkpoint after completing significant work
- **Pre-Commit**: Save state before committing code
- **Session Checkpoint**: Manual save during long work sessions

## Non-Blocking Execution

Checkpoints execute asynchronously without blocking Claude Code. Individual task checkpoint failures are isolated and logged without affecting other tasks.

## Example Output

Success:
```
✅ Checkpoint triggered: 3 active tasks saved (reason: manual)
```

No active tasks:
```
ℹ️ No active tasks found - checkpoint skipped
```

Failure:
```
⚠️ Checkpoint failed: Database connection error
```

## Related Commands

- `/devstream-list-tasks` - View all tasks and their status
- `/devstream-update-task` - Change task status (triggers status_change checkpoint)

## Technical Details

**MCP Tool**: `devstream_trigger_checkpoint`
**Service**: AutoSaveService.triggerImmediateCheckpoint()
**Database**: semantic_memory table (content_type: "context")
**Reason**: "manual"
**Timeout**: 30 seconds (configurable)

## Notes

- Checkpoints do NOT change task status (status updates trigger separate status_change checkpoints)
- Manual checkpoints are in addition to automatic saves, not a replacement
- Checkpoint history can be queried via `devstream_search_memory` with keyword "checkpoint"
