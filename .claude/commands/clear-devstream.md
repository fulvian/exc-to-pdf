---
description: Clear conversation history while preserving DevStream session summary
---

## Clear DevStream Conversation

This command preserves your session summary in DevStream memory before clearing the conversation history.

## Workflow

Execute the following steps in sequence:

### 1. Generate Session Summary

Analyze the current conversation and extract:

- **Session goal**: Infer the primary objective from context
- **Completed tasks**: Review work_sessions table for finished tasks in last 24 hours
- **Modified files**: Identify files changed during this session from semantic_memory
- **Key decisions**: Capture architectural choices and trade-offs discussed
- **Learnings**: Document insights, debugging discoveries, and lessons learned
- **Next steps**: Note any pending work or follow-up items

Create a comprehensive summary in this format:

```
DevStream Session Summary - [Date]

Session Goal: [Primary objective]

Completed Tasks:
- [Task ID]: [Task description]
- [Task ID]: [Task description]

Modified Files:
- [file path]: [purpose/changes]
- [file path]: [purpose/changes]

Key Decisions:
- [Decision]: [Rationale]

Learnings:
- [Learning]: [Impact]

Next Steps:
- [Action item]
```

### 2. Store Summary in DevStream Memory

Use the MCP tool to preserve the summary for future retrieval:

```
mcp__devstream__devstream_store_memory:
  content: [Full session summary text]
  content_type: "context"
  keywords: ["session", "summary", "clear-devstream", "session-end", "[YYYY-MM-DD]"]
```

This ensures the summary is:
- Stored in semantic memory with vector embeddings
- Searchable by keywords and semantic similarity
- Available for context injection in future sessions

### 3. Write Session Marker File

Create a marker file for the SessionStart hook to display on next session:

- **Path**: `~/.claude/state/devstream_last_session.txt`
- **Content**: Full session summary text
- **Purpose**: SessionStart hook will detect this file and display the summary automatically

Create parent directories if needed:

```bash
mkdir -p ~/.claude/state
echo "[Full summary text]" > ~/.claude/state/devstream_last_session.txt
```

### 4. Display Summary Preview

Show the user a preview of the preserved summary:

```
Session Summary Preview (first 500 characters):

[First 500 characters of summary...]

Summary preserved successfully:
- Stored in DevStream memory (searchable)
- Written to marker file: ~/.claude/state/devstream_last_session.txt
- Will be displayed automatically in next session
```

Confirm that:
- Memory storage completed successfully
- Marker file written successfully
- Summary is ready for next session

### 5. Execute Conversation Clear

Inform the user:

```
Conversation history will now be cleared. Your session summary has been preserved and will be available in your next session.

To access the summary later:
- Use: mcp__devstream__devstream_search_memory with query "session summary"
- Or: Start a new session and it will be displayed automatically
```

Then clear the current conversation history.

## Notes

- This command is designed to work seamlessly with the SessionStart hook
- The summary will be automatically displayed when you start your next session
- All summaries are preserved in DevStream memory for long-term retrieval
- Use `/clear-devstream` instead of regular clear to maintain continuity across sessions
