# Task Creation Protocol

This workflow defines how to create a DevStream-compliant task before any implementation work begins. Follow it whenever a new body of work is requested.

## 1. Understand the request
- Capture the user’s exact ask and clarify scope, deliverables, and priorities.
- Identify constraints (deadline, architecture, review requirements, external dependencies).

## 2. Draft the task
- Title: concise, actionable, prefixed with priority if applicable (e.g., `[P1] Project foundation`).
- Description: summarize goals, include context and explicit success criteria.
- Type: choose the dominant activity (`planning`, `analysis`, `implementation`, `testing`, `documentation`, `review`).
- Priority: map perceived urgency to a value between 1 (highest) and 5 (lowest).
- Phase: align with CLAUDE.md guidance (`foundation`, `implementation`, `stabilization`, etc.).

## 3. Validate with the user
- Present the draft, ask for confirmation or adjustments.
- Record clarification notes; update the description accordingly.

## 4. Create the task
- Use the DevStream task creation tool (direct DB client or MCP tool).
- Confirm the task appears in the project database and the current session state.

## 5. Log follow-up actions
- Note any preparatory steps (design discussion, research, spike).
- Link related resources (protocol references, specs, docs).

Once the task is active, move to the Task Startup Protocol before executing work.***
