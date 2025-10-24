# Context Compaction Protocol

Apply this protocol when chats, task histories, or context windows become unwieldy. The objective is to preserve critical information while keeping the working context lean.

## 1. Assess current context
- Identify active tasks, recent decisions, and in-flight discussions.
- Detect obsolete or redundant snippets that inflate the prompt.

## 2. Summarize essentials
- Produce short summaries per topic (requirements, design choices, blockers).
- Capture key links: tasks, commits, documents, dashboards.

## 3. Archive details
- Move verbose transcripts, logs, or exploratory notes to project storage (e.g., docs/, knowledge base).
- Reference archive locations in the compact summary for future retrieval.

## 4. Refresh the working set
- Replace the bloated context with the concise summary and references.
- Reintroduce only the information necessary for the next action.

## 5. Communicate the update
- Share the compact summary with collaborators.
- Note when the next compaction should occur (e.g., after major milestones).

Repeat the protocol whenever conversation history or memory exceeds the model’s comfort threshold.***
