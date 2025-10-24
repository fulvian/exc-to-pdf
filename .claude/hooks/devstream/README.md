# DevStream Claude Code Hooks - LOCAL PROJECT SETUP

Sistema di hook automatici per integrazione DevStream con Claude Code (Project-Specific).

## 🎯 Obiettivo
Automatizzare completamente task management e memory operations durante sviluppo con Claude Code nel progetto DevStream, seguendo best practice **LOCAL setup**.

## 📍 Location Strategy
**✅ LOCAL setup** (raccomandato da Context7 research):
- Location: `./devstream/.claude/hooks/devstream/`
- Configuration: `./devstream/.claude/settings.json`
- Scope: Project-specific DevStream automation
- Benefits: Project isolation, team sharing, version control

## 📁 Struttura Directory

```
devstream/.claude/hooks/devstream/
├── README.md              # Questa documentazione
├── config/                # Configuration e schema validation
│   ├── settings.py        # Schema configuration hooks
│   └── validation.py      # JSON schema validation
├── memory/                # Memory automation hooks
│   ├── user_prompt_submit.py    # Auto memory storage
│   ├── post_tool_use.py         # Results/learning capture
│   └── pre_tool_use.py          # Context injection
├── tasks/                 # Task management hooks
│   ├── session_start.py   # Task planning detection
│   ├── stop.py           # Task completion auto
│   └── progress_tracker.py # Code generation tracking
├── context/               # Context injection system
│   ├── project_context.py # Git repo, README detection
│   ├── memory_context.py  # Memory-based retrieval
│   └── standards_injection.py # Dev standards auto
└── utils/                 # Utility functions
    ├── mcp_client.py     # DevStream MCP integration
    ├── logger.py         # Structured logging
    └── common.py         # Shared utilities
```

## 🔧 Hook Types Implementati

### Memory Automation (DevStream-Specific)
- **UserPromptSubmit**: Automatic memory storage nel DevStream database
- **PostToolUse**: Capture risultati, learning, errors nel sistema semantico
- **PreToolUse**: Context injection da DevStream memoria

### Task Management (DevStream Integration)
- **SessionStart**: Detection task planning, auto-creation nel DevStream
- **Stop**: Task completion automatico con DevStream MCP
- **Progress Tracking**: Code generation monitoring per DevStream tasks

### Context System (DevStream Project Context)
- **Project Context**: Auto-detection DevStream standards, CLAUDE.md
- **Memory Context**: Retrieval da DevStream semantic memory
- **Standards Injection**: DevStream methodology e best practices

## 📊 DevStream MCP Integration

Tutti gli hook utilizzano DevStream MCP server locale per:
- Automatic memory storage (`mcp__devstream__devstream_store_memory`)
- Memory retrieval (`mcp__devstream__devstream_search_memory`)
- Task management (`mcp__devstream__devstream_create_task`, `mcp__devstream__devstream_update_task`)

## 🚀 Local Setup Benefits

1. **Project Isolation**: Hook specifici per DevStream project
2. **Team Sharing**: Hooks versioned con git, condivisi col team
3. **Customization**: Logic specifica per DevStream methodology
4. **No Interference**: Non impatta altri progetti Claude Code

## 📋 Configuration

Hook configuration in `./devstream/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "uv run .claude/hooks/devstream/memory/user_prompt_submit.py"
          }
        ]
      }
    ]
  }
}
```

---
*Creato: 2025-09-29 - DevStream Hook System Implementation (LOCAL Setup)*
*Context7 Research-Backed: Local project-specific hooks preferred*