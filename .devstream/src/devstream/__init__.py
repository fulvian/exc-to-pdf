"""
DevStream: Sistema Integrato Task Management & Memoria Cross-Session per Claude Code

Un framework rivoluzionario che trasforma l'interazione utente-AI imponendo un workflow
strutturato basato su task granulari con memoria semantica persistente.

Features principali:
- Task-Forced Workflow: Ogni interazione passa attraverso task strutturati
- Memoria Semantica: Storage persistente con embedding vettoriali
- Micro-Task Granulari: Task limitati a 10min e 256K token
- Team Virtuale Agenti: Architect, Coder, Reviewer, Documenter
- Context Injection: Memoria rilevante iniettata automaticamente
"""

__version__ = "0.1.0"
__author__ = "Claude AI Assistant"
__email__ = "noreply@anthropic.com"
__license__ = "MIT"

# Public API exports (temporarily commented for memory system testing)
# from devstream.core.config import DevStreamConfig
# from devstream.core.exceptions import DevStreamError, DatabaseError, MemoryError

__all__ = [
    "__version__",
    # "DevStreamConfig",
    # "DevStreamError",
    # "DatabaseError",
    # "MemoryError",
]