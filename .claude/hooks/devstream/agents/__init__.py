"""
Agent Auto-Delegation System.

This package provides automatic agent selection using hybrid pattern matching
and LLM routing for the DevStream custom agent system.

Modules:
    pattern_catalog: Agent routing patterns and metadata
"""

from .pattern_catalog import (
    PatternRule,
    PatternMatch,
    PATTERN_CATALOG,
    AGENT_METADATA,
    get_agent_by_name,
    get_all_agents,
    get_agents_by_role,
)

__all__ = [
    "PatternRule",
    "PatternMatch",
    "PATTERN_CATALOG",
    "AGENT_METADATA",
    "get_agent_by_name",
    "get_all_agents",
    "get_agents_by_role",
]
