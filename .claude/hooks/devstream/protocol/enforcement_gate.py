#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyinquirer>=1.0.3",
#     "aiofiles>=23.0.0",
#     "structlog>=23.0.0",
#     "python-dotenv>=1.0.0",
# ]
# ///

"""
Protocol Enforcement Gate - Blocking Validation for DevStream 7-Step Protocol

Phase 1 Component: Physical blocking of execution until user response.
Provides three options: Protocol, Override (with risks), Cancel.

Architecture Principles:
- Blocking interactive prompts using PyInquirer (Trust Score 10.0)
- Mandatory override logging to DevStream memory for audit trail
- Integration with existing hook system
- Graceful degradation if PyInquirer unavailable

Enforcement Flow:
1. Analyze task complexity and trigger conditions
2. Display protocol requirement warning
3. Present interactive options to user
4. Block execution until user makes choice
5. Log decision to memory with full audit trail

User Options:
✅ [RECOMMENDED] Follow DevStream protocol (research-driven, quality-assured)
⚠️  [OVERRIDE] Skip protocol (quick fix, NO quality assurance, NO Context7, NO testing)
❌ [CANCEL] Abort current operation

Override Risks:
- ❌ No Context7 research (potential outdated/incorrect patterns)
- ❌ No @code-reviewer validation (OWASP Top 10 security gaps)
- ❌ No testing requirements (95%+ coverage waived)
- ❌ No approval workflow (decisions undocumented)
"""

import asyncio
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from logger import get_devstream_logger

# PyInquirer import with graceful degradation
try:
    from PyInquirer import prompt, Separator
    from PyInquirer import Validator, ValidationError
    PYINQUIRER_AVAILABLE = True
except ImportError as e:
    PYINQUIRER_AVAILABLE = False
    PYINQUIRER_IMPORT_ERROR = str(e)

logger = get_devstream_logger(__name__)


class EnforcementDecision(Enum):
    """User decision from enforcement gate."""
    PROTOCOL = "protocol"
    OVERRIDE = "override"
    CANCEL = "cancel"


@dataclass
class EnforcementContext:
    """Context for enforcement gate decision."""
    task_description: str
    estimated_duration: int  # minutes
    complexity_score: float  # 0.0-1.0
    involves_code: bool
    involves_architecture: bool
    requires_context7: bool
    trigger_reasons: List[str]
    session_id: str
    timestamp: str


class EnforcementGate:
    """
    Blocking enforcement gate for DevStream protocol compliance.

    Provides physical blocking of execution until user makes an informed decision
    about following the 7-step protocol workflow.
    """

    def __init__(self):
        """Initialize enforcement gate."""
        self.decisions_logged = 0

        if not PYINQUIRER_AVAILABLE:
            logger.logger.info(
                "pyinquirer_unavailable",
                error=PYINQUIRER_IMPORT_ERROR,
                fallback="non-interactive mode"
            )

    def should_enforce_protocol(
        self,
        task_description: str,
        estimated_duration: int = 0,
        involves_code: bool = False,
        involves_architecture: bool = False,
        requires_context7: bool = False,
        file_count: int = 0
    ) -> Tuple[bool, List[str]]:
        """
        Analyze if protocol enforcement should be triggered.

        Args:
            task_description: Description of the task
            estimated_duration: Estimated duration in minutes
            involves_code: Whether task involves code implementation
            involves_architecture: Whether task involves architectural decisions
            requires_context7: Whether task requires Context7 research
            file_count: Number of files involved

        Returns:
            Tuple of (should_enforce, trigger_reasons)
        """
        trigger_reasons = []

        # Enforcement Trigger Criteria (from CLAUDE.md)
        if estimated_duration > 15:
            trigger_reasons.append(f"Estimated duration > 15min ({estimated_duration}min)")

        if involves_code:
            trigger_reasons.append("Involves code implementation (Write/Edit tools)")

        if involves_architecture:
            trigger_reasons.append("Involves architectural decisions")

        if file_count > 1:
            trigger_reasons.append(f"Involves multiple files ({file_count} files)")

        if requires_context7:
            trigger_reasons.append("Requires Context7 research")

        # Additional complexity triggers
        complexity_indicators = [
            "implement", "create", "build", "design", "architecture",
            "integration", "migration", "refactor", "optimization"
        ]
        if any(indicator in task_description.lower() for indicator in complexity_indicators):
            trigger_reasons.append("High complexity indicators detected")

        should_enforce = len(trigger_reasons) > 0

        logger.info(
            "enforcement_analysis",
            should_enforce=should_enforce,
            trigger_count=len(trigger_reasons),
            reasons=trigger_reasons
        )

        return should_enforce, trigger_reasons

    async def show_enforcement_gate(
        self,
        context: EnforcementContext,
        memory_client=None
    ) -> EnforcementDecision:
        """
        Display blocking enforcement gate and wait for user decision.

        Args:
            context: Enforcement context with task details
            memory_client: Optional MCP memory client for logging

        Returns:
            User's enforcement decision

        Raises:
            KeyboardInterrupt: If user interrupts (treated as CANCEL)
        """
        logger.info(
            "enforcement_gate_shown",
            session_id=context.session_id,
            task_description=context.task_description[:100]
        )

        try:
            if PYINQUIRER_AVAILABLE:
                decision = await self._interactive_prompt(context)
            else:
                decision = await self._non_interactive_fallback(context)

            # Log decision to memory
            await self._log_decision_to_memory(context, decision, memory_client)

            return decision

        except KeyboardInterrupt:
            logger.info("user_interrupted_enforcement_gate")
            return EnforcementDecision.CANCEL
        except Exception as e:
            logger.error(
                "enforcement_gate_error",
                error=str(e),
                error_type=type(e).__name__
            )
            return EnforcementDecision.CANCEL

    async def _interactive_prompt(self, context: EnforcementContext) -> EnforcementDecision:
        """
        Show interactive prompt using PyInquirer.

        Args:
            context: Enforcement context

        Returns:
            User decision
        """
        # Build warning message
        warning_message = self._build_warning_message(context)

        questions = [
            {
                'type': 'list',
                'name': 'decision',
                'message': warning_message,
                'choices': [
                    {
                        'name': '✅ [RECOMMENDED] Follow DevStream protocol (research-driven, quality-assured)',
                        'value': 'protocol'
                    },
                    Separator(),
                    {
                        'name': '⚠️  [OVERRIDE] Skip protocol (quick fix, NO quality assurance)',
                        'value': 'override'
                    },
                    Separator(),
                    {
                        'name': '❌ [CANCEL] Abort current operation',
                        'value': 'cancel'
                    }
                ],
                'default': 'protocol'
            }
        ]

        # Display additional context
        self._display_additional_context(context)

        try:
            answers = prompt(questions)
            decision_value = answers.get('decision', 'cancel')

            logger.info(
                "interactive_decision_made",
                decision=decision_value,
                session_id=context.session_id
            )

            return EnforcementDecision(decision_value)

        except Exception as e:
            logger.error(
                "interactive_prompt_error",
                error=str(e)
            )
            return EnforcementDecision.CANCEL

    async def _non_interactive_fallback(self, context: EnforcementContext) -> EnforcementDecision:
        """
        Non-interactive fallback when PyInquirer unavailable.

        Args:
            context: Enforcement context

        Returns:
            User decision (defaults to PROTOCOL)
        """
        print("\n" + "="*80)
        print("⚠️  DEVSTREAM PROTOCOL REQUIRED")
        print("="*80)
        print(self._build_warning_message(context))
        print("\nPyInquirer not available - defaulting to PROTOCOL mode")
        print("To override protocol, set environment variable: DEVSTREAM_PROTOCOL_OVERRIDE=true")
        print("="*80)

        # Check environment variable override
        import os
        if os.getenv("DEVSTREAM_PROTOCOL_OVERRIDE", "").lower() == "true":
            logger.logger.info(
                "protocol_override_via_env",
                session_id=context.session_id
            )
            return EnforcementDecision.OVERRIDE

        return EnforcementDecision.PROTOCOL

    def _build_warning_message(self, context: EnforcementContext) -> str:
        """Build comprehensive warning message for user."""
        message = "⚠️ DevStream Protocol Required\n\n"
        message += "This task requires following the DevStream 7-step workflow:\n"
        message += "DISCUSSION → ANALYSIS → RESEARCH → PLANNING → APPROVAL → IMPLEMENTATION → VERIFICATION\n\n"

        message += "Task Details:\n"
        message += f"• Description: {context.task_description}\n"
        message += f"• Estimated Duration: {context.estimated_duration} minutes\n"
        message += f"• Complexity Score: {context.complexity_score:.1f}/1.0\n"
        message += f"• Trigger Reasons: {', '.join(context.trigger_reasons)}\n\n"

        message += "OPTIONS:\n"
        message += "✅ [RECOMMENDED] Follow DevStream protocol (research-driven, quality-assured)\n"
        message += "⚠️  [OVERRIDE] Skip protocol (quick fix, NO quality assurance, NO Context7, NO testing)\n"
        message += "❌ [CANCEL] Abort current operation\n\n"

        message += "Risks of override:\n"
        message += "- ❌ No Context7 research (potential outdated/incorrect patterns)\n"
        message += "- ❌ No @code-reviewer validation (OWASP Top 10 security gaps)\n"
        message += "- ❌ No testing requirements (95%+ coverage waived)\n"
        message += "- ❌ No approval workflow (decisions undocumented)\n"

        return message

    def _display_additional_context(self, context: EnforcementContext):
        """Display additional context information."""
        print("\n📋 Task Analysis Summary:")
        print(f"   • Session ID: {context.session_id}")
        print(f"   • Complexity Score: {context.complexity_score:.1f}/1.0")
        print(f"   • Estimated Duration: {context.estimated_duration} minutes")

        if context.involves_code:
            print("   • ⚡ Code implementation required")
        if context.involves_architecture:
            print("   • 🏗️  Architectural decisions involved")
        if context.requires_context7:
            print("   • 🔍 Context7 research required")

        print(f"\n🎯 Trigger Reasons:")
        for i, reason in enumerate(context.trigger_reasons, 1):
            print(f"   {i}. {reason}")

        print("\n" + "-"*80)

    async def _log_decision_to_memory(
        self,
        context: EnforcementContext,
        decision: EnforcementDecision,
        memory_client=None
    ) -> None:
        """
        Log enforcement decision to DevStream memory.

        Args:
            context: Enforcement context
            decision: User decision
            memory_client: Optional MCP memory client
        """
        try:
            # Build decision log content
            content = (
                f"Protocol Enforcement Decision: {decision.value.upper()}\n"
                f"Session ID: {context.session_id}\n"
                f"Task Description: {context.task_description}\n"
                f"Estimated Duration: {context.estimated_duration} minutes\n"
                f"Complexity Score: {context.complexity_score:.1f}/1.0\n"
                f"Trigger Reasons: {', '.join(context.trigger_reasons)}\n"
                f"Timestamp: {context.timestamp}"
            )

            if decision == EnforcementDecision.OVERRIDE:
                content += "\n\n⚠️ PROTOCOL OVERRIDE - RISKS ACCEPTED:\n"
                content += "- No Context7 research\n"
                content += "- No @code-reviewer validation\n"
                content += "- No testing requirements\n"
                content += "- No approval workflow"

            # Keywords for memory search
            keywords = [
                "protocol-enforcement",
                decision.value,
                "decision",
                context.session_id,
                "override" if decision == EnforcementDecision.OVERRIDE else ""
            ]

            # Store in memory if client available
            if memory_client:
                await memory_client.store_memory(
                    content=content,
                    content_type="decision",
                    keywords=keywords
                )
                logger.debug("decision_logged_to_memory", decision=decision.value)
            else:
                # Fallback: log to file
                await self._log_decision_to_file(context, decision)

            self.decisions_logged += 1

        except Exception as e:
            logger.error(
                "decision_logging_failed",
                error=str(e),
                error_type=type(e).__name__
            )

    async def _log_decision_to_file(
        self,
        context: EnforcementContext,
        decision: EnforcementDecision
    ) -> None:
        """
        Fallback: Log decision to file.

        Args:
            context: Enforcement context
            decision: User decision
        """
        try:
            log_file = Path(".claude/logs/protocol_decisions.jsonl")
            log_file.parent.mkdir(parents=True, exist_ok=True)

            log_entry = {
                "timestamp": context.timestamp,
                "session_id": context.session_id,
                "decision": decision.value,
                "task_description": context.task_description,
                "complexity_score": context.complexity_score,
                "trigger_reasons": context.trigger_reasons
            }

            # Append to log file
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')

            logger.debug("decision_logged_to_file", log_file=str(log_file))

        except Exception as e:
            logger.error(
                "file_logging_failed",
                error=str(e)
            )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get enforcement gate statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "decisions_logged": self.decisions_logged,
            "pyinquirer_available": PYINQUIRER_AVAILABLE,
            "enforcement_active": True
        }


# Global instance
_global_gate: Optional[EnforcementGate] = None


def get_enforcement_gate() -> EnforcementGate:
    """Get global enforcement gate instance."""
    global _global_gate
    if _global_gate is None:
        _global_gate = EnforcementGate()
    return _global_gate


# Example usage and testing
if __name__ == "__main__":
    async def test_enforcement_gate():
        """Test enforcement gate functionality."""

        gate = EnforcementGate()

        # Test 1: Enforcement trigger analysis
        print("🧪 Test 1: Enforcement trigger analysis")
        should_enforce, reasons = gate.should_enforce_protocol(
            task_description="Implement new API endpoint with authentication",
            estimated_duration=25,
            involves_code=True,
            involves_architecture=True,
            requires_context7=True,
            file_count=3
        )
        assert should_enforce == True
        assert len(reasons) > 0
        print(f"✅ Should enforce: {should_enforce}")
        print(f"   Reasons: {reasons}")

        # Test 2: Simple task (no enforcement)
        print("\n🧪 Test 2: Simple task analysis")
        should_enforce, reasons = gate.should_enforce_protocol(
            task_description="Fix typo in README",
            estimated_duration=5,
            involves_code=False,
            involves_architecture=False,
            requires_context7=False,
            file_count=1
        )
        assert should_enforce == False
        assert len(reasons) == 0
        print(f"✅ Should enforce: {should_enforce}")

        # Test 3: Context creation
        print("\n🧪 Test 3: Enforcement context creation")
        import uuid
        from datetime import datetime, timezone

        context = EnforcementContext(
            task_description="Build comprehensive user authentication system",
            estimated_duration=45,
            complexity_score=0.8,
            involves_code=True,
            involves_architecture=True,
            requires_context7=True,
            trigger_reasons=[
                "Estimated duration > 15min (45min)",
                "Involves code implementation",
                "Involves architectural decisions",
                "Requires Context7 research"
            ],
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        print(f"✅ Context created: {context.session_id}")

        # Test 4: Statistics
        print("\n🧪 Test 4: Statistics")
        stats = gate.get_statistics()
        assert "decisions_logged" in stats
        assert "pyinquirer_available" in stats
        print(f"✅ Statistics: {stats}")

        # Test 5: Warning message building
        print("\n🧪 Test 5: Warning message building")
        warning = gate._build_warning_message(context)
        assert "DevStream Protocol Required" in warning
        assert context.task_description in warning
        print(f"✅ Warning message generated ({len(warning)} chars)")

        print("\n🎉 All enforcement gate tests PASSED")
        print("\n💡 Note: Interactive testing requires manual execution")
        print("   Use in real hook execution to see interactive prompts")

    # Run tests
    asyncio.run(test_enforcement_gate())