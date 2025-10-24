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
Interactive Step Validator - Enhanced Step Validation for DevStream Protocol

FASE 3 Component: Interactive validation system with step completion summaries
and user confirmation requirements for step transitions.

Architecture Principles:
- Interactive step validation using PyInquirer with graceful degradation
- Step completion summaries with detailed requirements
- User confirmation blocking for step transitions
- Integration with existing protocol state management
- Context7-based validation patterns

Step Validation Features:
1. Completion requirement checking for each protocol step
2. Interactive summaries showing what's been accomplished
3. Blocking confirmation dialogs for step progression
4. Detailed feedback on missing requirements
5. Audit trail logging for all step transitions

Integration Points:
- UserPromptSubmit hook: Interactive validation workflow
- ProtocolStateManager: State progression tracking
- DevStream memory: Audit trail logging
- Context7: Research validation patterns
"""

import asyncio
import json
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from logger import get_devstream_logger
from protocol_state_manager import ProtocolStep

# PyInquirer import with graceful degradation
try:
    from PyInquirer import prompt, Separator, Validator, ValidationError
    PYINQUIRER_AVAILABLE = True
except ImportError as e:
    PYINQUIRER_AVAILABLE = False
    PYINQUIRER_IMPORT_ERROR = str(e)

logger = get_devstream_logger(__name__)


class ValidationResult(Enum):
    """Step validation result types."""
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    PARTIAL = "partial"
    BLOCKED = "blocked"


@dataclass
class StepValidationResult:
    """Result of step validation with detailed feedback."""
    step: ProtocolStep
    result: ValidationResult
    completion_percentage: float  # 0.0-1.0
    requirements_met: List[str]
    requirements_missing: List[str]
    evidence: Dict[str, Any]
    next_step_ready: bool
    user_confirmation_required: bool


@dataclass
class StepTransition:
    """Step transition data for audit trail."""
    from_step: ProtocolStep
    to_step: ProtocolStep
    session_id: str
    user_confirmed: bool
    timestamp: str
    evidence: Dict[str, Any]
    notes: Optional[str]


class InteractiveStepValidator:
    """
    Interactive step validator with completion summaries and confirmation dialogs.

    Provides comprehensive validation for each protocol step with user interaction
    requirements for progression and detailed feedback on completion status.
    """

    def __init__(self, memory_client=None):
        """
        Initialize interactive step validator.

        Args:
            memory_client: Optional MCP memory client for audit logging
        """
        self.memory_client = memory_client
        self.transitions_completed = 0
        self.validations_performed = 0

        if not PYINQUIRER_AVAILABLE:
            logger.logger.info(
                "pyinquirer_unavailable",
                error=PYINQUIRER_IMPORT_ERROR,
                fallback="non-interactive validation"
            )

    async def validate_step_completion(
        self,
        step: ProtocolStep,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> StepValidationResult:
        """
        Validate completion of a specific protocol step.

        Args:
            step: Protocol step to validate
            user_input: User's input/prompt for validation
            context: Additional context for validation

        Returns:
            Detailed step validation result
        """
        self.validations_performed += 1

        try:
            logger.logger.info(
                "step_validation_started",
                step=step.value,
                step_name=str(step),
                session_id=context.get("session_id") if context else None
            )

            # Perform step-specific validation
            validation_data = await self._perform_step_validation(step, user_input, context)

            # Calculate completion percentage and requirements
            completion_percentage = self._calculate_completion_percentage(validation_data)
            requirements_met = validation_data.get("requirements_met", [])
            requirements_missing = validation_data.get("requirements_missing", [])
            evidence = validation_data.get("evidence", {})

            # Determine validation result
            result = self._determine_validation_result(completion_percentage, step)

            # Check if next step is ready
            next_step_ready = result == ValidationResult.COMPLETED

            # Determine if user confirmation is required
            user_confirmation_required = next_step_ready

            step_result = StepValidationResult(
                step=step,
                result=result,
                completion_percentage=completion_percentage,
                requirements_met=requirements_met,
                requirements_missing=requirements_missing,
                evidence=evidence,
                next_step_ready=next_step_ready,
                user_confirmation_required=user_confirmation_required
            )

            logger.logger.info(
                "step_validation_completed",
                step=step.value,
                result=result.value,
                completion_percentage=completion_percentage,
                requirements_met=len(requirements_met),
                requirements_missing=len(requirements_missing)
            )

            return step_result

        except Exception as e:
            logger.logger.error(
                "step_validation_error",
                step=step.value,
                error=str(e),
                error_type=type(e).__name__
            )

            # Return error result
            return StepValidationResult(
                step=step,
                result=ValidationResult.BLOCKED,
                completion_percentage=0.0,
                requirements_met=[],
                requirements_missing=["Validation system error"],
                evidence={"error": str(e)},
                next_step_ready=False,
                user_confirmation_required=False
            )

    async def handle_step_transition(
        self,
        validation_result: StepValidationResult,
        session_id: str,
        from_step: Optional[ProtocolStep] = None
    ) -> Tuple[bool, Optional[StepTransition]]:
        """
        Handle step transition with user confirmation.

        Args:
            validation_result: Result from step validation
            session_id: Current session ID
            from_step: Previous step (for transition tracking)

        Returns:
            Tuple of (transition_approved, transition_data)
        """
        try:
            if not validation_result.next_step_ready:
                logger.logger.info(
                    "step_transition_blocked",
                    step=validation_result.step.value,
                    reason="Step not completed"
                )
                return False, None

            # Show step completion summary
            await self._show_completion_summary(validation_result)

            # Request user confirmation
            user_confirmed = await self._request_step_confirmation(validation_result)

            if not user_confirmed:
                logger.logger.info(
                    "step_transition_cancelled_by_user",
                    step=validation_result.step.value,
                    session_id=session_id
                )
                return False, None

            # Create transition record
            transition = StepTransition(
                from_step=from_step or validation_result.step,
                to_step=self._get_next_step(validation_result.step),
                session_id=session_id,
                user_confirmed=True,
                timestamp=datetime.now(timezone.utc).isoformat(),
                evidence=validation_result.evidence,
                notes=f"Step {validation_result.step.value} completed with {validation_result.completion_percentage:.1%} completion"
            )

            # Log transition to memory
            await self._log_step_transition(transition)

            self.transitions_completed += 1

            logger.logger.info(
                "step_transition_approved",
                from_step=transition.from_step.value,
                to_step=transition.to_step.value,
                session_id=session_id
            )

            return True, transition

        except Exception as e:
            logger.logger.error(
                "step_transition_error",
                step=validation_result.step.value,
                error=str(e),
                error_type=type(e).__name__
            )
            return False, None

    async def _perform_step_validation(
        self,
        step: ProtocolStep,
        user_input: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform step-specific validation logic."""

        validation_methods = {
            ProtocolStep.DISCUSSION: self._validate_discussion_step,
            ProtocolStep.ANALYSIS: self._validate_analysis_step,
            ProtocolStep.RESEARCH: self._validate_research_step,
            ProtocolStep.PLANNING: self._validate_planning_step,
            ProtocolStep.APPROVAL: self._validate_approval_step,
            ProtocolStep.IMPLEMENTATION: self._validate_implementation_step,
            ProtocolStep.VERIFICATION: self._validate_verification_step,
        }

        validator = validation_methods.get(step)
        if validator:
            return await validator(user_input, context)
        else:
            return {
                "requirements_met": [],
                "requirements_missing": ["Unknown step validation"],
                "evidence": {"error": f"No validator for step {step}"}
            }

    async def _validate_discussion_step(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate DISCUSSION step with enhanced checking."""
        requirements_met = []
        requirements_missing = []
        evidence = {}

        # Check for discussion indicators
        discussion_indicators = [
            "discuss", "consider", "trade-off", "alternative", "approach",
            "strategy", "pros and cons", "let's talk", "what about"
        ]

        has_discussion = any(indicator in user_input.lower() for indicator in discussion_indicators)
        if has_discussion:
            requirements_met.append("Problem discussion present")
        else:
            requirements_missing.append("Discussion of problem/objective")

        # Check for sufficient length (meaningful discussion)
        if len(user_input) > 150:
            requirements_met.append("Sufficient discussion depth")
            evidence["word_count"] = len(user_input.split())
        else:
            requirements_missing.append("More detailed discussion needed")

        # Check for consideration of multiple perspectives
        perspective_indicators = ["alternatives", "options", "different", "multiple", "various"]
        has_perspectives = any(indicator in user_input.lower() for indicator in perspective_indicators)
        if has_perspectives:
            requirements_met.append("Multiple perspectives considered")

        # Check for identification of constraints or requirements
        constraint_indicators = ["requirement", "constraint", "limitation", "boundary", "scope"]
        has_constraints = any(indicator in user_input.lower() for indicator in constraint_indicators)
        if has_constraints:
            requirements_met.append("Constraints/requirements identified")

        return {
            "requirements_met": requirements_met,
            "requirements_missing": requirements_missing,
            "evidence": evidence
        }

    async def _validate_analysis_step(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate ANALYSIS step with technical analysis checking."""
        requirements_met = []
        requirements_missing = []
        evidence = {}

        # Check for analysis indicators
        analysis_indicators = [
            "analyze", "examine", "break down", "component", "structure",
            "impact", "dependency", "complexity", "assessment"
        ]

        has_analysis = any(indicator in user_input.lower() for indicator in analysis_indicators)
        if has_analysis:
            requirements_met.append("Technical analysis performed")
        else:
            requirements_missing.append("Technical analysis of requirements")

        # Check for file/component identification
        file_indicators = ["file", "component", "module", "service", "class", "function"]
        has_files = any(indicator in user_input.lower() for indicator in file_indicators)
        if has_files:
            requirements_met.append("Components identified")
            evidence["components_mentioned"] = True

        # Check for complexity estimation
        complexity_indicators = ["complex", "simple", "effort", "duration", "timeline", "estimate"]
        has_complexity = any(indicator in user_input.lower() for indicator in complexity_indicators)
        if has_complexity:
            requirements_met.append("Complexity assessed")

        # Check for risk identification
        risk_indicators = ["risk", "challenge", "difficulty", "obstacle", "problem"]
        has_risks = any(indicator in user_input.lower() for indicator in risk_indicators)
        if has_risks:
            requirements_met.append("Risks/Challenges identified")

        return {
            "requirements_met": requirements_met,
            "requirements_missing": requirements_missing,
            "evidence": evidence
        }

    async def _validate_research_step(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate RESEARCH step with Context7 integration."""
        requirements_met = []
        requirements_missing = []
        evidence = {}

        # Check for research indicators
        research_indicators = [
            "research", "study", "investigate", "explore", "documentation",
            "best practice", "pattern", "library", "framework"
        ]

        has_research = any(indicator in user_input.lower() for indicator in research_indicators)
        if has_research:
            requirements_met.append("Research conducted")
        else:
            requirements_missing.append("Best practices research")

        # Check for Context7 usage (if available in context)
        if context and context.get("context7_triggered"):
            requirements_met.append("Context7 research performed")
            evidence["context7_used"] = True

        # Check for documentation of findings
        doc_indicators = ["documentation", "findings", "results", "conclusion", "summary"]
        has_documentation = any(indicator in user_input.lower() for indicator in doc_indicators)
        if has_documentation:
            requirements_met.append("Research findings documented")

        # Check for validation of approach
        validation_indicators = ["validate", "confirm", "verify", "test approach", "proof"]
        has_validation = any(indicator in user_input.lower() for indicator in validation_indicators)
        if has_validation:
            requirements_met.append("Approach validated")

        return {
            "requirements_met": requirements_met,
            "requirements_missing": requirements_missing,
            "evidence": evidence
        }

    async def _validate_planning_step(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate PLANNING step with TodoWrite integration."""
        requirements_met = []
        requirements_missing = []
        evidence = {}

        # Check for planning indicators
        planning_indicators = [
            "plan", "todo", "steps", "implementation", "breakdown",
            "milestone", "phase", "iteration", "sprint"
        ]

        has_planning = any(indicator in user_input.lower() for indicator in planning_indicators)
        if has_planning:
            requirements_met.append("Implementation plan created")
        else:
            requirements_missing.append("Detailed implementation plan")

        # Check for micro-task breakdown
        micro_task_indicators = ["micro-task", "small task", "breakdown", "subtask", "step"]
        has_micro_tasks = any(indicator in user_input.lower() for indicator in micro_task_indicators)
        if has_micro_tasks:
            requirements_met.append("Micro-task breakdown defined")

        # Check for acceptance criteria
        acceptance_indicators = ["acceptance", "criteria", "requirement", "definition of done", "complete"]
        has_acceptance = any(indicator in user_input.lower() for indicator in acceptance_indicators)
        if has_acceptance:
            requirements_met.append("Acceptance criteria defined")

        # Check for timeline estimation
        timeline_indicators = ["timeline", "duration", "estimate", "deadline", "milestone"]
        has_timeline = any(indicator in user_input.lower() for indicator in timeline_indicators)
        if has_timeline:
            requirements_met.append("Timeline estimated")

        return {
            "requirements_met": requirements_met,
            "requirements_missing": requirements_missing,
            "evidence": evidence
        }

    async def _validate_approval_step(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate APPROVAL step with explicit confirmation checking."""
        requirements_met = []
        requirements_missing = []
        evidence = {}

        # Check for explicit approval indicators
        approval_indicators = [
            "approve", "approved", "confirm", "proceed", "ok", "agreed",
            "accept", "go ahead", "move forward", "ready to implement"
        ]

        has_approval = any(indicator in user_input.lower() for indicator in approval_indicators)
        if has_approval:
            requirements_met.append("Explicit approval obtained")
        else:
            requirements_missing.append("Explicit approval of implementation plan")

        # Check for risk acknowledgment
        risk_ack_indicators = ["acknowledge risk", "understand", "aware", "accept risk"]
        has_risk_ack = any(indicator in user_input.lower() for indicator in risk_ack_indicators)
        if has_risk_ack:
            requirements_met.append("Risks acknowledged")

        # Check for resource confirmation
        resource_indicators = ["resource", "time", "effort", "available", "sufficient"]
        has_resources = any(indicator in user_input.lower() for indicator in resource_indicators)
        if has_resources:
            requirements_met.append("Resources confirmed")

        return {
            "requirements_met": requirements_met,
            "requirements_missing": requirements_missing,
            "evidence": evidence
        }

    async def _validate_implementation_step(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate IMPLEMENTATION step with code completion checking."""
        requirements_met = []
        requirements_missing = []
        evidence = {}

        # Check for implementation indicators
        implementation_indicators = [
            "implement", "code", "write", "create", "build", "develop",
            "complete", "finish", "done", "implemented", "functioning"
        ]

        has_implementation = any(indicator in user_input.lower() for indicator in implementation_indicators)
        if has_implementation:
            requirements_met.append("Implementation completed")
        else:
            requirements_missing.append("Code implementation completed")

        # Check for testing mentions
        test_indicators = ["test", "testing", "tested", "validate", "verify"]
        has_testing = any(indicator in user_input.lower() for indicator in test_indicators)
        if has_testing:
            requirements_met.append("Testing performed")

        # Check for documentation
        doc_indicators = ["document", "documented", "readme", "comment", "explain"]
        has_documentation = any(indicator in user_input.lower() for indicator in doc_indicators)
        if has_documentation:
            requirements_met.append("Documentation created")

        return {
            "requirements_met": requirements_met,
            "requirements_missing": requirements_missing,
            "evidence": evidence
        }

    async def _validate_verification_step(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate VERIFICATION step with quality assurance checking."""
        requirements_met = []
        requirements_missing = []
        evidence = {}

        # Check for verification indicators
        verification_indicators = [
            "verify", "validate", "check", "confirm", "quality assurance",
            "passes", "successful", "working", "functional", "meets requirements"
        ]

        has_verification = any(indicator in user_input.lower() for indicator in verification_indicators)
        if has_verification:
            requirements_met.append("Verification completed")
        else:
            requirements_missing.append("Quality assurance verification")

        # Check for test results
        test_result_indicators = ["test pass", "all tests", "coverage", "successful test"]
        has_test_results = any(indicator in user_input.lower() for indicator in test_result_indicators)
        if has_test_results:
            requirements_met.append("Test results confirmed")

        # Check for requirement validation
        requirement_indicators = ["requirement met", "specification", "criteria met", "fulfill"]
        has_requirements = any(indicator in user_input.lower() for indicator in requirement_indicators)
        if has_requirements:
            requirements_met.append("Requirements validated")

        return {
            "requirements_met": requirements_met,
            "requirements_missing": requirements_missing,
            "evidence": evidence
        }

    def _calculate_completion_percentage(self, validation_data: Dict[str, Any]) -> float:
        """Calculate completion percentage from validation data."""
        requirements_met = validation_data.get("requirements_met", [])
        requirements_missing = validation_data.get("requirements_missing", [])

        total_requirements = len(requirements_met) + len(requirements_missing)
        if total_requirements == 0:
            return 0.0

        return len(requirements_met) / total_requirements

    def _determine_validation_result(
        self,
        completion_percentage: float,
        step: ProtocolStep
    ) -> ValidationResult:
        """Determine validation result from completion percentage."""
        if completion_percentage >= 0.9:
            return ValidationResult.COMPLETED
        elif completion_percentage >= 0.6:
            return ValidationResult.PARTIAL
        elif completion_percentage >= 0.3:
            return ValidationResult.INCOMPLETE
        else:
            return ValidationResult.BLOCKED

    def _get_next_step(self, current_step: ProtocolStep) -> ProtocolStep:
        """Get the next protocol step."""
        if current_step == ProtocolStep.IDLE:
            return ProtocolStep.DISCUSSION
        elif current_step == ProtocolStep.DISCUSSION:
            return ProtocolStep.ANALYSIS
        elif current_step == ProtocolStep.ANALYSIS:
            return ProtocolStep.RESEARCH
        elif current_step == ProtocolStep.RESEARCH:
            return ProtocolStep.PLANNING
        elif current_step == ProtocolStep.PLANNING:
            return ProtocolStep.APPROVAL
        elif current_step == ProtocolStep.APPROVAL:
            return ProtocolStep.IMPLEMENTATION
        elif current_step == ProtocolStep.IMPLEMENTATION:
            return ProtocolStep.VERIFICATION
        else:
            return ProtocolStep.IDLE  # End of protocol

    async def _show_completion_summary(self, validation_result: StepValidationResult) -> None:
        """Show step completion summary to user."""
        print(f"\n📋 STEP {validation_result.step.value} COMPLETION SUMMARY")
        print("=" * 70)
        print(f"Step: {validation_result.step}")
        print(f"Completion: {validation_result.completion_percentage:.1%}")
        print(f"Status: {validation_result.result.value.upper()}")

        if validation_result.requirements_met:
            print("\n✅ Requirements Met:")
            for req in validation_result.requirements_met:
                print(f"  • {req}")

        if validation_result.requirements_missing:
            print("\n❌ Requirements Missing:")
            for req in validation_result.requirements_missing:
                print(f"  • {req}")

        print("=" * 70)

    async def _request_step_confirmation(self, validation_result: StepValidationResult) -> bool:
        """Request user confirmation for step progression."""
        if not PYINQUIRER_AVAILABLE:
            # Fallback to simple input
            try:
                response = input(f"\nProceed to next step? (y/N): ").lower()
                return response.startswith('y')
            except (KeyboardInterrupt, EOFError):
                return False

        try:
            from PyInquirer import prompt

            questions = [{
                'type': 'confirm',
                'name': 'proceed',
                'message': f'Step {validation_result.step.value} ready. Proceed to next step?',
                'default': validation_result.result == ValidationResult.COMPLETED
            }]

            answers = prompt(questions)
            return answers.get('proceed', False)

        except Exception as e:
            logger.logger.error("step_confirmation_prompt_error", error=str(e))
            return False

    async def _log_step_transition(self, transition: StepTransition) -> None:
        """Log step transition to memory for audit trail."""
        try:
            if self.memory_client:
                content = (
                    f"Step Transition Completed\n"
                    f"From: Step {transition.from_step.value} ({transition.from_step})\n"
                    f"To: Step {transition.to_step.value} ({transition.to_step})\n"
                    f"Session ID: {transition.session_id}\n"
                    f"User Confirmed: {transition.user_confirmed}\n"
                    f"Timestamp: {transition.timestamp}\n"
                    f"Notes: {transition.notes or 'N/A'}\n"
                    f"Evidence: {json.dumps(transition.evidence, indent=2)}"
                )

                keywords = [
                    "step-transition",
                    f"step-{transition.from_step.value}",
                    f"step-{transition.to_step.value}",
                    transition.session_id,
                    "protocol-workflow"
                ]

                await self.memory_client.store_memory(
                    content=content,
                    content_type="decision",
                    keywords=keywords
                )

                logger.logger.debug("step_transition_logged", session_id=transition.session_id)

        except Exception as e:
            logger.logger.error("step_transition_logging_error", error=str(e))

    def get_statistics(self) -> Dict[str, Any]:
        """Get interactive step validator statistics."""
        return {
            "validations_performed": self.validations_performed,
            "transitions_completed": self.transitions_completed,
            "pyinquirer_available": PYINQUIRER_AVAILABLE,
            "memory_integration": self.memory_client is not None
        }


# Example usage and testing
if __name__ == "__main__":
    async def test_interactive_step_validator():
        """Test interactive step validator functionality."""

        validator = InteractiveStepValidator()

        # Test 1: DISCUSSION step validation
        print("🧪 Test 1: DISCUSSION step validation")
        result = await validator.validate_step_completion(
            ProtocolStep.DISCUSSION,
            "Let's discuss the implementation approach for the user authentication system. "
            "We need to consider security trade-offs and alternatives like JWT vs sessions."
        )
        assert result.step == ProtocolStep.DISCUSSION
        assert result.completion_percentage > 0.5
        print(f"✅ Discussion validation: {result.completion_percentage:.1%} complete")

        # Test 2: RESEARCH step validation
        print("\n🧪 Test 2: RESEARCH step validation")
        result = await validator.validate_step_completion(
            ProtocolStep.RESEARCH,
            "I researched best practices for API authentication and found that JWT tokens "
            "are recommended for stateless authentication. I also studied the OAuth2 specification."
        )
        assert result.step == ProtocolStep.RESEARCH
        assert len(result.requirements_met) > 0
        print(f"✅ Research validation: {len(result.requirements_met)} requirements met")

        # Test 3: PLANNING step validation
        print("\n🧪 Test 3: PLANNING step validation")
        result = await validator.validate_step_completion(
            ProtocolStep.PLANNING,
            "I'll break this down into micro-tasks: 1) Create user model, 2) Implement JWT utilities, "
            "3) Create auth endpoints, 4) Add middleware. Each task should take 15-30 minutes."
        )
        assert result.step == ProtocolStep.PLANNING
        assert "micro-task" in result.requirements_met or "breakdown" in result.requirements_met
        print(f"✅ Planning validation: {result.result.value}")

        # Test 4: Statistics
        print("\n🧪 Test 4: Validator statistics")
        stats = validator.get_statistics()
        assert "validations_performed" in stats
        assert "transitions_completed" in stats
        print(f"✅ Statistics: {stats}")

        print("\n🎉 All interactive step validator tests PASSED")
        print("\n💡 Note: Interactive confirmation testing requires manual execution")
        print("   Use in real hook execution to see interactive prompts and summaries")

    # Run tests
    asyncio.run(test_interactive_step_validator())