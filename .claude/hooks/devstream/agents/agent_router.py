"""
Agent Router for Agent Auto-Delegation System.

This module implements task complexity assessment and auto-approval logic for
delegating tasks to specialized agents. Routes tasks based on complexity,
architectural impact, and confidence thresholds.

Phase: 3 - Agent Router Implementation
Architecture: Task assessment + auto-approval + advisory message generation
Decision Threshold: ≥0.95 confidence + LOW complexity + NONE/LOW impact

Usage:
    router = AgentRouter()
    assessment = await router.assess_task_complexity(pattern_match, context)
    if router.should_auto_approve(assessment):
        # Auto-delegate to agent
    else:
        # Escalate to @tech-lead with advisory message
        message = router.format_advisory_message(assessment)
"""

from typing import Optional, Dict, Any, Literal, List
from dataclasses import dataclass
from .pattern_catalog import PatternMatch


@dataclass
class TaskAssessment:
    """
    Task complexity and delegation assessment result.

    Attributes:
        complexity: Task complexity level (LOW <30min, MEDIUM <2h, HIGH >2h)
        architectural_impact: Impact on system architecture
        recommendation: Delegation recommendation
        suggested_agent: Agent to delegate to (if DELEGATE/COORDINATE)
        confidence: Confidence score 0.0-1.0
        reason: Human-readable explanation of assessment
    """

    complexity: Literal["LOW", "MEDIUM", "HIGH"]
    architectural_impact: Literal["NONE", "LOW", "MEDIUM", "HIGH"]
    recommendation: Literal["DELEGATE", "COORDINATE", "ESCALATE"]
    suggested_agent: Optional[str]
    confidence: float
    reason: str


class AgentRouter:
    """
    Task complexity assessor and delegation router.

    Implements three-tier routing logic:
    1. DELEGATE: High confidence (≥0.95) + LOW complexity + NONE/LOW impact → Auto-approve
    2. COORDINATE: Medium confidence (0.85-0.94) OR MEDIUM complexity/impact → Advisory to @tech-lead
    3. ESCALATE: Low confidence (<0.85) OR HIGH complexity/impact → Full @tech-lead analysis

    Performance target: <5ms assessment time
    """

    # Auto-approval thresholds (configurable)
    AUTO_APPROVE_CONFIDENCE_THRESHOLD = 0.95
    AUTO_APPROVE_MAX_COMPLEXITY = "LOW"
    AUTO_APPROVE_MAX_IMPACT = "LOW"

    async def assess_task_complexity(
        self,
        pattern_match: PatternMatch,
        context: Dict[str, Any]
    ) -> TaskAssessment:
        """
        Assess task complexity and delegation suitability.

        Analyzes:
        - Pattern match confidence
        - File count and scope
        - Tool usage patterns
        - Architectural signals (migrations, refactoring, new services)

        Args:
            pattern_match: Pattern match result from PatternMatcher
            context: Tool execution context containing:
                - file_path: Optional file path
                - content: Optional file content
                - user_query: Optional user query
                - tool_name: Optional tool name
                - affected_files: Optional list of affected files

        Returns:
            TaskAssessment with complexity, impact, and recommendation

        Raises:
            ValueError: If pattern_match is invalid
        """
        if not pattern_match:
            raise ValueError("pattern_match cannot be None")

        # Extract context signals
        file_path = context.get("file_path")
        user_query = context.get("user_query", "")
        affected_files = context.get("affected_files", [])
        tool_name = context.get("tool_name")

        # Assess complexity based on signals
        complexity = self._assess_complexity_signals(
            pattern_match=pattern_match,
            user_query=user_query,
            file_path=file_path,
            affected_files=affected_files
        )

        # Assess architectural impact
        architectural_impact = self._assess_architectural_impact(
            user_query=user_query,
            file_path=file_path,
            tool_name=tool_name
        )

        # Determine recommendation
        recommendation = self._determine_recommendation(
            confidence=pattern_match["confidence"],
            complexity=complexity,
            architectural_impact=architectural_impact
        )

        # Build reason
        reason = self._build_assessment_reason(
            pattern_match=pattern_match,
            complexity=complexity,
            architectural_impact=architectural_impact,
            recommendation=recommendation
        )

        return TaskAssessment(
            complexity=complexity,
            architectural_impact=architectural_impact,
            recommendation=recommendation,
            suggested_agent=pattern_match["agent"],
            confidence=pattern_match["confidence"],
            reason=reason
        )

    def should_auto_approve(self, assessment: TaskAssessment) -> bool:
        """
        Determine if task should be auto-approved for delegation.

        Auto-approval criteria (ALL must be true):
        1. Confidence ≥ 0.95
        2. Complexity = LOW
        3. Architectural impact = NONE or LOW

        Args:
            assessment: TaskAssessment result

        Returns:
            True if task should be auto-delegated, False otherwise

        Example:
            >>> assessment = TaskAssessment(
            ...     complexity="LOW",
            ...     architectural_impact="NONE",
            ...     recommendation="DELEGATE",
            ...     suggested_agent="@python-specialist",
            ...     confidence=0.95,
            ...     reason="High confidence match"
            ... )
            >>> router.should_auto_approve(assessment)
            True
        """
        return (
            assessment.confidence >= self.AUTO_APPROVE_CONFIDENCE_THRESHOLD
            and assessment.complexity == self.AUTO_APPROVE_MAX_COMPLEXITY
            and assessment.architectural_impact in ["NONE", self.AUTO_APPROVE_MAX_IMPACT]
        )

    def format_advisory_message(self, assessment: TaskAssessment) -> str:
        """
        Format advisory message for @tech-lead context injection.

        Creates structured message with:
        - Recommended action
        - Suggested agent
        - Confidence score
        - Complexity and impact analysis
        - Reasoning

        Args:
            assessment: TaskAssessment result

        Returns:
            Formatted advisory message for @tech-lead

        Example:
            >>> assessment = TaskAssessment(...)
            >>> router.format_advisory_message(assessment)
            "ADVISORY: COORDINATE with @python-specialist\\n..."
        """
        # Header with recommendation
        header = f"ADVISORY: {assessment.recommendation}"
        if assessment.suggested_agent:
            header += f" with {assessment.suggested_agent}"

        # Metrics section
        metrics = [
            f"Confidence: {assessment.confidence:.2f}",
            f"Complexity: {assessment.complexity}",
            f"Architectural Impact: {assessment.architectural_impact}"
        ]

        # Reasoning section
        reasoning = f"Reasoning: {assessment.reason}"

        # Action recommendation
        if assessment.recommendation == "DELEGATE":
            action = f"✅ Auto-delegation approved to {assessment.suggested_agent}"
        elif assessment.recommendation == "COORDINATE":
            action = f"⚠️ Coordination recommended with {assessment.suggested_agent} (review suggested)"
        else:  # ESCALATE
            action = "🚨 Full @tech-lead analysis required (high complexity/impact)"

        # Assemble message
        message = "\n".join([
            header,
            "",
            "Metrics:",
            *[f"  • {metric}" for metric in metrics],
            "",
            reasoning,
            "",
            f"Action: {action}"
        ])

        return message

    def _assess_complexity_signals(
        self,
        pattern_match: PatternMatch,
        user_query: str,
        file_path: Optional[str],
        affected_files: List[str]
    ) -> Literal["LOW", "MEDIUM", "HIGH"]:
        """
        Assess task complexity from signals.

        Complexity levels:
        - LOW: Single file, clear pattern match, <30min estimated
        - MEDIUM: Multiple files (2-5), moderate scope, <2h estimated
        - HIGH: Many files (>5), complex refactoring, >2h estimated

        Args:
            pattern_match: Pattern match result
            user_query: User query string
            file_path: Optional file path
            affected_files: List of affected files

        Returns:
            Complexity level
        """
        # HIGH complexity signals
        high_complexity_keywords = [
            "refactor", "migrate", "redesign", "architecture",
            "multi-service", "distributed", "microservice"
        ]
        if any(keyword in user_query.lower() for keyword in high_complexity_keywords):
            return "HIGH"

        # File count signal
        file_count = len(affected_files) if affected_files else (1 if file_path else 0)
        if file_count > 5:
            return "HIGH"
        elif file_count >= 2:
            return "MEDIUM"

        # MEDIUM complexity signals
        medium_complexity_keywords = [
            "multiple", "several", "batch", "update all",
            "integrate", "connect", "extend"
        ]
        if any(keyword in user_query.lower() for keyword in medium_complexity_keywords):
            return "MEDIUM"

        # Default: LOW complexity (single file, clear pattern)
        return "LOW"

    def _assess_architectural_impact(
        self,
        user_query: str,
        file_path: Optional[str],
        tool_name: Optional[str]
    ) -> Literal["NONE", "LOW", "MEDIUM", "HIGH"]:
        """
        Assess architectural impact from signals.

        Impact levels:
        - NONE: No architectural changes
        - LOW: Minor changes to existing components
        - MEDIUM: New components, API changes
        - HIGH: Core architecture changes, migrations, breaking changes

        Args:
            user_query: User query string
            file_path: Optional file path
            tool_name: Optional tool name

        Returns:
            Architectural impact level
        """
        query_lower = user_query.lower()

        # HIGH impact signals
        high_impact_keywords = [
            "migration", "migrate", "migrating",  # Database/infrastructure changes
            "breaking change", "architecture",
            "redesign", "refactor architecture",
            "new service",  # Microservice creation
            "service", "microservice",  # Service-level changes (when combined with "new", "create")
            "new database", "database schema",
            "authentication system", "authorization system",
            "auth system"
        ]
        if any(keyword in query_lower for keyword in high_impact_keywords):
            return "HIGH"

        # MEDIUM impact signals
        medium_impact_keywords = [
            "new api", "new endpoint", "new feature",
            "integration", "integrate", "integrating",  # Third-party integrations (verb/noun forms)
            "third-party", "external api", "external service",
            "webhook", "webhooks",
            "api endpoint"
        ]
        if any(keyword in query_lower for keyword in medium_impact_keywords):
            return "MEDIUM"

        # LOW impact signals
        low_impact_keywords = [
            "fix", "update", "improve", "optimize",
            "test", "document", "refactor function"
        ]
        if any(keyword in query_lower for keyword in low_impact_keywords):
            return "LOW"

        # Default: NONE (no clear architectural impact)
        return "NONE"

    def _determine_recommendation(
        self,
        confidence: float,
        complexity: Literal["LOW", "MEDIUM", "HIGH"],
        architectural_impact: Literal["NONE", "LOW", "MEDIUM", "HIGH"]
    ) -> Literal["DELEGATE", "COORDINATE", "ESCALATE"]:
        """
        Determine delegation recommendation.

        Logic:
        - DELEGATE: Confidence ≥0.95 + LOW complexity + (NONE or LOW impact)
        - COORDINATE: Confidence ≥0.85 + MEDIUM complexity/impact
        - ESCALATE: Confidence <0.85 OR HIGH complexity/impact

        Args:
            confidence: Pattern match confidence
            complexity: Task complexity
            architectural_impact: Architectural impact

        Returns:
            Delegation recommendation
        """
        # ESCALATE for high complexity/impact or low confidence
        if complexity == "HIGH" or architectural_impact == "HIGH" or confidence < 0.85:
            return "ESCALATE"

        # DELEGATE for high confidence + low complexity/impact
        if (
            confidence >= self.AUTO_APPROVE_CONFIDENCE_THRESHOLD
            and complexity == "LOW"
            and architectural_impact in ["NONE", "LOW"]
        ):
            return "DELEGATE"

        # Default: COORDINATE (medium confidence/complexity)
        return "COORDINATE"

    def _build_assessment_reason(
        self,
        pattern_match: PatternMatch,
        complexity: Literal["LOW", "MEDIUM", "HIGH"],
        architectural_impact: Literal["NONE", "LOW", "MEDIUM", "HIGH"],
        recommendation: Literal["DELEGATE", "COORDINATE", "ESCALATE"]
    ) -> str:
        """
        Build human-readable assessment reason.

        Args:
            pattern_match: Pattern match result
            complexity: Task complexity
            architectural_impact: Architectural impact
            recommendation: Delegation recommendation

        Returns:
            Human-readable reason string
        """
        # Base reason from pattern match
        base_reason = pattern_match["reason"]

        # Add complexity context
        complexity_context = {
            "LOW": "single-file, well-defined scope",
            "MEDIUM": "multi-file or moderate scope",
            "HIGH": "complex refactoring or broad scope"
        }

        # Add impact context
        impact_context = {
            "NONE": "no architectural impact",
            "LOW": "minor component changes",
            "MEDIUM": "new components or API changes",
            "HIGH": "core architecture modifications"
        }

        # Build full reason
        reason_parts = [
            base_reason,
            f"Task complexity: {complexity} ({complexity_context[complexity]})",
            f"Architectural impact: {architectural_impact} ({impact_context[architectural_impact]})"
        ]

        # Add recommendation context
        if recommendation == "DELEGATE":
            reason_parts.append("Auto-delegation criteria met (high confidence, low risk)")
        elif recommendation == "COORDINATE":
            reason_parts.append("Coordination recommended (moderate complexity/impact)")
        else:
            reason_parts.append("Full analysis required (high complexity/impact or low confidence)")

        return "; ".join(reason_parts)
