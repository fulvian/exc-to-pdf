"""
Pattern Matcher for Agent Auto-Delegation System.

This module implements fast pattern-based agent routing (<10ms target) using
file extensions, import statements, keyword frequency analysis, and context matching.

Phase: 2 - Pattern Matcher Implementation
Architecture: Hybrid pattern matching + LLM routing
Performance Target: <10ms for 90% cases

Usage:
    matcher = PatternMatcher()
    result = matcher.match_patterns(
        file_path="/path/to/file.py",
        content="import fastapi",
        user_query="create API endpoint"
    )
    if result:
        print(f"Agent: {result['agent']}, Confidence: {result['confidence']}")
"""

from typing import Optional, Dict, List, Pattern
import re
import os
import sys
from pathlib import Path

# Multi-project import handling with fallback mechanisms
def _import_pattern_catalog():
    """Import pattern catalog with multi-project fallback support."""

    # Strategy 1: Try relative import (when run as module)
    try:
        from .pattern_catalog import (
            PatternMatch,
            PatternRule,
            PATTERN_CATALOG,
            QUALITY_GATE_PATTERNS,
            PYTHON_PATTERNS,
            TYPESCRIPT_PATTERNS,
            RUST_PATTERNS,
            GO_PATTERNS,
            SECURITY_PATTERNS,
            DATABASE_PATTERNS,
            TESTING_PATTERNS,
            PERFORMANCE_PATTERNS,
            REFACTORING_PATTERNS,
            API_PATTERNS,
            DEBUGGING_PATTERNS,
            INTEGRATION_PATTERNS,
            DEPLOYMENT_PATTERNS,
            DOCUMENTATION_PATTERNS,
            MIGRATION_PATTERNS,
        )
        return {
            'PatternMatch': PatternMatch,
            'PatternRule': PatternRule,
            'PATTERN_CATALOG': PATTERN_CATALOG,
            'QUALITY_GATE_PATTERNS': QUALITY_GATE_PATTERNS,
            'PYTHON_PATTERNS': PYTHON_PATTERNS,
            'TYPESCRIPT_PATTERNS': TYPESCRIPT_PATTERNS,
            'RUST_PATTERNS': RUST_PATTERNS,
            'GO_PATTERNS': GO_PATTERNS,
            'SECURITY_PATTERNS': SECURITY_PATTERNS,
            'DATABASE_PATTERNS': DATABASE_PATTERNS,
            'TESTING_PATTERNS': TESTING_PATTERNS,
            'PERFORMANCE_PATTERNS': PERFORMANCE_PATTERNS,
            'REFACTORING_PATTERNS': REFACTORING_PATTERNS,
            'API_PATTERNS': API_PATTERNS,
            'DEBUGGING_PATTERNS': DEBUGGING_PATTERNS,
            'INTEGRATION_PATTERNS': INTEGRATION_PATTERNS,
            'DEPLOYMENT_PATTERNS': DEPLOYMENT_PATTERNS,
            'DOCUMENTATION_PATTERNS': DOCUMENTATION_PATTERNS,
            'MIGRATION_PATTERNS': MIGRATION_PATTERNS,
        }
    except ImportError:
        pass

    # Strategy 2: Try absolute import from current directory
    try:
        from pattern_catalog import (
            PatternMatch,
            PatternRule,
            PATTERN_CATALOG,
            QUALITY_GATE_PATTERNS,
            PYTHON_PATTERNS,
            TYPESCRIPT_PATTERNS,
            RUST_PATTERNS,
            GO_PATTERNS,
            SECURITY_PATTERNS,
            DATABASE_PATTERNS,
            TESTING_PATTERNS,
            PERFORMANCE_PATTERNS,
            REFACTORING_PATTERNS,
            API_PATTERNS,
            DEBUGGING_PATTERNS,
            INTEGRATION_PATTERNS,
            DEPLOYMENT_PATTERNS,
            DOCUMENTATION_PATTERNS,
            MIGRATION_PATTERNS,
        )
        return {
            'PatternMatch': PatternMatch,
            'PatternRule': PatternRule,
            'PATTERN_CATALOG': PATTERN_CATALOG,
            'QUALITY_GATE_PATTERNS': QUALITY_GATE_PATTERNS,
            'PYTHON_PATTERNS': PYTHON_PATTERNS,
            'TYPESCRIPT_PATTERNS': TYPESCRIPT_PATTERNS,
            'RUST_PATTERNS': RUST_PATTERNS,
            'GO_PATTERNS': GO_PATTERNS,
            'SECURITY_PATTERNS': SECURITY_PATTERNS,
            'DATABASE_PATTERNS': DATABASE_PATTERNS,
            'TESTING_PATTERNS': TESTING_PATTERNS,
            'PERFORMANCE_PATTERNS': PERFORMANCE_PATTERNS,
            'REFACTORING_PATTERNS': REFACTORING_PATTERNS,
            'API_PATTERNS': API_PATTERNS,
            'DEBUGGING_PATTERNS': DEBUGGING_PATTERNS,
            'INTEGRATION_PATTERNS': INTEGRATION_PATTERNS,
            'DEPLOYMENT_PATTERNS': DEPLOYMENT_PATTERNS,
            'DOCUMENTATION_PATTERNS': DOCUMENTATION_PATTERNS,
            'MIGRATION_PATTERNS': MIGRATION_PATTERNS,
        }
    except ImportError:
        pass

    # Strategy 3: Try import from DevStream root (multi-project context)
    current_file = Path(__file__).resolve()
    agents_dir = current_file.parent

    # Look for pattern_catalog.py in various locations
    search_paths = [
        agents_dir / 'pattern_catalog.py',  # Same directory
        agents_dir.parent / 'pattern_catalog.py',  # Parent directory
        Path.cwd() / '.claude' / 'hooks' / 'devstream' / 'agents' / 'pattern_catalog.py',  # Multi-project context
        Path.cwd() / '.claude' / 'hooks' / 'devstream' / 'pattern_catalog.py',  # Alternative multi-project structure
    ]

    for search_path in search_paths:
        if search_path.exists():
            try:
                # Add the directory to Python path
                import_dir = str(search_path.parent)
                if import_dir not in sys.path:
                    sys.path.insert(0, import_dir)

                # Import the module
                import importlib.util
                spec = importlib.util.spec_from_file_location("pattern_catalog", search_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                return {
                    'PatternMatch': module.PatternMatch,
                    'PatternRule': module.PatternRule,
                    'PATTERN_CATALOG': module.PATTERN_CATALOG,
                    'QUALITY_GATE_PATTERNS': module.QUALITY_GATE_PATTERNS,
                    'PYTHON_PATTERNS': module.PYTHON_PATTERNS,
                    'TYPESCRIPT_PATTERNS': module.TYPESCRIPT_PATTERNS,
                    'RUST_PATTERNS': module.RUST_PATTERNS,
                    'GO_PATTERNS': module.GO_PATTERNS,
                    'SECURITY_PATTERNS': module.SECURITY_PATTERNS,
                    'DATABASE_PATTERNS': module.DATABASE_PATTERNS,
                    'TESTING_PATTERNS': module.TESTING_PATTERNS,
                    'PERFORMANCE_PATTERNS': module.PERFORMANCE_PATTERNS,
                    'REFACTORING_PATTERNS': module.REFACTORING_PATTERNS,
                    'API_PATTERNS': module.API_PATTERNS,
                    'DEBUGGING_PATTERNS': module.DEBUGGING_PATTERNS,
                    'INTEGRATION_PATTERNS': module.INTEGRATION_PATTERNS,
                    'DEPLOYMENT_PATTERNS': module.DEPLOYMENT_PATTERNS,
                    'DOCUMENTATION_PATTERNS': module.DOCUMENTATION_PATTERNS,
                    'MIGRATION_PATTERNS': module.MIGRATION_PATTERNS,
                }
            except Exception:
                continue

    # Strategy 4: Create minimal fallback patterns
    return _create_fallback_patterns()

def _create_fallback_patterns():
    """Create minimal fallback patterns when pattern catalog is unavailable."""

    # Define basic TypedDict classes
    class PatternMatch(dict):
        def __init__(self, agent=None, confidence=0.0, reason="", method=""):
            super().__init__({
                'agent': agent,
                'confidence': confidence,
                'reason': reason,
                'method': method
            })

    class PatternRule(dict):
        def __init__(self, **kwargs):
            super().__init__(kwargs)

    # Create minimal pattern catalog with essential patterns
    PATTERN_CATALOG = {
        "python_development": {
            "extensions": [".py", ".pyi"],
            "keywords": ["python", "fastapi", "django", "async"],
            "agent": "@python-specialist",
            "confidence": 0.9
        },
        "typescript_development": {
            "extensions": [".ts", ".tsx"],
            "keywords": ["typescript", "react", "nextjs"],
            "agent": "@typescript-specialist",
            "confidence": 0.9
        },
        "database_work": {
            "keywords": ["database", "sql", "postgresql", "migration"],
            "agent": "@database-specialist",
            "confidence": 0.85
        },
        "testing_work": {
            "keywords": ["test", "pytest", "testing"],
            "agent": "@testing-specialist",
            "confidence": 0.85
        },
        "pre_commit_review": {
            "keywords": ["review", "commit", "code review"],
            "agent": "@code-reviewer",
            "confidence": 1.0,
            "mandatory": True
        }
    }

    return {
        'PatternMatch': PatternMatch,
        'PatternRule': PatternRule,
        'PATTERN_CATALOG': PATTERN_CATALOG,
        'QUALITY_GATE_PATTERNS': {k: v for k, v in PATTERN_CATALOG.items() if v.get('mandatory')},
        'PYTHON_PATTERNS': {'python_development': PATTERN_CATALOG['python_development']},
        'TYPESCRIPT_PATTERNS': {'typescript_development': PATTERN_CATALOG['typescript_development']},
        'RUST_PATTERNS': {},
        'GO_PATTERNS': {},
        'SECURITY_PATTERNS': {},
        'DATABASE_PATTERNS': {'database_work': PATTERN_CATALOG['database_work']},
        'TESTING_PATTERNS': {'testing_work': PATTERN_CATALOG['testing_work']},
        'PERFORMANCE_PATTERNS': {},
        'REFACTORING_PATTERNS': {},
        'API_PATTERNS': {},
        'DEBUGGING_PATTERNS': {},
        'INTEGRATION_PATTERNS': {},
        'DEPLOYMENT_PATTERNS': {},
        'DOCUMENTATION_PATTERNS': {},
        'MIGRATION_PATTERNS': {},
    }

# Import pattern catalog with fallback support
_pattern_exports = _import_pattern_catalog()

# Extract all exports
PatternMatch = _pattern_exports['PatternMatch']
PatternRule = _pattern_exports['PatternRule']
PATTERN_CATALOG = _pattern_exports['PATTERN_CATALOG']
QUALITY_GATE_PATTERNS = _pattern_exports['QUALITY_GATE_PATTERNS']
PYTHON_PATTERNS = _pattern_exports['PYTHON_PATTERNS']
TYPESCRIPT_PATTERNS = _pattern_exports['TYPESCRIPT_PATTERNS']
RUST_PATTERNS = _pattern_exports['RUST_PATTERNS']
GO_PATTERNS = _pattern_exports['GO_PATTERNS']
SECURITY_PATTERNS = _pattern_exports['SECURITY_PATTERNS']
DATABASE_PATTERNS = _pattern_exports['DATABASE_PATTERNS']
TESTING_PATTERNS = _pattern_exports['TESTING_PATTERNS']
PERFORMANCE_PATTERNS = _pattern_exports['PERFORMANCE_PATTERNS']
REFACTORING_PATTERNS = _pattern_exports['REFACTORING_PATTERNS']
API_PATTERNS = _pattern_exports['API_PATTERNS']
DEBUGGING_PATTERNS = _pattern_exports['DEBUGGING_PATTERNS']
INTEGRATION_PATTERNS = _pattern_exports['INTEGRATION_PATTERNS']
DEPLOYMENT_PATTERNS = _pattern_exports['DEPLOYMENT_PATTERNS']
DOCUMENTATION_PATTERNS = _pattern_exports['DOCUMENTATION_PATTERNS']
MIGRATION_PATTERNS = _pattern_exports['MIGRATION_PATTERNS']


class PatternMatcher:
    """
    Fast pattern-based agent routing (<10ms target).

    Implements priority-based matching:
    1. Quality gates (if tool = 'git commit') - Confidence 1.0
    2. Coding patterns (if file_path provided) - Confidence 0.9
    3. Security patterns (if keywords match) - Confidence 0.95
    4. Task-specific patterns (if contexts match) - Confidence 0.85

    Performance optimizations:
    - Precompiled regex patterns
    - Dict lookup for file extensions (O(1))
    - Early exit when confidence >= 0.95
    - Cached keyword extraction
    """

    def __init__(self) -> None:
        """Initialize pattern matcher with precompiled regex patterns."""
        # Precompile all import regex patterns for performance
        self._import_patterns: Dict[str, List[Pattern[str]]] = {}
        self._shebang_patterns: Dict[str, Pattern[str]] = {}
        self._extension_map: Dict[str, str] = {}  # Fast O(1) lookup

        # Build extension lookup map
        for pattern_name, rule in PATTERN_CATALOG.items():
            extensions = rule.get("extensions")
            if extensions:
                for ext in extensions:
                    # Map extension to pattern name (highest confidence wins)
                    if ext not in self._extension_map:
                        self._extension_map[ext] = pattern_name
                    else:
                        # Keep pattern with higher confidence
                        existing_conf = PATTERN_CATALOG[self._extension_map[ext]]["confidence"]
                        new_conf = rule["confidence"]
                        if new_conf > existing_conf:
                            self._extension_map[ext] = pattern_name

            # Store precompiled import patterns
            imports = rule.get("imports")
            if imports:
                if pattern_name not in self._import_patterns:
                    self._import_patterns[pattern_name] = []
                self._import_patterns[pattern_name].append(imports)

            # Store precompiled shebang patterns
            shebang = rule.get("shebang")
            if shebang:
                self._shebang_patterns[pattern_name] = shebang

    def match_patterns(
        self,
        file_path: Optional[str] = None,
        content: Optional[str] = None,
        user_query: Optional[str] = None,
        tool_name: Optional[str] = None
    ) -> Optional[PatternMatch]:
        """
        Match context against pattern catalog.

        Priority order:
        1. Quality gates (if tool = 'git commit')
        2. Coding patterns (if file_path provided)
        3. Security patterns (if keywords match)
        4. Task-specific patterns (if contexts match)

        Args:
            file_path: Optional file path being worked on
            content: Optional file content or user query text
            user_query: Optional user query string
            tool_name: Optional tool name (e.g., 'git', 'Bash')

        Returns:
            Highest confidence match or None if no match found

        Performance:
            - Target: <10ms for 90% cases
            - Early exit when confidence >= 0.95
            - Precompiled regex for fast matching
        """
        # Priority 1: Quality gates (mandatory, highest confidence)
        if tool_name:
            match = self.check_quality_gates(tool_name)
            if match and match["confidence"] >= 0.95:
                return match  # Early exit for mandatory gates

        # Priority 2: Coding patterns (file-based, high confidence)
        if file_path:
            match = self.check_coding_patterns(file_path, content)
            if match:
                return match  # Return any coding pattern match

        # Priority 3: Security patterns (keyword-based, high confidence)
        if content or user_query:
            combined_text = f"{content or ''} {user_query or ''}"
            match = self.check_security_patterns(combined_text)
            if match:
                return match  # Return any security pattern match

        # Priority 4: Task-specific patterns (context-based, medium confidence)
        match = self.check_task_patterns(content, user_query)
        if match:
            return match

        return None

    def check_quality_gates(self, tool_name: Optional[str]) -> Optional[PatternMatch]:
        """
        Check mandatory quality gate patterns.

        Quality gates trigger on specific tool usage (e.g., git commit)
        and route to @code-reviewer with confidence 1.0 (mandatory).

        Args:
            tool_name: Tool name being invoked (e.g., 'git', 'Bash')

        Returns:
            PatternMatch for quality gate or None
        """
        if not tool_name:
            return None

        tool_lower = tool_name.lower()

        # Check pre-commit review pattern
        if "git" in tool_lower or "commit" in tool_lower:
            return PatternMatch(
                agent="@code-reviewer",
                confidence=1.0,
                reason="Quality gate: Pre-commit code review required",
                method="mandatory"
            )

        # Check pre-merge review pattern
        if any(keyword in tool_lower for keyword in ["merge", "pull", "pr"]):
            return PatternMatch(
                agent="@code-reviewer",
                confidence=1.0,
                reason="Quality gate: Pre-merge review required",
                method="mandatory"
            )

        return None

    def check_coding_patterns(
        self,
        file_path: str,
        content: Optional[str] = None
    ) -> Optional[PatternMatch]:
        """
        Check language-based coding patterns.

        Uses fast path:
        1. File extension lookup (O(1)) - Confidence 0.95
        2. Import statement detection - Confidence 0.90
        3. Shebang detection - Confidence 0.90

        Args:
            file_path: File path being worked on
            content: Optional file content for import detection

        Returns:
            PatternMatch for coding pattern or None
        """
        # Fast path: File extension lookup (O(1))
        _, ext = os.path.splitext(file_path)
        if ext in self._extension_map:
            pattern_name = self._extension_map[ext]
            rule = PATTERN_CATALOG[pattern_name]
            return PatternMatch(
                agent=rule["agent"],
                confidence=0.95,  # High confidence for exact extension match
                reason=f"File extension '{ext}' matched {rule['agent']}",
                method="extension"
            )

        # Check shebang if content provided
        if content:
            first_line = content.split("\n", 1)[0] if content else ""
            for pattern_name, shebang_pattern in self._shebang_patterns.items():
                if shebang_pattern.match(first_line):
                    rule = PATTERN_CATALOG[pattern_name]
                    return PatternMatch(
                        agent=rule["agent"],
                        confidence=0.90,
                        reason=f"Shebang matched {rule['agent']}",
                        method="shebang"
                    )

            # Check import statements
            for pattern_name, import_patterns in self._import_patterns.items():
                for import_pattern in import_patterns:
                    if import_pattern.search(content):
                        rule = PATTERN_CATALOG[pattern_name]
                        return PatternMatch(
                            agent=rule["agent"],
                            confidence=0.90,
                            reason=f"Import statement matched {rule['agent']}",
                            method="import"
                        )

        return None

    def check_security_patterns(self, content: str) -> Optional[PatternMatch]:
        """
        Check security-related patterns.

        Matches against security keywords and imports:
        - Authentication, OAuth, JWT
        - Cryptography, encryption, hashing
        - Vulnerability scanning, OWASP

        Args:
            content: Content to check for security patterns

        Returns:
            PatternMatch for security pattern or None
        """
        if not content:
            return None

        content_lower = content.lower()

        # Check security patterns in priority order
        for pattern_name, rule in SECURITY_PATTERNS.items():
            # Check keywords
            keywords = rule.get("keywords", [])
            keyword_matches = sum(1 for kw in keywords if kw.lower() in content_lower)

            if keyword_matches >= 2:  # Require at least 2 keyword matches
                return PatternMatch(
                    agent=rule["agent"],
                    confidence=rule["confidence"],
                    reason=f"Security keywords matched: {pattern_name}",
                    method="keyword"
                )

            # Check import patterns
            imports = rule.get("imports")
            if imports and imports.search(content):
                return PatternMatch(
                    agent=rule["agent"],
                    confidence=rule["confidence"],
                    reason=f"Security import matched: {pattern_name}",
                    method="import"
                )

        return None

    def check_task_patterns(
        self,
        content: Optional[str] = None,
        user_query: Optional[str] = None
    ) -> Optional[PatternMatch]:
        """
        Check task-specific patterns.

        Matches against:
        - Database work
        - Testing work
        - Performance optimization
        - Refactoring
        - API design
        - Debugging
        - Integration
        - Deployment
        - Documentation
        - Migration

        Args:
            content: Optional file content
            user_query: Optional user query string

        Returns:
            PatternMatch for task pattern or None
        """
        combined_text = f"{content or ''} {user_query or ''}".lower()

        if not combined_text.strip():
            return None

        # Aggregate all task-specific patterns
        task_pattern_groups = [
            DATABASE_PATTERNS,
            TESTING_PATTERNS,
            PERFORMANCE_PATTERNS,
            REFACTORING_PATTERNS,
            API_PATTERNS,
            DEBUGGING_PATTERNS,
            INTEGRATION_PATTERNS,
            DEPLOYMENT_PATTERNS,
            DOCUMENTATION_PATTERNS,
            MIGRATION_PATTERNS,
        ]

        best_match: Optional[PatternMatch] = None
        best_score = 0.0

        for pattern_group in task_pattern_groups:
            for pattern_name, rule in pattern_group.items():
                # Keyword frequency scoring
                keywords = rule.get("keywords", [])
                keyword_matches = sum(1 for kw in keywords if kw.lower() in combined_text)

                # Context phrase matching
                contexts = rule.get("contexts", [])
                context_matches = sum(1 for ctx in contexts if ctx.lower() in combined_text)

                # Calculate score (keyword + context matches)
                total_matches = keyword_matches + context_matches
                if total_matches > best_score:
                    best_score = total_matches
                    best_match = PatternMatch(
                        agent=rule["agent"],
                        confidence=rule["confidence"],
                        reason=f"Task pattern matched: {pattern_name} ({total_matches} matches)",
                        method="context"
                    )

        # Require at least 2 matches for confidence
        if best_match and best_score >= 2:
            return best_match

        return None
