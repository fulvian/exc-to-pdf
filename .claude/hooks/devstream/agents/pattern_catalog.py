"""
Pattern Catalog for Agent Auto-Delegation System.

This module defines routing patterns for automatic agent selection using
pattern matching (90% cases, <10ms latency). Patterns include file extensions,
import statements, keywords, and context phrases.

Phase: 1 - Pattern Catalog Definition
Architecture: Hybrid pattern matching + LLM routing
Performance Target: <10ms for pattern matching, ~500ms for LLM fallback
"""

from typing import TypedDict, List, Optional, Pattern, Dict
import re


class PatternRule(TypedDict, total=False):
    """
    Agent routing pattern rule.

    Attributes:
        keywords: Keyword list for matching user queries
        imports: Compiled regex for import statement detection
        files: File glob patterns (e.g., "*.py", "Dockerfile")
        contexts: Context phrases for semantic matching
        extensions: File extensions (e.g., [".py", ".pyi"])
        shebang: Shebang pattern for script detection
        agent: Agent name (e.g., '@python-specialist')
        confidence: Confidence score (0.0-1.0)
        mandatory: True for quality gates (e.g., pre-commit reviews)
    """
    keywords: List[str]
    imports: Optional[Pattern[str]]
    files: Optional[List[str]]
    contexts: Optional[List[str]]
    extensions: Optional[List[str]]
    shebang: Optional[Pattern[str]]
    agent: str
    confidence: float
    mandatory: Optional[bool]


class PatternMatch(TypedDict):
    """
    Result of pattern matching.

    Attributes:
        agent: Matched agent name (None if no match)
        confidence: Confidence score (0.0-1.0)
        reason: Human-readable matching reason
        method: Matching method used
    """
    agent: Optional[str]
    confidence: float
    reason: str
    method: str  # 'extension', 'import', 'keyword', 'context', 'file', 'mandatory'


# ============================================================================
# QUALITY GATE PATTERNS (Mandatory, Confidence 1.0)
# ============================================================================

QUALITY_GATE_PATTERNS: Dict[str, PatternRule] = {
    "pre_commit_review": {
        "keywords": [
            "review", "commit", "before commit", "pre-commit",
            "code review", "check code", "validate code"
        ],
        "contexts": [
            "before committing",
            "ready to commit",
            "review my changes",
            "check before merge",
            "validate implementation"
        ],
        "agent": "@code-reviewer",
        "confidence": 1.0,
        "mandatory": True
    },
    "pre_merge_review": {
        "keywords": [
            "merge", "pull request", "pr review", "merge review",
            "before merge", "pre-merge"
        ],
        "contexts": [
            "before merging",
            "ready for pr",
            "create pull request",
            "review pull request"
        ],
        "agent": "@code-reviewer",
        "confidence": 1.0,
        "mandatory": True
    }
}


# ============================================================================
# SECURITY PATTERNS (Confidence 0.95)
# ============================================================================

SECURITY_PATTERNS: Dict[str, PatternRule] = {
    "authentication": {
        "keywords": [
            "authentication", "auth", "login", "password", "oauth",
            "jwt", "token", "session", "credentials", "security"
        ],
        "contexts": [
            "secure authentication",
            "user login",
            "access control",
            "authorization system",
            "password hashing"
        ],
        "agent": "@security-auditor",
        "confidence": 0.95,
        "mandatory": False
    },
    "cryptography": {
        "keywords": [
            "encryption", "cryptography", "crypto", "hash", "hashing",
            "cipher", "aes", "rsa", "signing", "certificate", "ssl", "tls"
        ],
        "contexts": [
            "encrypt data",
            "decrypt data",
            "secure communication",
            "cryptographic function",
            "hash password"
        ],
        "imports": re.compile(
            r"^\s*(?:from|import)\s+(?:cryptography|hashlib|bcrypt|passlib|jwt|secrets)\b",
            re.MULTILINE
        ),
        "agent": "@security-auditor",
        "confidence": 0.95,
        "mandatory": False
    },
    "vulnerability_scanning": {
        "keywords": [
            "vulnerability", "security audit", "penetration test",
            "owasp", "sql injection", "xss", "csrf", "security scan"
        ],
        "contexts": [
            "check for vulnerabilities",
            "security audit",
            "find security issues",
            "owasp top 10"
        ],
        "agent": "@security-auditor",
        "confidence": 0.95,
        "mandatory": False
    }
}


# ============================================================================
# CODING PATTERNS - PYTHON (Confidence 0.9)
# ============================================================================

PYTHON_PATTERNS: Dict[str, PatternRule] = {
    "python_development": {
        "keywords": [
            "python", "fastapi", "django", "flask", "async",
            "asyncio", "pydantic", "pytest", "type hints"
        ],
        "extensions": [".py", ".pyi", ".pyw"],
        "shebang": re.compile(r"^#!\s*/usr/bin/(?:env\s+)?python[0-9.]*$"),
        "imports": re.compile(
            r"^\s*(?:from|import)\s+(?:fastapi|django|flask|asyncio|pydantic|pytest|typing)\b",
            re.MULTILINE
        ),
        "contexts": [
            "python code",
            "python api",
            "fastapi endpoint",
            "django model",
            "async python",
            "python testing"
        ],
        "agent": "@python-specialist",
        "confidence": 0.9,
        "mandatory": False
    }
}


# ============================================================================
# CODING PATTERNS - TYPESCRIPT (Confidence 0.9)
# ============================================================================

TYPESCRIPT_PATTERNS: Dict[str, PatternRule] = {
    "typescript_development": {
        "keywords": [
            "typescript", "react", "nextjs", "next.js", "tsx",
            "node.js", "express", "nest.js"
        ],
        "extensions": [".ts", ".tsx", ".mts", ".cts"],
        "imports": re.compile(
            r"^\s*(?:import|export)\s+.*\s+from\s+['\"](?:react|next|express|@nestjs)\b",
            re.MULTILINE
        ),
        "contexts": [
            "typescript code",
            "react component",
            "next.js page",
            "typescript api",
            "node.js server",
            "tsx component"
        ],
        "agent": "@typescript-specialist",
        "confidence": 0.9,
        "mandatory": False
    },
    "javascript_development": {
        "keywords": [
            "javascript", "js", "node", "npm", "package.json",
            "webpack", "babel"
        ],
        "extensions": [".js", ".jsx", ".mjs", ".cjs"],
        "shebang": re.compile(r"^#!\s*/usr/bin/(?:env\s+)?node$"),
        "contexts": [
            "javascript code",
            "js script",
            "node script",
            "javascript function"
        ],
        "agent": "@typescript-specialist",
        "confidence": 0.85,  # Lower confidence for JS vs TS
        "mandatory": False
    }
}


# ============================================================================
# CODING PATTERNS - RUST (Confidence 0.9)
# ============================================================================

RUST_PATTERNS: Dict[str, PatternRule] = {
    "rust_development": {
        "keywords": [
            "rust", "cargo", "tokio", "async rust", "actix",
            "warp", "rocket", "rustc", "borrowing", "lifetime"
        ],
        "extensions": [".rs"],
        "files": ["Cargo.toml", "Cargo.lock"],
        "imports": re.compile(
            r"^\s*(?:use|extern\s+crate)\s+(?:tokio|actix|warp|rocket|serde)\b",
            re.MULTILINE
        ),
        "contexts": [
            "rust code",
            "rust program",
            "cargo project",
            "async rust",
            "rust api"
        ],
        "agent": "@rust-specialist",
        "confidence": 0.9,
        "mandatory": False
    }
}


# ============================================================================
# CODING PATTERNS - GO (Confidence 0.9)
# ============================================================================

GO_PATTERNS: Dict[str, PatternRule] = {
    "go_development": {
        "keywords": [
            "go", "golang", "goroutine", "channel", "go module",
            "gin", "echo", "fiber", "grpc"
        ],
        "extensions": [".go"],
        "files": ["go.mod", "go.sum"],
        "shebang": re.compile(r"^#!\s*/usr/bin/(?:env\s+)?go\s+run$"),
        "imports": re.compile(
            r"^\s*import\s+(?:\(|\")\s*(?:github\.com|golang\.org)",
            re.MULTILINE
        ),
        "contexts": [
            "go code",
            "golang program",
            "go module",
            "goroutine",
            "go api"
        ],
        "agent": "@go-specialist",
        "confidence": 0.9,
        "mandatory": False
    }
}


# ============================================================================
# TASK-SPECIFIC PATTERNS (Confidence 0.85)
# ============================================================================

DATABASE_PATTERNS: Dict[str, PatternRule] = {
    "database_work": {
        "keywords": [
            "database", "sql", "postgresql", "mysql", "mongodb",
            "redis", "migration", "schema", "query optimization",
            "index", "orm", "sqlalchemy", "prisma"
        ],
        "files": [
            "*.sql", "schema.sql", "migrations/*", "alembic/*",
            "prisma/schema.prisma"
        ],
        "imports": re.compile(
            r"^\s*(?:from|import)\s+(?:sqlalchemy|psycopg|pymongo|redis|prisma)\b",
            re.MULTILINE
        ),
        "contexts": [
            "database design",
            "sql query",
            "database migration",
            "query optimization",
            "database schema"
        ],
        "agent": "@database-specialist",
        "confidence": 0.85,
        "mandatory": False
    }
}

TESTING_PATTERNS: Dict[str, PatternRule] = {
    "testing_work": {
        "keywords": [
            "test", "testing", "pytest", "jest", "unit test",
            "integration test", "e2e", "coverage", "mock",
            "fixture", "test suite"
        ],
        "files": [
            "test_*.py", "*_test.py", "*.test.ts", "*.test.js",
            "*.spec.ts", "*.spec.js", "tests/*", "__tests__/*"
        ],
        "imports": re.compile(
            r"^\s*(?:from|import)\s+(?:pytest|unittest|jest|@testing-library)\b",
            re.MULTILINE
        ),
        "contexts": [
            "write tests",
            "test coverage",
            "unit testing",
            "integration testing",
            "test strategy"
        ],
        "agent": "@testing-specialist",
        "confidence": 0.85,
        "mandatory": False
    }
}

PERFORMANCE_PATTERNS: Dict[str, PatternRule] = {
    "performance_work": {
        "keywords": [
            "performance", "optimization", "profiling", "benchmark",
            "scalability", "latency", "throughput", "caching",
            "load balancing", "bottleneck"
        ],
        "contexts": [
            "optimize performance",
            "improve speed",
            "reduce latency",
            "performance bottleneck",
            "scalability issue",
            "profiling results"
        ],
        "agent": "@performance-optimizer",
        "confidence": 0.85,
        "mandatory": False
    }
}

REFACTORING_PATTERNS: Dict[str, PatternRule] = {
    "refactoring_work": {
        "keywords": [
            "refactor", "refactoring", "clean code", "code smell",
            "technical debt", "extract", "rename", "restructure",
            "simplify", "modularize"
        ],
        "contexts": [
            "refactor code",
            "clean up code",
            "improve structure",
            "remove duplication",
            "technical debt reduction"
        ],
        "agent": "@refactoring-specialist",
        "confidence": 0.85,
        "mandatory": False
    }
}

API_PATTERNS: Dict[str, PatternRule] = {
    "api_design": {
        "keywords": [
            "api", "rest", "graphql", "endpoint", "route",
            "api design", "openapi", "swagger", "grpc",
            "rest api", "graphql schema"
        ],
        "files": [
            "openapi.yaml", "swagger.yaml", "*.graphql",
            "schema.graphql"
        ],
        "contexts": [
            "api design",
            "rest endpoint",
            "graphql schema",
            "api architecture",
            "api documentation"
        ],
        "agent": "@api-architect",
        "confidence": 0.85,
        "mandatory": False
    }
}

DEBUGGING_PATTERNS: Dict[str, PatternRule] = {
    "debugging_work": {
        "keywords": [
            "debug", "debugging", "bug", "error", "exception",
            "traceback", "stack trace", "breakpoint", "logging",
            "troubleshoot"
        ],
        "contexts": [
            "debug issue",
            "fix bug",
            "error investigation",
            "troubleshoot problem",
            "find root cause"
        ],
        "agent": "@debugger",
        "confidence": 0.85,
        "mandatory": False
    }
}

INTEGRATION_PATTERNS: Dict[str, PatternRule] = {
    "integration_work": {
        "keywords": [
            "integration", "integrate", "third party", "api integration",
            "webhook", "external service", "microservice",
            "service communication"
        ],
        "contexts": [
            "integrate service",
            "third party integration",
            "connect services",
            "microservice integration",
            "webhook setup"
        ],
        "agent": "@integration-specialist",
        "confidence": 0.85,
        "mandatory": False
    }
}

DEPLOYMENT_PATTERNS: Dict[str, PatternRule] = {
    "deployment_work": {
        "keywords": [
            "deployment", "deploy", "devops", "ci/cd", "docker",
            "kubernetes", "k8s", "helm", "terraform", "ansible",
            "github actions", "gitlab ci"
        ],
        "files": [
            "Dockerfile", "docker-compose.yaml", "docker-compose.yml",
            "*.tf", "*.tfvars", "k8s/*", "kubernetes/*",
            ".github/workflows/*", ".gitlab-ci.yml"
        ],
        "contexts": [
            "deploy application",
            "devops setup",
            "ci/cd pipeline",
            "docker container",
            "kubernetes deployment"
        ],
        "agent": "@devops-specialist",
        "confidence": 0.85,
        "mandatory": False
    }
}


# ============================================================================
# DOCUMENTATION PATTERNS (Confidence 0.85)
# ============================================================================

DOCUMENTATION_PATTERNS: Dict[str, PatternRule] = {
    "documentation_work": {
        "keywords": [
            "documentation", "docs", "readme", "guide",
            "tutorial", "api docs", "docstring", "comment",
            "markdown", "technical writing"
        ],
        "files": [
            "*.md", "README.md", "CONTRIBUTING.md",
            "docs/*", "documentation/*"
        ],
        "contexts": [
            "write documentation",
            "update readme",
            "create guide",
            "document api",
            "improve docs"
        ],
        "agent": "@documentation-specialist",
        "confidence": 0.85,
        "mandatory": False
    }
}


# ============================================================================
# MIGRATION PATTERNS (Confidence 0.85)
# ============================================================================

MIGRATION_PATTERNS: Dict[str, PatternRule] = {
    "migration_work": {
        "keywords": [
            "migration", "migrate", "upgrade", "port",
            "legacy code", "version upgrade", "framework migration",
            "language migration"
        ],
        "contexts": [
            "migrate code",
            "upgrade version",
            "port to new framework",
            "legacy migration",
            "framework upgrade"
        ],
        "agent": "@migration-specialist",
        "confidence": 0.85,
        "mandatory": False
    }
}


# ============================================================================
# MASTER PATTERN CATALOG
# ============================================================================

PATTERN_CATALOG: Dict[str, PatternRule] = {
    **QUALITY_GATE_PATTERNS,
    **SECURITY_PATTERNS,
    **PYTHON_PATTERNS,
    **TYPESCRIPT_PATTERNS,
    **RUST_PATTERNS,
    **GO_PATTERNS,
    **DATABASE_PATTERNS,
    **TESTING_PATTERNS,
    **PERFORMANCE_PATTERNS,
    **REFACTORING_PATTERNS,
    **API_PATTERNS,
    **DEBUGGING_PATTERNS,
    **INTEGRATION_PATTERNS,
    **DEPLOYMENT_PATTERNS,
    **DOCUMENTATION_PATTERNS,
    **MIGRATION_PATTERNS,
}


# ============================================================================
# AGENT METADATA
# ============================================================================

AGENT_METADATA: Dict[str, Dict[str, str]] = {
    "@tech-lead": {
        "role": "orchestrator",
        "description": "Coordinates multi-agent workflows, task decomposition, architecture decisions"
    },
    "@python-specialist": {
        "role": "domain",
        "description": "Python 3.11+, FastAPI, Django, async development, type safety"
    },
    "@typescript-specialist": {
        "role": "domain",
        "description": "TypeScript, React, Next.js, Node.js APIs, performance optimization"
    },
    "@rust-specialist": {
        "role": "domain",
        "description": "Rust, Cargo, Tokio, async systems, performance-critical code"
    },
    "@go-specialist": {
        "role": "domain",
        "description": "Go, goroutines, microservices, concurrent programming"
    },
    "@database-specialist": {
        "role": "domain",
        "description": "Database design, SQL optimization, migrations, ORM patterns"
    },
    "@devops-specialist": {
        "role": "domain",
        "description": "CI/CD, Docker, Kubernetes, infrastructure as code, deployment"
    },
    "@code-reviewer": {
        "role": "qa",
        "description": "Quality, security, performance validation (MANDATORY before commits)"
    },
    "@security-auditor": {
        "role": "qa",
        "description": "Security audits, vulnerability scanning, OWASP compliance"
    },
    "@testing-specialist": {
        "role": "task",
        "description": "Test strategy, test automation, coverage analysis, E2E testing"
    },
    "@performance-optimizer": {
        "role": "task",
        "description": "Performance profiling, optimization, scalability improvements"
    },
    "@api-architect": {
        "role": "task",
        "description": "API design, REST/GraphQL architecture, API documentation"
    },
    "@debugger": {
        "role": "task",
        "description": "Root cause analysis, debugging complex issues, error investigation"
    },
    "@refactoring-specialist": {
        "role": "task",
        "description": "Code refactoring, technical debt reduction, clean code practices"
    },
    "@integration-specialist": {
        "role": "task",
        "description": "Third-party integrations, microservice communication, webhooks"
    },
    "@documentation-specialist": {
        "role": "task",
        "description": "Technical documentation, API docs, user guides, tutorials"
    },
    "@migration-specialist": {
        "role": "task",
        "description": "Framework migrations, version upgrades, legacy code modernization"
    }
}


def get_agent_by_name(agent_name: str) -> Optional[Dict[str, str]]:
    """
    Get agent metadata by agent name.

    Args:
        agent_name: Agent name (e.g., '@python-specialist')

    Returns:
        Agent metadata dictionary or None if not found
    """
    return AGENT_METADATA.get(agent_name)


def get_all_agents() -> Dict[str, Dict[str, str]]:
    """
    Get all registered agents with metadata.

    Returns:
        Dictionary mapping agent names to metadata
    """
    return AGENT_METADATA.copy()


def get_agents_by_role(role: str) -> Dict[str, Dict[str, str]]:
    """
    Get all agents with a specific role.

    Args:
        role: Agent role ('orchestrator', 'domain', 'qa', 'task')

    Returns:
        Dictionary mapping agent names to metadata for specified role
    """
    return {
        name: metadata
        for name, metadata in AGENT_METADATA.items()
        if metadata["role"] == role
    }
