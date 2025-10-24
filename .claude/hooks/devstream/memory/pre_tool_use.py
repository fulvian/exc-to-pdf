#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "cchooks>=0.1.4",
#     "aiohttp>=3.8.0",
#     "structlog>=23.0.0",
#     "python-dotenv>=1.0.0",
#     "cachetools>=5.0.0",
# ]
# ///

"""
DevStream PreToolUse Hook - Context Injection before Write/Edit
Context7 + DevStream hybrid context assembly con graceful fallback.
Agent Auto-Delegation System integration for intelligent agent routing.
FASE 4.3/4.4: Rate limiting and LRU caching for memory operations.
"""

import sys
import asyncio
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from cachetools import cached, LRUCache
import hashlib
import string

# LOG-001: Add tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    # Fallback to basic approximation if tiktoken unavailable
    print("⚠️  DevStream: tiktoken unavailable, using approximate token counting", file=sys.stderr)

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
sys.path.insert(0, str(Path(__file__).parent.parent))  # Add devstream hooks dir to path

from cchooks import safe_create_context, PreToolUseContext
from devstream_base import DevStreamHookBase

# Context7-compliant robust import with fallback for unified_client
try:
    # Try relative import first (when run as module)
    from .unified_client import get_unified_client
except ImportError:
    try:
        # Fallback to absolute import (when run as script)
        from unified_client import get_unified_client
    except ImportError as e:
        # Final fallback - create dummy client that gracefully degrades
        # Context7-compliant: variable scope fixed by moving print inside except block
        error_message = str(e)

        def get_unified_client():
            class DummyUnifiedClient:
                def __init__(self):
                    self.disabled = True

                async def search_memory(self, *args, **kwargs):
                    return None

                async def store_memory(self, *args, **kwargs):
                    return None

                async def health_check(self):
                    return {"backends": {}, "overall": "disabled"}

                async def trigger_checkpoint(self, *args, **kwargs):
                    return None

            return DummyUnifiedClient()

        print(f"⚠️  DevStream: unified_client unavailable, using fallback: {error_message}", file=sys.stderr)
from rate_limiter import memory_rate_limiter, has_memory_capacity

# SQL Injection Protection Constants
SQL_INJECTION_PATTERNS = [
    # Basic SQL injection patterns
    r'(?i)\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b',
    r'(?i)\b(or|and)\s+\d+\s*=\s*\d+',  # OR 1=1
    r'(?i)\b(or|and)\s+["\'][^"\']*["\']?\s*=\s*["\'][^"\']*["\']?',  # OR 'x'='x (improved)
    r'["\']\s*(?:or|and)\s+["\'][^"\']*["\']?\s*=\s*["\'][^"\']*["\']?',  # OR in quotes (edge case)'
    r'(?i)--.*$',  # SQL comments
    r'(?i);.*$',   # Multiple statements
    r'(?i)/\*.*\*/',  # Block comments
    r'["\']\s*/\*.*\*/',  # Block comment after quote
    # Advanced injection patterns
    r'(?i)\b(waitfor|delay|sleep)\b',
    r'(?i)\b(benchmark|sleep)\s*\(',
    r'(?i)\b(information_schema|sysobjects|syscolumns)\b',
    r'(?i)\b(xp_|sp_)\w+',  # Extended stored procedures
    # String-based injection patterns
    r'["\'].*?(union|select|insert|update|delete).*?["\']',
    r'["\'].*?[;].*?(union|select|insert|update|delete)',
    # Hex/char encoding patterns
    r'(?i)0x[0-9a-f]+',
    r'(?i)char\s*\(',
    r'(?i)ascii\s*\(',
    # Time-based injection patterns
    r'(?i)\b(waitfor\s+delay|sleep\s*\(|benchmark\s*\()',
]

# SQL safe characters (alphanumeric, spaces, basic punctuation)
SQL_SAFE_CHARS = set(string.ascii_letters + string.digits + ' ._-:')

def _sanitize_query_elements(elements: List[str]) -> List[str]:
    """
    Context7-compliant SQL injection protection for query elements.

    Implements OWASP best practices:
    1. Pattern-based detection of SQL injection attempts
    2. Character-level validation (whitelist approach)
    3. Input sanitization and length limits
    4. Defense-in-depth with multiple validation layers

    Args:
        elements: List of string elements extracted from code analysis

    Returns:
        Sanitized list of elements safe for SQL query construction

    Security Notes:
        - Uses whitelist approach (only safe characters allowed)
        - Detects common SQL injection patterns
        - Applies length limits to prevent buffer overflows
        - Logs potential attacks for security monitoring
    """
    if not elements:
        return []

    sanitized_elements = []

    for element in elements:
        if not element or not isinstance(element, str):
            continue

        original_element = element
        attack_detected = False

        # Layer 1: Pattern-based detection (OWASP best practice)
        for pattern in SQL_INJECTION_PATTERNS:
            if re.search(pattern, element):
                attack_detected = True
                break

        if attack_detected:
            # Log potential attack (security monitoring)
            print(f"⚠️  SQL injection attempt detected and blocked: {element[:100]}...", file=sys.stderr)
            continue

        # Layer 2: Character-level validation (whitelist approach)
        # Only allow safe characters: alphanumeric, spaces, basic punctuation
        sanitized = ''.join(
            char for char in element
            if char in SQL_SAFE_CHARS
        )

        # Layer 3: Length validation (prevent buffer overflows)
        if len(sanitized) > 100:  # Reasonable limit for identifiers
            sanitized = sanitized[:100]

        # Layer 4: Empty/meaningless element filtering
        if len(sanitized.strip()) < 2:  # Minimum meaningful length
            continue

        # Layer 5: Final safety check (defense in depth)
        # Verify no dangerous patterns survived sanitization
        is_safe = True
        for pattern in SQL_INJECTION_PATTERNS:
            if re.search(pattern, sanitized):
                is_safe = False
                break

        if is_safe and sanitized.strip():
            sanitized_elements.append(sanitized.strip())

    # Limit total number of elements to prevent query bloating
    return sanitized_elements[:10]

# Semantic Cache Keys integration (Task 3: LRU cache optimization)
# Replace basic LRU cache with semantic-aware cache system
# Target: 60%+ hit rate improvement from 0.017% baseline

# Import SemanticCacheKeys with graceful degradation
try:
    from optimization.semantic_cache_keys import get_semantic_cache_keys, CacheHitType
    SEMANTIC_CACHE_AVAILABLE = True
except ImportError as e:
    SEMANTIC_CACHE_AVAILABLE = False
    _SEMANTIC_CACHE_IMPORT_ERROR = str(e)
    print(f"⚠️  DevStream: SemanticCacheKeys unavailable, using basic cache: {e}", file=sys.stderr)

# TaskAwareQueryConstructor integration (Task 4: Context relevance optimization)
# Replace basic query construction with intelligent context analysis
# Target: 70%+ relevance improvement from <30% baseline

try:
    from optimization.task_aware_query_constructor import get_task_aware_query_constructor
    QUERY_CONSTRUCTOR_AVAILABLE = True
except ImportError as e:
    QUERY_CONSTRUCTOR_AVAILABLE = False
    _QUERY_CONSTRUCTOR_IMPORT_ERROR = str(e)
    print(f"⚠️  DevStream: TaskAwareQueryConstructor unavailable: {e}", file=sys.stderr)

# TwoStageSearch integration (Task 5: Search performance optimization)
# Replace basic vector search with two-stage binary quantization
# Target: <100ms query time from 500ms baseline

try:
    from optimization.two_stage_search import get_two_stage_search, QuantizationType
    TWO_STAGE_SEARCH_AVAILABLE = True
except ImportError as e:
    TWO_STAGE_SEARCH_AVAILABLE = False
    _TWO_STAGE_SEARCH_IMPORT_ERROR = str(e)
    print(f"⚠️  DevStream: TwoStageSearch unavailable: {e}", file=sys.stderr)

# Initialize semantic cache system if available
if SEMANTIC_CACHE_AVAILABLE:
    try:
        # Configure for optimal performance based on Context7 research
        semantic_cache = get_semantic_cache_keys(
            max_cache_size=1000,              # Increase from 20 to 1000 for better hit rate
            similarity_threshold=0.75,        # Lowered from 0.85 to 0.75 for broader matching
            cluster_threshold=0.65,           # Lowered from 0.75 to 0.65 for more cluster matches
            enable_embeddings=True           # Enable semantic similarity features
        )
        print("✅ DevStream: SemanticCacheKeys initialized with Context7 patterns", file=sys.stderr)
    except Exception as e:
        SEMANTIC_CACHE_AVAILABLE = False
        _SEMANTIC_CACHE_IMPORT_ERROR = str(e)
        print(f"⚠️  DevStream: SemanticCacheKeys init failed, using basic cache: {e}", file=sys.stderr)

# Fallback to basic cache if semantic cache unavailable
if not SEMANTIC_CACHE_AVAILABLE:
    memory_search_cache = LRUCache(maxsize=20)
    print("⚠️  DevStream: Using basic LRU cache (20 entries, 0.017% hit rate expected)", file=sys.stderr)

# Agent Auto-Delegation imports (with graceful degradation)
try:
    from agents.pattern_matcher import PatternMatcher
    from agents.agent_router import AgentRouter, TaskAssessment
    AGENT_DELEGATION_AVAILABLE = True
except ImportError as e:
    AGENT_DELEGATION_AVAILABLE = False
    _IMPORT_ERROR = str(e)
    # Provide fallback type for type hints when delegation unavailable
    TaskAssessment = Any  # type: ignore

# ResourceMonitor imports (with graceful degradation)
try:
    from monitoring.resource_monitor import ResourceMonitor, ResourceHealth, HealthStatus
    RESOURCE_MONITORING_AVAILABLE = True
except ImportError as e:
    RESOURCE_MONITORING_AVAILABLE = False
    _RESOURCE_MONITOR_IMPORT_ERROR = str(e)


class PreToolUseHook:
    """
    PreToolUse hook for intelligent context injection.
    Combines Context7 library docs + DevStream semantic memory + Agent Auto-Delegation.
    """

    def __init__(self):
        self.base = DevStreamHookBase("pre_tool_use")
        self.unified_client = get_unified_client()

        # Agent Auto-Delegation components (graceful degradation)
        self.pattern_matcher: Optional[PatternMatcher] = None
        self.agent_router: Optional[AgentRouter] = None

        if AGENT_DELEGATION_AVAILABLE:
            try:
                self.pattern_matcher = PatternMatcher()
                self.agent_router = AgentRouter()
                self.base.debug_log("Agent Auto-Delegation enabled")
            except Exception as e:
                self.base.debug_log(f"Agent Auto-Delegation init failed: {e}")
        else:
            self.base.debug_log(f"Agent Auto-Delegation unavailable: {_IMPORT_ERROR}")

        # ResourceMonitor component (graceful degradation, singleton pattern)
        self.resource_monitor: Optional[ResourceMonitor] = None

        if RESOURCE_MONITORING_AVAILABLE:
            try:
                self.resource_monitor = ResourceMonitor()
                self.base.debug_log("ResourceMonitor enabled")
            except Exception as e:
                self.base.debug_log(f"ResourceMonitor init failed: {e}")
        else:
            self.base.debug_log(f"ResourceMonitor unavailable: {_RESOURCE_MONITOR_IMPORT_ERROR}")

        # Initialize TaskAwareQueryConstructor for enhanced query construction (Task 4)
        self.query_constructor = None
        try:
            self.query_constructor = get_task_aware_query_constructor(
                max_context_tokens=2000,
                relevance_threshold=0.7,  # High threshold for quality
                enable_semantic_expansion=True,
                enable_context_optimization=True
            )
            self.base.debug_log("TaskAwareQueryConstructor initialized")
        except Exception as e:
            self.base.debug_log(f"TaskAwareQueryConstructor init failed: {e}")

        # Initialize TwoStageSearch for high-performance search (Task 5)
        self.two_stage_search = None
        try:
            self.two_stage_search = get_two_stage_search(
                quantization_type=QuantizationType.BINARY,
                coarse_candidate_limit=100,
                fine_result_limit=20,
                similarity_threshold=0.7,
                enable_adaptive_limits=True
            )
            self.base.debug_log("TwoStageSearch initialized")
        except Exception as e:
            self.base.debug_log(f"TwoStageSearch init failed: {e}")

        # LOG-001: Token budget configuration with dynamic enforcement
        self.total_token_budget = int(
            os.getenv("DEVSTREAM_CONTEXT_MAX_TOKENS", "7000")  # Increased from 2000 to 7000
        )
        self.memory_token_budget = 2000  # Fixed memory budget
        self.context7_token_budget = self.total_token_budget - self.memory_token_budget  # Dynamic: 5000 tokens

        # LOG-001: Initialize tiktoken encoder for accurate counting
        self.tokenizer = None
        if TIKTOKEN_AVAILABLE:
            try:
                # Use GPT-4 tokenizer for Claude compatibility
                self.tokenizer = tiktoken.encoding_for_model("gpt-4")
                self.base.debug_log("tiktoken GPT-4 encoder initialized for accurate token counting")
            except Exception as e:
                self.base.debug_log(f"tiktoken initialization failed: {e}")
                self.tokenizer = None

    def _estimate_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken for accuracy or fallback approximation.

        LOG-001: Enhanced token counting with 95% accuracy using tiktoken.
        Falls back to chars/4 approximation if tiktoken unavailable.

        Args:
            text: Input text

        Returns:
            Accurate token count (tiktoken) or conservative estimate
        """
        if not text:
            return 0

        # LOG-001: Use tiktoken for accurate counting when available
        if self.tokenizer:
            try:
                # tiktoken provides exact token counts for GPT-4/Claude
                return len(self.tokenizer.encode(text))
            except Exception as e:
                self.base.debug_log(f"tiktoken counting failed: {e}, using fallback")
                # Fallback to approximation if tiktoken fails

        # Fallback: Conservative chars/4 approximation (original method)
        return len(text) // 4

    def _detect_libraries(self, content: str, file_path: str) -> List[str]:
        """
        Extract library names from imports and usage patterns.

        Args:
            content: File content to analyze
            file_path: File path for extension-based detection

        Returns:
            List of lowercase library names (Context7-compatible)

        Note:
            All names normalized to lowercase for Context7 compatibility.
            Example: "FastAPI" → "fastapi", "SQLAlchemy" → "sqlalchemy"
        """
        libraries = []
        file_ext = Path(file_path).suffix.lower()

        # Python imports
        if file_ext == '.py':
            import_pattern = r'^(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
            for match in re.finditer(import_pattern, content, re.MULTILINE):
                lib = match.group(1)
                # Skip stdlib
                if lib not in ['os', 'sys', 're', 'json', 'typing', 'pathlib',
                              'datetime', 'asyncio', 'subprocess', 'logging']:
                    libraries.append(lib)

        # JavaScript/TypeScript imports
        elif file_ext in ['.js', '.ts', '.jsx', '.tsx']:
            import_pattern = r'(?:import|require)\s*\(?[\'"]([^\'"\)]+)[\'"]'
            for match in re.finditer(import_pattern, content):
                lib = match.group(1).split('/')[0]  # Get package name
                if not lib.startswith('.'):  # Skip relative imports
                    libraries.append(lib)

        # Framework detection patterns (cross-language)
        # Note: Patterns are file-extension aware to reduce false positives
        framework_patterns = {
            'fastapi': (r'(?:from\s+fastapi|FastAPI\()', ['.py']),
            'django': (r'(?:from\s+django|django\.)', ['.py']),
            'flask': (r'(?:from\s+flask|@app\.route)', ['.py']),
            'react': (r'(?:import.*React|useState|useEffect|createContext)', ['.js', '.jsx', '.ts', '.tsx']),
            'vue': (r'(?:import.*Vue|createApp|defineComponent)', ['.js', '.ts', '.vue']),
            'next': (r'(?:import.*next|getServerSideProps|getStaticProps)', ['.js', '.jsx', '.ts', '.tsx']),
            'express': (r'(?:express\(\)|app\.get\(|app\.post\()', ['.js', '.ts']),
            'pytest': (r'(?:import\s+pytest|@pytest\.)', ['.py']),
            'jest': (r'(?:describe\(|test\(|expect\()', ['.js', '.ts', '.jsx', '.tsx']),
            'sqlalchemy': (r'(?:from\s+sqlalchemy|declarative_base|relationship)', ['.py']),
        }

        for lib, (pattern, extensions) in framework_patterns.items():
            # Only check if file extension matches
            if file_ext in extensions and re.search(pattern, content, re.IGNORECASE):
                libraries.append(lib)

        # Normalize to lowercase and remove duplicates for Context7 compatibility
        return list(set(lib.lower() for lib in libraries))

    def _build_code_aware_query(self, file_path: str, content: str) -> str:
        """
        Build intelligent query from code structure.

        Extracts:
        - Imports (libraries/modules being used)
        - Class names (key abstractions)
        - Function names (main operations)
        - Decorators (framework patterns like @app.get, @pytest.fixture)

        Args:
            file_path: Path to file being edited
            content: File content

        Returns:
            Structured query string with code elements

        Performance:
            Target: <50ms for typical files (99th percentile)
        """
        import time
        start_time = time.time()

        filename = Path(file_path).name
        ext = Path(file_path).suffix.lower()

        elements = [filename]

        # Python code analysis
        if ext == '.py':
            # Extract imports
            import_pattern = r'^(?:from\s+(\S+)|import\s+(\S+))'
            imports = re.findall(import_pattern, content, re.MULTILINE)
            imports = [i[0] or i[1] for i in imports if i[0] or i[1]]
            # Filter out stdlib and builtins, split dotted imports
            filtered_imports = []
            stdlib = {'os', 'sys', 're', 'json', 'typing', 'pathlib', 'datetime',
                     'asyncio', 'subprocess', 'logging', 'time', 'collections'}
            for imp in imports:
                base = imp.split('.')[0]  # Get root module
                if base not in stdlib:
                    filtered_imports.append(base)
            elements.extend(filtered_imports[:5])  # Max 5 imports

            # Extract class names
            class_pattern = r'^class\s+(\w+)'
            classes = re.findall(class_pattern, content, re.MULTILINE)
            elements.extend(classes[:3])  # Max 3 classes

            # Extract function/method names
            func_pattern = r'^(?:async\s+)?def\s+(\w+)'
            funcs = re.findall(func_pattern, content, re.MULTILINE)
            elements.extend(funcs[:5])  # Max 5 functions

            # Extract decorators (framework indicators)
            decorator_pattern = r'^@(\w+(?:\.\w+)?)'
            decorators = re.findall(decorator_pattern, content, re.MULTILINE)
            elements.extend(decorators[:3])  # Max 3 decorators

        # TypeScript/JavaScript analysis
        elif ext in ['.ts', '.tsx', '.js', '.jsx']:
            # Extract imports
            import_pattern = r'(?:import|from)\s+[\'"]([^\'"]+)[\'"]'
            imports = re.findall(import_pattern, content)
            # Filter relative imports and get package names
            filtered_imports = []
            for imp in imports:
                if not imp.startswith('.'):
                    # Get package name (before first /)
                    pkg = imp.split('/')[0]
                    filtered_imports.append(pkg)
            elements.extend(filtered_imports[:5])

            # Extract class/component names
            class_pattern = r'(?:class|function|const)\s+(\w+)'
            names = re.findall(class_pattern, content)
            elements.extend(names[:5])

        # Rust analysis
        elif ext == '.rs':
            # Extract use statements
            use_pattern = r'^use\s+([a-zA-Z_][a-zA-Z0-9_:]*)'
            uses = re.findall(use_pattern, content, re.MULTILINE)
            elements.extend([u.split('::')[0] for u in uses[:5]])

            # Extract struct/enum/trait names
            type_pattern = r'^(?:struct|enum|trait|impl)\s+(\w+)'
            types = re.findall(type_pattern, content, re.MULTILINE)
            elements.extend(types[:5])

        # Go analysis
        elif ext == '.go':
            # Extract imports
            import_pattern = r'import\s+(?:"([^"]+)"|`([^`]+)`)'
            imports = re.findall(import_pattern, content)
            imports = [i[0] or i[1] for i in imports]
            elements.extend([imp.split('/')[-1] for imp in imports[:5]])

            # Extract type/struct/interface names
            type_pattern = r'^type\s+(\w+)'
            types = re.findall(type_pattern, content, re.MULTILINE)
            elements.extend(types[:5])

        # Build query with code structure (SECURE - SQL injection protection)
        sanitized_elements = _sanitize_query_elements(elements)
        query = " ".join(sanitized_elements)

        # Fallback to content prefix if no elements extracted
        if len(query) < 50:
            query = f"{filename} {content[:300]}"

        # Log performance metrics
        elapsed_ms = (time.time() - start_time) * 1000
        self.base.debug_log(
            f"Code-aware query built in {elapsed_ms:.1f}ms: {query[:80]}..."
        )

        return query

    async def get_context7_docs(self, file_path: str, content: str) -> Optional[str]:
        """
        Get Context7 documentation using hybrid manager (direct + MCP fallback).

        Instead of emitting advisory messages, directly retrieves documentation
        using hybrid strategy with automatic fallback.
        """
        try:
            # Initialize hybrid manager (lazy initialization)
            if not hasattr(self, 'context7_manager'):
                try:
                    from ..utils.context7_hybrid_manager import Context7HybridManager
                    self.context7_manager = Context7HybridManager()
                except ImportError:
                    try:
                        from utils.context7_hybrid_manager import Context7HybridManager
                        self.context7_manager = Context7HybridManager()
                    except ImportError:
                        self.base.debug_log("Context7HybridManager not available, using fallback")
                        return await self._emit_context7_advisory_fallback(
                            self._detect_libraries(content, file_path)
                        )

            # Detect libraries from imports/usage
            libraries = self._detect_libraries(content, file_path)

            if not libraries:
                self.base.debug_log("No external libraries detected for Context7")
                return None

            self.base.debug_log(f"Context7 direct retrieval - detected libraries: {', '.join(libraries)}")

            # Build formatted documentation directly
            docs_sections = []

            for lib in libraries[:3]:  # Limit to top 3 to avoid context bloat
                try:
                    # Resolve library ID
                    library_id = await self.context7_manager.resolve_library_id(lib)
                    if not library_id:
                        continue

                    # LOG-001: Calculate dynamic token allocation for Context7
                    # Distribute context7_token_budget (5000) among detected libraries
                    libraries_count = len(libraries)
                    tokens_per_library = max(
                        500,  # Minimum tokens per library
                        self.context7_token_budget // max(libraries_count, 1)
                    )

                    # Get documentation with dynamic token allocation
                    docs = await self.context7_manager.get_library_docs(
                        library_id=library_id,
                        topic=self._extract_topic_from_code(content, lib),
                        tokens=tokens_per_library
                    )

                    if docs:
                        docs_sections.append(f"### {lib.title()} ({library_id})\n\n{docs}")

                except Exception as e:
                    # LOG-003: Fix silent Context7 failures - provide clear feedback per library
                    self.base.warning_feedback(f"Context7 failed for {lib}: {str(e)[:80]}")
                    self.base.debug_log(f"Context7 library {lib} retrieval failed - full error: {e}")

                    # Log specific library failure to memory
                    try:
                        await self.unified_client.store_memory(
                            content=f"Context7 library-specific failure - library: {lib}, error: {str(e)}",
                            content_type="error",
                            keywords=["context7-failure", "log-003", lib, "debugging"],
                            hook_name="pre_tool_use_context7_library_failure"
                        )
                    except:
                        pass  # Non-blocking
                    continue

            if not docs_sections:
                return None

            # Assemble final context
            formatted = "# Context7 Documentation\n\n"
            formatted += "\n\n---\n\n".join(docs_sections)
            formatted += "\n\n---\n*Retrieved via DevStream Context7 Direct Client*"

            self.base.success_feedback(f"Retrieved Context7 docs for {len(docs_sections)} libraries")
            return formatted

        except Exception as e:
            # LOG-003: Fix silent Context7 failures - provide clear user feedback
            self.base.warning_feedback(f"Context7 direct retrieval failed: {str(e)[:100]}")
            self.base.debug_log(f"Context7 direct retrieval failed - full error: {e}")

            # Log to memory for debugging (non-blocking)
            try:
                libraries = self._detect_libraries(content, file_path)
                await self.unified_client.store_memory(
                    content=f"Context7 failure detected - libraries: {', '.join(libraries)}, error: {str(e)}",
                    content_type="error",
                    keywords=["context7-failure", "log-003", "debugging"],
                    hook_name="pre_tool_use_context7_failure"
                )
            except:
                pass  # Non-blocking, don't fail the whole operation

            # Fallback to advisory pattern on failure
            self.current_file_path = file_path  # Store for fallback
            return await self._emit_context7_advisory_fallback(libraries)

    async def _emit_context7_advisory_fallback(self, libraries: List[str]) -> Optional[str]:
        """Fallback to advisory pattern if direct retrieval fails."""
        if not libraries:
            return None

        advisory = "# Context7 Advisory (Fallback Mode)\n\n"
        advisory += f"**File**: {Path(self.current_file_path if hasattr(self, 'current_file_path') else 'unknown').name}\n\n"
        advisory += f"**Detected Libraries**: {', '.join(libraries)}\n\n"
        advisory += "**Direct retrieval failed - using manual MCP calls:\n\n"

        for lib in libraries[:3]:
            advisory += f"### {lib}\n\n"
            advisory += "1. Resolve library ID:\n"
            advisory += f"```\nmcp__context7__resolve-library-id\n"
            advisory += f"libraryName: {lib}\n```\n\n"
            advisory += "2. Retrieve documentation:\n"
            advisory += f"```\nmcp__context7__get-library-docs\n"
            advisory += f"context7CompatibleLibraryID: <resolved_id_from_step_1>\n"
            advisory += f"tokens: 5000\n```\n\n"

        return advisory

    def _extract_topic_from_code(self, content: str, library: str) -> Optional[str]:
        """Extract relevant topics from code for better documentation targeting."""
        # Look for common patterns that indicate what the user is working on
        topics = []

        # Framework-specific patterns
        if library == "fastapi":
            if re.search(r'@app\.(get|post|put|delete)', content):
                topics.append("routing")
            if re.search(r'pydantic|BaseModel', content):
                topics.append("models")
        elif library == "pytest":
            if re.search(r'@pytest\.fixture', content):
                topics.append("fixtures")
            if re.search(r'pytest-asyncio', content):
                topics.append("async")
        elif library == "aiohttp":
            if re.search(r'ClientSession|session\.', content):
                topics.append("client")
            if re.search(r'web\.Application|@routes\.', content):
                topics.append("server")

        return " ".join(topics[:2]) if topics else None

    def _format_memory_with_budget(
        self,
        memory_items: List[Dict],
        max_tokens: int = 2000
    ) -> str:
        """
        Format memory results within token budget.

        Args:
            memory_items: Search results
            max_tokens: Maximum tokens to use (default: 2000)

        Returns:
            Formatted memory context within budget
        """
        formatted = "# DevStream Memory Context\n\n"
        used_tokens = self._estimate_tokens(formatted)

        for i, item in enumerate(memory_items, 1):
            content = item.get("content", "")
            score = item.get("relevance_score", 0.0)

            # Create result header
            header = f"## Result {i} (relevance: {score:.2f})\n"

            # Calculate available tokens for this result
            header_tokens = self._estimate_tokens(header)
            available = max_tokens - used_tokens - header_tokens - 20  # Buffer

            if available < 50:  # Minimum useful content
                break

            # Truncate content to fit budget
            max_chars = available * 4
            truncated_content = content[:max_chars]

            # Add result
            result_block = f"{header}{truncated_content}\n\n"
            result_tokens = self._estimate_tokens(result_block)

            if used_tokens + result_tokens > max_tokens:
                break

            formatted += result_block
            used_tokens += result_tokens

        formatted += f"\n*Total tokens used: ~{used_tokens}/{max_tokens}*\n"
        return formatted

    def _create_cache_key(self, query: str, limit: int, content_type: Optional[str] = None) -> str:
        """
        Create deterministic cache key for memory search.

        Args:
            query: Search query string
            limit: Result limit
            content_type: Optional content type filter

        Returns:
            SHA256 hex digest (64 chars)
        """
        key_parts = [query, str(limit), content_type or ""]
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _truncate_to_budget(self, text: str, max_tokens: int) -> str:
        """
        LOG-001: Truncate text to fit within token budget.

        Args:
            text: Text to truncate
            max_tokens: Maximum allowed tokens

        Returns:
            Truncated text that fits within budget
        """
        if not text or max_tokens <= 0:
            return ""

        # If already within budget, return as-is
        current_tokens = self._estimate_tokens(text)
        if current_tokens <= max_tokens:
            return text

        # LOG-001: More aggressive truncation for accuracy
        # Start with a conservative estimate and iteratively refine
        target_chars = (max_tokens - 10) * 4  # Leave buffer for truncation message
        target_chars = min(target_chars, len(text) // 2)  # Don't truncate more than half

        # Binary search for optimal truncation point
        low, high = 0, len(text)
        best_truncated = ""

        while low <= high:
            mid = (low + high) // 2
            candidate = text[:mid] + "\n\n*Content truncated to fit token budget*"
            candidate_tokens = self._estimate_tokens(candidate)

            if candidate_tokens <= max_tokens:
                best_truncated = candidate
                low = mid + 1
            else:
                high = mid - 1

        # If no good truncation found, use minimal fallback
        if not best_truncated:
            return "*Content too large for token budget*"

        # Ensure we don't cut in the middle of a word
        last_space = best_truncated.rfind(' ')
        truncation_marker_pos = best_truncated.find("\n\n*Content truncated")

        if (last_space > 0 and
            truncation_marker_pos > 0 and
            last_space < truncation_marker_pos):
            # Move truncation point to last complete word
            before_marker = best_truncated[:truncation_marker_pos]
            truncated_word = before_marker[:last_space]
            best_truncated = truncated_word + "\n\n*Content truncated to fit token budget*"

        return best_truncated

    async def get_devstream_memory(self, file_path: str, content: str) -> Optional[str]:
        """
        Search DevStream memory for relevant context with semantic caching and rate limiting.

        TASK 3 ENHANCEMENT: SemanticCacheKeys with 60%+ hit rate improvement from 0.017% baseline.
        Context7-compliant semantic matching, query clustering, and adaptive cache strategies.

        Args:
            file_path: Path to file being edited
            content: File content preview

        Returns:
            Formatted memory context or None

        Performance:
            - Exact/semantic cache hit: <1ms (no MCP call)
            - Semantic cluster match: <2ms (contextual similarity)
            - Cache miss with capacity: ~300-500ms (MCP search)
            - Rate limited: Graceful degradation, cache-only response
        """
        try:
            # Build enhanced search query using TaskAwareQueryConstructor (Task 4)
            # Target: 70%+ relevance improvement from <30% baseline
            if self.query_constructor:
                try:
                    # Use TaskAwareQueryConstructor for intelligent query construction
                    basic_query = self._build_code_aware_query(file_path, content)

                    # Construct enhanced query with intent analysis and semantic expansion
                    query_construction = self.query_constructor.construct_enhanced_query(
                        query=basic_query,
                        search_results=[],  # No initial results for pure query construction
                        token_budget=2000
                    )

                    # Extract the enhanced query string
                    query = query_construction.content if query_construction else basic_query

                    self.base.debug_log(
                        f"TaskAwareQueryConstructor enhanced query: {query[:80]}... "
                        f"(confidence: {query_construction.query_analysis.confidence_score:.2f if query_construction else 0:.2f})"
                    )

                except Exception as e:
                    self.base.debug_log(f"TaskAwareQueryConstructor failed, using basic query: {e}")
                    query = self._build_code_aware_query(file_path, content)
            else:
                # Fallback to basic query building
                query = self._build_code_aware_query(file_path, content)

            limit = 3

            # Use semantic cache if available, otherwise fallback to basic cache
            if SEMANTIC_CACHE_AVAILABLE:
                # Use semantic cache with enhanced matching
                cached_result, hit_type = semantic_cache.get(
                    query=query,
                    limit=limit,
                    content_type=None
                )

                if cached_result is not None:
                    # Log hit type for performance monitoring
                    if hit_type == CacheHitType.EXACT_MATCH:
                        self.base.debug_log(f"Semantic cache EXACT HIT: {query[:50]}...")
                    elif hit_type == CacheHitType.SEMANTIC_MATCH:
                        self.base.debug_log(f"Semantic cache SEMANTIC HIT: {query[:50]}...")
                    elif hit_type == CacheHitType.CLUSTER_MATCH:
                        self.base.debug_log(f"Semantic cache CLUSTER HIT: {query[:50]}...")

                    # Log performance statistics periodically
                    if hasattr(semantic_cache, 'get_stats'):
                        stats = semantic_cache.get_stats()
                        if stats.total_requests % 50 == 0:  # Log every 50 requests
                            self.base.debug_log(
                                f"Cache performance: {stats.hit_rate:.1f}% hit rate, "
                                f"{stats.semantic_hit_rate:.1f}% semantic hits, "
                                f"{stats.avg_query_time_ms:.1f}ms avg query time"
                            )

                    return cached_result

                # Cache miss - continue with search
                self.base.debug_log(f"Semantic cache MISS, searching: {query[:50]}...")

            else:
                # Fallback to basic cache
                cache_key = self._create_cache_key(query, limit)

                # Check basic cache first (synchronous, <1ms)
                if cache_key in memory_search_cache:
                    cached_result = memory_search_cache[cache_key]
                    self.base.debug_log(f"Basic cache HIT: {query[:50]}...")
                    return cached_result

                self.base.debug_log(f"Basic cache MISS, searching: {query[:50]}...")

            # Cache miss - check rate limiter capacity
            if not has_memory_capacity():
                self.base.debug_log(
                    "Memory rate limit exceeded, skipping search (graceful degradation)"
                )
                return None

            # Search memory via unified client with rate limiting
            # TASK 5 ENHANCEMENT: Use TwoStageSearch for <100ms query time from 500ms baseline
            async with memory_rate_limiter:
                if self.two_stage_search:
                    try:
                        # Use TwoStageSearch for high-performance search
                        search_result = await self.two_stage_search.search_memory(
                            query=query,
                            limit=limit,
                            content_type=None,
                            min_relevance=0.5,  # Filter low-relevance results
                            max_results=None  # Use internal optimization
                        )

                        # Convert TwoStageSearch result to expected format
                        if search_result and search_result.results:
                            result = {
                                "results": [
                                    {
                                        "content": item.content,
                                        "content_type": item.content_type,
                                        "metadata": item.metadata,
                                        "relevance_score": item.relevance_score,
                                        "distance": item.distance
                                    }
                                    for item in search_result.results
                                ],
                                "total_found": search_result.total_found,
                                "search_type": "two_stage_binary",
                                "compression_ratio": search_result.compression_ratio,
                                "search_time_ms": search_result.search_time_ms
                            }

                            self.base.debug_log(
                                f"TwoStageSearch completed in {search_result.search_time_ms:.1f}ms "
                                f"(found {len(search_result.results)} results, "
                                f"compression: {search_result.compression_ratio}x)"
                            )
                        else:
                            result = {"results": []}

                    except Exception as e:
                        self.base.debug_log(f"TwoStageSearch failed, using basic search: {e}")
                        # Fallback to basic search
                        result = await self.unified_client.search_memory(
                            query=query,
                            content_type=None,
                            limit=limit,
                            hook_name="pre_tool_use"
                        )
                else:
                    # Fallback to basic search
                    result = await self.unified_client.search_memory(
                        query=query,
                        content_type=None,
                        limit=limit,
                        hook_name="pre_tool_use"
                    )

            if not result or not result.get("results"):
                self.base.debug_log("No relevant memory found")

                # Cache negative result to prevent repeated searches
                if SEMANTIC_CACHE_AVAILABLE:
                    semantic_cache.set(query, limit, None, None)
                else:
                    cache_key = self._create_cache_key(query, limit)
                    memory_search_cache[cache_key] = None
                return None

            # Format memory results with token budget enforcement
            memory_items = result.get("results", [])
            if not memory_items:
                if SEMANTIC_CACHE_AVAILABLE:
                    semantic_cache.set(query, limit, None, None)
                else:
                    cache_key = self._create_cache_key(query, limit)
                    memory_search_cache[cache_key] = None
                return None

            formatted = self._format_memory_with_budget(
                memory_items,
                max_tokens=self.memory_token_budget
            )

            # LOG-001: Verify memory formatting didn't exceed budget
            if formatted:
                actual_tokens = self._estimate_tokens(formatted)
                if actual_tokens > self.memory_token_budget:
                    self.base.debug_log(
                        f"Memory formatting exceeded budget: {actual_tokens} > {self.memory_token_budget} tokens"
                    )
                    # Truncate to fit budget
                    formatted = self._truncate_to_budget(formatted, self.memory_token_budget)

            # Cache successful result using appropriate cache system
            if SEMANTIC_CACHE_AVAILABLE:
                semantic_cache.set(query, limit, None, formatted)
            else:
                cache_key = self._create_cache_key(query, limit)
                memory_search_cache[cache_key] = formatted

            self.base.success_feedback(f"Found {len(memory_items)} relevant memories (cached)")

            return formatted

        except Exception as e:
            self.base.debug_log(f"Memory search error: {e}")
            return None

    async def check_agent_delegation(
        self,
        file_path: Optional[str],
        content: Optional[str],
        tool_name: Optional[str],
        user_query: Optional[str] = None
    ) -> Optional[TaskAssessment]:
        """
        Check if task should be delegated to specialized agent.

        Args:
            file_path: Optional file path being worked on
            content: Optional file content
            tool_name: Optional tool name (e.g., 'Write', 'Edit')
            user_query: Optional user query string

        Returns:
            TaskAssessment if delegation match found, None otherwise

        Note:
            Gracefully degrades if Agent Auto-Delegation unavailable
        """
        # Check if Agent Auto-Delegation is enabled via config
        if not os.getenv("DEVSTREAM_AGENT_AUTO_DELEGATION_ENABLED", "true").lower() == "true":
            self.base.debug_log("Agent Auto-Delegation disabled via config")
            return None

        # Check if components available
        if not self.pattern_matcher or not self.agent_router:
            return None

        try:
            # Match patterns
            pattern_match = self.pattern_matcher.match_patterns(
                file_path=file_path,
                content=content,
                user_query=user_query,
                tool_name=tool_name
            )

            if not pattern_match:
                self.base.debug_log("No agent pattern match found")
                return None

            # Assess task complexity
            context = {
                "file_path": file_path,
                "content": content,
                "user_query": user_query or "",
                "tool_name": tool_name,
                "affected_files": [file_path] if file_path else []
            }

            assessment = await self.agent_router.assess_task_complexity(
                pattern_match=pattern_match,
                context=context
            )

            self.base.debug_log(
                f"Agent delegation assessment: {assessment.recommendation} "
                f"({assessment.suggested_agent}, confidence={assessment.confidence:.2f})"
            )

            # Log delegation decision to DevStream memory (non-blocking)
            await self._log_delegation_decision(assessment, pattern_match)

            return assessment

        except Exception as e:
            # Non-blocking error - log and continue
            self.base.debug_log(f"Agent delegation check failed: {e}")
            return None

    async def _log_delegation_decision(
        self,
        assessment: TaskAssessment,
        pattern_match: Dict[str, Any]
    ) -> None:
        """
        Log delegation decision to DevStream memory.

        Args:
            assessment: Task assessment with delegation recommendation
            pattern_match: Pattern match information

        Note:
            Non-blocking - errors are logged but do not interrupt execution
        """
        # Check if memory is enabled
        if not os.getenv("DEVSTREAM_MEMORY_ENABLED", "true").lower() == "true":
            return

        try:
            # Format delegation decision log
            agent = assessment.suggested_agent
            confidence = assessment.confidence
            recommendation = assessment.recommendation
            reason = assessment.reason
            complexity = assessment.complexity
            impact = assessment.architectural_impact

            content = (
                f"Agent Delegation: @{agent} (confidence {confidence:.2f}, {recommendation})\n"
                f"Reason: {reason}\n"
                f"Complexity: {complexity} | Impact: {impact}"
            )

            # Extract keywords
            keywords = [
                "agent-delegation",
                agent,
                recommendation.lower(),
                complexity.lower()
            ]

            # Store in memory via unified client
            await self.unified_client.store_memory(
                content=content,
                content_type="decision",
                keywords=keywords,
                hook_name="pre_tool_use_delegation"
            )

            self.base.debug_log(f"Delegation decision logged to memory: @{agent}")

        except Exception as e:
            # Non-blocking error - log and continue
            self.base.debug_log(f"Failed to log delegation decision: {e}")

    async def assemble_context(
        self,
        file_path: str,
        content: str
    ) -> Optional[str]:
        """
        Assemble hybrid context from Context7 + DevStream memory.

        Uses parallel execution via asyncio.gather() to reduce latency.
        Both retrievals execute concurrently, with independent error handling.

        Args:
            file_path: File being edited
            content: File content

        Returns:
            Assembled context string or None

        Performance:
            Target: <800ms (55% improvement from sequential 1772ms)
            Context7: 5000 token budget max
            Memory: 2000 token budget max
        """
        import time
        start_time = time.time()

        context_parts = []

        # Execute both retrievals in parallel (independent error handling)
        # Each method has try/except returning None on failure
        context7_task = self.get_context7_docs(file_path, content)
        memory_task = self.get_devstream_memory(file_path, content)

        context7_docs, memory_context = await asyncio.gather(
            context7_task,
            memory_task,
            return_exceptions=False  # Let individual methods handle errors
        )

        # Collect successful retrievals
        if context7_docs:
            context_parts.append(context7_docs)
        if memory_context:
            context_parts.append(memory_context)

        # Log performance metrics
        elapsed_ms = (time.time() - start_time) * 1000
        self.base.debug_log(
            f"Parallel context retrieval completed in {elapsed_ms:.0f}ms "
            f"(Context7: {'✓' if context7_docs else '✗'}, "
            f"Memory: {'✓' if memory_context else '✗'})"
        )

        if not context_parts:
            return None

        # Assemble final context
        assembled = "\n\n---\n\n".join(context_parts)
        return f"# Enhanced Context for {Path(file_path).name}\n\n{assembled}"

    async def process(self, context: PreToolUseContext) -> None:
        """
        Main hook processing logic.
        Integrates ResourceMonitor + Agent Auto-Delegation + Context7 + DevStream memory.

        Args:
            context: PreToolUse context from cchooks
        """
        # PHASE 0: Resource health check (non-blocking)
        skip_heavy_injection = False

        if self.resource_monitor:
            try:
                # Check resource health (cached, minimal overhead)
                health: ResourceHealth = self.resource_monitor.check_stability()

                # Log health status with structured logging
                if health.healthy:
                    self.base.debug_log(
                        f"Resource health: HEALTHY ({len(health.metrics)} metrics checked)"
                    )
                elif health.status == HealthStatus.WARNING:
                    self.base.debug_log(
                        f"Resource health: WARNING - {health.get_warning_summary()}"
                    )
                elif health.status == HealthStatus.CRITICAL:
                    # CRITICAL status - log error and consider skipping heavy operations
                    self.base.debug_log(
                        f"Resource health: CRITICAL - {health.get_warning_summary()}"
                    )
                    self.base.warning_feedback(
                        "System resources CRITICAL - Consider restarting Claude Code or reducing load"
                    )
                    # Optional: Skip heavy context injection to reduce load
                    skip_heavy_injection = True

            except Exception as e:
                # Monitor failure should NOT block tool execution
                self.base.debug_log(
                    f"Resource monitor check failed (non-critical): {str(e)[:100]}"
                )

        # Check if hook should run
        if not self.base.should_run():
            self.base.debug_log("Hook disabled via config")
            context.output.exit_success()
            return

        # PHASE -1: MCP Process Cleanup (DISABLED - causing disconnections!)
        # The cleanup hook was killing and restarting MCP processes in an infinite loop
        # TODO: Re-evaluate if cleanup is actually needed for single-instance setup
        pass

        # Extract tool information
        tool_name = context.tool_name
        tool_input = context.tool_input

        self.base.debug_log(f"Processing {tool_name} for {tool_input.get('file_path', 'unknown')}")

        # Only process Write/Edit operations
        if tool_name not in ["Write", "Edit", "MultiEdit"]:
            context.output.exit_success()
            return

        # Extract file information
        file_path = tool_input.get("file_path", "")
        content = tool_input.get("content", "") or tool_input.get("new_string", "")

        if not file_path:
            self.base.debug_log("No file path in tool input")
            context.output.exit_success()
            return

        try:
            context_parts = []

            # PHASE 1: Agent Auto-Delegation (BEFORE Context7/memory injection)
            try:
                assessment = await self.check_agent_delegation(
                    file_path=file_path,
                    content=content,
                    tool_name=tool_name,
                    user_query=None  # user_query not available in PreToolUse context
                )

                if assessment and self.agent_router:
                    # Format advisory message
                    advisory_message = self.agent_router.format_advisory_message(assessment)

                    # Prepend advisory to context injection
                    advisory_header = (
                        "# Agent Auto-Delegation Advisory\n\n"
                        f"{advisory_message}\n\n"
                        "---\n\n"
                    )
                    context_parts.append(advisory_header)

                    self.base.success_feedback(
                        f"Agent delegation: {assessment.recommendation} "
                        f"({assessment.suggested_agent})"
                    )

            except Exception as e:
                # Non-blocking error - log and continue
                self.base.debug_log(f"Agent delegation failed: {e}")

            # PHASE 2: Context7 + DevStream memory injection
            # Skip heavy injection if resources are CRITICAL (optimization)
            if not skip_heavy_injection:
                enhanced_context = await self.assemble_context(file_path, content)
                if enhanced_context:
                    context_parts.append(enhanced_context)
            else:
                self.base.debug_log(
                    "Skipping heavy context injection due to CRITICAL resource status"
                )

            # PHASE 3: Inject assembled context
            if context_parts:
                final_context = "\n".join(context_parts)
                self.base.inject_context(final_context)
                self.base.success_feedback(f"Context injected for {Path(file_path).name}")
            else:
                self.base.debug_log("No relevant context found")

            # Always allow the operation to proceed
            context.output.exit_success()

        except Exception as e:
            # Non-blocking error - log and continue
            self.base.warning_feedback(f"Context injection failed: {str(e)[:50]}")
            context.output.exit_success()


def main():
    """Main entry point for PreToolUse hook."""
    # Create context using cchooks
    ctx = safe_create_context()

    # Verify it's PreToolUse context
    if not isinstance(ctx, PreToolUseContext):
        print(f"Error: Expected PreToolUseContext, got {type(ctx)}", file=sys.stderr)
        sys.exit(1)

    # Create and run hook
    hook = PreToolUseHook()

    try:
        # Run async processing
        asyncio.run(hook.process(ctx))
    except Exception as e:
        # Graceful failure - non-blocking
        print(f"⚠️  DevStream: PreToolUse error", file=sys.stderr)
        ctx.output.exit_non_block(f"Hook error: {str(e)[:100]}")


if __name__ == "__main__":
    main()