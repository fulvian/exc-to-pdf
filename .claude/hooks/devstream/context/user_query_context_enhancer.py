#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "cchooks>=0.1.4",
#     "aiohttp>=3.8.0",
#     "structlog>=23.0.0",
#     "python-dotenv>=1.0.0",
# ]
# ///

"""
DevStream UserPromptSubmit Hook - Context Enhancement before User Query
Combines Context7 library docs, DevStream memory, and task detection.
"""

import sys
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from cchooks import safe_create_context, UserPromptSubmitContext
from devstream_base import DevStreamHookBase, FeedbackLevel
from context7_client import Context7Client
from unified_client import get_unified_client

try:
    from mcp_client import get_mcp_client
    MCP_CLIENT_AVAILABLE = True
    _MCP_IMPORT_ERROR = ""
except Exception as e:  # pragma: no cover - defensive import guard
    MCP_CLIENT_AVAILABLE = False
    _MCP_IMPORT_ERROR = str(e)
    get_mcp_client = None  # type: ignore[assignment]

# Agent Auto-Delegation imports (with graceful degradation)
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'agents'))
    from pattern_matcher import PatternMatcher
    from agent_router import AgentRouter
    AGENT_DELEGATION_AVAILABLE = True
except ImportError as e:
    AGENT_DELEGATION_AVAILABLE = False
    _DELEGATION_IMPORT_ERROR = str(e)

# Protocol State Manager imports (FASE 2 Integration)
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'protocol'))
    from protocol_state_manager import ProtocolStateManager, ProtocolStep
    from enforcement_gate import EnforcementGate
    from task_first_handler import TaskFirstHandler
    from interactive_step_validator import InteractiveStepValidator
    from implementation_plan_generator import ImplementationPlanGenerator
    PROTOCOL_ENFORCEMENT_AVAILABLE = True
except ImportError as e:
    PROTOCOL_ENFORCEMENT_AVAILABLE = False
    _PROTOCOL_IMPORT_ERROR = str(e)


class UserPromptSubmitHook:
    """
    UserPromptSubmit hook for intelligent query enhancement.
    Combines Context7 research + DevStream memory + task lifecycle detection.
    """

    def __init__(self):
        self.base = DevStreamHookBase("user_prompt_submit")
        if MCP_CLIENT_AVAILABLE and get_mcp_client is not None:
            self.mcp_client = get_mcp_client()
        else:
            self.mcp_client = None
            self.base.debug_log(
                f"MCP client unavailable for Context7 integration: {_MCP_IMPORT_ERROR}"
            )
        self.unified_client = get_unified_client()
        self.context7 = Context7Client(self.mcp_client)

        # Agent Auto-Delegation components (graceful degradation)
        self.pattern_matcher = None

        # Protocol Enforcement components (FASE 3 Integration)
        self.protocol_manager = None
        self.enforcement_gate = None
        self.task_handler = None
        self.step_validator = None
        self.plan_generator = None

        if PROTOCOL_ENFORCEMENT_AVAILABLE:
            try:
                self.protocol_manager = ProtocolStateManager()
                self.enforcement_gate = EnforcementGate()
                self.task_handler = TaskFirstHandler()
                self.step_validator = InteractiveStepValidator(self.mcp_client)
                self.plan_generator = ImplementationPlanGenerator(self.mcp_client)
                self.base.debug_log("FASE 3 protocol enforcement components initialized (with plan generator)")
            except Exception as e:
                self.base.user_feedback(
                    f"FASE 3 protocol enforcement initialization failed: {e}",
                    FeedbackLevel.MINIMAL
                )
        self.agent_router = None

        if AGENT_DELEGATION_AVAILABLE:
            try:
                self.pattern_matcher = PatternMatcher()
                self.agent_router = AgentRouter()
                self.base.debug_log("Agent Auto-Delegation enabled in UserPromptSubmit")
            except Exception as e:
                self.base.debug_log(f"Agent Auto-Delegation init failed: {e}")
        else:
            self.base.debug_log(f"Agent Auto-Delegation unavailable: {_DELEGATION_IMPORT_ERROR}")

    async def detect_context7_trigger(self, user_input: str) -> bool:
        """
        Detect if Context7 should be triggered for this query.

        Args:
            user_input: User input text

        Returns:
            True if Context7 should search for library docs
        """
        return await self.context7.should_trigger_context7(user_input)

    async def get_context7_research(self, user_input: str) -> Optional[str]:
        """
        Get Context7 research for user query.

        Args:
            user_input: User input text

        Returns:
            Formatted Context7 docs or None
        """
        try:
            if not self.context7.enabled:
                self.base.debug_log("Context7 integration disabled; skipping research")
                return None

            self.base.debug_log("Context7 triggered - searching for docs")

            # Search and retrieve documentation
            result = await self.context7.search_and_retrieve(user_input)

            if result.success and result.docs:
                self.base.success_feedback(f"Context7 docs: {result.library_id}")
                return self.context7.format_docs_for_context(result)
            else:
                # LOG-003: Fix silent Context7 failures - provide clear user feedback
                error_msg = result.error if result.error else "Unknown error"
                self.base.warning_feedback(f"Context7 search failed: {error_msg[:100]}")
                self.base.debug_log(f"Context7 search failed - full error: {error_msg}")

                # Log Context7 search failure to memory
                try:
                    await self.unified_client.store_memory(
                        content=f"Context7 search failure - error: {error_msg}, query: {user_input[:100]}",
                        content_type="error",
                        keywords=["context7-failure", "log-003", "search-failure", "debugging"],
                        hook_name="user_query_context7_search_failure"
                    )
                except:
                    pass  # Non-blocking

                return None

        except Exception as e:
            self.base.debug_log(f"Context7 error: {e}")
            return None

    async def search_devstream_memory(self, user_input: str) -> Optional[str]:
        """
        Search DevStream memory for relevant context.

        Args:
            user_input: User input text

        Returns:
            Formatted memory context or None
        """
        try:
            self.base.debug_log(f"Searching DevStream memory: {user_input[:50]}...")

            # Search memory via unified client
            result = await self.unified_client.search_memory(
                query=user_input,
                limit=3,
                hook_name="user_query_context_enhancer"
            )

            if not result or not result.get("results"):
                self.base.debug_log("No relevant memory found")
                return None

            # Format memory results
            memory_items = result.get("results", [])
            if not memory_items:
                return None

            formatted = "# DevStream Memory Context\n\n"
            for i, item in enumerate(memory_items[:3], 1):
                content = item.get("content", "")[:300]
                score = item.get("relevance_score", 0.0)
                formatted += f"## Result {i} (relevance: {score:.2f})\n{content}\n\n"

            self.base.success_feedback(f"Found {len(memory_items)} relevant memories")
            return formatted

        except Exception as e:
            self.base.debug_log(f"Memory search error: {e}")
            return None

    def estimate_task_complexity(self, user_input: str) -> Dict[str, Any]:
        """
        Estimate task complexity to determine if protocol enforcement is needed.

        Args:
            user_input: User input text

        Returns:
            Dict with complexity analysis and enforcement decision
        """
        input_lower = user_input.lower()
        triggers = []

        # Trigger 1: Duration estimation (keywords indicating complexity)
        duration_keywords = [
            "implement", "build", "create", "refactor", "migrate",
            "integrate", "optimize", "design", "architect"
        ]
        if any(kw in input_lower for kw in duration_keywords):
            triggers.append("estimated_duration_>15min")

        # Trigger 2: Code implementation (explicit tool mentions)
        implementation_keywords = [
            "write code", "edit file", "modify", "update code",
            "add function", "create class", "implement feature"
        ]
        if any(kw in input_lower for kw in implementation_keywords):
            triggers.append("code_implementation_required")

        # Trigger 3: Architectural decisions
        architecture_keywords = [
            "architecture", "design pattern", "system design",
            "api design", "database schema", "integration"
        ]
        if any(kw in input_lower for kw in architecture_keywords):
            triggers.append("architectural_decisions_required")

        # Trigger 4: Multiple files/components
        multi_file_keywords = [
            "files", "components", "modules", "services",
            "multi-", "across", "integrate"
        ]
        if any(kw in input_lower for kw in multi_file_keywords):
            triggers.append("multiple_files_or_components")

        # Trigger 5: Research requirement
        research_keywords = [
            "research", "best practices", "how to", "documentation",
            "library", "framework", "pattern"
        ]
        if any(kw in input_lower for kw in research_keywords):
            triggers.append("context7_research_required")

        # Decision: Enforce if ANY trigger detected
        enforce = len(triggers) > 0

        return {
            "enforce_protocol": enforce,
            "triggers": triggers,
            "complexity_score": len(triggers),
            "user_input_preview": user_input[:100]
        }

    def generate_enforcement_prompt(self, complexity: Dict[str, Any]) -> str:
        """
        Generate enforcement gate prompt for user.

        Args:
            complexity: Complexity analysis from estimate_task_complexity

        Returns:
            Formatted enforcement prompt
        """
        triggers_formatted = "\n".join([f"  - {t.replace('_', ' ').title()}" for t in complexity["triggers"]])

        prompt = f"""⚠️ DevStream Protocol Required

**Detected Complexity Triggers**:
{triggers_formatted}

This task requires following the DevStream 7-step workflow:
DISCUSSION → ANALYSIS → RESEARCH → PLANNING → APPROVAL → IMPLEMENTATION → VERIFICATION

**OPTIONS**:
✅ [RECOMMENDED] Follow DevStream protocol (research-driven, quality-assured)
   - Context7 research for best practices
   - @code-reviewer validation (OWASP Top 10 security)
   - 95%+ test coverage requirement
   - Approval workflow (decisions documented)

⚠️  [OVERRIDE] Skip protocol (quick fix, NO quality assurance)
   - ❌ No Context7 research (potential outdated/incorrect patterns)
   - ❌ No @code-reviewer validation (security gaps)
   - ❌ No testing requirements (95%+ coverage waived)
   - ❌ No approval workflow (decisions undocumented)

**Choose**:
1. Follow Protocol (RECOMMENDED)
2. Override (explicit risk acknowledgment required)
Cancel"""

        return prompt

    async def check_agent_delegation(self, user_input: str) -> Optional[str]:
        """
        Check if task should be delegated to specialized agent (ALWAYS-ON).

        Args:
            user_input: User input text

        Returns:
            Agent delegation advisory message if match found, None otherwise

        Note:
            This is ALWAYS checked for every user request (always-on mode).
            Confidence-based routing:
            - ≥ 0.95: AUTOMATIC delegation (immediate)
            - 0.85-0.94: ADVISORY delegation (suggest + request approval)
            - < 0.85: @tech-lead COORDINATION (multi-agent orchestration)
        """
        # Check if components available
        if not self.pattern_matcher or not self.agent_router:
            self.base.debug_log("Agent delegation unavailable (components not loaded)")
            return None

        try:
            # Match patterns from user query
            pattern_match = self.pattern_matcher.match_patterns(
                file_path=None,
                content=None,
                user_query=user_input,
                tool_name=None
            )

            if not pattern_match:
                self.base.debug_log("No agent pattern match found for user query")
                return None

            # Assess task complexity and delegation
            assessment = self.agent_router.assess_task(
                pattern_match=pattern_match,
                user_query=user_input
            )

            if not assessment:
                return None

            # Format delegation advisory based on confidence
            advisory_message = self.agent_router.format_advisory_message(assessment)

            # Log delegation event
            try:
                await self.unified_client.store_memory(
                    content=f"Agent delegation: {assessment.suggested_agent} (confidence: {assessment.confidence:.2f}). User query: {user_input[:200]}",
                    content_type="decision",
                    keywords=["agent-delegation", "auto-routing", "confidence-based"],
                    hook_name="user_query_context_enhancer_delegation"
                )
            except Exception as e:
                self.base.debug_log(f"Failed to log delegation event: {e}")

            self.base.success_feedback(
                f"Agent delegation: {assessment.recommendation} "
                f"({assessment.suggested_agent}, confidence: {assessment.confidence:.2f})"
            )

            return advisory_message

        except Exception as e:
            self.base.debug_log(f"Agent delegation error: {e}")
            return None

    async def check_protocol_enforcement(self, user_input: str) -> Optional[str]:
        """
        Check if protocol enforcement is required and return prompt if needed.

        Args:
            user_input: User input text

        Returns:
            Enforcement prompt if required, None otherwise
        """
        # Estimate complexity
        complexity = self.estimate_task_complexity(user_input)

        if not complexity["enforce_protocol"]:
            self.base.debug_log("Protocol enforcement not required (simple task)")
            return None

        self.base.debug_log(f"Protocol enforcement triggered: {complexity['triggers']}")

        # Generate enforcement prompt
        enforcement_prompt = self.generate_enforcement_prompt(complexity)

        # Store enforcement event in memory via unified_client (with embedding generation)
        try:
            await self.unified_client.store_memory(
                content=f"Protocol enforcement triggered: {complexity['triggers']}. User input: {user_input[:200]}",
                content_type="decision",
                keywords=["protocol-enforcement", "complexity-analysis", "workflow-gate"],
                hook_name="user_query_context_enhancer_enforcement"
            )
        except Exception as e:
            self.base.debug_log(f"Failed to log enforcement event: {e}")

        return enforcement_prompt

    async def detect_task_lifecycle_event(self, user_input: str) -> Optional[Dict[str, str]]:
        """
        Detect task lifecycle events from user input.

        Args:
            user_input: User input text

        Returns:
            Event data if detected, None otherwise
        """
        input_lower = user_input.lower()

        # Task creation patterns
        if any(pattern in input_lower for pattern in [
            "create task",
            "new task",
            "add task",
            "start working on"
        ]):
            return {
                "event_type": "task_creation",
                "pattern": "User initiated new task",
                "query": user_input[:100]
            }

        # Task completion patterns
        elif any(pattern in input_lower for pattern in [
            "complete",
            "finished",
            "done with",
            "completed the",
            "task complete"
        ]):
            return {
                "event_type": "task_completion",
                "pattern": "User indicated task completion",
                "query": user_input[:100]
            }

        # Implementation progress patterns
        elif any(pattern in input_lower for pattern in [
            "implement",
            "build",
            "create",
            "working on"
        ]):
            return {
                "event_type": "implementation_progress",
                "pattern": "User starting implementation work",
                "query": user_input[:100]
            }

        return None

    async def detect_direct_db_commands(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Detect Direct DB commands in natural language and trigger automatic operations.

        Args:
            user_input: User input text

        Returns:
            Command data if detected, None otherwise
        """
        input_lower = user_input.lower()

        # Direct DB UPDATE commands
        update_patterns = [
            "aggiorna il task devstream",
            "aggiorna il db devstream",
            "aggiorna la memoria di progetto",
            "aggiorna il database devstream",
            "update task devstream",
            "update db devstream",
            "update project memory",
            "aggiorna lo stato del task",
            "aggiorna il progresso del task",
            "salva lo stato del task",
            "store task status",
            "save project status",
            "aggiorna task debugging",
            "update debugging task"
        ]

        if any(pattern in input_lower for pattern in update_patterns):
            return {
                "command_type": "store_memory",
                "pattern": "Direct DB update command detected",
                "query": user_input,
                "content_type": "decision",
                "keywords": ["direct-db-update", "task-status", "devstream-database", "natural-language-command"]
            }

        # Direct DB SEARCH commands
        search_patterns = [
            "cerca il task",
            "cerca nel db devstream",
            "trova il task",
            "cerca nella memoria",
            "search task",
            "search in db devstream",
            "find task",
            "search in memory",
            "trova nel database",
            "cerca il progetto"
        ]

        if any(pattern in input_lower for pattern in search_patterns):
            return {
                "command_type": "search_memory",
                "pattern": "Direct DB search command detected",
                "query": user_input,
                "limit": 5,
                "content_type": None,
                "keywords": ["direct-db-search", "task-search", "devstream-database", "natural-language-command"]
            }

        return None

    async def execute_direct_db_command(self, command_data: Dict[str, Any]) -> Optional[str]:
        """
        Execute Direct DB command using enhanced async patterns from Context7 research.

        Applies async context manager pattern and structured error handling from
        aiosqlite and FastAPI best practices.

        Args:
            command_data: Command data from detect_direct_db_commands

        Returns:
            Result message or None
        """
        import time
        from contextlib import asynccontextmanager

        # Enhanced error handling with performance logging (FastAPI pattern)
        start_time = time.time()

        try:
            # Pattern 1: Async Context Manager with proper resource management
            from direct_client import store_memory_async, search_memory_async

            # Pattern 2: Dependency injection style with try/finally cleanup
            @asynccontextmanager
            async def direct_db_operation(operation_name: str):
                """Async context manager for Direct DB operations with logging."""
                self.base.debug_log(f"Starting Direct DB operation: {operation_name}")
                try:
                    yield
                    duration_ms = (time.time() - start_time) * 1000
                    await self._log_direct_call_success(
                        operation=operation_name,
                        parameters=command_data,
                        duration_ms=duration_ms
                    )
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    await self._log_direct_call_error(
                        operation=operation_name,
                        parameters=command_data,
                        duration_ms=duration_ms,
                        error=str(e)
                    )
                    raise

            if command_data["command_type"] == "store_memory":
                # Enhanced store operation with async context manager
                async with direct_db_operation("natural_language_store_memory"):
                    content = self._generate_store_content(command_data)

                    # Apply structured error handling with retry logic (PERF-001 pattern)
                    result = await self._execute_with_retry(
                        store_memory_async,
                        content=content,
                        content_type=command_data["content_type"],
                        keywords=command_data["keywords"]
                    )

                    if result:
                        memory_id = result.get('memory_id', 'unknown')
                        embedding_generated = result.get('embedding_generated', False)
                        return self._format_store_success(memory_id, embedding_generated)
                    else:
                        return "❌ Direct DB Update: Fallito"

            elif command_data["command_type"] == "search_memory":
                # Enhanced search operation with async context manager
                async with direct_db_operation("natural_language_search_memory"):
                    search_terms = self._extract_search_terms(command_data["query"])

                    # Apply structured error handling with fallback
                    result = await self._execute_with_retry(
                        search_memory_async,
                        query=search_terms,
                        content_type=command_data.get("content_type"),
                        limit=command_data.get("limit", 5)
                    )

                    return self._format_search_result(result, search_terms)

        except Exception as e:
            # Enhanced error logging with context (FastAPI middleware pattern)
            duration_ms = (time.time() - start_time) * 1000
            error_msg = f"❌ Direct DB Error: {str(e)[:50]}"

            self.base.debug_log(f"Direct DB command execution error after {duration_ms:.2f}ms: {e}")

            # Log error with structured format for debugging
            await self._log_direct_call_error(
                operation="natural_language_command",
                parameters=command_data,
                duration_ms=duration_ms,
                error=str(e)
            )

            return error_msg

        return None

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

    def _extract_search_terms(self, query: str) -> str:
        """Extract relevant search terms from natural language query."""
        # Remove common command patterns and keep the actual search terms
        patterns_to_remove = [
            "cerca il task", "cerca nel db devstream", "trova il task",
            "cerca nella memoria", "search task", "search in db devstream",
            "find task", "search in memory", "trova nel database",
            "cerca il progetto", "cerca task", "trova task"
        ]

        query_lower = query.lower()
        for pattern in patterns_to_remove:
            if pattern in query_lower:
                query = query.replace(pattern, "", 1).strip()

        # If no meaningful content left, use the original query
        return query if query.strip() else query_lower

    # Enhanced helper methods applying Context7 patterns

    def _generate_store_content(self, command_data: Dict[str, Any]) -> str:
        """
        Generate enhanced content for storage with structured format.

        Args:
            command_data: Command data from detect_direct_db_commands

        Returns:
            Formatted content string with metadata
        """
        return f"""# Direct DB Update Triggered by Natural Language Command

**Original Query**: {command_data['query']}

**Command Detected**: {command_data['pattern']}
**Execution Method**: Enhanced Direct DB Client (async context manager pattern)
**Timestamp**: {self._get_timestamp()}

## Enhanced Implementation:
This status update was automatically generated using Context7 best practices:
- Async context manager pattern for proper resource management
- Structured error handling with performance logging
- Retry logic with exponential backoff (PERF-001)
- Background task support for non-blocking operations

**Keywords**: {', '.join(command_data['keywords'])}
**Content Type**: {command_data['content_type']}
**Storage**: Enhanced Direct DB with automatic embedding generation
**Patterns Applied**: aiosqlite async context managers, FastAPI middleware logging

---
*This record demonstrates the enhanced Direct DB natural language interface with Context7 patterns applied.*"""

    async def _execute_with_retry(self, func, max_retries: int = 3, **kwargs):
        """
        Execute function with retry logic using PERF-001 exponential backoff pattern.

        Args:
            func: Async function to execute
            max_retries: Maximum retry attempts
            **kwargs: Function arguments

        Returns:
            Function result or None if all retries fail
        """
        import asyncio

        for attempt in range(max_retries):
            try:
                return await func(**kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    # Last attempt failed, re-raise
                    raise

                # Exponential backoff: 2^attempt seconds (PERF-001 pattern)
                backoff_time = 2 ** attempt
                self.base.debug_log(f"Direct DB operation failed (attempt {attempt + 1}), retrying in {backoff_time}s: {e}")
                await asyncio.sleep(backoff_time)

        return None

    def _format_store_success(self, memory_id: str, embedding_generated: bool) -> str:
        """
        Format success message for store operations.

        Args:
            memory_id: Generated memory ID
            embedding_generated: Whether embeddings were generated

        Returns:
            Formatted success message
        """
        embedding_status = "SÌ" if embedding_generated else "NO"
        return f"✅ Direct DB Update: Memorizzato con ID {memory_id} (Embeddings: {embedding_status}) [Enhanced Pattern]"

    def _format_search_result(self, result: Optional[Dict[str, Any]], search_terms: str) -> str:
        """
        Format search result with enhanced information.

        Args:
            result: Search result from Direct DB
            search_terms: Original search terms

        Returns:
            Formatted search result message
        """
        if result and result.get("success"):
            found_count = result.get("count", 0)
            if found_count > 0:
                return f"✅ Direct DB Search: Trovati {found_count} risultati per '{search_terms}' [Enhanced Pattern]"
            else:
                return f"⚠️ Direct DB Search: Nessun risultato trovato per '{search_terms}' [Enhanced Pattern]"
        else:
            return "❌ Direct DB Search: Fallito [Enhanced Pattern]"

    async def _log_direct_call_success(
        self,
        operation: str,
        parameters: Dict[str, Any],
        duration_ms: float
    ) -> None:
        """
        Log successful Direct DB operation using FastAPI middleware pattern.

        Args:
            operation: Operation name
            parameters: Operation parameters
            duration_ms: Execution time in milliseconds
        """
        try:
            # Use LoggerAdapter with structured logging (Context7 pattern)
            from logger_adapter import LoggerAdapter

            # Try to get logger through service locator
            try:
                from service_interfaces import get_logger
                logger = get_logger()

                if hasattr(logger, 'log_direct_call'):
                    logger.log_direct_call(
                        operation=operation,
                        parameters=parameters,
                        success=True,
                        duration_ms=duration_ms
                    )
                else:
                    # Fallback to standard logging
                    logger.info(f"Direct DB operation {operation} completed in {duration_ms:.2f}ms",
                              extra={"operation": operation, "duration_ms": duration_ms, **parameters})
            except:
                # Final fallback to base logger
                self.base.debug_log(f"Direct DB SUCCESS: {operation} in {duration_ms:.2f}ms")

        except Exception as e:
            # Non-blocking error - don't fail the operation
            self.base.debug_log(f"Failed to log Direct DB success: {e}")

    async def _log_direct_call_error(
        self,
        operation: str,
        parameters: Dict[str, Any],
        duration_ms: float,
        error: str
    ) -> None:
        """
        Log failed Direct DB operation using FastAPI middleware pattern.

        Args:
            operation: Operation name
            parameters: Operation parameters
            duration_ms: Execution time in milliseconds
            error: Error message
        """
        try:
            # Use LoggerAdapter with structured logging (Context7 pattern)
            try:
                from service_interfaces import get_logger
                logger = get_logger()

                if hasattr(logger, 'log_direct_call'):
                    logger.log_direct_call(
                        operation=operation,
                        parameters=parameters,
                        success=False,
                        duration_ms=duration_ms,
                        error=error
                    )
                else:
                    # Fallback to standard logging
                    logger.error(f"Direct DB operation {operation} failed in {duration_ms:.2f}ms: {error}",
                               extra={"operation": operation, "duration_ms": duration_ms, "error": error, **parameters})
            except:
                # Final fallback to base logger
                self.base.debug_log(f"Direct DB ERROR: {operation} in {duration_ms:.2f}ms - {error}")

        except Exception as e:
            # Non-blocking error - don't fail the operation
            self.base.debug_log(f"Failed to log Direct DB error: {e}")

    async def assemble_enhanced_context(
        self,
        user_input: str
    ) -> Optional[str]:
        """
        Assemble enhanced context from multiple sources.

        Args:
            user_input: User input text

        Returns:
            Assembled enhanced context or None
        """
        context_parts = []

        # PRIORITY 0: Agent Auto-Delegation (ALWAYS-ON, checked FIRST)
        delegation_advisory = await self.check_agent_delegation(user_input)
        if delegation_advisory:
            context_parts.append(f"# Agent Auto-Delegation Advisory\n\n{delegation_advisory}")
            self.base.success_feedback("Agent delegation advisory provided")

        # PRIORITY 1: Check protocol enforcement (MANDATORY)
        enforcement_prompt = await self.check_protocol_enforcement(user_input)
        if enforcement_prompt:
            context_parts.append(f"# Protocol Enforcement Gate\n\n{enforcement_prompt}")
            self.base.success_feedback("Protocol enforcement gate activated")

        # PRIORITY 2: Check if Context7 should trigger
        if await self.detect_context7_trigger(user_input):
            context7_docs = await self.get_context7_research(user_input)
            if context7_docs:
                context_parts.append(context7_docs)

        # PRIORITY 3: Search DevStream memory
        memory_context = await self.search_devstream_memory(user_input)
        if memory_context:
            context_parts.append(memory_context)

        # PRIORITY 4: Detect Direct DB natural language commands (NEW!)
        direct_db_command = await self.detect_direct_db_commands(user_input)
        if direct_db_command:
            # Execute the Direct DB command immediately
            command_result = await self.execute_direct_db_command(direct_db_command)
            if command_result:
                command_context = f"""# Direct DB Natural Language Command Executed

**Command Result**: {command_result}
**Original Query**: {direct_db_command['query']}
**Pattern**: {direct_db_command['pattern']}

The Direct DB client automatically executed your natural language command.
This demonstrates the working natural language interface to DevStream database.
"""
                context_parts.append(command_context)
                self.base.success_feedback("Direct DB natural language command executed")

        # PRIORITY 5: Detect task lifecycle events
        task_event = await self.detect_task_lifecycle_event(user_input)
        if task_event:
            event_context = f"""# Task Lifecycle Event Detected

**Event Type**: {task_event['event_type']}
**Pattern**: {task_event['pattern']}

This query appears to be related to task management. Consider using TodoWrite for tracking progress.
"""
            context_parts.append(event_context)

        if not context_parts:
            return None

        # Assemble final context
        assembled = "\n\n---\n\n".join(context_parts)
        return f"# Enhanced Context for Query\n\n{assembled}"

    # FASE 3: Enhanced Protocol Enforcement Methods

    async def _enhanced_protocol_enforcement(
        self,
        user_input: str,
        complexity: Dict[str, Any],
        current_state
    ) -> Dict[str, Any]:
        """
        Enhanced protocol enforcement with full interactive UI.

        Args:
            user_input: User's input prompt
            complexity: Complexity analysis result
            current_state: Current protocol state

        Returns:
            Dict with enforcement result and action
        """
        try:
            # Check if we need to enforce task creation first (Step 0)
            if current_state.protocol_step == ProtocolStep.IDLE:
                self.base.debug_log("Starting new protocol session - enforcing task creation")

                task_result = await self.task_handler.enforce_task_creation(user_input)

                if task_result[0] and task_result[1]:  # Success and task_id provided
                    # Update current state with new task
                    current_state = await self.protocol_manager.advance_step(
                        current_state,
                        ProtocolStep.DISCUSSION,
                        task_id=task_result[1]
                    )
                    self.base.success_feedback(
                        f"✅ Task created: {task_result[1]} → Step 1: DISCUSSION"
                    )
                elif not task_result[0]:  # User cancelled
                    return {"action": "cancel", "reason": "Task creation cancelled"}
                else:  # Task creation overridden or not needed
                    current_state = await self.protocol_manager.advance_step(
                        current_state,
                        ProtocolStep.DISCUSSION
                    )

            # Build enforcement context for interactive UI
            from enforcement_gate import EnforcementContext
            from datetime import datetime, timezone
            import uuid

            enforcement_context = EnforcementContext(
                task_description=user_input[:200],
                estimated_duration=max(15, complexity["complexity_score"] * 30),
                complexity_score=complexity["complexity_score"] / 5.0,  # Normalize to 0.0-1.0
                involves_code="code_implementation_required" in complexity["triggers"],
                involves_architecture="architectural_decisions_required" in complexity["triggers"],
                requires_context7="context7_research_required" in complexity["triggers"],
                trigger_reasons=complexity["triggers"],
                session_id=current_state.session_id,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

            # Show enhanced enforcement gate with full UI
            decision = await self.enforcement_gate.show_enforcement_gate(
                enforcement_context,
                self.mcp_client
            )

            return {
                "action": decision.value,
                "enforcement_context": enforcement_context,
                "current_state": current_state,
                "decision_timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            self.base.debug_log(f"Enhanced enforcement error: {e}")
            # Fallback to simple processing
            return {"action": "fallback", "reason": str(e)}

    async def _handle_protocol_override(
        self,
        user_input: str,
        complexity: Dict[str, Any],
        enforcement_result: Dict[str, Any]
    ) -> None:
        """
        Handle protocol override with risk acknowledgment.

        Args:
            user_input: User's input prompt
            complexity: Complexity analysis result
            enforcement_result: Result from enforcement gate
        """
        try:
            # Log override decision with full risk acknowledgment
            override_content = (
                f"PROTOCOL OVERRIDE - RISKS ACKNOWLEDGED\n"
                f"User Input: {user_input[:200]}\n"
                f"Complexity Triggers: {', '.join(complexity['triggers'])}\n"
                f"Session ID: {enforcement_result['current_state'].session_id}\n"
                f"Override Timestamp: {enforcement_result['decision_timestamp']}\n\n"
                f"RISKS ACCEPTED:\n"
                f"❌ No Context7 research (potential outdated/incorrect patterns)\n"
                f"❌ No @code-reviewer validation (OWASP Top 10 security gaps)\n"
                f"❌ No testing requirements (95%+ coverage waived)\n"
                f"❌ No approval workflow (decisions undocumented)\n"
                f"❌ No step-by-step validation (quality assurance bypassed)"
            )

            await self.base.safe_mcp_call(
                self.mcp_client,
                "devstream_store_memory",
                {
                    "content": override_content,
                    "content_type": "decision",
                    "keywords": [
                        "protocol-override",
                        "risk-acknowledgment",
                        "quality-bypass",
                        enforcement_result['current_state'].session_id
                    ]
                }
            )

            self.base.user_feedback(
                "⚠️ Protocol override logged - quality assurance disabled",
                FeedbackLevel.MINIMAL
            )

        except Exception as e:
            self.base.debug_log(f"Override handling error: {e}")

    async def _handle_protocol_workflow(
        self,
        user_input: str,
        current_state,
        enforcement_result: Dict[str, Any]
    ) -> str:
        """
        Handle full protocol workflow with interactive step validation.

        Args:
            user_input: User's input prompt
            current_state: Current protocol state
            enforcement_result: Result from enforcement gate

        Returns:
            Enhanced user input with protocol instructions
        """
        try:
            self.base.success_feedback(
                f"🎯 Protocol workflow started - Current: {current_state.protocol_step}"
            )

            # ENHANCED: Interactive step validation and progression
            workflow_result = await self._interactive_step_validation(
                user_input, current_state
            )

            if workflow_result["step_validated"]:
                # Build enhanced user input with protocol instructions
                protocol_instructions = self._build_protocol_instructions(
                    current_state.protocol_step,
                    workflow_result
                )

                enhanced_input = f"{user_input}\n\n{protocol_instructions}"

                self.base.success_feedback(
                    f"✅ Step {current_state.protocol_step.value} validated - Protocol active"
                )

                return enhanced_input
            else:
                # Step validation failed - show blocking message
                self.base.user_feedback(
                    f"⚠️ Step {current_state.protocol_step.value} requires completion before proceeding",
                    FeedbackLevel.MINIMAL
                )
                return user_input

        except Exception as e:
            self.base.debug_log(f"Protocol workflow handling error: {e}")
            return user_input

    async def _interactive_step_validation(
        self,
        user_input: str,
        current_state
    ) -> Dict[str, Any]:
        """
        FASE 3 Enhanced: Interactive step validation with comprehensive summaries.

        Args:
            user_input: User's input prompt
            current_state: Current protocol state

        Returns:
            Dict with validation result
        """
        try:
            current_step = current_state.protocol_step

            # Use the enhanced InteractiveStepValidator
            if self.step_validator:
                validation_result = await self.step_validator.validate_step_completion(
                    current_step, user_input, {"session_id": current_state.session_id}
                )

                # Handle step transition with user confirmation
                transition_approved, transition_data = await self.step_validator.handle_step_transition(
                    validation_result, current_state.session_id
                )

                if transition_approved and transition_data:
                    # Update protocol state to next step
                    next_state = await self.protocol_manager.advance_step(
                        current_state,
                        transition_data.to_step,
                        metadata_updates={
                            "step_completed_at": transition_data.timestamp,
                            "transition_notes": transition_data.notes
                        }
                    )

                    return {
                        "step_validated": True,
                        "validation_result": validation_result,
                        "next_step_ready": True,
                        "advanced_to_step": transition_data.to_step,
                        "new_state": next_state
                    }
                else:
                    # Step validation failed or user declined progression
                    return {
                        "step_validated": False,
                        "reason": "Step transition not approved",
                        "validation_result": validation_result,
                        "requires_more_work": True
                    }
            else:
                # Fallback to basic validation
                validation_result = await self._validate_current_step_completion(
                    current_step, user_input
                )

                if validation_result["completed"]:
                    return {
                        "step_validated": True,
                        "validation_result": validation_result,
                        "next_step_ready": True
                    }
                else:
                    return {
                        "step_validated": False,
                        "reason": validation_result.get("missing_requirements", []),
                        "requires_more_work": True
                    }

        except Exception as e:
            self.base.debug_log(f"FASE 3 interactive step validation error: {e}")
            return {"step_validated": False, "reason": str(e)}

    async def _validate_current_step_completion(
        self,
        step: ProtocolStep,
        user_input: str
    ) -> Dict[str, Any]:
        """
        Validate completion of current protocol step.

        Args:
            step: Current protocol step
            user_input: User's input prompt

        Returns:
            Dict with validation result
        """
        # Step-specific validation logic
        if step == ProtocolStep.DISCUSSION:
            return await self._validate_discussion_step(user_input)
        elif step == ProtocolStep.ANALYSIS:
            return await self._validate_analysis_step(user_input)
        elif step == ProtocolStep.RESEARCH:
            return await self._validate_research_step(user_input)
        elif step == ProtocolStep.PLANNING:
            return await self._validate_planning_step(user_input)
        elif step == ProtocolStep.APPROVAL:
            return await self._validate_approval_step(user_input)
        elif step == ProtocolStep.IMPLEMENTATION:
            return await self._validate_implementation_step(user_input)
        elif step == ProtocolStep.VERIFICATION:
            return await self._validate_verification_step(user_input)
        else:
            return {"completed": False, "reason": "Unknown step"}

    async def _validate_discussion_step(self, user_input: str) -> Dict[str, Any]:
        """Validate DISCUSSION step completion."""
        # Check for discussion indicators in user input
        discussion_indicators = [
            "discuss", "let's talk about", "consider", "trade-offs",
            "pros and cons", "alternatives", "approach", "strategy"
        ]

        has_discussion = any(indicator in user_input.lower() for indicator in discussion_indicators)
        sufficient_length = len(user_input) > 100  # Minimum meaningful discussion

        completed = has_discussion and sufficient_length

        return {
            "completed": completed,
            "has_discussion": has_discussion,
            "sufficient_length": sufficient_length,
            "missing_requirements": [] if completed else [
                "Meaningful discussion of the problem/objective",
                "Consideration of trade-offs and alternatives"
            ]
        }

    async def _validate_analysis_step(self, user_input: str) -> Dict[str, Any]:
        """Validate ANALYSIS step completion."""
        analysis_indicators = [
            "analyze", "examine", "break down", "components",
            "requirements", "constraints", "dependencies", "impact"
        ]

        has_analysis = any(indicator in user_input.lower() for indicator in analysis_indicators)

        return {
            "completed": has_analysis,
            "has_analysis": has_analysis,
            "missing_requirements": [] if has_analysis else [
                "Analysis of codebase for similar patterns",
                "Identification of files to modify",
                "Complexity estimation and constraints"
            ]
        }

    async def _validate_research_step(self, user_input: str) -> Dict[str, Any]:
        """Validate RESEARCH step completion."""
        # Check for Context7 usage or research indicators
        research_indicators = [
            "research", "best practices", "documentation", "library",
            "framework", "context7", "study", "investigate"
        ]

        has_research = any(indicator in user_input.lower() for indicator in research_indicators)

        # Also check if Context7 was triggered in this session
        context7_triggered = await self.detect_context7_trigger(user_input)

        completed = has_research or context7_triggered

        return {
            "completed": completed,
            "has_research": has_research,
            "context7_triggered": context7_triggered,
            "missing_requirements": [] if completed else [
                "Context7 research for best practices",
                "Documentation of findings",
                "Validation of approach with research"
            ]
        }

    async def _validate_planning_step(self, user_input: str) -> Dict[str, Any]:
        """Validate PLANNING step completion."""
        planning_indicators = [
            "plan", "todo", "steps", "implementation", "breakdown",
            "micro-tasks", "milestones", "acceptance criteria"
        ]

        has_planning = any(indicator in user_input.lower() for indicator in planning_indicators)

        return {
            "completed": has_planning,
            "has_planning": has_planning,
            "missing_requirements": [] if has_planning else [
                "TodoWrite list creation for implementation",
                "Micro-task breakdown (10-15 min tasks)",
                "Definition of completion criteria"
            ]
        }

    async def _validate_approval_step(self, user_input: str) -> Dict[str, Any]:
        """Validate APPROVAL step completion."""
        approval_indicators = [
            "approve", "approved", "confirm", "proceed", "ok", "agreed",
            "accept", "go ahead", "move forward"
        ]

        has_approval = any(indicator in user_input.lower() for indicator in approval_indicators)

        return {
            "completed": has_approval,
            "has_approval": has_approval,
            "missing_requirements": [] if has_approval else [
                "Explicit approval of the implementation plan",
                "Confirmation of approach and timeline"
            ]
        }

    async def _validate_implementation_step(self, user_input: str) -> Dict[str, Any]:
        """Validate IMPLEMENTATION step completion."""
        implementation_indicators = [
            "implement", "code", "write", "create", "build", "develop",
            "complete", "finished", "done", "implemented"
        ]

        has_implementation = any(indicator in user_input.lower() for indicator in implementation_indicators)

        return {
            "completed": has_implementation,
            "has_implementation": has_implementation,
            "missing_requirements": [] if has_implementation else [
                "Actual implementation of the planned features",
                "Code completion and testing"
            ]
        }

    async def _validate_verification_step(self, user_input: str) -> Dict[str, Any]:
        """Validate VERIFICATION step completion."""
        verification_indicators = [
            "test", "verify", "validate", "check", "confirm working",
            "quality assurance", "review", "passes", "successful"
        ]

        has_verification = any(indicator in user_input.lower() for indicator in verification_indicators)

        return {
            "completed": has_verification,
            "has_verification": has_verification,
            "missing_requirements": [] if has_verification else [
                "Testing of implemented features",
                "Quality assurance validation",
                "Confirmation that requirements are met"
            ]
        }

    async def _show_step_completion_summary(
        self,
        step: ProtocolStep,
        validation_result: Dict[str, Any]
    ) -> None:
        """Show step completion summary to user."""
        print(f"\n✅ STEP {step.value} COMPLETED: {step}")
        print("=" * 60)
        print(f"Step {step.value} has been successfully validated.")
        print("Ready to proceed to next step.")
        print("=" * 60)

    async def _show_step_incompletion_details(
        self,
        step: ProtocolStep,
        validation_result: Dict[str, Any]
    ) -> None:
        """Show step incompletion details to user."""
        print(f"\n⚠️ STEP {step.value} INCOMPLETE: {step}")
        print("=" * 60)
        print("This step requires completion before proceeding:")

        missing = validation_result.get("missing_requirements", [])
        for i, requirement in enumerate(missing, 1):
            print(f"  {i}. {requirement}")

        print("=" * 60)

    async def _prompt_step_progression(
        self,
        step: ProtocolStep,
        validation_result: Dict[str, Any]
    ) -> bool:
        """
        Prompt user for step progression confirmation.

        Args:
            step: Current protocol step
            validation_result: Validation result

        Returns:
            True if user confirms progression, False otherwise
        """
        try:
            # Try to use PyInquirer if available
            try:
                from PyInquirer import prompt

                questions = [{
                    'type': 'confirm',
                    'name': 'proceed',
                    'message': f'Step {step.value} completed. Proceed to next step?',
                    'default': True
                }]

                answers = prompt(questions)
                return answers.get('proceed', False)

            except ImportError:
                # Fallback to simple input
                response = input(f"\nStep {step.value} completed. Proceed to next step? (y/N): ").lower()
                return response.startswith('y')

        except Exception as e:
            self.base.debug_log(f"Step progression prompt error: {e}")
            return True  # Default to proceeding on error

    def _build_protocol_instructions(
        self,
        step: ProtocolStep,
        workflow_result: Dict[str, Any]
    ) -> str:
        """
        Build protocol instructions for enhanced user input.

        Args:
            step: Current protocol step
            workflow_result: Result from workflow validation

        Returns:
            Protocol instructions string
        """
        instructions = [
            "🎯 DevStream 7-Step Protocol Active",
            f"Current Step: {step.value} - {step}",
            "",
            "Protocol Workflow:",
            "DISCUSSION → ANALYSIS → RESEARCH → PLANNING → APPROVAL → IMPLEMENTATION → VERIFICATION",
            "",
            "Quality Requirements:",
            "• Context7 research for best practices",
            "• @code-reviewer validation (OWASP Top 10)",
            "• 95%+ test coverage",
            "• Full documentation with examples",
            "• Performance validation",
            "",
            f"Status: ✅ Step {step.value} validated - Ready for execution"
        ]

        return "\n".join(instructions)

    async def process(self, context: UserPromptSubmitContext) -> None:
        """
        Main hook processing logic with FASE 3 Enhanced Protocol Enforcement.

        Integrates full enforcement gate UI, interactive step validation, and complete
        workflow enforcement from TASK_CREATION through VERIFICATION.

        Args:
            context: UserPromptSubmit context from cchooks
        """
        # Check if hook should run
        if not self.base.should_run():
            self.base.debug_log("Hook disabled via config")
            context.output.exit_success()
            return

        # Extract user input
        user_input = getattr(context, 'user_input', getattr(context, 'prompt', ''))

        if not user_input or len(user_input) < 10:
            self.base.debug_log("User input too short for enhancement")
            context.output.exit_success()
            return

        # FASE 3: Enhanced Protocol Enforcement with Full Interactive UI
        if PROTOCOL_ENFORCEMENT_AVAILABLE and self.protocol_manager:
            try:
                # Get current protocol state
                current_state = await self.protocol_manager.get_current_state()

                # Analyze task complexity for protocol enforcement
                complexity = self.estimate_task_complexity(user_input)

                if complexity["enforce_protocol"]:
                    self.base.debug_log(
                        f"FASE 3 enforcement triggered: {complexity['triggers']} (score: {complexity['complexity_score']})"
                    )

                    # ENHANCED: Full enforcement gate with interactive UI
                    enforcement_result = await self._enhanced_protocol_enforcement(
                        user_input, complexity, current_state
                    )

                    if enforcement_result["action"] == "cancel":
                        self.base.debug_log("User cancelled protocol enforcement")
                        context.output.exit_success()
                        return
                    elif enforcement_result["action"] == "override":
                        await self._handle_protocol_override(
                            user_input, complexity, enforcement_result
                        )
                        # Continue with normal processing but with gates disabled
                    elif enforcement_result["action"] == "protocol":
                        # ENHANCED: Interactive step validation and progression
                        user_input = await self._handle_protocol_workflow(
                            user_input, current_state, enforcement_result
                        )
                    else:
                        # Fallback: continue with normal processing
                        self.base.debug_log("Using fallback processing mode")

            except Exception as e:
                self.base.user_feedback(
                    f"FASE 3 protocol enforcement error: {e}",
                    FeedbackLevel.MINIMAL
                )
                # Continue with normal processing on error

        self.base.debug_log(f"Processing user query: {len(user_input)} chars")

        try:
            # Assemble enhanced context
            enhanced_context = await self.assemble_enhanced_context(user_input)

            if enhanced_context:
                # Inject context
                self.base.inject_context(enhanced_context)
                self.base.success_feedback("Query context enhanced")
            else:
                self.base.debug_log("No relevant context found")

            # Always allow the operation to proceed
            context.output.exit_success()

        except Exception as e:
            # Non-blocking error - log and continue
            self.base.warning_feedback(f"Context enhancement failed: {str(e)[:50]}")
            context.output.exit_success()


def main():
    """Main entry point for UserPromptSubmit hook."""
    # Create context using cchooks
    ctx = safe_create_context()

    # Verify it's UserPromptSubmit context
    if not isinstance(ctx, UserPromptSubmitContext):
        print(f"Error: Expected UserPromptSubmitContext, got {type(ctx)}", file=sys.stderr)
        sys.exit(1)

    # Create and run hook
    hook = UserPromptSubmitHook()

    try:
        # Run async processing
        asyncio.run(hook.process(ctx))
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Graceful failure - non-blocking
        print(f"⚠️  DevStream: UserPromptSubmit error", file=sys.stderr)
        ctx.output.exit_non_block(f"Hook error: {str(e)[:100]}")


if __name__ == "__main__":
    main()
