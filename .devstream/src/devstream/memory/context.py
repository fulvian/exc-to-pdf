"""
Context Assembly System con Token Budget Management

Intelligent context assembly che combina relevant memories
con token budget management e optimization strategies.
"""

import logging
import time
from typing import List, Optional, Tuple

import tiktoken

from .models import (
    MemoryEntry,
    SearchQuery,
    ContextAssemblyResult,
    MemoryQueryResult,
    ContentType
)
from .search import HybridSearchEngine
from .exceptions import ContextError, TokenBudgetError

logger = logging.getLogger(__name__)


class ContextAssembler:
    """
    Intelligent context assembler con token budget management.

    Assembla context relevante dalla memory usando search results
    e optimizza per token budget constraints.
    """

    def __init__(self, search_engine: HybridSearchEngine, encoding_name: str = "cl100k_base"):
        """
        Initialize context assembler.

        Args:
            search_engine: Hybrid search engine instance
            encoding_name: tiktoken encoding name for token counting
        """
        self.search_engine = search_engine
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed, using fallback: {e}")
            # Fallback: rough estimation (4 chars per token average)
            return len(text) // 4

    async def assemble_context(
        self,
        query: str,
        token_budget: int,
        relevance_threshold: float = 0.3,
        max_memories: int = 20,
        strategy: str = "relevance"
    ) -> ContextAssemblyResult:
        """
        Assemble context from relevant memories con token budget.

        Args:
            query: Query for finding relevant context
            token_budget: Maximum tokens available for context
            relevance_threshold: Minimum relevance score for inclusion
            max_memories: Maximum number of memories to consider
            strategy: Assembly strategy ("relevance", "diversity", "chronological")

        Returns:
            Assembled context con metadata

        Raises:
            ContextError: Se il context assembly fallisce
            TokenBudgetError: Se il token budget è insufficiente
        """
        try:
            start_time = time.time()

            if token_budget <= 0:
                raise TokenBudgetError("Token budget must be positive")

            # Search for relevant memories
            search_query = SearchQuery(
                query_text=query,
                max_results=max_memories,
                min_relevance=relevance_threshold,
                semantic_weight=1.0,
                keyword_weight=0.7
            )

            search_results = await self.search_engine.search(search_query)

            if not search_results:
                logger.warning(f"No relevant memories found for query: {query[:100]}...")
                return self._empty_context_result(token_budget, strategy)

            # Apply assembly strategy
            selected_memories = self._apply_strategy(search_results, strategy)

            # Assemble context con token budget
            assembled_context, used_memories, truncated = self._assemble_with_budget(
                selected_memories, token_budget
            )

            assembly_time = (time.time() - start_time) * 1000

            result = ContextAssemblyResult(
                assembled_context=assembled_context,
                memory_entries=[result.memory_entry for result in used_memories],
                total_tokens=self.count_tokens(assembled_context),
                tokens_budget=token_budget,
                tokens_remaining=token_budget - self.count_tokens(assembled_context),
                relevance_threshold=relevance_threshold,
                truncated=truncated,
                assembly_strategy=strategy,
                assembly_time_ms=assembly_time
            )

            logger.info(
                f"Context assembled in {assembly_time:.2f}ms: "
                f"{len(used_memories)} memories, {result.total_tokens} tokens"
            )

            return result

        except Exception as e:
            logger.error(f"Context assembly failed: {e}")
            raise ContextError(f"Context assembly failed: {e}") from e

    def _apply_strategy(
        self,
        search_results: List[MemoryQueryResult],
        strategy: str
    ) -> List[MemoryQueryResult]:
        """
        Apply assembly strategy to order memories.

        Args:
            search_results: Search results to order
            strategy: Strategy to apply

        Returns:
            Ordered memory results
        """
        if strategy == "relevance":
            # Already sorted by relevance from search
            return search_results

        elif strategy == "diversity":
            # Diversify by content type and avoid similar content
            return self._diversify_memories(search_results)

        elif strategy == "chronological":
            # Sort by creation time (newest first)
            return sorted(
                search_results,
                key=lambda x: x.memory_entry.created_at,
                reverse=True
            )

        else:
            logger.warning(f"Unknown strategy '{strategy}', using relevance")
            return search_results

    def _diversify_memories(self, results: List[MemoryQueryResult]) -> List[MemoryQueryResult]:
        """
        Diversify memories by content type e similarity.

        Args:
            results: Search results to diversify

        Returns:
            Diversified results
        """
        diversified = []
        content_type_counts = {}

        for result in results:
            content_type = result.memory_entry.content_type
            type_count = content_type_counts.get(content_type, 0)

            # Prefer diverse content types (limit 3 per type initially)
            if type_count < 3:
                diversified.append(result)
                content_type_counts[content_type] = type_count + 1

        # Add remaining results if we have space
        remaining = [r for r in results if r not in diversified]
        diversified.extend(remaining)

        return diversified

    def _assemble_with_budget(
        self,
        memories: List[MemoryQueryResult],
        token_budget: int
    ) -> Tuple[str, List[MemoryQueryResult], bool]:
        """
        Assemble context respecting token budget.

        Args:
            memories: Ordered memory results
            token_budget: Available token budget

        Returns:
            Tuple of (assembled_context, used_memories, truncated)
        """
        context_parts = []
        used_memories = []
        current_tokens = 0
        truncated = False

        # Reserve tokens for formatting (separators, etc.)
        formatting_overhead = 100
        available_tokens = max(0, token_budget - formatting_overhead)

        for memory_result in memories:
            memory = memory_result.memory_entry

            # Format memory content
            formatted_content = self._format_memory_content(memory)
            content_tokens = self.count_tokens(formatted_content)

            # Check if adding this memory would exceed budget
            if current_tokens + content_tokens > available_tokens:
                if current_tokens == 0:
                    # First memory is too large, truncate it
                    truncated_content = self._truncate_content(
                        formatted_content, available_tokens
                    )
                    context_parts.append(truncated_content)
                    used_memories.append(memory_result)
                    truncated = True
                    break
                else:
                    # Stop adding memories
                    truncated = True
                    break

            # Add memory to context
            context_parts.append(formatted_content)
            used_memories.append(memory_result)
            current_tokens += content_tokens

        # Assemble final context
        if not context_parts:
            assembled_context = ""
        else:
            assembled_context = self._join_context_parts(context_parts)

        return assembled_context, used_memories, truncated

    def _format_memory_content(self, memory: MemoryEntry) -> str:
        """
        Format memory content for context inclusion.

        Args:
            memory: Memory entry to format

        Returns:
            Formatted content string
        """
        # Create header con metadata
        header_parts = []

        if memory.content_type:
            # Handle both enum and string content types
            if hasattr(memory.content_type, 'value'):
                header_parts.append(f"Type: {memory.content_type.value}")
            else:
                header_parts.append(f"Type: {memory.content_type}")

        if memory.task_id:
            header_parts.append(f"Task: {memory.task_id}")

        if memory.keywords:
            keywords_str = ", ".join(memory.keywords[:5])  # Limit keywords
            header_parts.append(f"Keywords: {keywords_str}")

        header = " | ".join(header_parts) if header_parts else "Memory"

        # Format content
        formatted = f"=== {header} ===\n{memory.content.strip()}\n"

        return formatted

    def _join_context_parts(self, parts: List[str]) -> str:
        """
        Join context parts con appropriate separators.

        Args:
            parts: List of formatted content parts

        Returns:
            Joined context string
        """
        if not parts:
            return ""

        # Join con double newlines for clear separation
        return "\n\n".join(parts)

    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """
        Truncate content to fit within token budget.

        Args:
            content: Content to truncate
            max_tokens: Maximum tokens allowed

        Returns:
            Truncated content
        """
        if max_tokens <= 0:
            return ""

        # Binary search for optimal truncation point
        left, right = 0, len(content)
        best_content = ""

        while left <= right:
            mid = (left + right) // 2
            candidate = content[:mid]
            candidate_tokens = self.count_tokens(candidate)

            if candidate_tokens <= max_tokens:
                best_content = candidate
                left = mid + 1
            else:
                right = mid - 1

        # Add truncation indicator
        if len(best_content) < len(content):
            best_content += "\n[... truncated ...]"

        return best_content

    def _empty_context_result(self, token_budget: int, strategy: str) -> ContextAssemblyResult:
        """Create empty context result when no memories found."""
        return ContextAssemblyResult(
            assembled_context="",
            memory_entries=[],
            total_tokens=0,
            tokens_budget=token_budget,
            tokens_remaining=token_budget,
            relevance_threshold=0.0,
            truncated=False,
            assembly_strategy=strategy,
            assembly_time_ms=0.0
        )

    async def assemble_task_context(
        self,
        task_id: str,
        token_budget: int,
        include_related: bool = True
    ) -> ContextAssemblyResult:
        """
        Assemble context specific for a task.

        Args:
            task_id: Task ID to assemble context for
            token_budget: Available token budget
            include_related: Whether to include related memories

        Returns:
            Task-specific context

        Raises:
            ContextError: Se il context assembly fallisce
        """
        try:
            # Build query for task-specific context
            query_parts = [f"task:{task_id}"]

            if include_related:
                # Add broader context query terms
                query_parts.extend(["implementation", "solution", "approach"])

            query = " ".join(query_parts)

            # Create search query con task filtering
            search_query = SearchQuery(
                query_text=query,
                max_results=15,
                task_id=task_id,  # Filter by specific task
                semantic_weight=0.8,
                keyword_weight=1.2  # Higher weight for keyword matching
            )

            search_results = await self.search_engine.search(search_query)

            if include_related and len(search_results) < 5:
                # Expand search to include phase/plan context
                expanded_query = SearchQuery(
                    query_text=query,
                    max_results=10,
                    min_relevance=0.2,  # Lower threshold for broader context
                    semantic_weight=1.0,
                    keyword_weight=0.8
                )
                expanded_results = await self.search_engine.search(expanded_query)
                search_results.extend(expanded_results)

                # Remove duplicates
                seen_ids = set()
                unique_results = []
                for result in search_results:
                    if result.memory_entry.id not in seen_ids:
                        unique_results.append(result)
                        seen_ids.add(result.memory_entry.id)
                search_results = unique_results

            # Assemble con chronological strategy per task context
            return await self.assemble_context(
                query=query,
                token_budget=token_budget,
                relevance_threshold=0.2,
                strategy="chronological"
            )

        except Exception as e:
            logger.error(f"Task context assembly failed: {e}")
            raise ContextError(f"Task context assembly failed: {e}") from e

    async def get_context_summary(self, context: str, max_tokens: int = 100) -> str:
        """
        Generate summary of assembled context.

        Args:
            context: Context to summarize
            max_tokens: Maximum tokens for summary

        Returns:
            Context summary

        Note: This is a placeholder - could be enhanced con LLM summarization
        """
        try:
            if not context or self.count_tokens(context) <= max_tokens:
                return context

            # Simple extractive summarization (first sentences)
            sentences = context.split('. ')
            summary_parts = []
            current_tokens = 0

            for sentence in sentences:
                sentence_tokens = self.count_tokens(sentence + '. ')
                if current_tokens + sentence_tokens > max_tokens:
                    break
                summary_parts.append(sentence)
                current_tokens += sentence_tokens

            summary = '. '.join(summary_parts)
            if summary and not summary.endswith('.'):
                summary += '.'

            return summary or context[:max_tokens * 4]  # Fallback character truncation

        except Exception as e:
            logger.warning(f"Context summarization failed: {e}")
            return context[:max_tokens * 4]  # Fallback