#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pydantic>=2.0.0",
#     "python-dotenv>=1.0.0",
#     "aiohttp>=3.8.0",
# ]
# ///

"""
DevStream Intelligent Context Injector - Advanced Context Assembly
Context7-compliant intelligent context injection con semantic search e relevance scoring.
"""

import json
import sys
import os
import asyncio
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass
from enum import Enum

# Import DevStream utilities
sys.path.append(str(Path(__file__).parent.parent / 'utils'))
from common import DevStreamHookBase, get_project_context
from logger import get_devstream_logger
from mcp_client import get_mcp_client

class ContextType(Enum):
    """Context injection types."""
    IMPLEMENTATION = "implementation"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"
    PATTERNS = "patterns"
    EXAMPLES = "examples"
    DOCUMENTATION = "documentation"
    ERROR_RESOLUTION = "error_resolution"

class RelevanceLevel(Enum):
    """Context relevance levels."""
    CRITICAL = "critical"      # 0.8-1.0
    HIGH = "high"             # 0.6-0.8
    MEDIUM = "medium"         # 0.4-0.6
    LOW = "low"              # 0.2-0.4
    MINIMAL = "minimal"      # 0.0-0.2

@dataclass
class ContextFragment:
    """Individual context fragment."""
    content: str
    content_type: str
    relevance_score: float
    keywords: List[str]
    source: str
    timestamp: datetime
    token_cost: int

@dataclass
class ContextAssembly:
    """Complete context assembly."""
    fragments: List[ContextFragment]
    total_tokens: int
    relevance_distribution: Dict[RelevanceLevel, int]
    context_types: Set[ContextType]
    assembly_confidence: float

class IntelligentContextInjector(DevStreamHookBase):
    """
    Intelligent Context Injector per advanced context assembly da DevStream memory.
    Implementa Context7-validated patterns per semantic context injection.
    """

    def __init__(self):
        super().__init__('intelligent_context_injector')
        self.structured_logger = get_devstream_logger('context_injector')
        self.mcp_client = get_mcp_client()
        self.start_time = time.time()

        # Context injection configuration
        self.max_total_tokens = 2000
        self.min_relevance_threshold = 0.4
        self.max_fragments = 8
        self.context_window_hours = 24

        # Scoring weights
        self.recency_weight = 0.3
        self.relevance_weight = 0.5
        self.diversity_weight = 0.2

        # Cache for efficiency
        self.context_cache = {}
        self.cache_ttl = 300  # 5 minutes

    async def inject_intelligent_context(
        self,
        query_context: Dict[str, Any],
        tool_context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Inject intelligent context based on query e tool context.

        Args:
            query_context: User query or prompt context
            tool_context: Optional tool execution context

        Returns:
            Assembled context string or None
        """
        self.structured_logger.log_hook_start(query_context, {
            "phase": "intelligent_context_injection"
        })

        try:
            # Generate semantic query from context
            semantic_query = await self.generate_semantic_query(query_context, tool_context)

            # Check cache first
            cache_key = self.generate_cache_key(semantic_query)
            cached_context = self.get_cached_context(cache_key)

            if cached_context:
                self.logger.debug("Using cached context")
                return cached_context

            # Retrieve relevant memories
            relevant_memories = await self.retrieve_relevant_memories(semantic_query)

            if not relevant_memories:
                self.logger.debug("No relevant memories found for context injection")
                return None

            # Score and rank memories
            scored_fragments = await self.score_and_rank_memories(
                relevant_memories,
                semantic_query,
                query_context
            )

            # Assemble optimal context
            context_assembly = await self.assemble_optimal_context(
                scored_fragments,
                semantic_query
            )

            # Generate final context
            if context_assembly and context_assembly.fragments:
                final_context = await self.generate_final_context(context_assembly)

                # Cache result
                self.cache_context(cache_key, final_context)

                # Log successful injection
                self.logger.info(f"Injected intelligent context: "
                               f"{len(context_assembly.fragments)} fragments, "
                               f"{context_assembly.total_tokens} tokens")

                # Store injection event
                await self.store_injection_event(context_assembly, semantic_query)

                return final_context

            return None

        except Exception as e:
            self.structured_logger.log_hook_error(e, {
                "query_context": query_context,
                "tool_context": tool_context
            })
            raise

    async def generate_semantic_query(
        self,
        query_context: Dict[str, Any],
        tool_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate semantic query from context.

        Args:
            query_context: Query context
            tool_context: Tool context

        Returns:
            Semantic search query
        """
        query_parts = []

        # Extract from user query
        user_query = query_context.get('user_input', '')
        if user_query:
            # Extract key terms
            key_terms = self.extract_key_terms(user_query)
            query_parts.extend(key_terms[:5])  # Top 5 terms

        # Extract from tool context
        if tool_context:
            tool_name = tool_context.get('tool_name', '')
            if tool_name:
                query_parts.append(tool_name)

            # Add tool-specific context
            if tool_name in ['Edit', 'Write', 'MultiEdit']:
                file_path = tool_context.get('tool_input', {}).get('file_path', '')
                if file_path:
                    path_obj = Path(file_path)
                    query_parts.append(path_obj.suffix.lstrip('.'))
                    query_parts.extend(path_obj.parts[-2:])

            elif tool_name == 'Bash':
                command = tool_context.get('tool_input', {}).get('command', '')
                command_words = command.split()[:3]
                query_parts.extend(command_words)

        # Add context-specific terms
        if 'devstream' in str(query_context).lower():
            query_parts.extend(['devstream', 'memory', 'task'])
        if 'hook' in str(query_context).lower():
            query_parts.extend(['hook', 'automation', 'claude'])

        # Clean and deduplicate
        query_parts = list(set([part.strip() for part in query_parts if part.strip()]))

        return ' '.join(query_parts[:10])  # Limit query length

    def extract_key_terms(self, text: str) -> List[str]:
        """
        Extract key terms from text.

        Args:
            text: Input text

        Returns:
            List of key terms
        """
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())

        # Filter out common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

        key_terms = [word for word in words if word not in stop_words and len(word) > 2]

        # Return most frequent terms
        term_counts = {}
        for term in key_terms:
            term_counts[term] = term_counts.get(term, 0) + 1

        return sorted(term_counts.keys(), key=lambda x: term_counts[x], reverse=True)

    async def retrieve_relevant_memories(self, semantic_query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories from DevStream.

        Args:
            semantic_query: Semantic search query

        Returns:
            List of relevant memories
        """
        try:
            # Primary search
            primary_results = await self.mcp_client.search_memory(
                query=semantic_query,
                limit=15
            )

            memories = []

            if primary_results and primary_results.get('content'):
                memories.extend(self.parse_memory_results(primary_results, 'primary'))

            # Secondary search with broader terms
            if len(memories) < 5:
                broad_query = ' '.join(semantic_query.split()[:3])  # Use first 3 terms
                secondary_results = await self.mcp_client.search_memory(
                    query=broad_query,
                    limit=10
                )

                if secondary_results and secondary_results.get('content'):
                    secondary_memories = self.parse_memory_results(secondary_results, 'secondary')
                    # Add non-duplicate memories
                    for memory in secondary_memories:
                        if not any(self.memories_similar(memory, existing) for existing in memories):
                            memories.append(memory)

            return memories[:20]  # Limit total memories

        except Exception as e:
            self.logger.warning(f"Failed to retrieve relevant memories: {e}")
            return []

    def parse_memory_results(self, results: Dict[str, Any], source: str) -> List[Dict[str, Any]]:
        """
        Parse memory search results.

        Args:
            results: MCP search results
            source: Result source

        Returns:
            List of parsed memories
        """
        memories = []

        content = results.get('content', [])
        if content and len(content) > 0:
            text_content = content[0].get('text', '')

            if text_content:
                # Simple parsing - in production would parse structured results
                memory_blocks = text_content.split('\n\n')

                for i, block in enumerate(memory_blocks):
                    if len(block.strip()) > 50:  # Minimum content length
                        memories.append({
                            'id': f"{source}_{i}",
                            'content': block.strip(),
                            'source': source,
                            'content_type': self.infer_content_type(block),
                            'timestamp': datetime.now().isoformat(),
                            'keywords': self.extract_keywords_from_content(block)
                        })

        return memories

    def memories_similar(self, memory1: Dict[str, Any], memory2: Dict[str, Any]) -> bool:
        """
        Check if two memories are similar.

        Args:
            memory1: First memory
            memory2: Second memory

        Returns:
            True if memories are similar
        """
        content1 = memory1.get('content', '').lower()
        content2 = memory2.get('content', '').lower()

        # Simple similarity check
        if len(content1) == 0 or len(content2) == 0:
            return False

        # Check for substantial overlap
        words1 = set(content1.split())
        words2 = set(content2.split())

        if len(words1) == 0 or len(words2) == 0:
            return False

        overlap = len(words1.intersection(words2))
        min_words = min(len(words1), len(words2))

        return (overlap / min_words) > 0.7  # 70% overlap threshold

    def infer_content_type(self, content: str) -> str:
        """
        Infer content type from content.

        Args:
            content: Content text

        Returns:
            Inferred content type
        """
        content_lower = content.lower()

        if any(keyword in content_lower for keyword in ['error', 'failed', 'exception']):
            return 'error'
        elif any(keyword in content_lower for keyword in ['implement', 'create', 'build']):
            return 'implementation'
        elif any(keyword in content_lower for keyword in ['test', 'verify', 'check']):
            return 'testing'
        elif any(keyword in content_lower for keyword in ['hook', 'automation']):
            return 'automation'
        elif any(keyword in content_lower for keyword in ['context', 'memory']):
            return 'context'
        else:
            return 'general'

    def extract_keywords_from_content(self, content: str) -> List[str]:
        """
        Extract keywords from content.

        Args:
            content: Content text

        Returns:
            List of keywords
        """
        return self.extract_key_terms(content)[:5]

    async def score_and_rank_memories(
        self,
        memories: List[Dict[str, Any]],
        semantic_query: str,
        query_context: Dict[str, Any]
    ) -> List[ContextFragment]:
        """
        Score and rank memories for relevance.

        Args:
            memories: Raw memories
            semantic_query: Semantic query
            query_context: Query context

        Returns:
            List of scored context fragments
        """
        fragments = []
        query_terms = set(semantic_query.lower().split())

        for memory in memories:
            content = memory.get('content', '')
            content_type = memory.get('content_type', 'general')
            keywords = memory.get('keywords', [])

            # Calculate relevance score
            relevance_score = await self.calculate_relevance_score(
                content,
                keywords,
                query_terms,
                content_type,
                memory.get('timestamp')
            )

            if relevance_score >= self.min_relevance_threshold:
                fragments.append(ContextFragment(
                    content=content,
                    content_type=content_type,
                    relevance_score=relevance_score,
                    keywords=keywords,
                    source=memory.get('source', 'unknown'),
                    timestamp=datetime.fromisoformat(memory.get('timestamp', datetime.now().isoformat())),
                    token_cost=self.estimate_token_cost(content)
                ))

        # Sort by relevance score
        fragments.sort(key=lambda x: x.relevance_score, reverse=True)

        return fragments

    async def calculate_relevance_score(
        self,
        content: str,
        keywords: List[str],
        query_terms: Set[str],
        content_type: str,
        timestamp_str: Optional[str]
    ) -> float:
        """
        Calculate relevance score for content.

        Args:
            content: Content text
            keywords: Content keywords
            query_terms: Query terms
            content_type: Content type
            timestamp_str: Timestamp string

        Returns:
            Relevance score (0.0-1.0)
        """
        score = 0.0

        # Term matching score
        content_terms = set(content.lower().split())
        keyword_terms = set([kw.lower() for kw in keywords])
        all_terms = content_terms.union(keyword_terms)

        if all_terms:
            term_overlap = len(query_terms.intersection(all_terms))
            max_possible = min(len(query_terms), len(all_terms))
            if max_possible > 0:
                term_score = term_overlap / max_possible
                score += term_score * self.relevance_weight

        # Content type bonus
        type_bonuses = {
            'implementation': 0.2,
            'error': 0.15,
            'automation': 0.2,
            'context': 0.1,
            'testing': 0.1
        }
        score += type_bonuses.get(content_type, 0.0)

        # Recency score
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                hours_old = (datetime.now() - timestamp).total_seconds() / 3600

                if hours_old <= 1:
                    recency_score = 1.0
                elif hours_old <= 6:
                    recency_score = 0.8
                elif hours_old <= 24:
                    recency_score = 0.5
                else:
                    recency_score = 0.2

                score += recency_score * self.recency_weight
            except:
                pass

        # Content quality score
        content_length = len(content)
        if content_length > 100:  # Substantial content
            score += 0.1
        if content_length > 500:  # Rich content
            score += 0.1

        return min(score, 1.0)

    def estimate_token_cost(self, content: str) -> int:
        """
        Estimate token cost for content.

        Args:
            content: Content text

        Returns:
            Estimated token count
        """
        return len(content) // 4  # Rough estimation

    async def assemble_optimal_context(
        self,
        scored_fragments: List[ContextFragment],
        semantic_query: str
    ) -> Optional[ContextAssembly]:
        """
        Assemble optimal context within token budget.

        Args:
            scored_fragments: Scored context fragments
            semantic_query: Semantic query

        Returns:
            Optimal context assembly
        """
        if not scored_fragments:
            return None

        selected_fragments = []
        total_tokens = 0
        context_types = set()
        relevance_dist = {level: 0 for level in RelevanceLevel}

        # Greedy selection with diversity
        type_counts = {}

        for fragment in scored_fragments:
            # Check token budget
            if total_tokens + fragment.token_cost > self.max_total_tokens:
                break

            # Check fragment limit
            if len(selected_fragments) >= self.max_fragments:
                break

            # Promote diversity - limit same content types
            content_type = fragment.content_type
            if type_counts.get(content_type, 0) >= 3:  # Max 3 per type
                continue

            # Select fragment
            selected_fragments.append(fragment)
            total_tokens += fragment.token_cost
            context_types.add(ContextType(fragment.content_type) if fragment.content_type in [t.value for t in ContextType] else ContextType.IMPLEMENTATION)

            # Update type counts
            type_counts[content_type] = type_counts.get(content_type, 0) + 1

            # Update relevance distribution
            if fragment.relevance_score >= 0.8:
                relevance_dist[RelevanceLevel.CRITICAL] += 1
            elif fragment.relevance_score >= 0.6:
                relevance_dist[RelevanceLevel.HIGH] += 1
            elif fragment.relevance_score >= 0.4:
                relevance_dist[RelevanceLevel.MEDIUM] += 1
            else:
                relevance_dist[RelevanceLevel.LOW] += 1

        if selected_fragments:
            # Calculate assembly confidence
            avg_relevance = sum(f.relevance_score for f in selected_fragments) / len(selected_fragments)
            diversity_score = len(context_types) / max(len(ContextType), 1)
            assembly_confidence = (avg_relevance * 0.7) + (diversity_score * 0.3)

            return ContextAssembly(
                fragments=selected_fragments,
                total_tokens=total_tokens,
                relevance_distribution=relevance_dist,
                context_types=context_types,
                assembly_confidence=assembly_confidence
            )

        return None

    async def generate_final_context(self, assembly: ContextAssembly) -> str:
        """
        Generate final context string.

        Args:
            assembly: Context assembly

        Returns:
            Final context string
        """
        context_parts = [
            "🧠 DevStream Intelligent Context",
            f"📊 Relevance: {assembly.assembly_confidence:.2f} | Tokens: {assembly.total_tokens}",
            ""
        ]

        # Add fragments by relevance level
        critical_fragments = [f for f in assembly.fragments if f.relevance_score >= 0.8]
        high_fragments = [f for f in assembly.fragments if 0.6 <= f.relevance_score < 0.8]
        medium_fragments = [f for f in assembly.fragments if 0.4 <= f.relevance_score < 0.6]

        if critical_fragments:
            context_parts.append("🎯 Critical Context:")
            for fragment in critical_fragments:
                context_parts.append(f"   {fragment.content[:200]}...")
                context_parts.append("")

        if high_fragments:
            context_parts.append("📋 High Relevance:")
            for fragment in high_fragments:
                context_parts.append(f"   {fragment.content[:150]}...")
                context_parts.append("")

        if medium_fragments:
            context_parts.append("💡 Additional Context:")
            for fragment in medium_fragments:
                context_parts.append(f"   {fragment.content[:100]}...")
                context_parts.append("")

        # Add footer
        context_parts.extend([
            "---",
            "🚀 This context was intelligently assembled from DevStream memory to enhance your understanding."
        ])

        return '\n'.join(context_parts)

    def generate_cache_key(self, semantic_query: str) -> str:
        """
        Generate cache key for query.

        Args:
            semantic_query: Semantic query

        Returns:
            Cache key
        """
        import hashlib
        return hashlib.md5(semantic_query.encode()).hexdigest()

    def get_cached_context(self, cache_key: str) -> Optional[str]:
        """
        Get cached context.

        Args:
            cache_key: Cache key

        Returns:
            Cached context or None
        """
        if cache_key in self.context_cache:
            cached_item = self.context_cache[cache_key]
            if (datetime.now() - cached_item['timestamp']).total_seconds() < self.cache_ttl:
                return cached_item['context']

        return None

    def cache_context(self, cache_key: str, context: str) -> None:
        """
        Cache context.

        Args:
            cache_key: Cache key
            context: Context to cache
        """
        self.context_cache[cache_key] = {
            'context': context,
            'timestamp': datetime.now()
        }

        # Clean old cache entries
        if len(self.context_cache) > 50:  # Limit cache size
            oldest_key = min(
                self.context_cache.keys(),
                key=lambda k: self.context_cache[k]['timestamp']
            )
            del self.context_cache[oldest_key]

    async def store_injection_event(
        self,
        assembly: ContextAssembly,
        semantic_query: str
    ) -> None:
        """
        Store context injection event.

        Args:
            assembly: Context assembly
            semantic_query: Semantic query
        """
        event_content = (
            f"INTELLIGENT CONTEXT INJECTION: Query '{semantic_query}' processed. "
            f"Assembled {len(assembly.fragments)} fragments, "
            f"{assembly.total_tokens} tokens, "
            f"confidence {assembly.assembly_confidence:.2f}"
        )

        await self.mcp_client.store_memory(
            content=event_content,
            content_type="context",
            keywords=["context-injection", "intelligent", "automation", "devstream"]
        )

# Convenience functions for hook integration

async def inject_context_for_query(query_context: Dict[str, Any]) -> Optional[str]:
    """
    Inject intelligent context for user query.

    Args:
        query_context: Query context

    Returns:
        Assembled context or None
    """
    injector = IntelligentContextInjector()
    return await injector.inject_intelligent_context(query_context)

async def inject_context_for_tool(tool_context: Dict[str, Any]) -> Optional[str]:
    """
    Inject intelligent context for tool execution.

    Args:
        tool_context: Tool context

    Returns:
        Assembled context or None
    """
    injector = IntelligentContextInjector()
    return await injector.inject_intelligent_context({}, tool_context)

async def main():
    """Main execution for testing."""
    injector = IntelligentContextInjector()

    # Test context injection
    test_query = {
        'user_input': 'implement hook system for automated task management'
    }

    print("🧠 Testing intelligent context injection...")

    context = await injector.inject_intelligent_context(test_query)

    if context:
        print("✅ Context injected successfully:")
        print(context[:500] + "..." if len(context) > 500 else context)
    else:
        print("ℹ️  No relevant context found for injection")

if __name__ == "__main__":
    asyncio.run(main())