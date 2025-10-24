#!/usr/bin/env .devstream/bin/python
"""
DevStream TaskAwareQueryConstructor System

Context7-compliant query construction system for improving context injection relevance
from <30% to 70%+ through intelligent query analysis, semantic enhancement, and task-aware optimization.

Based on Context7 research from:
- Semantic Router for query routing and classification
- Context Engineering for relevance optimization and token budgeting
- Hybrid retrieval patterns for improved precision
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from collections import defaultdict, Counter

import structlog
from cachetools import LRUCache
import numpy as np

logger = structlog.get_logger()


class QueryIntent(str, Enum):
    """Types of query intents for context optimization."""
    CODE_GENERATION = "code_generation"          # Writing code snippets
    CODE_ANALYSIS = "code_analysis"             # Analyzing existing code
    DEBUGGING = "debugging"                     # Finding and fixing bugs
    DOCUMENTATION = "documentation"             # Writing docs/explanations
    REFACTORING = "refactoring"                 # Code improvement/restructuring
    TESTING = "testing"                         # Writing tests
    ARCHITECTURE = "architecture"               # System design decisions
    OPTIMIZATION = "optimization"               # Performance tuning
    INTEGRATION = "integration"                 # Connecting components
    DEPLOYMENT = "deployment"                   # Deployment procedures
    GENERAL = "general"                         # General assistance


class ContextScope(str, Enum):
    """Scope of context needed for the query."""
    IMMEDIATE = "immediate"                     # Current file/function only
    LOCAL = "local"                            # Related files in same module
    MODULE = "module"                          # Entire module/package
    PROJECT = "project"                        # Cross-project context
    EXTERNAL = "external"                       # External documentation/resources


@dataclass
class QueryAnalysis:
    """Analysis result for user query."""
    original_query: str
    cleaned_query: str
    intent: QueryIntent
    scope: ContextScope
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    file_patterns: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    relevance_threshold: float = 0.5
    expansion_terms: List[str] = field(default_factory=list)


@dataclass
class ContextConstruction:
    """Constructed context with metadata."""
    content: str
    sources: List[Dict[str, Any]]
    relevance_scores: List[float]
    total_tokens: int
    construction_time_ms: float
    query_analysis: QueryAnalysis
    optimization_applied: List[str] = field(default_factory=list)


@dataclass
class QueryMetrics:
    """Performance metrics for query construction."""
    total_queries: int = 0
    avg_relevance_score: float = 0.0
    avg_construction_time_ms: float = 0.0
    context_relevance_improvement: float = 0.0
    token_efficiency: float = 0.0
    intent_accuracy: float = 0.0


class TaskAwareQueryConstructor:
    """
    Advanced query construction system for DevStream memory operations.

    Features:
    - Intent-aware query analysis and classification
    - Semantic query expansion and enhancement
    - Context scope optimization based on task requirements
    - Token budget management and efficiency optimization
    - Relevance scoring and filtering
    - Context7-compliant construction patterns

    Target Performance:
    - Context relevance: 70%+ (from <30% baseline)
    - Construction time: <10ms per query
    - Token efficiency: 80%+ relevance per token
    - Intent classification accuracy: 85%+
    """

    def __init__(
        self,
        max_context_tokens: int = 2000,
        relevance_threshold: float = 0.5,
        enable_semantic_expansion: bool = True,
        enable_context_optimization: bool = True
    ):
        """
        Initialize TaskAwareQueryConstructor system.

        Args:
            max_context_tokens: Maximum tokens for constructed context
            relevance_threshold: Minimum relevance score for content inclusion
            enable_semantic_expansion: Whether to use semantic query expansion
            enable_context_optimization: Whether to apply context optimization techniques
        """
        self.max_context_tokens = max_context_tokens
        self.relevance_threshold = relevance_threshold
        self.enable_semantic_expansion = enable_semantic_expansion
        self.enable_context_optimization = enable_context_optimization

        # Caching for query analysis results
        self._analysis_cache: LRUCache = LRUCache(maxsize=500)
        self._construction_cache: LRUCache = LRUCache(maxsize=200)

        # Performance metrics
        self._metrics = QueryMetrics()
        self._construction_times: List[float] = []
        self._relevance_scores: List[float] = []

        # Query patterns and classification rules
        self._intent_patterns = self._initialize_intent_patterns()
        self._scope_patterns = self._initialize_scope_patterns()
        self._entity_extractors = self._initialize_entity_extractors()

        logger.info(
            "TaskAwareQueryConstructor initialized",
            max_context_tokens=max_context_tokens,
            relevance_threshold=relevance_threshold,
            semantic_expansion=enable_semantic_expansion,
            context_optimization=enable_context_optimization
        )

    def _initialize_intent_patterns(self) -> Dict[QueryIntent, List[str]]:
        """Initialize intent classification patterns."""
        return {
            QueryIntent.CODE_GENERATION: [
                r'\b(create|write|implement|build|develop|generate)\b.*\b(code|function|method|class|component)\b',
                r'\b(add|insert)\b.*\b(feature|functionality)\b',
                r'\b(help me)\b.*\b(code|program|implement)\b'
            ],
            QueryIntent.CODE_ANALYSIS: [
                r'\b(analyze|explain|understand|review|examine)\b.*\b(code|function|method)\b',
                r'\b(what|how|why)\b.*\b(this|the)\b.*\b(code|function|implementation)\b',
                r'\b(show me|tell me about)\b.*\b(code|logic|algorithm)\b'
            ],
            QueryIntent.DEBUGGING: [
                r'\b(debug|fix|error|issue|problem|bug)\b',
                r'\b(not working|broken|failed|crashed)\b',
                r'\b(why isn\'t|what\'s wrong|how to fix)\b'
            ],
            QueryIntent.DOCUMENTATION: [
                r'\b(document|explain|comment|describe)\b',
                r'\b(write|add)\b.*\b(docs|documentation|comments)\b',
                r'\b(what does|how to describe)\b'
            ],
            QueryIntent.REFACTORING: [
                r'\b(refactor|improve|optimize|clean up)\b.*\b(code|function)\b',
                r'\b(make better|restructure|reorganize)\b',
                r'\b(better way to|more efficient)\b'
            ],
            QueryIntent.TESTING: [
                r'\b(test|spec|unit test|integration test)\b',
                r'\b(write|create|add)\b.*\b(test|tests)\b',
                r'\b(how to test|testing strategy)\b'
            ],
            QueryIntent.ARCHITECTURE: [
                r'\b(architecture|design|structure|pattern)\b',
                r'\b(should I|how to) (design|structure|organize)\b',
                r'\b(best practice|pattern|approach)\b'
            ],
            QueryIntent.OPTIMIZATION: [
                r'\b(optimize|performance|speed|efficiency)\b',
                r'\b(make faster|improve performance|reduce latency)\b',
                r'\b(bottleneck|slow|inefficient)\b'
            ],
            QueryIntent.INTEGRATION: [
                r'\b(integrate|connect|combine|merge)\b',
                r'\b(hook up|link)\b.*\b(with|to)\b',
                r'\b(api|service|library)\b.*\b(integration|connection)\b'
            ],
            QueryIntent.DEPLOYMENT: [
                r'\b(deploy|deployment|release|publish)\b',
                r'\b(ship|production|staging)\b',
                r'\b(ci/cd|pipeline|build)\b'
            ]
        }

    def _initialize_scope_patterns(self) -> Dict[ContextScope, List[str]]:
        """Initialize context scope classification patterns."""
        return {
            ContextScope.IMMEDIATE: [
                r'\b(this|current)\s+(function|method|class|file)\b',
                r'\b(right here|in this place)\b'
            ],
            ContextScope.LOCAL: [
                r'\b(this\s+module|same\s+file|related\s+files)\b',
                r'\b(nearby|close|local)\b'
            ],
            ContextScope.MODULE: [
                r'\b(module|package|library)\b',
                r'\b(entire|whole)\s+(module|package)\b'
            ],
            ContextScope.PROJECT: [
                r'\b(project|codebase|application)\b',
                r'\b(across|throughout)\s+(project|codebase)\b'
            ],
            ContextScope.EXTERNAL: [
                r'\b(documentation|docs|external|outside)\b',
                r'\b(stack overflow|github|npm|pypi)\b'
            ]
        }

    def _initialize_entity_extractors(self) -> Dict[str, re.Pattern]:
        """Initialize entity extraction patterns."""
        return {
            'file_paths': re.compile(r'\b[\w\-\/\.]+\.(py|js|ts|jsx|tsx|go|rs|java|cpp|c|sql|md|json|yaml|yml)\b'),
            'function_names': re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*(?:\(\))?'),
            'class_names': re.compile(r'\b[A-Z][a-zA-Z0-9_]*(?=\s|\.|\(|\])'),
            'variables': re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*(?=\s*=|\s*\()'),
            'technologies': re.compile(r'\b(python|javascript|typescript|react|vue|angular|fastapi|django|flask|node|express|postgres|mysql|redis|docker|kubernetes)\b', re.IGNORECASE),
            'concepts': re.compile(r'\b(api|endpoint|database|cache|queue|auth|authentication|authorization|middleware|service|component|module|package|library)\b', re.IGNORECASE)
        }

    def _analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze user query to extract intent, scope, and entities.

        Args:
            query: Original user query string

        Returns:
            QueryAnalysis with extracted information
        """
        # Check cache first
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if query_hash in self._analysis_cache:
            return self._analysis_cache[query_hash]

        start_time = time.time()

        # Clean and normalize query
        cleaned_query = self._clean_query(query)

        # Extract entities
        entities = self._extract_entities(cleaned_query)
        keywords = self._extract_keywords(cleaned_query)
        file_patterns = self._extract_file_patterns(cleaned_query)

        # Classify intent
        intent, intent_confidence = self._classify_intent(cleaned_query)

        # Determine scope
        scope, scope_confidence = self._classify_scope(cleaned_query)

        # Generate expansion terms
        expansion_terms = self._generate_expansion_terms(cleaned_query, intent, entities)

        # Calculate overall confidence
        confidence_score = (intent_confidence + scope_confidence) / 2

        analysis = QueryAnalysis(
            original_query=query,
            cleaned_query=cleaned_query,
            intent=intent,
            scope=scope,
            entities=entities,
            keywords=keywords,
            file_patterns=file_patterns,
            confidence_score=confidence_score,
            relevance_threshold=self.relevance_threshold,
            expansion_terms=expansion_terms
        )

        # Cache result
        self._analysis_cache[query_hash] = analysis

        logger.debug(
            "Query analysis completed",
            intent=intent.value,
            scope=scope.value,
            confidence=confidence_score,
            entities_count=len(entities),
            keywords_count=len(keywords),
            analysis_time_ms=(time.time() - start_time) * 1000
        )

        return analysis

    def _clean_query(self, query: str) -> str:
        """Clean and normalize query string."""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())

        # Normalize common contractions
        contractions = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "doesn't": "does not",
            "didn't": "did not",
            "isn't": "is not",
            "aren't": "are not"
        }

        for contraction, expansion in contractions.items():
            query = query.replace(contraction, expansion)

        return query

    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities from query using patterns."""
        entities = []

        for entity_type, pattern in self._entity_extractors.items():
            matches = pattern.findall(query)
            entities.extend(matches)

        # Remove duplicates and filter
        entities = list(set([e for e in entities if len(e) > 1]))

        return entities

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract technical keywords from query."""
        # Technical keywords to track
        technical_keywords = {
            'function', 'method', 'class', 'variable', 'parameter', 'argument',
            'api', 'endpoint', 'route', 'request', 'response', 'service',
            'database', 'query', 'sql', 'table', 'schema', 'migration',
            'test', 'pytest', 'unit', 'integration', 'fixture', 'mock',
            'error', 'exception', 'bug', 'debug', 'fix', 'issue',
            'performance', 'optimize', 'speed', 'memory', 'cache',
            'deploy', 'build', 'release', 'production', 'staging',
            'security', 'auth', 'authentication', 'authorization', 'token',
            'frontend', 'backend', 'client', 'server', 'middleware'
        }

        words = re.findall(r'\b\w+\b', query.lower())
        return [word for word in words if word in technical_keywords]

    def _extract_file_patterns(self, query: str) -> List[str]:
        """Extract file patterns from query."""
        file_matches = self._entity_extractors['file_paths'].findall(query)
        return list(set(file_matches))

    def _classify_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Classify query intent based on patterns."""
        query_lower = query.lower()

        best_intent = QueryIntent.GENERAL
        best_score = 0.0

        for intent, patterns in self._intent_patterns.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1.0

            # Normalize score
            score = score / len(patterns) if patterns else 0.0

            if score > best_score:
                best_score = score
                best_intent = intent

        confidence = min(best_score, 1.0)
        return best_intent, confidence

    def _classify_scope(self, query: str) -> Tuple[ContextScope, float]:
        """Classify context scope based on patterns."""
        query_lower = query.lower()

        best_scope = ContextScope.PROJECT  # Default scope
        best_score = 0.0

        for scope, patterns in self._scope_patterns.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1.0

            # Normalize score
            score = score / len(patterns) if patterns else 0.0

            if score > best_score:
                best_score = score
                best_scope = scope

        confidence = min(best_score, 1.0)
        return best_scope, confidence

    def _generate_expansion_terms(self, query: str, intent: QueryIntent, entities: List[str]) -> List[str]:
        """Generate semantic expansion terms for the query."""
        if not self.enable_semantic_expansion:
            return []

        expansion_terms = []

        # Intent-based expansions
        intent_expansions = {
            QueryIntent.CODE_GENERATION: ['implement', 'create', 'build', 'develop', 'write'],
            QueryIntent.CODE_ANALYSIS: ['explain', 'understand', 'analyze', 'examine', 'review'],
            QueryIntent.DEBUGGING: ['error', 'issue', 'problem', 'bug', 'fix', 'debug'],
            QueryIntent.DOCUMENTATION: ['document', 'explain', 'comment', 'describe', 'guide'],
            QueryIntent.REFACTORING: ['improve', 'optimize', 'clean', 'restructure', 'reorganize'],
            QueryIntent.TESTING: ['test', 'spec', 'assert', 'verify', 'validate'],
            QueryIntent.ARCHITECTURE: ['design', 'structure', 'pattern', 'approach', 'strategy'],
            QueryIntent.OPTIMIZATION: ['performance', 'speed', 'efficiency', 'fast', 'optimize'],
            QueryIntent.INTEGRATION: ['connect', 'integrate', 'combine', 'link', 'merge'],
            QueryIntent.DEPLOYMENT: ['deploy', 'release', 'publish', 'ship', 'production']
        }

        if intent in intent_expansions:
            expansion_terms.extend(intent_expansions[intent])

        # Entity-based expansions
        for entity in entities:
            if '.' in entity:  # File path
                parts = entity.split('.')
                if len(parts) > 1:
                    expansion_terms.append(parts[0])  # Add filename without extension

            # Add common variations
            if entity.endswith('s'):  # Plural
                expansion_terms.append(entity[:-1])  # Singular
            elif not entity.endswith('s'):
                expansion_terms.append(entity + 's')  # Plural

        # Remove duplicates and original query terms
        query_words = set(query.lower().split())
        expansion_terms = [term for term in set(expansion_terms) if term not in query_words]

        return expansion_terms[:10]  # Limit to top 10

    def construct_enhanced_query(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        token_budget: Optional[int] = None
    ) -> ContextConstruction:
        """
        Construct enhanced query context with optimized relevance.

        Args:
            query: Original user query
            search_results: Raw search results from memory system
            token_budget: Optional token budget override

        Returns:
            ContextConstruction with enhanced context and metadata
        """
        start_time = time.time()

        # Use provided token budget or default
        token_budget = token_budget or self.max_context_tokens

        # Analyze query
        query_analysis = self._analyze_query(query)

        # Calculate relevance scores for each result
        scored_results = []
        for result in search_results:
            relevance_score = self._calculate_relevance_score(
                query_analysis, result
            )
            scored_results.append((result, relevance_score))

        # Sort by relevance
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Apply context optimization
        if self.enable_context_optimization:
            scored_results = self._optimize_context_selection(
                query_analysis, scored_results, token_budget
            )

        # Construct context content
        context_content, selected_sources, final_scores = self._build_context_content(
            query_analysis, scored_results, token_budget
        )

        # Calculate metrics
        construction_time = (time.time() - start_time) * 1000
        total_tokens = self._estimate_tokens(context_content)

        # Record metrics
        self._record_metrics(construction_time, final_scores)

        context_construction = ContextConstruction(
            content=context_content,
            sources=selected_sources,
            relevance_scores=final_scores,
            total_tokens=total_tokens,
            construction_time_ms=construction_time,
            query_analysis=query_analysis,
            optimization_applied=self._get_applied_optimizations(query_analysis)
        )

        logger.info(
            "Enhanced query context constructed",
            intent=query_analysis.intent.value,
            scope=query_analysis.scope.value,
            sources_count=len(selected_sources),
            tokens_used=total_tokens,
            avg_relevance=np.mean(final_scores) if final_scores else 0.0,
            construction_time_ms=construction_time
        )

        return context_construction

    def _calculate_relevance_score(
        self,
        query_analysis: QueryAnalysis,
        result: Dict[str, Any]
    ) -> float:
        """
        Calculate relevance score for a search result.

        Args:
            query_analysis: Analyzed query information
            result: Search result with content and metadata

        Returns:
            Relevance score (0.0-1.0)
        """
        content = result.get('content', '').lower()
        metadata = result.get('metadata', {})

        score = 0.0

        # Keyword matching (40% weight)
        keyword_matches = 0
        for keyword in query_analysis.keywords:
            if keyword in content:
                keyword_matches += 1

        if query_analysis.keywords:
            keyword_score = keyword_matches / len(query_analysis.keywords)
            score += keyword_score * 0.4

        # Entity matching (30% weight)
        entity_matches = 0
        for entity in query_analysis.entities:
            if entity.lower() in content:
                entity_matches += 1

        if query_analysis.entities:
            entity_score = entity_matches / len(query_analysis.entities)
            score += entity_score * 0.3

        # Semantic expansion matching (20% weight)
        expansion_matches = 0
        for expansion_term in query_analysis.expansion_terms:
            if expansion_term in content:
                expansion_matches += 1

        if query_analysis.expansion_terms:
            expansion_score = expansion_matches / len(query_analysis.expansion_terms)
            score += expansion_score * 0.2

        # Content type and scope relevance (10% weight)
        content_type = result.get('content_type', '')
        file_path = metadata.get('file_path', '')

        scope_score = 0.0

        # File pattern matching
        for pattern in query_analysis.file_patterns:
            if pattern in file_path:
                scope_score += 0.5

        # Content type relevance based on intent
        intent_type_mapping = {
            QueryIntent.CODE_GENERATION: ['code'],
            QueryIntent.CODE_ANALYSIS: ['code', 'documentation'],
            QueryIntent.DEBUGGING: ['code', 'error', 'log'],
            QueryIntent.DOCUMENTATION: ['documentation', 'code'],
            QueryIntent.TESTING: ['code', 'test'],
            QueryIntent.ARCHITECTURE: ['documentation', 'code']
        }

        if query_analysis.intent in intent_type_mapping:
            if content_type in intent_type_mapping[query_analysis.intent]:
                scope_score += 0.5

        score += min(scope_score, 1.0) * 0.1

        return min(score, 1.0)

    def _optimize_context_selection(
        self,
        query_analysis: QueryAnalysis,
        scored_results: List[Tuple[Dict[str, Any], float]],
        token_budget: int
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Apply context optimization techniques to improve relevance.

        Args:
            query_analysis: Analyzed query information
            scored_results: Results with relevance scores
            token_budget: Available token budget

        Returns:
            Optimized list of results with scores
        """
        # Filter by minimum relevance threshold
        filtered_results = [
            (result, score) for result, score in scored_results
            if score >= query_analysis.relevance_threshold
        ]

        # Diversity-based selection to avoid redundant content
        if query_analysis.scope in [ContextScope.MODULE, ContextScope.PROJECT]:
            filtered_results = self._apply_diversity_filtering(filtered_results)

        # Apply adaptive token allocation based on importance
        filtered_results = self._apply_adaptive_allocation(
            query_analysis, filtered_results, token_budget
        )

        return filtered_results

    def _apply_diversity_filtering(
        self,
        scored_results: List[Tuple[Dict[str, Any], float]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Apply diversity filtering to avoid redundant content.

        Args:
            scored_results: Results with relevance scores

        Returns:
            Diversified list of results
        """
        if not scored_results:
            return []

        diversified = []
        seen_content_hashes = set()

        for result, score in scored_results:
            content = result.get('content', '')
            content_hash = hashlib.md5(content[:200].encode()).hexdigest()

            # Skip if very similar to already seen content
            if content_hash not in seen_content_hashes:
                diversified.append((result, score))
                seen_content_hashes.add(content_hash)

                # Limit diversity set to maintain relevance
                if len(diversified) >= 10:
                    break

        return diversified

    def _apply_adaptive_allocation(
        self,
        query_analysis: QueryAnalysis,
        scored_results: List[Tuple[Dict[str, Any], float]],
        token_budget: int
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Apply adaptive token allocation based on result importance.

        Args:
            query_analysis: Analyzed query information
            scored_results: Results with relevance scores
            token_budget: Available token budget

        Returns:
            Results with adaptive allocation applied
        """
        if not scored_results:
            return []

        # Calculate token requirements
        allocated_results = []
        used_tokens = 0

        for result, score in scored_results:
            content = result.get('content', '')
            estimated_tokens = len(content.split()) * 1.3  # Rough token estimation

            # Prioritize high-relevance results
            if score >= 0.8:  # High relevance - full allocation
                if used_tokens + estimated_tokens <= token_budget:
                    allocated_results.append((result, score))
                    used_tokens += estimated_tokens
            elif score >= 0.5:  # Medium relevance - truncated allocation
                remaining_budget = token_budget - used_tokens
                if remaining_budget > 0:
                    # Truncate content to fit remaining budget
                    max_words = int(remaining_budget / 1.3)
                    words = content.split()[:max_words]
                    truncated_result = result.copy()
                    truncated_result['content'] = ' '.join(words)
                    allocated_results.append((truncated_result, score))
                    used_tokens += len(words) * 1.3
                    break
            else:  # Low relevance - skip
                continue

            if used_tokens >= token_budget * 0.95:  # Leave 5% buffer
                break

        return allocated_results

    def _build_context_content(
        self,
        query_analysis: QueryAnalysis,
        scored_results: List[Tuple[Dict[str, Any], float]],
        token_budget: int
    ) -> Tuple[str, List[Dict[str, Any]], List[float]]:
        """
        Build the final context content from selected results.

        Args:
            query_analysis: Analyzed query information
            scored_results: Selected results with scores
            token_budget: Available token budget

        Returns:
            Tuple of (context_content, selected_sources, relevance_scores)
        """
        context_parts = []
        selected_sources = []
        relevance_scores = []

        # Add query context header
        header = f"Query Context for: {query_analysis.original_query}\n"
        header += f"Intent: {query_analysis.intent.value}\n"
        header += f"Scope: {query_analysis.scope.value}\n\n"

        context_parts.append(header)
        header_tokens = len(header.split())

        used_tokens = header_tokens

        # Add relevant sources
        for i, (result, score) in enumerate(scored_results):
            if used_tokens >= token_budget * 0.9:
                break

            content = result.get('content', '')
            metadata = result.get('metadata', {})
            file_path = metadata.get('file_path', 'Unknown')
            content_type = result.get('content_type', 'unknown')

            # Format source entry
            source_header = f"Source {i+1}: {file_path} (relevance: {score:.2f})\n"
            source_content = f"Content: {content}\n\n"

            source_entry = source_header + source_content
            entry_tokens = len(source_entry.split())

            if used_tokens + entry_tokens <= token_budget:
                context_parts.append(source_entry)
                selected_sources.append({
                    'file_path': file_path,
                    'content_type': content_type,
                    'relevance_score': score,
                    'metadata': metadata
                })
                relevance_scores.append(score)
                used_tokens += entry_tokens
            else:
                # Truncate to fit remaining budget
                remaining_tokens = int((token_budget - used_tokens) / 1.3)
                if remaining_tokens > 10:  # Only add if meaningful content fits
                    words = content.split()[:remaining_tokens]
                    truncated_content = ' '.join(words)
                    truncated_entry = source_header + f"Content: {truncated_content}...\n\n"

                    context_parts.append(truncated_entry)
                    selected_sources.append({
                        'file_path': file_path,
                        'content_type': content_type,
                        'relevance_score': score,
                        'metadata': metadata,
                        'truncated': True
                    })
                    relevance_scores.append(score)
                    used_tokens += len(truncated_entry.split())
                break

        # Combine all parts
        final_context = ''.join(context_parts)

        return final_context, selected_sources, relevance_scores

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimation: ~1.3 tokens per word
        return int(len(text.split()) * 1.3)

    def _record_metrics(self, construction_time: float, relevance_scores: List[float]):
        """Record performance metrics."""
        self._construction_times.append(construction_time)
        self._relevance_scores.extend(relevance_scores)

        # Keep only recent metrics
        if len(self._construction_times) > 1000:
            self._construction_times = self._construction_times[-500:]
            self._relevance_scores = self._relevance_scores[-500:]

    def _get_applied_optimizations(self, query_analysis: QueryAnalysis) -> List[str]:
        """Get list of optimizations applied for this query."""
        optimizations = []

        if self.enable_semantic_expansion and query_analysis.expansion_terms:
            optimizations.append("semantic_expansion")

        if query_analysis.scope in [ContextScope.MODULE, ContextScope.PROJECT]:
            optimizations.append("diversity_filtering")

        optimizations.append("adaptive_allocation")
        optimizations.append("relevance_scoring")

        return optimizations

    def get_metrics(self) -> QueryMetrics:
        """
        Get current performance metrics.

        Returns:
            QueryMetrics with current performance data
        """
        if self._construction_times:
            self._metrics.avg_construction_time_ms = sum(self._construction_times) / len(self._construction_times)

        if self._relevance_scores:
            self._metrics.avg_relevance_score = sum(self._relevance_scores) / len(self._relevance_scores)

        return self._metrics

    def reset_metrics(self):
        """Reset performance metrics."""
        self._metrics = QueryMetrics()
        self._construction_times = []
        self._relevance_scores = []
        logger.info("Query constructor metrics reset")


# Global instance for import
_query_constructor = None


def get_task_aware_query_constructor(
    max_context_tokens: int = 2000,
    relevance_threshold: float = 0.5,
    enable_semantic_expansion: bool = True,
    enable_context_optimization: bool = True
) -> TaskAwareQueryConstructor:
    """
    Get or create global TaskAwareQueryConstructor instance.

    Args:
        max_context_tokens: Maximum tokens for constructed context
        relevance_threshold: Minimum relevance score for content inclusion
        enable_semantic_expansion: Whether to use semantic query expansion
        enable_context_optimization: Whether to apply context optimization techniques

    Returns:
        TaskAwareQueryConstructor instance
    """
    global _query_constructor

    if (_query_constructor is None or
        _query_constructor.max_context_tokens != max_context_tokens or
        _query_constructor.relevance_threshold != relevance_threshold):

        _query_constructor = TaskAwareQueryConstructor(
            max_context_tokens=max_context_tokens,
            relevance_threshold=relevance_threshold,
            enable_semantic_expansion=enable_semantic_expansion,
            enable_context_optimization=enable_context_optimization
        )

    return _query_constructor