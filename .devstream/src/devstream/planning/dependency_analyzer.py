"""
Advanced Dependency Analysis

Enhanced dependency detection with cycle prevention and graph validation.
Integrates with existing TaskDependencyGraph for consistency.
"""

import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, deque

from ..tasks.models import TaskDependencyGraph
from .models import AITaskSuggestion, TaskDependencySuggestion
from .protocols import ValidationError, CyclicDependencyError

logger = logging.getLogger(__name__)


class DependencyGraphValidator:
    """Advanced dependency graph validation with cycle detection."""

    def __init__(self):
        self.graph: Dict[str, List[str]] = defaultdict(list)
        self.reverse_graph: Dict[str, List[str]] = defaultdict(list)

    def build_graph(
        self,
        dependencies: List[TaskDependencySuggestion],
        tasks: List[AITaskSuggestion]
    ) -> None:
        """Build dependency graph from suggestions."""
        self.graph.clear()
        self.reverse_graph.clear()

        # Initialize with all task IDs
        task_ids = {task.id for task in tasks}
        for task_id in task_ids:
            self.graph[task_id] = []
            self.reverse_graph[task_id] = []

        # Add dependencies
        for dep in dependencies:
            if (dep.prerequisite_task_id in task_ids and
                dep.dependent_task_id in task_ids):
                self.graph[dep.prerequisite_task_id].append(dep.dependent_task_id)
                self.reverse_graph[dep.dependent_task_id].append(dep.prerequisite_task_id)

    def detect_cycles(self) -> List[List[str]]:
        """Detect all cycles in the dependency graph."""
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str) -> bool:
            """DFS with cycle detection."""
            if node in rec_stack:
                # Found cycle, extract it from path
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return True

            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.graph[node]:
                if dfs(neighbor):
                    # Continue to find all cycles
                    pass

            rec_stack.remove(node)
            path.pop()
            return False

        # Check all nodes for cycles
        for node in self.graph:
            if node not in visited:
                dfs(node)

        return cycles

    def get_strongly_connected_components(self) -> List[List[str]]:
        """Find strongly connected components (cycle groups)."""
        def kosaraju_dfs(graph: Dict[str, List[str]], node: str, visited: Set[str], stack: List[str]) -> None:
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    kosaraju_dfs(graph, neighbor, visited, stack)
            stack.append(node)

        # First DFS to get finish times
        visited = set()
        stack = []
        for node in self.graph:
            if node not in visited:
                kosaraju_dfs(self.graph, node, visited, stack)

        # Second DFS on transposed graph
        visited = set()
        components = []

        while stack:
            node = stack.pop()
            if node not in visited:
                component = []
                kosaraju_dfs(self.reverse_graph, node, visited, component)
                if len(component) > 1:  # Only cycles with multiple nodes
                    components.append(component)

        return components

    def topological_sort(self) -> List[str]:
        """Topological sort to validate DAG and get execution order."""
        in_degree = defaultdict(int)

        # Calculate in-degrees
        for node in self.graph:
            for neighbor in self.graph[node]:
                in_degree[neighbor] += 1

        # Initialize queue with nodes having 0 in-degree
        queue = deque([node for node in self.graph if in_degree[node] == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            for neighbor in self.graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # If result doesn't include all nodes, there's a cycle
        if len(result) != len(self.graph):
            missing_nodes = set(self.graph.keys()) - set(result)
            raise CyclicDependencyError(
                f"Cyclic dependencies detected. Nodes in cycles: {missing_nodes}"
            )

        return result

    def analyze_dependency_strength(
        self,
        dependencies: List[TaskDependencySuggestion],
        tasks: List[AITaskSuggestion]
    ) -> Dict[str, Any]:
        """Analyze dependency graph properties."""
        task_map = {task.id: task for task in tasks}

        # Calculate metrics
        total_deps = len(dependencies)
        blocking_deps = len([d for d in dependencies if d.dependency_type == "blocking"])
        soft_deps = len([d for d in dependencies if d.dependency_type == "soft"])

        # Find critical path
        critical_path = self._find_critical_path(dependencies, tasks)

        # Calculate parallelization potential
        parallel_groups = self._find_parallel_groups()

        return {
            "total_dependencies": total_deps,
            "blocking_dependencies": blocking_deps,
            "soft_dependencies": soft_deps,
            "dependency_ratio": total_deps / len(tasks) if tasks else 0,
            "critical_path_length": len(critical_path),
            "critical_path_duration": sum(
                task_map[task_id].estimated_minutes
                for task_id in critical_path
                if task_id in task_map
            ),
            "parallel_groups": len(parallel_groups),
            "max_parallelization": max(len(group) for group in parallel_groups) if parallel_groups else 1,
            "average_dependency_strength": sum(d.strength for d in dependencies) / len(dependencies) if dependencies else 0
        }

    def _find_critical_path(
        self,
        dependencies: List[TaskDependencySuggestion],
        tasks: List[AITaskSuggestion]
    ) -> List[str]:
        """Find critical path (longest path in terms of duration)."""
        task_map = {task.id: task for task in tasks}

        # Build adjacency list with weights (task durations)
        adj = defaultdict(list)
        for dep in dependencies:
            if dep.dependency_type == "blocking":  # Only consider blocking dependencies
                adj[dep.prerequisite_task_id].append(dep.dependent_task_id)

        # Find longest path using DFS
        memo = {}

        def longest_path_from(node: str) -> Tuple[int, List[str]]:
            if node in memo:
                return memo[node]

            if not adj[node]:  # No outgoing edges
                duration = task_map[node].estimated_minutes if node in task_map else 0
                memo[node] = (duration, [node])
                return memo[node]

            max_duration = 0
            best_path = [node]

            for neighbor in adj[node]:
                sub_duration, sub_path = longest_path_from(neighbor)
                total_duration = sub_duration

                if total_duration > max_duration:
                    max_duration = total_duration
                    best_path = [node] + sub_path

            current_duration = task_map[node].estimated_minutes if node in task_map else 0
            memo[node] = (max_duration + current_duration, best_path)
            return memo[node]

        # Find the starting node with maximum path
        max_duration = 0
        critical_path = []

        for task_id in task_map.keys():
            duration, path = longest_path_from(task_id)
            if duration > max_duration:
                max_duration = duration
                critical_path = path

        return critical_path

    def _find_parallel_groups(self) -> List[List[str]]:
        """Find groups of tasks that can be executed in parallel."""
        # Tasks with no dependencies can run in parallel
        no_deps = [node for node in self.graph if not self.reverse_graph[node]]

        # Group tasks by their depth level
        levels = defaultdict(list)
        visited = set()

        def assign_level(node: str, level: int) -> None:
            if node in visited:
                return
            visited.add(node)
            levels[level].append(node)

            for neighbor in self.graph[node]:
                assign_level(neighbor, level + 1)

        # Start from root nodes
        for root in no_deps:
            assign_level(root, 0)

        return [level_tasks for level_tasks in levels.values() if len(level_tasks) > 1]

    def optimize_dependencies(
        self,
        dependencies: List[TaskDependencySuggestion],
        tasks: List[AITaskSuggestion]
    ) -> List[TaskDependencySuggestion]:
        """Optimize dependencies by removing redundant ones."""
        # Build transitive closure
        self.build_graph(dependencies, tasks)

        # Find transitive dependencies (A->B->C implies A->C is redundant)
        optimized = []

        for dep in dependencies:
            prereq = dep.prerequisite_task_id
            dependent = dep.dependent_task_id

            # Check if there's an indirect path
            if not self._has_indirect_path(prereq, dependent, exclude_direct=True):
                optimized.append(dep)
            else:
                logger.info(f"Removing redundant dependency: {prereq} -> {dependent}")

        return optimized

    def _has_indirect_path(self, start: str, end: str, exclude_direct: bool = False) -> bool:
        """Check if there's an indirect path between two nodes."""
        visited = set()
        queue = deque([start])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            for neighbor in self.graph[current]:
                if neighbor == end:
                    # Direct connection
                    if exclude_direct and current == start:
                        continue
                    return True
                queue.append(neighbor)

        return False


class SmartDependencyDetector:
    """Smart dependency detection with pattern recognition."""

    def __init__(self):
        self.common_patterns = {
            "setup_before_impl": {
                "keywords": ["setup", "configure", "install"],
                "depends_on": ["implement", "create", "build"]
            },
            "test_after_impl": {
                "keywords": ["test", "validate", "verify"],
                "depends_on": ["implement", "create", "build", "add"]
            },
            "docs_after_impl": {
                "keywords": ["document", "write docs"],
                "depends_on": ["implement", "create", "build"]
            },
            "deploy_after_test": {
                "keywords": ["deploy", "release"],
                "depends_on": ["test", "validate"]
            }
        }

    def detect_implicit_dependencies(
        self,
        tasks: List[AITaskSuggestion]
    ) -> List[TaskDependencySuggestion]:
        """Detect implicit dependencies based on task patterns."""
        implicit_deps = []

        for i, task in enumerate(tasks):
            for j, other_task in enumerate(tasks):
                if i == j:
                    continue

                dependency = self._check_pattern_dependency(task, other_task)
                if dependency:
                    implicit_deps.append(dependency)

        return implicit_deps

    def _check_pattern_dependency(
        self,
        task: AITaskSuggestion,
        potential_prereq: AITaskSuggestion
    ) -> Optional[TaskDependencySuggestion]:
        """Check if task depends on potential_prereq based on patterns."""
        task_title_lower = task.title.lower()
        prereq_title_lower = potential_prereq.title.lower()

        for pattern_name, pattern in self.common_patterns.items():
            # Check if task matches pattern keywords
            task_matches = any(keyword in task_title_lower for keyword in pattern["keywords"])
            # Check if potential_prereq matches dependency keywords
            prereq_matches = any(keyword in prereq_title_lower for keyword in pattern["depends_on"])

            if task_matches and prereq_matches:
                # Additional semantic check
                if self._semantic_dependency_check(task, potential_prereq):
                    return TaskDependencySuggestion(
                        prerequisite_task_id=potential_prereq.id,
                        dependent_task_id=task.id,
                        dependency_type="soft",
                        strength=0.7,
                        reasoning=f"Pattern-based dependency: {pattern_name}",
                        confidence_score=0.75,
                        detected_from=["pattern_analysis", pattern_name]
                    )

        return None

    def _semantic_dependency_check(
        self,
        task: AITaskSuggestion,
        potential_prereq: AITaskSuggestion
    ) -> bool:
        """Additional semantic check for dependency validity."""
        # Check if tasks are related by shared keywords
        task_keywords = set(word.lower() for word in task.title.split() + task.description.split())
        prereq_keywords = set(word.lower() for word in potential_prereq.title.split() + potential_prereq.description.split())

        # Remove common words
        common_words = {"the", "and", "or", "for", "with", "to", "a", "an", "in", "on", "at"}
        task_keywords -= common_words
        prereq_keywords -= common_words

        # Check for keyword overlap
        overlap = task_keywords & prereq_keywords
        overlap_ratio = len(overlap) / min(len(task_keywords), len(prereq_keywords)) if task_keywords and prereq_keywords else 0

        return overlap_ratio > 0.2  # At least 20% keyword overlap


def enhance_dependency_analysis(
    ai_dependencies: List[TaskDependencySuggestion],
    tasks: List[AITaskSuggestion]
) -> Dict[str, Any]:
    """Enhanced dependency analysis with validation and optimization."""

    # Initialize components
    validator = DependencyGraphValidator()
    detector = SmartDependencyDetector()

    # Detect additional implicit dependencies
    implicit_deps = detector.detect_implicit_dependencies(tasks)

    # Combine AI and implicit dependencies
    all_dependencies = ai_dependencies + implicit_deps

    # Build and validate graph
    validator.build_graph(all_dependencies, tasks)

    # Detect cycles
    cycles = validator.detect_cycles()
    if cycles:
        logger.warning(f"Detected {len(cycles)} cycles in dependency graph")

    # Analyze graph properties
    analysis = validator.analyze_dependency_strength(all_dependencies, tasks)

    # Optimize dependencies
    optimized_deps = validator.optimize_dependencies(all_dependencies, tasks)

    # Get execution order
    try:
        execution_order = validator.topological_sort()
        analysis["execution_order"] = execution_order
        analysis["is_dag"] = True
    except CyclicDependencyError as e:
        analysis["execution_order"] = []
        analysis["is_dag"] = False
        analysis["cycle_error"] = str(e)

    return {
        "original_dependencies": len(ai_dependencies),
        "implicit_dependencies": len(implicit_deps),
        "total_dependencies": len(all_dependencies),
        "optimized_dependencies": len(optimized_deps),
        "cycles_detected": len(cycles),
        "cycles": cycles,
        "analysis": analysis,
        "optimized_deps": optimized_deps,
        "validation_passed": len(cycles) == 0
    }