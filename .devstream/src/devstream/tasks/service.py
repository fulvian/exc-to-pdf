"""
Task Management Service Layer - High-Level Operations

Provides high-level task management operations that combine multiple engine
operations into cohesive workflows. This layer handles complex business logic
and provides simplified interfaces for common task management scenarios.

Key Features:
- Workflow orchestration for common task patterns
- Intelligent task planning and breakdown
- Progress monitoring and reporting
- Integration points for AI planning and memory systems
- Transaction management for complex operations

Based on Context7 research patterns for service layer architecture.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union, TYPE_CHECKING
from uuid import UUID, uuid4

from devstream.tasks.engine import TaskEngine, TaskEngineConfig
from devstream.tasks.models import (
    InterventionPlan, Phase, MicroTask, TaskDependencyGraph,
    TaskStatus, TaskPriority, TaskType, TaskComplexity
)
from devstream.tasks.repository import TaskRepository
from devstream.memory.storage import MemoryStorage
from devstream.exceptions import DevStreamError

if TYPE_CHECKING:
    from devstream.planning import (
        OllamaPlanner, TaskBreakdownRequest, PlanGenerationRequest,
        EstimationRequest, PlanningResult, AIPlannerConfig,
        PlanningMode, EstimationApproach, ContextSource
    )


class TaskServiceError(DevStreamError):
    """Task service specific errors"""
    pass


class WorkflowError(TaskServiceError):
    """Workflow execution errors"""
    pass


class PlanningError(TaskServiceError):
    """Task planning errors"""
    pass


class TaskWorkflowTemplate:
    """Template for common task workflows"""

    def __init__(
        self,
        name: str,
        description: str,
        phases: List[Dict[str, Any]],
        default_tasks: List[Dict[str, Any]]
    ):
        self.name = name
        self.description = description
        self.phases = phases
        self.default_tasks = default_tasks


class TaskService:
    """
    High-level task management service

    Provides simplified interfaces for complex task management operations
    while maintaining full control over the underlying task engine.
    """

    def __init__(
        self,
        engine: Optional[TaskEngine] = None,
        repository: Optional[TaskRepository] = None,
        memory_storage: Optional[MemoryStorage] = None,
        config: Optional[TaskEngineConfig] = None,
        ai_planner: Optional["OllamaPlanner"] = None
    ):
        self.engine = engine or TaskEngine(repository, memory_storage, config)
        self.repository = self.engine.repository
        self.memory_storage = self.engine.memory_storage
        self.ai_planner = ai_planner

        # Workflow templates
        self._workflow_templates: Dict[str, TaskWorkflowTemplate] = {}
        self._initialize_default_templates()

        # Operation statistics
        self._operation_stats = {
            "plans_created": 0,
            "tasks_created": 0,
            "tasks_completed": 0,
            "workflows_executed": 0
        }

    # ============================================================================
    # QUICK START OPERATIONS
    # ============================================================================

    async def create_simple_plan(
        self,
        title: str,
        objective: str,
        tasks: List[Dict[str, Any]],
        estimated_hours: Optional[float] = None
    ) -> Dict[str, Any]:
        """Create a simple plan with tasks in a single phase"""

        # Calculate estimated hours if not provided
        if not estimated_hours:
            task_minutes = sum(task.get("estimated_minutes", 5) for task in tasks)
            estimated_hours = task_minutes / 60.0

        # Create intervention plan
        plan = await self.engine.create_intervention_plan(
            title=title,
            description=f"Simple plan for: {objective}",
            objective=objective,
            estimated_hours=estimated_hours,
            category="development"
        )

        # Create single phase
        phase = await self.engine.create_phase(
            plan_id=plan.id,
            name="Implementation",
            description="Implementation of all tasks",
            objective=objective,
            order_index=1,
            estimated_hours=estimated_hours
        )

        # Create tasks
        created_tasks = []
        for i, task_data in enumerate(tasks):
            task = await self.engine.create_micro_task(
                phase_id=phase.id,
                title=task_data["title"],
                description=task_data.get("description", task_data["title"]),
                task_type=TaskType(task_data.get("task_type", "implementation")),
                complexity=TaskComplexity(task_data.get("complexity", "simple")),
                estimated_minutes=task_data.get("estimated_minutes", 5),
                priority=TaskPriority(task_data.get("priority", "medium")),
                keywords=task_data.get("keywords", []),
                acceptance_criteria=task_data.get("acceptance_criteria", ["Task completed successfully"])
            )
            created_tasks.append(task)

        self._operation_stats["plans_created"] += 1
        self._operation_stats["tasks_created"] += len(created_tasks)

        return {
            "plan": plan,
            "phase": phase,
            "tasks": created_tasks,
            "ready_tasks": await self.engine.get_ready_tasks(plan.id)
        }

    async def create_from_template(
        self,
        template_name: str,
        title: str,
        objective: str,
        customizations: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a plan from a predefined template"""

        template = self._workflow_templates.get(template_name)
        if not template:
            raise WorkflowError(f"Template '{template_name}' not found")

        customizations = customizations or {}

        # Create plan
        plan = await self.engine.create_intervention_plan(
            title=title,
            description=customizations.get("description", template.description),
            objective=objective,
            estimated_hours=customizations.get("estimated_hours", 8.0),
            category=customizations.get("category", "development")
        )

        # Create phases
        created_phases = []
        for phase_data in template.phases:
            phase = await self.engine.create_phase(
                plan_id=plan.id,
                name=phase_data["name"],
                description=phase_data["description"],
                objective=phase_data["objective"],
                order_index=phase_data["order_index"],
                estimated_hours=phase_data.get("estimated_hours", 2.0)
            )
            created_phases.append(phase)

        # Create default tasks
        created_tasks = []
        for task_data in template.default_tasks:
            # Find appropriate phase
            phase = next(
                (p for p in created_phases if p.order_index == task_data["phase_index"]),
                created_phases[0]
            )

            task = await self.engine.create_micro_task(
                phase_id=phase.id,
                title=task_data["title"],
                description=task_data["description"],
                task_type=TaskType(task_data["task_type"]),
                complexity=TaskComplexity(task_data.get("complexity", "simple")),
                estimated_minutes=task_data.get("estimated_minutes", 5),
                priority=TaskPriority(task_data.get("priority", "medium")),
                keywords=task_data.get("keywords", [])
            )
            created_tasks.append(task)

        self._operation_stats["plans_created"] += 1
        self._operation_stats["tasks_created"] += len(created_tasks)
        self._operation_stats["workflows_executed"] += 1

        return {
            "plan": plan,
            "phases": created_phases,
            "tasks": created_tasks,
            "template_used": template_name
        }

    # ============================================================================
    # WORKFLOW EXECUTION
    # ============================================================================

    async def execute_next_ready_task(
        self,
        plan_id: UUID,
        assignee: Optional[str] = None,
        auto_complete: bool = False
    ) -> Optional[MicroTask]:
        """Execute the next ready task in a plan"""

        # Get ready tasks
        ready_tasks = await self.engine.get_ready_tasks(plan_id)

        if not ready_tasks:
            return None

        # Select highest priority task
        next_task = ready_tasks[0]  # Already sorted by priority

        # Start task
        started_task = await self.engine.start_task(next_task.id, assignee)

        # Auto-complete if requested (for testing/simulation)
        if auto_complete:
            await asyncio.sleep(0.1)  # Simulate work
            completed_task = await self.engine.complete_task(
                started_task.id,
                actual_minutes=started_task.estimated_minutes,
                completion_notes="Auto-completed by task service"
            )
            self._operation_stats["tasks_completed"] += 1
            return completed_task

        return started_task

    async def execute_task_sequence(
        self,
        task_ids: List[UUID],
        assignee: Optional[str] = None,
        parallel: bool = False
    ) -> List[MicroTask]:
        """Execute a sequence of tasks"""

        if parallel:
            # Execute tasks in parallel
            tasks = await asyncio.gather(*[
                self._execute_single_task(task_id, assignee)
                for task_id in task_ids
            ])
        else:
            # Execute tasks sequentially
            tasks = []
            for task_id in task_ids:
                task = await self._execute_single_task(task_id, assignee)
                tasks.append(task)

        self._operation_stats["tasks_completed"] += len(tasks)
        return tasks

    async def auto_execute_plan(
        self,
        plan_id: UUID,
        max_parallel_tasks: int = 3,
        assignee: Optional[str] = None,
        simulation_mode: bool = True
    ) -> Dict[str, Any]:
        """Automatically execute an entire plan"""

        execution_log = []
        active_tasks = set()

        while True:
            # Get ready tasks
            ready_tasks = await self.engine.get_ready_tasks(plan_id)

            # Filter out tasks that are already active
            available_tasks = [task for task in ready_tasks if task.id not in active_tasks]

            if not available_tasks:
                # Wait for active tasks to complete
                if active_tasks:
                    await asyncio.sleep(1)
                    continue
                else:
                    # No more tasks to execute
                    break

            # Start as many tasks as possible up to limit
            tasks_to_start = available_tasks[:max_parallel_tasks - len(active_tasks)]

            for task in tasks_to_start:
                try:
                    started_task = await self.engine.start_task(task.id, assignee)
                    active_tasks.add(task.id)

                    execution_log.append({
                        "action": "started",
                        "task": started_task,
                        "timestamp": datetime.now()
                    })

                    # In simulation mode, auto-complete after a delay
                    if simulation_mode:
                        asyncio.create_task(
                            self._simulate_task_completion(task.id, active_tasks, execution_log)
                        )

                except Exception as e:
                    execution_log.append({
                        "action": "error",
                        "task_id": task.id,
                        "error": str(e),
                        "timestamp": datetime.now()
                    })

            # Short delay before next iteration
            await asyncio.sleep(0.5)

        # Get final progress
        progress = await self.engine.get_plan_progress(plan_id)

        return {
            "execution_log": execution_log,
            "final_progress": progress,
            "tasks_executed": len([log for log in execution_log if log["action"] == "completed"])
        }

    # ============================================================================
    # PLANNING AND ANALYSIS
    # ============================================================================

    async def analyze_plan_feasibility(self, plan_id: UUID) -> Dict[str, Any]:
        """Analyze plan feasibility and identify potential issues"""

        # Get plan progress
        progress = await self.engine.get_plan_progress(plan_id)

        # Get execution metrics
        metrics = await self.engine.get_task_execution_metrics(plan_id)

        # Analyze task distribution
        task_types = {}
        complexity_distribution = {}
        for task in progress["ready_tasks"] + progress["active_tasks"]:
            task_types[task.task_type.value] = task_types.get(task.task_type.value, 0) + 1
            complexity_distribution[task.complexity.value] = complexity_distribution.get(task.complexity.value, 0) + 1

        # Identify issues
        issues = []

        # Check for bottlenecks
        if metrics["bottleneck_tasks"]:
            issues.append({
                "type": "bottleneck",
                "severity": "high",
                "description": f"Found {len(metrics['bottleneck_tasks'])} bottleneck tasks",
                "details": metrics["bottleneck_tasks"]
            })

        # Check for dependency violations
        if metrics["dependency_violations"]:
            issues.append({
                "type": "dependency_violation",
                "severity": "critical",
                "description": f"Found {len(metrics['dependency_violations'])} dependency violations",
                "details": metrics["dependency_violations"]
            })

        # Check for imbalanced complexity
        complex_tasks = complexity_distribution.get("complex", 0)
        total_tasks = sum(complexity_distribution.values())
        if total_tasks > 0 and complex_tasks / total_tasks > 0.5:
            issues.append({
                "type": "complexity_imbalance",
                "severity": "medium",
                "description": f"Too many complex tasks ({complex_tasks}/{total_tasks})",
                "recommendation": "Consider breaking down complex tasks"
            })

        return {
            "plan": progress["plan"],
            "feasibility_score": self._calculate_feasibility_score(progress, metrics, issues),
            "task_distribution": task_types,
            "complexity_distribution": complexity_distribution,
            "estimated_completion": metrics["completion_estimates"]["estimated_completion"],
            "issues": issues,
            "recommendations": self._generate_recommendations(issues, progress)
        }

    # ============================================================================
    # AI-POWERED PLANNING METHODS
    # ============================================================================

    async def create_ai_powered_plan(
        self,
        title: str,
        description: str,
        objectives: List[str],
        context: Optional[str] = None,
        planning_mode: "PlanningMode" = "detailed",
        estimation_approach: "EstimationApproach" = "realistic"
    ) -> Dict[str, Any]:
        """
        Create an intervention plan using AI-powered task breakdown and dependency analysis.

        This method leverages the OllamaPlanner for intelligent task decomposition,
        dependency detection, and time estimation based on Context7 research.
        """
        if not self.ai_planner:
            raise PlanningError("AI Planner not available. Please configure OllamaPlanner.")

        try:
            # Create plan generation request
            plan_request = PlanGenerationRequest(
                title=title,
                description=description,
                objectives=objectives,
                planning_mode=planning_mode,
                estimation_approach=estimation_approach,
                provided_context=context,
                include_memory_context=True
            )

            # Generate AI-powered plan
            planning_result = await self.ai_planner.generate_plan(plan_request)

            # Create intervention plan from AI results
            total_hours = planning_result.total_estimated_minutes / 60.0

            plan = await self.engine.create_intervention_plan(
                title=title,
                description=description,
                objective="\n".join(objectives),
                estimated_hours=total_hours,
                category="ai_generated"
            )

            # Create phases based on AI suggestions
            phases_created = []
            if planning_result.suggested_phases:
                for i, phase_name in enumerate(planning_result.suggested_phases):
                    # Group tasks by suggested phase
                    phase_tasks = [
                        task for task in planning_result.suggested_tasks
                        if task.suggested_phase == phase_name
                    ]

                    phase_duration = sum(task.estimated_minutes for task in phase_tasks) / 60.0

                    phase = await self.engine.create_phase(
                        plan_id=plan.id,
                        name=phase_name,
                        description=f"AI-generated phase: {phase_name}",
                        objective=phase_name,
                        order_index=i + 1,
                        estimated_hours=phase_duration
                    )
                    phases_created.append({"phase": phase, "tasks": phase_tasks})
            else:
                # Create single phase if no phases suggested
                phase = await self.engine.create_phase(
                    plan_id=plan.id,
                    name="Implementation",
                    description="AI-generated task implementation",
                    objective="\n".join(objectives),
                    order_index=1,
                    estimated_hours=total_hours
                )
                phases_created.append({"phase": phase, "tasks": planning_result.suggested_tasks})

            # Create tasks from AI suggestions
            all_created_tasks = []
            task_id_mapping = {}  # AI task ID -> real task ID

            for phase_info in phases_created:
                phase = phase_info["phase"]
                ai_tasks = phase_info["tasks"]

                for ai_task in ai_tasks:
                    # Map AI task type to our enum
                    task_type = self._map_ai_task_type(ai_task.task_type)
                    complexity = self._map_ai_complexity(ai_task.complexity_score)
                    priority = self._map_ai_priority(ai_task.priority_score)

                    created_task = await self.engine.create_micro_task(
                        phase_id=phase.id,
                        title=ai_task.title,
                        description=ai_task.description,
                        task_type=task_type,
                        complexity=complexity,
                        estimated_minutes=ai_task.estimated_minutes,
                        priority=priority,
                        keywords=[],
                        acceptance_criteria=[f"AI Confidence: {ai_task.confidence_score:.2f}"]
                    )

                    all_created_tasks.append(created_task)
                    task_id_mapping[ai_task.id] = created_task.id

            # Add dependencies from AI analysis
            dependencies_created = []
            if planning_result.suggested_dependencies:
                dependency_graph = TaskDependencyGraph()

                for ai_dep in planning_result.suggested_dependencies:
                    prereq_id = task_id_mapping.get(ai_dep.prerequisite_task_id)
                    dep_id = task_id_mapping.get(ai_dep.dependent_task_id)

                    if prereq_id and dep_id:
                        await self.engine.add_task_dependency(prereq_id, dep_id)
                        dependency_graph.add_dependency(prereq_id, dep_id)
                        dependencies_created.append({
                            "prerequisite": prereq_id,
                            "dependent": dep_id,
                            "reasoning": ai_dep.reasoning,
                            "confidence": ai_dep.confidence_score
                        })

            # Update statistics
            self._operation_stats["plans_created"] += 1
            self._operation_stats["tasks_created"] += len(all_created_tasks)

            return {
                "plan": plan,
                "phases": [info["phase"] for info in phases_created],
                "tasks": all_created_tasks,
                "dependencies": dependencies_created,
                "ai_analysis": {
                    "planning_confidence": planning_result.planning_confidence,
                    "completeness_score": planning_result.completeness_score,
                    "total_estimated_minutes": planning_result.total_estimated_minutes,
                    "average_complexity": planning_result.average_complexity,
                    "reasoning": planning_result.planning_reasoning
                },
                "task_count": len(all_created_tasks),
                "dependency_count": len(dependencies_created)
            }

        except Exception as e:
            raise PlanningError(f"AI-powered plan creation failed: {e}") from e

    async def ai_task_breakdown(
        self,
        objective: str,
        context: Optional[str] = None,
        max_tasks: int = 15,
        target_duration: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Break down a complex objective into micro-tasks using AI analysis.

        Returns task suggestions that can be used with create_simple_plan()
        or integrated into existing plans.
        """
        if not self.ai_planner:
            raise PlanningError("AI Planner not available. Please configure OllamaPlanner.")

        try:
            # Create breakdown request
            breakdown_request = TaskBreakdownRequest(
                objective=objective,
                context=context,
                max_tasks=max_tasks,
                max_task_duration_minutes=target_duration,
                include_memory_context=True
            )

            # Generate task breakdown
            planning_result = await self.ai_planner.generate_task_breakdown(breakdown_request)

            # Convert AI tasks to simple format
            task_suggestions = []
            for ai_task in planning_result.suggested_tasks:
                task_data = {
                    "title": ai_task.title,
                    "description": ai_task.description,
                    "estimated_minutes": ai_task.estimated_minutes,
                    "task_type": self._map_ai_task_type(ai_task.task_type).value,
                    "complexity": self._map_ai_complexity(ai_task.complexity_score).value,
                    "priority": self._map_ai_priority(ai_task.priority_score).value,
                    "ai_confidence": ai_task.confidence_score,
                    "ai_reasoning": ai_task.reasoning
                }
                task_suggestions.append(task_data)

            return task_suggestions

        except Exception as e:
            raise PlanningError(f"AI task breakdown failed: {e}") from e

    async def estimate_task_complexity(
        self,
        task_title: str,
        task_description: str,
        project_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get AI-powered complexity and time estimation for a task.

        Uses research-backed estimation methodologies including
        Lizard complexity analysis and SD-Metrics velocity patterns.
        """
        if not self.ai_planner:
            raise PlanningError("AI Planner not available. Please configure OllamaPlanner.")

        try:
            # Create estimation request
            estimation_request = EstimationRequest(
                task_title=task_title,
                task_description=task_description,
                project_context=project_context
            )

            # Get AI estimation
            estimation = await self.ai_planner.estimate_complexity(estimation_request)

            return {
                "estimated_minutes": estimation.estimated_minutes,
                "complexity_score": estimation.complexity_score,
                "uncertainty_factor": estimation.uncertainty_factor,
                "confidence_score": estimation.confidence_score,
                "analysis_factors": estimation.analysis_factors,
                "risk_factors": estimation.risk_factors,
                "assumptions": estimation.assumptions,
                "reasoning": estimation.estimation_reasoning,
                "approach_used": estimation.approach_used.value
            }

        except Exception as e:
            raise PlanningError(f"AI complexity estimation failed: {e}") from e

    def _map_ai_task_type(self, ai_task_type: str) -> TaskType:
        """Map AI task type to our TaskType enum."""
        type_mapping = {
            "implementation": TaskType.IMPLEMENTATION,
            "testing": TaskType.TESTING,
            "documentation": TaskType.DOCUMENTATION,
            "research": TaskType.RESEARCH,
            "planning": TaskType.PLANNING,
            "review": TaskType.TESTING,  # Map review to testing
            "setup": TaskType.IMPLEMENTATION,  # Map setup to implementation
            "deployment": TaskType.IMPLEMENTATION,
            "analysis": TaskType.RESEARCH
        }
        return type_mapping.get(ai_task_type.lower(), TaskType.IMPLEMENTATION)

    def _map_ai_complexity(self, complexity_score: int) -> TaskComplexity:
        """Map AI complexity score (1-10) to our TaskComplexity enum."""
        if complexity_score <= 3:
            return TaskComplexity.SIMPLE
        elif complexity_score <= 6:
            return TaskComplexity.MODERATE
        elif complexity_score <= 8:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.VERY_COMPLEX

    def _map_ai_priority(self, priority_score: float) -> TaskPriority:
        """Map AI priority score (0.0-1.0) to our TaskPriority enum."""
        if priority_score <= 0.3:
            return TaskPriority.LOW
        elif priority_score <= 0.6:
            return TaskPriority.MEDIUM
        elif priority_score <= 0.8:
            return TaskPriority.HIGH
        else:
            return TaskPriority.URGENT

    async def suggest_task_breakdown(
        self,
        task: MicroTask,
        target_complexity: TaskComplexity = TaskComplexity.SIMPLE
    ) -> List[Dict[str, Any]]:
        """Suggest how to break down a complex task"""

        if task.complexity.value in ["trivial", "simple"]:
            return []  # Task doesn't need breakdown

        # Generate breakdown suggestions based on task type
        suggestions = []

        if task.task_type == TaskType.IMPLEMENTATION:
            suggestions.extend([
                {
                    "title": f"Design {task.title.replace('Implement', '').strip()}",
                    "description": f"Design the architecture and approach for {task.description}",
                    "task_type": "research",
                    "complexity": "simple",
                    "estimated_minutes": 3
                },
                {
                    "title": f"Implement core {task.title.replace('Implement', '').strip()}",
                    "description": f"Implement the main functionality for {task.description}",
                    "task_type": "implementation",
                    "complexity": "moderate",
                    "estimated_minutes": 6
                },
                {
                    "title": f"Test {task.title.replace('Implement', '').strip()}",
                    "description": f"Test the implementation of {task.description}",
                    "task_type": "testing",
                    "complexity": "simple",
                    "estimated_minutes": 4
                }
            ])

        elif task.task_type == TaskType.RESEARCH:
            suggestions.extend([
                {
                    "title": f"Research {task.title.replace('Research', '').strip()} - Initial",
                    "description": f"Initial research phase for {task.description}",
                    "task_type": "research",
                    "complexity": "simple",
                    "estimated_minutes": 4
                },
                {
                    "title": f"Research {task.title.replace('Research', '').strip()} - Deep dive",
                    "description": f"Deep dive research for {task.description}",
                    "task_type": "research",
                    "complexity": "moderate",
                    "estimated_minutes": 6
                }
            ])

        # Add dependencies between suggested tasks
        for i in range(1, len(suggestions)):
            suggestions[i]["depends_on_previous"] = True

        return suggestions

    async def optimize_task_dependencies(self, plan_id: UUID) -> Dict[str, Any]:
        """Optimize task dependencies for better execution flow"""

        # Build current dependency graph
        graph = await self.engine.build_dependency_graph(plan_id)

        # Analyze current dependencies
        optimization_suggestions = []

        # Find unnecessary dependencies
        for task_id, task in graph.tasks.items():
            for dep_id in task.depends_on:
                # Check if dependency is really necessary
                dep_task = graph.tasks.get(dep_id)
                if dep_task:
                    # If tasks have no overlapping keywords/resources, dependency might be unnecessary
                    task_keywords = set(task.keywords)
                    dep_keywords = set(dep_task.keywords)

                    if not task_keywords.intersection(dep_keywords):
                        optimization_suggestions.append({
                            "type": "remove_dependency",
                            "task_id": task_id,
                            "task_title": task.title,
                            "dependency_id": dep_id,
                            "dependency_title": dep_task.title,
                            "reason": "No resource overlap detected"
                        })

        # Find missing dependencies
        for task_id, task in graph.tasks.items():
            for other_id, other_task in graph.tasks.items():
                if task_id != other_id and other_id not in task.depends_on:
                    # Check if dependency should exist
                    task_keywords = set(task.keywords)
                    other_keywords = set(other_task.keywords)

                    overlap = task_keywords.intersection(other_keywords)
                    if len(overlap) >= 2:  # Significant overlap
                        optimization_suggestions.append({
                            "type": "add_dependency",
                            "task_id": task_id,
                            "task_title": task.title,
                            "suggested_dependency_id": other_id,
                            "suggested_dependency_title": other_task.title,
                            "reason": f"Shared keywords: {', '.join(overlap)}"
                        })

        return {
            "current_dependencies": len([(t, d) for t in graph.tasks.values() for d in t.depends_on]),
            "optimization_suggestions": optimization_suggestions,
            "critical_path_length": len(graph.get_critical_path()),
            "potential_parallelism": self._calculate_potential_parallelism(graph)
        }

    # ============================================================================
    # MONITORING AND REPORTING
    # ============================================================================

    async def get_execution_dashboard(self, plan_id: UUID) -> Dict[str, Any]:
        """Get comprehensive execution dashboard data"""

        # Get basic progress
        progress = await self.engine.get_plan_progress(plan_id)

        # Get metrics
        metrics = await self.engine.get_task_execution_metrics(plan_id)

        # Calculate velocity
        velocity = await self._calculate_velocity(plan_id)

        # Get recent activity
        recent_activity = await self._get_recent_activity(plan_id)

        return {
            "plan_overview": {
                "title": progress["plan"].title,
                "status": progress["plan"].status.value,
                "overall_progress": progress["overall_progress"],
                "estimated_completion": metrics["completion_estimates"]["estimated_completion"]
            },
            "task_summary": {
                "total": sum(progress["task_statistics"].values()),
                "completed": progress["task_statistics"].get("completed", 0),
                "active": progress["task_statistics"].get("active", 0),
                "pending": progress["task_statistics"].get("pending", 0),
                "ready": len(progress["ready_tasks"])
            },
            "velocity": velocity,
            "bottlenecks": metrics["bottleneck_tasks"],
            "recent_activity": recent_activity,
            "next_actions": await self._suggest_next_actions(plan_id),
            "health_score": await self._calculate_health_score(plan_id)
        }

    async def generate_progress_report(self, plan_id: UUID) -> str:
        """Generate a human-readable progress report"""

        dashboard = await self.get_execution_dashboard(plan_id)

        report = f"""
# Progress Report: {dashboard['plan_overview']['title']}

## Executive Summary
- **Overall Progress**: {dashboard['plan_overview']['overall_progress']:.1f}%
- **Status**: {dashboard['plan_overview']['status'].title()}
- **Health Score**: {dashboard['health_score']:.1f}/10

## Task Summary
- **Total Tasks**: {dashboard['task_summary']['total']}
- **Completed**: {dashboard['task_summary']['completed']}
- **In Progress**: {dashboard['task_summary']['active']}
- **Ready to Start**: {dashboard['task_summary']['ready']}
- **Pending**: {dashboard['task_summary']['pending']}

## Performance Metrics
- **Velocity**: {dashboard['velocity'].get('tasks_per_hour', 0):.1f} tasks/hour
- **Average Task Duration**: {dashboard['velocity'].get('avg_duration_minutes', 0):.1f} minutes

## Bottlenecks ({len(dashboard['bottlenecks'])})
"""

        for bottleneck in dashboard['bottlenecks'][:3]:  # Top 3
            report += f"- **{bottleneck['task_title']}**: {bottleneck['dependent_task_count']} dependent tasks\n"

        report += f"""
## Next Actions
"""
        for action in dashboard['next_actions'][:5]:  # Top 5
            report += f"- {action['action']}: {action['description']}\n"

        if dashboard['plan_overview']['estimated_completion']:
            completion_date = dashboard['plan_overview']['estimated_completion']
            report += f"\n**Estimated Completion**: {completion_date.strftime('%Y-%m-%d %H:%M')}\n"

        return report

    # ============================================================================
    # TEMPLATE MANAGEMENT
    # ============================================================================

    def register_workflow_template(self, template: TaskWorkflowTemplate) -> None:
        """Register a new workflow template"""
        self._workflow_templates[template.name] = template

    def get_available_templates(self) -> List[str]:
        """Get list of available workflow templates"""
        return list(self._workflow_templates.keys())

    def get_template_details(self, template_name: str) -> Optional[TaskWorkflowTemplate]:
        """Get details of a specific template"""
        return self._workflow_templates.get(template_name)

    # ============================================================================
    # STATISTICS AND ANALYTICS
    # ============================================================================

    def get_service_statistics(self) -> Dict[str, Any]:
        """Get service operation statistics"""
        return {
            **self._operation_stats,
            "templates_available": len(self._workflow_templates),
            "uptime": "Not implemented",  # Would track service uptime
            "cache_hit_rate": "Not implemented"  # Would track caching metrics
        }

    async def get_historical_metrics(
        self,
        plan_id: UUID,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get historical metrics for a plan"""

        # This would typically query historical data
        # For now, return current snapshot
        current_progress = await self.engine.get_plan_progress(plan_id)

        return {
            "plan_id": str(plan_id),
            "period_days": days,
            "current_snapshot": current_progress,
            "historical_data": "Not implemented"  # Would contain time series data
        }

    # ============================================================================
    # HELPER METHODS
    # ============================================================================

    def _initialize_default_templates(self) -> None:
        """Initialize default workflow templates"""

        # Software Development Template
        dev_template = TaskWorkflowTemplate(
            name="software_development",
            description="Standard software development workflow",
            phases=[
                {
                    "name": "Planning & Design",
                    "description": "Requirements analysis and design",
                    "objective": "Define clear requirements and architecture",
                    "order_index": 1,
                    "estimated_hours": 3.0
                },
                {
                    "name": "Implementation",
                    "description": "Core development work",
                    "objective": "Implement the designed solution",
                    "order_index": 2,
                    "estimated_hours": 8.0
                },
                {
                    "name": "Testing & Validation",
                    "description": "Testing and quality assurance",
                    "objective": "Ensure solution quality and reliability",
                    "order_index": 3,
                    "estimated_hours": 3.0
                }
            ],
            default_tasks=[
                {
                    "title": "Analyze requirements",
                    "description": "Analyze and document requirements",
                    "task_type": "research",
                    "phase_index": 1,
                    "estimated_minutes": 8,
                    "complexity": "moderate"
                },
                {
                    "title": "Design architecture",
                    "description": "Design system architecture and components",
                    "task_type": "documentation",
                    "phase_index": 1,
                    "estimated_minutes": 10,
                    "complexity": "complex"
                },
                {
                    "title": "Implement core functionality",
                    "description": "Implement main system functionality",
                    "task_type": "implementation",
                    "phase_index": 2,
                    "estimated_minutes": 10,
                    "complexity": "complex"
                },
                {
                    "title": "Write unit tests",
                    "description": "Create comprehensive unit tests",
                    "task_type": "testing",
                    "phase_index": 3,
                    "estimated_minutes": 8,
                    "complexity": "moderate"
                }
            ]
        )

        self.register_workflow_template(dev_template)

        # Research Template
        research_template = TaskWorkflowTemplate(
            name="research_project",
            description="Research and analysis workflow",
            phases=[
                {
                    "name": "Literature Review",
                    "description": "Review existing research and documentation",
                    "objective": "Understand current state of knowledge",
                    "order_index": 1,
                    "estimated_hours": 4.0
                },
                {
                    "name": "Investigation",
                    "description": "Conduct detailed investigation",
                    "objective": "Gather and analyze new information",
                    "order_index": 2,
                    "estimated_hours": 6.0
                },
                {
                    "name": "Documentation",
                    "description": "Document findings and recommendations",
                    "objective": "Create actionable documentation",
                    "order_index": 3,
                    "estimated_hours": 2.0
                }
            ],
            default_tasks=[
                {
                    "title": "Review existing documentation",
                    "description": "Review and catalog existing research",
                    "task_type": "research",
                    "phase_index": 1,
                    "estimated_minutes": 10,
                    "complexity": "complex"
                },
                {
                    "title": "Conduct primary research",
                    "description": "Gather new information through investigation",
                    "task_type": "research",
                    "phase_index": 2,
                    "estimated_minutes": 10,
                    "complexity": "complex"
                },
                {
                    "title": "Document findings",
                    "description": "Create comprehensive documentation of results",
                    "task_type": "documentation",
                    "phase_index": 3,
                    "estimated_minutes": 8,
                    "complexity": "moderate"
                }
            ]
        )

        self.register_workflow_template(research_template)

    async def _execute_single_task(self, task_id: UUID, assignee: Optional[str]) -> MicroTask:
        """Execute a single task (helper method)"""
        started_task = await self.engine.start_task(task_id, assignee)

        # Simulate work (for testing/demo)
        await asyncio.sleep(0.1)

        completed_task = await self.engine.complete_task(
            started_task.id,
            actual_minutes=started_task.estimated_minutes,
            completion_notes="Completed by task service"
        )

        return completed_task

    async def _simulate_task_completion(
        self,
        task_id: UUID,
        active_tasks: Set[UUID],
        execution_log: List[Dict[str, Any]]
    ) -> None:
        """Simulate task completion (for auto-execution)"""

        # Get task for estimated duration
        task = await self.engine.get_micro_task(task_id)
        if not task:
            return

        # Simulate work time (scaled down for demo)
        await asyncio.sleep(task.estimated_minutes * 0.1)  # 0.1 seconds per minute

        try:
            completed_task = await self.engine.complete_task(
                task_id,
                actual_minutes=task.estimated_minutes,
                completion_notes="Auto-completed by simulation"
            )

            active_tasks.discard(task_id)
            self._operation_stats["tasks_completed"] += 1

            execution_log.append({
                "action": "completed",
                "task": completed_task,
                "timestamp": datetime.now()
            })

        except Exception as e:
            active_tasks.discard(task_id)
            execution_log.append({
                "action": "completion_error",
                "task_id": task_id,
                "error": str(e),
                "timestamp": datetime.now()
            })

    def _calculate_feasibility_score(
        self,
        progress: Dict[str, Any],
        metrics: Dict[str, Any],
        issues: List[Dict[str, Any]]
    ) -> float:
        """Calculate feasibility score (0-10)"""

        base_score = 10.0

        # Deduct for issues
        for issue in issues:
            if issue["severity"] == "critical":
                base_score -= 3.0
            elif issue["severity"] == "high":
                base_score -= 2.0
            elif issue["severity"] == "medium":
                base_score -= 1.0

        # Deduct for bottlenecks
        bottleneck_count = len(metrics.get("bottleneck_tasks", []))
        base_score -= min(bottleneck_count * 0.5, 3.0)

        # Deduct for dependency violations
        violation_count = len(metrics.get("dependency_violations", []))
        base_score -= min(violation_count * 1.0, 4.0)

        return max(0.0, base_score)

    def _generate_recommendations(
        self,
        issues: List[Dict[str, Any]],
        progress: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on analysis"""

        recommendations = []

        for issue in issues:
            if issue["type"] == "bottleneck":
                recommendations.append("Consider parallelizing bottleneck tasks or breaking them down")
            elif issue["type"] == "dependency_violation":
                recommendations.append("Review and fix dependency violations before proceeding")
            elif issue["type"] == "complexity_imbalance":
                recommendations.append("Break down complex tasks into smaller, manageable units")

        # General recommendations
        ready_count = len(progress.get("ready_tasks", []))
        if ready_count > 5:
            recommendations.append("Many tasks are ready - consider increasing parallel execution")

        return recommendations

    def _calculate_potential_parallelism(self, graph: TaskDependencyGraph) -> int:
        """Calculate potential parallel execution opportunities"""

        # Count tasks with no dependencies (can run in parallel)
        independent_tasks = sum(
            1 for task in graph.tasks.values()
            if not task.depends_on
        )

        return independent_tasks

    async def _calculate_velocity(self, plan_id: UUID) -> Dict[str, float]:
        """Calculate task execution velocity"""

        # Get completed tasks (simplified - would use historical data)
        stats = await self.engine.repository.get_task_statistics(plan_id)
        completed_count = stats.get("completed", 0)

        # Calculate simple velocity metrics
        # In real implementation, would use time series data
        return {
            "tasks_per_hour": completed_count * 2.0,  # Simplified calculation
            "avg_duration_minutes": 6.5,  # Would calculate from actual data
            "completion_rate": 0.85  # Simplified rate
        }

    async def _get_recent_activity(self, plan_id: UUID) -> List[Dict[str, Any]]:
        """Get recent activity for the plan"""

        # Simplified - would query actual activity log
        return [
            {
                "timestamp": datetime.now() - timedelta(minutes=30),
                "action": "task_completed",
                "description": "Completed task: Example task"
            },
            {
                "timestamp": datetime.now() - timedelta(hours=1),
                "action": "task_started",
                "description": "Started task: Another example"
            }
        ]

    async def _suggest_next_actions(self, plan_id: UUID) -> List[Dict[str, Any]]:
        """Suggest next actions for the plan"""

        ready_tasks = await self.engine.get_ready_tasks(plan_id)

        actions = []

        if ready_tasks:
            actions.append({
                "action": "start_task",
                "description": f"Start next ready task: {ready_tasks[0].title}",
                "priority": "high"
            })

        # Get bottlenecks
        metrics = await self.engine.get_task_execution_metrics(plan_id)
        if metrics["bottleneck_tasks"]:
            actions.append({
                "action": "resolve_bottleneck",
                "description": f"Resolve bottleneck: {metrics['bottleneck_tasks'][0]['task_title']}",
                "priority": "high"
            })

        return actions

    async def _calculate_health_score(self, plan_id: UUID) -> float:
        """Calculate overall plan health score"""

        # Get various metrics
        progress = await self.engine.get_plan_progress(plan_id)
        metrics = await self.engine.get_task_execution_metrics(plan_id)

        base_score = 10.0

        # Deduct for violations and bottlenecks
        base_score -= len(metrics.get("dependency_violations", [])) * 2.0
        base_score -= len(metrics.get("bottleneck_tasks", [])) * 1.0

        # Boost for good progress
        if progress["overall_progress"] > 50:
            base_score += 1.0

        return max(0.0, min(10.0, base_score))