"""Task Graph for inter-agent coordination.

This module implements the Task Graph abstraction that replaces Query-based
communication with Task-based communication. As specified in AGENT_FRAMEWORK.md:

"Make the Task message (rather than the Query message) as the main mechanism
for agents to request context from other agents. This unifies two seemingly
different concepts: task assignment (parent-child exchange) and context
retrieval (peer-to-peer exchange)."

Key features:
- Tasks form a DAG in the blackboard
- Task results can be reused (creates DAG structure)
- Tasks can be picked up by any capable agent
- Supports dynamic task generation (agents can customize task prompts)
- Enables cache-aware scheduling

Integration with existing code:
- Stored in EnhancedBlackboard with event system
- Uses existing AgentResourceRequirements for scheduling
- Integrates with VCM for cache-aware scheduling
- Uses ScopeAwareResult pattern for task results
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# Import from patterns (domain-agnostic)
from ..patterns.scope import AnalysisScope, ScopeAwareResult

# Import existing models
from ..models import AgentResourceRequirements

# Import blackboard types
from .types import BlackboardEntry, BlackboardEvent
from .blackboard import EnhancedBlackboard


class TaskStatus(str, Enum):
    """Status of a task in the task graph."""

    PENDING = "pending"  # Created but not yet claimed
    CLAIMED = "claimed"  # Claimed by an agent but not started
    RUNNING = "running"  # Currently being executed
    BLOCKED = "blocked"  # Blocked by unsatisfied dependencies
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Failed with error
    CANCELLED = "cancelled"  # Cancelled by requester


class TaskPriority(int, Enum):
    """Priority levels for task scheduling."""

    CRITICAL = 0  # Must execute immediately
    HIGH = 1  # Important, execute soon
    NORMAL = 2  # Regular priority
    LOW = 3  # Background task
    DEFERRED = 4  # Can be delayed


class Task(BaseModel):
    """A task in the task graph.

    Tasks unify:
    - Task assignment (parent → child)
    - Context retrieval (peer → peer)
    - Analysis requests (coordinator → specialist)

    Every task is a node in the task graph DAG and can spawn child tasks.
    Task results can be reused by other tasks.

    Examples:
        Code analysis task:
        ```python
        task = Task(
            task_type="code_analysis",
            goal="Analyze authentication flow",
            description="Identify security issues in auth module",
            requirements={
                "analysis_types": ["taint_analysis", "dependency_analysis"],
                "focus_modules": ["auth", "session"]
            },
            constraints={
                "max_tokens": 100000,
                "deadline_seconds": 300
            },
            capable_agent_types=["code_analysis_specialist"],
            requesting_agent_id="coordinator_001"
        )
        ```

        Query-style task (replaces PageQuery):
        ```python
        task = Task(
            task_type="query",
            goal="Find AuthManager implementation",
            description="Locate where AuthManager.validate() is implemented",
            requirements={
                "query_text": "AuthManager.validate implementation",
                "query_type": "structural"
            },
            capable_agent_types=["query_processor"],
            requesting_agent_id="analyzer_042"
        )
        ```
    """

    # Identification
    task_id: str = Field(
        default_factory=lambda: f"task_{uuid.uuid4().hex}",
        description="Unique task identifier"
    )

    task_type: str = Field(
        description="Type of task: 'code_analysis', 'query', 'synthesis', 'validation', etc."
    )

    # Task specification
    goal: str = Field(
        description="What the task aims to achieve"
    )

    description: str = Field(
        description="Detailed task description"
    )

    requirements: dict[str, Any] = Field(
        default_factory=dict,
        description="Task requirements (what needs to be done, input data, etc.)"
    )

    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Constraints (time, resources, quality thresholds, etc.)"
    )

    # Customizable prompt (allows agents to generate/customize task prompts!)
    custom_prompt: str | None = Field(
        default=None,
        description="Custom prompt for executing this task (LLM-customizable)"
    )

    prompt_template: str | None = Field(
        default=None,
        description="Template for generating execution prompt"
    )

    # DAG structure
    parent_task_id: str | None = Field(
        default=None,
        description="Parent task that spawned this task"
    )

    child_task_ids: list[str] = Field(
        default_factory=list,
        description="Child tasks spawned by this task"
    )

    depends_on: list[str] = Field(
        default_factory=list,
        description="Task IDs that must complete before this task can run"
    )

    blocks: list[str] = Field(
        default_factory=list,
        description="Task IDs that are blocked by this task"
    )

    # Assignment and capabilities
    requesting_agent_id: str = Field(
        description="Agent that requested this task"
    )

    assigned_agent_id: str | None = Field(
        default=None,
        description="Agent assigned to execute this task"
    )

    capable_agent_types: list[str] = Field(
        default_factory=list,
        description="Types of agents capable of executing this task"
    )

    required_capabilities: list[str] = Field(
        default_factory=list,
        description="Specific capabilities required (tools, knowledge domains, etc.)"
    )

    # Execution state
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="Current task status"
    )

    result: Any | None = Field(
        default=None,
        description="Task result (should be ScopeAwareResult for most tasks)"
    )

    error: str | None = Field(
        default=None,
        description="Error message if task failed"
    )

    error_details: dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed error information"
    )

    # Scope awareness (from patterns)
    scope: AnalysisScope = Field(
        default_factory=AnalysisScope,
        description="Scope of this task (what it covers, what it needs)"
    )

    # Scheduling and prioritization
    priority: TaskPriority = Field(
        default=TaskPriority.NORMAL,
        description="Task priority for scheduling"
    )

    deadline: float | None = Field(
        default=None,
        description="Deadline timestamp (None = no deadline)"
    )

    # Resource requirements
    resource_requirements: AgentResourceRequirements = Field(
        default_factory=AgentResourceRequirements,
        description="Resource requirements for executing this task"
    )

    # Cost tracking
    estimated_cost: dict[str, float] = Field(
        default_factory=dict,
        description="Estimated costs (tokens, time, etc.)"
    )

    actual_cost: dict[str, float] = Field(
        default_factory=dict,
        description="Actual costs incurred"
    )

    # Timestamps
    created_at: float = Field(
        default_factory=time.time,
        description="When task was created"
    )

    claimed_at: float | None = Field(
        default=None,
        description="When task was claimed"
    )

    started_at: float | None = Field(
        default=None,
        description="When execution started"
    )

    completed_at: float | None = Field(
        default=None,
        description="When execution completed"
    )

    # Retry and recovery
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts on failure"
    )

    retry_count: int = Field(
        default=0,
        description="Number of retries so far"
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional task metadata"
    )

    # Methods

    def can_execute(self) -> bool:
        """Check if task can be executed (dependencies met).

        Returns:
            True if all dependencies are satisfied
        """
        # Task can execute if:
        # - Status is PENDING
        # - No blocking dependencies
        return self.status == TaskStatus.PENDING

    def is_overdue(self) -> bool:
        """Check if task is overdue.

        Returns:
            True if past deadline
        """
        if self.deadline is None:
            return False
        return time.time() > self.deadline

    def can_retry(self) -> bool:
        """Check if task can be retried.

        Returns:
            True if retries remaining
        """
        return self.retry_count < self.max_retries

    def mark_claimed(self, agent_id: str) -> None:
        """Mark task as claimed by an agent.

        Args:
            agent_id: Agent claiming the task
        """
        self.status = TaskStatus.CLAIMED
        self.assigned_agent_id = agent_id
        self.claimed_at = time.time()

    def mark_running(self) -> None:
        """Mark task as running."""
        self.status = TaskStatus.RUNNING
        self.started_at = time.time()

    def mark_completed(self, result: Any) -> None:
        """Mark task as completed with result.

        Args:
            result: Task result
        """
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()

        # Update scope if result is ScopeAwareResult
        if isinstance(result, ScopeAwareResult):
            self.scope = result.scope

    def mark_failed(self, error: str, error_details: dict[str, Any] | None = None) -> None:
        """Mark task as failed.

        Args:
            error: Error message
            error_details: Optional error details
        """
        self.status = TaskStatus.FAILED
        self.error = error
        if error_details:
            self.error_details = error_details
        self.completed_at = time.time()

    def to_blackboard_entry(self) -> dict[str, Any]:
        """Convert to blackboard entry format.

        Returns:
            Dictionary for blackboard storage
        """
        return self.model_dump()

    @classmethod
    def from_blackboard_entry(cls, entry: dict[str, Any]) -> Task:
        """Reconstruct from blackboard entry.

        Args:
            entry: Blackboard entry

        Returns:
            Task instance
        """
        return cls(**entry)


class TaskGraph:
    """Manages task graph in blackboard.

    The task graph is stored in the blackboard and provides:
    - Task creation and retrieval
    - Dependency management with cycle detection
    - Atomic task claiming
    - Task lifecycle management
    - Event-driven notifications
    - Critical path analysis
    - Progress tracking

    All operations are backed by the blackboard for distributed coordination.
    """

    def __init__(self, blackboard: EnhancedBlackboard):
        """Initialize task graph.

        Args:
            blackboard: EnhancedBlackboard instance for storage
        """
        self.blackboard = blackboard
        self.namespace = "task_graph"
        self._task_cache: dict[str, Task] = {}  # Local cache for performance

    async def add_task(self, task: Task) -> str:
        """Add task to graph with cycle detection.

        Args:
            task: Task to add

        Returns:
            Task ID

        Raises:
            ValueError: If adding task would create a cycle
        """
        # Check for dependency cycles before adding
        if task.depends_on:
            if await self._would_create_cycle(task.task_id, task.depends_on):
                raise ValueError(f"Adding task {task.task_id} would create a dependency cycle")

        key = f"{self.namespace}:{task.task_id}"

        await self.blackboard.write(
            key=key,
            value=task.to_blackboard_entry(),
            tags={
                "task",
                task.task_type,
                task.status.value,
                f"priority_{task.priority.value}"
            },
            created_by=task.requesting_agent_id,
            metadata={
                "parent_task_id": task.parent_task_id,
                "depends_on": task.depends_on,
                "priority": task.priority.value
            }
        )

        # Update cache
        self._task_cache[task.task_id] = task

        # Update parent's child list if this is a child task
        if task.parent_task_id:
            await self._add_child_to_parent(task.parent_task_id, task.task_id)

        # Update dependency tracking
        for dep_task_id in task.depends_on:
            await self._add_dependent(dep_task_id, task.task_id)

        # Emit event for new task
        await self.blackboard.emit_event(
            BlackboardEvent(
                event_type="task_created",
                key=f"{self.namespace}:{task.task_id}",
                value=task.to_blackboard_entry(),
                metadata={"task_type": task.task_type}
            )
        )

        return task.task_id

    async def get_task(self, task_id: str) -> Task | None:
        """Get task by ID.

        Args:
            task_id: Task ID

        Returns:
            Task or None if not found
        """
        key = f"{self.namespace}:{task_id}"
        entry = await self.blackboard.read(key)

        if entry is None:
            return None

        return Task.from_blackboard_entry(entry)

    async def get_available_tasks(
        self,
        agent_type: str | None = None,
        max_tasks: int = 10,
        capabilities: list[str] | None = None
    ) -> list[Task]:
        """Get tasks available for execution.

        Args:
            agent_type: Optional filter by capable agent type
            max_tasks: Maximum number of tasks to return
            capabilities: Optional filter by required capabilities

        Returns:
            List of available tasks, sorted by priority
        """
        # Query for PENDING tasks
        all_tasks = await self._query_tasks_by_status(TaskStatus.PENDING)

        # Filter by agent type if specified
        if agent_type:
            all_tasks = [
                t for t in all_tasks
                if not t.capable_agent_types or agent_type in t.capable_agent_types
            ]

        # Filter by capabilities if specified
        if capabilities:
            all_tasks = [
                t for t in all_tasks
                if not t.required_capabilities or all(
                    cap in capabilities for cap in t.required_capabilities
                )
            ]

        # Filter by dependencies (only include tasks whose dependencies are met)
        executable_tasks = []
        for task in all_tasks:
            if await self._are_dependencies_satisfied(task):
                executable_tasks.append(task)
            else:
                # Mark as blocked if dependencies not satisfied
                task.status = TaskStatus.BLOCKED
                await self.update_task(task)

        # Sort by priority (lower number = higher priority)
        executable_tasks.sort(key=lambda t: (t.priority.value, t.created_at))

        return executable_tasks[:max_tasks]

    async def claim_task(
        self,
        task_id: str,
        agent_id: str,
        force: bool = False
    ) -> bool:
        """Claim a task atomically.

        Args:
            task_id: Task to claim
            agent_id: Agent claiming the task
            force: Whether to force claim even if already claimed

        Returns:
            True if successfully claimed
        """
        # Use transaction for atomic claim
        async with self.blackboard.transaction() as txn:
            key = f"{self.namespace}:{task_id}"
            entry = await txn.read(key)

            if not entry:
                return False

            task = Task.from_blackboard_entry(entry.value)

            # Check if can be claimed
            if not force and task.status != TaskStatus.PENDING:
                return False  # Already claimed or running

            # Claim task
            task.mark_claimed(agent_id)

            # Create updated entry
            updated_entry = BlackboardEntry(
                key=key,
                value=task.to_blackboard_entry(),
                version=entry.version + 1,
                created_at=entry.created_at,
                updated_at=time.time(),
                created_by=entry.created_by,
                updated_by=agent_id,
                ttl_seconds=entry.ttl_seconds,
                tags=entry.tags,
                metadata=entry.metadata
            )

            # Write back
            await txn.write(key, updated_entry)

            return True

    async def start_task(self, task_id: str, agent_id: str) -> bool:
        """Mark task as running.

        Args:
            task_id: Task ID
            agent_id: Agent starting the task

        Returns:
            True if successfully marked as running
        """
        task = await self.get_task(task_id)
        if not task:
            return False

        # Verify claimed by this agent
        if task.assigned_agent_id != agent_id:
            return False

        task.mark_running()
        await self.update_task(task)

        return True

    async def complete_task(
        self,
        task_id: str,
        result: Any,
        agent_id: str
    ) -> bool:
        """Mark task as completed with result.

        Args:
            task_id: Task ID
            result: Task result (should be ScopeAwareResult)
            agent_id: Agent completing the task

        Returns:
            True if successfully completed
        """
        task = await self.get_task(task_id)
        if not task:
            return False

        # Verify assigned to this agent
        if task.assigned_agent_id != agent_id:
            return False

        task.mark_completed(result)
        await self.update_task(task)

        # Emit completion event
        await self.blackboard.emit_event(
            BlackboardEvent(
                event_type="task_completed",
                key=f"{self.namespace}:{task_id}",
                value=task.to_blackboard_entry(),
                metadata={"task_type": task.task_type}
            )
        )

        # Unblock dependent tasks
        await self._unblock_dependents(task_id)

        return True

    async def fail_task(
        self,
        task_id: str,
        error: str,
        agent_id: str,
        error_details: dict[str, Any] | None = None
    ) -> bool:
        """Mark task as failed.

        Args:
            task_id: Task ID
            error: Error message
            agent_id: Agent that encountered the error
            error_details: Optional error details

        Returns:
            True if successfully marked as failed
        """
        task = await self.get_task(task_id)
        if not task:
            return False

        # Verify assigned to this agent
        if task.assigned_agent_id != agent_id:
            return False

        task.mark_failed(error, error_details)
        await self.update_task(task)

        # Emit failure event
        await self.blackboard.emit_event(
            BlackboardEvent(
                event_type="task_failed",
                key=f"{self.namespace}:{task_id}",
                value=task.to_blackboard_entry(),
                metadata={"error": error}
            )
        )

        return True

    async def update_task(self, task: Task) -> None:
        """Update task in blackboard.

        Args:
            task: Task with updates
        """
        key = f"{self.namespace}:{task.task_id}"

        await self.blackboard.write(
            key=key,
            value=task.to_blackboard_entry(),
            created_by=task.assigned_agent_id,
            tags={
                "task",
                task.task_type,
                task.status.value,
                f"priority_{task.priority.value}"
            },
            metadata={
                "parent_task_id": task.parent_task_id,
                "depends_on": task.depends_on,
                "priority": task.priority.value
            }
        )

    async def get_task_dependencies(self, task_id: str) -> list[Task]:
        """Get all tasks this task depends on.

        Args:
            task_id: Task ID

        Returns:
            List of dependency tasks
        """
        task = await self.get_task(task_id)
        if not task:
            return []

        dependencies = []
        for dep_id in task.depends_on:
            dep_task = await self.get_task(dep_id)
            if dep_task:
                dependencies.append(dep_task)

        return dependencies

    async def get_task_dependents(self, task_id: str) -> list[Task]:
        """Get all tasks that depend on this task.

        Args:
            task_id: Task ID

        Returns:
            List of dependent tasks
        """
        task = await self.get_task(task_id)
        if not task:
            return []

        dependents = []
        for dependent_id in task.blocks:
            dependent_task = await self.get_task(dependent_id)
            if dependent_task:
                dependents.append(dependent_task)

        return dependents

    async def get_blocked_tasks(self) -> list[Task]:
        """Get tasks that are blocked by incomplete dependencies.

        Returns:
            List of blocked tasks
        """
        blocked = []

        # Get all BLOCKED status tasks
        all_tasks = await self._query_tasks_by_status(TaskStatus.BLOCKED)

        # Verify they're still blocked
        for task in all_tasks:
            if not await self._are_dependencies_satisfied(task):
                blocked.append(task)
            else:
                # Dependencies now satisfied, unblock
                task.status = TaskStatus.PENDING
                await self.update_task(task)

        return blocked

    async def get_critical_path(self, root_task_id: str) -> list[str]:
        """Get the critical path from root task.

        The critical path is the longest path through the DAG
        that determines the minimum completion time.

        Args:
            root_task_id: Root task ID

        Returns:
            List of task IDs forming the critical path
        """
        # Build adjacency list
        adj = {}
        task_durations = {}

        # Get all tasks in the subgraph
        visited = set()
        queue = [root_task_id]

        while queue:
            task_id = queue.pop(0)
            if task_id in visited:
                continue
            visited.add(task_id)

            task = await self.get_task(task_id)
            if not task:
                continue

            # Estimate duration from resource requirements
            duration = task.resource_requirements.estimated_duration_seconds or 60
            task_durations[task_id] = duration

            # Add edges to children
            adj[task_id] = task.child_task_ids
            queue.extend(task.child_task_ids)

        # Find critical path using topological sort and dynamic programming
        critical_path = self._find_critical_path(adj, task_durations, root_task_id)

        return critical_path

    def _find_critical_path(
        self,
        adj: dict[str, list[str]],
        durations: dict[str, float],
        root: str
    ) -> list[str]:
        """Find critical path using dynamic programming.

        Args:
            adj: Adjacency list (task -> children)
            durations: Task durations
            root: Root task

        Returns:
            Critical path task IDs
        """
        # Calculate earliest start times
        earliest_start = {root: 0}
        latest_start = {}

        # Topological order
        topo_order = self._topological_sort(adj)

        # Forward pass: calculate earliest start times
        for task_id in topo_order:
            if task_id not in earliest_start:
                # Find max of predecessors' finish times
                max_finish = 0
                for pred_id, children in adj.items():
                    if task_id in children:
                        pred_finish = earliest_start.get(pred_id, 0) + durations.get(pred_id, 0)
                        max_finish = max(max_finish, pred_finish)
                earliest_start[task_id] = max_finish

        # Find project completion time
        completion_time = max(
            earliest_start.get(t, 0) + durations.get(t, 0)
            for t in topo_order
        )

        # Backward pass: calculate latest start times
        for task_id in reversed(topo_order):
            children = adj.get(task_id, [])
            if not children:
                # Leaf node
                latest_start[task_id] = completion_time - durations.get(task_id, 0)
            else:
                # Min of children's latest starts minus our duration
                min_child_start = min(
                    latest_start.get(child, completion_time)
                    for child in children
                )
                latest_start[task_id] = min_child_start - durations.get(task_id, 0)

        # Critical tasks have earliest_start == latest_start (zero slack)
        critical_tasks = []
        for task_id in topo_order:
            slack = latest_start.get(task_id, 0) - earliest_start.get(task_id, 0)
            if abs(slack) < 0.01:  # Account for floating point
                critical_tasks.append(task_id)

        return critical_tasks

    def _topological_sort(self, adj: dict[str, list[str]]) -> list[str]:
        """Topological sort of DAG.

        Args:
            adj: Adjacency list

        Returns:
            Topologically sorted task IDs
        """
        # Calculate in-degrees
        in_degree = {node: 0 for node in adj}
        for children in adj.values():
            for child in children:
                if child not in in_degree:
                    in_degree[child] = 0
                in_degree[child] += 1

        # Start with nodes having in-degree 0
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            # Reduce in-degree of children
            for child in adj.get(node, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return result

    async def _would_create_cycle(
        self,
        new_task_id: str,
        dependencies: list[str]
    ) -> bool:
        """Check if adding dependencies would create a cycle.

        Uses DFS to detect if any dependency can reach back to new_task_id.

        Args:
            new_task_id: New task being added
            dependencies: Its dependencies

        Returns:
            True if would create cycle
        """
        for dep_id in dependencies:
            if await self._can_reach(dep_id, new_task_id):
                return True
        return False

    async def _can_reach(self, from_id: str, to_id: str) -> bool:
        """Check if from_id can reach to_id through dependencies.

        Args:
            from_id: Starting task
            to_id: Target task

        Returns:
            True if path exists
        """
        visited = set()
        stack = [from_id]

        while stack:
            current = stack.pop()
            if current == to_id:
                return True

            if current in visited:
                continue
            visited.add(current)

            # Get task's dependencies
            task = await self.get_task(current)
            if task:
                stack.extend(task.depends_on)

        return False

    async def get_task_progress(self, root_task_id: str) -> dict[str, Any]:
        """Get progress statistics for task tree.

        Args:
            root_task_id: Root task ID

        Returns:
            Progress statistics
        """
        tree = await self.get_task_tree(root_task_id)

        stats = {
            "total": 0,
            "pending": 0,
            "claimed": 0,
            "running": 0,
            "blocked": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "progress_percentage": 0.0,
            "estimated_remaining_seconds": 0
        }

        # Count tasks by status
        def count_tasks(node: dict):
            task = Task(**node["task"])
            stats["total"] += 1
            stats[task.status.value] += 1

            for child in node.get("children", []):
                count_tasks(child)

        if tree:
            count_tasks(tree)

            # Calculate progress
            if stats["total"] > 0:
                completed_weight = stats["completed"]
                running_weight = stats["running"] * 0.5  # Running tasks count as half
                stats["progress_percentage"] = (
                    (completed_weight + running_weight) / stats["total"] * 100
                )

                # Estimate remaining time based on average task duration
                # This is simplified - real implementation would use historical data
                avg_duration = 60  # seconds
                remaining_tasks = stats["total"] - stats["completed"]
                stats["estimated_remaining_seconds"] = remaining_tasks * avg_duration

        return stats


    async def _are_dependencies_satisfied(self, task: Task) -> bool:
        """Check if all dependencies are satisfied.

        Args:
            task: Task to check

        Returns:
            True if all dependencies completed
        """
        if not task.depends_on:
            return True

        for dep_id in task.depends_on:
            dep_task = await self.get_task(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False

        return True

    async def _add_child_to_parent(self, parent_id: str, child_id: str) -> None:
        """Add child to parent's child list.

        Args:
            parent_id: Parent task ID
            child_id: Child task ID
        """
        parent = await self.get_task(parent_id)
        if parent and child_id not in parent.child_task_ids:
            parent.child_task_ids.append(child_id)
            await self.update_task(parent)

    async def _add_dependent(self, task_id: str, dependent_id: str) -> None:
        """Add dependent to task's blocks list.

        Args:
            task_id: Task ID
            dependent_id: Dependent task ID
        """
        task = await self.get_task(task_id)
        if task and dependent_id not in task.blocks:
            task.blocks.append(dependent_id)
            await self.update_task(task)

    async def _unblock_dependents(self, task_id: str) -> None:
        """Unblock tasks that were waiting for this task.

        Args:
            task_id: Completed task ID
        """
        # Get all tasks that depend on this one
        dependents = await self.get_task_dependents(task_id)

        for dependent in dependents:
            if dependent.status == TaskStatus.BLOCKED:
                # Check if all dependencies now satisfied
                if await self._are_dependencies_satisfied(dependent):
                    dependent.status = TaskStatus.PENDING
                    await self.update_task(dependent)

    async def _query_tasks_by_status(self, status: TaskStatus) -> list[Task]:
        """Query tasks by status using blackboard query.

        Args:
            status: Task status to filter by

        Returns:
            List of tasks with that status
        """
        # Query blackboard for tasks with status tag
        entries = await self.blackboard.query(
            namespace=self.namespace,
            tags={status.value},
            limit=1000  # Large limit to get all matching tasks
        )

        tasks = []
        for entry in entries:
            try:
                task = Task.from_blackboard_entry(entry.value)
                # Double-check status matches (in case of stale tags)
                if task.status == status:
                    tasks.append(task)
            except Exception:
                # Skip invalid entries
                continue

        return tasks

    async def get_task_tree(self, root_task_id: str) -> dict[str, Any]:
        """Get task tree rooted at given task.

        Args:
            root_task_id: Root task ID

        Returns:
            Tree structure with task and children
        """
        root = await self.get_task(root_task_id)
        if not root:
            return {}

        tree = {
            "task": root.model_dump(),
            "children": []
        }

        # Recursively get children
        for child_id in root.child_task_ids:
            child_tree = await self.get_task_tree(child_id)
            if child_tree:
                tree["children"].append(child_tree)

        return tree


# Utility functions

async def create_analysis_task(
    goal: str,
    description: str,
    analysis_types: list[str],
    requesting_agent_id: str,
    parent_task_id: str | None = None,
    priority: TaskPriority = TaskPriority.NORMAL
) -> Task:
    """Create a code analysis task.

    Args:
        goal: Analysis goal
        description: Detailed description
        analysis_types: Types of analysis to perform
        requesting_agent_id: Requesting agent ID
        parent_task_id: Optional parent task
        priority: Task priority

    Returns:
        Created task
    """
    return Task(
        task_type="code_analysis",
        goal=goal,
        description=description,
        requirements={"analysis_types": analysis_types},
        requesting_agent_id=requesting_agent_id,
        parent_task_id=parent_task_id,
        priority=priority,
        capable_agent_types=["code_analysis_specialist", "code_analyzer"]
    )


async def create_query_task(
    query_text: str,
    requesting_agent_id: str,
    query_type: str = "semantic",
    parent_task_id: str | None = None
) -> Task:
    """Create a query task (replaces PageQuery).

    Args:
        query_text: Query text
        requesting_agent_id: Requesting agent ID
        query_type: Type of query
        parent_task_id: Optional parent task

    Returns:
        Created task
    """
    return Task(
        task_type="query",
        goal=f"Answer query: {query_text}",
        description=query_text,
        requirements={
            "query_text": query_text,
            "query_type": query_type
        },
        requesting_agent_id=requesting_agent_id,
        parent_task_id=parent_task_id,
        capable_agent_types=["query_processor", "general_agent"]
    )

