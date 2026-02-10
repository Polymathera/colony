"""Data models for the agent system.

This module defines all core data structures for agents, actions, plans,
tools, and communication.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal, TYPE_CHECKING, AsyncContextManager
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    from ..distributed.state_management import SharedState


# ============================================================================
# Action Models
# ============================================================================


class ActionType(str, Enum):
    """Types of actions agents can perform."""

    PERFORMANCE_UPDATE = "performance_update"

    # Planning
    PLAN_CREATE = "plan_create"
    PLAN_REVISE = "plan_revise"
    PLAN_BACKTRACK = "plan_backtrack"

    # Reasoning
    ANALYZE = "analyze"
    HYPOTHESIS_TEST = "hypothesis_test"
    DECISION_MAKE = "decision_make"

    # Tool usage
    TOOL_DISCOVER = "tool_discover"
    TOOL_USE = "tool_use"
    TOOL_FIX = "tool_fix"

    # Tool building
    TOOL_CREATE = "tool_create"
    TOOL_REFINE = "tool_refine"

    # Context management
    CONTEXT_FETCH = "context_fetch"
    CONTEXT_COMPACT = "context_compact"
    CONTEXT_SUMMARIZE = "context_summarize"

    # Communication
    MESSAGE_SEND = "message_send"  # Unicast, multicast, broadcast
    MESSAGE_RECEIVE = "message_receive"
    NEGOTIATE = "negotiate"

    # Memory management
    MEMORY_SEARCH = "memory_search"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    CHANGESET_MAINTAIN = "changeset_maintain"

    # Agent orchestration
    AGENT_SPAWN = "agent_spawn"  # Can spawn multiple agents
    AGENT_TERMINATE = "agent_terminate"
    AGENT_MONITOR = "agent_monitor"  # Blocking or non-blocking

    # Output
    OUTPUT_WRITE = "output_write"
    OUTPUT_FORMAT = "output_format"
    REPORT_GENERATE = "report_generate"

    # Reasoning loop specific actions (for code analysis and similar tasks)
    ANALYZE_PAGE = "analyze_page"  # Load and analyze a single page
    GENERATE_QUERIES = "generate_queries"  # Generate queries from analysis findings
    ROUTE_QUERY = "route_query"  # Find relevant pages for a query using routing policy
    PROCESS_QUERY = "process_query"  # Load relevant pages and get LLM answer to query
    SPAWN_SUBAGENT = "spawn_subagent"  # Delegate to child agent
    SYNTHESIZE = "synthesize"  # Combine results from multiple sources
    WRITE_BLACKBOARD = "write_blackboard"  # Write to shared memory

    # Code Analysis Actions
    # TODO: Should these be tools instead? Or should they be abstract actions that are specialized for code analysis?
    COMPUTE_SLICE = "compute_slice"  # Compute program slice
    INFER_CONTRACTS = "infer_contracts"  # Infer contracts from code
    INFER_INTENT = "infer_intent"  # Infer intent from code
    ANALYZE_COMPLIANCE = "analyze_compliance"  # Analyze compliance
    ANALYZE_IMPACT = "analyze_impact"  # Analyze change impact
    READ_BLACKBOARD = "read_blackboard"  # Read from shared memory
    WAIT_FOR_SUBAGENTS = "wait_for_subagents"  # Wait for child agents
    CUSTOM = "custom"  # Domain-specific action

    SPAWN_CHILD_AGENT = "spawn_child_agent"
    WAIT_FOR_CHILDREN = "wait_for_children"
    AGGREGATE_CHILD_RESULTS = "aggregate_child_results"

    # REPL execution
    EXECUTE_CODE = "execute_code"  # Execute Python code in PolicyPythonREPL


class ActionStatus(str, Enum):
    """Status of an action execution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"  # Alias for RUNNING
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"  # For long-running actions
    BLOCKED = "blocked"  # For actions waiting on dependencies



class ActionResult(BaseModel):
    """Result of executing an action.

    Example:
        ```python
        result = ActionResult(
            success=True,
            output={"pages_found": ["page_042", "page_015"]},
            metrics={"time_ms": 150, "pages_loaded": 2}
        )
        ```
    """

    success: bool
    output: Any = None
    error: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)  # tokens, latency, cost, etc.
    metadata: dict[str, Any] = Field(default_factory=dict)
    blocked: bool = False  # If action is blocked waiting on dependencies
    blocked_reason: str | None = None
    completed: bool = Field(
        default=False,
        description="If action fully completed (vs. partial result). Used mostly by ActionPolicies. If True, action is done. If False, more iterations needed."
    )


class ActionCheckpoint(BaseModel):
    """Checkpoint for long-running actions."""

    checkpoint_id: str
    action_id: str
    step_number: int
    state: dict[str, Any]
    created_at: float = Field(default_factory=time.time)



class Action(BaseModel):
    """An action to be executed by an agent.
    Rich action representation with tracking and results.

    This represents a single action taken by an agent, with full context
    about what was done, why, and what the outcome was.

    Example:
        ```python
        action = Action(
            action_type=ActionType.ROUTE_QUERY,
            parameters={
                "query": "Where is AuthManager defined?",
                "max_results": 5
            },
            reasoning="Need to find authentication implementation"
        )
        ```

    Attributes:
        action_id: Unique identifier for this action
        agent_id: Agent that performed this action
        action_type: Type of action
        parameters: Action-specific parameters
        status: Current status
        result: Result if completed
        created_at: When action was created
        started_at: When action started executing
        completed_at: When action completed
        parent_action_id: If this is a sub-action of a long-running action
        plan_snapshot: Snapshot of the plan when this action was taken
        reasoning_trace: Full reasoning trace for this action
        checkpoints: Checkpoints for long-running actions
        metadata: Additional context
    """
    action_id: str
    agent_id: str
    action_type: ActionType | str
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: ActionStatus = ActionStatus.PENDING
    result: ActionResult | None = None

    reasoning: str | None = None  # Why this action?
    expected_outcome: str | None = None  # What do we expect?
    priority: int = 1  # Higher = more important

    created_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None

    # For hierarchical actions
    parent_action_id: str | None = None

    # Rich context - full stream of consciousness
    plan_snapshot: dict[str, Any] | None = None  # Plan state when action was taken
    reasoning_trace: str | None = None  # Full reasoning for this action
    checkpoints: list[ActionCheckpoint] = Field(default_factory=list)

    # Retry configuration
    max_retries: int = 3  # Maximum number of retries
    retry_count: int = 0  # Current retry count
    timeout_seconds: float | None = None  # Timeout for action execution

    # Additional metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Hierarchical planning support - action can contain sub-plan
    sub_plan: ActionPlan | None = None  # For composite actions
    description: str | None = None  # Human-readable description

    # REPL integration - for dataflow between actions
    result_var: str | None = Field(
        default=None,
        description="REPL variable name to assign result.output to after execution"
    )
    code: str | None = Field(
        default=None,
        description="Python code to execute (for EXECUTE_CODE action type)"
    )
    storage_hint: Any | None = Field(
        default=None,
        description="StorageHint for how to store result (value vs reference to backing store)"
    )

    def start(self) -> None:
        """Mark action as started."""
        self.status = ActionStatus.RUNNING
        self.started_at = time.time()

    def complete(self, result: ActionResult) -> None:
        """Mark action as completed."""
        self.status = ActionStatus.COMPLETED if result.success else ActionStatus.FAILED
        self.result = result
        self.completed_at = time.time()

    def pause(self) -> None:
        """Pause long-running action."""
        self.status = ActionStatus.PAUSED

    def cancel(self) -> None:
        """Cancel action."""
        self.status = ActionStatus.CANCELLED
        self.completed_at = time.time()

    def add_checkpoint(self, checkpoint: ActionCheckpoint) -> None:
        """Add a checkpoint for this action."""
        self.checkpoints.append(checkpoint)

    def is_atomic(self) -> bool:
        """Check if action is atomic (no sub-plan)."""
        return self.sub_plan is None

    def is_composite(self) -> bool:
        """Check if action has sub-plan."""
        return self.sub_plan is not None

    def is_long_running(self) -> bool:
        """Check if action is long-running (has checkpoints or sub-plan)."""
        return len(self.checkpoints) > 0 or self.sub_plan is not None

    # -------------------------------------------------------------------------
    # Memory System Integration
    # -------------------------------------------------------------------------

    def get_blackboard_key(self, scope_id: str) -> str:
        """Generate blackboard key for storing this action in memory.

        Args:
            scope_id: Memory scope ID (e.g., "agent:abc123:working")

        Returns:
            Key like "agent:abc123:working:action:action_001"
        """
        return f"{scope_id}:action:{self.action_id}"

    @staticmethod
    def get_key_pattern(scope_id: str) -> str:
        """Pattern for matching all actions in a scope.

        Args:
            scope_id: Memory scope ID

        Returns:
            Pattern like "agent:abc123:working:action:*"
        """
        return f"{scope_id}:action:*"



# ============================================================================
# Planning Models
# ============================================================================


class PlanStatus(str, Enum):
    """Plan execution status."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    ACTIVE = "active"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPENDED = "suspended"


class PlanVisibility(str, Enum):
    """Who can see this plan."""

    PRIVATE = "private"
    TEAM = "team"
    HIERARCHY = "hierarchy"
    GLOBAL = "global"


class PlanningStrategy(str, Enum):
    """Planning strategy along top-down to bottom-up spectrum."""

    TOP_DOWN = "top_down"  # Plan all steps upfront
    BOTTOM_UP = "bottom_up"  # Plan one step at a time, fully reactive
    HYBRID = "hybrid"  # Plan a few steps ahead, adapt as needed
    MPC = "mpc"  # Model-predictive control (plan horizon, re-evaluate)


class RevisionTrigger(str, Enum):
    """Triggers that cause plan revision."""

    FAILURE = "failure"  # Action failed
    BLOCKED = "blocked"  # Action blocked
    QUALITY_THRESHOLD = "quality_threshold"  # Plan quality below threshold
    NEW_INFORMATION = "new_information"  # Significant new information available
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # Resources (cache, memory) exhausted
    TIME_LIMIT = "time_limit"  # Execution time exceeded
    EXTERNAL_REQUEST = "external_request"  # External agent/system requests revision
    PERIODIC = "periodic"  # Periodic re-evaluation (MPC)


class RevisionStrategy(str, Enum):
    """Strategies for revising plans."""

    REPLAN_FROM_SCRATCH = "replan_from_scratch"  # Discard and create new plan
    INCREMENTAL_REPAIR = "incremental_repair"  # Fix broken parts
    BACKTRACK = "backtrack"  # Revert to previous version
    REORDER_ACTIONS = "reorder_actions"  # Change action sequence
    ADD_ACTIONS = "add_actions"  # Insert new actions
    REMOVE_ACTIONS = "remove_actions"  # Remove unnecessary actions
    SUBSTITUTE_ACTIONS = "substitute_actions"  # Replace actions with alternatives


class ConflictType(str, Enum):
    """Types of conflicts between plans."""

    RESOURCE_CONTENTION = "resource_contention"  # Both need same resource
    CACHE_CONTENTION = "cache_contention"  # Cache conflict
    SEQUENTIAL_DEPENDENCY = "sequential_dependency"  # Must execute in order
    MUTUAL_EXCLUSION = "mutual_exclusion"  # Cannot execute simultaneously
    PRIORITY_CONFLICT = "priority_conflict"  # Priority ordering conflict
    GOAL_CONFLICT = "goal_conflict"  # Conflicting goals
    DEPENDENCY_CYCLE = "dependency_cycle"  # Circular dependency
    TEMPORAL_CONFLICT = "temporal_conflict"  # Time-based conflict


class ConflictResolutionStrategy(str, Enum):
    """Strategies for resolving conflicts between plans."""

    PRIORITY = "priority"  # Resolve by priority ordering
    TEMPORAL = "temporal"  # Stagger execution over time
    RESOURCE_EFFICIENCY = "efficiency"  # Optimize resource usage
    NEGOTIATION = "negotiation"  # Negotiate between agents
    ESCALATION = "escalation"  # Escalate to higher authority
    REPLAN = "replan"  # Replan one or both conflicting plans


class PlanScope(str, Enum):
    """Hierarchical scope of a plan."""

    SYSTEM = "system"  # System-wide coordination
    CLUSTER = "cluster"  # Cluster-level analysis
    PAGE = "page"  # Single page analysis
    SUBGOAL = "subgoal"  # Sub-goal within larger plan


# ============================================================================
# Plan Evaluation Models
# ============================================================================


class CostModel(BaseModel):
    """Model for estimating plan execution costs."""

    total_tokens: float = 0.0
    prompt_tokens: float = 0.0
    completion_tokens: float = 0.0
    estimated_duration_seconds: float = 0.0
    pages_to_load: int = 0
    llm_calls: int = 0
    memory_mb: float = 0.0
    compute_cost_usd: float = 0.0


class BenefitModel(BaseModel):
    """Model for estimating plan execution benefits."""

    expected_quality: float = 0.0  # 0.0 to 1.0
    information_gain: float = 0.0
    goal_progress: float = 0.0
    learning_value: float = 0.0
    reusability: float = 0.0


class RiskModel(BaseModel):
    """Model for estimating plan execution risks."""

    failure_probability: float = 0.0  # 0.0 to 1.0
    partial_completion_risk: float = 0.0
    resource_exhaustion_risk: float = 0.0
    coordination_conflict_risk: float = 0.0
    max_loss_if_failed: float = 0.0


class PlanEvaluation(BaseModel):
    """Complete evaluation of a plan's cost, benefit, and risk."""

    plan_id: str
    cost: CostModel
    benefit: BenefitModel
    risk: RiskModel
    utility_score: float = 0.0  # benefit / (cost * risk)

    def calculate_utility(self) -> float:
        """Calculate utility score from cost, benefit, and risk.

        Returns:
            Utility score (higher is better)
        """
        # Avoid division by zero
        cost_factor = max(self.cost.total_tokens, 1.0)
        risk_factor = max(self.risk.failure_probability, 0.01)

        # Simple utility calculation: benefit / (cost * risk)
        # Can be refined with more sophisticated models
        benefit_sum = (
            self.benefit.expected_quality
            + self.benefit.information_gain
            + self.benefit.goal_progress
            + self.benefit.learning_value
            + self.benefit.reusability
        )

        self.utility_score = benefit_sum / (cost_factor * risk_factor)
        return self.utility_score


# ============================================================================
# Cache-Aware Planning Models
# ============================================================================


class CacheContext(BaseModel):
    """Cache-aware planning context.

    Tracks cache requirements, access patterns, and page relationships
    for cache-aware plan optimization.
    """

    # Working set - pages expected to be accessed during plan execution
    working_set: list[str] = Field(default_factory=list, description="Hot pages currently loaded")
    working_set_priority: dict[str, float] = Field(default_factory=dict)
    estimated_access_pattern: dict[str, int] = Field(default_factory=dict)
    access_sequence: list[str] = Field(default_factory=list)
    prefetch_pages: list[str] = Field(default_factory=list)

    page_temperatures: dict[str, float] = Field(default_factory=dict, description="page_id -> access frequency")

    # Cache sizing
    min_cache_size: int = 0
    ideal_cache_size: int = 0
    cache_hit_prediction: float = 0.0

    # Page graph and relationships (from CACHE_AWARE_PLANNING.md)
    page_graph_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of page relationships (dependencies, clusters, etc.)"
    )

    # Current cache state
    current_cached_pages: list[str] = Field(
        default_factory=list,
        description="Pages currently loaded in cache"
    )

    # Page sharing and exclusivity
    shareable_pages: list[str] = Field(
        default_factory=list,
        description="Pages that can be safely shared between agents"
    )
    exclusive_pages: list[str] = Field(
        default_factory=list,
        description="Pages requiring exclusive access (no sharing)"
    )

    # Detailed cache requirements per resource
    estimated_cache_requirements: dict[str, int] = Field(
        default_factory=dict,
        description="Detailed cache requirements (page_id -> size in KB)"
    )

    # Locality information
    spatial_locality: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Pages typically accessed together (page_id -> related_pages)"
    )
    temporal_locality: dict[str, int] = Field(
        default_factory=dict,
        description="Expected re-access intervals in seconds (page_id -> interval)"
    )


class SpawnedChildInfo(BaseModel):
    """Info about spawned child agent."""

    agent_id: str
    purpose: str
    plan_id: str | None = None
    spawned_at: float = Field(default_factory=time.time)
    # Note: Status is tracked via event subscriptions, NOT stored here

class PlanExecutionContext(BaseModel):
    """Strongly-typed execution context (replaces dict[str, Any])."""

    completed_action_ids: list[str] = Field(default_factory=list)
    action_results: dict[str, ActionResult] = Field(default_factory=dict)
    spawned_children: list[SpawnedChildInfo] = Field(default_factory=list)
    cache_state: dict[str, Any] = Field(default_factory=dict)
    findings: dict[str, Any] = Field(default_factory=dict)
    analyzed_pages: set[str] = Field(default_factory=set)
    synthesis_results: dict[str, Any] = Field(default_factory=dict)
    custom_data: dict[str, Any] = Field(default_factory=dict)  # Escape hatch


class ManualPlanSpec(BaseModel):
    """Specification for a manually-created plan."""

    actions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of action dictionaries (will be converted to Action objects)"
    )


class PlanningContext(BaseModel):
    """Strongly-typed planning context.
    
    This context is passed to planning methods to provide all necessary information
    for plan generation, evaluation, and execution.
    
    Fields:
        parent_plan_id: ID of parent plan if this is a sub-plan (for hierarchical planning)
        manual_plan: Optional manual plan specification (if provided, plan is created manually)
        execution_context: Current execution context with progress, findings, etc.
        bound_pages: List of page IDs this agent is bound to (for cache-aware planning)
        custom_data: Additional custom context data (escape hatch for extensibility)
    """
    parent_plan_id: str | None = Field(
        default=None,
        description="ID of parent plan if this is a sub-plan (for hierarchical planning)"
    )

    goals: list[str] = Field(
        default_factory=list,
        description="List of goals to achieve"
    )

    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution constraints (time limits, resource limits, etc.)"
    )

    action_descriptions: list[tuple[str, dict[str, str]]] = Field(
        default_factory=list,
        description="Descriptions of available action groups for planning by their keys in "
        "the action map of the ActionDispatcher. The action policy will get it from "
        "ActionDispatcher.get_action_descriptions(). Each tuple contains the group description and a dictionary of action descriptions."
    )

    manual_plan: ManualPlanSpec | None = Field(
        default=None,
        description="Optional manual plan specification. If provided, plan is created manually instead of LLM-generated."
    )

    page_ids: list[str] = Field(
        default_factory=list,
        description="List of page IDs relevant to the planning context (used for cache-aware planning)"
    )

    execution_context: PlanExecutionContext | None = Field(
        default=None,
        description="Current execution context with progress, findings, spawned children, etc."
    )

    recalled_memories: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Memories recalled from agent memory levels for this planning step. "
        "Populated by AgentContextEngine.gather_context() before LLM planning."
    )

    custom_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom context data (escape hatch for extensibility)"
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanningContext:
        """Create PlanningContext from dictionary (for backward compatibility).

        Args:
            data: Dictionary with context data
            
        Returns:
            PlanningContext instance

        Note:
            This method handles conversion of nested dicts to proper types.
        """
        # Convert execution_context dict to PlanExecutionContext if needed
        exec_context = data.get("execution_context")
        if exec_context is not None and isinstance(exec_context, dict):
            exec_context = PlanExecutionContext(**exec_context)

        # Convert manual_plan dict to ManualPlanSpec if needed
        manual_plan = data.get("manual_plan")
        if manual_plan is not None and isinstance(manual_plan, dict):
            manual_plan = ManualPlanSpec(**manual_plan)

        return cls(
            parent_plan_id=data.get("parent_plan_id"),
            manual_plan=manual_plan,
            execution_context=exec_context,
            page_ids=data.get("page_ids", []),
            recalled_memories=data.get("recalled_memories", []),
            custom_data=data.get("custom_data", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert PlanningContext to dictionary (for backward compatibility).

        Returns:
            Dictionary representation of context
        """
        result: dict[str, Any] = {}
        if self.parent_plan_id:
            result["parent_plan_id"] = self.parent_plan_id
        if self.manual_plan:
            result["manual_plan"] = self.manual_plan.model_dump()
        if self.execution_context:
            result["execution_context"] = self.execution_context
        if self.page_ids:
            result["page_ids"] = self.page_ids
        if self.recalled_memories:
            result["recalled_memories"] = self.recalled_memories
        if self.custom_data:
            result["custom_data"] = self.custom_data
        return result


class ReasoningContext(BaseModel):
    """Context for reasoning operations.
    
    Used in reasoning.py for goal-oriented reasoning loops.
    
    Fields:
        available_resources: Available resources (pages, tools, etc.)
        current_state: Current system state (pages_analyzed_count, etc.)
        constraints: Execution constraints (time, memory, etc.)
        findings: Current findings from analysis
        pending_queries: List of pending queries
        last_query_routing: Last query routing result
        last_query_answer: Last query answer
        custom_data: Additional custom context data
    """
    
    available_resources: dict[str, Any] = Field(default_factory=dict, description="Available resources (pages, tools, etc.)")
    current_state: dict[str, Any] = Field(default_factory=dict, description="Current system state (pages_analyzed_count, etc.)")
    constraints: dict[str, Any] = Field(default_factory=dict, description="Execution constraints (time, memory, etc.)")
    findings: dict[str, Any] = Field(default_factory=dict, description="Current findings from analysis")
    pending_queries: list[dict[str, Any]] = Field(default_factory=list, description="List of pending queries")
    last_query_routing: dict[str, Any] | None = Field(default=None, description="Last query routing result")
    last_query_answer: dict[str, Any] | None = Field(default=None, description="Last query answer")
    custom_data: dict[str, Any] = Field(default_factory=dict, description="Additional custom context data")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from context (dict-like access for backward compatibility).
        
        Checks current_state, findings, and custom_data in order.
        """
        if key in self.current_state:
            return self.current_state[key]
        if key in self.findings:
            return self.findings[key]
        if key in self.custom_data:
            return self.custom_data[key]
        # Special handling for common keys
        if key == "pending_queries":
            return self.pending_queries
        if key == "last_query_routing":
            return self.last_query_routing
        if key == "last_query_answer":
            return self.last_query_answer
        return default


class QueryContext(BaseModel):
    """Context for query generation and execution.
    
    Used in query patterns (query_driven.py, query/explorer.py, etc.).
    
    Fields:
        analysis_goal: Goal of the analysis
        current_findings: Current findings from analysis
        query_history: History of previous queries
        scope: Current analysis scope
        metadata: Additional metadata
        custom_data: Additional custom context data
    """
    
    analysis_goal: str | None = Field(default=None, description="Goal of the analysis")
    current_findings: list[dict[str, Any]] = Field(default_factory=list, description="Current findings from analysis")
    query_history: list[dict[str, Any]] = Field(default_factory=list, description="History of previous queries")
    scope: dict[str, Any] = Field(default_factory=dict, description="Current analysis scope")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    custom_data: dict[str, Any] = Field(default_factory=dict, description="Additional custom context data")
    direction: Literal["forward", "backward"] = Field(default="forward", description="Direction of the query routing.")


class AttentionContext(BaseModel):
    """Context for attention mechanisms.
    
    Used in attention.py and attention_policy.py for computing attention scores.
    
    Fields:
        analysis_context: Current analysis context
        findings: Findings from current analysis
        metadata: Optional metadata
        custom_data: Additional custom context data
    """
    source_page_id: str | None = Field(default=None, description="ID of the page being evaluated")
    analysis_context: dict[str, Any] = Field(default_factory=dict, description="Current analysis context")
    findings: list[dict[str, Any]] = Field(default_factory=list, description="Findings from current analysis")
    metadata: dict[str, Any] | None = Field(default=None, description="Optional metadata")
    custom_data: dict[str, Any] = Field(default_factory=dict, description="Additional custom context data")
    scope_clusters: list[str] = Field(
        default_factory=list,
        description="List of cluster IDs defining the current analysis scope."
    )
    current_pages: list[str] = Field(
        default_factory=list,
        description="List of page IDs currently loaded in the agent's working set."
    )


class RunContext(BaseModel):
    """Context for job/work coordination.
    
    Used in cache_coordination.py for working set management.
    
    Fields:
        analysis_goal: Goal of the analysis job
        run_id: Job identifier
        tenant_id: Tenant identifier
        priority: Job priority
        constraints: Job constraints (time, resources, etc.)
        custom_data: Additional custom context data
    """
    
    analysis_goal: str = Field(description="Goal of the analysis job")
    run_id: str | None = Field(default=None, description="Identifier of AgentRun")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    priority: int = Field(default=0, description="Job priority")
    constraints: dict[str, Any] = Field(default_factory=dict, description="Job constraints (time, resources, etc.)")
    custom_data: dict[str, Any] = Field(default_factory=dict, description="Additional custom context data")


class ErrorContext(BaseModel):
    """Context for error escalation.
    
    Used in base.py for error escalation to parent agents.
    
    Fields:
        error_type: Type of error
        error_details: Detailed error information
        action_context: Context about the action that failed
        recovery_suggestions: Suggested recovery actions
        custom_data: Additional custom context data
    """
    
    error_type: str | None = Field(default=None, description="Type of error")
    error_details: dict[str, Any] = Field(default_factory=dict, description="Detailed error information")
    action_context: dict[str, Any] = Field(default_factory=dict, description="Context about the action that failed")
    recovery_suggestions: list[str] = Field(default_factory=list, description="Suggested recovery actions")
    custom_data: dict[str, Any] = Field(default_factory=dict, description="Additional custom context data")



class PlanningParameters(BaseModel):
    """Parameters controlling planning behavior."""

    # Strategy
    strategy: PlanningStrategy = PlanningStrategy.MPC

    # Planning horizon (MPC)
    planning_horizon: int = 5
    replan_every_n_steps: int = 3
    replan_on_failure: bool = True

    # Plan complexity limits
    max_actions: int = 50
    max_hierarchical_depth: int = 5

    # Planning effort limits
    max_planning_time_seconds: float = 30.0
    max_planning_tokens: int = 10000

    # Convergence criteria for iterative planning
    max_iterations: int = 10
    min_improvement_threshold: float = 0.05
    quality_threshold: float = 0.8

    # Action granularity preferences
    prefer_atomic_actions: bool = False
    prefer_composite_actions: bool = False

    # Top-down vs bottom-up weight (0.0 = pure bottom-up, 1.0 = pure top-down)
    top_down_weight: float = 0.5

    # Dynamic agent allocation
    allow_child_spawning: bool = True
    max_child_agents: int = 10

    # Cache sizing
    ideal_cache_size: int = 0




# ============================================================================
# Execution History and Learning Models
# ============================================================================


class ExecutionOutcome(str, Enum):
    """Outcome of plan execution."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class PlanExecutionRecord(BaseModel):
    """Record of plan execution for learning and analysis.

    Captures complete execution history including costs, quality metrics,
    revisions, conflicts, and lessons learned.
    """

    plan_id: str
    agent_id: str
    goal: str
    scope: str  # What was being analyzed/processed
    strategy: str | None = None  # Planning strategy used

    # Plan structure
    actions: list[dict[str, Any]] = Field(default_factory=list)
    reasoning: str | None = None

    # Timing
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    duration_seconds: float = 0.0

    # Outcome
    outcome: ExecutionOutcome
    success_rate: float = 0.0  # Percentage of successful actions

    # Cost tracking
    actual_cost: dict[str, float] = Field(
        default_factory=dict,
        description="Actual costs (tokens, time, pages_loaded, etc.)"
    )
    estimated_cost: dict[str, float] = Field(
        default_factory=dict,
        description="Estimated costs from planning phase"
    )

    # Quality and performance
    quality_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Quality scores (accuracy, completeness, etc.)"
    )
    cache_stats: dict[str, float] = Field(
        default_factory=dict,
        description="Cache hit rate, pages loaded, etc."
    )

    # Adaptation tracking
    revision_count: int = 0
    revision_history: list[dict] = Field(default_factory=list)
    conflicts_encountered: int = 0
    coordination_events: list[dict] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)

    # Context snapshot
    context_snapshot: dict[str, Any] = Field(
        default_factory=dict,
        description="Relevant context at execution time"
    )

    # Lessons learned
    lessons_learned: list[str] = Field(
        default_factory=list,
        description="Insights from this execution"
    )

    def cost_accuracy(self) -> dict[str, float]:
        """Calculate accuracy of cost estimates.

        Returns:
            Dict mapping cost dimension to accuracy ratio (actual/estimated)
        """
        accuracy = {}
        for key in self.estimated_cost:
            if key in self.actual_cost and self.estimated_cost[key] > 0:
                accuracy[key] = self.actual_cost[key] / self.estimated_cost[key]
        return accuracy

    def was_efficient(self, max_cost_ratio: float = 1.2) -> bool:
        """Check if execution was efficient (didn't exceed estimates by much).

        Args:
            max_cost_ratio: Maximum acceptable actual/estimated ratio

        Returns:
            True if all costs within acceptable ratio
        """
        accuracies = self.cost_accuracy()
        return all(ratio <= max_cost_ratio for ratio in accuracies.values())


class PlanPattern(BaseModel):
    """Learned pattern from execution history.

    Represents a reusable planning pattern extracted from successful
    (or failed) executions, with applicability conditions and confidence.
    """

    pattern_id: str
    pattern_type: str  # "success", "failure", "cache_optimization", etc.
    description: str
    applicability: str  # When this pattern applies

    # Evidence
    supporting_executions: list[str] = Field(
        default_factory=list,
        description="Plan IDs that support this pattern"
    )
    confidence: float = 0.0  # 0.0 to 1.0

    # Conditions for applicability
    context_conditions: dict[str, Any] = Field(
        default_factory=dict,
        description="Context conditions when pattern applies"
    )

    # Pattern content
    recommended_actions: list[dict] = Field(
        default_factory=list,
        description="Recommended action sequence"
    )
    anti_patterns: list[str] = Field(
        default_factory=list,
        description="What to avoid"
    )

    # Performance metrics
    avg_success_rate: float = 0.0
    avg_cost_accuracy: float = 0.0
    avg_cache_efficiency: float = 0.0

    # Metadata
    discovered_at: float = Field(default_factory=time.time)
    last_validated: float = Field(default_factory=time.time)
    usage_count: int = 0


class TodoItemStatus(str, Enum):
    """Status of a todo item."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class TodoItem(BaseModel):
    """A single item in a hierarchical todo list.

    Supports recursive nesting for sub-tasks and dependencies.
    """

    item_id: str
    description: str
    status: TodoItemStatus = TodoItemStatus.PENDING

    # Hierarchical structure
    parent_item_id: str | None = None
    sub_items: list[str] = Field(default_factory=list)  # IDs of sub-items

    # Dependencies
    depends_on: list[str] = Field(default_factory=list)  # IDs of items that must complete first

    # Priority
    priority: int = 0  # Higher = more urgent/important

    # Tracking
    created_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None

    # Context
    assigned_agent_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionPlan(BaseModel):
    """Multi-goal hierarchical plan.
    Unified hierarchical plan model (supports both reasoning.py and planning package).

    A plan consists of ordered actions at a specific abstraction level.
    Actions can contain sub-plans for hierarchical decomposition.

    Supports:
    - Sequential execution (reasoning.py style)
    - Hierarchical decomposition (via action sub-plans)
    - Revision and backtracking
    - Progress tracking
    - TodoItem structure (optional)
    - Goal hierarchy CAN be LLM-generated OR manually specified

    Example (simple sequential):
        ```python
        plan = Plan(
            plan_id="plan_001",
            agent_id="agent_123",
            goal="Understand authentication flow",
            actions=[
                Action(action_type=ActionType.ROUTE_QUERY, ...),
                Action(action_type=ActionType.PROCESS_QUERY, ...),
                Action(action_type=ActionType.ANALYZE_PAGE, ...),
                Action(action_type=ActionType.SYNTHESIZE, ...),
            ],
            strategy="sequential"
        )
        ```

    Example (hierarchical with sub-plans):
        ```python
        plan = Plan(
            plan_id="plan_cluster",
            agent_id="coordinator",
            goal="Analyze authentication cluster",
            actions=[
                Action(
                    action_type=ActionType.ANALYZE_CLUSTER,
                    sub_plan=Plan(
                        goal="Analyze all pages",
                        actions=[...]
                    )
                )
            ]
        )
        ```
    """

    plan_id: str = Field(
        default_factory=lambda: f"action_plan_{uuid.uuid4()}",
        description="Unique identifier for the plan"
    )
    agent_id: str

    # Goals - support both single goal (reasoning.py) and multiple goals (planning package)
    goals: list[str] = Field(default_factory=list)  # Multiple goals for planning package

    # Goal hierarchy - CAN be LLM-generated OR manually specified
    # Format: {goal_id: {"goal": str, "sub_goals": [goal_ids], "parent": goal_id}}
    goal_hierarchy: dict[str, dict] = Field(default_factory=dict)

    # Generation method, Strategy and constraints
    generation_method: str = "manual"  # "llm" or "manual"
    strategy: str | None = None  # What strategy are we using?
    constraints: dict[str, Any] = Field(default_factory=dict)

    initial_reasoning: str | None = None

    # Action sequence at this level
    actions: list[Action] = Field(default_factory=list)
    current_action_index: int = 0

    # Planning parameters (planning package)
    planning_horizon: int = 5
    replan_every_n_steps: int = 3

    # Execution context (planning package)
    execution_context: PlanExecutionContext = Field(default_factory=PlanExecutionContext)

    # Cache context (planning package)
    cache_context: CacheContext = Field(default_factory=CacheContext)

    # Parent-child relationships (planning package)
    # NOTE: Parent doesn't track child status here - that's event-driven via blackboard
    parent_plan_id: str | None = None
    parent_agent_id: str | None = None
    child_plan_ids: list[str] = Field(default_factory=list)  # Child plan references
    depends_on: list[str] = Field(default_factory=list)

    # Multi-agent coordination
    subscribers: list[str] = Field(default_factory=list)  # Agent IDs subscribed to plan events
    scope: PlanScope | None = None  # Hierarchical scope (SYSTEM, CLUSTER, PAGE, SUBGOAL)

    # Progress tracking
    progress: float = 0.0  # Execution progress 0.0 to 1.0
    execution_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Historical execution events"
    )

    # Cost tracking
    estimated_cost: dict[str, float] = Field(
        default_factory=dict,
        description="Cost estimates (tokens, time, etc.)"
    )
    actual_cost: dict[str, float] = Field(
        default_factory=dict,
        description="Actual costs incurred during execution"
    )

    # Status (planning package)
    status: PlanStatus | None = None # PlanStatus.PROPOSED
    visibility: PlanVisibility = PlanVisibility.TEAM
    approval_required: bool = True
    approved_by: str | None = None
    blocked_reason: str | None = None

    # Revision history
    version: int = 1
    revision_history: list[dict[str, Any]] = Field(default_factory=list)

    # Timestamps
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None

    # Optional TodoItem structure for complex plans
    items: dict[str, TodoItem] = Field(default_factory=dict)
    root_items: list[str] = Field(default_factory=list)

    # Hierarchical context
    parent_action_id: str | None = None  # If this is a sub-plan
    abstraction_level: int = 0  # 0=top-level, 1=sub-plan, etc.

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Methods for reasoning.py compatibility
    def pop_next_action(self) -> Action | None:
        """Get and remove next action (reasoning.py style)."""
        if self.actions:
            return self.actions.pop(0)
        return None

    def has_actions(self) -> bool:
        """Check if plan has remaining actions (reasoning.py style)."""
        return len(self.actions) > 0

    def prepend_action(self, action: Action) -> None:
        """Add action to front of plan (urgent)."""
        self.actions.insert(0, action)
        self.updated_at = time.time()

    def append_action(self, action: Action) -> None:
        """Add action to end of plan."""
        self.actions.append(action)
        self.updated_at = time.time()

    # Methods for hierarchical planning
    def get_next_action(self) -> Action | None:
        """Get next action without removing it."""
        if self.current_action_index < len(self.actions):
            return self.actions[self.current_action_index]
        return None

    def advance(self) -> None:
        """Move to next action."""
        self.current_action_index += 1
        self.updated_at = time.time()

    def has_remaining_actions(self) -> bool:
        """Check if plan has remaining actions."""
        return self.current_action_index < len(self.actions)

    def is_complete(self) -> bool:
        """Check if plan is complete (planning package style)."""
        if self.status:
            return self.status == PlanStatus.COMPLETED
        # Fallback: check if all actions completed
        return all(
            action.status == ActionStatus.COMPLETED
            for action in self.actions
        )

    def get_pending_actions(self) -> list[Action]:
        """Get pending actions."""
        return [a for a in self.actions if a.status == ActionStatus.PENDING]

    def add_spawned_agent(self, agent_id: str, purpose: str, plan_id: str | None = None) -> None:
        """Record spawned child agent in execution context."""
        self.execution_context.spawned_children.append(
            SpawnedChildInfo(
                agent_id=agent_id,
                purpose=purpose,
                plan_id=plan_id,
            )
        )

    @staticmethod
    def get_plan_key(agent_id: str) -> str:
        """Get blackboard key for agent's plan (static method)."""
        return f"agent:{agent_id}:plan"


    @staticmethod
    def get_all_plans_key_pattern() -> str:
        return "agent:*:plan"

    # Modification methods
    def insert_action(self, action: Action, position: int | None = None) -> None:
        """Insert action at position (default: after current)."""
        if position is None:
            position = self.current_action_index + 1
        self.actions.insert(position, action)
        self.updated_at = time.time()

    def replace_action(self, index: int, new_action: Action) -> None:
        """Replace action at index."""
        self.actions[index] = new_action
        self.updated_at = time.time()

    def remove_action(self, index: int) -> Action:
        """Remove and return action at index."""
        action = self.actions.pop(index)
        if index < self.current_action_index:
            self.current_action_index -= 1
        self.updated_at = time.time()
        return action

    # Revision methods
    def revise(self, changes: dict[str, Any], reason: str = "") -> None:
        """Revise plan, keeping revision history."""
        self.revision_history.append({
            "version": self.version,
            "timestamp": self.updated_at,
            "reason": reason,
            "snapshot": {
                "actions": [a.model_dump() for a in self.actions],
                "current_action_index": self.current_action_index,
                "strategy": self.strategy,
            }
        })
        self.version += 1
        self.updated_at = time.time()
        for key, value in changes.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def backtrack_to_version(self, version: int) -> bool:
        """Backtrack to a previous version."""
        for snapshot in self.revision_history:
            if snapshot["version"] == version:
                self.actions = [
                    Action(**action_dict)
                    for action_dict in snapshot["snapshot"]["actions"]
                ]
                self.current_action_index = snapshot["snapshot"]["current_action_index"]
                self.strategy = snapshot["snapshot"]["strategy"]
                self.version = version
                self.updated_at = time.time()
                return True
        return False

    # TodoItem methods (optional structure)
    def add_item(self, item: TodoItem, parent_id: str | None = None) -> None:
        """Add a todo item to the plan."""
        self.items[item.item_id] = item
        if parent_id:
            item.parent_item_id = parent_id
            if parent_id in self.items:
                self.items[parent_id].sub_items.append(item.item_id)
        else:
            self.root_items.append(item.item_id)
        self.updated_at = time.time()

    # Hierarchy methods
    def get_all_actions_recursive(self) -> list[Action]:
        """Get all actions including sub-plans (BFS traversal)."""
        all_actions = []
        queue = list(self.actions)
        while queue:
            action = queue.pop(0)
            all_actions.append(action)
            if action.sub_plan:
                queue.extend(action.sub_plan.actions)
        return all_actions

    def get_total_action_count(self) -> int:
        """Count all actions including sub-plans."""
        return len(self.get_all_actions_recursive())

    def get_depth(self) -> int:
        """Get maximum depth of plan hierarchy."""
        max_depth = 0
        for action in self.actions:
            if action.sub_plan:
                sub_depth = action.sub_plan.get_depth()
                max_depth = max(max_depth, sub_depth + 1)
        return max_depth

    # -------------------------------------------------------------------------
    # Memory System Integration
    # -------------------------------------------------------------------------

    def get_blackboard_key(self, scope_id: str) -> str:
        """Generate blackboard key for storing this plan in memory.

        Args:
            scope_id: Memory scope ID (e.g., "agent:abc123:working")

        Returns:
            Key like "agent:abc123:working:plan:plan_001"
        """
        return f"{scope_id}:plan:{self.plan_id}"

    @staticmethod
    def get_key_pattern(scope_id: str) -> str:
        """Pattern for matching all plans in a scope.

        Args:
            scope_id: Memory scope ID

        Returns:
            Pattern like "agent:abc123:working:plan:*"
        """
        return f"{scope_id}:plan:*"



# ============================================================================
# Tool Models
# ============================================================================


class ToolParameterSchema(BaseModel):
    """Schema for a tool parameter."""

    name: str
    type: str  # "string", "int", "bool", etc.
    description: str
    required: bool = False
    default: Any = None


class ToolMetadata(BaseModel):
    """Metadata about a tool."""

    tool_id: str
    name: str
    description: str
    category: str  # "search", "code_analysis", "execution", etc.

    # Schema
    input_schema: list[ToolParameterSchema] = Field(default_factory=list)
    output_schema: dict[str, Any] = Field(default_factory=dict)

    # Usage
    usage_tips: list[str] = Field(default_factory=list)
    common_errors: list[str] = Field(default_factory=list)

    # Deployment info
    deployment_app_name: str
    deployment_name: str

    # Auth/config
    requires_auth: bool = False
    auth_token_key: str | None = None  # Key in auth dict
    rate_limit: int | None = None  # Calls per minute


class ToolCall(BaseModel):
    """A call to a tool by an agent."""

    call_id: str
    agent_id: str
    tool_id: str
    parameters: dict[str, Any]

    # Authentication
    auth_token: str | None = None

    status: ActionStatus = ActionStatus.PENDING
    result: Any = None
    error: str | None = None

    created_at: float = Field(default_factory=time.time)
    completed_at: float | None = None


# ============================================================================
# Communication Models
# ============================================================================


class ResultStatus(str, Enum):
    """Status of results written to blackboard by child agents.

    These are streamed via EnhancedBlackboard + EventBus for persistent,
    queryable result storage with push notifications.
    """
    PROGRESS_UPDATE = "progress"
    INTERMEDIATE_RESULT = "intermediate"
    FINAL_RESULT = "final"
    ERROR = "error"


class ChildResult(BaseModel):
    """Result written to blackboard by child agent.

    Children stream results to parents via blackboard writes. Parents
    subscribe to child result events and receive push notifications.

    Key encapsulation ensures single source of truth for blackboard keys.

    Example:
        ```python
        # Child publishes progress
        progress = ChildResult(
            result_status=ResultStatus.PROGRESS_UPDATE,
            role="analyzer",
            agent_id=child_id,
            sequence_num=0,
            progress_pct=0.5,
            current_phase="analyzing functions"
        )
        await board.write(
            key=progress.get_key(),
            value=progress.dict(),
            ttl_seconds=progress.get_ttl(),
            tags=progress.get_tags()
        )

        # Parent subscribes to all child results
        board.subscribe(
            callback=on_child_result,
            filter=KeyPatternFilter(ChildResult.key_pattern_all(child_id))
        )
        ```
    """
    result_status: ResultStatus
    role: str  # Role assigned by parent (e.g., "analyzer", "synthesizer")
    agent_id: str
    sequence_num: int  # Monotonically increasing
    timestamp: float = Field(default_factory=time.time)
    payload: dict[str, Any] = Field(default_factory=dict)

    # For progress updates
    progress_pct: float | None = None  # 0.0 to 1.0
    current_phase: str | None = None
    estimated_completion: float | None = None

    # For intermediate/final results
    data: Any | None = None

    # For errors
    error_type: str | None = None
    error_message: str | None = None
    is_recoverable: bool = True

    # ========================================================================
    # Key Encapsulation (NEVER hardcode these patterns elsewhere!)
    # ========================================================================

    @staticmethod
    def key_progress(agent_id: str) -> str:
        """Blackboard key for progress updates (ephemeral, TTL=300s)."""
        return f"child:{agent_id}:progress"

    @staticmethod
    def key_intermediate(agent_id: str, seq: int) -> str:
        """Blackboard key for intermediate results (persistent, queryable)."""
        return f"child:{agent_id}:intermediate:{seq}"

    @staticmethod
    def key_final(agent_id: str) -> str:
        """Blackboard key for final result (persistent)."""
        return f"child:{agent_id}:final"

    @staticmethod
    def key_error(agent_id: str, seq: int) -> str:
        """Blackboard key for errors (persistent, for debugging)."""
        return f"child:{agent_id}:error:{seq}"

    @staticmethod
    def key_pattern_all(agent_id: str) -> str:
        """Glob pattern to subscribe to ALL results from this child."""
        return f"child:{agent_id}:*"

    def get_key(self) -> str:
        """Get the blackboard key for this result based on its type."""
        if self.result_status == ResultStatus.PROGRESS_UPDATE:
            return self.key_progress(self.agent_id)
        elif self.result_status == ResultStatus.INTERMEDIATE_RESULT:
            return self.key_intermediate(self.agent_id, self.sequence_num)
        elif self.result_status == ResultStatus.FINAL_RESULT:
            return self.key_final(self.agent_id)
        elif self.result_status == ResultStatus.ERROR:
            return self.key_error(self.agent_id, self.sequence_num)
        else:
            raise ValueError(f"Unknown result type: {self.result_status}")

    def get_ttl(self) -> float | None:
        """Get TTL for this result type (None = persist forever).

        Returns:
            TTL in seconds for ephemeral results (progress updates),
            None for persistent results (intermediate, final, error).
        """
        if self.result_status == ResultStatus.PROGRESS_UPDATE:
            return 300.0  # Progress expires after 5 minutes
        else:
            return None  # Intermediate/final/error persist indefinitely

    def get_tags(self) -> set[str]:
        """Get tags for this result (for blackboard querying/filtering).

        Returns:
            Set of tags including result type, role, and special markers.
        """
        tags = {self.result_status.value, self.role}
        if self.result_status == ResultStatus.ERROR and not self.is_recoverable:
            tags.add("critical_error")
        return tags


class AgentState(str, Enum):
    """Agent lifecycle states."""

    INITIALIZED = "initialized"  # Just created
    RUNNING = "running"  # Actively executing
    WAITING = "waiting"  # Waiting for input/tool/etc.
    LOADED = "loaded"  # Page-bound agent with loaded pages
    UNLOADED = "unloaded"  # Page-bound agent with unloaded pages
    IDLE = "idle"  # Not actively doing anything
    SUSPENDED = "suspended"  # Suspended (resources freed, can be resumed)
    STOPPED = "stopped"  # Stopped gracefully
    FAILED = "failed"  # Failed with error




class ActionPolicyIterationResult(BaseModel):
    """Result of a single action policy iteration.

    Attributes:
        action: Action chosen to execute
        updated_state: Updated internal state after action
        error_context: Optional error context if action failed
    """

    policy_completed: bool = False
    success: bool
    error_context: ErrorContext | None = None
    requires_termination: bool = False
    blocked_reason: str | None = None

    # TODO: Do we need these fields?
    action_executed: Action | None = None
    result: ActionResult | None = None



class Ref(BaseModel):
    """Reference to a value in scope for dataflow between actions.

    References follow a path syntax:
        $variable           - Scope variable from current/parent scope
        $results.action_id  - Previous action's result
        $global.CapName     - Agent capability
        $shared.key         - Blackboard entry

    Examples:
        Ref.var("query")                      # Variable from scope
        Ref.result("route_001", "output.page_ids")  # Previous action result
        Ref.capability("QueryGenerator")      # Agent capability
        Ref.shared("analysis_findings")       # Blackboard entry
    """
    path: str = Field(description="Reference path (e.g., '$results.action_id.output')")

    @classmethod
    def var(cls, name: str) -> "Ref":
        """Reference a scope variable."""
        return cls(path=f"${name}")

    @classmethod
    def result(cls, action_id: str, output_path: str = "output") -> "Ref":
        """Reference a previous action's result.

        Args:
            action_id: ID of the action whose result to reference
            output_path: Path within the result (default: "output")
        """
        return cls(path=f"$results.{action_id}.{output_path}")

    @classmethod
    def capability(cls, name: str, attr: str = "") -> "Ref":
        """Reference an agent capability.

        Args:
            name: Capability class name
            attr: Optional attribute path within capability
        """
        path = f"$global.{name}"
        if attr:
            path += f".{attr}"
        return cls(path=path)

    @classmethod
    def shared(cls, key: str) -> "Ref":
        """Reference a blackboard entry.

        Args:
            key: Blackboard key
        """
        return cls(path=f"$shared.{key}")

    def is_result_ref(self) -> bool:
        """Check if this is a result reference."""
        return self.path.startswith("$results.")

    def is_var_ref(self) -> bool:
        """Check if this is a scope variable reference."""
        return self.path.startswith("$") and not any(
            self.path.startswith(p) for p in ["$results.", "$global.", "$shared."]
        )

    def is_capability_ref(self) -> bool:
        """Check if this is a capability reference."""
        return self.path.startswith("$global.")

    def is_shared_ref(self) -> bool:
        """Check if this is a blackboard reference."""
        return self.path.startswith("$shared.")

    def get_parts(self) -> list[str]:
        """Get path parts for navigation."""
        # Remove leading $ and split
        path = self.path[1:] if self.path.startswith("$") else self.path
        return path.split(".")



class ActionSharedDataDependency(BaseModel):
    """A single shared data dependency recorded on an Action.
    This is used to implement optimistic concurrency control (OCC)
    for shared data items accessed by multiple agents during planning and execution.
    That shared data may be read during planning by each agent to inform its plan,
    and updated during execution by actions within the plan. So, to avoid race
    conditions and ensure consistency, we use versioned data dependencies.

    When an agent plans its actions, it records the expected version of each
    shared data item it reads as a dependency. Then, before executing the plan,
    the dispatcher checks that the current version of each data item matches
    the expected version recorded by the agent. If any data item has changed
    (i.e., the versions don't match), the dispatcher rejects the plan execution
    and asks the agent to replan based on the updated data. If the versions match,
    the dispatcher allows the plan to proceed, by acquiring locks on the data items
    in a deterministic order (based on the data_key) to avoid deadlocks and ensure
    global mutual exclusion.

    `expected_version` can be any version token chosen by the data owner.
    The dispatcher validates it against the current version observed via the transactor.

    NOTE: Using digests or hashes instead of a monotonically increasing version number
    can result in ABA problems if the data changes back to a previous state.
    Agents should design their versioning scheme accordingly.
    """

    data_key: str = Field(
        description=(
            "Logical name of the data object (e.g., game state key). "
            "This is used as the key to read the data object from the transaction object. "
            "It is also used as the ordering key to sort the data dependencies in "
            "a deterministic order for lock acquisition to avoid deadlocks."
        )
    )
    data_value: Any = Field(description="Value of the data item read during planning")
    expected_version: Any = Field(description="Expected version of the data item used/read during planning")

    # Runtime-only transactors (not serialized). These are async context managers
    # that acquire coordination/transaction context for a dependency.
    transactor: AsyncContextManager[Any] = Field(
        default=None,
        description="Async context manager for data item transaction (runtime only, not serialized)"
    )


class PolicyREPL(ABC):

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        pass

    @abstractmethod
    def has(self, key: str) -> bool:
        pass

    @abstractmethod
    async def delete_variable(self, name: str) -> bool:
        """Delete a variable.

        Also deletes from backing store if it's a reference.

        Args:
            name: Variable name

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def list_variables(self) -> list[dict[str, Any]]:
        """List all variables with metadata for LLM planning context.

        Returns:
            List of variable info dicts with name, type, description, etc.
        """
        pass

    @abstractmethod
    def get_variable_summary(self) -> str:
        """Get formatted summary of all variables for LLM context.

        Returns:
            Multi-line string describing all REPL variables
        """
        pass

    @abstractmethod
    def set_result(self, action_id: str, result: "ActionResult") -> None:
        """Store action result by ID.

        Args:
            action_id: Action ID
            result: Action result
        """
        pass

    @abstractmethod
    def get_result(self, action_id: str) -> "ActionResult | None":
        """Get action result by ID.

        Args:
            action_id: Action ID

        Returns:
            ActionResult or None if not found
        """
        pass

    @abstractmethod
    def has_result(self, action_id: str) -> bool:
        """Check if result exists for action ID."""
        pass

    @abstractmethod
    @property
    def results(self) -> dict[str, "ActionResult"]:
        """Get all action results (copy)."""
        pass


class PolicyScope(BaseModel):
    """Variable scope for an `ActionPolicy` execution.

    NOTE: This should no longer be used. It is only kept for historical reasons.

    Similar to variable scopes in programming languages:
    - Actions can read/write variables in the current scope
    - Nested policies get child scopes (parent scope accessible via chain)
    - Parallel policies share parent scope (concurrent access)
    - Capabilities ($global) and blackboard ($shared) are global scope

    Example:
        ```python
        # Create scope
        scope = PolicyScope()

        # Set variables
        scope.set("query", "How does auth work?")
        scope.set("max_results", 10)

        # Get variables (checks parent scopes)
        query = scope.get("query")

        # Store action results
        scope.set_result("route_001", ActionResult(...))

        # Create child scope for nested policy (NO LONGER SUPPORTED - TODO - Remove child scopes?)
        child_scope = scope.child_scope()
        child_scope.set("local_var", "only in child")

        # Child can access parent variables
        assert child_scope.get("query") == "How does auth work?"
        ```
    """

    # Local variable bindings (set by actions in this policy)
    bindings: dict[str, Any] = Field(default_factory=dict)

    # Action result bindings (keyed by action_id)
    results: dict[str, ActionResult] = Field(default_factory=dict)

    # Parent scope reference (for nested policies) - stored as ID for serialization
    # TODO: Remove support for nested scopes? Nested policies are no longer supported.
    _parent: PolicyScope | None = None

    model_config = {"arbitrary_types_allowed": True}

    def get(self, key: str, default: Any = None) -> Any:
        """Get variable from scope (checks parent scopes).

        Args:
            key: Variable name
            default: Default value if not found

        Returns:
            Variable value or default
        """
        if key in self.bindings:
            return self.bindings[key]
        if self._parent:
            return self._parent.get(key, default)
        return default

    def set(self, key: str, value: Any) -> None:
        """Set variable in current scope.

        Args:
            key: Variable name
            value: Value to set
        """
        self.bindings[key] = value

    def has(self, key: str) -> bool:
        """Check if variable exists in scope chain.

        Args:
            key: Variable name

        Returns:
            True if variable exists in this or parent scope
        """
        if key in self.bindings:
            return True
        if self._parent:
            return self._parent.has(key)
        return False

    def get_result(self, action_id: str) -> ActionResult | None:
        """Get action result by ID (checks parent scopes).

        Args:
            action_id: Action ID

        Returns:
            ActionResult or None if not found
        """
        if action_id in self.results:
            return self.results[action_id]
        if self._parent:
            return self._parent.get_result(action_id)
        return None

    def set_result(self, action_id: str, result: ActionResult) -> None:
        """Store action result in scope.

        Args:
            action_id: Action ID
            result: Action result
        """
        self.results[action_id] = result

    def child_scope(self) -> PolicyScope:
        """Create child scope for nested policy.

        The child scope inherits from this scope (can read parent variables)
        but writes are local to the child.

        Returns:
            New child PolicyScope
        """
        child = PolicyScope()
        child._parent = self
        return child

    def merge_from(self, other: PolicyScope, keys: list[str] | None = None) -> None:
        """Merge variables from another scope into this scope.

        Args:
            other: Scope to merge from
            keys: Optional list of keys to merge (None = all)
        """
        if keys is None:
            keys = list(other.bindings.keys())
        for key in keys:
            if key in other.bindings:
                self.bindings[key] = other.bindings[key]

    def to_dict(self) -> dict[str, Any]:
        """Convert scope to dictionary for serialization."""
        return {
            "bindings": self.bindings.copy(),
            "results": {k: v.model_dump() for k, v in self.results.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicyScope":
        """Create scope from dictionary."""
        scope = cls()
        scope.bindings = data.get("bindings", {})
        for k, v in data.get("results", {}).items():
            scope.results[k] = ActionResult(**v)
        return scope

    # ---------------------------------------------------------------------
    # Shared State dependency tracking (for optimistic concurrency + multi-state coordination)
    # ---------------------------------------------------------------------

    # Tracked volatile dependencies
    # These are values (and their versions) read during planning of
    # an action that may change before execution of that action (e.g., due to
    # concurrent execution of another agent), causing the action executor to
    # abort so that the agent can replan based on updated state.
    shared_data_dependencies: dict[str, ActionSharedDataDependency] = Field(
        default_factory=dict,
        description=(
            "Shared state objects and expected versions used during action planning. "
            "The dispatcher can validate these dependencies before executing the action."
        ),
    )

    def set_shared(
        self,
        data_key: str,
        data_value: Any,
        data_version: Any,
        transactor: AsyncContextManager[Any],
    ) -> Action:
        """Record that this action depends on a particular version of data shared with other agents.

        This method is called by ActionPolicy.plan_step (NOT action executors).
        It allows the dispatcher to:
        - acquire coordination/transaction contexts in a deterministic order
        - validate that the planned-against state versions are still current
        - execute the action and commit changes atomically (by exiting transactors)

        Notes:
        - The `transactor` is runtime-only and is not serialized. If actions are
          persisted/suspended, the caller must ensure they are recreated.

        Args:
            `data_key`: Logical name of the data item (e.g., game state key).
            `data_value`: Value of the data item read during planning.
            `data_version`: Expected version used during planning.
            `transactor`: Runtime-only transactor that acquires coordination/transaction context for the data item.
                        It is a context manager that can be entered to acquire the transaction context.
                        The returned transaction object can be used to read/write the data item with
                        a key `data_key`.

        Returns:
            The action object itself.
        """
        # Replace existing dependency for same data_key (last write wins)
        self.shared_data_dependencies[data_key] = ActionSharedDataDependency(
            data_key=data_key,
            data_value=data_value,
            expected_version=data_version,
            transactor=transactor
        )
        return self

    def get_shared(self, data_key: str) -> Any:
        """Get shared data dependency value by key.

        Args:
            data_key: Logical name of the data item

        Returns:
            Any or None if not found
        """
        dep = self.shared_data_dependencies.get(data_key)
        if dep is not None:
            return dep.data_value
        return None

    def _has_shared_data_dependencies(self) -> bool:
        """Internal: check if action has any shared data dependencies."""
        return bool(self.shared_data_dependencies)

    def _get_sorted_shared_data_dependencies(self) -> list[ActionSharedDataDependency]:
        """Internal: get sorted list of shared data dependencies."""
        deps: list[ActionSharedDataDependency] = sorted(
            self.shared_data_dependencies.values(),
            key=lambda d: str(d.data_key),
        )
        return deps

    def _clear_shared_data_dependencies(self) -> None:
        """Internal: clear all shared data dependencies."""
        self.shared_data_dependencies.clear()




class ActionPolicyIO(BaseModel):
    """Contract for `ActionPolicy` inputs and outputs.

    Declares what variables a policy expects in its scope (inputs)
    and what variables it produces (outputs). This enables:
    - Type-safe composition of policies
    - Clear documentation of policy interface
    - Validation at policy start/end

    Example:
        ```python
        class MyPolicy(BaseActionPolicy):
            io = ActionPolicyIO(
                inputs={"query": str, "max_results": int},
                outputs={"page_ids": list[str], "analysis": dict}
            )
        ```
    """

    inputs: dict[str, type] = Field(
        default_factory=dict,
        description="Input variables expected in scope (name -> type)"
    )

    outputs: dict[str, type] = Field(
        default_factory=dict,
        description="Output variables produced by policy (name -> type)"
    )

    def validate_inputs(self, repl: PolicyREPL) -> list[str]:
        """Validate that required inputs are in scope.

        Args:
            repl: PolicyREPL to validate

        Returns:
            List of missing input variable names
        """
        missing = []
        for name in self.inputs:
            if not repl.has(name):
                missing.append(name)
        return missing

    def extract_outputs(self, repl: PolicyREPL) -> dict[str, Any]:
        """Extract declared outputs from scope.

        Args:
            repl: PolicyREPL to extract from

        Returns:
            Dictionary of output variable values
        """
        outputs = {}
        for name in self.outputs:
            value = repl.get(name)
            if value is not None:
                outputs[name] = value
        return outputs


class ActionPolicyExecutionState(BaseModel):
    """Holds the execution state of an ActionPolicy for suspension/resumption.

    This decouples the action policy instance from its execution state so that:
    - The same action policy instance can be used across multiple executions.
    - The same action policy instance can be used by other action policies.
    - The action policy can be re-instantiated on resumption if needed.

    Dataflow between actions within the policy is provided by PolicyPythonREPL
    via REPLCapability, not by this execution state. REPL state is managed by the capability and
    serialized with the agent's capabilities when suspending.

    For hierarchical composition (nested policies), spawn child agents instead
    of nesting policies. Use agent spawning and communication mechanisms.
    """

    current_plan: ActionPlan | None = None
    iteration_history: list[ActionPolicyIterationResult] = Field(default_factory=list)
    iteration_num: int = 0

    # Arbitrary custom state for the action policy
    custom: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary context state for the action policy"
    )

    def get_custom_field(
        self,
        field_name: str,
        default: Any = None) -> Any:
        """Get field from state context."""
        return self.custom[field_name] if field_name in self.custom else default

    def set_custom_field(
        self,
        field_name: str,
        value: Any) -> None:
        """Set field in state context."""
        self.custom[field_name] = value


# ============================================================================
# Agent Suspension State
# ============================================================================


class AgentSuspensionState(SharedState):
    """Persistent state for suspended agents.

    When agents are suspended (either self-suspended due to blocking dependencies
    or manager-suspended for resource management), their execution state must be
    preserved so they can resume seamlessly when conditions allow.

    This state is persisted to StateManager (Redis/etcd) and includes:
    1. Execution State: Plan progress, current action
    2. Communication State: Child tracking, subscriptions, message sequences
    3. Cache State: Working set, page access patterns, page graph snapshot

    The cache state is critical for VCM integration - when resuming an agent,
    we can route it to a replica that already has its working set loaded,
    or prefetch pages before resumption.

    Example:
        ```python
        # Suspend agent and save state
        state = AgentSuspensionState(
            agent_id=agent.agent_id,
            suspension_reason="resource_exhaustion",
            suspended_at=time.time(),

            # Execution state
            plan_id=agent.current_plan.plan_id,
            current_action_index=agent.current_plan.current_action_index,

            # Communication state
            child_agents=agent.child_agents,
            child_progress=agent.child_progress,
            parent_agent_id=agent.metadata.get("parent_agent_id"),

            # Cache state
            working_set_pages=list(agent.working_set_manager.working_set),
            page_access_counts=dict(agent.working_set_manager.page_access_counts),
            last_page_access_times=dict(agent.working_set_manager.last_access_time),
        )
        await state_manager.set(state.get_state_key(), state)

        # Resume agent
        state = await state_manager.get(
            AgentSuspensionState.get_state_key(app_name, agent_id)
        )
        await agent._restore_from_suspension(state)
        ```

    OS Analogy:
        - Agent = Process
        - Suspension = Process context switch / swap to disk
        - SuspensionState = Process control block (PCB)
        - Working set = Process memory pages
        - Resume = Restore process from PCB
    """

    # ========================================================================
    # Identity and Timing
    # ========================================================================

    agent_id: str = Field(description="Agent being suspended")
    agent_type: str = Field(
        description=(
            "Type/class of agent (e.g., 'code_analyzer', 'supervisor'). "
            "Required for resume_agent() to know which class to instantiate."
        )
    )
    suspended_at: float = Field(
        default_factory=time.time,
        description="Unix timestamp of suspension"
    )
    suspension_reason: str = Field(
        description=(
            "Why agent was suspended: "
            "'dependency_blocking' (waiting on children), "
            "'resource_exhaustion' (manager eviction), "
            "'cache_pressure' (VCM optimization), "
            "'load_balancing' (proactive rebalancing)"
        )
    )

    # ========================================================================
    # Execution State (Plan Progress)
    # ========================================================================

    plan_id: str | None = Field(
        default=None,
        description="PlanBlackboard key for current plan (already persisted separately)"
    )
    current_action_index: int | None = Field(
        default=None,
        description="Index of current/next action in plan"
    )
    agent_state: str | None = Field(
        default=None,
        description="AgentState enum value (RUNNING, BLOCKED, etc.)"
    )

    # The agent calls self.action_policy.serialize_suspension_state(state)
    # So, the policy can store whatever it needs here in the action_policy_state.
    action_policy_state: ActionPolicyExecutionState | None = Field(
        default=None,
        description="Execution state of the agent's ActionPolicy"
    )

    # ========================================================================
    # Communication State (Parent-Child Coordination)
    # ========================================================================

    # Child management
    child_agents: dict[str, str] = Field(
        default_factory=dict,
        description="Map of role -> agent_id for spawned children"
    )
    child_progress: dict[str, float] = Field(
        default_factory=dict,
        description="Map of role -> progress percentage (0.0-1.0)"
    )

    custom_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary custom data for each agent type."
    )

    # Parent communication
    parent_agent_id: str | None = Field(
        default=None,
        description="Parent agent ID if this is a child agent"
    )
    role: str | None = Field(
        default=None,
        description="Role assigned by parent (e.g., 'analyzer', 'synthesizer')"
    )

    # Blackboard subscriptions (need to re-subscribe on resume)
    blackboard_subscriptions: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of subscription metadata for re-subscribing on resume. "
            "Each entry: {'scope': 'shared', 'scope_id': '...', 'pattern': '...', 'handler': 'method_name'}"
        )
    )

    # ========================================================================
    # Resource State (For Resource Reclamation)
    # ========================================================================

    allocated_cpu_cores: float = Field(
        default=0.1,
        description="CPU cores allocated to this agent (freed on suspend)"
    )
    allocated_memory_mb: int = Field(
        default=512,
        description="Memory allocated to this agent (freed on suspend)"
    )
    allocated_gpu_cores: float = Field(
        default=0.0,
        description="GPU cores allocated (freed on suspend)"
    )
    allocated_gpu_memory_mb: int = Field(
        default=0,
        description="GPU memory allocated (freed on suspend)"
    )

    # ========================================================================
    # Cache State (VCM Integration) - NEW, critical for cache-aware resumption
    # ========================================================================

    working_set_pages: list[str] = Field(
        default_factory=list,
        description=(
            "List of VCM page IDs in agent's working set. "
            "These are the pages the agent was actively using before suspension. "
            "On resume, route agent to replica with most of these pages loaded, "
            "or prefetch them before resuming execution."
        )
    )

    page_access_counts: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Map of page_id -> access count. "
            "Tracks how frequently agent accessed each page. "
            "Used for prioritizing page prefetching on resume."
        )
    )

    last_page_access_times: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Map of page_id -> last access timestamp. "
            "Used for LRU-based page prioritization on resume."
        )
    )

    page_graph_snapshot: str | None = Field(
        default=None,
        description=(
            "Serialized NetworkX graph capturing page relationships "
            "(e.g., call graphs, dependency graphs). "
            "Used for prefetching related pages on resume. "
            "Format: base64-encoded pickled networkx.DiGraph"
        )
    )

    bound_pages: list[str] = Field(
        default_factory=list,
        description=(
            "Original bound_pages from AgentSpawnSpec. "
            "These are the pages the agent was routed to initially. "
            "Used for soft-affinity routing on resume."
        )
    )

    # ========================================================================
    # Resumption Metadata
    # ========================================================================

    resumption_priority: int = Field(
        default=0,
        description=(
            "Priority for resumption (higher = resume sooner). "
            "Used when multiple agents are suspended and resources become available."
        )
    )

    suspension_count: int = Field(
        default=1,
        description=(
            "Number of times this agent has been suspended. "
            "Used for fairness - agents suspended many times get higher priority."
        )
    )

    max_suspension_duration: float | None = Field(
        default=None,
        description=(
            "Optional TTL for suspension (seconds). "
            "If agent is suspended longer than this, it's garbage collected. "
            "None = no time limit (suspend indefinitely until resources available)."
        )
    )

    # ========================================================================
    # Key Management (StateManager Integration)
    # ========================================================================

    @staticmethod
    def get_state_key(app_name: str, agent_id: str) -> str:
        """Get StateManager key for this agent's suspension state.

        Args:
            app_name: Application name (from serving system)
            agent_id: Agent ID

        Returns:
            Redis/etcd key for storing suspension state
        """
        return f"polymathera:serving:{app_name}:agents:suspension:{agent_id}"

    def get_key(self, app_name: str) -> str:
        """Get StateManager key for this suspension state instance."""
        return self.get_state_key(app_name, self.agent_id)


# ============================================================================
# Agent Spawning Models
# ============================================================================


class AgentResourceRequirements(BaseModel):
    """Resource requirements for an agent.

    Specifies CPU, memory, and GPU resources needed by an agent.
    Used for resource-aware scheduling and capacity planning.

    Example:
        ```python
        # Lightweight agent
        lightweight = AgentResourceRequirements(
            cpu_cores=0.1,
            memory_mb=256,
        )

        # GPU-intensive agent
        gpu_agent = AgentResourceRequirements(
            cpu_cores=1.0,
            memory_mb=4096,
            gpu_cores=0.5,
            gpu_memory_mb=8192,
        )
        ```
    """

    cpu_cores: float = Field(
        default=0.1,
        ge=0.0,
        description="Fractional CPU cores (0.1 = 10% of 1 core)"
    )
    memory_mb: int = Field(
        default=512,
        ge=0,
        description="Memory in megabytes"
    )
    gpu_cores: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fractional GPU cores (0.0 = no GPU, 1.0 = full GPU)"
    )
    gpu_memory_mb: int = Field(
        default=0,
        ge=0,
        description="GPU memory in megabytes"
    )


class AgentMetadata(BaseModel):
    """Metadata for agents."""
    role: str | None = None
    tenant_id: str = "default"  # Tenant/namespace for multi-tenant deployments
    team_id: str | None = None
    session_id: str = Field(
        default="default",
        description=(
            "Session that created this agent. Used for tracking purposes only. "
            "An Agent is not associated with a session."
        )
    )
    run_id: str = Field(
        default="default",
        description=(
            "AgentRun that created this agent. Used for tracking purposes only. "
            "An Agent is not associated with a session or a run."
        )
    )
    parent_agent_id: str | None = None
    group_id: str | None = None

    goals: list[str] = Field(default_factory=list)

    # Optional page binding
    bound_pages: list[str] = Field(default_factory=list)

    max_iterations: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum reasoning loop iterations"
    )

    resuming_from_suspension: bool = False
    resumption_priority: int = 0

    action_policy_config: dict[str, Any] = Field(default_factory=dict)
    suspended_agent_id: str | None = None
    max_suspension_duration: float | None = None

    suspension_count: int = 0
    child_heartbeat_timeout: float = 300.0  # Default 5 minutes
    priority: int = 0
    parent_clarifications: list[Any] = Field(default_factory=list)
    performance_metrics: dict[str, Any] = Field(default_factory=dict)
    performance_last_updated: float = Field(default_factory=time.time)
    last_resumed_at: float = Field(default_factory=time.time)
    last_action_time: float = 0.0

    parameters: dict[str, Any] = Field(
        description="Additional parameters for the agent capabilities",
        default_factory=dict
    )

    def update(self, **kwargs) -> None:
        """Update metadata fields."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class AgentSpawnSpec(BaseModel):
    """Specification for spawning an agent.

    This model provides strong typing for agent creation parameters,
    including routing requirements for page-affinity agents.

    Example:
        ```python
        # Page-affinity agent with specific requirements
        spec = AgentSpawnSpec(
            agent_type="code_analyzer",
            bound_pages=["repo-123-context"],
            requirements=LLMClientRequirements(
                model_family="llama",
                min_context_window=32000,
                tenant_id="customer-123",
            ),
            metadata={"repo_id": "123"},
        )

        # Standalone agent (no page affinity or requirements)
        spec = AgentSpawnSpec(
            agent_type="supervisor",
            metadata={"role": "coordinator"},
        )
        ```

    Routing Logic:
        - If bound_pages or requirements specified → Page-affinity agent on VLLMDeployment
          - requirements used to select appropriate VLLM deployment
        - Otherwise → Standalone agent on StandaloneAgentDeployment
    """

    agent_type: str = Field(
        description="Type of agent to create (e.g., 'code_analyzer', 'supervisor', 'researcher')"
    )

    agent_id: str | None = Field(
        default=None,
        description="Optional unique ID (auto-generated if None)"
    )

    capabilities: list[str] = Field(
        default_factory=list,
        description="List of capability class paths to attach to the agent"
    )

    action_policy: str | None = Field(
        default=None,
        description="Action policy class path to use for the agent"
    )

    bound_pages: list[str] = Field(
        default_factory=list,
        description="Virtual page IDs this agent is bound to (empty for non-affinity agents)"
    )

    requirements: Any | None = Field(  # Import LLMClientRequirements would be circular, so use Any
        default=None,
        description="LLM deployment requirements for routing (LLMClientRequirements)"
    )

    resource_requirements: AgentResourceRequirements = Field(
        default_factory=AgentResourceRequirements,
        description="CPU/memory/GPU requirements for this agent"
    )

    metadata: AgentMetadata = Field(
        default_factory=AgentMetadata,
        description="Additional metadata for the agent"
    )

    def has_deployment_affinity(self) -> bool:
        """Check if this agent requires specific deployment (vs standalone).

        Returns:
            True if agent has page bindings or LLM requirements
        """
        return bool(self.bound_pages) or (self.requirements is not None)


# ============================================================================
# Resource Exhausted Handling
# ============================================================================


class ResourceExhaustedStrategy(str, Enum):
    """Strategy for handling ResourceExhausted errors during agent spawning.

    When an agent spawn fails due to resource exhaustion, the system
    can apply different strategies depending on the agent type and deployment.
    """

    SCALE_DEPLOYMENT = "scale_deployment"
    """Scale the deployment by adding replicas (standalone agents only)."""

    REPLICATE_PAGES = "replicate_pages"
    """Replicate VCM pages to new VLLM replica (VLLM-resident agents only)."""

    SOFT_CONSTRAINT = "soft_constraint"
    """Violate page affinity, schedule to replica without pages loaded."""

    SUSPEND_AGENTS = "suspend_agents"
    """Suspend existing agents to make room for new agent."""


class ResourceExhaustedConfig(BaseModel):
    """Configuration for handling ResourceExhausted errors during agent spawning.

    This configures how the system responds when agent deployment replicas
    run out of capacity (CPU, memory, GPU, or agent count limits).

    Example:
        ```python
        config = ResourceExhaustedConfig(
            standalone_strategy=ResourceExhaustedStrategy.SCALE_DEPLOYMENT,
            vllm_strategy_order=[
                ResourceExhaustedStrategy.REPLICATE_PAGES,
                ResourceExhaustedStrategy.SOFT_CONSTRAINT,
            ],
            max_retries=3,
            retry_delay_s=2.0,
        )

        agent_system = AgentSystemDeployment(
            resource_exhausted_config=config
        )
        ```
    """

    standalone_strategy: ResourceExhaustedStrategy = Field(
        default=ResourceExhaustedStrategy.SCALE_DEPLOYMENT,
        description="Strategy for standalone agents (default: scale deployment)"
    )

    vllm_strategy_order: list[ResourceExhaustedStrategy] = Field(
        default_factory=lambda: [
            ResourceExhaustedStrategy.REPLICATE_PAGES,
            ResourceExhaustedStrategy.SOFT_CONSTRAINT,
            ResourceExhaustedStrategy.SUSPEND_AGENTS,
        ],
        description="Ordered list of strategies to try for VLLM-resident agents"
    )

    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retry attempts after ResourceExhausted"
    )

    retry_delay_s: float = Field(
        default=2.0,
        ge=0.0,
        description="Delay between retry attempts (seconds)"
    )

    page_replication_timeout_s: float = Field(
        default=60.0,
        ge=0.0,
        description="Timeout for page replication requests (seconds)"
    )

    enable_auto_scaling: bool = Field(
        default=True,
        description="Enable automatic scaling on ResourceExhausted"
    )

