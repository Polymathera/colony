"""Types for the agent memory system.

This module provides shared types for memory capabilities:
- MemoryQuery: Query parameters for recalling memories
- MemorySubscription: Configuration for listening to data types to ingest from other scopes (subscriptions)
- MemoryRecord: Generic container for LLM-produced dict data (auto-wraps in store())
- MemoryProducerConfig: Configuration for hook-based memory capture
- MaintenanceResult: Result of maintenance operations
- RetrievalContext: Context for goal-aware retrieval
- ScoredEntry: Entry with retrieval score
- MemoryLens: Read-only view over memory with custom filtering
- ConsolidationContext: Context passed to consolidation transformers
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Awaitable, Literal

from pydantic import BaseModel, Field

from ....distributed.hooks import Pointcut, HookContext

if TYPE_CHECKING:
    from ...blackboard.types import BlackboardEntry
    from ...models import Action


class TagFilter(BaseModel):
    """Logical filter expression over entry tags.

    Supports AND (all_of), OR (any_of), NOT (none_of) combinators.
    The LLM planner constructs these to do precise tag-based retrieval.

    Example:
        ```python
        # Entries that are actions AND successful
        TagFilter(all_of={"action", "success"})

        # Entries that are either infer or plan actions
        TagFilter(any_of={"action_type:infer", "action_type:plan"})

        # Successful actions, excluding infer
        TagFilter(all_of={"action", "success"}, none_of={"action_type:infer"})
        ```
    """

    all_of: set[str] = Field(
        default_factory=set,
        description="Entry must have ALL of these tags",
    )
    any_of: set[str] = Field(
        default_factory=set,
        description="Entry must have at least ONE of these tags",
    )
    none_of: set[str] = Field(
        default_factory=set,
        description="Entry must have NONE of these tags",
    )

    def matches(self, entry_tags: set[str]) -> bool:
        """Check if a set of entry tags satisfies this filter."""
        if self.all_of and not self.all_of.issubset(entry_tags):
            return False
        if self.any_of and not self.any_of.intersection(entry_tags):
            return False
        if self.none_of and self.none_of.intersection(entry_tags):
            return False
        return True

    @property
    def is_empty(self) -> bool:
        return not self.all_of and not self.any_of and not self.none_of


class MemoryQuery(BaseModel):
    """Query parameters for recalling memories.

    Used by `MemoryCapability.recall()` and `AgentContextEngine.gather_context()`.

    Supports three query modes based on which fields are populated:
    - **Semantic**: `query` text for vector similarity search
    - **Logical**: `tag_filter`, `time_range`, `key_pattern` for structured filtering
    - **Hybrid**: both semantic + logical (results are filtered then ranked by similarity)

    The LLM planner constructs MemoryQuery objects. Use `list_tags` action to
    discover available tags before constructing tag-based queries.

    Example:
        ```python
        # Semantic query
        MemoryQuery(query="What authentication approach was used?")

        # Logical query by tags
        MemoryQuery(tag_filter=TagFilter(all_of={"action", "success"}))

        # Hybrid: semantic + tag filter
        MemoryQuery(
            query="security analysis results",
            tag_filter=TagFilter(any_of={"action_type:infer", "action_type:plan"}),
            max_results=10,
        )
        ```
    """

    # Semantic search
    query: str | None = Field(
        default=None,
        description="Natural language query for semantic similarity search",
    )

    # Logical filters
    tag_filter: TagFilter = Field(
        default_factory=TagFilter,
        description="Tag filter with AND/OR/NOT logic for precise retrieval",
    )

    time_range: tuple[float, float] | None = Field(
        default=None,
        description="(start_timestamp, end_timestamp) absolute time filter",
    )

    key_pattern: str | None = Field(
        default=None,
        description="Key glob pattern filter (e.g., 'scope:Action:*')",
    )

    # Result control
    max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return"
    )

    min_relevance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score (0-1). Applied to semantic scores.",
    )

    include_expired: bool = Field(
        default=False,
        description="Include memories past their TTL (for audit/debug)"
    )

    max_age_seconds: float | None = Field(
        default=None,
        description="Only return memories created within this time window"
    )

    @property
    def has_semantic(self) -> bool:
        """Whether this query requests semantic similarity search."""
        return bool(self.query)

    @property
    def has_logical(self) -> bool:
        """Whether this query has logical filter constraints."""
        return (
            not self.tag_filter.is_empty
            or self.time_range is not None
            or self.key_pattern is not None
            or self.max_age_seconds is not None
        )


class MemoryRecord(BaseModel):
    """Generic container for storing arbitrary data in the memory system.

    When the LLM plans a memory store action, it produces plain JSON dicts.
    The memory system requires data objects to have a ``record_id`` property.
    ``MemoryRecord`` bridges this gap: ``MemoryCapability.store()`` auto-wraps
    raw dicts into a ``MemoryRecord`` so they satisfy the blackboard protocol.

    Example::

        record = MemoryRecord(
            content={"task": "impact_analysis", "repo_id": "my-project"},
            tags={"task_context", "config"},
        )
    """
    record_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:8],
        description="Unique ID for this record (default: random hash)",
    )
    content: dict[str, Any] = Field(
        description="The stored data payload.",
    )
    tags: set[str] = Field(
        default_factory=set,
        description="Tags for categorization and retrieval.",
    )
    created_at: float = Field(
        default_factory=time.time,
        description="Timestamp when this record was created.",
    )


@dataclass
class MemorySubscription:
    """Configuration for a memory capability to pull data from another scope.

    Each subscription defines a source for data ingestion. The capability:
    1. Listens for new entries matching `data_type` in `source_scope_id`
    2. Applies optional `filters` to select relevant entries
    3. Collects entries into pending queue

    The capability's `ingestion_transformer` (at capability level, not per-subscription)
    processes ALL pending entries together before writing to this scope.

    Example:
        ```python
        # STM pulls from working memory and sensory memory
        stm = MemoryCapability(
            agent=agent,
            scope_id=MemoryScope.agent_stm(agent),
            ingestion_policy=MemoryIngestPolicy(
                subscriptions=[
                    MemorySubscription(source_scope_id=MemoryScope.agent_working(agent)),
                    MemorySubscription(source_scope_id=MemoryScope.agent_sensory(agent)),
                ],
                transformer=SummarizingTransformer(  # Consolidates ALL inputs
                    agent=agent,
                    prompt="Summarize inputs into coherent STM entries.",
                ),
        )
        ```
    """

    source_scope_id: str
    """Blackboard scope to listen to."""

    data_type: type[BaseModel] | None = None
    """Data type to subscribe to."""

    key_pattern: str | None = None
    """Optional key pattern to filter events from the source scope.
    Can be an arbitrary glob pattern (e.g., "agent:working:*") or a specific pattern defined by the data type. If None, subscribes to "*" which matches all data in the source scope."""

    filters: list[Callable[["BlackboardEntry"], bool]] | None = None
    """Optional filter predicates to select entries."""


@dataclass
class MaintenanceConfig:
    """Configuration for memory level maintenance policies.

    Each memory level can have different maintenance behaviors:
    - Decay: Reduce relevance over time
    - Pruning: Remove low-value memories
    - Deduplication: Merge similar memories
    - Reindexing: Update embeddings periodically

    Example:
        ```python
        stm = MemoryCapability(
            agent=agent,
            scope_id=MemoryScope.agent_stm(agent),
            maintenance=MaintenanceConfig(
                decay_rate=0.01,  # 1% per minute
                prune_threshold=0.1,  # Remove if relevance < 10%
                dedup_threshold=0.95,  # Merge if similarity > 95%
            ),
        )
        ```
    """

    # Decay settings
    decay_rate: float = 0.0
    """Relevance decay rate per minute (0 = no decay)."""

    decay_min: float = 0.0
    """Minimum relevance after decay (floor value)."""

    # Pruning settings
    prune_threshold: float = 0.0
    """Relevance threshold below which to prune (0 = no pruning)."""

    prune_interval_seconds: float = 300.0
    """How often to run pruning (default: 5 minutes)."""

    # Deduplication settings
    dedup_threshold: float = 0.95
    """Similarity threshold above which to deduplicate (0.95 = 95% similar)."""

    dedup_interval_seconds: float = 600.0
    """How often to run deduplication (default: 10 minutes)."""

    # Reindexing settings
    reindex_interval_seconds: float = 3600.0
    """How often to reindex embeddings (default: 1 hour)."""

    # Access tracking
    track_access: bool = True
    """Whether to track access counts and timestamps."""


# Type alias for memory extractor functions
# Extractor receives hook context and method result, returns storable data or None, tags, metadata
MemoryTagsMetadataTuple = tuple[BaseModel | None, set[str], dict[str, Any]]

MemoryExtractor = (
    Callable[["HookContext", Any], MemoryTagsMetadataTuple] |
    Callable[["HookContext", Any], Awaitable[MemoryTagsMetadataTuple]]
)


@dataclass
class MemoryProducerConfig:
    """Configuration for hook-based memory capture.

    Memory capabilities use this to "observe" agent behavior by registering
    hooks on memory-producing methods. When the hooked method returns, the
    memory capability extracts storable data and stores it.

    This implements the principle: "memory is an observer of agent behavior."

    Attributes:
        pointcut: Which method(s) to observe (e.g., Pointcut.pattern("*.dispatch"))
        extractor: Function to extract storable data from hook context and result.
                   If None, stores the return value directly (must be BaseModel).
        priority: Hook priority (higher = runs later). Default 1000 for memory hooks.
        ttl_seconds: Time to live for the stored memory entries.
        tags_extractor: Function to extract tags from hook context and result.
        metadata_extractor: Function to extract metadata from hook context and result.

    Example:
        ```python
        # Working memory observes action execution
        working_memory = MemoryCapability(
            agent=agent,
            scope_id=MemoryScope.agent_working(agent),
            producers=[
                # Store completed actions (extract from args, not return value)
                MemoryProducerConfig(
                    pointcut=Pointcut.pattern("ActionDispatcher.dispatch"),
                    extractor=lambda ctx, result: (
                        ctx.args[0],  # Action is first arg
                        {"action", f"action_type:{ctx.args[0].action_type}"},
                        {"action_created_at": ctx.args[0].created_at}
                    ),
                    ttl_seconds=3600,  # 1 hour
                ),
                # Store plans directly (return value is the plan)
                MemoryProducerConfig(
                    pointcut=Pointcut.pattern("*._create_initial_plan"),
                    extractor=lambda ctx, result: (
                        result,
                        {"plan", f"plan_type:{result.plan_type}"},
                        {"plan_created_at": result.created_at, "plan_updated_at": result.updated_at}
                    ),
                    ttl_seconds=3600,  # 1 hour
                ),
            ],
        )
        ```

    Note:
        The pointcut must match `@hookable` methods. If a method is not hookable,
        the hook registration will succeed but the hook will never fire.
    """

    pointcut: Pointcut
    """Pointcut specifying which method(s) to observe."""

    extractor: MemoryExtractor | None = None
    """
    Function to extract storable data from hook context and result.
    
    Signature: (ctx: HookContext, result: Any) -> BaseModel | None
    
    - `ctx.args`: Positional arguments passed to the method
    - `ctx.kwargs`: Keyword arguments passed to the method
    - `ctx.instance`: The object whose method was called
    - `result`: The method's return value
    
    Return None to skip storage (e.g., for filtering).
    If None (default), stores the result directly (must have record_id).
    """

    priority: int = 1000
    """Hook priority. Higher values run later in the hook chain.
    Default 1000 ensures memory hooks run after other hooks."""

    ttl_seconds: float | None = None
    """Time to live for the stored memory entries.
    If None, uses the memory level's default TTL.
    """

def extract_event_from_event_driven_policy(ctx: HookContext, result: Any) -> MemoryTagsMetadataTuple:
    return result, {"observation", "event", f"event_type:{result.event_type}"}, {
        "event_type": result.event_type,
        "event_id": result.event_id,
        "event_created_at": result.created_at,
        "event_updated_at": result.updated_at,
    },

# Common extractors for standard memory producers
def extract_action_from_dispatch(ctx: HookContext, result: Any) -> MemoryTagsMetadataTuple:
    """Extract the Action (with result attached) from dispatch() call.

    ActionDispatcher.dispatch(action, scope) -> ActionResult
    We want to store the Action (which has the result attached), not the ActionResult.

    Also extracts _reflection_learnings and _critique_learnings from the result output
    for use by ReflectionCapability and CriticCapability when gathering context from memory.
    """
    if ctx.args:
        action = ctx.args[0]  # First positional arg is the Action
        tags: set[str] = {"action", f"action_type:{action.action_type}"}
        metadata: dict[str, Any] = {"action_created_at": action.created_at}

        # Extract learnings from result if available
        # Action executors write learnings to result.output["_reflection_learnings"] and ["_critique_learnings"]
        if action.result and action.result.output and isinstance(action.result.output, dict):
            # Extract reflection learnings for ReflectionCapability
            reflection_learnings = action.result.output.get("_reflection_learnings")
            if reflection_learnings:
                metadata["_reflection_learnings"] = reflection_learnings

            # Extract critique learnings for CriticCapability
            critique_learnings = action.result.output.get("_critique_learnings")
            if critique_learnings:
                metadata["_critique_learnings"] = critique_learnings

        # Add success/failure tag for easier filtering
        if action.result:
            if action.result.success:
                tags.add("success")
            else:
                tags.add("failure")

        return action, tags, metadata
    return None, set(), {}


def extract_plan_from_policy(ctx: HookContext, result: Any) -> MemoryTagsMetadataTuple:
    """Extract ActionPlan after plan creation.
    
    The plan is typically stored on the policy, not returned directly.
    Access via ctx.instance.current_plan after the method completes.
    """
    ### instance = ctx.instance
    ### if hasattr(instance, 'current_plan') and instance.current_plan:
    ###     return instance.current_plan
    ### return None
    return result, {"plan", f"status:{result.status}"}, {
        "action_count": len(result.actions),
        "goals": result.goals,
    }



def extract_terminal_game_state(ctx: HookContext, result: Any) -> MemoryTagsMetadataTuple:
    """Extract terminal game state from submit_move() call.
    
    GameProtocolCapability.submit_move() -> (success, reason, GameState)
    Only returns the GameState if the game is terminal.
    """
    if isinstance(result, tuple) and len(result) >= 3:
        success, reason, game_state = result[0], result[1], result[2]
        if game_state and hasattr(game_state, 'is_terminal'):
            if game_state.is_terminal() or (hasattr(game_state, 'phase') and 
                    hasattr(game_state.phase, 'value') and game_state.phase.value == 'terminal'):

                tags = {
                    "game_experience",
                    f"game_type:{game_state.game_type}",
                    f"outcome:{game_state.outcome.outcome_type if game_state.outcome else 'unknown'}",
                }
                if game_state.outcome and game_state.outcome.success:
                    tags.add("success")
                else:
                    tags.add("failure")

                # Get agent_id from the capability instance (ctx.instance is GameProtocolCapability)
                agent_id = ctx.instance.agent.agent_id if hasattr(ctx.instance, 'agent') else None

                metadata = {
                    "duration_s": (game_state.ended_at or time.time()) - game_state.started_at if game_state.started_at else None,
                    "move_count": len(game_state.history),
                    "participants": game_state.participants,
                    "my_role": game_state.get_role(agent_id) if agent_id else None,
                }
                return game_state, tags, metadata

    return None, set(), {}



def extract_reflection(ctx: HookContext, result: Any) -> MemoryTagsMetadataTuple:
    """Extract Reflection from ReflectionCapability.reflect() call."""

    tags = {"reflection", f"confidence:{int(result.confidence * 10)}/10"}
    if result.needs_more_info:
        tags.add("needs_followup")

    metadata = {
        "learned_count": len(result.learned),
        "assumptions_violated": len(result.assumptions_violated),
    }

    return result, tags, metadata


def extract_critique(ctx: HookContext, result: Any) -> MemoryTagsMetadataTuple:
    """Extract Critique from CritiqueCapability.critique() call."""

    tags = {"critique", f"quality:{int(result.quality_score * 10)}/10"}
    if result.requires_replanning:
        tags.add("requires_replanning")
    if result.requires_revision:
        tags.add("requires_revision")

    metadata = {
        "issues_count": len(result.issues),
        "suggestions_count": len(result.suggestions),
    }

    return result, tags, metadata


# =============================================================================
# Retrieval Types
# =============================================================================


@dataclass
class RetrievalContext:
    """Context for goal-aware retrieval.

    Justification: <mark>Embedding similarity ≠ task relevance.</mark>
    Relevance depends on current goal and agent state.

    Attributes:
        current_goal: Current task or goal description
        agent_state: Relevant agent state for context
        recent_actions: Recent actions for temporal context
    """

    current_goal: str | None = None
    """Current task or goal description for relevance scoring."""

    agent_state: dict[str, Any] | None = None
    """Relevant agent state for context-aware retrieval."""

    recent_actions: list["Action"] | None = None
    """Recent actions for temporal context."""


@dataclass
class ScoredEntry:
    """Entry with retrieval score and component breakdown.

    Used by RetrievalStrategy to return scored results with explanations.

    Attributes:
        entry: The retrieved blackboard entry
        score: Overall relevance score (0-1)
        components: Breakdown of score components (e.g., recency, similarity)
    """

    entry: "BlackboardEntry"
    """The retrieved blackboard entry."""

    score: float
    """Overall relevance score (0-1)."""

    components: dict[str, float] = field(default_factory=dict)
    """Score component breakdown, e.g., {"recency": 0.3, "similarity": 0.6}."""


# =============================================================================
# Maintenance Types
# =============================================================================


@dataclass
class MaintenanceResult:
    """Result of a maintenance operation.

    Returned by MaintenancePolicy.execute() to report what was done.

    Attributes:
        entries_processed: Total entries examined
        entries_removed: Entries deleted
        entries_modified: Entries updated (e.g., decayed)
        duration_seconds: Time taken for the operation
    """

    entries_processed: int = 0
    """Total entries examined."""

    entries_removed: int = 0
    """Entries deleted."""

    entries_modified: int = 0
    """Entries updated (e.g., decayed)."""

    duration_seconds: float = 0.0
    """Time taken for the operation."""


@dataclass
class ConsolidationContext:
    """Context passed to ConsolidationTransformer.consolidate().

    Provides information about the consolidation operation for transformers
    that need context-aware processing.

    Attributes:
        source_scope: Scope entries are being consolidated from
        target_scope: Scope entries will be written to
        agent_goals: Current agent goals for goal-aware consolidation
        task_description: Current task for task-aware summarization
    """

    source_scope: str
    """Scope entries are being consolidated from."""

    target_scope: str
    """Scope entries will be written to."""

    agent_goals: list[str] | None = None
    """Current agent goals for goal-aware consolidation."""

    task_description: str | None = None
    """Current task for task-aware summarization."""


# =============================================================================
# Memory Lens
# =============================================================================


@dataclass
class MemoryLens:
    """Read-only view over memory with custom filtering/ranking.

    Justification: <mark>Different contexts need different perspectives.</mark>
    - Planning: Recent actions, current goals, immediate obstacles
    - Reflection: Past successes/failures, patterns, lessons learned
    - Learning: Examples similar to current task

    A lens does NOT copy data—it's a configured query interface.

    Attributes:
        name: Identifier for this lens
        description: Human-readable description
        scopes: Which memory scopes to query
        tags_include: Only include entries with these tags
        tags_exclude: Exclude entries with these tags
        time_range: Relative time range (seconds before now)
        max_results: Maximum results to return

    Example:
        ```python
        PLANNING_LENS = MemoryLens(
            name="planning",
            description="Recent context for action planning",
            scopes=["working", "stm"],
            time_range=(-3600, 0),  # Last hour
            max_results=10,
        )
        ```
    """

    name: str
    """Identifier for this lens."""

    description: str = ""
    """Human-readable description."""

    scopes: list[str] = field(default_factory=list)
    """Which memory scopes to query."""

    data_types: list[type[BaseModel]] | None = None
    """Filter to specific data types."""

    tags_include: set[str] | None = None
    """Only include entries with these tags."""

    tags_exclude: set[str] | None = None
    """Exclude entries with these tags."""

    time_range: tuple[float, float] | None = None
    """Relative time range (seconds from now). E.g., (-3600, 0) = last hour."""

    max_results: int = 20
    """Maximum results to return."""

    context_formatter: Callable[[list["BlackboardEntry"]], str] | None = None
    """Optional formatter to convert entries to context string."""


# =============================================================================
# Predefined Lenses
# =============================================================================

PLANNING_LENS = MemoryLens(
    name="planning",
    description="Recent context for action planning",
    scopes=["working", "stm"],
    time_range=(-3600, 0),  # Last hour
    max_results=10,
)

REFLECTION_LENS = MemoryLens(
    name="reflection",
    description="Past experiences for self-reflection",
    scopes=["stm", "ltm:episodic"],
    tags_include={"action_result", "game_experience"},
    max_results=20,
)

SKILL_LENS = MemoryLens(
    name="skills",
    description="Procedural knowledge for task execution",
    scopes=["ltm:procedural"],
    max_results=5,
)


# =============================================================================
# Memory Introspection Types
# =============================================================================


@dataclass
class MemoryScopeInfo:
    """Information about a single memory scope for introspection.

    Used by AgentContextEngine.inspect_memory_map() to give the LLM
    a structured view of the memory layout.
    """

    scope_id: str
    """Scope identifier (e.g., 'agent:abc123:stm')."""

    scope_type: str
    """Human-readable type: 'working', 'stm', 'ltm:episodic', etc."""

    purpose: str
    """What this scope stores and why."""

    # Configuration
    ttl_seconds: float | None = None
    """Time-to-live for entries (None = no expiration)."""

    max_entries: int | None = None
    """Maximum capacity (None = no limit)."""

    # Current state
    entry_count: int = 0
    """Current number of entries in scope."""

    oldest_entry_age_seconds: float | None = None
    """Age of oldest entry."""

    newest_entry_age_seconds: float | None = None
    """Age of newest entry."""

    # Dataflow relationships
    subscribes_to: list[str] = field(default_factory=list)
    """Scope IDs this capability pulls data from."""

    subscribers: list[str] = field(default_factory=list)
    """Scope IDs that pull data from this capability."""

    producer_count: int = 0
    """Number of hook-based producers capturing data into this scope."""

    # Maintenance status
    pending_ingestion_count: int = 0
    """Entries waiting to be ingested from subscriptions."""

    maintenance_policy_count: int = 0
    """Number of active maintenance policies."""


@dataclass
class MemoryMap:
    """Complete map of agent's memory layout.

    Generated on-demand by AgentContextEngine.inspect_memory_map().
    Provides the LLM with a global view of the memory system.
    """

    agent_id: str
    """Agent this map belongs to."""

    scopes: dict[str, MemoryScopeInfo] = field(default_factory=dict)
    """All memory scopes by scope_id."""

    dataflow_edges: list[tuple[str, str]] = field(default_factory=list)
    """Directed edges: (source_scope, target_scope) for subscriptions."""

    total_entries: int = 0
    """Total entries across all scopes."""

    total_pending_ingestion: int = 0
    """Total entries pending ingestion across all scopes."""

    generated_at: float = field(default_factory=time.time)
    """When this map was generated."""

    def get_scope_by_type(self, scope_type: str) -> MemoryScopeInfo | None:
        """Get scope info by type (e.g., 'working', 'stm')."""
        for info in self.scopes.values():
            if info.scope_type == scope_type:
                return info
        return None


@dataclass
class ScopeInspectionResult:
    """Detailed inspection of a single memory scope.

    Returned by AgentContextEngine.inspect_scope().
    """

    scope_info: MemoryScopeInfo
    """Basic scope information."""

    # Detailed configuration
    retrieval_strategy: str = "unknown"
    """Name of the retrieval strategy in use."""

    maintenance_policies: list[str] = field(default_factory=list)
    """Names of active maintenance policies."""

    lens_names: list[str] = field(default_factory=list)
    """Available lens configurations."""

    # Sample entries (optional)
    sample_entries: list[dict[str, Any]] = field(default_factory=list)
    """Sample entries for content preview."""


@dataclass
class MemoryStatistics:
    """Health and usage statistics for the memory system.

    Returned by AgentContextEngine.get_memory_statistics().
    """

    total_entries: int = 0
    """Total entries across all scopes."""

    total_pending_ingestion: int = 0
    """Total entries awaiting ingestion."""

    scope_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per-scope statistics: {scope_id: {entry_count, capacity_pct, ...}}."""

    generated_at: float = field(default_factory=time.time)
    """When these statistics were generated."""


@dataclass
class MemorySearchResult:
    """Result from cross-scope memory search.

    Returned by AgentContextEngine.search_memory().
    """

    entry: "BlackboardEntry"
    """The matched entry."""

    scope_id: str
    """Which scope this entry came from."""

    relevance_score: float = 0.0
    """Relevance score from retrieval strategy."""


@dataclass
class MemoryValidationIssue:
    """Issue found during memory system validation."""

    severity: Literal["error", "warning", "info"]
    """How serious the issue is."""

    message: str
    """Description of the issue."""

    scope: str | None = None
    """Affected scope (if applicable)."""

    capability: str | None = None
    """Affected capability class (if applicable)."""


@dataclass
class CapabilityMemoryRequirements:
    """Memory requirements declared by a capability.

    Capabilities override get_memory_requirements() to declare what
    memory resources they need. Used by AgentContextEngine to:
    - Validate memory system integrity at initialization
    - Generate accurate memory map descriptions
    - Warn about missing dependencies
    """

    required_scopes: list[str] = field(default_factory=list)
    """Memory scope types this capability requires (e.g., ['working', 'stm'])."""

    produces_tags: set[str] = field(default_factory=set)
    """Tags this capability produces in memory."""

    consumes_tags: set[str] = field(default_factory=set)
    """Tags this capability reads from memory."""

