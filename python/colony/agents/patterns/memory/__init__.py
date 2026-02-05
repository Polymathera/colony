"""Agent Memory System.

This module provides the unified `MemoryCapability` class for managing agent memory:

- `MemoryCapability`: Unified memory capability (ingestion + storage + maintenance)
- `WorkingMemoryCapability`: Specialized for token-bounded working memory
- `AgentContextEngine`: Unified interface for cross-level memory operations
- `MemoryLifecycleHooks`: Lifecycle hooks for task/shutdown integration
- `MemoryScope`: Standardized scope ID generation
- `MemoryManagementAgent`: System administrator agent for memory lifecycle
- Memory management capabilities for recycling and collective memory

Architecture (Pull Model):
    Each MemoryCapability manages ONE scope. It:
    1. INGESTS data from sources (subscriptions to other scopes + hook-based producers)
    2. Transforms ALL pending entries together via `ingestion_transformer`
    3. MAINTAINS the scope (decay, prune, dedupe, within-scope consolidation)

    If CapabilityB wants data from CapabilityA's scope:
    → CapabilityB subscribes to CapabilityA's scope
    → CapabilityB's ingestion_transformer consolidates the data
    → CapabilityA's maintenance (TTL, capacity) cleans up

    No push/export logic. Each capability only manages its own scope.

These capabilities provide:
- Storage, retrieval, and maintenance for each memory level
- Transformation and transfer between memory levels
- Unified context gathering for action planning
- Working memory compaction and task completion draining
- Agent lifecycle integration (shutdown, termination)
- Collective memory for agent types (transfer learning)

Usage:
    ```python
    #######################################################

    from polymathera.colony.agents.patterns.memory import (
        create_default_memory_hierarchy,
        MemoryScope,
    )

    # Create standard memory hierarchy (recommended)
    capabilities = await create_default_memory_hierarchy(agent)

    #######################################################

    # Or build custom hierarchy using MemoryCapability
    from polymathera.colony.agents.patterns.memory import (
        MemoryCapability,
        WorkingMemoryCapability,
        MemorySubscription,
        MemoryProducerConfig,
        SummarizingTransformer,
    )
    from polymathera.colony.agents.patterns.hooks import Pointcut

    agent_id = agent.agent_id

    # Working memory with hook-based capture (no transformer)
    working = WorkingMemoryCapability(
        agent=agent,
        scope_id=MemoryScope.agent_working(agent_id),
        max_tokens=8000,
        producers=[
            MemoryProducerConfig(
                pointcut=Pointcut.pattern("ActionDispatcher.dispatch"),
                extractor=extract_action_from_dispatch,
            ),
        ],
    )

    # STM subscribes to working memory, consolidates ALL inputs
    stm = MemoryCapability(
        agent=agent,
        scope_id=MemoryScope.agent_stm(agent_id),
        ingestion_policy=MemoryIngestPolicy(
            subscriptions=[
                MemorySubscription(source_scope_id=MemoryScope.agent_working(agent_id)),
            ],
            trigger=PeriodicMemoryIngestPolicyTrigger(
                interval_seconds=120.0,  # Every 2 minutes
            ),
            transformer=SummarizingTransformer(
                agent=agent,
                prompt="Summarize recent actions into a coherent narrative for short-term memory.",
            ),
        ),
        ttl_seconds=86400,
    )

    # LTM subscribes to STM
    ltm = MemoryCapability(
        agent=agent,
        scope_id=MemoryScope.agent_ltm_episodic(agent_id),
        subscriptions=[
            MemorySubscription(source_scope_id=MemoryScope.agent_stm(agent_id)),
        ],
        ingestion_transformer=EpisodicConsolidationTransformer(),
    )

    await working.initialize()
    await stm.initialize()
    await ltm.initialize()

    agent.add_capability(working)
    agent.add_capability(stm)
    agent.add_capability(ltm)

    #######################################################

    # Context engine discovers all capabilities
    ctx_engine = AgentContextEngine(agent)
    await ctx_engine.initialize()
    agent.add_capability(ctx_engine)

    # In action policy
    ctx_engine = agent.get_capability(AgentContextEngine.get_capability_name())
    memories = await ctx_engine.gather_context(query=MemoryQuery(query="authentication flow", max_results=20))

    #######################################################
    ```
"""

# Core types
from .types import (
    MemoryQuery,
    MemorySubscription,
    MaintenanceConfig,
    MemoryProducerConfig,
    MemoryTagsMetadataTuple,
    RetrievalContext,
    ScoredEntry,
    MaintenanceResult,
    ConsolidationContext,
    MemoryLens,
    PLANNING_LENS,
    REFLECTION_LENS,
    SKILL_LENS,
    extract_event_from_event_driven_policy,
    extract_action_from_dispatch,
    extract_plan_from_policy,
    extract_terminal_game_state,
    extract_reflection,
    extract_critique,
    # Memory introspection types
    MemoryScopeInfo,
    MemoryMap,
    ScopeInspectionResult,
    MemoryStatistics,
    MemorySearchResult,
    MemoryValidationIssue,
    CapabilityMemoryRequirements,
)

# Scope ID generation
from .scopes import MemoryScope

# Unified memory capability (primary)
from .capability import MemoryCapability

# Specialized capabilities
from .working import WorkingMemoryCapability, CompactionSummary
from .session_memory import SessionMemoryCapability

# Protocols and default implementations
from .protocols import (
    StorageBackend,
    StorageBackendFactory,
    RetrievalStrategy,
    MaintenancePolicy,
    ConsolidationTransformer,
    UtilityScorer,
    RecencyRetrieval,
    TTLMaintenancePolicy,
    CapacityMaintenancePolicy,
    DecayMaintenancePolicy,
    ConsolidationMaintenancePolicy,
    IdentityConsolidationTransformer,
    SummarizingTransformer,
    FilteringTransformer,
    TSource,
    TTarget,
    MemoryIngestPolicy,
    MemoryIngestPolicyTrigger,
    OnDemandMemoryIngestPolicyTrigger,
    PeriodicMemoryIngestPolicyTrigger,
    ThresholdMemoryIngestPolicyTrigger,
    CompositeMemoryIngestPolicyTrigger,
)

# Storage backends
from .backends import BlackboardStorageBackend, BlackboardStorageBackendFactory

# Context engine
from .context import AgentContextEngine

# Memory architecture prompt guidance
from .prompts import get_memory_architecture_guidance

# Lifecycle hooks
from .lifecycle import (
    MemoryLifecycleHooks,
    AgentTerminationEvent,
    AgentCreationEvent,
)

# Memory management capabilities and agent
from .management import (
    AgentMemoryRecycler,
    CollectiveMemoryInitializer,
    CollectiveMemoryMaintainer,
    MemoryManagementAgent,
)

# Factory functions
from .defaults import (
    create_default_memory_hierarchy,
    create_minimal_memory_hierarchy,
    create_session_memory,
)


__all__ = [
    # Core types
    "MemoryQuery",
    "MemorySubscription",
    "MaintenanceConfig",
    "MemoryProducerConfig",
    "MemoryTagsMetadataTuple",
    "RetrievalContext",
    "ScoredEntry",
    "MaintenanceResult",
    "ConsolidationContext",
    "MemoryLens",
    "PLANNING_LENS",
    "REFLECTION_LENS",
    "SKILL_LENS",
    "extract_event_from_event_driven_policy",
    "extract_action_from_dispatch",
    "extract_plan_from_policy",
    "extract_terminal_game_state",
    "extract_reflection",
    "extract_critique",
    # Scope ID generation
    "MemoryScope",
    # Unified memory capability (primary)
    "MemoryCapability",
    # Specialized capabilities
    "WorkingMemoryCapability",
    "CompactionSummary",
    "SessionMemoryCapability",
    # Protocols
    "StorageBackend",
    "StorageBackendFactory",
    "RetrievalStrategy",
    "MaintenancePolicy",
    "ConsolidationTransformer",
    "UtilityScorer",
    # Default implementations
    "RecencyRetrieval",
    "TTLMaintenancePolicy",
    "CapacityMaintenancePolicy",
    "DecayMaintenancePolicy",
    "ConsolidationMaintenancePolicy",
    "IdentityConsolidationTransformer",
    "SummarizingTransformer",
    "FilteringTransformer",
    "MemoryIngestPolicy",
    "MemoryIngestPolicyTrigger",
    "OnDemandMemoryIngestPolicyTrigger",
    "PeriodicMemoryIngestPolicyTrigger",
    "ThresholdMemoryIngestPolicyTrigger",
    "CompositeMemoryIngestPolicyTrigger",
    # Storage backends
    "BlackboardStorageBackend",
    "BlackboardStorageBackendFactory",
    # Memory introspection types
    "MemoryScopeInfo",
    "MemoryMap",
    "ScopeInspectionResult",
    "MemoryStatistics",
    "MemorySearchResult",
    "MemoryValidationIssue",
    "CapabilityMemoryRequirements",
    # Context engine
    "AgentContextEngine",
    # Memory architecture prompt guidance
    "get_memory_architecture_guidance",
    # Lifecycle hooks
    "MemoryLifecycleHooks",
    "AgentTerminationEvent",
    "AgentCreationEvent",
    # Memory management
    "AgentMemoryRecycler",
    "CollectiveMemoryInitializer",
    "CollectiveMemoryMaintainer",
    "MemoryManagementAgent",
    # Factory functions
    "create_default_memory_hierarchy",
    "create_minimal_memory_hierarchy",
    "create_session_memory",
    "TSource",
    "TTarget",
]
