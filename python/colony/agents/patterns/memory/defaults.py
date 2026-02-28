"""Default Memory Hierarchy Factory.

This module provides factory functions to create standard memory hierarchies
for agents. Users can use these as-is or as templates for custom configurations.

The default hierarchy includes:
- Sensory memory (optional, raw observations)
- Working memory (current task context, with compaction)
- Short-term memory (recent observations, decaying)
- Long-term memory (episodic, semantic, procedural - persistent)
- Pull model for Intra-agent transfers (working→STM, STM→LTM): higher levels subscribe
  to lower levels and apply ingestion_policy.transformer
- Lifecycle hooks (task completion, shutdown)

Example:
    ```python
    from polymathera.colony.agents.patterns.memory.defaults import (
        create_default_memory_hierarchy,
    )

    # Create standard hierarchy
    capabilities = await create_default_memory_hierarchy(agent)

    # Or customize
    capabilities = await create_default_memory_hierarchy(
        agent,
        stm_ttl=7200,  # 2 hours
        working_max_tokens=16000,  # Larger working memory
    )

    # All capabilities are automatically added to agent and initialized
    ```
"""

from __future__ import annotations

import logging
from typing import Any

from ...base import Agent, AgentCapability
from .scopes import MemoryScope
from .types import (
    MaintenanceConfig,
    MemorySubscription,
    MemoryProducerConfig,
    extract_action_from_dispatch,
    extract_event_from_event_driven_policy,
    extract_plan_from_policy,
    extract_terminal_game_state,
    extract_reflection,
    extract_critique,
)
from .capability import MemoryCapability
from .working import WorkingMemoryCapability
from .session_memory import SessionMemoryCapability
from .context import AgentContextEngine
from .lifecycle import MemoryLifecycleHooks
from .protocols import (
    DecayMaintenancePolicy,
    TTLMaintenancePolicy,
    CapacityMaintenancePolicy,
    PruneMaintenancePolicy,
    DeduplicationMaintenancePolicy,
    ConsolidationMaintenancePolicy,
    SummarizingTransformer,
    FilteringTransformer,
    IdentityConsolidationTransformer,
    MemoryIngestPolicy,
    ThresholdMemoryIngestPolicyTrigger,
    PeriodicMemoryIngestPolicyTrigger,
    CompositeMemoryIngestPolicyTrigger,
)
from ..hooks import Pointcut

logger = logging.getLogger(__name__)


async def create_default_memory_hierarchy(
    agent: Agent,
    include_sensory: bool = False,
    stm_ttl: float = 3600,       # 1 hour
    stm_max_entries: int = 100,
    ltm_ttl: float | None = None,  # No expiration
    working_max_tokens: int = 8000,
    auto_add_to_agent: bool = True,
) -> dict[str, AgentCapability]:
    """Create the standard memory hierarchy for an agent.

    This creates using the unified MemoryCapability class:
    - Sensory memory (optional)
    - Working memory (with context compaction)
    - Short-term memory
    - Three LTM sub-levels (episodic, semantic, procedural)
    - Pull model: higher levels subscribe to lower levels
        - Working → STM transfer (on-demand, triggered by lifecycle hooks)
        - STM → LTM transfers (threshold and periodic)
    - AgentContextEngine for unified access
    - MemoryLifecycleHooks for task/shutdown integration

    Args:
        agent: The agent to create memory for
        include_sensory: Whether to include sensory memory level
        stm_ttl: TTL for STM entries (seconds)
        stm_max_entries: Max entries in STM before eviction
        ltm_ttl: TTL for LTM entries (None = no expiration)
        working_max_tokens: Token budget for working memory
        auto_add_to_agent: If True, add all capabilities to agent

    Returns:
        Dict of capability_name -> capability instance
    """
    capabilities: dict[str, AgentCapability] = {}
    agent_id = agent.agent_id

    # -------------------------------------------------------------------------
    # Memory Capabilities (Unified)
    # -------------------------------------------------------------------------

    # Sensory memory (optional) - captures raw observations (events)
    if include_sensory:
        sensory = MemoryCapability(
            agent=agent,
            scope_id=MemoryScope.agent_sensory(agent_id),
            capability_key="sensory",
            ttl_seconds=10,  # Very short retention
            max_entries=50,
            maintenance_policies=[
                DecayMaintenancePolicy(decay_rate=0.01),  # 1% per minute
                TTLMaintenancePolicy(check_interval_seconds=60.0),
                CapacityMaintenancePolicy(max_entries=50, check_interval_seconds=60.0),
                PruneMaintenancePolicy(prune_threshold=0.1, check_interval_seconds=60.0),
            ],
            # Observe blackboard events via hook on get_next_event
            # Note: extractor=None means directly store the result
            producers=[
                MemoryProducerConfig(
                    pointcut=Pointcut.pattern("EventDrivenActionPolicy.get_next_event"),
                    extractor=extract_event_from_event_driven_policy,
                    ttl_seconds=60,  # 1 minute retention
                ),
            ],
        )
        capabilities["sensory"] = sensory

    # Working memory - captures actions, plans, and filtered sensory input
    # Uses WorkingMemoryCapability for token tracking and compaction
    # Conditionally subscribes to sensory memory when enabled
    working_subscriptions = []
    if include_sensory:
        working_subscriptions.append(
            MemorySubscription(source_scope_id=MemoryScope.agent_sensory(agent_id)),
        )

    working = WorkingMemoryCapability(
        agent=agent,
        scope_id=MemoryScope.agent_working(agent_id),
        capability_key="working",
        max_tokens=working_max_tokens,
        compaction_threshold=0.9,
        maintenance=MaintenanceConfig(
            decay_rate=0.0,  # No decay in working memory
            track_access=False,
        ),
        # Pull from sensory frequently to keep working memory current
        ingestion_policy=MemoryIngestPolicy(
            # Subscribe to sensory memory when enabled - filter raw input for relevance
            subscriptions=working_subscriptions if working_subscriptions else None,
            trigger=PeriodicMemoryIngestPolicyTrigger(
                interval_seconds=30.0,  # Every 30 seconds
            ),
            # Filter sensory input to keep only relevant observations
            transformer=FilteringTransformer(
                min_relevance=0.4,  # Filter out low-relevance noise from sensory
            ) if include_sensory else None,
        ) if include_sensory else None,
        # Observe action execution via hook on dispatch
        producers=[
            MemoryProducerConfig(
                pointcut=Pointcut.pattern("ActionDispatcher.dispatch"),
                extractor=extract_action_from_dispatch,
                ttl_seconds=3600,  # 1 hour
            ),
            # Observe plan creation
            MemoryProducerConfig(
                pointcut=Pointcut.pattern("CacheAwareActionPolicy._create_initial_plan"),
                extractor=extract_plan_from_policy, # Cannot be None. The return value is the plan but we need tags and metadata.
                ttl_seconds=3600,  # 1 hour
            ),
            MemoryProducerConfig(
                pointcut=Pointcut.pattern("CacheAwareActionPolicy._replan_horizon"),
                extractor=extract_plan_from_policy, # Cannot be None. The return value is the plan but we need tags and metadata.
                ttl_seconds=3600,  # 1 hour
            ),
            # TODO: CriticCapability.critique_self_output is an @action_executor and
            # we already store actions and their results above in the ActionDispatcher.dispatch pointcut.
            MemoryProducerConfig(
                pointcut=Pointcut.pattern("CriticCapability.critique_self_output"),
                extractor=extract_critique, # Cannot be None. The return value is the critique but we need tags and metadata.
                ttl_seconds=3600,  # 1 hour
            ),
        ],
    )
    capabilities["working"] = working

    # Short-term memory - subscribes to working memory and captures messages
    # Uses summarizing transformer to consolidate incoming working memory items
    stm = MemoryCapability(
        agent=agent,
        scope_id=MemoryScope.agent_stm(agent_id),
        capability_key="stm",
        ttl_seconds=stm_ttl,
        max_entries=stm_max_entries,
        # Trigger ingestion when enough items accumulate or periodically
        ingestion_policy=MemoryIngestPolicy(
            # Pull model: subscribe to working memory
            subscriptions=[
                MemorySubscription(source_scope_id=MemoryScope.agent_working(agent_id)),
            ],
            trigger=CompositeMemoryIngestPolicyTrigger(
                policies=[
                    ThresholdMemoryIngestPolicyTrigger(min_items=5, max_items=20),
                    PeriodicMemoryIngestPolicyTrigger(interval_seconds=120.0),  # Every 2 minutes
                ],
                require_all=False,  # OR: trigger on either condition
            ),
            # Summarize incoming entries from working memory (consolidation at ingestion)
            transformer=SummarizingTransformer(
                agent=agent,
                prompt=(
                    "Summarize the following recent actions and observations into a coherent "
                    "narrative. Preserve key decisions, outcomes, and any important context. "
                    "Focus on what happened and why it matters for ongoing tasks."
                ),
                max_tokens=300,
            ),
        ),
        maintenance_policies=[
            DecayMaintenancePolicy(decay_rate=0.01),  # 1% per minute
            TTLMaintenancePolicy(check_interval_seconds=60.0),
            CapacityMaintenancePolicy(max_entries=stm_max_entries, check_interval_seconds=60.0),
            PruneMaintenancePolicy(prune_threshold=0.1, check_interval_seconds=60.0),
            DeduplicationMaintenancePolicy(deduplication_threshold=0.95, check_interval_seconds=60.0),
            # Within-scope consolidation (summarize old entries already in STM)
            ConsolidationMaintenancePolicy(
                transformer=SummarizingTransformer(
                    agent=agent,
                    prompt=(
                        "Compress and consolidate these short-term memory entries. "
                        "Merge related events, remove redundancy, and preserve the temporal "
                        "sequence of important happenings. Keep essential details that may "
                        "be needed for immediate decision-making."
                    ),
                    max_tokens=500,
                ),
                threshold=15,
                check_interval_seconds=300.0,  # 5 minutes
                delete_originals=True,
            ),
        ],
        # TODO: Capture any input stream (e.g., incoming messages) directly via hook
    )
    capabilities["stm"] = stm

    # Long-term episodic memory - subscribes to STM and captures terminal game states
    # Stores event sequences, experiences, game outcomes
    # Filters for episodic content (events, actions, game states) and consolidates
    ltm_episodic = MemoryCapability(
        agent=agent,
        scope_id=MemoryScope.agent_ltm_episodic(agent_id),
        capability_key="ltm:episodic",
        ttl_seconds=ltm_ttl,
        # Trigger ingestion less frequently (LTM consolidation is slower)
        ingestion_policy=MemoryIngestPolicy(
            # Pull model: subscribe to STM
            subscriptions=[
                MemorySubscription(source_scope_id=MemoryScope.agent_stm(agent_id)),
            ],
            trigger=CompositeMemoryIngestPolicyTrigger(
                policies=[
                    ThresholdMemoryIngestPolicyTrigger(min_items=10, max_items=50),
                    PeriodicMemoryIngestPolicyTrigger(interval_seconds=600.0),  # Every 10 minutes
                ],
                require_all=False,
            ),
            # Filter for episodic content (events, actions, outcomes) and summarize
            # FilteringTransformer keeps only entries with episodic-relevant tags
            transformer=FilteringTransformer(
                min_relevance=0.3,  # Keep moderately relevant and above
                required_tags=None,  # Don't require specific tags (allow all)
                excluded_tags={"reflection", "semantic", "knowledge", "skill"},  # Exclude semantic/procedural
            ),
        ),
        maintenance_policies=[
            DecayMaintenancePolicy(decay_rate=0.001),  # Slow decay
            PruneMaintenancePolicy(prune_threshold=0.05, check_interval_seconds=600.0),
            # Within-scope consolidation for LTM episodic
            ConsolidationMaintenancePolicy(
                transformer=SummarizingTransformer(
                    agent=agent,
                    prompt=(
                        "Consolidate these episodic memories into coherent experiences. "
                        "Group related events into episodes, preserve causal relationships "
                        "and outcomes. Focus on WHAT happened, WHEN, and what the RESULTS were. "
                        "Maintain the narrative structure of experiences."
                    ),
                    max_tokens=1000,
                ),
                threshold=20,
                check_interval_seconds=900.0,  # 15 minutes
                delete_originals=True,
            ),
        ],
        # Also capture completed games directly via hook
        producers=[
            MemoryProducerConfig(
                pointcut=Pointcut.pattern("GameProtocolCapability.submit_move"),
                extractor=extract_terminal_game_state,  # Only stores terminal states
                ttl_seconds=None,  # No TTL for episodic memories
            ),
        ],
    )
    capabilities["ltm_episodic"] = ltm_episodic

    # Long-term semantic memory - subscribes to STM for distilled knowledge and captures reflections directly via hook
    # Stores facts, patterns, learned concepts
    # Filters for semantic content (reflections, knowledge, patterns) and consolidates
    ltm_semantic = MemoryCapability(
        agent=agent,
        scope_id=MemoryScope.agent_ltm_semantic(agent_id),
        capability_key="ltm:semantic",
        ttl_seconds=ltm_ttl,
        # Semantic consolidation is infrequent (knowledge distillation takes time)
        ingestion_policy=MemoryIngestPolicy(
            # Pull model: subscribe to STM
            subscriptions=[
                MemorySubscription(source_scope_id=MemoryScope.agent_stm(agent_id)),
            ],
            trigger=CompositeMemoryIngestPolicyTrigger(
                policies=[
                    ThresholdMemoryIngestPolicyTrigger(min_items=15, max_items=100),
                    PeriodicMemoryIngestPolicyTrigger(interval_seconds=900.0),  # Every 15 minutes
                ],
                require_all=False,
            ),
            # Filter for semantic content (knowledge, patterns, reflections)
            transformer=FilteringTransformer(
                min_relevance=0.5,  # Keep high-relevance content for semantic memory
                required_tags=None,
                excluded_tags={"ephemeral", "procedural", "skill", "action"},  # Exclude procedural/action
            ),
        ),
        maintenance_policies=[
            DecayMaintenancePolicy(decay_rate=0.0005),  # Very slow decay
            PruneMaintenancePolicy(prune_threshold=0.02, check_interval_seconds=1200.0),
            # Within-scope consolidation for semantic knowledge
            ConsolidationMaintenancePolicy(
                transformer=SummarizingTransformer(
                    agent=agent,
                    prompt=(
                        "Distill these memories into abstract knowledge and facts. "
                        "Extract general principles, patterns, and concepts. "
                        "Remove episodic details and focus on WHAT is true, not when it happened. "
                        "Identify relationships between concepts and create structured knowledge."
                    ),
                    max_tokens=1500,
                ),
                threshold=25,
                check_interval_seconds=1800.0,  # 30 minutes
                delete_originals=True,
            ),
        ],
        # Also capture reflections directly via hook
        # TODO: Should reflections be stored in semantic memory?
        # They represent distilled knowledge. But they might have immediate
        # relevance too. So, they could also go to working memory.
        producers=[
            MemoryProducerConfig(
                pointcut=Pointcut.pattern("ReflectionCapability.reflect"),
                extractor=extract_reflection,
                ttl_seconds=None,
            ),
        ],
    )
    capabilities["ltm_semantic"] = ltm_semantic

    # Long-term procedural memory - skills, strategies, learned patterns
    # Subscribes to episodic memory to extract action patterns from ALL experiences
    # Both successes (what to do) and failures (what to avoid) contribute to skill learning
    ltm_procedural = MemoryCapability(
        agent=agent,
        scope_id=MemoryScope.agent_ltm_procedural(agent_id),
        capability_key="ltm:procedural",
        ttl_seconds=ltm_ttl,
        # Skill learning is slower - less frequent consolidation
        ingestion_policy=MemoryIngestPolicy(
            # Pull model: subscribe to episodic memory to learn from ALL experiences
            subscriptions=[
                MemorySubscription(source_scope_id=MemoryScope.agent_ltm_episodic(agent_id)),
            ],
            trigger=CompositeMemoryIngestPolicyTrigger(
                policies=[
                    ThresholdMemoryIngestPolicyTrigger(min_items=20, max_items=100),
                    PeriodicMemoryIngestPolicyTrigger(interval_seconds=1800.0),  # Every 30 minutes
                ],
                require_all=False,
            ),
            # Extract action patterns and strategies from episodic experiences
            transformer=SummarizingTransformer(
                agent=agent,
                prompt=(
                    "Extract actionable skills and strategies from these experiences. "
                    "Identify: (1) successful action patterns to REPEAT, (2) failed approaches to AVOID, "
                    "(3) conditions that determine when each approach works. "
                    "Focus on HOW to do things, not just what happened. "
                    "Generalize specific instances into reusable procedures."
                ),
                max_tokens=1000,
            ),
        ),
        maintenance_policies=[
            # Very slow decay - skills persist longer than episodes
            DecayMaintenancePolicy(decay_rate=0.0001),
            # Deduplicate similar procedures
            DeduplicationMaintenancePolicy(deduplication_threshold=0.9, check_interval_seconds=1800.0),
            # Consolidate similar skills into generalized procedures
            ConsolidationMaintenancePolicy(
                transformer=SummarizingTransformer(
                    agent=agent,
                    prompt=(
                        "Merge and generalize these procedural skills. "
                        "Combine similar strategies into more general procedures. "
                        "Identify common patterns across different contexts. "
                        "Create hierarchical skills where specific procedures are cases of general ones. "
                        "Preserve both the general rule and important exceptions/edge cases."
                    ),
                    max_tokens=2000,
                ),
                threshold=30,
                check_interval_seconds=3600.0,  # Every hour
                delete_originals=True,
            ),
        ],
    )
    capabilities["ltm_procedural"] = ltm_procedural

    # -------------------------------------------------------------------------
    # Context Engine (Unified Access)
    # -------------------------------------------------------------------------

    context_engine = AgentContextEngine(agent=agent)
    capabilities["context_engine"] = context_engine

    # -------------------------------------------------------------------------
    # Lifecycle Hooks
    # -------------------------------------------------------------------------

    lifecycle_hooks = MemoryLifecycleHooks(
        agent=agent,
        stm_scope_id=MemoryScope.agent_stm(agent_id),
    )
    capabilities["lifecycle_hooks"] = lifecycle_hooks

    # -------------------------------------------------------------------------
    # Add to Agent, then Initialize
    # -------------------------------------------------------------------------
    # Add all capabilities first so they can discover each other during
    # initialization (e.g., AgentContextEngine discovers MemoryCapability
    # instances on the agent).

    if auto_add_to_agent:
        for name, capability in capabilities.items():
            agent.add_capability(capability)

    for name, capability in capabilities.items():
        try:
            await capability.initialize()
            logger.debug(f"Initialized memory capability: {name}")
        except Exception as e:
            logger.error(f"Failed to initialize {name}: {e}")
            raise

    logger.info(
        f"Created default memory hierarchy for {agent_id}: "
        f"{len(capabilities)} capabilities"
    )

    return capabilities


async def create_minimal_memory_hierarchy(
    agent: "Agent",
    auto_add_to_agent: bool = True,
) -> dict[str, AgentCapability]:
    """Create a minimal memory hierarchy with only STM and working memory.

    Useful for simple agents or testing.

    Args:
        agent: The agent to create memory for
        auto_add_to_agent: If True, add all capabilities to agent

    Returns:
        Dict of capability_name -> capability instance
    """
    capabilities: dict[str, AgentCapability] = {}
    agent_id = agent.agent_id

    # Working memory
    working = WorkingMemoryCapability(
        agent=agent,
        scope_id=MemoryScope.agent_working(agent_id),
        capability_key="working",
        max_tokens=4000,
    )
    capabilities["working"] = working

    # Short-term memory - subscribes to working memory
    stm = MemoryCapability(
        agent=agent,
        scope_id=MemoryScope.agent_stm(agent_id),
        capability_key="stm",
        ttl_seconds=1800,  # 30 minutes
        max_entries=50,
        # Trigger on threshold or periodic
        ingestion_policy=MemoryIngestPolicy(
            subscriptions=[
                MemorySubscription(source_scope_id=MemoryScope.agent_working(agent_id)),
            ],
            trigger=ThresholdMemoryIngestPolicyTrigger(
                min_items=3,
                max_items=10,
            ),
            # Consolidate incoming entries from working memory
            transformer=SummarizingTransformer(
                agent=agent,
                prompt=(
                    "Briefly summarize these recent actions and observations. "
                    "Keep essential context for ongoing tasks."
                ),
                max_tokens=200,
            ),
        ),
    )
    capabilities["stm"] = stm

    # Context engine
    context_engine = AgentContextEngine(agent=agent)
    capabilities["context_engine"] = context_engine

    # Add all first, then initialize (so AgentContextEngine can discover others)
    if auto_add_to_agent:
        for name, capability in capabilities.items():
            agent.add_capability(capability)

    for name, capability in capabilities.items():
        await capability.initialize()

    return capabilities


async def create_session_memory(
    agent: Agent,
    tenant_id: str,
    *,
    include_cross_session: bool = False,
    cross_session_weight: float = 0.3,
    ttl_seconds: float | None = None,
    max_entries: int | None = 10000,
    auto_add_to_agent: bool = True,
) -> SessionMemoryCapability:
    """Create a session-scoped memory capability for an agent.

    `SessionMemoryCapability` is a single instance per tenant that stores
    entries tagged with the current session_id. Unlike regular memory
    capabilities, it automatically filters by session when recalling.

    Use this when an agent needs to handle multiple user sessions within
    a tenant, with each session having isolated memory.

    Args:
        `agent`: The agent to create session memory for
        `tenant_id`: Tenant this session memory serves
        `include_cross_session`: Whether to include cross-session memories
            in retrieval (with lower weight). Default: False.
        `cross_session_weight`: Weight for cross-session entries when
            include_cross_session is True (0.0-1.0). Default: 0.3.
        `ttl_seconds`: TTL for session memories (None = no expiration)
        `max_entries`: Maximum entries before eviction (default: 10000)
        `auto_add_to_agent`: If True, add capability to agent

    Returns:
        Initialized `SessionMemoryCapability`

    Example:
        ```python
        from polymathera.colony.agents.patterns.memory import create_session_memory
        from polymathera.colony.agents.sessions import session_context

        # Create session memory for tenant
        session_memory = await create_session_memory(
            agent=agent,
            tenant_id="my-tenant",
        )

        # Use within session context
        async with session_context(session):
            # Store automatically tagged with session_id
            await session_memory.store(observation)

            # Recall automatically filtered by session_id
            memories = await session_memory.recall()
        ```
    """
    session_memory = SessionMemoryCapability(
        agent=agent,
        tenant_id=tenant_id,
        include_cross_session=include_cross_session,
        cross_session_weight=cross_session_weight,
        ttl_seconds=ttl_seconds,
        max_entries=max_entries,
        maintenance_policies=[
            CapacityMaintenancePolicy(
                max_entries=max_entries or 10000,
                check_interval_seconds=300.0,  # 5 minutes
            ),
        ] if max_entries else None,
    )

    await session_memory.initialize()

    if auto_add_to_agent:
        agent.add_capability(session_memory)

    logger.info(
        f"Created session memory for tenant {tenant_id} on agent {agent.agent_id}"
    )

    return session_memory

