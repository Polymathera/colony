"""Query-driven discovery patterns for incremental exploration.

This package implements patterns for discovering relevant context through
iterative querying:
- Incremental query processing with feedback loops (`IncrementalQueryCapability`)
- Scope-based query generation from gaps (`ScopeBasedQueryGenerator`)
- Multi-hop search and query routing across relationships (`MultiHopSearchCapability`)
- Query-driven and hypothesis-driven exploration (`QueryDrivenExplorationCapability`)
- Adaptive query generation (`AdaptiveQueryGenerator`)

These patterns are implemented as AgentCapabilities with @action_executor
methods. Use the `create_*_policy()` functions to combine them with
CacheAwareActionPolicy, where the planner decides iteration flow.
"""

from .incremental import (
    IncrementalQueryCapability,
    create_incremental_query_policy,
)
from .scope_based import ScopeBasedQueryGenerator
from .multi_hop import (
    MultiHopSearchResult,
    MultiHopSearchCapability,
    create_multi_hop_search_policy,
)
from .explorer import (
    ExplorationResult,
    QueryDrivenExplorationCapability,
    create_exploration_policy,
)
from .adaptive import AdaptiveQueryGenerator, create_adaptive_query_policy
from .attention import (
    PageQuery,
    PageKey,
    AttentionScore,
    AttentionScoringMechanism,
    KeyGenerator,
    QueryGenerator,
    ErrorQueryGenerator,
    SemanticQueryGenerator,
    DependencyQueryGenerator,
    HybridKeyGenerator,
    SemanticKeyGenerator,
    StructuralKeyGenerator,
    BatchedLLMAttention,
    EmbeddingBasedAttention,
)
from .attention_policy import (
    AttentionPolicy,
    HierarchicalAttentionPolicy,
    GlobalAttentionPolicy,
    LocalAttentionPolicy,
)
from .key_registry import GlobalPageKeyRegistry
from .query_routing import (
    PageQueryRoutingPolicy,
    DirectAttentionRouting,
    HierarchicalAttentionRouting,
    CacheAwareRouting,
    GraphBasedRouting,
    HybridQueryRouting,
    BatchedQueryRouting,
    create_page_query_router1,
    create_page_query_router2
)

__all__ = [
    # Capabilities (provide @action_executor methods)
    "IncrementalQueryCapability",
    "MultiHopSearchCapability",
    "QueryDrivenExplorationCapability",
    "AdaptiveQueryGenerator",
    "ScopeBasedQueryGenerator",
    # Factory functions (create policies with planners)
    "create_incremental_query_policy",
    "create_multi_hop_search_policy",
    "create_exploration_policy",
    "create_adaptive_query_policy",
    # Result types
    "MultiHopSearchResult",
    "ExplorationResult",
    # Attention components
    "PageQuery",
    "PageKey",
    "AttentionScore",
    "AttentionScoringMechanism",
    "KeyGenerator",
    "QueryGenerator",
    "ErrorQueryGenerator",
    "SemanticQueryGenerator",
    "DependencyQueryGenerator",
    "HybridKeyGenerator",
    "SemanticKeyGenerator",
    "StructuralKeyGenerator",
    "BatchedLLMAttention",
    "EmbeddingBasedAttention",
    # Attention Policies
    "AttentionPolicy",
    "HierarchicalAttentionPolicy",
    "GlobalAttentionPolicy",
    "LocalAttentionPolicy",
    # Key Registry
    "GlobalPageKeyRegistry",
    # Query Routing
    "PageQueryRoutingPolicy",
    "DirectAttentionRouting",
    "HierarchicalAttentionRouting",
    "CacheAwareRouting",
    "GraphBasedRouting",
    "HybridQueryRouting",
    "BatchedQueryRouting",
    "create_page_query_router1",
    "create_page_query_router2"
]
