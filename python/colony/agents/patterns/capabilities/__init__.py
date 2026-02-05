"""Capabilities for multi-agent execution framework primitives.

These capabilities provide @action_executor methods that ActionPolicies
can compose to achieve various distributed multi-agent execution strategies.

Layer 0 (Universal Primitives):
- WorkingSetCapability: Cluster-wide working set management
- PageGraphCapability: Page graph traversal and updates
- AgentPoolCapability: Agent lifecycle management

Layer 1 (Query/Result):
- QueryAttentionCapability: Query generation and routing
- ResultCapability: Result storage, merging, and synthesis

Policies:
- BatchingPolicy: How to group work items for processing
- PrefetchPolicy: How to predict and preload pages
- CoordinationPolicy: How to assign work to agents
"""

from .working_set import (
    WorkingSetCapability,
    PageEvictionPolicy,
    LRUEvictionPolicy,
    ReferenceCountEvictionPolicy,
    CompositeEvictionPolicy,
)
from .page_graph import PageGraphCapability
from .agent_pool import AgentPoolCapability
from .query_attention import QueryAttentionCapability
from .result import ResultCapability
from .reputation import ReputationCapability
from .consistency import ConsistencyCapability
from .grounding import GroundingCapability
from .goal_alignment import ObjectiveGuardCapability
from .consciousness import ConsciousnessCapability
from .reflection import ReflectionCapability
from .merge import MergeCapability
from .refinement import RefinementCapability
from .synthesis import SynthesisCapability
from .validation import ValidationCapability
from .vcm_analysis import VCMAnalysisCapability

# Batching policies
from .batching import (
    BatchingPolicy,
    ClusteringBatchPolicy,
    ContinuousBatchPolicy,
    HybridBatchPolicy,
)
# Prefetch policies
from .prefetching import (
    PrefetchPolicy,
    GraphPrefetchPolicy,
    QueryPrefetchPolicy,
    FeedbackPrefetchPolicy,
    CompositePrefetchPolicy,
)
# Coordination policies
from .coordination import (
    CoordinationPolicy,
    RoundRobinCoordinationPolicy,
    AffinityCoordinationPolicy,
    LoadBalancingCoordinationPolicy,
    CompositeCoordinationPolicy,
)

__all__ = [
    # Layer 0: Universal Primitives
    "WorkingSetCapability",
    "PageGraphCapability",
    "AgentPoolCapability",
    # Layer 0: Eviction Policies
    "PageEvictionPolicy",
    "LRUEvictionPolicy",
    "ReferenceCountEvictionPolicy",
    "CompositeEvictionPolicy",
    # Layer 1: Query/Result
    "QueryAttentionCapability",
    "ResultCapability",
    # Batching Policies
    "BatchingPolicy",
    "ClusteringBatchPolicy",
    "ContinuousBatchPolicy",
    "HybridBatchPolicy",
    # Prefetch Policies
    "PrefetchPolicy",
    "GraphPrefetchPolicy",
    "QueryPrefetchPolicy",
    "FeedbackPrefetchPolicy",
    "CompositePrefetchPolicy",
    # Coordination Policies
    "CoordinationPolicy",
    "RoundRobinCoordinationPolicy",
    "AffinityCoordinationPolicy",
    "LoadBalancingCoordinationPolicy",
    "CompositeCoordinationPolicy",
    # Capabilities
    "ReputationCapability",
    "ConsistencyCapability",
    "GroundingCapability",
    "ObjectiveGuardCapability",
    "ConsciousnessCapability",
    "ReflectionCapability",
    "MergeCapability",
    "RefinementCapability",
    "SynthesisCapability",
    "ValidationCapability",
    # VCM Analysis Pattern
    "VCMAnalysisCapability",
]
