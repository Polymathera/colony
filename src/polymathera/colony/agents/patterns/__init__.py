"""Domain-agnostic patterns for distributed agent collaboration.

This package contains reusable abstractions extracted from code analysis
that are generalizable to other domains requiring:
- Distributed analysis over large contexts
- Incremental refinement and discovery
- Multi-agent collaboration
- Partial knowledge management
- Confidence-aware reasoning

Core patterns:
- ScopeAwareResult: Wraps results with completeness metadata
- MergePolicy: Strategies for combining results
- RefinementPolicy: Iterative improvement strategies
- ValidationPolicy: Multi-level validation
- QueryDrivenExplorationCapability: Hypothesis-driven discovery
- RelationshipGraphBuilder: Knowledge graph construction
- IncrementalSynthesizer: Progressive synthesis with feedback

All patterns follow policy-based design for pluggability and customization.
"""

from .scope import (
    AnalysisScope,
    ScopeAwareResult,
    merge_scopes,
)

# ---------------------------------------------------------------------------
# ALL imports are deferred via __getattr__ to break circular dependencies:
#   base.py → patterns.hooks.decorator → patterns/__init__.py
#     → capabilities → working_set → base.py
# ---------------------------------------------------------------------------

__all__ = [
    # Scope awareness
    "AnalysisScope",
    "ScopeAwareResult",
    "merge_scopes",
    # Critique
    "Critique",
    "CritiqueContext",
    "CritiquePolicy",
    "CriticCapability",
    "CritiqueRequest",
    "LLMCritiquePolicy",
    "MetricBasedCritiquePolicy",
    "OutputRelationship",
    # Merging
    "MergePolicy",
    "MergeContext",
    "MergeCapability",
    "SemanticMergePolicy",
    "StatisticalMergePolicy",
    "GraphMergePolicy",
    "ListMergePolicy",
    "DictMergePolicy",
    "HierarchicalMerger",
    # Refinement
    "RefinementPolicy",
    "RefinementContext",
    "RefinementCapability",
    "LLMRefinementPolicy",
    "IncrementalRefiner",
    "ProgressiveRefinementTracker",
    # Confidence
    "ConfidenceTracker",
    "UncertaintyPropagator",
    "ConfidenceMetrics",
    # Validation
    "AnalysisValidationPolicy",
    "ValidationContext",
    "ValidationResult",
    "ValidationIssue",
    "ValidationCapability",
    "CrossShardConsistencyValidator",
    "EvidenceBasedValidator",
    "MultiLevelValidator",
    "ConsensusValidator",
    "ContradictionResolver",
    "Contradiction",
    # Relationships
    "Relationship",
    "RelationshipGraph",
    "RelationshipGraphBuilder",
    # Hypothesis game system (unified)
    "Hypothesis",
    "SubjectType",
    "SubjectReference",
    "ObservationType",
    "Observation",
    "HypothesisDomain",
    "HypothesisContext",
    "EvidenceType",
    "Evidence",
    "TriggerType",
    "HypothesisFormationTrigger",
    "EvaluationDecision",
    "EvaluationResult",
    "HypothesisStatus",
    "HypothesisFilter",
    "HypothesisFormationStrategy",
    "EvidenceGatheringStrategy",
    "HypothesisEvaluationStrategy",
    "LLMHypothesisFormation",
    "RuleBasedHypothesisFormation",
    "QueryBasedEvidence",
    "LLMReasoningEvidence",
    "CompositeEvidenceStrategy",
    "LLMEvaluation",
    "RuleBasedEvaluation",
    "TrackedHypothesis",
    "HypothesisTrackingCapability",
    # Legacy hypothesis (deprecated)
    "HypothesisTestResult",
    "HypothesisDrivenExplorer",
    # Query-driven discovery
    "QueryDrivenExplorationCapability",
    "ScopeBasedQueryGenerator",
    "AdaptiveQueryGenerator",
    "ExplorationResult",
    "IncrementalQueryCapability",
    # Synthesis
    "IncrementalSynthesizer",
    "SynthesisCapability",
    "SynthesisUpdate",
    # Models
    "Reflection",
    # Consciousness
    "ConsciousnessCapability",
    "SystemDocumentation",
    # Hooks
    "hookable",
    "register_hook",
    "auto_register_hooks",
    "Pointcut",
    "HookType",
    "ErrorMode",
    "HookContext",
    "RegisteredHook",
    "AgentHookRegistry",
    # Event processing
    "event_handler",
    "EventProcessingResult",
    "PROCESSED",
    # Capabilities (multi-agent execution framework primitives)
    "WorkingSetCapability",
    "PageGraphCapability",
    "AgentPoolCapability",
    "QueryAttentionCapability",
    "ResultCapability",
    # Eviction policies
    "PageEvictionPolicy",
    "LRUEvictionPolicy",
    "ReferenceCountEvictionPolicy",
    "CompositeEvictionPolicy",
    # Batching policies
    "BatchingPolicy",
    "ClusteringBatchPolicy",
    "ContinuousBatchPolicy",
    "HybridBatchPolicy",
    # Prefetch policies
    "PrefetchPolicy",
    "GraphPrefetchPolicy",
    "QueryPrefetchPolicy",
    "FeedbackPrefetchPolicy",
    "CompositePrefetchPolicy",
    # Coordination policies
    "CoordinationPolicy",
    "RoundRobinCoordinationPolicy",
    "AffinityCoordinationPolicy",
    "LoadBalancingCoordinationPolicy",
    "CompositeCoordinationPolicy",
]

# ---------------------------------------------------------------------------
# Lazy import registry: name → relative module path
# ---------------------------------------------------------------------------
_LAZY_IMPORTS = {
    # .capabilities.critique
    "CritiqueContext": ".capabilities.critique",
    "CritiquePolicy": ".capabilities.critique",
    "CriticCapability": ".capabilities.critique",
    "CritiqueRequest": ".capabilities.critique",
    "LLMCritiquePolicy": ".capabilities.critique",
    "MetricBasedCritiquePolicy": ".capabilities.critique",
    "OutputRelationship": ".capabilities.critique",
    # .capabilities.merge
    "MergePolicy": ".capabilities.merge",
    "MergeContext": ".capabilities.merge",
    "MergeCapability": ".capabilities.merge",
    "SemanticMergePolicy": ".capabilities.merge",
    "StatisticalMergePolicy": ".capabilities.merge",
    "GraphMergePolicy": ".capabilities.merge",
    "ListMergePolicy": ".capabilities.merge",
    "DictMergePolicy": ".capabilities.merge",
    "HierarchicalMerger": ".capabilities.merge",
    # .capabilities.refinement
    "RefinementPolicy": ".capabilities.refinement",
    "RefinementContext": ".capabilities.refinement",
    "RefinementCapability": ".capabilities.refinement",
    "LLMRefinementPolicy": ".capabilities.refinement",
    "IncrementalRefiner": ".capabilities.refinement",
    "ProgressiveRefinementTracker": ".capabilities.refinement",
    # .confidence
    "ConfidenceTracker": ".confidence",
    "UncertaintyPropagator": ".confidence",
    "ConfidenceMetrics": ".confidence",
    # .capabilities.validation
    "AnalysisValidationPolicy": ".capabilities.validation",
    "ValidationContext": ".capabilities.validation",
    "ValidationResult": ".capabilities.validation",
    "ValidationIssue": ".capabilities.validation",
    "ValidationCapability": ".capabilities.validation",
    "CrossShardConsistencyValidator": ".capabilities.validation",
    "EvidenceBasedValidator": ".capabilities.validation",
    "MultiLevelValidator": ".capabilities.validation",
    "ConsensusValidator": ".capabilities.validation",
    "ContradictionResolver": ".capabilities.validation",
    "Contradiction": ".capabilities.validation",
    # .relationships
    "Relationship": ".relationships",
    "RelationshipGraph": ".relationships",
    "RelationshipGraphBuilder": ".relationships",
    # .games.hypothesis.types
    "SubjectType": ".games.hypothesis.types",
    "SubjectReference": ".games.hypothesis.types",
    "ObservationType": ".games.hypothesis.types",
    "Observation": ".games.hypothesis.types",
    "HypothesisDomain": ".games.hypothesis.types",
    "HypothesisContext": ".games.hypothesis.types",
    "EvidenceType": ".games.hypothesis.types",
    "Evidence": ".games.hypothesis.types",
    "TriggerType": ".games.hypothesis.types",
    "HypothesisFormationTrigger": ".games.hypothesis.types",
    "EvaluationDecision": ".games.hypothesis.types",
    "EvaluationResult": ".games.hypothesis.types",
    "HypothesisStatus": ".games.hypothesis.types",
    "HypothesisFilter": ".games.hypothesis.types",
    # .games.hypothesis.strategies
    "HypothesisFormationStrategy": ".games.hypothesis.strategies",
    "EvidenceGatheringStrategy": ".games.hypothesis.strategies",
    "HypothesisEvaluationStrategy": ".games.hypothesis.strategies",
    "LLMHypothesisFormation": ".games.hypothesis.strategies",
    "RuleBasedHypothesisFormation": ".games.hypothesis.strategies",
    "QueryBasedEvidence": ".games.hypothesis.strategies",
    "LLMReasoningEvidence": ".games.hypothesis.strategies",
    "CompositeEvidenceStrategy": ".games.hypothesis.strategies",
    "LLMEvaluation": ".games.hypothesis.strategies",
    "RuleBasedEvaluation": ".games.hypothesis.strategies",
    # .games.hypothesis.tracking
    "TrackedHypothesis": ".games.hypothesis.tracking",
    "HypothesisTrackingCapability": ".games.hypothesis.tracking",
    # .hypothesis.driven (legacy)
    "HypothesisTestResult": ".hypothesis.driven",
    "HypothesisDrivenExplorer": ".hypothesis.driven",
    # .attention
    "IncrementalQueryCapability": ".attention",
    "QueryDrivenExplorationCapability": ".attention",
    "ExplorationResult": ".attention",
    "ScopeBasedQueryGenerator": ".attention",
    "AdaptiveQueryGenerator": ".attention",
    # .capabilities.synthesis
    "IncrementalSynthesizer": ".capabilities.synthesis",
    "SynthesisCapability": ".capabilities.synthesis",
    "SynthesisUpdate": ".capabilities.synthesis",
    # .models
    "Critique": ".models",
    "Reflection": ".models",
    "Hypothesis": ".models",
    # .capabilities.consciousness
    "ConsciousnessCapability": ".capabilities.consciousness",
    "SystemDocumentation": ".capabilities.consciousness",
    # .hooks
    "hookable": ".hooks",
    "register_hook": ".hooks",
    "auto_register_hooks": ".hooks",
    "Pointcut": ".hooks",
    "HookType": ".hooks",
    "ErrorMode": ".hooks",
    "HookContext": ".hooks",
    "RegisteredHook": ".hooks",
    "AgentHookRegistry": ".hooks",
    # .events
    "event_handler": ".events",
    "EventProcessingResult": ".events",
    "PROCESSED": ".events",
    # .capabilities.working_set
    "WorkingSetCapability": ".capabilities.working_set",
    "PageEvictionPolicy": ".capabilities.working_set",
    "LRUEvictionPolicy": ".capabilities.working_set",
    "ReferenceCountEvictionPolicy": ".capabilities.working_set",
    "CompositeEvictionPolicy": ".capabilities.working_set",
    # .capabilities.page_graph
    "PageGraphCapability": ".capabilities.page_graph",
    # .capabilities.agent_pool
    "AgentPoolCapability": ".capabilities.agent_pool",
    # .capabilities.query_attention
    "QueryAttentionCapability": ".capabilities.query_attention",
    # .capabilities.result
    "ResultCapability": ".capabilities.result",
    # .capabilities.batching
    "BatchingPolicy": ".capabilities.batching",
    "ClusteringBatchPolicy": ".capabilities.batching",
    "ContinuousBatchPolicy": ".capabilities.batching",
    "HybridBatchPolicy": ".capabilities.batching",
    # .capabilities.prefetching
    "PrefetchPolicy": ".capabilities.prefetching",
    "GraphPrefetchPolicy": ".capabilities.prefetching",
    "QueryPrefetchPolicy": ".capabilities.prefetching",
    "FeedbackPrefetchPolicy": ".capabilities.prefetching",
    "CompositePrefetchPolicy": ".capabilities.prefetching",
    # .capabilities.coordination
    "CoordinationPolicy": ".capabilities.coordination",
    "RoundRobinCoordinationPolicy": ".capabilities.coordination",
    "AffinityCoordinationPolicy": ".capabilities.coordination",
    "LoadBalancingCoordinationPolicy": ".capabilities.coordination",
    "CompositeCoordinationPolicy": ".capabilities.coordination",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value  # cache for subsequent access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
