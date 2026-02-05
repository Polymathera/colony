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

from .scope import AnalysisScope, ScopeAwareResult, merge_scopes
from .capabilities.critique import (
    Critique,
    CritiqueContext,
    CritiquePolicy,
    CriticCapability,
    CritiqueRequest,
    LLMCritiquePolicy,
    MetricBasedCritiquePolicy,
    OutputRelationship
)
from .capabilities.merge import (
    MergePolicy,
    MergeContext,
    MergeCapability,
    SemanticMergePolicy,
    StatisticalMergePolicy,
    GraphMergePolicy,
    ListMergePolicy,
    DictMergePolicy,
    HierarchicalMerger,
)
from .capabilities.refinement import (
    RefinementPolicy,
    RefinementContext,
    RefinementCapability,
    LLMRefinementPolicy,
    IncrementalRefiner,
    ProgressiveRefinementTracker,
)
from .confidence import (
    ConfidenceTracker,
    UncertaintyPropagator,
    ConfidenceMetrics,
)
from .capabilities.validation import (
    AnalysisValidationPolicy,
    ValidationContext,
    ValidationResult,
    ValidationIssue,
    ValidationCapability,
    CrossShardConsistencyValidator,
    EvidenceBasedValidator,
    MultiLevelValidator,
    ConsensusValidator,
    ContradictionResolver,
    Contradiction,
)
from .relationships import (
    Relationship,
    RelationshipGraph,
    RelationshipGraphBuilder,
)
# Unified hypothesis game system
from .games.hypothesis.types import (
    SubjectType,
    SubjectReference,
    ObservationType,
    Observation,
    HypothesisDomain,
    HypothesisContext,
    EvidenceType,
    Evidence,
    TriggerType,
    HypothesisFormationTrigger,
    EvaluationDecision,
    EvaluationResult,
    HypothesisStatus,
    HypothesisFilter,
)
from .games.hypothesis.strategies import (
    HypothesisFormationStrategy,
    EvidenceGatheringStrategy,
    HypothesisEvaluationStrategy,
    LLMHypothesisFormation,
    RuleBasedHypothesisFormation,
    QueryBasedEvidence,
    LLMReasoningEvidence,
    CompositeEvidenceStrategy,
    LLMEvaluation,
    RuleBasedEvaluation,
)
from .games.hypothesis.tracking import (
    TrackedHypothesis,
    HypothesisTrackingCapability,
)

# Legacy hypothesis support (deprecated - use game-based hypothesis system)
from .hypothesis.driven import (
    HypothesisTestResult,
    HypothesisDrivenExplorer,
)
from .attention import (
    IncrementalQueryCapability,
    QueryDrivenExplorationCapability,
    ExplorationResult,
    ScopeBasedQueryGenerator,
    AdaptiveQueryGenerator,
)
from .capabilities.synthesis import (
    IncrementalSynthesizer,
    SynthesisCapability,
    SynthesisUpdate,
)
from .models import (
    Reflection,
    Hypothesis,
)

# Consciousness integration
from .capabilities.consciousness import (
    ConsciousnessCapability,
    SystemDocumentation,
)

# Hook system
from .hooks import (
    hookable,
    register_hook,
    auto_register_hooks,
    Pointcut,
    HookType,
    ErrorMode,
    HookContext,
    RegisteredHook,
    AgentHookRegistry,
)

# Event processing
from .events import (
    event_handler,
    EventProcessingResult,
    PROCESSED,
)

# Capabilities for multi-agent execution framework primitives
from .capabilities import (
    WorkingSetCapability,
    PageGraphCapability,
    AgentPoolCapability,
    QueryAttentionCapability,
    ResultCapability,
    # Eviction policies
    PageEvictionPolicy,
    LRUEvictionPolicy,
    ReferenceCountEvictionPolicy,
    CompositeEvictionPolicy,
    # Batching policies
    BatchingPolicy,
    ClusteringBatchPolicy,
    ContinuousBatchPolicy,
    HybridBatchPolicy,
    # Prefetch policies
    PrefetchPolicy,
    GraphPrefetchPolicy,
    QueryPrefetchPolicy,
    FeedbackPrefetchPolicy,
    CompositePrefetchPolicy,
    # Coordination policies
    CoordinationPolicy,
    RoundRobinCoordinationPolicy,
    AffinityCoordinationPolicy,
    LoadBalancingCoordinationPolicy,
    CompositeCoordinationPolicy,
)


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

