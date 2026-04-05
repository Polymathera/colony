"""Planning capabilities — cognitive capabilities for plan-aware agents.

These capabilities augment an agent's reasoning about its own planning.
They support three consumption modes:

1. **Pre-programmed**: ``CacheAwareActionPlanner`` calls the programmatic API
   in a hardcoded sequence (learning → cache → strategy → replan).
2. **LLM-selected**: Simple action policies (e.g., ``MinimalActionPolicy``)
   expose ``@action_executor`` wrappers that the LLM can optionally call.
3. **Code-generated**: ``CodeGenerationActionPolicy`` calls the programmatic
   API directly in LLM-generated Python code.

Each capability provides both a **programmatic API** (complex parameters,
system objects) and an **LLM API** (``@action_executor`` methods with simple,
LLM-producible parameters).

Example::

    from polymathera.colony.agents.patterns.planning.capabilities import (
        CacheAnalysisCapability,
        PlanLearningCapability,
        PlanCoordinationCapability,
    )

    # Register on agent — CacheAwareActionPlanner does this automatically
    agent.add_capability(CacheAnalysisCapability(agent=agent))

    # Programmatic API (used by CacheAwareActionPlanner and CodeGenerationActionPolicy)
    cache_ctx = await cache_cap.analyze_cache_requirements(planning_context)

    # LLM API (used by MinimalActionPolicy via JSON action selection)
    # The LLM sees: analyze_cache — Analyze cache requirements for given pages.
    #   Parameters: page_ids?: list[str]
"""

from .cache_analysis import CacheAnalysisCapability
from .learning import PlanLearningCapability
from .replanning import (
    ReplanningCapability,
    ReplanningDecision,
    ReplanningPolicy,
    PeriodicReplanningPolicy,
    AdaptiveReplanningPolicy,
    PlanExhaustionReplanningPolicy,
    CompositeReplanningPolicy,
)
from .evaluator import (
    PlanEvaluator,
    PlanSelector,
    PlanEvaluationCapability,
)
from .coordination import (
    CoordinationStrategy,
    PartialGlobalPlanning,
    NegotiationProtocol,
    ConsensusNegotiation,
    MarketBasedNegotiation,
    CacheAwarePlanConflictDetector,
    CacheAwarePlanConflictResolver,
    ActionPlanConflictResolver,
    PlanCoordinationCapability,
)

__all__ = [
    "CacheAnalysisCapability",
    "PlanLearningCapability",
    "PlanEvaluationCapability",
    "ReplanningCapability",
    "ReplanningDecision",
    "ReplanningPolicy",
    "PeriodicReplanningPolicy",
    "AdaptiveReplanningPolicy",
    "PlanExhaustionReplanningPolicy",
    "CompositeReplanningPolicy",
    # Evaluation and Selection
    "PlanEvaluator",
    "PlanSelector",
    # Coordination
    "CoordinationStrategy",
    "PartialGlobalPlanning",
    "NegotiationProtocol",
    "MarketBasedNegotiation",
    "ConsensusNegotiation",
    "CacheAwarePlanConflictDetector",
    "CacheAwarePlanConflictResolver",
    "ActionPlanConflictResolver",
    "PlanCoordinationCapability",
]
