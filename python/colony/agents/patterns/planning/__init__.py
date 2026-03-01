"""Multi-agent planning framework for Polymathera.

This package provides a comprehensive framework for planning and coordination:

Core Components:
- Plan: Multi-goal hierarchical plan with cache context
- Agent: Base class for agents that use multi-step planning
- Planner: Creates and executes plans incrementally
- PlanBlackboard: Distributed plan storage with event-driven coordination

Planning Strategies:
- ModelPredictiveControlStrategy: Plan horizon steps, execute, re-evaluate
- TopDownPlanningStrategy: Break goals into sub-goals first
- BottomUpPlanningStrategy: Start with concrete actions

Example:
    ```python
    from polymathera.colony.agents import (
        Agent,
        PlanningParameters,
        PlanningStrategy,
    )

    # Define custom planning agent
    class MyAgent(Agent):
        pass

    # Spawn with planning configuration
    spec = AgentBlueprint(
        agent_class="MyAgent",
        metadata={
            "goals": ["Analyze repository", "Generate report"],
            "planning_params": {
                "strategy": "mpc",
                "planning_horizon": 5,
            }
        }
    )
    ```
"""

from .blackboard import PlanBlackboard
from .coordination import (
    CoordinationStrategy,
    ConsensusNegotiation,
    MarketBasedNegotiation,
    NegotiationProtocol,
    PartialGlobalPlanning,
)
from .evaluator import PlanEvaluator, PlanSelector
from .planner import (
    ActionPlanner,
    CacheAwareActionPlanner,
    SequentialPlanner,
    create_cache_aware_planner,
)
from .policies import (
    CacheAwarePlanningPolicy,
    LearningPlanningPolicy,
    CoordinationPlanningPolicy,
    CacheAwareConflictDetector,
    CacheAwareConflictResolver,
    ConflictResolver,
    PlanAccessPolicy,
    HierarchicalAccessPolicy,
)
from .replanning import (
    ReplanningDecision,
    ReplanningPolicy,
    PeriodicReplanningPolicy,
    AdaptiveReplanningPolicy,
    CompositeReplanningPolicy,
)
from .strategies import (
    BottomUpPlanningStrategy,
    ModelPredictiveControlStrategy,
    PlanningStrategyPolicy,
    SCOPE_SELECTION_THRESHOLD,
    ScopeSelectionResponse,
    TopDownPlanningStrategy,
    get_default_planning_strategy,
)

__all__ = [
    # Core classes
    "ActionPlanner",
    "PlanBlackboard",
    "CacheAwareActionPlanner",
    "SequentialPlanner",
    "create_cache_aware_planner",
    # Evaluation and Selection
    "PlanEvaluator",
    "PlanSelector",
    # Policies
    "CacheAwarePlanningPolicy",
    "LearningPlanningPolicy",
    "CoordinationPlanningPolicy",
    "CacheAwareConflictDetector",
    "CacheAwareConflictResolver",
    "ConflictResolver",
    "PlanAccessPolicy",
    "HierarchicalAccessPolicy",
    # Coordination
    "CoordinationStrategy",
    "PartialGlobalPlanning",
    "NegotiationProtocol",
    "MarketBasedNegotiation",
    "ConsensusNegotiation",
    # Enums
    "PlanStatus",
    "PlanVisibility",
    "ActionStatus",
    "PlanningStrategy",
    # Replanning policies
    "ReplanningDecision",
    "ReplanningPolicy",
    "PeriodicReplanningPolicy",
    "AdaptiveReplanningPolicy",
    "CompositeReplanningPolicy",
    # Strategies
    "PlanningStrategyPolicy",
    "ModelPredictiveControlStrategy",
    "TopDownPlanningStrategy",
    "BottomUpPlanningStrategy",
    "get_default_planning_strategy",
    # Scope selection
    "ScopeSelectionResponse",
    "SCOPE_SELECTION_THRESHOLD",
]