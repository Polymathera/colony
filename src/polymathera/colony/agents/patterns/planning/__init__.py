"""Multi-agent planning framework for Polymathera.

This package provides a comprehensive framework for planning and coordination:

Core Components:
- Plan: Multi-goal hierarchical plan with cache context
- Agent: Base class for agents that use multi-step planning
- Planner: Creates and executes plans incrementally
- PlanBlackboard: Distributed plan storage with event-driven coordination

Planning Strategies:
- ModelPredictiveActionPlanningStrategy: Plan horizon steps, execute, re-evaluate
- TopDownActionPlanningStrategy: Break goals into sub-goals first
- BottomUpActionPlanningStrategy: Start with concrete actions

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
from .planner import (
    ActionPlanner,
    CacheAwareActionPlanner,
    SequentialPlanner,
    create_cache_aware_planner,
)
from .access import PlanAccessPolicy, HierarchicalAccessPolicy
from .strategies import (
    BottomUpActionPlanningStrategy,
    ModelPredictiveActionPlanningStrategy,
    ActionPlanningStrategy,
    SCOPE_SELECTION_THRESHOLD,
    ScopeSelectionResponse,
    TopDownActionPlanningStrategy,
    get_default_planning_strategy,
)
from .prompts import (
    PromptFormattingStrategy,
    MarkdownPromptFormatting,
    XMLPromptFormatting,
    NumericIDPromptFormatting,
    AliasPromptFormatting,
)

__all__ = [
    # Core classes
    "ActionPlanner",
    "PlanBlackboard",
    "CacheAwareActionPlanner",
    "SequentialPlanner",
    "create_cache_aware_planner",
    # Policies
    "PlanAccessPolicy",
    "HierarchicalAccessPolicy",
    # Strategies
    "ActionPlanningStrategy",
    # Prompt formatting
    "PromptFormattingStrategy",
    "MarkdownPromptFormatting",
    "XMLPromptFormatting",
    "AliasPromptFormatting",
    "NumericIDPromptFormatting",
    "ModelPredictiveActionPlanningStrategy",
    "TopDownActionPlanningStrategy",
    "BottomUpActionPlanningStrategy",
    "get_default_planning_strategy",
    # Scope selection
    "ScopeSelectionResponse",
    "SCOPE_SELECTION_THRESHOLD",
]