"""Planning strategy policy implementations.

This module provides different strategies for generating plans:
- PlanningStrategyPolicy: Abstract base class
- ModelPredictiveControlStrategy: Plan horizon steps, execute, re-evaluate
- TopDownPlanningStrategy: Break goals into sub-goals first, then plan actions
- BottomUpPlanningStrategy: Start with concrete actions, infer goal structure
"""

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any
from overrides import override
from pydantic import BaseModel, Field

from ...models import (
    Action,
    ActionStatus,
    ActionType,
    CacheContext,
    ActionPlan,
    PlanExecutionContext,
    PlanningContext,
    PlanningParameters,
)
from ...base import Agent

logger = logging.getLogger(__name__)


def _get_action_types_description() -> str:
    """Get a formatted description of all available action types."""
    action_types = [at.value for at in ActionType]
    return ", ".join(sorted(action_types))


# ============================================================================
# Response Models for LLM Planning
# ============================================================================


class GoalHierarchyNode(BaseModel):
    """A node in the goal hierarchy."""

    goal: str = Field(description="Goal description")
    sub_goals: list[str] = Field(default_factory=list, description="List of sub-goal IDs")
    parent: str | None = Field(default=None, description="Parent goal ID, if any")


class ActionSpec(BaseModel):
    """Specification for an action (from LLM response)."""

    action_type: str = Field(
        description=f"Type of action. Available types: {_get_action_types_description()}"
    )
    description: str = Field(description="Human-readable description of the action")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Action-specific parameters")
    reasoning: str | None = Field(default=None, description="Why this action is needed")
    expected_outcome: str | None = Field(default=None, description="What we expect to learn")


class PlanGenerationResponse(BaseModel):
    """Response model for plan generation."""

    reasoning: str = Field(description="Step-by-step reasoning about how to achieve the goals")
    goal_hierarchy: dict[str, GoalHierarchyNode] = Field(
        default_factory=dict, description="Hierarchical goal structure"
    )
    actions: list[ActionSpec] = Field(default_factory=list, description="List of planned actions")


class ReplanningResponse(BaseModel):
    """Response model for replanning."""

    reasoning: str | None = Field(default=None, description="Reasoning for the replanning")
    actions: list[ActionSpec] = Field(default_factory=list, description="List of new actions")


class GoalHierarchyDecompositionResponse(BaseModel):
    """Response model for goal hierarchy decomposition."""

    goal_hierarchy: dict[str, GoalHierarchyNode] = Field(
        default_factory=dict, description="Hierarchical goal structure"
    )


class GoalHierarchyInferenceResponse(BaseModel):
    """Response model for goal hierarchy inference."""

    goal_hierarchy: dict[str, GoalHierarchyNode] = Field(
        default_factory=dict, description="Hierarchical goal structure"
    )


class PlanningStrategyPolicy(ABC):
    """Policy for how to generate plans.

    This is NOT an enum - it's an actual implementation.
    """

    def __init__(self, agent: Agent | None = None):
        """Initialize planning strategy with agent reference.
        
        Args:
            agent: Agent instance for LLM inference. If None, must be set via set_agent().
        """
        self.agent = agent

    def set_agent(self, agent: Agent) -> None:
        """Set the agent reference for LLM inference."""
        self.agent = agent

    @abstractmethod
    async def generate_plan(
        self,
        planning_context: PlanningContext,
        params: PlanningParameters,
        learned_patterns: list[Any] | None = None,
        cache_context: CacheContext | None = None,
    ) -> ActionPlan:
        """Generate initial plan.

        Args:
            goals: List of goals to achieve
            planning_context: Execution context and other context information
            params: Planning parameters
            learned_patterns: Learned patterns from similar successful plans
            cache_context: Cache-aware planning context
        """
        pass

    @abstractmethod
    async def replan_horizon(
        self,
        plan: ActionPlan,
        planning_context: PlanningContext,
        params: PlanningParameters,
        learned_patterns: list[Any] | None = None,
        cache_context: CacheContext | None = None,
    ) -> list[Action]:
        """Replan next N actions (for MPC).
        
        Args:
            plan: Current plan with execution progress
            planning_context: Execution context and other context information
            params: Planning parameters
            learned_patterns: Learned patterns from similar successful plans
            cache_context: Cache-aware planning context
        """
        pass

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Strip markdown code fences (```json ... ```) from LLM responses."""
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
        stripped = re.sub(r"\n?```\s*$", "", stripped)
        return stripped.strip()

    def _parse_plan_response(self, response: Any) -> PlanGenerationResponse:
        """Parse LLM response into PlanGenerationResponse using pydantic."""
        text = (
            response.generated_text if hasattr(response, "generated_text") else str(response)
        )
        # Extract JSON from response
        text = self._strip_markdown_fences(text)
        try:
            # TODO: Extract action parameters and populate ActionSpec properly
            return PlanGenerationResponse.model_validate_json(text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response as PlanGenerationResponse: {e}")
            # Return minimal valid response
            return PlanGenerationResponse(
                reasoning=text,
                goal_hierarchy={},
                actions=[],
            )

    def _parse_replanning_response(self, response: Any) -> ReplanningResponse:
        """Parse LLM response into ReplanningResponse using pydantic."""
        text = (
            response.generated_text if hasattr(response, "generated_text") else str(response)
        )
        # Extract JSON from response
        text = self._strip_markdown_fences(text)
        try:
            # TODO: Extract action parameters and populate ActionSpec properly
            return ReplanningResponse.model_validate_json(text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response as ReplanningResponse: {e}")
            # Return minimal valid response
            return ReplanningResponse(
                reasoning=text,
                actions=[],
            )

    def _convert_actions(self, action_specs: list[ActionSpec]) -> list[Action]:
        """Convert ActionSpec objects to Action objects."""
        actions = []
        for i, spec in enumerate(action_specs):
            try:
                actions.append(
                    Action(
                        action_id=f"action_{int(time.time() * 1000)}_{i}",
                        agent_id="",  # Will be set by caller
                        action_type=ActionType(spec.action_type),
                        description=spec.description,
                        parameters=spec.parameters,
                        reasoning=spec.reasoning,
                        expected_outcome=spec.expected_outcome,
                    )
                )
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to convert action {i}: {e}")
                continue
        return actions

    def _build_planning_prompt(
        self,
        goals: list[str],
        planning_context: PlanningContext,
        params: PlanningParameters,
        learned_patterns: list[Any] | None = None,
        cache_context: CacheContext | None = None,
    ) -> str:
        """Build LLM prompt for planning with learned patterns and cache context."""

        # TODO: Action descriptions, context summary, action arguments parsing, etc.
        # TODO: Structure the prompt so that the more stable parts are in the prefix
        #       and only the variable parts (goals, context) are in the suffix to
        #       optimize KV caching.

        goals_str = "\n".join(f"- {g}" for g in goals)

        # Add learned patterns if available
        patterns_section = ""
        if learned_patterns and len(learned_patterns) > 0:
            patterns_section = "\n\n## Learned Patterns from Similar Successful Plans\n\n"
            for i, pattern in enumerate(learned_patterns[:3], 1):  # Top 3 patterns
                if isinstance(pattern, dict):
                    patterns_section += f"{i}. **Pattern**: {pattern.get('description', 'N/A')}\n"
                    patterns_section += f"   - **Success Rate**: {pattern.get('success_rate', 0):.1%}\n"
                    patterns_section += f"   - **Key Actions**: {', '.join(pattern.get('common_action_types', [])[:5])}\n"
                else:
                    # PlanPattern model
                    patterns_section += f"{i}. **Pattern**: {pattern.description}\n"
                    patterns_section += f"   - **Success Rate**: {pattern.success_rate:.1%}\n"
                    patterns_section += f"   - **Key Actions**: {', '.join(pattern.common_action_types[:5])}\n"

        # Add cache context if available
        cache_section = ""
        if cache_context and cache_context.working_set:
            cache_section = "\n\n## Cache Context\n\n"
            cache_section += f"- **Working Set Size**: {len(cache_context.working_set)} pages\n"
            cache_section += f"- **Cache Capacity**: Min={cache_context.min_cache_size}, Ideal={cache_context.ideal_cache_size}\n"

            # Add central pages from page graph summary
            if cache_context.page_graph_summary:
                central_pages = cache_context.page_graph_summary.get("central_pages", [])
                if central_pages:
                    cache_section += f"- **Central Pages** (high connectivity): {', '.join(central_pages[:5])}\n"

            # Add spatial locality hints
            if cache_context.spatial_locality:
                cache_section += f"- **Pages with Spatial Locality**: {len(cache_context.spatial_locality)} groups\n"
                cache_section += "  *Hint: Group actions by spatially related pages for cache efficiency*\n"

        # Add execution context if available
        exec_context_section = ""
        if planning_context.execution_context:
            completed = len(planning_context.execution_context.completed_action_ids)
            exec_context_section = f"\n\n## Current Execution Context\n\n"
            exec_context_section += f"- **Completed Actions**: {completed}\n"

        # TODO: The following context fields are not yet included:
        # - planning_context.action_descriptions
        # - planning_context.custom_data # Includes memory architecture
        # - planning_context.constraints
        # - planning_context.recalled_memories

        action_descriptions = ""
        for group_desc, group_actions in planning_context.action_descriptions:
            if group_desc:
                action_descriptions += f"\n### {group_desc}\n"
            for action_key, action_desc in group_actions.items():
                action_descriptions += f"- {action_key}: {action_desc}\n"

        # Include memory architecture guidance if available
        memory_guidance = planning_context.custom_data.get("memory_architecture_guidance", "")
        memory_section = f"\n{memory_guidance}\n" if memory_guidance else ""

        return f"""You are creating a plan to achieve multiple goals.

## Goals

{goals_str}
{patterns_section}
{cache_section}
{exec_context_section}

Available actions:
{action_descriptions}
{memory_section}

## Your Task

1. **Decompose goals** into a hierarchical structure (main goals -> sub-goals)
2. **Plan concrete actions** for the next {params.planning_horizon} steps
3. **Consider learned patterns** from similar successful plans above
4. **Optimize for cache efficiency** by grouping actions on spatially related pages
5. **Explain your reasoning** step-by-step

## Output Format

The response must be valid JSON matching the expected schema. The available action types are specified in the schema.
"""


class ModelPredictiveControlStrategy(PlanningStrategyPolicy):
    """Model-Predictive Control strategy: plan horizon steps, execute, re-evaluate."""

    def __init__(self, horizon: int = 5, agent: Agent | None = None):
        super().__init__(agent=agent)
        self.horizon = horizon

    async def generate_plan(
        self,
        planning_context: PlanningContext,
        params: PlanningParameters,
        learned_patterns: list[Any] | None = None,
        cache_context: CacheContext | None = None,
    ) -> ActionPlan:
        """Generate initial plan with horizon steps."""
        if not self.agent:
            raise RuntimeError("Agent reference not set. Call set_agent() or pass agent to __init__")

        # Build prompt for LLM
        prompt = self._build_planning_prompt(
            planning_context=planning_context,
            params=params,
            learned_patterns=learned_patterns,
            cache_context=cache_context,
        )

        # Get LLM response using Agent.infer
        logger.warning(
            f"\n"
            f"            ╔══════════════════════════════════════╗\n"
            f"            ║  💬 MPC: agent.infer() — LLM CALL   ║\n"
            f"            ║  prompt_len={len(prompt):<24}║\n"
            f"            ║  max_tokens={params.max_planning_tokens:<24}║\n"
            f"            ╚══════════════════════════════════════╝"
        )
        response = await self.agent.infer(
            prompt=prompt,
            max_tokens=params.max_planning_tokens,
            json_schema=PlanGenerationResponse.model_json_schema(),
        )
        resp_text = response.generated_text if hasattr(response, "generated_text") else str(response)
        logger.warning(
            f"            💬 MPC: agent.infer() returned — response_len={len(resp_text)}"
        )

        # Parse into plan structure using pydantic
        plan_response = self._parse_plan_response(response)

        # Convert goal_hierarchy from pydantic models to dict format expected by ActionPlan
        goal_hierarchy_dict = {}
        for goal_id, goal_node in plan_response.goal_hierarchy.items():
            goal_hierarchy_dict[goal_id] = {
                "goal": goal_node.goal,
                "sub_goals": goal_node.sub_goals,
                "parent": goal_node.parent,
            }

        # Convert actions
        actions = self._convert_actions(plan_response.actions)[: params.planning_horizon]

        return ActionPlan(
            plan_id=f"plan_{int(time.time() * 1000)}",
            agent_id="",  # Set by caller
            goals=planning_context.goals,
            goal_hierarchy=goal_hierarchy_dict,
            actions=actions,
            planning_horizon=params.planning_horizon,
            replan_every_n_steps=params.replan_every_n_steps,
            generation_method="llm",
            initial_reasoning=plan_response.reasoning,
        )

    @override
    async def replan_horizon(
        self,
        plan: ActionPlan,
        planning_context: PlanningContext,
        params: PlanningParameters,
        learned_patterns: list[Any] | None = None,
        cache_context: CacheContext | None = None,
    ) -> list[Action]:
        """Replan next horizon steps based on execution so far."""
        if not self.agent:
            raise RuntimeError("Agent reference not set. Call set_agent() or pass agent to __init__")

        # Build prompt with execution history
        prompt = self._build_replanning_prompt(plan, planning_context, params)

        # Get LLM response using Agent.infer
        response = await self.agent.infer(
            prompt=prompt,
            max_tokens=params.max_planning_tokens,
            json_schema=ReplanningResponse.model_json_schema(),
        )

        # Parse using pydantic
        replan_response = self._parse_replanning_response(response)
        new_actions = self._convert_actions(replan_response.actions)[: params.planning_horizon]
        return new_actions

    def _build_replanning_prompt(self, plan: ActionPlan, planning_context: PlanningContext, params: PlanningParameters) -> str:
        """Build replanning prompt with detailed context."""

        # TODO: This prompt is very weak. Needs improvement.
        # TODO: It does not even provide action descriptions or context summary.

        # TODO: Structure the prompt so that the more stable parts are in the prefix
        #       and only the variable parts (goals, context) are in the suffix to
        #       optimize KV caching.

        # Summarize what's been done
        completed_actions = [
            a for a in plan.actions if a.status == ActionStatus.COMPLETED
        ]

        summary = "\n".join(
            [
                f"- {a.action_type}: {a.description} -> {'✓' if a.result and a.result.success else '✗'}"
                for a in completed_actions
            ]
        )

        # TODO: The following context fields are not yet included:
        # - planning_context.goals
        # - planning_context.action_descriptions
        # - planning_context.custom_data # Includes memory architecture
        # - planning_context.constraints
        # - planning_context.recalled_memories

        action_descriptions = ""
        for group_desc, group_actions in planning_context.action_descriptions:
            if group_desc:
                action_descriptions += f"\n### {group_desc}\n"
            for action_key, action_desc in group_actions.items():
                action_descriptions += f"- {action_key}: {action_desc}\n"

        # Include memory architecture guidance if available
        memory_guidance = planning_context.custom_data.get("memory_architecture_guidance", "")
        memory_section = f"\n{memory_guidance}\n" if memory_guidance else ""

        return f"""You are replanning based on progress so far.

Original Goals:
{chr(10).join(f"- {g}" for g in plan.goals)}

Available actions:
{action_descriptions}
{memory_section}

Progress So Far:
{summary}

Current Context:
- Completed: {len(completed_actions)}/{len(plan.actions)} actions
- Findings: {len(plan.execution_context.findings)} items
- Children: {len(plan.execution_context.spawned_children)} spawned

Replan the next {params.planning_horizon} actions based on:
1. What we've learned
2. What's still needed
3. Any issues encountered

The response must be valid JSON matching the expected schema."""


class TopDownPlanningStrategy(PlanningStrategyPolicy):
    """Top-down: Break goals into sub-goals first, then plan actions."""

    def __init__(self, agent: Agent | None = None):
        super().__init__(agent=agent)

    @override
    async def generate_plan(
        self,
        planning_context: PlanningContext,
        params: PlanningParameters,
        learned_patterns: list[Any] | None = None,
        cache_context: CacheContext | None = None,
    ) -> ActionPlan:
        """Generate plan via top-down decomposition."""
        if not self.agent:
            raise RuntimeError("Agent reference not set. Call set_agent() or pass agent to __init__")

        # First, decompose goals into hierarchy
        decomposition_prompt = f"""Break down these goals into a hierarchical structure:

Goals: {', '.join(planning_context.goals)}

The response must be valid JSON matching the expected schema."""

        decomp_response = await self.agent.infer(
            prompt=decomposition_prompt,
            max_tokens=params.max_planning_tokens,
            json_schema=GoalHierarchyDecompositionResponse.model_json_schema(),
        )

        # Parse using pydantic
        try:
            text = decomp_response.generated_text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                decomp_data = GoalHierarchyDecompositionResponse(**data)
            else:
                decomp_data = GoalHierarchyDecompositionResponse()
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse decomposition response: {e}")
            decomp_data = GoalHierarchyDecompositionResponse()

        # Convert to dict format
        goal_hierarchy = {}
        for goal_id, goal_node in decomp_data.goal_hierarchy.items():
            goal_hierarchy[goal_id] = {
                "goal": goal_node.goal,
                "sub_goals": goal_node.sub_goals,
                "parent": goal_node.parent,
            }

        # Then, plan actions for leaf goals
        leaf_goals = [
            g for g, data in goal_hierarchy.items() if not data.get("sub_goals")
        ]

        # TODO: The following context fields are not yet included:
        # - planning_context.action_descriptions
        # - planning_context.custom_data # Includes memory architecture
        # - planning_context.constraints
        # - planning_context.recalled_memories

        action_descriptions = ""
        for group_desc, group_actions in planning_context.action_descriptions:
            if group_desc:
                action_descriptions += f"\n### {group_desc}\n"
            for action_key, action_desc in group_actions.items():
                action_descriptions += f"- {action_key}: {action_desc}\n"

        # Include memory architecture guidance if available
        memory_guidance = planning_context.custom_data.get("memory_architecture_guidance", "")
        memory_section = f"\n{memory_guidance}\n" if memory_guidance else ""

        planning_prompt = f"""Create actions for these leaf goals:

Leaf goals: {leaf_goals}

Available actions:
{action_descriptions}
{memory_section}

The response must be valid JSON matching the expected schema."""

        action_response = await self.agent.infer(
            prompt=planning_prompt,
            max_tokens=params.max_planning_tokens,
            json_schema=ReplanningResponse.model_json_schema(),  # Reuse replanning response for actions
        )
        action_data = self._parse_replanning_response(action_response)
        actions = self._convert_actions(action_data.actions)[: params.planning_horizon]

        return ActionPlan(
            plan_id=f"plan_{int(time.time() * 1000)}",
            agent_id="",
            goals=planning_context.goals,
            goal_hierarchy=goal_hierarchy,
            actions=actions,
            planning_horizon=params.planning_horizon,
            replan_every_n_steps=params.replan_every_n_steps,
            generation_method="llm",
            initial_reasoning=action_data.reasoning,
        )

    @override
    async def replan_horizon(
        self,
        plan: ActionPlan,
        planning_context: PlanningContext,
        params: PlanningParameters,
        learned_patterns: list[Any] | None = None,
        cache_context: CacheContext | None = None,
    ) -> list[Action]:
        """Replan based on current progress."""
        if not self.agent:
            raise RuntimeError("Agent reference not set. Call set_agent() or pass agent to __init__")

        # TODO: Build a proper prompt
        # prompt = self._build_replanning_prompt(plan, planning_context, params)

        # TODO: The following context fields are not yet included:
        # - planning_context.action_descriptions
        # - planning_context.custom_data # Includes memory architecture
        # - planning_context.constraints
        # - planning_context.recalled_memories

        prompt = f"Replan actions for goals: {plan.goals} given current progress. The response must be valid JSON matching the expected schema."
        
        response = await self.agent.infer(
            prompt=prompt,
            max_tokens=params.max_planning_tokens,
            json_schema=ReplanningResponse.model_json_schema(),
        )
        replan_data = self._parse_replanning_response(response)
        return self._convert_actions(replan_data.actions)[: params.planning_horizon]


class BottomUpPlanningStrategy(PlanningStrategyPolicy):
    """Bottom-up: Start with concrete actions, infer goal structure."""

    def __init__(self, agent: Agent | None = None):
        super().__init__(agent=agent)

    @override
    async def generate_plan(
        self,
        planning_context: PlanningContext,
        params: PlanningParameters,
        learned_patterns: list[Any] | None = None,
        cache_context: CacheContext | None = None,
    ) -> ActionPlan:
        """Generate plan bottom-up from concrete actions."""
        if not self.agent:
            raise RuntimeError("Agent reference not set. Call set_agent() or pass agent to __init__")

        # TODO: Build a proper prompt
        # prompt = self._build_planning_prompt(plan, planning_context, params)

        # TODO: The following context fields are not yet included:
        # - planning_context.action_descriptions
        # - planning_context.custom_data # Includes memory architecture
        # - planning_context.constraints
        # - planning_context.recalled_memories

        # Generate concrete actions first
        action_prompt = f"""Generate concrete actions to achieve: {', '.join(planning_context.goals)}

Focus on specific, executable steps. The response must be valid JSON matching the expected schema."""

        action_response = await self.agent.infer(
            prompt=action_prompt,
            max_tokens=params.max_planning_tokens,
            json_schema=ReplanningResponse.model_json_schema(),
        )
        action_data = self._parse_replanning_response(action_response)
        actions = self._convert_actions(action_data.actions)[: params.planning_horizon]

        # Infer goal hierarchy from actions
        hierarchy_prompt = f"""Given these actions, infer the goal hierarchy:

Actions: {[a.description for a in actions]}

The response must be valid JSON matching the expected schema."""

        hier_response = await self.agent.infer(
            prompt=hierarchy_prompt,
            max_tokens=params.max_planning_tokens,
            json_schema=GoalHierarchyInferenceResponse.model_json_schema(),
        )
        
        # Parse using pydantic
        try:
            text = hier_response.generated_text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                hier_data = GoalHierarchyInferenceResponse(**data)
            else:
                hier_data = GoalHierarchyInferenceResponse()
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse hierarchy response: {e}")
            hier_data = GoalHierarchyInferenceResponse()

        # Convert to dict format
        goal_hierarchy = {}
        for goal_id, goal_node in hier_data.goal_hierarchy.items():
            goal_hierarchy[goal_id] = {
                "goal": goal_node.goal,
                "sub_goals": goal_node.sub_goals,
                "parent": goal_node.parent,
            }

        return ActionPlan(
            plan_id=f"plan_{int(time.time() * 1000)}",
            agent_id="",
            goals=planning_context.goals,
            goal_hierarchy=goal_hierarchy,
            actions=actions,
            planning_horizon=params.planning_horizon,
            replan_every_n_steps=params.replan_every_n_steps,
            generation_method="llm",
            initial_reasoning=action_data.reasoning,
        )

    @override
    async def replan_horizon(
        self,
        plan: ActionPlan,
        planning_context: PlanningContext,
        params: PlanningParameters,
        learned_patterns: list[Any] | None = None,
        cache_context: CacheContext | None = None,
    ) -> list[Action]:
        """Replan based on current progress."""
        if not self.agent:
            raise RuntimeError("Agent reference not set. Call set_agent() or pass agent to __init__")

        # TODO: Build a proper prompt
        # prompt = self._build_replanning_prompt(plan, planning_context, params)

        # TODO: The following context fields are not yet included:
        # - planning_context.goals
        # - planning_context.action_descriptions
        # - planning_context.custom_data # Includes memory architecture
        # - planning_context.constraints
        # - planning_context.recalled_memories
        prompt = f"Continue planning actions for goals: {plan.goals}. The response must be valid JSON matching the expected schema."
        
        response = await self.agent.infer(
            prompt=prompt,
            max_tokens=params.max_planning_tokens,
            json_schema=ReplanningResponse.model_json_schema(),
        )
        replan_data = self._parse_replanning_response(response)
        return self._convert_actions(replan_data.actions)[: params.planning_horizon]


def get_default_planning_strategy(params: PlanningParameters, agent: Agent | None = None) -> PlanningStrategyPolicy:
    """Get default planning strategy based on parameters.
    
    Args:
        params: Planning parameters
        agent: Optional agent reference (can be set later via set_agent())
    """
    from ...models import PlanningStrategy

    if params.strategy == PlanningStrategy.MPC:
        return ModelPredictiveControlStrategy(horizon=params.planning_horizon, agent=agent)
    elif params.strategy == PlanningStrategy.TOP_DOWN:
        return TopDownPlanningStrategy(agent=agent)
    elif params.strategy == PlanningStrategy.BOTTOM_UP:
        return BottomUpPlanningStrategy(agent=agent)
    else:  # HYBRID
        return ModelPredictiveControlStrategy(horizon=params.planning_horizon, agent=agent)

