"""Planning strategy policy implementations.

This module provides different strategies for generating plans:
- ActionPlanningStrategy: Abstract base class
- ModelPredictiveActionPlanningStrategy: Plan horizon steps, execute, re-evaluate
- TopDownActionPlanningStrategy: Break goals into sub-goals first, then plan actions
- BottomUpActionPlanningStrategy: Start with concrete actions, infer goal structure
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any
from overrides import override
from pydantic import BaseModel, Field

from ...models import (
    Action,
    ActionType,
    CacheContext,
    ActionPlan,
    PlanningContext,
    PlanningParameters,
    PlanStatus,
)
from ...base import Agent
from .prompts import (
    JsonExtractor,
    PromptFormattingStrategy,
    MarkdownPromptFormatting,
    XMLPromptFormatting,
    NumericIDPromptFormatting,
    AliasPromptFormatting,
)

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


class ScopeSelectionResponse(BaseModel):
    """Response model for scope selection (Phase 1 of hierarchical action scoping)."""

    reasoning: str = Field(description="Brief reasoning for scope selection")
    selected_groups: list[str] = Field(description="group_key values for relevant action groups")


# Skip scope selection when total groups are at or below this count.
# The overhead of an extra LLM call exceeds the savings at small group counts.
SCOPE_SELECTION_THRESHOLD: int = 6


class ActionPlanningStrategy(ABC):
    """Policy for how to generate plans.

    This is NOT an enum - it's an actual implementation.
    """

    def __init__(
        self,
        agent: Agent | None = None,
        prompt_formatting: PromptFormattingStrategy | None = None,
    ):
        """Initialize planning strategy with agent reference.

        Args:
            agent: Agent instance for LLM inference. If None, must be set via set_agent().
            prompt_formatting: Strategy for formatting action descriptions in the
                planner prompt. Defaults to XMLPromptFormatting.
        """
        self.agent = agent
        self.prompt_formatting = prompt_formatting or XMLPromptFormatting()

    def set_agent(self, agent: Agent) -> None:
        """Set the agent reference for LLM inference."""
        self.agent = agent

    def _get_goal_hierarchy(self, goal_hierarchy: dict[str, GoalHierarchyNode]) -> dict[str, dict[str, Any]]:
        # Convert to dict format
        goal_hierarchy_dict = {}
        for goal_id, goal_node in goal_hierarchy.items():
            goal_hierarchy_dict[goal_id] = {
                "goal": goal_node.goal,
                "sub_goals": goal_node.sub_goals,
                "parent": goal_node.parent,
            }
        return goal_hierarchy_dict

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

    async def replan_horizon(
        self,
        plan: ActionPlan,
        planning_context: PlanningContext,
        params: PlanningParameters,
        learned_patterns: list[Any] | None = None,
        cache_context: CacheContext | None = None,
    ) -> list[Action]:
        """Replan next horizon actions based on execution so far (for MPC).

        Args:
            plan: Current plan with execution progress
            planning_context: Execution context and other context information
            params: Planning parameters
            learned_patterns: Learned patterns from similar successful plans
            cache_context: Cache-aware planning context
        """
        if not self.agent:
            raise RuntimeError("Agent reference not set. Call set_agent() or pass agent to __init__")

        # Reuse cached scope selection from initial plan
        self._apply_cached_scope_filter(planning_context, plan)

        # Build prompt with execution history
        prompt = self.prompt_formatting.build_replanning_prompt(planning_context, plan, params)
        _log_planning_context_sizes(planning_context, prompt)

        # Get LLM response using Agent.infer
        response = await self.agent.infer(
            prompt=prompt,
            max_tokens=params.max_planning_tokens,
            temperature=0.3,  # More deterministic - TODO: Make configurable
            context_page_ids=[],  # TODO: Planning runs only on the prompt. Right?
            json_schema=ReplanningResponse.model_json_schema(),
        )

        # Parse using pydantic
        replan_response = self._parse_replanning_response(response)
        valid_keys = self._get_valid_action_keys(planning_context)
        new_actions = self._convert_actions(replan_response.actions, valid_keys)[: params.planning_horizon]
        return new_actions

    # ── Scope selection (Phase 1 of hierarchical action scoping) ──────────────

    @staticmethod
    def _should_use_scope_selection(planning_context: PlanningContext) -> bool:
        """Return True if scope selection is beneficial (enough groups to justify the extra call)."""
        return len(planning_context.action_group_summaries) > SCOPE_SELECTION_THRESHOLD

    async def _run_scope_selection(self, planning_context: PlanningContext) -> list[str]:
        """Run scope selection (Phase 1) and return selected group keys.

        On parse failure, falls back to all groups (no filtering).
        """
        if not self.agent:
            raise RuntimeError("Agent reference not set for scope selection")

        prompt = self.prompt_formatting.build_scope_selection_prompt(planning_context)
        all_keys = [s.group_key for s in planning_context.action_group_summaries]
        _log_planning_context_sizes(planning_context, prompt)

        logger.info(
            f"Scope selection: {len(all_keys)} groups available, "
            f"prompt_len={len(prompt)}"
        )

        try:
            response = await self.agent.infer(
                prompt=prompt,
                max_tokens=512,
                temperature=0.1,
                context_page_ids=[],
                json_schema=ScopeSelectionResponse.model_json_schema(),
            )
            text = response.generated_text if hasattr(response, "generated_text") else str(response)
            json_text = JsonExtractor.extract(text)
            result = ScopeSelectionResponse.model_validate_json(json_text)

            # Validate: only keep keys that actually exist
            valid = [k for k in result.selected_groups if k in all_keys]
            if not valid:
                logger.warning("Scope selection returned no valid group keys — using all groups")
                return all_keys

            logger.info(
                f"Scope selection: {len(valid)}/{len(all_keys)} groups selected — "
                f"reasoning: {result.reasoning[:120]}"
            )
            return valid

        except Exception as e:
            logger.warning(f"Scope selection failed ({e}) — using all groups")
            return all_keys

    async def _apply_scope_selection(self, planning_context: PlanningContext) -> list[str] | None:
        """Run scope selection if beneficial and filter planning_context.action_descriptions in place.

        Returns the selected group keys (for caching on ActionPlan), or None if scope selection
        was skipped (all groups included).
        """
        if not self._should_use_scope_selection(planning_context):
            return None

        selected = await self._run_scope_selection(planning_context)
        all_keys = [s.group_key for s in planning_context.action_group_summaries]
        if len(selected) == len(all_keys):
            return None  # All groups selected — no filtering needed

        # Filter action_descriptions to only include selected groups
        selected_set = set(selected)
        planning_context.action_descriptions = [
            desc for desc in planning_context.action_descriptions
            if desc.group_key in selected_set
        ]
        planning_context.selected_groups = selected
        return selected

    @staticmethod
    def _apply_cached_scope_filter(
        planning_context: PlanningContext,
        plan: ActionPlan,
    ) -> None:
        """Reuse cached selected_groups from a plan to filter action_descriptions for replanning."""
        if plan.selected_groups is None:
            return
        selected_set = set(plan.selected_groups)
        planning_context.action_descriptions = [
            desc for desc in planning_context.action_descriptions
            if desc.group_key in selected_set
        ]

    # ── End scope selection ─────────────────────────────────────────────────────

    def _parse_plan_response(self, response: Any) -> PlanGenerationResponse:
        """Parse LLM response into PlanGenerationResponse using pydantic."""
        text = (
            response.generated_text if hasattr(response, "generated_text") else str(response)
        )
        # Extract JSON from response
        text = JsonExtractor.extract(text)
        try:
            # TODO: Extract action parameters and populate ActionSpec properly
            return PlanGenerationResponse.model_validate_json(text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response as PlanGenerationResponse:\nRaw response: {text}\nError: {e}")
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
        text = JsonExtractor.extract(text)
        try:
            # TODO: Extract action parameters and populate ActionSpec properly
            return ReplanningResponse.model_validate_json(text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response as ReplanningResponse:\nRaw response: {text}\nError: {e}")
            # Return minimal valid response
            return ReplanningResponse(
                reasoning=text,
                actions=[],
            )

    @staticmethod
    def _get_valid_action_keys(planning_context: PlanningContext) -> set[str]:
        """Extract all valid action keys from planning context action descriptions."""
        valid_keys: set[str] = set()
        for group in planning_context.action_descriptions:
            for key in group.action_descriptions:
                valid_keys.add(key)
        # Also include built-in ActionType enum values
        for at in ActionType:
            valid_keys.add(at.value)
        return valid_keys

    @staticmethod
    def _resolve_action_key(
        raw_key: str, valid_keys: set[str]
    ) -> tuple[str, str]:
        """Resolve an action key against valid keys.

        Returns (resolved_key, resolution_type) where resolution_type is one of:
        "exact", "suffix", "prefix_strip", "method_name", "unresolved".
        """
        if raw_key in valid_keys:
            return raw_key, "exact"

        # Suffix match: LLM emitted just the method name
        suffix_matches = [k for k in valid_keys if k.endswith(f".{raw_key}")]
        if len(suffix_matches) == 1:
            return suffix_matches[0], "suffix"

        # Prefix-strip: LLM added a spurious prefix (e.g. "working.working_memory_store")
        parts = raw_key.split(".")
        for start in range(1, len(parts)):
            candidate = ".".join(parts[start:])
            # Check if candidate is a valid key directly
            if candidate in valid_keys:
                return candidate, "prefix_strip"
            # Check if candidate matches as a suffix of a valid key
            candidate_suffix_matches = [k for k in valid_keys if k.endswith(f".{candidate}")]
            if len(candidate_suffix_matches) == 1:
                return candidate_suffix_matches[0], "method_name"

        return raw_key, "unresolved"

    def _convert_actions(
        self,
        action_specs: list[ActionSpec],
        valid_action_keys: set[str] | None = None,
    ) -> list[Action]:
        """Convert ActionSpec objects to Action objects.

        action_type may be an ActionType enum value (e.g. "tool_use") or a
        capability action key (e.g. "AgentContextEngine.8dd64d54.inspect_memory_map").
        Action.action_type is typed ActionType | str, and the dispatcher uses
        str(action.action_type) for executor lookup, so both forms work.

        If valid_action_keys is provided, validates and attempts to resolve
        invalid keys via suffix/prefix matching.  Logs metrics for monitoring.
        """
        actions = []
        exact_matches = 0
        fuzzy_matches = 0
        unresolved = 0

        for i, spec in enumerate(action_specs):
            raw_key = spec.action_type

            # Validate and resolve if valid keys are available
            if valid_action_keys is not None:
                # First, try the formatting strategy's alias/numeric resolution
                strategy_resolved = self.prompt_formatting.resolve_action_key(raw_key)
                if strategy_resolved and strategy_resolved in valid_action_keys:
                    exact_matches += 1
                    raw_key = strategy_resolved
                else:
                    resolved_key, resolution = self._resolve_action_key(
                        raw_key, valid_action_keys
                    )
                    if resolution == "exact":
                        exact_matches += 1
                    elif resolution in ("suffix", "prefix_strip"):
                        fuzzy_matches += 1
                        logger.warning(
                            f"Action key resolved ({resolution}): "
                            f"'{raw_key}' → '{resolved_key}'"
                        )
                        raw_key = resolved_key
                    else:
                        unresolved += 1
                        logger.error(
                            f"Unresolved action key: '{raw_key}' — "
                            f"no match in {len(valid_action_keys)} valid keys"
                        )

            # Prefer ActionType enum when it matches; otherwise keep the raw
            # action key string — the dispatcher resolves it against executors.
            try:
                action_type: ActionType | str = ActionType(raw_key)
            except ValueError:
                action_type = raw_key

            actions.append(
                Action(
                    action_id=f"action_{int(time.time() * 1000)}_{i}",
                    agent_id="",  # Will be set by caller
                    action_type=action_type,
                    description=spec.description,
                    parameters=spec.parameters,
                    reasoning=spec.reasoning,
                    expected_outcome=spec.expected_outcome,
                )
            )

        total = len(action_specs)
        if valid_action_keys is not None and total > 0:
            logger.info(
                f"Action key accuracy: {exact_matches}/{total} exact, "
                f"{fuzzy_matches}/{total} fuzzy-resolved, "
                f"{unresolved}/{total} unresolved"
            )

        return actions


class ModelPredictiveActionPlanningStrategy(ActionPlanningStrategy):
    """Model-Predictive Control strategy: plan horizon steps, execute, re-evaluate."""

    def __init__(
        self,
        horizon: int = 5,
        agent: Agent | None = None,
        prompt_formatting: PromptFormattingStrategy | None = None,
    ):
        super().__init__(agent=agent, prompt_formatting=prompt_formatting)
        self.horizon = horizon

    @override
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

        # Phase 1: Scope selection (filters action_descriptions in place if beneficial)
        selected_groups = await self._apply_scope_selection(planning_context)

        # Phase 2: Build prompt with (possibly filtered) action descriptions
        prompt = self.prompt_formatting.build_planning_prompt(
            planning_context=planning_context,
            params=params,
            learned_patterns=learned_patterns,
            cache_context=cache_context,
        )

        # Log component sizes to diagnose prompt bloat
        _log_planning_context_sizes(planning_context, prompt)

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
            temperature=0.3,  # More deterministic - TODO: Make configurable
            context_page_ids=[],  # TODO: Planning runs only on the prompt. Right?
            json_schema=PlanGenerationResponse.model_json_schema(),
        )
        resp_text = response.generated_text if hasattr(response, "generated_text") else str(response)
        logger.warning(
            f"            💬 MPC: agent.infer() returned — response_len={len(resp_text)}"
        )

        # Parse into plan structure using pydantic
        plan_response = self._parse_plan_response(response)

        # Convert goal_hierarchy from pydantic models to dict format expected by ActionPlan
        goal_hierarchy_dict = self._get_goal_hierarchy(plan_response.goal_hierarchy)

        # Convert actions (with key validation)
        valid_keys = self._get_valid_action_keys(planning_context)
        actions = self._convert_actions(plan_response.actions, valid_keys)[: params.planning_horizon]

        return ActionPlan(
            plan_id=f"plan_{int(time.time() * 1000)}",
            agent_id="",  # Set by caller
            goals=planning_context.goals,
            goal_hierarchy=goal_hierarchy_dict,
            actions=actions,
            status=PlanStatus.PROPOSED,
            planning_horizon=params.planning_horizon,
            replan_every_n_steps=params.replan_every_n_steps,
            generation_method="llm",
            initial_reasoning=plan_response.reasoning,
            selected_groups=selected_groups,
        )


class TopDownActionPlanningStrategy(ActionPlanningStrategy):
    """Top-down: Break goals into sub-goals first, then plan actions."""

    def __init__(self, agent: Agent | None = None, prompt_formatting: PromptFormattingStrategy | None = None):
        super().__init__(agent=agent, prompt_formatting=prompt_formatting)

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

        # Phase 1: Scope selection (filters action_descriptions in place if beneficial)
        selected_groups = await self._apply_scope_selection(planning_context)

        # Log context sizes for prompt bloat diagnosis
        decomp_prompt = self.prompt_formatting.build_decomposition_prompt(planning_context)
        _log_planning_context_sizes(planning_context, decomp_prompt)

        # Step 1: Decompose goals into hierarchy (with action awareness)
        decomp_response = await self.agent.infer(
            prompt=decomp_prompt,
            max_tokens=params.max_planning_tokens,
            temperature=0.3,  # More deterministic - TODO: Make configurable
            context_page_ids=[],  # TODO: Planning runs only on the prompt. Right?
            json_schema=GoalHierarchyDecompositionResponse.model_json_schema(),
        )

        # Parse using pydantic
        try:
            text = decomp_response.generated_text
            json_text = JsonExtractor.extract(text)
            decomp_data = GoalHierarchyDecompositionResponse.model_validate_json(json_text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse decomposition response:\nRaw response: {text}\nError: {e}")
            decomp_data = GoalHierarchyDecompositionResponse()

        # Convert to dict format
        goal_hierarchy_dict = self._get_goal_hierarchy(decomp_data.goal_hierarchy)

        # Step 2: Plan actions for leaf goals
        leaf_goals = [
            g for g, data in goal_hierarchy_dict.items() if not data.get("sub_goals")
        ]

        action_planning_prompt = self.prompt_formatting.build_action_planning_prompt(planning_context, leaf_goals, params)
        _log_planning_context_sizes(planning_context, action_planning_prompt)

        action_response = await self.agent.infer(
            prompt=action_planning_prompt,
            max_tokens=params.max_planning_tokens,
            temperature=0.3,  # More deterministic - TODO: Make configurable
            context_page_ids=[],  # TODO: Planning runs only on the prompt. Right?
            json_schema=ReplanningResponse.model_json_schema(),  # Reuse replanning response for actions
        )
        action_data = self._parse_replanning_response(action_response)
        valid_keys = self._get_valid_action_keys(planning_context)
        actions = self._convert_actions(action_data.actions, valid_keys)[: params.planning_horizon]

        return ActionPlan(
            plan_id=f"plan_{int(time.time() * 1000)}",
            agent_id="",
            goals=planning_context.goals,
            goal_hierarchy=goal_hierarchy_dict,
            actions=actions,
            status=PlanStatus.PROPOSED,
            planning_horizon=params.planning_horizon,
            replan_every_n_steps=params.replan_every_n_steps,
            generation_method="llm",
            initial_reasoning=action_data.reasoning,
            selected_groups=selected_groups,
        )


class BottomUpActionPlanningStrategy(ActionPlanningStrategy):
    """Bottom-up: Start with concrete actions, infer goal structure."""

    def __init__(self, agent: Agent | None = None, prompt_formatting: PromptFormattingStrategy | None = None):
        super().__init__(agent=agent, prompt_formatting=prompt_formatting)

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

        # Phase 1: Scope selection (filters action_descriptions in place if beneficial)
        selected_groups = await self._apply_scope_selection(planning_context)

        # Log context sizes for prompt bloat diagnosis
        action_prompt = self.prompt_formatting.build_action_planning_prompt(planning_context, planning_context.goals, params)
        _log_planning_context_sizes(planning_context, action_prompt)

        # Step 1: Generate concrete actions
        action_response = await self.agent.infer(
            prompt=action_prompt,
            max_tokens=params.max_planning_tokens,
            temperature=0.3,  # More deterministic - TODO: Make configurable
            context_page_ids=[],  # TODO: Planning runs only on the prompt. Right?
            json_schema=ReplanningResponse.model_json_schema(),
        )
        action_data = self._parse_replanning_response(action_response)
        valid_keys = self._get_valid_action_keys(planning_context)
        actions = self._convert_actions(action_data.actions, valid_keys)[: params.planning_horizon]

        # Step 2: Infer goal hierarchy from actions
        hierarchy_prompt = self.prompt_formatting.build_goal_hierarchy_inference_prompt(actions)
        _log_planning_context_sizes(planning_context, hierarchy_prompt)

        hier_response = await self.agent.infer(
            prompt=hierarchy_prompt,
            max_tokens=params.max_planning_tokens,
            temperature=0.3,  # More deterministic - TODO: Make configurable
            context_page_ids=[],  # TODO: Planning runs only on the prompt. Right?
            json_schema=GoalHierarchyInferenceResponse.model_json_schema(),
        )

        try:
            text = hier_response.generated_text
            json_text = JsonExtractor.extract(text)
            hier_data = GoalHierarchyInferenceResponse.model_validate_json(json_text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse hierarchy response:\nRaw response: {text}\nError: {e}")
            hier_data = GoalHierarchyInferenceResponse()

        # Convert to dict format
        goal_hierarchy_dict = self._get_goal_hierarchy(hier_data.goal_hierarchy)

        return ActionPlan(
            plan_id=f"plan_{int(time.time() * 1000)}",
            agent_id="",
            goals=planning_context.goals,
            goal_hierarchy=goal_hierarchy_dict,
            actions=actions,
            status=PlanStatus.PROPOSED,
            planning_horizon=params.planning_horizon,
            replan_every_n_steps=params.replan_every_n_steps,
            generation_method="llm",
            initial_reasoning=action_data.reasoning,
            selected_groups=selected_groups,
        )



def _log_planning_context_sizes(ctx: PlanningContext, prompt: str) -> None:
    """Log the size of each PlanningContext component for diagnosing prompt bloat."""
    action_desc_chars = sum(
        len(desc)
        for group in ctx.action_descriptions
        for desc in group.action_descriptions.values()
    )
    action_group_count = len(ctx.action_descriptions)
    action_count = sum(len(g.action_descriptions) for g in ctx.action_descriptions)
    memories_chars = sum(len(json.dumps(m, default=str)) for m in ctx.recalled_memories)
    custom_data_chars = len(json.dumps(ctx.custom_data, default=str)) if ctx.custom_data else 0
    system_prompt_chars = len(ctx.system_prompt)
    exec_ctx_chars = len(ctx.execution_context.model_dump_json()) if ctx.execution_context else 0

    logger.warning(
        f"\n            ┌─ Planning Context Sizes ───────────────────┐\n"
        f"            │  system_prompt:      {system_prompt_chars:>8} chars          │\n"
        f"            │  action_descriptions:{action_desc_chars:>8} chars ({action_group_count} groups, {action_count} actions) │\n"
        f"            │  recalled_memories:  {memories_chars:>8} chars ({len(ctx.recalled_memories)} entries) │\n"
        f"            │  execution_context:  {exec_ctx_chars:>8} chars          │\n"
        f"            │  custom_data:        {custom_data_chars:>8} chars          │\n"
        f"            │  goals:              {len(ctx.goals):>8} entries        │\n"
        f"            │  ─────────────────────────────────────────── │\n"
        f"            │  total prompt:       {len(prompt):>8} chars          │\n"
        f"            └─────────────────────────────────────────────┘"
    )


def _get_prompt_formatting(params: PlanningParameters) -> PromptFormattingStrategy:
    """Resolve prompt formatting strategy from parameters."""
    from ...models import PromptFormattingType

    fmt = params.prompt_formatting
    if fmt == PromptFormattingType.MARKDOWN:
        return MarkdownPromptFormatting()
    elif fmt == PromptFormattingType.ALIAS:
        return AliasPromptFormatting()
    elif fmt == PromptFormattingType.NUMERIC:
        return NumericIDPromptFormatting()
    else:  # XML (default)
        return XMLPromptFormatting()


def get_default_planning_strategy(params: PlanningParameters, agent: Agent | None = None) -> ActionPlanningStrategy:
    """Get default planning strategy based on parameters.

    Args:
        params: Planning parameters
        agent: Optional agent reference (can be set later via set_agent())
    """
    from ...models import PlanningStrategy

    formatting = _get_prompt_formatting(params)

    if params.strategy == PlanningStrategy.MPC:
        return ModelPredictiveActionPlanningStrategy(horizon=params.planning_horizon, agent=agent, prompt_formatting=formatting)
    elif params.strategy == PlanningStrategy.TOP_DOWN:
        return TopDownActionPlanningStrategy(agent=agent, prompt_formatting=formatting)
    elif params.strategy == PlanningStrategy.BOTTOM_UP:
        return BottomUpActionPlanningStrategy(agent=agent, prompt_formatting=formatting)
    else:  # HYBRID
        return ModelPredictiveActionPlanningStrategy(horizon=params.planning_horizon, agent=agent, prompt_formatting=formatting)

