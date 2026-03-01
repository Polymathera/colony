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
    ActionGroupDescription,
    ActionStatus,
    ActionType,
    CacheContext,
    ActionPlan,
    PlanExecutionContext,
    PlanningContext,
    PlanningParameters,
    PlanStatus,
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


class ScopeSelectionResponse(BaseModel):
    """Response model for scope selection (Phase 1 of hierarchical action scoping)."""

    reasoning: str = Field(description="Brief reasoning for scope selection")
    selected_groups: list[str] = Field(description="group_key values for relevant action groups")


# Skip scope selection when total groups are at or below this count.
# The overhead of an extra LLM call exceeds the savings at small group counts.
SCOPE_SELECTION_THRESHOLD: int = 6


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
    def _extract_json(text: str) -> str:
        """Extract the JSON object from LLM responses that may contain
        markdown fences, natural language preamble, or trailing text.
        """
        # Try extracting from markdown fences first
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()
        # Fall back to outermost { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return text[start:end + 1]
        return text.strip()

    @staticmethod
    def _format_action_descriptions(planning_context: PlanningContext) -> str:
        """Format action descriptions from planning context into prompt text."""
        sections = []
        for group in planning_context.action_descriptions:
            if group.group_description:
                sections.append(f"\n### {group.group_description}")  # TODO: Format this using XML tags?
            for action_key, action_desc in group.action_descriptions.items():
                sections.append(f"- {action_key}: {action_desc}")  # TODO: Format this using XML tags?
        return "\n".join(sections)

    @staticmethod
    def _format_custom_data(planning_context: PlanningContext) -> str:
        """Format custom_data entries (memory guidance, task guidance, etc.)."""
        # Ensure the memory architecture guidance is available
        if "memory_architecture_guidance" not in planning_context.custom_data:
            raise ValueError("PlanningContext.custom_data must include 'memory_architecture_guidance' key")

        parts = []
        for key, value in planning_context.custom_data.items():
            if value:
                parts.append(str(value))
        return "\n\n".join(parts)

    @staticmethod
    def _format_recalled_memories(planning_context: PlanningContext) -> str:
        """Format recalled memories into prompt text."""
        if not planning_context.recalled_memories:
            return ""
        lines = ["## Recalled Memories"]
        for mem in planning_context.recalled_memories[:10]:
            content = mem.get("content", mem.get("text", str(mem)))
            scope = mem.get("scope", "")
            prefix = f"[{scope}] " if scope else ""
            lines.append(f"- {prefix}{content}")
        return "\n".join(lines)

    @staticmethod
    def _format_constraints(planning_context: PlanningContext) -> str:
        """Format constraints into prompt text."""
        if not planning_context.constraints:
            return ""
        lines = ["## Constraints"]
        for key, value in planning_context.constraints.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    # ── Scope selection (Phase 1 of hierarchical action scoping) ──────────────

    @staticmethod
    def _format_action_group_summaries(summaries: list[ActionGroupDescription]) -> str:
        """Format action group summaries as one line per group for the scope selection prompt."""
        lines = []
        for s in summaries:
            tags_str = f", tags: {', '.join(sorted(s.tags))}" if s.tags else ""
            lines.append(f"- [{s.group_key}] ({s.action_count} actions{tags_str}): {s.group_description}")
        return "\n".join(lines)

    @staticmethod
    def _should_use_scope_selection(planning_context: PlanningContext) -> bool:
        """Return True if scope selection is beneficial (enough groups to justify the extra call)."""
        return len(planning_context.action_group_summaries) > SCOPE_SELECTION_THRESHOLD

    def _build_scope_selection_prompt(self, planning_context: PlanningContext) -> str:
        """Build a lightweight prompt for scope selection (Phase 1).

        Includes only: system identity, goals, minimal execution context, and group summaries.
        Omits: full action descriptions, memory guidance, recalled memories — those go in Phase 2.
        """
        goals_str = "\n".join(f"- {g}" for g in planning_context.goals)
        summaries_str = self._format_action_group_summaries(planning_context.action_group_summaries)

        # Minimal execution context — just enough for the LLM to know what phase we're in
        exec_hint = ""
        if planning_context.execution_context:
            ctx = planning_context.execution_context
            completed = len(ctx.completed_action_ids)
            if completed > 0:
                exec_hint = f"\n\nExecution progress: {completed} actions completed."

        return f"""You are selecting which action groups are relevant for the current planning step.

## Goals

{goals_str}
{exec_hint}

## Available Action Groups

{summaries_str}

## Instructions

Select the action groups that are relevant to the current goals. Include groups whose actions
might be needed for planning. When in doubt, include the group — it is better to include an
extra group than to miss a needed one.

Respond with ONLY a JSON object (no markdown fences, no surrounding text):

{{
  "reasoning": "<brief reasoning for your selection>",
  "selected_groups": ["<group_key_1>", "<group_key_2>", ...]
}}"""

    async def _run_scope_selection(self, planning_context: PlanningContext) -> list[str]:
        """Run scope selection (Phase 1) and return selected group keys.

        On parse failure, falls back to all groups (no filtering).
        """
        if not self.agent:
            raise RuntimeError("Agent reference not set for scope selection")

        prompt = self._build_scope_selection_prompt(planning_context)
        all_keys = [s.group_key for s in planning_context.action_group_summaries]

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
            json_text = self._extract_json(text)
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

    def _build_decomposition_prompt(self, planning_context: PlanningContext) -> str:
        """Build prompt for goal decomposition (step 1 of top-down planning)."""
        action_descriptions = self._format_action_descriptions(planning_context)
        custom_data_section = self._format_custom_data(planning_context)
        constraints_section = self._format_constraints(planning_context)
        memories_section = self._format_recalled_memories(planning_context)

        return f"""{planning_context.system_prompt}

## Goals

{', '.join(planning_context.goals)}

## Available Actions

{action_descriptions}
{constraints_section}
{memories_section}
{custom_data_section}

Break down the goals into a hierarchical structure.
Each leaf goal should map to one or more of the available actions.

Respond with ONLY a JSON object (no markdown fences, no surrounding text) in this exact format:

{{
  "goal_hierarchy": {{
    "<goal_id>": {{
      "goal": "<description of this goal>",
      "sub_goals": ["<child_goal_id>", ...],
      "parent": "<parent_goal_id or null>"
    }},
    ...
  }}
}}

Rules:
- "goal_hierarchy" is a flat dict keyed by goal ID (not a nested tree).
- Every node must have a "goal" field (string), a "sub_goals" field (list of goal IDs), and a "parent" field (string or null).
- Leaf nodes have empty "sub_goals" lists.
- The root node has "parent": null."""

    def _build_action_planning_prompt(
        self,
        planning_context: PlanningContext,
        leaf_goals: list[str],
        params: PlanningParameters,
    ) -> str:
        """Build prompt for action planning (step 2 of top-down planning)."""
        action_descriptions = self._format_action_descriptions(planning_context)
        custom_data_section = self._format_custom_data(planning_context)
        constraints_section = self._format_constraints(planning_context)
        memories_section = self._format_recalled_memories(planning_context)

        return f"""{planning_context.system_prompt}

## Leaf Goals

{leaf_goals}

## Available Actions

{action_descriptions}
{constraints_section}
{memories_section}
{custom_data_section}

Create actions for each leaf goal. Each action must use one of the available action types listed above.
Plan the next {params.planning_horizon} actions.

Respond with ONLY a JSON object (no markdown fences, no surrounding text) in this exact format:

{{
  "reasoning": "<step-by-step reasoning>",
  "actions": [
    {{
      "action_type": "<one of the available action types above>",
      "description": "<what this action does>",
      "parameters": {{}},
      "reasoning": "<why this action is needed>"
    }},
    ...
  ]
}}"""

    def _build_replan_prompt(
        self,
        planning_context: PlanningContext,
        plan: Any,
        summary: str,
        params: Any,
    ) -> str:
        """Build prompt for replanning given current progress."""
        action_descriptions = self._format_action_descriptions(planning_context)
        custom_data_section = self._format_custom_data(planning_context)
        constraints_section = self._format_constraints(planning_context)
        memories_section = self._format_recalled_memories(planning_context)

        return f"""{planning_context.system_prompt}

## Goals

{', '.join(plan.goals)}

## Available Actions

{action_descriptions}
{constraints_section}
{memories_section}
{custom_data_section}

## Progress

{summary}
Completed: {len([a for a in plan.actions if a.status.value == 'completed'])}/{len(plan.actions)} actions

Plan the next {params.planning_horizon} actions.

Respond with ONLY a JSON object (no markdown fences, no surrounding text) in this exact format:

{{
  "reasoning": "<why these actions are needed given current progress>",
  "actions": [
    {{
      "action_type": "<one of the available action types above>",
      "description": "<what this action does>",
      "parameters": {{}},
      "reasoning": "<why this action is needed>"
    }},
    ...
  ]
}}"""

    def _parse_plan_response(self, response: Any) -> PlanGenerationResponse:
        """Parse LLM response into PlanGenerationResponse using pydantic."""
        text = (
            response.generated_text if hasattr(response, "generated_text") else str(response)
        )
        # Extract JSON from response
        text = self._extract_json(text)
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
        text = self._extract_json(text)
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

    def _convert_actions(self, action_specs: list[ActionSpec]) -> list[Action]:
        """Convert ActionSpec objects to Action objects.

        action_type may be an ActionType enum value (e.g. "tool_use") or a
        capability action key (e.g. "AgentContextEngine.8dd64d54.inspect_memory_map").
        Action.action_type is typed ActionType | str, and the dispatcher uses
        str(action.action_type) for executor lookup, so both forms work.
        """
        actions = []
        for i, spec in enumerate(action_specs):
            # Prefer ActionType enum when it matches; otherwise keep the raw
            # action key string — the dispatcher resolves it against executors.
            try:
                action_type: ActionType | str = ActionType(spec.action_type)
            except ValueError:
                action_type = spec.action_type

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
        return actions

    def _build_planning_prompt(
        self,
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
        goals_str = "\n".join(f"- {g}" for g in planning_context.goals)
        action_descriptions = self._format_action_descriptions(planning_context)
        custom_data_section = self._format_custom_data(planning_context)
        constraints_section = self._format_constraints(planning_context)
        memories_section = self._format_recalled_memories(planning_context)

        # Learned patterns from similar successful plans
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

        # Execution context — render all available state so the LLM can make informed decisions
        exec_context_section = ""
        if planning_context.execution_context:
            ctx = planning_context.execution_context
            parts = ["\n\n## Current Execution Context\n"]
            parts.append(f"- **Completed Actions**: {len(ctx.completed_action_ids)}")
            if ctx.analyzed_pages:
                parts.append(f"- **Pages Analyzed**: {len(ctx.analyzed_pages)}")
            if ctx.spawned_children:
                parts.append(f"- **Spawned Children**: {len(ctx.spawned_children)}")
            if ctx.findings:
                parts.append(f"- **Findings**: {len(ctx.findings)} entries")
                for key, value in list(ctx.findings.items())[:5]:
                    preview = str(value)[:120]
                    parts.append(f"  - {key}: {preview}")
            if ctx.synthesis_results:
                parts.append(f"- **Synthesis Results**: {len(ctx.synthesis_results)} entries")
            # Show recent action results (last 3) so planner sees what just happened
            if ctx.action_results:
                recent = list(ctx.action_results.items())[-3:]
                parts.append("- **Recent Action Results**:")
                for action_id, result in recent:
                    status = result.status if hasattr(result, 'status') else 'unknown'
                    parts.append(f"  - {action_id}: {status}")
            exec_context_section = "\n".join(parts) + "\n"

        return f"""{planning_context.system_prompt}

## Goals

{goals_str}
{patterns_section}
{cache_section}
{exec_context_section}
{constraints_section}
{memories_section}

## Available Actions

{action_descriptions}

{custom_data_section}

## Your Task

1. **Decompose goals** into a hierarchical structure (main goals -> sub-goals)
2. **Plan concrete actions** for the next {params.planning_horizon} steps using the available actions above
3. **Consider learned patterns** from similar successful plans above
4. **Optimize for cache efficiency** by grouping actions on spatially related pages
5. **Explain your reasoning** step-by-step

## Output Format

Respond with ONLY a JSON object (no markdown fences, no surrounding text) in this exact format:

{{
  "reasoning": "<step-by-step reasoning>",
  "goal_hierarchy": {{
    "<goal_id>": {{
      "goal": "<description>",
      "sub_goals": ["<child_id>", ...],
      "parent": "<parent_id or null>"
    }},
    ...
  }},
  "actions": [
    {{
      "action_type": "<one of the available action types above>",
      "description": "<what this action does>",
      "parameters": {{}},
      "reasoning": "<why this action is needed>"
    }},
    ...
  ]
}}
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

        # Phase 1: Scope selection (filters action_descriptions in place if beneficial)
        selected_groups = await self._apply_scope_selection(planning_context)

        # Phase 2: Build prompt with (possibly filtered) action descriptions
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
            status=PlanStatus.PROPOSED,
            planning_horizon=params.planning_horizon,
            replan_every_n_steps=params.replan_every_n_steps,
            generation_method="llm",
            initial_reasoning=plan_response.reasoning,
            selected_groups=selected_groups,
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

        # Reuse cached scope selection from initial plan
        self._apply_cached_scope_filter(planning_context, plan)

        # Build prompt with execution history
        prompt = self._build_replanning_prompt(plan, planning_context, params)

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
        new_actions = self._convert_actions(replan_response.actions)[: params.planning_horizon]
        return new_actions

    def _build_replanning_prompt(self, plan: ActionPlan, planning_context: PlanningContext, params: PlanningParameters) -> str:
        """Build replanning prompt with detailed context."""
        action_descriptions = self._format_action_descriptions(planning_context)
        custom_data_section = self._format_custom_data(planning_context)
        constraints_section = self._format_constraints(planning_context)
        memories_section = self._format_recalled_memories(planning_context)

        completed_actions = [
            a for a in plan.actions if a.status == ActionStatus.COMPLETED
        ]

        summary = "\n".join(
            [
                f"- {a.action_type}: {a.description} -> {'completed' if a.result and a.result.success else 'failed'}"
                for a in completed_actions
            ]
        )

        return f"""{planning_context.system_prompt}

## Replanning

Original Goals:
{chr(10).join(f"- {g}" for g in plan.goals)}

Available actions:
{action_descriptions}
{constraints_section}
{memories_section}
{custom_data_section}

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

Respond with ONLY a JSON object (no markdown fences, no surrounding text) in this exact format:

{{
  "reasoning": "<why these actions are needed given current progress>",
  "actions": [
    {{
      "action_type": "<one of the available action types above>",
      "description": "<what this action does>",
      "parameters": {{}},
      "reasoning": "<why this action is needed>"
    }},
    ...
  ]
}}"""


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

        # Phase 1: Scope selection (filters action_descriptions in place if beneficial)
        selected_groups = await self._apply_scope_selection(planning_context)

        # Step 1: Decompose goals into hierarchy (with action awareness)
        decomp_response = await self.agent.infer(
            prompt=self._build_decomposition_prompt(planning_context),
            max_tokens=params.max_planning_tokens,
            temperature=0.3,  # More deterministic - TODO: Make configurable
            context_page_ids=[],  # TODO: Planning runs only on the prompt. Right?
            json_schema=GoalHierarchyDecompositionResponse.model_json_schema(),
        )

        # Parse using pydantic
        try:
            text = decomp_response.generated_text
            json_text = self._extract_json(text)
            decomp_data = GoalHierarchyDecompositionResponse.model_validate_json(json_text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse decomposition response:\nRaw response: {text}\nError: {e}")
            decomp_data = GoalHierarchyDecompositionResponse()

        # Convert to dict format
        goal_hierarchy = {}
        for goal_id, goal_node in decomp_data.goal_hierarchy.items():
            goal_hierarchy[goal_id] = {
                "goal": goal_node.goal,
                "sub_goals": goal_node.sub_goals,
                "parent": goal_node.parent,
            }

        # Step 2: Plan actions for leaf goals
        leaf_goals = [
            g for g, data in goal_hierarchy.items() if not data.get("sub_goals")
        ]

        action_response = await self.agent.infer(
            prompt=self._build_action_planning_prompt(planning_context, leaf_goals, params),
            max_tokens=params.max_planning_tokens,
            temperature=0.3,  # More deterministic - TODO: Make configurable
            context_page_ids=[],  # TODO: Planning runs only on the prompt. Right?
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
            status=PlanStatus.PROPOSED,
            planning_horizon=params.planning_horizon,
            replan_every_n_steps=params.replan_every_n_steps,
            generation_method="llm",
            initial_reasoning=action_data.reasoning,
            selected_groups=selected_groups,
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

        # Reuse cached scope selection from initial plan
        self._apply_cached_scope_filter(planning_context, plan)

        completed_actions = [
            a for a in plan.actions if a.status == ActionStatus.COMPLETED
        ]
        summary = "\n".join(
            f"- {a.action_type}: {a.description} -> {'completed' if a.result and a.result.success else 'failed'}"
            for a in completed_actions
        )

        response = await self.agent.infer(
            prompt=self._build_replan_prompt(planning_context, plan, summary, params),
            max_tokens=params.max_planning_tokens,
            temperature=0.3,  # More deterministic - TODO: Make configurable
            context_page_ids=[],  # TODO: Planning runs only on the prompt. Right?
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

        # Phase 1: Scope selection (filters action_descriptions in place if beneficial)
        selected_groups = await self._apply_scope_selection(planning_context)

        # Step 1: Generate concrete actions
        action_response = await self.agent.infer(
            prompt=self._build_action_planning_prompt(planning_context, planning_context.goals, params),
            max_tokens=params.max_planning_tokens,
            temperature=0.3,  # More deterministic - TODO: Make configurable
            context_page_ids=[],  # TODO: Planning runs only on the prompt. Right?
            json_schema=ReplanningResponse.model_json_schema(),
        )
        action_data = self._parse_replanning_response(action_response)
        actions = self._convert_actions(action_data.actions)[: params.planning_horizon]

        # Step 2: Infer goal hierarchy from actions
        hierarchy_prompt = f"""Given these actions, infer the goal hierarchy:

Actions: {[a.description for a in actions]}

Respond with ONLY a JSON object (no markdown fences, no surrounding text) in this exact format:

{{
  "goal_hierarchy": {{
    "<goal_id>": {{
      "goal": "<description of this goal>",
      "sub_goals": ["<child_goal_id>", ...],
      "parent": "<parent_goal_id or null>"
    }},
    ...
  }}
}}"""

        hier_response = await self.agent.infer(
            prompt=hierarchy_prompt,
            max_tokens=params.max_planning_tokens,
            temperature=0.3,  # More deterministic - TODO: Make configurable
            context_page_ids=[],  # TODO: Planning runs only on the prompt. Right?
            json_schema=GoalHierarchyInferenceResponse.model_json_schema(),
        )

        try:
            text = hier_response.generated_text
            json_text = self._extract_json(text)
            hier_data = GoalHierarchyInferenceResponse.model_validate_json(json_text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse hierarchy response:\nRaw response: {text}\nError: {e}")
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
            status=PlanStatus.PROPOSED,
            planning_horizon=params.planning_horizon,
            replan_every_n_steps=params.replan_every_n_steps,
            generation_method="llm",
            initial_reasoning=action_data.reasoning,
            selected_groups=selected_groups,
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

        # Reuse cached scope selection from initial plan
        self._apply_cached_scope_filter(planning_context, plan)

        completed_actions = [
            a for a in plan.actions if a.status == ActionStatus.COMPLETED
        ]
        summary = "\n".join(
            f"- {a.action_type}: {a.description} -> {'completed' if a.result and a.result.success else 'failed'}"
            for a in completed_actions
        )

        response = await self.agent.infer(
            prompt=self._build_replan_prompt(planning_context, plan, summary, params),
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

