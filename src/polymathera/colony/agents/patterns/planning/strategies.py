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


class PromptFormattingStrategy(ABC):
    """Strategy for formatting action descriptions in the planner prompt.

    Different formatting strategies may affect LLM accuracy on action key
    selection. This abstraction allows experimentation and evaluation.
    """

    @abstractmethod
    def format_action_descriptions(self, action_descriptions: list[ActionGroupDescription]) -> str:
        """Format action group descriptions into prompt text."""
        ...

    @abstractmethod
    def format_action_type_instruction(self) -> str:
        """Return the instruction text for the action_type field in the JSON output format."""
        ...

    def resolve_action_key(self, raw_key: str) -> str | None:
        """Resolve a key emitted by the LLM back to the full action key.

        Override in strategies that use aliases or numeric IDs. Returns None
        if this strategy has no special resolution (fall through to default
        suffix/prefix matching in _resolve_action_key).
        """
        return None


class MarkdownPromptFormatting(PromptFormattingStrategy):
    """Original markdown list formatting (legacy).

    Action keys are presented as markdown list items:
        ### Group description
        - ClassName.hash.action_name: Action description
    """

    @override
    def format_action_descriptions(self, action_descriptions: list[ActionGroupDescription]) -> str:
        sections = []
        for group in action_descriptions:
            if group.group_description:
                sections.append(f"\n### {group.group_description}")
            for action_key, action_desc in group.action_descriptions.items():
                sections.append(f"- {action_key}: {action_desc}")
        return "\n".join(sections)

    @override
    def format_action_type_instruction(self) -> str:
        return '"action_type": "<one of the available action types above>"'


class XMLPromptFormatting(PromptFormattingStrategy):
    """XML-structured formatting for unambiguous action key delimitation.

    Action keys are presented as XML attributes, making the full key
    structurally unambiguous:
        <action-group key="ClassName.hash">
          <description>Group description</description>
          <action key="ClassName.hash.action_name">Action description</action>
        </action-group>

    Includes explicit copy instructions (strategy A2) to reduce truncation.
    """

    @override
    def format_action_descriptions(self, action_descriptions: list[ActionGroupDescription]) -> str:
        sections = []
        for group in action_descriptions:
            sections.append(f'<action-group key="{group.group_key}">')
            if group.group_description:
                sections.append(f"  <description>{group.group_description}</description>")
            for action_key, action_desc in group.action_descriptions.items():
                sections.append(f'  <action key="{action_key}">{action_desc}</action>')
            sections.append("</action-group>")
        sections.append("")
        sections.append(
            "CRITICAL: The action_type value MUST be copied EXACTLY from the key= attribute "
            "of an <action> element above. Include ALL parts: ClassName.hash.method_name. "
            "Do NOT abbreviate, truncate, or modify the key in any way."
        )
        return "\n".join(sections)

    @override
    def format_action_type_instruction(self) -> str:
        return '"action_type": "<the exact key= attribute from an <action> element above>"'


class AliasPromptFormatting(PromptFormattingStrategy):
    """Short alias formatting — LLM outputs human-readable action aliases, framework resolves.

    Generates short, unique aliases from the method name suffix of each action key.
    The LLM outputs the alias; the framework resolves it to the full key.

    Example prompt output:
        <action-group name="consciousness">
          <action alias="update_self_concept" key="ConsciousnessCapability.b79b5858.consciousness_update_self_concept">
            Update the agent's self-concept...
          </action>
        </action-group>

    LLM outputs: "action_type": "update_self_concept"
    Framework resolves to: "ConsciousnessCapability.b79b5858.consciousness_update_self_concept"
    """

    def __init__(self):
        self._alias_to_key: dict[str, str] = {}

    @staticmethod
    def _generate_alias(full_key: str, all_keys: list[str]) -> str:
        """Extract shortest unambiguous suffix as alias."""
        parts = full_key.split(".")
        if len(parts) < 2:
            return full_key
        suffix = parts[-1]
        # Check if suffix is unique across all keys
        if sum(1 for k in all_keys if k.split(".")[-1] == suffix) == 1:
            return suffix
        # Prefix with short group name for disambiguation
        group = parts[0].replace("Capability", "").lower()
        return f"{group}.{suffix}"

    @override
    def format_action_descriptions(self, action_descriptions: list[ActionGroupDescription]) -> str:
        self._alias_to_key.clear()
        all_keys = [
            key
            for group in action_descriptions
            for key in group.action_descriptions
        ]

        # Pre-compute aliases and detect collisions
        key_to_alias: dict[str, str] = {}
        alias_counts: dict[str, int] = {}
        for full_key in all_keys:
            alias = self._generate_alias(full_key, all_keys)
            alias_counts[alias] = alias_counts.get(alias, 0) + 1
            key_to_alias[full_key] = alias

        # Resolve collisions: if alias maps to multiple keys, use the full key
        for full_key, alias in key_to_alias.items():
            if alias_counts[alias] > 1:
                key_to_alias[full_key] = full_key

        sections = []
        for group in action_descriptions:
            group_name = group.group_key.split(".")[0].replace("Capability", "").lower() if group.group_key else "unknown"
            sections.append(f'<action-group name="{group_name}">')
            if group.group_description:
                sections.append(f"  <description>{group.group_description}</description>")
            for action_key, action_desc in group.action_descriptions.items():
                alias = key_to_alias[action_key]
                self._alias_to_key[alias] = action_key
                sections.append(
                    f'  <action alias="{alias}" key="{action_key}">{action_desc}</action>'
                )
            sections.append("</action-group>")
        sections.append("")
        sections.append(
            'Use the alias= value for action_type (e.g. "update_self_concept"), '
            "NOT the full key. The framework resolves aliases automatically."
        )
        return "\n".join(sections)

    @override
    def format_action_type_instruction(self) -> str:
        return '"action_type": "<the alias= value from an <action> element above>"'

    @override
    def resolve_action_key(self, raw_key: str) -> str | None:
        return self._alias_to_key.get(raw_key)


class NumericIDPromptFormatting(PromptFormattingStrategy):
    """Numeric ID formatting — LLM outputs an integer, framework resolves to full key.

    Assigns each action a sequential integer. The LLM outputs just the number,
    eliminating the string-copying problem entirely.

    Example prompt output:
        ## Available Actions
        [1] Update self-concept (ConsciousnessCapability) — Update the agent's self-concept...
        [2] Get self-concept (ConsciousnessCapability) — Get the agent's current self-concept...

    LLM outputs: "action_type": 1
    Framework resolves to: "ConsciousnessCapability.b79b5858.consciousness_update_self_concept"
    """

    def __init__(self):
        self._id_to_key: dict[str, str] = {}

    @override
    def format_action_descriptions(self, action_descriptions: list[ActionGroupDescription]) -> str:
        self._id_to_key.clear()
        sections = ["## Available Actions\n"]
        idx = 1
        for group in action_descriptions:
            group_name = group.group_key.split(".")[0] if group.group_key else "Unknown"
            if group.group_description:
                sections.append(f"### {group.group_description}")
            for action_key, action_desc in group.action_descriptions.items():
                self._id_to_key[str(idx)] = action_key
                # Show short method name + group for readability
                method = action_key.split(".")[-1] if "." in action_key else action_key
                sections.append(f"[{idx}] {method} ({group_name}) — {action_desc}")
                idx += 1
            sections.append("")
        sections.append(
            "Use the number in brackets (e.g. 1, 2, 3) as the action_type value. "
            "Do NOT use the action name or full key."
        )
        return "\n".join(sections)

    @override
    def format_action_type_instruction(self) -> str:
        return '"action_type": <number from the brackets [N] above>'

    @override
    def resolve_action_key(self, raw_key: str) -> str | None:
        # Handle both "1" and 1 (LLM may emit either)
        key = str(raw_key).strip().strip('"')
        return self._id_to_key.get(key)


class PlanningStrategyPolicy(ABC):
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

    def _format_action_descriptions(self, planning_context: PlanningContext) -> str:
        """Format action descriptions from planning context into prompt text."""
        return self.prompt_formatting.format_action_descriptions(planning_context.action_descriptions)

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
        """Format recalled memories into prompt text.

        Handles different memory entry types stored by producer hooks:
        - Action entries (from ActionDispatcher.dispatch hook): value is an Action object
        - Plan entries (from plan creation/replan hooks): value is an ActionPlan object
        - Generic entries: falls back to string representation
        """
        if not planning_context.recalled_memories:
            return ""
        lines = ["## Recalled Memories"]
        for mem in planning_context.recalled_memories[:10]:  # TODO: make this limit configurable
            tags = set(mem.get("tags", []))
            value = mem.get("value")

            # Action entries stored by extract_action_from_dispatch
            if "action" in tags and hasattr(value, "action_type"):
                status = value.status.value if hasattr(value.status, "value") else str(value.status)
                desc = value.description if hasattr(value, "description") else ""
                result_info = ""
                if value.result:
                    result_info = " (success)" if value.result.success else " (failed)"
                lines.append(f"- [action] {value.action_type}: {desc}{result_info} [{status}]")
                continue

            # Plan entries stored by extract_plan_from_policy
            if "plan" in tags and hasattr(value, "actions"):
                n_actions = len(value.actions) if value.actions else 0
                plan_status = value.status.value if hasattr(value.status, "value") else str(value.status)
                lines.append(f"- [plan] {n_actions} actions, status={plan_status}")
                continue

            # Generic entries — try content/text keys, then str(value)
            if isinstance(value, str):
                lines.append(f"- {value}")
            else:
                content = str(value)[:200] if value is not None else "(empty)"
                lines.append(f"- {content}")
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

        Includes: system prompt (agent identity, role, goals, constraints), minimal
        execution context, and group summaries.
        Omits: full action descriptions, recalled memories, custom data — those go in Phase 2.
        """
        summaries_str = self._format_action_group_summaries(planning_context.action_group_summaries)

        # Minimal execution context — just enough for the LLM to know what phase we're in
        exec_hint = ""
        if planning_context.execution_context:
            ctx = planning_context.execution_context
            completed = len(ctx.completed_action_ids)
            if completed > 0:
                exec_hint = f"\n\nExecution progress: {completed} actions completed."

        return f"""{planning_context.system_prompt}

## Task: Select Relevant Action Groups
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
      {self.prompt_formatting.format_action_type_instruction()},
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
      {self.prompt_formatting.format_action_type_instruction()},
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
      {self.prompt_formatting.format_action_type_instruction()},
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

    def __init__(
        self,
        horizon: int = 5,
        agent: Agent | None = None,
        prompt_formatting: PromptFormattingStrategy | None = None,
    ):
        super().__init__(agent=agent, prompt_formatting=prompt_formatting)
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
        valid_keys = self._get_valid_action_keys(planning_context)
        new_actions = self._convert_actions(replan_response.actions, valid_keys)[: params.planning_horizon]
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
      {self.prompt_formatting.format_action_type_instruction()},
      "description": "<what this action does>",
      "parameters": {{}},
      "reasoning": "<why this action is needed>"
    }},
    ...
  ]
}}"""


class TopDownPlanningStrategy(PlanningStrategyPolicy):
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
        valid_keys = self._get_valid_action_keys(planning_context)
        actions = self._convert_actions(action_data.actions, valid_keys)[: params.planning_horizon]

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
        valid_keys = self._get_valid_action_keys(planning_context)
        return self._convert_actions(replan_data.actions, valid_keys)[: params.planning_horizon]


class BottomUpPlanningStrategy(PlanningStrategyPolicy):
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

        # Step 1: Generate concrete actions
        action_response = await self.agent.infer(
            prompt=self._build_action_planning_prompt(planning_context, planning_context.goals, params),
            max_tokens=params.max_planning_tokens,
            temperature=0.3,  # More deterministic - TODO: Make configurable
            context_page_ids=[],  # TODO: Planning runs only on the prompt. Right?
            json_schema=ReplanningResponse.model_json_schema(),
        )
        action_data = self._parse_replanning_response(action_response)
        valid_keys = self._get_valid_action_keys(planning_context)
        actions = self._convert_actions(action_data.actions, valid_keys)[: params.planning_horizon]

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
        valid_keys = self._get_valid_action_keys(planning_context)
        return self._convert_actions(replan_data.actions, valid_keys)[: params.planning_horizon]


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


def get_default_planning_strategy(params: PlanningParameters, agent: Agent | None = None) -> PlanningStrategyPolicy:
    """Get default planning strategy based on parameters.

    Args:
        params: Planning parameters
        agent: Optional agent reference (can be set later via set_agent())
    """
    from ...models import PlanningStrategy

    formatting = _get_prompt_formatting(params)

    if params.strategy == PlanningStrategy.MPC:
        return ModelPredictiveControlStrategy(horizon=params.planning_horizon, agent=agent, prompt_formatting=formatting)
    elif params.strategy == PlanningStrategy.TOP_DOWN:
        return TopDownPlanningStrategy(agent=agent, prompt_formatting=formatting)
    elif params.strategy == PlanningStrategy.BOTTOM_UP:
        return BottomUpPlanningStrategy(agent=agent, prompt_formatting=formatting)
    else:  # HYBRID
        return ModelPredictiveControlStrategy(horizon=params.planning_horizon, agent=agent, prompt_formatting=formatting)

