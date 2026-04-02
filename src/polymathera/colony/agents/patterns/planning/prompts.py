
import re
from abc import ABC, abstractmethod
from overrides import override
from typing import Any

from ...models import (
    Action,
    ActionPlan,
    ActionStatus,
    ActionGroupDescription,
    PlanningContext,
    PlanningParameters,
    CacheContext,
)

class JsonExtractor:
    """Utility for robustly extracting JSON objects from LLM responses, handling common formatting variations."""

    @staticmethod
    def extract(text: str) -> str:
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
    def _format_constraints(planning_context: PlanningContext) -> str:
        """Format constraints into prompt text."""
        if not planning_context.constraints:
            return ""
        lines = ["## Constraints"]
        for key, value in planning_context.constraints.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

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
    def _format_action_group_summaries(summaries: list[ActionGroupDescription]) -> str:
        """Format action group summaries as one line per group for the scope selection prompt."""
        lines = []
        for s in summaries:
            tags_str = f", tags: {', '.join(sorted(s.tags))}" if s.tags else ""
            lines.append(f"- [{s.group_key}] ({s.action_count} actions{tags_str}): {s.group_description}")
        return "\n".join(lines)

    def build_scope_selection_prompt(self, planning_context: PlanningContext) -> str:
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

    def build_decomposition_prompt(self, planning_context: PlanningContext) -> str:
        """Build prompt for goal decomposition (step 1 of top-down planning)."""
        action_descriptions = self.format_action_descriptions(planning_context.action_descriptions)
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

    def build_action_planning_prompt(
        self,
        planning_context: PlanningContext,
        leaf_goals: list[str],
        params: PlanningParameters,
    ) -> str:
        """Build prompt for action planning (step 2 of top-down planning)."""
        action_descriptions = self.format_action_descriptions(planning_context.action_descriptions)
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
      {self.format_action_type_instruction()},
      "description": "<what this action does>",
      "parameters": {{}},
      "reasoning": "<why this action is needed>"
    }},
    ...
  ]
}}"""

    def build_planning_prompt(
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
        action_descriptions = self.format_action_descriptions(planning_context.action_descriptions)
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
      {self.format_action_type_instruction()},
      "description": "<what this action does>",
      "parameters": {{}},
      "reasoning": "<why this action is needed>"
    }},
    ...
  ]
}}
"""

    def build_replanning_prompt(
        self,
        planning_context: PlanningContext,
        plan: ActionPlan,
        params: PlanningParameters,
    ) -> str:
        """Build prompt for replanning given current progress."""
        action_descriptions = self.format_action_descriptions(planning_context.action_descriptions)
        custom_data_section = self._format_custom_data(planning_context)
        constraints_section = self._format_constraints(planning_context)
        memories_section = self._format_recalled_memories(planning_context)

        completed_actions = [
            a for a in plan.actions if a.status == ActionStatus.COMPLETED
        ]
        summary = "\n".join(
            f"- {a.action_type}: {a.description} -> {'completed' if a.result and a.result.success else 'failed'}"
            for a in completed_actions
        )

        return f"""{planning_context.system_prompt}

# Replanning

## Original Goals:
{chr(10).join(f"- {g}" for g in plan.goals)}

## Available Actions

{action_descriptions}
{constraints_section}
{memories_section}
{custom_data_section}

## Progress so Far

{summary}
Completed: {len(completed_actions)}/{len(plan.actions)} actions

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
      {self.format_action_type_instruction()},
      "description": "<what this action does>",
      "parameters": {{}},
      "reasoning": "<why this action is needed>"
    }},
    ...
  ]
}}"""

    def build_goal_hierarchy_inference_prompt(self, actions: list[Action]) -> str:
        """Build prompt for inferring goal hierarchy from actions."""

        return f"""Given these actions, infer the goal hierarchy:

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


