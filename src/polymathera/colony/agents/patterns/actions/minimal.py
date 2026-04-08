"""Minimal action policy — the simplest possible action selection.

``MinimalActionPolicy`` gathers available ``@action_executor`` methods from
the agent's capabilities, shows them to the LLM, and dispatches whatever
the LLM selects. No planning infrastructure, no event processing, no
pre-programmed enrichment sequence.

If planning capabilities (``CacheAnalysisCapability``,
``PlanLearningCapability``, etc.) are registered on the agent, their
``@action_executor`` methods appear in the action list. The LLM decides
whether to use them — they are not forced.

This is the **baseline** for evaluating the value that structure adds.
It occupies the bottom-left of the action policy space:

- **Execution mode**: JSON action selection (not code generation)
- **Structure/guidance**: None (LLM decides everything)

Usage::

    from polymathera.colony.agents.patterns.actions.minimal import (
        MinimalActionPolicy,
        create_minimal_action_policy,
    )

    policy = await create_minimal_action_policy(agent)
    # Agent's run loop calls execute_iteration() which calls plan_step()
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from overrides import override

from ...base import Agent
from ...models import (
    Action,
    ActionType,
    ActionPolicyExecutionState,
)
from .policies import BaseActionPolicy

logger = logging.getLogger(__name__)


class MinimalActionPolicy(BaseActionPolicy):
    """Bare-bones action policy. LLM picks from available actions.

    No planning infrastructure, no event processing, no cache analysis,
    no learned patterns, no coordination, no replanning. The LLM sees
    all available ``@action_executor`` methods and selects which one to
    call next based on its own judgment.

    If planning capabilities are registered on the agent (by the user or
    by a planner that ran previously), their ``@action_executor`` methods
    appear in the action list and the LLM can use them — but is not
    required to.

    Args:
        agent: The owning agent.
        max_iterations: Maximum planning iterations before signaling completion.
        temperature: LLM sampling temperature (lower = more deterministic).
        max_tokens: Maximum tokens for LLM response.

    Example::

        policy = MinimalActionPolicy(agent=agent, max_iterations=20)
        await policy.initialize()
    """

    def __init__(
        self,
        agent: Agent,
        max_iterations: int = 50,
        temperature: float = 0.3,
        max_tokens: int = 512,
        **kwargs,
    ):
        super().__init__(agent=agent, **kwargs)
        self._max_iterations = max_iterations
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._iteration_count = 0
        self._last_result_summary: str | None = None

    @override
    async def plan_step(
        self, state: ActionPolicyExecutionState
    ) -> Action | None:
        """Select the next action by asking the LLM.

        1. Gathers action descriptions from all registered capabilities.
        2. Builds a minimal prompt with goals, last result, and actions.
        3. Calls the LLM with constrained JSON output.
        4. Parses the response and returns the selected Action.

        Returns None to signal completion (``state.custom["policy_complete"]``)
        or idle (``state.custom["idle"]``).
        """
        self._iteration_count += 1

        if self._iteration_count > self._max_iterations:
            logger.info(f"MinimalActionPolicy: max iterations ({self._max_iterations}) reached")
            state.custom["policy_complete"] = True
            return None

        # 1. Gather action descriptions
        action_descriptions = await self.get_action_descriptions()

        if not action_descriptions:
            logger.warning("MinimalActionPolicy: no actions available")
            state.custom["idle"] = True
            return None

        # 2. Build prompt
        prompt = self._build_prompt(state, action_descriptions)

        # 3. Call LLM
        try:
            response = await self.agent.infer(
                prompt=prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        except Exception as e:
            logger.error(f"MinimalActionPolicy: LLM inference failed: {e}")
            return None

        # 4. Parse response
        action = self._parse_response(response, action_descriptions)

        if action is None:
            # LLM signaled completion or produced unparseable output
            if self._iteration_count > 1:
                state.custom["policy_complete"] = True
            return None

        # Track last result for next iteration's context
        return action

    def _build_prompt(
        self,
        state: ActionPolicyExecutionState,
        action_descriptions: list,
    ) -> str:
        """Build the minimal planning prompt.

        Includes: agent identity, goals, last action result, and available actions.
        No memories, no cache context, no learned patterns — just the essentials.
        """
        goals = self.agent.metadata.goals or ["Complete the assigned task"]
        goals_str = "\n".join(f"- {g}" for g in goals)

        # Format action descriptions
        actions_str = ""
        all_action_keys = []
        for group in action_descriptions:
            group_key = group.group_key if hasattr(group, 'group_key') else str(group)
            group_desc = group.group_description if hasattr(group, 'group_description') else ""
            actions_str += f"\n## {group_key}\n{group_desc}\n"

            action_descs = group.action_descriptions if hasattr(group, 'action_descriptions') else {}
            for action_key, desc in action_descs.items():
                actions_str += f"  - **{action_key}**: {desc}\n"
                all_action_keys.append(action_key)

        # Last result context
        last_result_str = ""
        if self._last_result_summary:
            last_result_str = f"\n## Last Action Result\n{self._last_result_summary}\n"

        prompt = f"""You are an autonomous agent.

## Goals
{goals_str}
{last_result_str}
## Available Actions
{actions_str}

## Iteration {self._iteration_count} of {self._max_iterations}

Select the next action to take. If all goals are achieved, respond with
{{"action_type": "DONE", "parameters": {{}}}}.

Respond with ONLY a JSON object:
{{
  "reasoning": "<why this action>",
  "action_type": "<exact action key from above>",
  "parameters": {{<parameters for the action>}}
}}
"""
        return prompt

    def _parse_response(
        self,
        response: Any,
        action_descriptions: list,
    ) -> Action | None:
        """Parse LLM response into an Action."""
        # Extract text
        if hasattr(response, 'generated_text'):
            text = response.generated_text
        elif hasattr(response, 'text'):
            text = response.text
        elif isinstance(response, str):
            text = response
        else:
            text = str(response)

        # Extract JSON
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON in the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(text[start:end])
                except json.JSONDecodeError:
                    logger.warning("MinimalActionPolicy: could not parse LLM response as JSON")
                    return None
            else:
                logger.warning("MinimalActionPolicy: no JSON found in LLM response")
                return None

        action_type = data.get("action_type", "")
        parameters = data.get("parameters", {})

        # Check for completion signal
        if action_type.upper() == "DONE":
            return None

        # Resolve action key (fuzzy matching)
        all_keys = []
        for group in action_descriptions:
            action_descs = group.action_descriptions if hasattr(group, 'action_descriptions') else {}
            all_keys.extend(action_descs.keys())

        resolved_key = self._resolve_action_key(action_type, all_keys)

        if not resolved_key:
            logger.warning(f"MinimalActionPolicy: unknown action '{action_type}'")
            return None

        return Action(
            action_id=f"minimal_{uuid.uuid4().hex[:8]}",
            agent_id=self.agent.agent_id,
            action_type=resolved_key,
            parameters=parameters,
            reasoning=data.get("reasoning", ""),
        )

    def _resolve_action_key(self, raw_key: str, valid_keys: list[str]) -> str | None:
        """Resolve an LLM-produced action key against valid keys.

        Tries: exact match, suffix match, case-insensitive match.
        """
        # Exact match
        if raw_key in valid_keys:
            return raw_key

        # Suffix match (LLM emitted just the method name)
        for key in valid_keys:
            if key.endswith(f".{raw_key}"):
                return key

        # Case-insensitive
        raw_lower = raw_key.lower()
        for key in valid_keys:
            if key.lower() == raw_lower:
                return key

        # Method name extraction
        for key in valid_keys:
            method_name = key.split(".")[-1] if "." in key else key
            if method_name == raw_key or method_name.lower() == raw_lower:
                return key

        return None

    @classmethod
    def bind(cls, **kwargs):
        """Create a blueprint for this policy."""
        from ...blueprint import ActionPolicyBlueprint
        return ActionPolicyBlueprint(cls=cls, kwargs=kwargs)


async def create_minimal_action_policy(
    agent: Agent,
    max_iterations: int = 50,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> MinimalActionPolicy:
    """Create a minimal action policy.

    Args:
        agent: The owning agent.
        max_iterations: Maximum planning iterations.
        temperature: LLM sampling temperature.
        max_tokens: Maximum tokens for LLM response.

    Returns:
        Initialized MinimalActionPolicy.
    """
    policy = MinimalActionPolicy(
        agent=agent,
        max_iterations=max_iterations,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    await policy.initialize()
    return policy
