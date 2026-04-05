

from typing import Any

from ....utils import setup_logger
from ...base import Agent
from ...models import (
    PlanningContext,
    PlanExecutionContext,
)
from .capabilities.replanning import ReplanningDecision


logger = setup_logger(__name__)


class PlanningContextBuilder:
    """Builds the planning context for an agent's planning process.

    The planning context includes the system prompt (agent identity, role, capabilities),
    current goals, constraints, available actions, and relevant memories. This context is
    used by the LLM planner to generate plans that are coherent with the agent's identity
    and informed by its past experiences.

    The builder gathers information from the agent's metadata, capabilities (e.g. memory),
    and optionally from a plan blackboard if one is used for sharing information across
    plans or agents.
    """

    def __init__(self, agent: Agent):
        self.agent = agent

    async def get_replanning_context(
        self, execution_context: PlanExecutionContext,
        decision: ReplanningDecision | None = None
    ) -> PlanningContext:

        planning_context = await self.get_planning_context(execution_context)

        # Pass replanning decision info to planner via custom_data
        if decision:
            planning_context.custom_data["revision_triggers"] = [
                t.value for t in decision.triggers
            ]
            planning_context.custom_data["revision_strategy"] = decision.strategy.value
            planning_context.custom_data["revision_reason"] = decision.reason

        return planning_context

    async def get_planning_context(self, execution_context: PlanExecutionContext) -> PlanningContext:
        """Build the planning context for the current planning step (initial or replanning).
        """
        goals = self.agent.metadata.goals or []

        # Recall memories for planning context
        recalled_memories = await self._gather_memories()

        # Get memory architecture guidance for LLM reasoning
        memory_guidance = await self._get_memory_architecture_guidance()
        custom_data: dict[str, Any] = {}
        if memory_guidance:
            custom_data["memory_architecture_guidance"] = memory_guidance

        from ..actions.policies import BaseActionPolicy
        action_policy: BaseActionPolicy = self.agent.action_policy
        assert isinstance(action_policy, BaseActionPolicy), "Agent's action_policy must be a BaseActionPolicy for planning context building"

        # Create new planning context based on current state
        return PlanningContext(
            system_prompt=await self._build_system_prompt(),
            execution_context=execution_context,
            page_ids=list(self.agent.bound_pages) if hasattr(self.agent, 'bound_pages') else [],
            goals=goals,
            constraints=self._get_constraints(),
            action_descriptions=await action_policy.get_action_descriptions(),
            action_group_summaries=await action_policy.get_action_group_summaries(),
            recalled_memories=recalled_memories,
            custom_data=custom_data,
            # parent_plan_id=self.current_plan.parent_plan_id,
        )

    async def _gather_memories(self) -> list[dict[str, Any]]:
        """Gather memories for planning context via AgentContextEngine.

        If the agent has an AgentContextEngine capability, this retrieves
        relevant memories from working memory and potentially STM/LTM.

        Returns:
            List of recalled memory dicts for the planning context
        """
        # Import here to avoid circular imports
        from ..memory import AgentContextEngine, MemoryQuery
        from ...scopes import MemoryScope

        ctx_engine: AgentContextEngine = self.agent.get_capability_by_type(AgentContextEngine)
        if ctx_engine is None:
            return []

        try:
            # Gather context from all available memory scopes
            entries = await ctx_engine.gather_context(
                query=MemoryQuery(max_results=50),
                ### scopes=[
                ###     MemoryScope.agent_working(self.agent),
                ###     MemoryScope.agent_stm(self.agent),
                ###     MemoryScope.agent_ltm_episodic(self.agent),
                ### ],
            )

            # Diagnostic: log what gather_context returned
            action_entries = [e for e in entries if e.tags and "action" in e.tags]
            logger.warning(
                f"gather_context returned {len(entries)} entries "
                f"({len(action_entries)} actions): {[e.key for e in entries[:10]]}"
            )

            # Convert BlackboardEntry objects to dicts for the planning context
            recalled_memories = []
            for entry in entries:
                recalled_memories.append({
                    "key": entry.key,
                    "value": entry.value,
                    "tags": list(entry.tags) if entry.tags else [],
                    "created_at": entry.created_at,
                    "relevance": entry.metadata.get("relevance", 1.0),
                })
            return recalled_memories
        except Exception as e:
            logger.warning(f"Failed to gather planning context: {e}")
            return []

    async def _get_memory_architecture_guidance(self) -> str | None:
        """Get memory architecture guidance for inclusion in planning prompts.

        If the agent has an AgentContextEngine, generates a description of the
        agent's memory system (levels, dataflow, available actions, capacity)
        that the LLM planner can use to reason about memory as a first-class
        cognitive resource.

        Returns:
            Guidance string, or None if no context engine is available.
        """
        from ..memory import AgentContextEngine

        ctx_engine: AgentContextEngine = self.agent.get_capability_by_type(AgentContextEngine)
        if ctx_engine is None:
            return None

        try:
            return await ctx_engine.get_memory_architecture_guidance()
        except Exception as e:
            logger.warning(f"Failed to get memory architecture guidance: {e}")
            return None

    def _indefinite_article(self, word: str) -> str:
        """Return 'a' or 'an' based on the first letter of the word."""
        return "an" if word[0].lower() in "aeiou" else "a"

    def _format_role(self, name: str, role: str | None) -> str:
        identity = f"You are {self._indefinite_article(name)} {name}"
        if role:
            identity += f", {self._indefinite_article(role)} {role}"
        return identity

    async def _build_system_prompt(self) -> str:
        """Build stable agent identity prompt.

        Uses AgentSelfConcept from ConsciousnessCapability when available,
        falling back to agent metadata, class docstring, and capabilities.
        """
        from ..capabilities.consciousness import ConsciousnessCapability

        agent = self.agent
        parts: list[str] = []

        # Try to get self-concept from ConsciousnessCapability
        consciousness: ConsciousnessCapability | None = agent.get_capability_by_type(ConsciousnessCapability)
        self_concept = await consciousness.get_self_concept() if consciousness else None

        if self_concept:
            # Build from self-concept
            identity = self._format_role(self_concept.name, self_concept.role)
            parts.append(identity)

            if self_concept.description:
                parts.append(self_concept.description)

            if self_concept.identity:
                parts.append(self_concept.identity)

            if self_concept.goals:
                parts.append("Your goals are:\n" + "\n".join(f"- {g}" for g in self_concept.goals))

            if self_concept.constraints:
                parts.append("Your constraints are:\n" + "\n".join(f"- {c}" for c in self_concept.constraints))

            if self_concept.capabilities:
                parts.append(f"Your capabilities are: {', '.join(self_concept.capabilities)}")

            if self_concept.limitations:
                parts.append("Your limitations are:\n" + "\n".join(f"- {l}" for l in self_concept.limitations))

            if self_concept.world_model:
                parts.append(f"Your world model is: {self_concept.world_model}")

            if self_concept.frame_of_mind:
                parts.append(f"Your frame of mind is: {self_concept.frame_of_mind}")
        else:
            # Fallback: build from agent metadata and class info
            identity = self._format_role(agent.__class__.__name__, agent.metadata.role)
            identity += f", a {agent.agent_type} agent."
            parts.append(identity)

            doc = agent.__class__.__doc__
            if doc:
                parts.append(doc.strip().split('\n\n')[0].strip())

            cap_names = agent.get_capability_names()
            if cap_names:
                parts.append(f"Your capabilities: {', '.join(cap_names)}")

        # Task parameters (always from metadata — these are per-run, not part of self-concept)
        params = agent.metadata.parameters
        if params:
            param_lines = [f"- {k}: {v}" for k, v in params.items()
                           if not k.startswith("_") and k != "planning_params"]
            if param_lines:
                parts.append("Task parameters:\n" + "\n".join(param_lines))

        return "\n\n".join(parts)

    def _get_constraints(self) -> dict[str, Any]:
        """Extract execution constraints from agent metadata."""
        constraints: dict[str, Any] = {}
        meta = self.agent.metadata
        if meta.max_iterations:
            constraints["max_iterations"] = meta.max_iterations
        params = meta.parameters
        if "max_agents" in params:
            constraints["max_parallel_workers"] = params["max_agents"]
        if "quality_threshold" in params:
            constraints["quality_threshold"] = params["quality_threshold"]
        return constraints


