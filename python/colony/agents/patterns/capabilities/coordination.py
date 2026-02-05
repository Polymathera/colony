"""Policy hierarchies for multi-agent execution framework.

Provides pluggable policies for:
- Batching: How to group work items for processing
- Prefetching: How to predict and preload pages
- Coordination: How to assign work to agents

These policies are used by capabilities to implement various
distributed multi-agent execution strategies.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)



# =============================================================================
# Coordination Policies (Agent Work Assignment)
# =============================================================================

class CoordinationPolicy(ABC):
    """Abstract base for agent coordination strategies.

    Determines how to assign work to agents in a pool.
    Different strategies trade off between:
    - Load balancing
    - Cache affinity
    - Specialization
    """

    @abstractmethod
    async def select_agent(
        self,
        work_unit: dict[str, Any],
        available_agents: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> str | None:
        """Select agent to assign work to.

        Args:
            work_unit: Work to assign
            available_agents: List of agent info dicts
            context: Additional context

        Returns:
            Agent ID to assign to, or None if no suitable agent
        """
        pass


class RoundRobinCoordinationPolicy(CoordinationPolicy):
    """Round-robin work assignment.

    Cycles through agents in order, ensuring even distribution.

    Use when:
    - All agents are equivalent
    - Work items are independent
    - You want simple, fair distribution
    """

    def __init__(self):
        """Initialize round-robin policy."""
        self._last_index = -1

    async def select_agent(
        self,
        work_unit: dict[str, Any],
        available_agents: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> str | None:
        """Select next agent in rotation."""
        if not available_agents:
            return None

        self._last_index = (self._last_index + 1) % len(available_agents)
        return available_agents[self._last_index].get("agent_id")


class AffinityCoordinationPolicy(CoordinationPolicy):
    """Cache affinity-based work assignment.

    Assigns work to agents that have relevant pages loaded,
    maximizing cache reuse.

    Use when:
    - Cache locality is critical
    - Agents have different page sets loaded
    - Work items have clear page associations
    """

    def __init__(self, min_affinity: float = 0.3):
        """Initialize affinity policy.

        Args:
            min_affinity: Minimum affinity score to consider
        """
        self.min_affinity = min_affinity

    async def select_agent(
        self,
        work_unit: dict[str, Any],
        available_agents: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> str | None:
        """Select agent with highest page affinity."""
        if not available_agents:
            return None

        work_pages = set(work_unit.get("pages", []))
        if not work_pages:
            # No page info, use round-robin fallback
            return available_agents[0].get("agent_id")

        best_agent = None
        best_affinity = 0.0

        for agent in available_agents:
            agent_pages = set(agent.get("loaded_pages", []))
            if not agent_pages:
                continue

            # Calculate affinity as overlap ratio
            affinity = len(work_pages & agent_pages) / len(work_pages)

            if affinity > best_affinity and affinity >= self.min_affinity:
                best_affinity = affinity
                best_agent = agent.get("agent_id")

        return best_agent or available_agents[0].get("agent_id")


class LoadBalancingCoordinationPolicy(CoordinationPolicy):
    """Load-balanced work assignment.

    Assigns work to agents with lowest current workload.

    Use when:
    - Work items have varying completion times
    - You want to minimize overall completion time
    - Agent capacity varies
    """

    def __init__(self, max_workload: int = 5):
        """Initialize load balancing policy.

        Args:
            max_workload: Maximum work items per agent
        """
        self.max_workload = max_workload

    async def select_agent(
        self,
        work_unit: dict[str, Any],
        available_agents: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> str | None:
        """Select agent with lowest workload."""
        if not available_agents:
            return None

        # Filter agents under max workload
        eligible = [
            a for a in available_agents
            if a.get("current_workload", 0) < self.max_workload
        ]

        if not eligible:
            return None

        # Select agent with lowest workload
        best_agent = min(
            eligible,
            key=lambda a: a.get("current_workload", 0)
        )

        return best_agent.get("agent_id")


class CompositeCoordinationPolicy(CoordinationPolicy):
    """Composite coordination combining affinity and load balancing.

    First filters by affinity, then selects by load.
    """

    def __init__(
        self,
        affinity_policy: AffinityCoordinationPolicy | None = None,
        load_policy: LoadBalancingCoordinationPolicy | None = None,
    ):
        """Initialize composite policy.

        Args:
            affinity_policy: Affinity-based policy
            load_policy: Load balancing policy
        """
        self.affinity_policy = affinity_policy or AffinityCoordinationPolicy()
        self.load_policy = load_policy or LoadBalancingCoordinationPolicy()

    async def select_agent(
        self,
        work_unit: dict[str, Any],
        available_agents: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> str | None:
        """Select by affinity first, then load balance."""
        if not available_agents:
            return None

        # Try affinity first
        work_pages = set(work_unit.get("pages", []))
        if work_pages:
            # Filter to agents with any affinity
            affinity_agents = []
            for agent in available_agents:
                agent_pages = set(agent.get("loaded_pages", []))
                if agent_pages & work_pages:
                    affinity_agents.append(agent)

            if affinity_agents:
                # Load balance among affinity agents
                return await self.load_policy.select_agent(
                    work_unit, affinity_agents, context
                )

        # No affinity info, pure load balancing
        return await self.load_policy.select_agent(
            work_unit, available_agents, context
        )
