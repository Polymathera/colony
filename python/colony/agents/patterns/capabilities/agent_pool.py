"""Agent pool capability for managing child agent lifecycle.

Provides @action_executor methods for spawning, monitoring, and managing
a dynamic pool of worker agents. Wraps existing Agent spawn/suspend methods.

Wraps:
- Agent.spawn_child_agents()
- Agent.suspend()
- AgentHandle for result retrieval
- AgentSuspensionState for state preservation

Usage:
    # Add capability to agent
    pool_cap = AgentPoolCapability(agent=self)
    self.add_capability(pool_cap)

    # ActionPolicy can now use these actions:
    # - create_agent(agent_type, capabilities, bound_pages, metadata)
    # - get_agent_status(agent_ids, filter_state)
    # - assign_work(agent_id, work_unit, priority)
    # - get_work_results(agent_ids, result_type)
    # - suspend_agent(agent_id, reason)
    # - resume_agent(agent_id)
    # - terminate_agent(agent_id)
"""

from __future__ import annotations

import logging
import time
from typing import Any, TYPE_CHECKING
from overrides import override

from ...base import AgentCapability, AgentHandle
from ...models import AgentSpawnSpec, AgentMetadata, AgentSuspensionState
from ..actions.policies import action_executor

if TYPE_CHECKING:
    from ...base import Agent

logger = logging.getLogger(__name__)


class AgentPoolCapability(AgentCapability):
    """Manages a dynamic pool of worker agents.

    Wraps Agent spawn/suspend methods as @action_executors.
    Uses existing AgentSuspensionState for state preservation including
    working set pages for cache-aware resume.

    No assumptions about:
    - What work agents do (tasks, pages, clusters - all abstracted as "work units")
    - When work completes (agents can work indefinitely, store partial results)
    - How agents are structured (hierarchical, flat, swarm - all supported)

    The ActionPolicy decides when to create/assign/suspend/terminate agents.
    """

    def __init__(self, agent: Agent, scope_id: str | None = None):
        """Initialize agent pool capability.

        Args:
            agent: Owning agent (coordinator)
            scope_id: Blackboard scope (defaults to agent_id)
        """
        super().__init__(agent=agent, scope_id=scope_id)

        # Track created agents
        self._agent_handles: dict[str, AgentHandle] = {}
        self._agent_work: dict[str, dict[str, Any]] = {}  # agent_id -> current work
        self._agent_results: dict[str, list[dict[str, Any]]] = {}  # agent_id -> results

    def get_action_group_description(self) -> str:
        return (
            "Agent Pool — manages dynamic pool of worker agents. "
            "create_agent is expensive (spawns new process with capability setup). "
            "Supports bound_pages for cache-affine routing (agent placed near cached pages). "
            "suspend/resume preserves full state including working set for efficient restart. "
            "terminate is irreversible — use suspend if the agent may be needed again."
        )

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        # TODO: Implement
        logger.warning("serialize_suspension_state not implemented for AgentPoolCapability")
        return state

    @override
    async def deserialize_suspension_state(self, state: AgentSuspensionState) -> None:
        # TODO: Implement
        logger.warning("deserialize_suspension_state not implemented for AgentPoolCapability")
        pass

    # === Action Executors ===

    @action_executor()
    async def create_agent(
        self,
        agent_type: str,
        capabilities: list[str] | None = None,
        bound_pages: list[str] | None = None,
        metadata: AgentMetadata | None = None,
        role: str | None = None,
    ) -> dict[str, Any]:
        """Spawn a new child agent in the pool.

        Use bound_pages for cache affinity - the agent will be routed to a
        replica that has those pages loaded in KV cache.

        Wraps: Agent.spawn_child_agents()

        Args:
            agent_type: Fully qualified class path
                (e.g., "polymathera.colony.agents.base.Agent")
            capabilities: Capability class names to attach
            bound_pages: Pages for affinity routing (routes to replica with these loaded)
            metadata: Passed to child agent's metadata
            role: Role identifier for tracking (e.g., "analyzer", "synthesizer")

        Returns:
            Dict with:
            - agent_id: Created agent's ID
            - role: Role if specified
            - created: Whether creation succeeded
        """
        try:
            # Inherit tenant_id and group_id from parent when not explicitly set
            if metadata is None:
                metadata = AgentMetadata(
                    tenant_id=self.agent.tenant_id,
                    group_id=self.agent.group_id,
                    parent_agent_id=self.agent.agent_id,
                )
            handles: list[AgentHandle] = await self.agent.spawn_child_agents(
                agent_specs=[
                    AgentSpawnSpec(
                        agent_type=agent_type,
                        capability_types=capabilities,
                        bound_pages=bound_pages or [],
                        metadata=metadata,
                    )
                ],
                capability_types=[capabilities],
                return_handles=True,
            )
            handle = handles[0]
            agent_id = handle.child_agent_id
            self._agent_handles[agent_id] = handle
            self._agent_results[agent_id] = []

            # Track role if provided
            if role:
                self.agent.child_agents[role] = agent_id

            logger.info(
                f"AgentPoolCapability: created agent {agent_id} "
                f"(type={agent_type}, bound_pages={len(bound_pages or [])})"
            )

            return {
                "agent_id": agent_id,
                "role": role,
                "created": True,
            }

        except Exception as e:
            logger.error(f"AgentPoolCapability: failed to create agent: {e}")
            return {
                "agent_id": None,
                "role": role,
                "created": False,
                "error": str(e),
            }

    @action_executor()
    async def get_agent_status(
        self,
        agent_ids: list[str] | None = None,
        filter_state: str | None = None,
    ) -> dict[str, Any]:
        """Get status of agents in the pool.

        Args:
            agent_ids: Specific agents to query (None = all tracked agents)
            filter_state: Filter by state ("RUNNING", "STOPPED", "SUSPENDED", etc.)

        Returns:
            Dict with:
            - agents: List of agent status dicts with agent_id, state, current_work
        """
        if agent_ids is None:
            agent_ids = list(self._agent_handles.keys())

        agents = []
        for agent_id in agent_ids:
            handle = self._agent_handles.get(agent_id)

            # Get state from child_agents tracking
            state = "UNKNOWN"
            if agent_id in self.agent.child_agents.values():
                # Agent is tracked as a child
                state = "RUNNING"

            # Check if we have work assigned
            current_work = self._agent_work.get(agent_id)

            agent_status = {
                "agent_id": agent_id,
                "state": state,
                "current_work": current_work,
                "has_handle": handle is not None,
                "result_count": len(self._agent_results.get(agent_id, [])),
            }

            if filter_state is None or state == filter_state:
                agents.append(agent_status)

        return {
            "agents": agents,
            "total": len(agents),
        }

    @action_executor()
    async def assign_work(
        self,
        agent_id: str,
        work_unit: dict[str, Any],
        priority: int = 0,
    ) -> dict[str, Any]:
        """Assign work to an agent.

        Work units are opaque to this primitive. Could be:
        - A page to analyze
        - A cluster of pages
        - A query to answer
        - A hypothesis to validate

        The actual work assignment happens via blackboard communication
        to the target agent.

        Args:
            agent_id: Agent to assign to
            work_unit: Opaque work specification
            priority: Work priority (higher = more urgent)

        Returns:
            Dict with:
            - assigned: Whether work was assigned
            - agent_id: Agent that received work
        """
        if agent_id not in self._agent_handles:
            return {
                "assigned": False,
                "agent_id": agent_id,
                "error": "agent_not_tracked",
            }

        # Store work assignment locally
        self._agent_work[agent_id] = {
            "work_unit": work_unit,
            "priority": priority,
            "assigned_at": time.time(),
        }

        # Send work assignment via blackboard
        blackboard = await self.get_blackboard()
        await blackboard.write(
            f"{agent_id}:work_assignment",
            {
                "work_unit": work_unit,
                "priority": priority,
                "assigned_by": self.agent.agent_id,
                "assigned_at": time.time(),
            },
            created_by=self.agent.agent_id,
        )

        logger.debug(
            f"AgentPoolCapability: assigned work to {agent_id} "
            f"(priority={priority})"
        )

        return {
            "assigned": True,
            "agent_id": agent_id,
        }

    @action_executor()
    async def get_work_results(
        self,
        agent_ids: list[str] | None = None,
        result_type: str = "all",
    ) -> dict[str, Any]:
        """Get work results from agents.

        Results are collected from blackboard where child agents write them.

        Args:
            agent_ids: Specific agents to get results from (None = all)
            result_type: Filter by type ("partial", "final", "all")

        Returns:
            Dict with:
            - results: List of result dicts with agent_id, result_type, data
        """
        if agent_ids is None:
            agent_ids = list(self._agent_handles.keys())

        results = []
        blackboard = await self.get_blackboard()

        for agent_id in agent_ids:
            # Check for results on blackboard
            # Convention: children write to {child_id}:result:{type}
            for rtype in ["partial", "final"]:
                if result_type != "all" and result_type != rtype:
                    continue

                key = f"{agent_id}:result:{rtype}"
                data = await blackboard.read(key)
                if data:
                    results.append({
                        "agent_id": agent_id,
                        "result_type": rtype,
                        "data": data,
                    })

        return {
            "results": results,
            "total": len(results),
        }

    @action_executor()
    async def suspend_agent(
        self,
        agent_id: str,
        reason: str = "",
    ) -> dict[str, Any]:
        """Suspend an agent to free resources.

        State is preserved in AgentSuspensionState including working set
        pages for cache-aware resume later.

        Wraps: AgentManagerBase.suspend_agent()

        Args:
            agent_id: Agent to suspend
            reason: Why suspending (logged and stored)

        Returns:
            Dict with:
            - suspended: Whether suspension succeeded
            - agent_id: Agent that was suspended
        """
        if agent_id not in self._agent_handles:
            return {
                "suspended": False,
                "agent_id": agent_id,
                "error": "agent_not_tracked",
            }

        try:
            # Get manager and suspend
            manager = self.agent._manager
            if manager is None:
                return {
                    "suspended": False,
                    "agent_id": agent_id,
                    "error": "no_manager",
                }

            success = await manager.suspend_agent(agent_id, reason=reason or "")

            if success:
                # Remove from tracking (will be re-added on resume)
                self._agent_handles.pop(agent_id, None)
                self._agent_work.pop(agent_id, None)

                # Remove from child_agents
                roles_to_remove = [
                    role for role, aid in self.agent.child_agents.items()
                    if aid == agent_id
                ]
                for role in roles_to_remove:
                    del self.agent.child_agents[role]

                logger.info(
                    f"AgentPoolCapability: suspended agent {agent_id} "
                    f"(reason: {reason})"
                )

            return {
                "suspended": success,
                "agent_id": agent_id,
            }

        except Exception as e:
            logger.error(f"AgentPoolCapability: failed to suspend {agent_id}: {e}")
            return {
                "suspended": False,
                "agent_id": agent_id,
                "error": str(e),
            }

    @action_executor()
    async def resume_agent(
        self,
        agent_id: str,
    ) -> dict[str, Any]:
        """Resume a suspended agent.

        Uses SoftPageAffinityRouter to route to replica with agent's
        working set already loaded for optimal cache locality.

        Wraps: AgentManagerBase.resume_agent()

        Args:
            agent_id: Agent to resume (uses suspension state key)

        Returns:
            Dict with:
            - resumed: Whether resume succeeded
            - new_agent_id: New agent ID (may differ from original)
        """
        try:
            manager = self.agent._manager
            if manager is None:
                return {
                    "resumed": False,
                    "agent_id": agent_id,
                    "error": "no_manager",
                }

            new_agent_id = await manager.resume_agent(agent_id)

            if new_agent_id:
                # Create new handle for resumed agent
                handle = AgentHandle(
                    child_agent_id=new_agent_id,
                    owner=self.agent,
                )
                self._agent_handles[new_agent_id] = handle
                self._agent_results[new_agent_id] = []

                logger.info(
                    f"AgentPoolCapability: resumed agent {agent_id} -> {new_agent_id}"
                )

            return {
                "resumed": new_agent_id is not None,
                "agent_id": agent_id,
                "new_agent_id": new_agent_id,
            }

        except Exception as e:
            logger.error(f"AgentPoolCapability: failed to resume {agent_id}: {e}")
            return {
                "resumed": False,
                "agent_id": agent_id,
                "error": str(e),
            }

    @action_executor()
    async def terminate_agent(
        self,
        agent_id: str,
        collect_results: bool = True,
    ) -> dict[str, Any]:
        """Terminate an agent and optionally collect final results.

        Args:
            agent_id: Agent to terminate
            collect_results: Whether to collect results before terminating

        Returns:
            Dict with:
            - terminated: Whether termination succeeded
            - final_results: Collected results (if collect_results=True)
        """
        final_results = []

        # Collect results if requested
        if collect_results:
            result_data = await self.get_work_results(
                agent_ids=[agent_id],
                result_type="all",
            )
            final_results = result_data.get("results", [])

        # Remove from tracking
        self._agent_handles.pop(agent_id, None)
        self._agent_work.pop(agent_id, None)
        stored_results = self._agent_results.pop(agent_id, [])
        final_results.extend(stored_results)

        # Remove from child_agents
        roles_to_remove = [
            role for role, aid in self.agent.child_agents.items()
            if aid == agent_id
        ]
        for role in roles_to_remove:
            del self.agent.child_agents[role]

        logger.info(
            f"AgentPoolCapability: terminated agent {agent_id} "
            f"(collected {len(final_results)} results)"
        )

        return {
            "terminated": True,
            "agent_id": agent_id,
            "final_results": final_results,
        }

    @action_executor()
    async def list_available_agents(
        self,
        filter_capabilities: list[str] | None = None,
        filter_state: str | None = None,
        filter_workload: str | None = None,
    ) -> dict[str, Any]:
        """List agents available for work assignment.

        Args:
            filter_capabilities: Require specific capabilities
            filter_state: Filter by state
            filter_workload: Filter by workload ("idle", "light", "heavy")

        Returns:
            Dict with:
            - agents: List of available agent dicts
        """
        # Get all agents
        status_result = await self.get_agent_status()
        all_agents = status_result.get("agents", [])

        available = []
        for agent in all_agents:
            # Filter by state
            if filter_state and agent.get("state") != filter_state:
                continue

            # Filter by workload
            if filter_workload:
                has_work = agent.get("current_work") is not None
                if filter_workload == "idle" and has_work:
                    continue
                if filter_workload in ("light", "heavy") and not has_work:
                    continue

            # TODO: Filter by capabilities (requires querying agent metadata)

            available.append(agent)

        return {
            "agents": available,
            "total": len(available),
        }

    @action_executor()
    async def broadcast_to_agents(
        self,
        message: dict[str, Any],
        agent_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Broadcast a message to multiple agents.

        Uses blackboard to publish message that agents can subscribe to.

        Args:
            message: Message to broadcast
            agent_ids: Specific agents to target (None = all tracked agents)

        Returns:
            Dict with:
            - sent_to: List of agent IDs that received the message
        """
        if agent_ids is None:
            agent_ids = list(self._agent_handles.keys())

        blackboard = await self.get_blackboard()
        sent_to = []

        for agent_id in agent_ids:
            await blackboard.write(
                f"{agent_id}:broadcast",
                {
                    "message": message,
                    "from": self.agent.agent_id,
                    "timestamp": time.time(),
                },
                created_by=self.agent.agent_id,
            )
            sent_to.append(agent_id)

        logger.debug(
            f"AgentPoolCapability: broadcast message to {len(sent_to)} agents"
        )

        return {
            "sent_to": sent_to,
            "total": len(sent_to),
        }
