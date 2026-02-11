"""Page-affinity router for agent scheduling.

This router implements intelligent agent spawning policies:
- Routes agents to replicas that have their bound pages loaded
- Supports both hard affinity (ALL pages required) and soft affinity (best effort)
- Respects component boundaries (VCM owns memory, router owns routing)
"""

import logging
from typing import Any

from ..distributed.ray_utils.serving import (
    RequestRouter,
    RoutingHints,
    DeploymentReplicaInfo,
    DeploymentRequest,
)
from ..system import get_vcm, get_agent_system
from ..vcm.models import PageLocation


logger = logging.getLogger(__name__)


class SoftPageAffinityRouter(RequestRouter):
    """Router that selects VLLM replicas based on page affinity.

    This router implements a simplified agent spawning policy:
    1. Query VCM for page locations across replicas
    2. Score replicas by number of bound_pages already loaded
    3. Select replica based on affinity mode:
       - Hard affinity (soft_affinity=False): Only consider replicas with ALL bound pages
       - Soft affinity (soft_affinity=True): Select replica with maximum bound pages loaded
    4. Return selected replica for proxy to call start_agent()

    Component Responsibilities:
    - VCM: Owns all memory management (page loading, eviction)
    - VLLMDeployment: Checks local capacity, starts/stops agents, handles suspension
    - SoftPageAffinityRouter: Routing policy (select best replica)
    - AgentSystemDeployment: High-level orchestration, scaling decisions
    """

    def __init__(self, strip_routing_params: list[str] | None = None):
        """Initialize router with lazy VCM handle.

        Args:
            strip_routing_params: List of parameter names to strip from kwargs
                before passing to the method. Defaults to ["soft_affinity"].
        """
        super().__init__()
        self._vcm_handle = None
        self.strip_routing_params = strip_routing_params or ["soft_affinity"]

    async def _get_vcm_handle(self):
        """Lazy initialization of VCM handle."""
        if self._vcm_handle is None:
            # Get deployment names
            self._vcm_handle = get_vcm()
        return self._vcm_handle

    @staticmethod
    def extract_routing_hints(
        method_name: str,
        args: tuple,  # noqa: ARG004 - Required by RequestRouter interface
        kwargs: dict[str, Any],
    ) -> RoutingHints | None:
        """Extract page affinity routing hints from start_agent() arguments.

        Args:
            method_name: Name of the method being called
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            RoutingHints with bound_pages and soft_affinity metadata
        """
        if method_name != "start_agent":
            return None

        # Extract bound_pages and soft_affinity from kwargs
        bound_pages = kwargs.get("bound_pages", [])
        soft_affinity = kwargs.get("soft_affinity", False)

        if not bound_pages:
            return None  # No page affinity, use default routing

        return RoutingHints(
            router_class=SoftPageAffinityRouter,
            metadata={
                "bound_pages": bound_pages,
                "soft_affinity": soft_affinity,
            }
        )

    async def route_request(
        self,
        request: DeploymentRequest,
        replicas: list[DeploymentReplicaInfo],
    ) -> DeploymentReplicaInfo:
        """Route start_agent request using page affinity scoring.

        This router ONLY selects a replica. It does NOT call methods or replicate pages.

        Args:
            request: The start_agent request
            replicas: Available VLLM replicas

        Returns:
            Selected replica for proxy to call start_agent()

        Raises:
            ValueError: If no healthy replicas or no suitable replica found
        """
        if not replicas:
            raise ValueError("No healthy replicas available")

        # Strip routing parameters from kwargs before passing to method
        # This prevents TypeError when the target method doesn't accept these params
        for param in self.strip_routing_params:
            request.kwargs.pop(param, None)

        # Extract routing hints
        if not request.routing_hints or not request.routing_hints.metadata:
            # No page affinity, use least-loaded
            return self._select_least_loaded(replicas)

        bound_pages = request.routing_hints.metadata.get("bound_pages", [])
        soft_affinity = request.routing_hints.metadata.get("soft_affinity", False)

        if not bound_pages:
            return self._select_least_loaded(replicas)

        # Get VCM handle
        vcm_handle = await self._get_vcm_handle()

        # Query VCM for page locations
        page_locations = await self._get_page_locations(vcm_handle, bound_pages)

        # Score and rank replicas
        ranked_replicas = await self._rank_replicas_by_affinity(
            replicas, bound_pages, page_locations
        )

        logger.info(
            f"Ranked {len(ranked_replicas)} replicas for agent with "
            f"{len(bound_pages)} bound pages (soft_affinity={soft_affinity})"
        )

        # Select best replica based on affinity mode
        for replica_info in ranked_replicas:
            replica = replica_info["replica"]
            pages_on_replica = replica_info["pages_on_replica"]
            missing_pages = replica_info["missing_pages"]

            # Hard affinity: require ALL pages
            if not soft_affinity and missing_pages:
                logger.debug(
                    f"Skipping replica {replica.replica_id}: "
                    f"hard affinity requires all pages, missing {len(missing_pages)}"
                )
                continue

            # Found suitable replica (all pages for hard, or best for soft)
            logger.info(
                f"Selected replica {replica.replica_id} for agent "
                f"({len(pages_on_replica)}/{len(bound_pages)} pages loaded)"
            )
            return replica

        # No suitable replica found
        raise ValueError(
            f"No suitable replica found for agent with {len(bound_pages)} bound pages "
            f"(soft_affinity={soft_affinity})"
        )

    async def _get_page_locations(
        self,
        vcm_handle: Any,
        page_ids: list[str],
    ) -> dict[str, list[PageLocation]]:
        """Query VCM for page locations.

        Args:
            vcm_handle: VCM deployment handle
            page_ids: Page IDs to query

        Returns:
            Dict mapping page_id -> list of PageLocation objects
        """
        page_locations = {}
        for page_id in page_ids:
            try:
                locations = await vcm_handle.get_page_locations(page_id=page_id)
                page_locations[page_id] = locations or []
            except Exception as e:
                logger.warning(f"Failed to get locations for page {page_id}: {e}")
                page_locations[page_id] = []

        return page_locations

    async def _rank_replicas_by_affinity(
        self,
        replicas: list[DeploymentReplicaInfo],
        bound_pages: list[str],
        page_locations: dict[str, list[PageLocation]],
    ) -> list[dict[str, Any]]:
        """Rank replicas by how many bound pages they have loaded.

        Args:
            replicas: Available replicas
            bound_pages: Pages needed by agent
            page_locations: Page locations from VCM

        Returns:
            List of dicts sorted by score (descending):
            [
                {
                    "replica": DeploymentReplicaInfo,
                    "score": int,  # Number of pages on replica
                    "pages_on_replica": set[str],
                    "missing_pages": set[str],
                },
                ...
            ]
        """
        replica_scores = []

        for replica in replicas:
            pages_on_replica = set()
            missing_pages = set()

            for page_id in bound_pages:
                locations = page_locations.get(page_id, [])

                # Check if this replica has the page
                has_page = any(
                    loc.client_id == replica.replica_id
                    for loc in locations
                )

                if has_page:
                    pages_on_replica.add(page_id)
                else:
                    missing_pages.add(page_id)

            score = len(pages_on_replica)

            replica_scores.append({
                "replica": replica,
                "score": score,
                "pages_on_replica": pages_on_replica,
                "missing_pages": missing_pages,
            })

        # Sort by score descending (most pages first)
        replica_scores.sort(key=lambda x: x["score"], reverse=True)

        return replica_scores

    def _select_least_loaded(
        self,
        replicas: list[DeploymentReplicaInfo]
    ) -> DeploymentReplicaInfo:
        """Fallback to least-loaded routing when no page affinity.

        Args:
            replicas: Available replicas

        Returns:
            Replica with lowest load
        """
        return min(
            replicas,
            key=lambda r: r.queue_length + r.in_flight_requests
        )


class AgentAffinityRouter(RequestRouter):
    """Routes agent lifecycle calls to the replica that owns the agent.

    Queries AgentSystemDeployment.agent_locations to find which replica
    has the agent, then routes the call there.

    Used for: suspend_agent(), stop_agent(), get_agent_state()

    NOT used for resume_agent() - resumed agents are deleted from replica during suspension,
    so there is no "owning replica" to route to. Resume goes through AgentSystem.resume_agent()
    which spawns via SoftPageAffinityRouter.
    """

    def __init__(self):
        super().__init__()
        self._agent_system_handle = None

    async def _get_agent_system_handle(self):
        """Lazy init agent system handle."""
        if self._agent_system_handle is None:
            self._agent_system_handle = get_agent_system()
        return self._agent_system_handle

    @staticmethod
    def extract_routing_hints(
        method_name: str,
        args: tuple,
        kwargs: dict[str, Any],
    ) -> RoutingHints | None:
        """Extract agent_id from call arguments.

        Args:
            method_name: Name of the method being called
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            RoutingHints with agent_id metadata
        """
        # Methods that need agent affinity routing
        if method_name not in ["suspend_agent", "stop_agent", "get_agent_state"]:
            return None

        # Agent ID is first positional arg or 'agent_id' kwarg
        agent_id = None
        if args:
            agent_id = args[0]
        elif "agent_id" in kwargs:
            agent_id = kwargs["agent_id"]

        if not agent_id:
            return None

        return RoutingHints(
            router_class=AgentAffinityRouter,
            metadata={"agent_id": agent_id}
        )

    async def route_request(
        self,
        request: DeploymentRequest,
        replicas: list[DeploymentReplicaInfo],
    ) -> DeploymentReplicaInfo:
        """Route to replica that owns the agent.

        Args:
            request: The lifecycle request (suspend/stop/get_state)
            replicas: Available replicas

        Returns:
            Selected replica that owns the agent

        Raises:
            ValueError: If no healthy replicas available
        """
        if not replicas:
            raise ValueError("No healthy replicas available")

        # Extract routing hints
        if not request.routing_hints or not request.routing_hints.metadata:
            # Fallback to least-loaded
            return self._select_least_loaded(replicas)

        agent_id = request.routing_hints.metadata.get("agent_id")
        if not agent_id:
            return self._select_least_loaded(replicas)

        # Query AgentSystem for agent location
        agent_system = await self._get_agent_system_handle()

        try:
            replica_id = await agent_system.get_agent_location(agent_id)

            if not replica_id:
                logger.warning(
                    f"Agent {agent_id} not found in agent_locations. "
                    f"Routing to fallback replica."
                )
                return self._select_least_loaded(replicas)

            # Find matching replica
            for replica in replicas:
                if replica.replica_id == replica_id:
                    logger.debug(f"Routing {request.method_name}({agent_id}) to {replica_id}")
                    return replica

            # Agent's replica not in healthy replicas list
            logger.warning(
                f"Agent {agent_id} on replica {replica_id}, but replica not healthy. "
                f"Routing to fallback."
            )
            return self._select_least_loaded(replicas)

        except Exception as e:
            logger.error(f"Failed to get agent location for {agent_id}: {e}")
            return self._select_least_loaded(replicas)

    def _select_least_loaded(
        self,
        replicas: list[DeploymentReplicaInfo]
    ) -> DeploymentReplicaInfo:
        """Fallback to least-loaded routing when agent location unknown.

        Args:
            replicas: Available replicas

        Returns:
            Replica with lowest load
        """
        return min(
            replicas,
            key=lambda r: r.queue_length + r.in_flight_requests
        )



