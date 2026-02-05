"""Request routing policies for deployments."""

import asyncio
import logging
from abc import ABC, abstractmethod

from .models import DeploymentReplicaInfo, DeploymentRequest

logger = logging.getLogger(__name__)


class RequestRouter(ABC):
    """Base class for request routing policies.

    Subclass this to implement custom routing logic for deployments.
    The router determines which replica should handle each incoming request.
    """

    def __init__(self):
        """Initialize the router."""
        pass

    @abstractmethod
    async def route_request(
        self,
        request: DeploymentRequest,
        replicas: list[DeploymentReplicaInfo],
    ) -> DeploymentReplicaInfo:
        """Route a request to a replica.

        Args:
            request: The incoming request to route.
            replicas: Available healthy replicas.

        Returns:
            The replica that should handle this request.

        Raises:
            ValueError: If no suitable replica can be found.
        """
        pass


class RoundRobinRouter(RequestRouter):
    """Round-robin load balancer across replicas."""

    def __init__(self):
        super().__init__()
        self._index = 0

    async def route_request(
        self,
        request: DeploymentRequest,
        replicas: list[DeploymentReplicaInfo],
    ) -> DeploymentReplicaInfo:
        """Route request using round-robin selection."""
        if not replicas:
            raise ValueError("No healthy replicas available")

        replica = replicas[self._index % len(replicas)]
        self._index += 1
        return replica


class LeastLoadedRouter(RequestRouter):
    """Route requests to the replica with the least load.

    Load is measured as: queue_length + in_flight_requests
    """

    async def route_request(
        self,
        request: DeploymentRequest,
        replicas: list[DeploymentReplicaInfo],
    ) -> DeploymentReplicaInfo:
        """Route request to the replica with lowest load."""
        if not replicas:
            raise ValueError("No healthy replicas available")

        # Calculate load for each replica
        def get_load(replica: DeploymentReplicaInfo) -> int:
            return replica.queue_length + replica.in_flight_requests

        # Select replica with minimum load
        return min(replicas, key=get_load)
