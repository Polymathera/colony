

import ray

# from ray.core.scheduler import ResourceScheduler
# from ray.core.scheduler.scheduling_algorithm import SchedulingAlgorithm
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


class TopologyAwareScheduler:  # (SchedulingAlgorithm):
    """
    A custom Ray scheduler that makes placement decisions based on network topology.

    This scheduler attempts to maximize data locality and minimize network latency by:
    1. First trying to schedule tasks on the same node as the originating task
    2. Then trying nodes in the same rack
    3. Then trying nodes in the same datacenter
    4. Finally falling back to any available node

    The topology is represented as a map of datacenters to lists of node IDs:
    ```python
    {
        "us-east-1a": ["node1", "node2"],  # Nodes in first datacenter
        "us-east-1b": ["node3", "node4"]   # Nodes in second datacenter
    }
    ```

    Usage with Ray:
    ```python
    topology_map = {
        "us-east-1a": ["i-1234", "i-5678"],
        "us-east-1b": ["i-9012", "i-3456"]
    }
    scheduler = TopologyAwareScheduler(topology_map)
    ray.init(scheduling_algorithm=scheduler)
    ```

    Args:
        `topology_map`: dict mapping datacenter IDs to lists of node IDs

    Topology-aware task scheduling for Ray using placement groups and custom resources.

    This scheduler uses Ray's placement groups and custom resources to implement
    topology-aware scheduling. It works by:
    1. Creating placement groups that map to physical topology (datacenters/racks)
    2. Using custom resources to track topology information
    3. Using Ray's placement group scheduling strategies

    Example:

    Usage with Ray:
    ```python
    scheduler = TopologyAwareScheduler(topology_map={
        "us-east-1a": ["node1", "node2"],
        "us-east-1b": ["node3", "node4"]
    })

    # Create actor with topology awareness
    MyActor = ray.remote(Actor)
    actor = scheduler.create_actor(
        MyActor,
        origin_node="node1",
        resource_demands={"GPU": 1}
    )
    ```
    """

    def __init__(self, topology_map: dict[str, list[str]]):
        self.topology_map = topology_map  # Map of datacenter to list of node IDs
        self._placement_groups = {}
        self._setup_topology_resources()

    def _setup_topology_resources(self):
        """Set up custom resources for topology"""
        for dc, nodes in self.topology_map.items():
            # Create placement group for datacenter
            bundles = [
                {"CPU": 0.001, f"node_{node}": 1, f"dc_{dc}": 1} for node in nodes
            ]
            pg = ray.util.placement_group(bundles, strategy="STRICT_PACK")
            ray.get(pg.ready())
            self._placement_groups[dc] = pg

    def create_actor(
        self,
        actor_cls: ray.remote,
        origin_node: str | None = None,
        resource_demands: dict[str, float] | None = None,
        **kwargs,
    ) -> ray.actor.ActorHandle:
        """Create an actor with topology-aware placement"""
        if origin_node:
            # Find datacenter containing origin node
            dc = next(
                (dc for dc, nodes in self.topology_map.items() if origin_node in nodes),
                None,
            )
            if dc:
                pg = self._placement_groups[dc]
                # Try same node first
                try:
                    actor = actor_cls.options(
                        placement_group=pg,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg, placement_group_bundle_index=0
                        ),
                        resources={f"node_{origin_node}": 0.001, **resource_demands},
                    ).remote(**kwargs)
                    return actor
                except ray.exceptions.GetTimeoutError:
                    # Fall back to same datacenter
                    return actor_cls.options(
                        placement_group=pg,
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg
                        ),
                        resources={f"dc_{dc}": 0.001, **resource_demands},
                    ).remote(**kwargs)

        # Fall back to any node
        return actor_cls.options(resources=resource_demands).remote(**kwargs)

    def schedule(
        self,
        resource_demands: list[dict],
        node_resources: dict[str, dict],
        node_constraints: dict[str, list[str]],
    ) -> str | None:
        # Get the originating node (assuming it's passed in the constraints)
        origin_node = node_constraints.get("origin_node", [None])[0]
        if not origin_node:
            return None

        # Find the datacenter of the origin node
        origin_dc = next(
            (dc for dc, nodes in self.topology_map.items() if origin_node in nodes),
            None,
        )

        # First, try to schedule on the same node
        if origin_node in node_resources and self._resources_fit(
            resource_demands, node_resources[origin_node]
        ):
            return origin_node

        # Then, try to schedule in the same datacenter
        if origin_dc:
            for node in self.topology_map[origin_dc]:
                if node in node_resources and self._resources_fit(
                    resource_demands, node_resources[node]
                ):
                    return node

        # If no suitable node found in the same datacenter, fall back to any available node
        for node, resources in node_resources.items():
            if self._resources_fit(resource_demands, resources):
                return node

        return None

    def _resources_fit(self, demands: list[dict], available: dict) -> bool:
        for demand in demands:
            for resource, amount in demand.items():
                if available.get(resource, 0) < amount:
                    return False
        return True
