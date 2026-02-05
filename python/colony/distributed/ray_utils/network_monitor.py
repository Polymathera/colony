import asyncio
from collections import defaultdict
import logging

import numpy as np
import ray
from sklearn.cluster import DBSCAN

from .rate_limit import RateLimitConfig, TokenBucketRateLimiter

logger = logging.getLogger(__name__)


class NetworkMonitor:
    def __init__(
        self,
        update_interval: int = 60,
        sampling_ratio: float = 0.1,
        tokens_per_second: float = 100,
        burst: int = 1000,
    ):
        self.update_interval = update_interval
        self.sampling_ratio = sampling_ratio
        self.node_latencies: dict[str, dict[str, float]] = {}
        self.rack_nodes_clusters: dict[int, list[str]] = {}
        self.datacenter_nodes_clusters: dict[int, list[str]] = {}
        self.datacenter_racks_clusters: dict[
            int, list[int]
        ] = {}  # Now stores rack IDs instead of node IDs
        self.rate_limiter = TokenBucketRateLimiter(
            RateLimitConfig(tokens_per_second, burst)
        )

    @staticmethod
    def get_latency_resource_name(node_id: str) -> str:
        return f"latency_{node_id}"

    async def run(self):
        while True:
            await self.update_network_metrics()
            self.cluster_nodes_hierarchical()
            await asyncio.sleep(self.update_interval)

    async def update_network_metrics(self):
        # TODO: You might want to add some rate limiting or batching to prevent overwhelming
        #       the network or the nodes.
        # TODO: For very large clusters, you might want to consider sampling techniques rather
        #       than measuring all-to-all latencies.
        nodes = ray.nodes()
        # sampled_nodes = random.sample(nodes, int(len(nodes) * self.sampling_ratio))

        measurement_tasks = [
            NetworkMonitor.options(resources={f"node:{node['NodeID']}": 0.01})
            .remote()
            .measure_latencies(node)
            for node in nodes
        ]
        # This parallelized approach should significantly speed up the latency measurements for your large cluster.
        # This could potentially reduce the measurement time from O(n^2) to O(n), where n is
        # the number of nodes (assuming sufficient parallelism in the Ray cluster).
        # This approach will create a lot of network traffic all at once. Make sure your
        # network can handle this burst of activity.
        # NOTE: Using ray.get in this async context will block the event loop and issue a warning.
        latencies_list = await asyncio.gather(*measurement_tasks)

        for node, latencies in zip(nodes, latencies_list, strict=True):
            self.node_latencies[node["NodeID"]] = latencies
            await self.update_node_resources(node, latencies)

    async def measure_latencies(
        self, node: dict, rate_limit: bool = True
    ) -> dict[str, float]:
        all_nodes = ray.nodes()
        other_nodes = [n for n in all_nodes if n["NodeID"] != node["NodeID"]]
        if rate_limit:
            ping_tasks = [
                self.ping(other_node["NodeManagerAddress"])
                for other_node in other_nodes
            ]
            latencies = await asyncio.gather(*ping_tasks)
            latency_map = {
                other_node["NodeID"]: latency
                for other_node, latency in zip(other_nodes, latencies, strict=True)
            }
            return latency_map
        else:
            latency_map = {}
            for other_node in other_nodes:
                await self.rate_limiter.acquire()  # Rate limit each ping
                latency = await self.ping(other_node["NodeManagerAddress"])
                latency_map[other_node["NodeID"]] = latency
            return latency_map

    async def ping_self(self) -> str:
        """A simple method call to check liveness"""
        return "pong"

    async def ping(self, ip: str, count: int = 3) -> float:
        try:
            process = await asyncio.create_subprocess_exec(
                "ping",
                "-c",
                str(count),
                ip,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                output = stdout.decode()
                lines = output.split("\n")
                for line in lines:
                    if "avg" in line:
                        return float(line.split("/")[4])
            else:
                print(f"Error pinging {ip}: {stderr.decode()}")
        except Exception as e:
            print(f"Exception while pinging {ip}: {e}")
        return float("inf")

    async def update_node_resources(self, node: dict, latencies: dict[str, float]):
        # TODO: The ray.experimental.set_resource is deprecated and has no effect.
        #       We need to find the correct Ray >= 2.x method to update node resources dynamically.
        #       This might involve interacting with the Ray Agent API or rethinking the approach.
        resources = {
            self.get_latency_resource_name(node_id): latency
            for node_id, latency in latencies.items()
        }
        # Commenting out the deprecated call
        # ray.experimental.set_resource(node["NodeID"], resources)
        logger.warning(f"Skipping resource update for node {node['NodeID']} due to deprecated API (ray.experimental.set_resource). Resources not set: {resources}")

    def _cluster_nodes_into_racks(self, eps_rack: float = 1, min_nodes_rack: int = 2):
        # NOTE: Not tested yet
        node_ids = list(self.node_latencies.keys())
        distance_matrix = np.array(
            [
                [self.node_latencies[n1].get(n2, float("inf")) for n2 in node_ids]
                for n1 in node_ids
            ]
        )

        # Cluster nodes into racks
        # This implementation assumes that:
        # - Nodes in the same rack have very low latency (< 1ms),
        # - Nodes in the same data center but different racks have slightly higher latency (< 5ms).
        # - Nodes in different data centers would have even higher latency.
        # TODO: Tune these parameters based on your actual network topology and latency distributions.
        # - eps: The maximum distance between two points to be considered in the same cluster.
        # - min_samples: The minimum number of samples in a cluster to form a cluster.
        rack_clustering = DBSCAN(eps=eps_rack, min_samples=min_nodes_rack).fit(
            distance_matrix
        )
        self.rack_nodes_clusters = defaultdict(list)
        for node_id, label in zip(node_ids, rack_clustering.labels_, strict=True):
            self.rack_nodes_clusters[label].append(node_id)

        return node_ids, distance_matrix

    def cluster_nodes_flat(
        self,
        *,
        eps_rack: float = 1,
        eps_datacenter: float = 5,
        min_nodes_rack: int = 2,
        min_nodes_datacenter: int = 2,
    ):
        # NOTE: Not tested yet
        node_ids, distance_matrix = self._cluster_nodes_into_racks(
            eps_rack, min_nodes_rack
        )

        datacenter_clustering = DBSCAN(
            eps=eps_datacenter, min_samples=min_nodes_datacenter
        ).fit(distance_matrix)
        self.datacenter_nodes_clusters = defaultdict(list)
        for node_id, label in zip(node_ids, datacenter_clustering.labels_, strict=True):
            self.datacenter_nodes_clusters[label].append(node_id)

    def cluster_nodes_hierarchical(
        self,
        *,
        eps_rack: float = 1,
        eps_datacenter: float = 5,
        min_nodes_rack: int = 2,
        min_racks_datacenter: int = 2,
    ):
        # NOTE: Not tested yet
        _, _ = self._cluster_nodes_into_racks(eps_rack, min_nodes_rack)

        # Compute average latencies between racks
        rack_ids = list(self.rack_nodes_clusters.keys())
        rack_distance_matrix = np.zeros((len(rack_ids), len(rack_ids)))
        for i, rack1 in enumerate(rack_ids):
            for j, rack2 in enumerate(rack_ids):
                if i != j:
                    latencies = [
                        self.node_latencies[n1].get(n2, float("inf"))
                        for n1 in self.rack_nodes_clusters[rack1]
                        for n2 in self.rack_nodes_clusters[rack2]
                    ]
                    rack_distance_matrix[i][j] = np.mean(latencies)

        # Cluster racks into datacenters
        datacenter_clustering = DBSCAN(
            eps=eps_datacenter, min_samples=min_racks_datacenter
        ).fit(rack_distance_matrix)
        self.datacenter_racks_clusters = defaultdict(list)
        for rack_id, label in zip(rack_ids, datacenter_clustering.labels_, strict=True):
            self.datacenter_racks_clusters[label].append(rack_id)

    def get_node_rack(self, node_id: str) -> int:
        for rack, nodes in self.rack_nodes_clusters.items():
            if node_id in nodes:
                return rack
        return -1  # Not assigned to any rack

    def get_node_datacenter_flat(self, node_id: str) -> int:
        for dc, nodes in self.datacenter_nodes_clusters.items():
            if node_id in nodes:
                return dc
        return -1  # Not assigned to any datacenter

    def get_node_datacenter_hierarchical(self, node_id: str) -> int:
        rack = self.get_node_rack(node_id)
        for dc, racks in self.datacenter_racks_clusters.items():
            if rack in racks:
                return dc
        return -1  # Not assigned to any datacenter

    async def update_topology_resources(self):
        for node in ray.nodes():
            rack = self.get_node_rack(node["NodeID"])
            datacenter = self.get_node_datacenter_flat(node["NodeID"])
            if datacenter == -1:
                datacenter = self.get_node_datacenter_hierarchical(node["NodeID"])
            resources = {"rack": rack, "datacenter": datacenter}
            ray.experimental.set_resource(node["NodeID"], resources)
