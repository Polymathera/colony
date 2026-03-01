"""Colony cluster connection via Ray Client API.

Manages the Ray connection and provides cached deployment handles
for calling Colony's serving deployments (AgentSystem, SessionManager, VCM).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ColonyConnection:
    """Manages connection to the running Colony Ray cluster."""

    def __init__(
        self,
        ray_client_address: str = "ray://ray-head:10001",
        ray_dashboard_url: str = "http://ray-head:8265",
        prometheus_url: str = "http://ray-head:9090",
    ):
        self.ray_client_address = ray_client_address
        self.ray_dashboard_url = ray_dashboard_url
        self.prometheus_url = prometheus_url
        self._connected = False
        self._handle_cache: dict[str, Any] = {}
        self._http_client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        """Connect to the Ray cluster via Client API."""
        import ray

        if not ray.is_initialized():
            try:
                ray.init(
                    address=self.ray_client_address,
                    namespace="polymathera",
                    logging_level=logging.WARNING,
                )
                self._connected = True
                logger.info("Ray Client connected")
            except Exception as e:
                logger.warning(f"Ray Client connection failed: {e}. Dashboard will run in degraded mode.")
                self._connected = False

        self._http_client = httpx.AsyncClient(timeout=10.0)

    async def disconnect(self) -> None:
        """Disconnect from the Ray cluster."""
        import ray

        if ray.is_initialized():
            ray.shutdown()
        self._connected = False
        self._handle_cache.clear()
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_deployment_handle(self, app_name: str, deployment_name: str) -> Any:
        """Get a cached deployment handle.

        Returns a DeploymentHandle that can call @serving.endpoint methods
        on the target deployment via Ray actor RPC.
        """
        cache_key = f"{app_name}/{deployment_name}"
        if cache_key not in self._handle_cache:
            from colony.distributed.ray_utils.serving import get_deployment
            self._handle_cache[cache_key] = get_deployment(app_name, deployment_name)
        return self._handle_cache[cache_key]

    async def get_ray_cluster_status(self) -> dict[str, Any]:
        """Get Ray cluster status via the Ray Dashboard API."""
        if not self._http_client:
            return {"status": "disconnected"}
        try:
            resp = await self._http_client.get(f"{self.ray_dashboard_url}/api/cluster_status")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def get_ray_nodes(self) -> list[dict[str, Any]]:
        """Get Ray node info via the Dashboard API."""
        if not self._http_client:
            return []
        try:
            resp = await self._http_client.get(f"{self.ray_dashboard_url}/nodes?view=summary")
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", {}).get("summary", [])
        except Exception:
            return []

    async def query_prometheus(self, query: str) -> dict[str, Any]:
        """Run a PromQL instant query."""
        if not self._http_client:
            return {"status": "error", "error": "No HTTP client"}
        try:
            resp = await self._http_client.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def fetch_prometheus_metrics(self) -> str:
        """Fetch raw Prometheus metrics text."""
        if not self._http_client:
            return ""
        try:
            resp = await self._http_client.get(f"{self.prometheus_url}/metrics")
            resp.raise_for_status()
            return resp.text
        except Exception:
            return ""
