"""Colony cluster connection via Ray Client API.

Manages the Ray connection and provides cached deployment handles
for calling Colony's serving deployments (AgentSystem, SessionManager, VCM).

Uses the proper system helpers from colony.system (get_session_manager,
get_agent_system, get_vcm, etc.) which resolve deployment names through
colony.deployment_names.get_deployment_names().
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Default app name — matches polymath.py default
_DEFAULT_APP_NAME = "polymathera"


class ColonyConnection:
    """Manages connection to the running Colony Ray cluster."""

    def __init__(
        self,
        ray_client_address: str = "ray://ray-head:10001",
        ray_dashboard_url: str = "http://ray-head:8265",
        prometheus_url: str = "http://ray-head:9090",
        app_name: str = _DEFAULT_APP_NAME,
    ):
        self.ray_client_address = ray_client_address
        self.ray_dashboard_url = ray_dashboard_url
        self.prometheus_url = prometheus_url
        self.app_name = app_name
        self._connected = False
        self._handle_cache: dict[str, Any] = {}
        self._http_client: httpx.AsyncClient | None = None
        # Observability (direct DB + Kafka access for traces)
        self._kafka_bootstrap: str | None = None
        self._db_pool: Any | None = None
        self._span_consumer: Any | None = None
        self._span_query_store: Any | None = None

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
                logger.info("Ray Client connected to %s", self.ray_client_address)
            except Exception as e:
                logger.warning(f"Ray Client connection failed: {e}. Dashboard will run in degraded mode.")
                self._connected = False

        self._http_client = httpx.AsyncClient(timeout=10.0)

    async def disconnect(self) -> None:
        """Disconnect from the Ray cluster."""
        import ray

        # Stop observability consumer
        if self._span_consumer:
            await self._span_consumer.stop()
            self._span_consumer = None
        if self._db_pool:
            await self._db_pool.close()
            self._db_pool = None

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

    def _get_handle(self, name_attr: str) -> Any:
        """Get a cached deployment handle using colony.system helpers.

        Uses get_deployment_names() to resolve the actual deployment name,
        then serving.get_deployment() to get the handle.
        """
        if name_attr in self._handle_cache:
            return self._handle_cache[name_attr]

        from colony.deployment_names import get_deployment_names
        from colony.distributed.ray_utils.serving import get_deployment

        names = get_deployment_names()
        deployment_name = getattr(names, name_attr)
        logger.info(
            "Resolving deployment: %s → %s (app=%s)",
            name_attr, deployment_name, self.app_name,
        )
        handle = get_deployment(self.app_name, deployment_name)
        self._handle_cache[name_attr] = handle
        return handle

    def get_session_manager(self) -> Any:
        """Get SessionManagerDeployment handle."""
        return self._get_handle("session_manager")

    def get_agent_system(self) -> Any:
        """Get AgentSystemDeployment handle."""
        return self._get_handle("agent_system")

    def get_vcm(self) -> Any:
        """Get VirtualContextManager handle."""
        return self._get_handle("vcm")

    def get_llm_cluster(self) -> Any:
        """Get LLM cluster handle."""
        return self._get_handle("llm_cluster")

    def get_tool_manager(self) -> Any:
        """Get ToolManager handle."""
        return self._get_handle("tool_manager")

    def get_deployment_handle(self, app_name: str, deployment_name: str) -> Any:
        """Get a deployment handle by explicit app/deployment name.

        Prefer the typed methods (get_session_manager, get_agent_system, etc.)
        which resolve names through get_deployment_names(). This method is
        only for cases where the caller has a dynamic app_name/deployment_name.
        """
        cache_key = f"{app_name}/{deployment_name}"
        if cache_key not in self._handle_cache:
            from colony.distributed.ray_utils.serving import get_deployment
            self._handle_cache[cache_key] = get_deployment(app_name, deployment_name)
        return self._handle_cache[cache_key]

    async def init_observability(
        self,
        pg_host: str, pg_port: int, pg_user: str, pg_password: str, pg_database: str,
        kafka_bootstrap: str,
    ) -> None:
        """Initialize PostgreSQL pool, schema, and Kafka→PG span consumer."""
        self._kafka_bootstrap = kafka_bootstrap
        try:
            import asyncpg
            self._db_pool = await asyncpg.create_pool(
                host=pg_host, port=pg_port,
                user=pg_user, password=pg_password,
                database=pg_database,
                min_size=2, max_size=10,
            )
            # Ensure spans table exists
            from colony.agents.observability.migrations import ensure_schema
            await ensure_schema(self._db_pool)

            # Create query store
            from colony.agents.observability.store import SpanQueryStore
            self._span_query_store = SpanQueryStore(self._db_pool)

            # Start Kafka→PG consumer
            from colony.agents.observability.consumer import SpanConsumer
            self._span_consumer = SpanConsumer(
                kafka_bootstrap=kafka_bootstrap,
                db_pool=self._db_pool,
            )
            await self._span_consumer.start()
            logger.info("Observability initialized (PG pool + Kafka consumer)")
        except Exception as e:
            logger.warning(f"Observability init failed: {e}. Traces will be unavailable.")

    def get_span_query_store(self) -> Any:
        """Get the SpanQueryStore for trace queries."""
        return self._span_query_store

    @property
    def db_pool(self) -> Any:
        """Get the asyncpg connection pool."""
        return self._db_pool

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

    async def query_prometheus_range(
        self, query: str, start: float, end: float, step: str = "15s",
    ) -> dict[str, Any]:
        """Run a PromQL range query."""
        if not self._http_client:
            return {"status": "error", "error": "No HTTP client"}
        try:
            resp = await self._http_client.get(
                f"{self.prometheus_url}/api/v1/query_range",
                params={"query": query, "start": start, "end": end, "step": step},
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
