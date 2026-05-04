"""Runtime API for the dashboard backend.

The frozen dataclass is the surface FastAPI consumes. Source-of-truth fields
live in the typed :class:`WebUIConfig` (env-bound + tier-aware); ``from_env``
materialises the dataclass from there, preserving the existing call sites.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DashboardConfig:
    """Configuration for the Colony dashboard backend.

    The dashboard connects to Colony via Ray Client and uses deployment
    handles for all data access. No direct backend connections (Redis,
    PostgreSQL, etc.) are needed.
    """

    # Dashboard server
    host: str = "0.0.0.0"
    port: int = 8080

    # Ray cluster
    ray_client_address: str = "ray://ray-head:10001"

    # Ray services (HTTP APIs — these are Ray services, not Colony backends)
    ray_dashboard_url: str = "http://ray-head:8265"
    prometheus_url: str = "http://ray-head:9090"

    # Frontend
    static_dir: str | None = None  # Path to built frontend files

    # Observability (Kafka + PostgreSQL for traces)
    kafka_bootstrap: str = "kafka:9092"
    pg_host: str = "postgres"
    pg_port: int = 5432
    pg_user: str = "colony"
    pg_password: str = ""
    pg_database: str = "colony"

    @classmethod
    def from_env(cls) -> DashboardConfig:
        """Materialise from the registered :class:`WebUIConfig`."""
        from .configs import get_web_ui_config
        c = get_web_ui_config()
        return cls(
            host=c.host,
            port=c.port,
            ray_client_address=c.ray_client_address,
            ray_dashboard_url=c.ray_dashboard_url,
            prometheus_url=c.prometheus_url,
            static_dir=c.static_dir,
            kafka_bootstrap=c.kafka_bootstrap,
            pg_host=c.pg_host,
            pg_port=c.pg_port,
            pg_user=c.pg_user,
            pg_password=c.pg_password,
            pg_database=c.pg_database,
        )
