"""Dashboard configuration loaded from environment variables."""

from __future__ import annotations

import os
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
    pg_password: str = "colony_dev"
    pg_database: str = "colony"

    @classmethod
    def from_env(cls) -> DashboardConfig:
        """Load configuration from environment variables."""
        return cls(
            host=os.environ.get("DASHBOARD_HOST", "0.0.0.0"),
            port=int(os.environ.get("DASHBOARD_PORT", "8080")),
            ray_client_address=os.environ.get("RAY_CLIENT_ADDRESS", "ray://ray-head:10001"),
            ray_dashboard_url=os.environ.get("RAY_DASHBOARD_URL", "http://ray-head:8265"),
            prometheus_url=os.environ.get("PROMETHEUS_URL", "http://ray-head:9090"),
            static_dir=os.environ.get("DASHBOARD_STATIC_DIR"),
            kafka_bootstrap=os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092"),
            pg_host=os.environ.get("RDS_HOST", "postgres"),
            pg_port=int(os.environ.get("RDS_PORT", "5432")),
            pg_user=os.environ.get("RDS_USER", "colony"),
            pg_password=os.environ.get("RDS_PASSWORD", "colony_dev"),
            pg_database=os.environ.get("RDS_DB_NAME", "colony"),
        )
