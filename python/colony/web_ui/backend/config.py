"""Dashboard configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DashboardConfig:
    """Configuration for the Colony dashboard backend."""

    # Dashboard server
    host: str = "0.0.0.0"
    port: int = 8080

    # Ray cluster
    ray_address: str = "ray-head:6379"
    ray_client_address: str = "ray://ray-head:10001"

    # Redis
    redis_host: str = "redis"
    redis_port: int = 6379

    # PostgreSQL
    rds_host: str = "postgres"
    rds_port: int = 5432
    rds_user: str = "colony"
    rds_password: str = "colony_dev"
    rds_db_name: str = "colony"

    # Ray services (inside Docker network)
    ray_dashboard_url: str = "http://ray-head:8265"
    prometheus_url: str = "http://ray-head:9090"

    # Frontend
    static_dir: str | None = None  # Path to built frontend files

    @classmethod
    def from_env(cls) -> DashboardConfig:
        """Load configuration from environment variables."""
        return cls(
            host=os.environ.get("DASHBOARD_HOST", "0.0.0.0"),
            port=int(os.environ.get("DASHBOARD_PORT", "8080")),
            ray_address=os.environ.get("RAY_ADDRESS", "ray-head:6379"),
            ray_client_address=os.environ.get("RAY_CLIENT_ADDRESS", "ray://ray-head:10001"),
            redis_host=os.environ.get("REDIS_HOST", "redis"),
            redis_port=int(os.environ.get("REDIS_PORT", "6379")),
            rds_host=os.environ.get("RDS_HOST", "postgres"),
            rds_port=int(os.environ.get("RDS_PORT", "5432")),
            rds_user=os.environ.get("RDS_USER", "colony"),
            rds_password=os.environ.get("RDS_PASSWORD", "colony_dev"),
            rds_db_name=os.environ.get("RDS_DB_NAME", "colony"),
            ray_dashboard_url=os.environ.get("RAY_DASHBOARD_URL", "http://ray-head:8265"),
            prometheus_url=os.environ.get("PROMETHEUS_URL", "http://ray-head:9090"),
            static_dir=os.environ.get("DASHBOARD_STATIC_DIR"),
        )
