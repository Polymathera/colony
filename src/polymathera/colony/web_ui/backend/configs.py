"""Typed config slot for the dashboard (web-UI) backend.

Replaces the ad-hoc ``os.environ`` reads in ``DashboardConfig.from_env``;
``DashboardConfig`` itself stays as the runtime API consumed by the FastAPI
app and now derives its values from this :class:`WebUIConfig`.
"""

from __future__ import annotations

from pydantic import Field

from ...distributed.config import (
    ConfigComponent,
    Mutability,
    Tier,
    register_polymathera_config,
    tier_metadata,
)


@register_polymathera_config(path="web_ui")
class WebUIConfig(ConfigComponent):
    """Settings for the dashboard backend service.

    All fields env-bound to preserve the legacy bootstrap path: the Docker
    `.env` keeps populating these, and operator YAML can override per-deploy.
    """

    host: str = Field(
        default="0.0.0.0",
        json_schema_extra={
            "env": "DASHBOARD_HOST", "optional": True,
            **tier_metadata(tier=Tier.L1_OPERATOR),
        },
    )
    port: int = Field(
        default=8080,
        json_schema_extra={
            "env": "DASHBOARD_PORT", "optional": True,
            **tier_metadata(tier=Tier.L1_OPERATOR),
        },
    )
    ray_client_address: str = Field(
        default="ray://ray-head:10001",
        json_schema_extra={
            "env": "RAY_CLIENT_ADDRESS", "optional": True,
            **tier_metadata(tier=Tier.L1_OPERATOR),
        },
    )
    ray_dashboard_url: str = Field(
        default="http://ray-head:8265",
        json_schema_extra={
            "env": "RAY_DASHBOARD_URL", "optional": True,
            **tier_metadata(tier=Tier.L1_OPERATOR),
        },
    )
    prometheus_url: str = Field(
        default="http://ray-head:9090",
        json_schema_extra={
            "env": "PROMETHEUS_URL", "optional": True,
            **tier_metadata(tier=Tier.L1_OPERATOR),
        },
    )
    static_dir: str | None = Field(
        default=None,
        json_schema_extra={
            "env": "DASHBOARD_STATIC_DIR", "optional": True,
            **tier_metadata(tier=Tier.L1_OPERATOR),
        },
    )
    kafka_bootstrap: str = Field(
        default="kafka:9092",
        json_schema_extra={
            "env": "KAFKA_BOOTSTRAP", "optional": True,
            **tier_metadata(tier=Tier.L1_OPERATOR, mutability=Mutability.RELOADABLE),
        },
    )
    pg_host: str = Field(
        default="postgres",
        json_schema_extra={
            "env": "RDS_HOST", "optional": True,
            **tier_metadata(tier=Tier.L1_OPERATOR),
        },
    )
    pg_port: int = Field(
        default=5432,
        json_schema_extra={
            "env": "RDS_PORT", "optional": True,
            **tier_metadata(tier=Tier.L1_OPERATOR),
        },
    )
    pg_user: str = Field(
        default="colony",
        json_schema_extra={
            "env": "RDS_USER", "optional": True,
            **tier_metadata(tier=Tier.L1_OPERATOR),
        },
    )
    pg_password: str = Field(
        default="",
        json_schema_extra={
            "env": "RDS_PASSWORD", "optional": True,
            **tier_metadata(tier=Tier.L1_OPERATOR),
        },
    )
    pg_database: str = Field(
        default="colony",
        json_schema_extra={
            "env": "RDS_DB_NAME", "optional": True,
            **tier_metadata(tier=Tier.L1_OPERATOR),
        },
    )


def get_web_ui_config() -> WebUIConfig:
    """Sync fetch of the registered :class:`WebUIConfig` (defaults if uninit)."""
    from ...distributed.config import get_component_or_default
    return get_component_or_default("web_ui", WebUIConfig)


__all__ = ("WebUIConfig", "get_web_ui_config")
