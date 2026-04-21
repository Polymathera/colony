"""Pydantic response models for the dashboard API.

Thin wrappers and summaries over Colony's internal models.
Colony models are reused directly where possible; these models exist
for cases where we need a simplified view or aggregation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthStatus(BaseModel):
    """Overall cluster health summary."""

    ray_connected: bool = False
    redis_connected: bool = False
    deployments_ready: bool = False
    ray_cluster_status: str = "unknown"
    node_count: int = 0


class RedisInfo(BaseModel):
    """Subset of Redis INFO for the dashboard."""

    connected_clients: int = 0
    used_memory_human: str = ""
    total_commands_processed: int = 0
    keyspace_hits: int = 0
    keyspace_misses: int = 0
    uptime_in_seconds: int = 0


class DeploymentSummary(BaseModel):
    """Summary of a serving deployment."""

    app_name: str
    deployment_name: str
    proxy_actor_name: str


class ApplicationSummary(BaseModel):
    """Summary of a serving application."""

    app_name: str
    created_at: float
    deployments: list[DeploymentSummary] = Field(default_factory=list)


class AgentSummary(BaseModel):
    """Lightweight agent info for list views."""

    agent_id: str
    agent_type: str = ""
    state: str = ""
    capabilities: list[str] = Field(default_factory=list)


class SessionSummary(BaseModel):
    """Lightweight session info for list views."""

    session_id: str
    tenant_id: str = ""
    colony_id: str = ""
    state: str = ""
    created_at: float = 0.0
    run_count: int = 0


class RunSummary(BaseModel):
    """Lightweight run info for list views."""

    run_id: str
    session_id: str = ""
    agent_id: str = ""
    colony_id: str = ""
    status: str = ""
    started_at: float | None = None
    completed_at: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class PageSummary(BaseModel):
    """Lightweight page info for the VCM tab."""

    page_id: str
    source: str = ""
    tokens: int = 0
    loaded: bool = False
    files: list[str] = Field(default_factory=list)


class AgentHierarchyNode(BaseModel):
    """Agent with parent-child relationship info for hierarchy view."""

    agent_id: str
    agent_type: str = ""
    state: str = ""
    role: str | None = None
    parent_agent_id: str | None = None
    capability_names: list[str] = Field(default_factory=list)
    bound_pages: list[str] = Field(default_factory=list)
    tenant_id: str = ""
    colony_id: str = ""


class VCMStats(BaseModel):
    """VCM statistics summary."""

    total_pages: int = 0
    loaded_pages: int = 0
    page_groups: int = 0
    pending_faults: int = 0
