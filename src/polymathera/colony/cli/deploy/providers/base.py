"""Base classes for deployment providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class ProviderStatus(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class ServiceInfo:
    name: str
    status: ProviderStatus
    host: str | None = None
    port: int | None = None
    details: dict[str, str] = field(default_factory=dict)


class DeploymentProvider(ABC):
    """Base class for deployment providers (Compose, K8s, etc.)."""

    @abstractmethod
    async def up(
        self,
        build: bool = True,
        workers: int = 1,
        config_path: str | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> list[ServiceInfo]:
        """Start all infrastructure. Returns status of each service."""

    @abstractmethod
    async def down(self) -> None:
        """Stop and remove all infrastructure."""

    @abstractmethod
    async def status(self) -> list[ServiceInfo]:
        """Get status of all services."""

    @abstractmethod
    async def run(
        self,
        origin_url: str | None = None,
        local_repo: str | None = None,
        branch: str = "main",
        commit: str = "HEAD",
        config_path: str | None = None,
        extra_env: dict[str, str] | None = None,
        extra_args: list[str] | None = None,
    ) -> int:
        """Run polymath.py inside the cluster. Returns exit code."""

    @abstractmethod
    async def doctor(self) -> dict[str, bool]:
        """Check prerequisites. Returns {check_name: passed}."""
