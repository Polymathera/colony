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
        bake: bool = False,
    ) -> list[ServiceInfo]:
        """Start all infrastructure. Returns status of each service.

        ``bake=True`` snapshots the resolved L1-G ``cluster.extensions.packages``
        into a pinned ``colony-local:<hash>`` image so the container-start
        hook does not need to pip-install on every boot. Default fast path
        (``bake=False``) installs into a persistent overlay volume at
        container start and re-uses it across restarts via the resolved
        hash.
        """

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

    @abstractmethod
    async def image_info(self) -> dict[str, list[str]]:
        """Inspect the running cluster's images: which polymathera-* packages
        are baked into the runtime image vs. overlay-installed at container
        start. Returns ``{"baked": [pkg_lines], "overlay": [pkg_lines]}``.
        Each entry is a ``"name version"`` line as ``pip list`` prints it.
        """

    @abstractmethod
    async def image_build(
        self,
        config_path: str | None = None,
        bake: bool = False,
        on_status: Callable[[str], None] | None = None,
    ) -> str:
        """Build the base + runtime images (and optionally a bake image)
        WITHOUT bringing the cluster up. Returns the final image tag the
        cluster would run."""
