"""Kind + KubeRay deployment provider (stub for future implementation)."""

from __future__ import annotations

from collections.abc import Callable

from ..config import DeployConfig
from .base import DeploymentProvider, ServiceInfo


class KindKubeRayProvider(DeploymentProvider):
    """Manages a local Ray cluster via Kind + KubeRay.

    Not yet implemented. Uses the same colony:local Docker image as the
    Compose provider, but orchestrates via Kubernetes for full autoscaling
    and production-parity testing.

    Workflow (when implemented):
        1. kind create cluster --config kind-config.yaml
        2. docker build + kind load docker-image colony:local
        3. helm install kuberay-operator
        4. kubectl apply -f ray-cluster-local.yaml
        5. kubectl port-forward for dashboard + client
    """

    def __init__(self, config: DeployConfig) -> None:
        self._config = config

    async def up(
        self,
        build: bool = True,
        workers: int = 1,
        on_status: Callable[[str], None] | None = None,
    ) -> list[ServiceInfo]:
        raise NotImplementedError(
            "Kind + KubeRay deployment is not yet implemented. "
            "Use Docker Compose mode (default) for now."
        )

    async def down(self) -> None:
        raise NotImplementedError("Kind + KubeRay deployment is not yet implemented.")

    async def status(self) -> list[ServiceInfo]:
        raise NotImplementedError("Kind + KubeRay deployment is not yet implemented.")

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
        raise NotImplementedError("Kind + KubeRay deployment is not yet implemented.")

    async def doctor(self) -> dict[str, bool]:
        raise NotImplementedError("Kind + KubeRay deployment is not yet implemented.")
