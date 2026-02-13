"""DeploymentManager — orchestrates infrastructure providers."""

from __future__ import annotations

from .config import DeployConfig
from .providers.base import DeploymentProvider, ServiceInfo
from .providers.compose import DockerComposeProvider


class DeploymentManager:
    """Orchestrates deployment via the configured provider."""

    def __init__(self, config: DeployConfig | None = None) -> None:
        self._config = config or DeployConfig()
        self._provider = self._create_provider()

    def _create_provider(self) -> DeploymentProvider:
        if self._config.mode == "compose":
            return DockerComposeProvider(self._config)
        elif self._config.mode == "k8s":
            from .providers.k8s import KindKubeRayProvider
            return KindKubeRayProvider(self._config)
        raise ValueError(f"Unknown deployment mode: {self._config.mode}")

    async def up(self, build: bool = True, workers: int = 1) -> list[ServiceInfo]:
        return await self._provider.up(build=build, workers=workers)

    async def down(self) -> None:
        return await self._provider.down()

    async def status(self) -> list[ServiceInfo]:
        return await self._provider.status()

    async def run(
        self,
        codebase_path: str,
        config_path: str | None = None,
        extra_env: dict[str, str] | None = None,
        extra_args: list[str] | None = None,
    ) -> int:
        return await self._provider.run(
            codebase_path=codebase_path,
            config_path=config_path,
            extra_env=extra_env,
            extra_args=extra_args,
        )

    async def doctor(self) -> dict[str, bool]:
        return await self._provider.doctor()
