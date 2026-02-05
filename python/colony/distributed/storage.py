from __future__ import annotations
import asyncio
import logging
from pathlib import Path
from typing import Any

from multipledispatch import dispatch
from pydantic import HttpUrl, UUID4

from ..agents.types import ActionPlan
from ..schema.exceptions import AuthorizationError
from ..schema.base_types import RepoId
from ..schema.vmr import VirtualMonorepo, Repository
from .auth import AuthToken, DistributedAuthManager
from .stores.databases import RelationalStorage
from .stores.files import FileStorage, ScalableDistributedFileSystem
from .stores.git import GitFileStorage
from .stores.json import JsonStorage
from .stores.objects import ObjectStorage
from .configs import StorageConfig



# TODO: Create all the necessary S3 buckets, DynamoDB tables, EFS
# instances beforehand and ensure that your AWS credentials have
# the necessary permissions to access all of these instances.



logger = logging.getLogger(__name__)



class Storage:
    """
    PROMPT:
    Implement this class and make it production ready. Implement proper error handling, retries, and handle edge cases
    and ensure no messages or data are lost. This class is used by Polymathera agent microservices for various needs:
    persisting and updating VMR structures in a database, providing a graph database to store agent knowledge bases,
    distributed file system to clone and share millions of git repositories. Perhaps splitting the class into multiple
    classes each for a different storage type is a better approach.
    """

    def __init__(self, config: StorageConfig | None = None):
        self.config: StorageConfig | None = config
        self.object_storage: ObjectStorage | None = None
        self.relational_storage: RelationalStorage | None = None
        self.distributed_file_system: ScalableDistributedFileSystem | None = None
        self.json_storage: JsonStorage | None = None
        self.file_storage: FileStorage | None = None
        self.git_storage: GitFileStorage | None = None
        self.auth_manager: DistributedAuthManager | None = None

    async def initialize(self) -> None:
        """Initialize all storage subsystems with per-component timing logs.

        The helper _run_timed logs *start*, *done* (with elapsed seconds) or
        *fail* for every underlying init coroutine.  This lets a single test
        run show exactly which subsystem hangs.
        """

        import time

        async def _run_timed(name: str, coro: "Coroutine[Any, Any, Any]"):  # type: ignore[name-defined]
            start = time.time()
            logger.info("Storage.initialize: %s start", name)
            try:
                result = await coro
                logger.info("Storage.initialize: %s done %.2fs", name, time.time() - start)
                return result
            except Exception as exc:
                logger.error("Storage.initialize: %s failed after %.2fs – %s", name, time.time() - start, exc)
                raise

        self.config = await StorageConfig.check_or_get_component(self.config)
        self.object_storage = ObjectStorage(self.config.object_storage)
        self.relational_storage = RelationalStorage(self.config.relational_storage)
        self.distributed_file_system = ScalableDistributedFileSystem(self.config.distributed_file_system)
        self.json_storage = JsonStorage(self.config.json_storage)
        self.file_storage = FileStorage(self.distributed_file_system, self.config.file_storage)
        self.git_storage = GitFileStorage(self.distributed_file_system, self.config.git_storage)
        self.auth_manager = DistributedAuthManager(self.config.auth_config)

        init_tasks = [
            _run_timed("object_storage", self.object_storage.initialize()),
            _run_timed("relational_storage", self.relational_storage.initialize()),
            _run_timed("distributed_file_system", self.distributed_file_system.initialize()),
            _run_timed("json_storage", self.json_storage.initialize()),
            _run_timed("file_storage", self.file_storage.initialize()),
            _run_timed("git_storage", self.git_storage.initialize()),
            _run_timed("auth_manager", self.auth_manager.initialize()),
        ]

        logger.info("Storage.initialize: awaiting %d init tasks", len(init_tasks))
        await asyncio.gather(*init_tasks)
        logger.info("Storage.initialize: all subsystems initialized")

    async def cleanup(self) -> None:
        """Cleanup background tasks and resources"""
        from ..utils import cleanup_dynamic_asyncio_tasks

        try:
            await cleanup_dynamic_asyncio_tasks(self, raise_exceptions=False)
            if self.object_storage:
                await self.object_storage.cleanup()
            if self.relational_storage:
                await self.relational_storage.cleanup()
            if self.distributed_file_system:
                await self.distributed_file_system.cleanup()
            if self.json_storage:
                await self.json_storage.cleanup()
            if self.file_storage:
                await self.file_storage.cleanup()
            if self.git_storage:
                await self.git_storage.cleanup()
            if self.auth_manager:
                await self.auth_manager.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up Storage tasks: {e}")

    async def join(self, *paths):
        return await self.distributed_file_system.join(*paths)

    async def exists(self, path):
        return await self.distributed_file_system.exists(path)

    async def read_file(self, path: str | Path) -> bytes:
        return await self.distributed_file_system.read_file(path)

    async def read_compressed_file(self, path: str | Path) -> bytes:
        return await self.distributed_file_system.read_compressed_file(path)

    async def walk(self, path: str | Path):
        async for item in self.distributed_file_system.walk(path):
            yield item

    async def glob(self, path: str | Path, pattern: str = "*"):
        return await self.distributed_file_system.glob(path, pattern)

    async def write_file(self, path: str | Path, data: bytes):
        return await self.distributed_file_system.write_file(path, data)

    async def write_compressed_file(self, path: str | Path, data: bytes):
        return await self.distributed_file_system.write_compressed_file(path, data)

    async def store_object(self, bucket, key, data):
        return await self.object_storage.store_object(bucket, key, data)

    async def get_object(self, bucket, key):
        return await self.object_storage.get_object(bucket, key)

    async def add_repo_to_vmr(self, repo: Repository, vmr: VirtualMonorepo) -> bool:
        return await self.relational_storage.add_repo_to_vmr(repo, vmr)

    async def remove_repo_from_vmr(self, repo: Repository, vmr: VirtualMonorepo) -> bool:
        return await self.relational_storage.remove_repo_from_vmr(repo, vmr)

    async def update_repo(self, repo: Repository) -> bool:
        """Update repository metadata in the relational storage"""
        return await self.relational_storage.update_repo(repo)

    async def store_vmr(self, **vmr_data: Any) -> str:
        return await self.relational_storage.store_vmr(**vmr_data)

    async def update_vmr(self, vmr_id: str, **vmr_updates: Any):
        return await self.relational_storage.update_vmr(vmr_id, **vmr_updates)

    async def load_vmr(self, auth_token: AuthToken, vmr_id: str) -> VirtualMonorepo:
        """Load VMR with authorization check"""
        await self._verify_access(
            auth_token=auth_token,
            resource_type="vmr",
            resource_id=vmr_id,
            required_permission="read",
        )
        return await self.relational_storage.load_vmr(vmr_id)

    async def delete_vmr(self, vmr_id: str):
        return await self.relational_storage.delete_vmr(vmr_id)

    async def list_vmrs(self) -> list[VirtualMonorepo]:
        return await self.relational_storage.list_vmrs()

    async def cleanup_all_data(self) -> None:
        return await self.relational_storage.cleanup_all_data()

    async def update_vmr_status(self, vmr_id: str, status: str):
        return await self.relational_storage.update_vmr_status(vmr_id, status)

    async def get_vmr_status(self, vmr_id: str) -> str:
        return await self.relational_storage.get_vmr_status(vmr_id)

    async def store_action_plan(self, vmr_id: str, action_plan: ActionPlan):
        return await self.update_vmr(vmr_id, action_plan=action_plan.to_dict())

    async def load_action_plan(self, auth_token: AuthToken, vmr_id: str) -> ActionPlan:
        vmr = await self.load_vmr(auth_token, vmr_id)
        return vmr.action_plan # TODO: This can be made more efficient by loading the action plan from the database directly.

    async def save_json(self, data: dict[str, Any], metadata: dict[str, Any]):
        # TODO: Add compression (msgpack, zstd, etc.), error handling and retries
        return await self.json_storage.save(data, metadata)

    async def load_json(self, metadata: dict[str, Any]) -> dict[str, Any]:
        # TODO: Add compression (msgpack, zstd, etc.), error handling and retries
        return await self.json_storage.load(metadata)

    async def store_code_stats(
        self,
        auth_token: AuthToken,
        vmr_id: str,
        repo_id: RepoId,
        code_stats: dict[str, Any],
    ):
        """Store code stats with authorization checks for both VMR and repo"""
        # Check VMR access
        await self._verify_access(
            auth_token=auth_token,
            resource_type="vmr",
            resource_id=str(vmr_id),
            required_permission="write",
        )

        # Check repository access
        await self._verify_access(
            auth_token=auth_token,
            resource_type="repository",
            resource_id=str(repo_id),
            required_permission="write",
        )

        return await self.json_storage.save(
            code_stats, metadata={"vmr_id": str(vmr_id), "repo_id": str(repo_id)}
        )

    async def load_code_stats(self, vmr_id: str, repo_id: str):
        return await self.json_storage.load(
            metadata={"vmr_id": str(vmr_id), "repo_id": str(repo_id)}
        )

    def _validate_repo_url(self, origin_url: str):
        try:
            HttpUrl(origin_url)
        except ValueError as e:
            raise ValueError(f"Invalid repository URL: {origin_url}") from e

    async def git_operation(
        self, auth_token: AuthToken, origin_url: str, operation: str, *args, **kwargs
    ) -> Any:
        # TODO: Add authorization check
        return await self.git_storage.git_operation(
            origin_url, operation, *args, **kwargs
        )

    async def clone_or_retrieve_repository(
        self, *, auth_token: AuthToken, origin_url: str, branch: str, commit: str, vmr_id: str | UUID4
    ) -> Path:
        """Wrapper that delegates to GitFileStorage and emits detailed timing logs.

        Extra logging helps diagnose long-running operations during integration
        tests where the clone step may hang.  The commit and branch names are
        included so we can correlate with the test case.  No behavioural change
        - only instrumentation.
        """
        import time  # Local import to avoid polluting module namespace at top

        self._validate_repo_url(origin_url)
        # TODO: Add authorization check

        start_ts = time.time()
        logger.info(
            "Storage.clone_or_retrieve_repository: start origin=%s branch=%s commit=%s vmr_id=%s",
            origin_url,
            branch,
            commit,
            vmr_id,
        )

        # Delegate to GitFileStorage
        repo_path = await self.git_storage.clone_or_retrieve_repository(
            origin_url=origin_url,
            branch=branch,
            commit=commit,
            vmr_id=vmr_id,
        )

        elapsed = time.time() - start_ts
        logger.info(
            "Storage.clone_or_retrieve_repository: done origin=%s elapsed=%.2fs path=%s",
            origin_url,
            elapsed,
            repo_path,
        )

        return repo_path

    async def bulk_clone_repositories(
        self, auth_token: AuthToken, origin_urls: list[str]
    ):
        for origin_url in origin_urls:
            self._validate_repo_url(origin_url)
        # TODO: Add authorization check
        return await self.git_storage.bulk_clone_repositories(origin_urls)

    async def get_repository_path(self, auth_token: AuthToken, origin_url: str) -> str:
        self._validate_repo_url(origin_url)
        # TODO: Add authorization check
        return await self.git_storage.get_repository_path(origin_url)

    async def escalate_failure(self, error_message):
        logger.critical(f"Unrecoverable failure in Storage: {error_message}")
        # Here you would implement your alerting logic, e.g., sending an email or a message to a dedicated channel

    async def get_size_mb(self, path: str) -> float:
        return await self.distributed_file_system.get_size_mb(path)

    @dispatch(AuthToken, str, str, str, str)
    async def release_reference(
        self,
        auth_token: AuthToken,
        origin_url: str,
        branch: str,
        commit: str,
        vmr_id: str,
    ):
        self._validate_repo_url(origin_url)
        # TODO: Add authorization check
        return await self.git_storage.release_reference(
            origin_url, branch, commit, vmr_id
        )

    async def _verify_access(
        self,
        auth_token: AuthToken,
        resource_type: str,
        resource_id: str,
        required_permission: str,
    ):
        """Verify access with proper error handling and auditing"""
        if not self.config.enable_auth:
            return

        try:
            has_permission = await self.auth_manager.verify_permission(
                auth_token=auth_token,
                resource_type=resource_type,
                resource_id=resource_id,
                required_permission=required_permission,
            )

            if not has_permission:
                raise AuthorizationError(
                    f"Access denied: {auth_token.actor_id} does not have "
                    f"{required_permission} permission for {resource_type}:{resource_id}"
                )

        except AuthorizationError:
            raise
        except Exception as e:
            logger.error(f"Authorization check failed: {e}")
            raise AuthorizationError("Authorization check failed due to system error")
