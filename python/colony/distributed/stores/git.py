import asyncio
import hashlib
import json
import logging
import os
import tarfile
import tempfile
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field, UUID4
import shutil

import boto3
from botocore.exceptions import ClientError
from git import Repo  # Using GitPython
from git.exc import GitError, GitCommandError
from multipledispatch import dispatch

from ...utils import create_dynamic_asyncio_task, call_async
from ...utils.retry import standard_retry
from .files import FileSystemInterface
from ...distributed import get_polymathera
from ..configs import GitCacheManagerConfig, GitColdStorageConfig, GitFileStorageConfig
from ...schema.base_types import RepoId
from ...metrics.common import BaseMetricsMonitor

logger = logging.getLogger(__name__)


def build_repo_cache_key(
    origin_url: str, operation: str | None = None, *args, **kwargs
) -> str:
    """Build cache key for repository operations"""
    if operation:
        return f"{origin_url}:{operation}:{json.dumps(args)}:{json.dumps(kwargs)}"
    return origin_url


def get_replica_id(origin_url: str, branch: str, commit: str) -> str:
    """Generate a unique hash based on origin_url, branch, and commit"""
    return hashlib.sha256(
        f"{origin_url}_{branch}_{commit}".encode()
    ).hexdigest()[:16]


class GitCacheManager:

    """
    Manages caching for git operations using a distributed cache.
    Provides a clean interface for git-specific caching needs.

    Key features:
    1. Uses a distributed cache for consistent caching behavior
    2. Separates different types of git caches with namespaces
    3. Supports metadata and TTL configuration
    4. Handles locking for concurrent operations
    5. Manages reference counting for repositories
    """

    def __init__(self, config: GitCacheManagerConfig | None = None):
        self.config: GitCacheManagerConfig | None = config
        # Initialize different caches for different purposes
        self.clone_cache = None
        self.operations_cache = None
        self.references_cache = None
        self.locks_cache = None
        self.default_ttl = None
        self.lock_ttl = None

    async def initialize(self):
        self.config = await GitCacheManagerConfig.check_or_get_component(self.config)
        self.default_ttl = timedelta(seconds=self.config.cache_ttl)
        self.lock_ttl = timedelta(seconds=self.config.lock_ttl)
        self.clone_cache = await get_polymathera().create_distributed_simple_cache(
            # bool,
            namespace="git:clones",  # TODO: Scope is global to all VMRs?
            config=self.config.clone_cache_config,
        )
        self.operations_cache = await get_polymathera().create_distributed_simple_cache(
            # Any,
            namespace="git:operations",  # TODO: Scope is global to all VMRs?
            config=self.config.operations_cache_config,
        )
        self.references_cache = await get_polymathera().create_distributed_simple_cache(
            # set[str],
            namespace="git:references",  # TODO: Scope is global to all VMRs?
            config=self.config.references_cache_config,
        )
        self.locks_cache = await get_polymathera().create_distributed_simple_cache(
            # str,
            namespace="git:locks",  # TODO: Scope is global to all VMRs?
            config=self.config.locks_cache_config,
        )

    async def is_cloned(self, origin_url: str) -> bool:
        """Check if repository is marked as cloned"""
        key = build_repo_cache_key(origin_url)
        return await self.clone_cache.exists(key)

    async def mark_as_cloned(self, origin_url: str, ttl: timedelta | None = None):
        """Mark repository as successfully cloned"""
        key = build_repo_cache_key(origin_url)
        await self.clone_cache.set(key, True, ttl=ttl or self.default_ttl)

    async def unmark_as_cloned(self, origin_url: str):
        """Unmark repository as cloned"""
        key = build_repo_cache_key(origin_url)
        await self.clone_cache.delete(key)

    async def get_cached_operation(
        self, origin_url: str, operation: str, *args, **kwargs
    ) -> Any:
        """Get cached git operation result"""
        key = build_repo_cache_key(origin_url, operation, *args, **kwargs)
        return await self.operations_cache.get(key)

    async def cache_operation(
        self,
        origin_url: str,
        operation: str,
        result: Any,
        *args,
        ttl: timedelta | None = None,
        **kwargs,
    ):
        """Cache git operation result"""
        key = build_repo_cache_key(origin_url, operation, *args, **kwargs)
        await self.operations_cache.set(key, result, ttl=ttl or self.default_ttl)

    async def add_reference(self, repo_id: RepoId, vmr_id: str):
        """Add VMR reference to repository"""
        key = build_repo_cache_key(repo_id)
        await self.references_cache.add_to_set(key, vmr_id)

    async def remove_reference(self, repo_id: RepoId, vmr_id: str) -> int:
        """
        Remove VMR reference from repository
        Returns number of remaining references
        """
        key = build_repo_cache_key(repo_id)
        await self.references_cache.remove_from_set(key, vmr_id)
        return await self.references_cache.scard(key)

    async def get_reference_count(self, repo_id: RepoId) -> int:
        """Get number of references to repository"""
        key = build_repo_cache_key(repo_id)
        return await self.references_cache.get_set_cardinality(key)

    async def get_all_repo_ids(self) -> list[RepoId]:
        """Get all repository IDs"""
        try:
            all_keys = await self.references_cache.get_all_keys()

            # Redis may return keys as bytes.  Convert everything to str before
            # further processing to avoid "a bytes-like object is required, not 'str'"
            decoded_keys: list[str] = []
            for key in all_keys:
                if isinstance(key, bytes):
                    try:
                        key = key.decode()
                    except UnicodeDecodeError:
                        # Fallback – replace undecodable bytes to ensure we don't crash
                        logger.warning(f"Error decoding key: {key!s}")
                        key = key.decode(errors="replace")
                decoded_keys.append(key)

            return [RepoId(k.split(":")[1]) for k in decoded_keys]
        except Exception as e:
            logger.error(f"Error during get_all_repo_ids: {e!s}")
            return []

    @asynccontextmanager
    async def repository_lock(self, origin_url: str):
        """Lock repository for exclusive access with retry logic"""
        import asyncio
        key = build_repo_cache_key(origin_url, "lock")
        max_retries = 5
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Try to acquire lock atomically
                locked = await self.locks_cache.acquire_lock(key, self.lock_ttl)
                if locked:
                    try:
                        yield
                        return
                    finally:
                        # Release lock
                        await self.locks_cache.release_lock(key)
                else:
                    if attempt < max_retries - 1:
                        # Wait with exponential backoff before retrying
                        delay = base_delay * (2 ** attempt) + (0.1 * attempt)  # Add jitter
                        logger.info(f"Repository lock busy for {origin_url}, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                    else:
                        raise Exception(f"Could not acquire repository lock for {origin_url} after {max_retries} attempts")
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                # For other exceptions, also retry
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Lock acquisition failed for {origin_url}: {e}, retrying in {delay:.1f}s")
                await asyncio.sleep(delay)

    async def close(self):
        """Close all cache connections"""
        await self.clone_cache.close()
        await self.operations_cache.close()
        await self.references_cache.close()
        await self.locks_cache.close()


class GitColdStorage:
    def __init__(self, config: GitColdStorageConfig | None = None):
        self.config = config
        self.s3_client = None
        self.s3_buckets = None
        self.dynamodb = None
        self.repo_table = None
        self.bucket_count = None

    async def initialize(self) -> None:
        self.config = await GitColdStorageConfig.check_or_get_component(self.config)
        self.s3_buckets = self.config.s3_buckets
        self.bucket_count = len(self.s3_buckets)
        self.dynamodb = boto3.resource("dynamodb", region_name=self.config.aws_region)
        self.repo_table = self.dynamodb.Table(self.config.repo_metadata_table)
        self.s3_client = boto3.client("s3", region_name=self.config.aws_region)
        create_dynamic_asyncio_task(self, self._monitor_s3_usage())
        create_dynamic_asyncio_task(self, self._repair_metadata_periodic())

    async def cleanup(self) -> None:
        """Cleanup background tasks and resources"""
        from polymathera.utils import cleanup_dynamic_asyncio_tasks

        try:
            await cleanup_dynamic_asyncio_tasks(self, raise_exceptions=False)
        except Exception as e:
            logger.warning(f"Error cleaning up GitColdStorage tasks: {e}")

    async def repository_exists(self, origin_url: str) -> bool:
        """Return True only when **both** metadata and the S3 tar-ball exist.

        This guards against the drift you have observed where metadata was
        written but the upload never completed (or the object was deleted).
        In that case we treat the repo as *not* in cold-storage and allow the
        calling code to (re)upload; we also purge the stale metadata record so
        we don't keep making the same wrong decision.
        """

        metadata = await self._get_repo_metadata(origin_url)
        if not metadata:
            return False

        bucket = metadata.get("s3_bucket")
        key    = metadata.get("s3_key")

        if not bucket or not key:
            # Corrupt metadata – delete it and start fresh
            await self._store_repo_metadata({})  # overwrite/clear
            logger.warning(
                "GitColdStorage: corrupt metadata for %s - bucket/key missing; purged.",
                origin_url,
            )
            return False

        if not await self._s3_object_exists(bucket, key):
            logger.warning(
                "GitColdStorage: metadata for %s present but S3 object %s/%s missing – treating as absent.",
                origin_url,
                bucket,
                key,
            )
            # Remove stale record so we don't keep skipping uploads.
            await call_async(self.repo_table.delete_item, Key={"repo_id": metadata["repo_id"]})
            return False

        return True

    async def _s3_object_exists(self, bucket: str, key: str) -> bool:
        """Check object existence with a HEAD request (does not incur data transfer)."""
        try:
            await call_async(self.s3_client.head_object, Bucket=bucket, Key=key)
            return True
        except self.s3_client.exceptions.NoSuchKey:
            return False
        except Exception as exc:
            logger.warning("GitColdStorage: head_object failed for %s/%s: %s", bucket, key, exc)
            return False

    async def _get_repo_metadata(self, origin_url: str) -> dict[str, Any] | None:
        repo_id = self._make_ddb_repo_id(origin_url)
        response = await call_async(self.repo_table.get_item, Key={"repo_id": repo_id})
        return response.get("Item")

    async def _store_repo_metadata(self, metadata: dict[str, Any]):
        await call_async(self.repo_table.put_item, Item=metadata)

    async def get_all_repo_metadata(self) -> list[dict[str, Any]]:
        response = await call_async(self.repo_table.scan)
        return response.get("Items", [])

    async def store_repository(self, origin_url: str, repo_path: Path) -> None:
        try:
            bucket_name, s3_key = self._get_bucket_for_repo(origin_url)

            with tempfile.NamedTemporaryFile() as tmp:
                with tarfile.open(tmp.name, "w:gz") as tar:
                    tar.add(repo_path, arcname=os.path.basename(repo_path))
                await self._upload_to_s3(tmp.name, bucket_name, s3_key)

            # Store metadata in DynamoDB
            # Check https://gitpython.readthedocs.io/en/stable/tutorial.html
            repo = Repo(repo_path)
            # active_branch = repo.head # This will only work if the repository has a master branch
            active_branch = repo.active_branch
            head_commit = active_branch.commit.hexsha
            metadata = {
                "repo_id": self._make_ddb_repo_id(origin_url),
                "origin_url": origin_url,
                "s3_bucket": bucket_name,
                "s3_key": s3_key,
                "last_commit": head_commit,
                "active_branch": active_branch.name,
                "last_updated": datetime.now().isoformat(),
            }
            await self._store_repo_metadata(metadata)
        except Exception as e:
            logger.error(f"_________ store_repository: Error storing repository {origin_url}: {e!s}")
            raise

    async def retrieve_repository(
        self, origin_url: str, repo_path: Path, force_download: bool = False
    ):
        if os.path.exists(repo_path) and not force_download:
            return

        metadata = await self._get_repo_metadata(origin_url)
        if not metadata:
            raise FileNotFoundError(
                f"Repository {origin_url} not found in cold storage"
            )

        bucket_name, s3_key = metadata["s3_bucket"], metadata["s3_key"]
        # bucket_name, s3_key = self._get_bucket_for_repo(origin_url)
        with tempfile.NamedTemporaryFile() as tmp:
            await self._download_from_s3(bucket_name, s3_key, tmp.name)
            with tarfile.open(tmp.name, "r:gz") as tar:
                tar.extractall(path=os.path.dirname(repo_path))

    async def delete_repository(self, origin_url: str):
        bucket_name, s3_key = self._get_bucket_for_repo(origin_url)
        await call_async(self.s3_client.delete_object, Bucket=bucket_name, Key=s3_key)

    def _get_bucket_for_repo(self, origin_url: str):
        # Use a hash function to determine which bucket to use
        bucket_index = (
            int(hashlib.md5(origin_url.encode()).hexdigest(), 16) % self.bucket_count
        )
        s3_key = f"repos/{origin_url}.tar.gz"
        return self.s3_buckets[bucket_index], s3_key

    async def _download_from_s3(self, bucket_name: str, s3_key: str, fpath: str):
        await call_async(self.s3_client.download_file, bucket_name, s3_key, fpath)
        # response = await self.s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        # with open(fpath, 'wb') as data:
        #     data.write(response['Body'].read())

    async def _upload_to_s3(self, fpath: str, bucket_name: str, s3_key: str):
        await call_async(self.s3_client.upload_file, fpath, bucket_name, s3_key)

        # with open(fpath, 'rb') as data:
        #     await self.s3_client.put_object(
        #         Bucket=bucket_name,
        #         Key=s3_key,
        #         Body=data
        #     )

    async def _monitor_s3_usage(self):
        while True:
            for bucket in self.s3_buckets:
                response: dict[str, Any] = await call_async(self.s3_client.list_objects_v2, Bucket=bucket)
                total_size = sum(obj["Size"] for obj in response.get("Contents", []))
                total_objects = response.get("KeyCount", 0)

                # You can log this information or use it to make scaling decisions
                logger.info(
                    f"Bucket {bucket}: {total_objects} objects, {total_size} bytes"
                )

            await asyncio.sleep(self.config.s3_monitor_interval)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _make_ddb_repo_id(origin_url: str) -> str:
        """Generate a stable repository identifier.

        For now we simply reuse the *origin_url* as the primary key. This keeps
        the logic backward-compatible with any code that still expects to see
        the raw URL in the metadata while matching the DynamoDB table schema
        that uses *repo_id* as the partition key.
        """
        return origin_url

    # ------------------------------------------------------------------
    # Metadata-repair – public entry and periodic scheduler
    # ------------------------------------------------------------------

    async def repair_metadata_once(self, batch_size: int = 100) -> None:
        """Scan the metadata table and delete entries whose S3 objects are
        missing or whose row is malformed.  Runs quickly at start-up and can be
        invoked periodically.  *batch_size* controls the number of HEAD
        requests executed concurrently.
        """

        items = await self.get_all_repo_metadata()
        if not items:
            return

        sem = asyncio.Semaphore(batch_size)
        tasks: list[asyncio.Task] = []

        async def _check(item: dict[str, Any]):
            async with sem:
                bucket = item.get("s3_bucket")
                key = item.get("s3_key")
                repo_id = item.get("repo_id")

                if not bucket or not key or not repo_id:
                    await call_async(self.repo_table.delete_item, Key={"repo_id": repo_id})
                    logger.warning("Repair-metadata: removed corrupt row %s", repo_id)
                    return

                if not await self._s3_object_exists(bucket, key):
                    await call_async(self.repo_table.delete_item, Key={"repo_id": repo_id})
                    logger.warning("Repair-metadata: removed stale row %s (S3 object missing)", repo_id)

        # Create tasks and track them properly to avoid orphaned tasks
        for it in items:
            tasks.append(asyncio.create_task(_check(it)))

        # Wait for all tasks to complete and handle any exceptions
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.warning(f"Some repair metadata tasks failed: {e}")
            # Cancel any remaining tasks to prevent orphaned tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise

    async def _repair_metadata_periodic(self) -> None:
        """Run *repair_metadata_once* at a fixed cadence (default 6 h)."""
        # Launch background repair task that periodically reconciles DynamoDB
        # metadata with the actual S3 objects so we do not accumulate stale or
        # corrupt entries.
        interval = getattr(self.config, "repair_interval", 6 * 3600)
        while True:
            try:
                await self.repair_metadata_once()
            except Exception as exc:
                logger.warning("GitColdStorage: metadata repair pass failed: %s", exc)
            await asyncio.sleep(interval)


class GitCloneTransaction(BaseModel):
    origin_url: str
    branch: str | None = None
    commit: str | None = None
    vmr_id: str
    source_path: Path | None = None
    replica_path: Path | None = None
    source_exists: bool = False
    cold_storage_exists: bool = False
    replica_exists: bool = False
    is_cloned: bool = False
    retrieved_from_cold_storage: bool = False
    cloned_to_efs: bool = False
    stored_in_cold_storage: bool = False
    updated_cache: bool = False
    added_source_reference: bool = False
    added_replica_reference: bool = False
    created_replica_path: bool = False
    checked_out_branch: bool = False
    checked_out_commit: bool = False


class GitFileStorageMetricsMonitor(BaseMetricsMonitor):
    """Prometheus metrics for GitFileStorage."""

    def __init__(self,
                 enable_http_server: bool = True,
                 service_name: str = "git_file_storage"):
        super().__init__(enable_http_server, service_name)

        self.logger.info(f"Initializing GitFileStorageMetricsMonitor instance {id(self)}...")

        self.clone_counter = self.get_or_create_counter(
            "repository_clones_total", "Total number of repository clones", ["origin_url"]
        )
        self.repo_size_gauge = self.get_or_create_gauge(
            "repository_size_bytes", "Repository size in bytes", ["origin_url"]
        )
        self.operation_latency = self.get_or_create_histogram(
            "git_operation_latency_seconds", "Latency of git operations", ["operation"]
        )



class GitFileStorage:
    """
    GitFileStorage provides git functionality using a FileSystemInterface for storage.
    It implements repository cloning, updating, and various git operations.

    Key features:
    1. Uses FileSystemInterface for file operations, allowing for different storage backends.
    2. Implements distributed caching of git operations.
    3. Provides bulk cloning of repositories.
    4. Implements periodic pruning of old branches and garbage collection.
    5. Uses S3 as cold storage for repositories.
    6. Implements error handling and retries for git operations.

    Cloning touches three separate subsystems (EFS, S3, Redis). If any of those steps fails,
    you must roll back the others or you end up with the inconsistencies.

    ──────────────────────────────────
    1 . Model a real "transaction"
    ──────────────────────────────────
    Use a context-manager so you get automatic rollback:

    ```python
    class GitCloneTransactionCM(GitCloneTransaction):
        async def __aenter__(self):          # start txn
            return self

        async def __aexit__(self, exc_type, exc, tb):    # either commit or rollback
            if exc_type:
                await self.rollback()
            return False      # propagate the exception

        async def rollback(self):
            try:
                if self.created_replica and await fs.exists(self.created_replica):
                    await fs.delete(self.created_replica)
                if self.added_replica_ref:
                    await cache.remove_reference(self.replica_cache_key, self.vmr_id)
                ...
            except Exception as e:
                logger.warning("Rollback failed: %s", e)
    ```

    Your cloning function becomes:

    ```python
    async def clone_or_retrieve_repository(...):
        async with GitCloneTransactionCM(origin_url, branch, commit, vmr_id) as tx:
            await self._fast_path_if_replica_exists(...)

            # STEP-1  ensure source exists (EFS or cold storage or fresh clone)
            if not await fs.exists(source_path):
                ...

            # STEP-2  update caches
            ...

            # STEP-3  create / refresh replica
            if not await fs.exists(replica_path):
                ...
                tx.created_replica = replica_path

            # STEP-4 checkout branch/commit if required
            ...

            tx.commit()               # optional: mark "committed" to disable rollback
            return replica_path       # exiting CM without error ⇒ no rollback
    ```

    ──────────────────────────────────
    2 . Make each step idempotent
    ──────────────────────────────────
    If the function crashes mid-way it will retry (because of tenacity).
    Ensure you can safely call it again:

    1. **Clone to EFS** - use a temporary path + atomic rename (`clone_path_tmp → source_path`) to guarantee the directory is either complete or absent.
    2. **Upload to S3** - wrap in `if not s3.object_exists(...)` or use `ETag` to verify the object is complete.
    3. **Cache / reference bookkeeping** - do all Redis writes in a single pipeline (`MULTI/EXEC`) so you never see partially-applied state.

    ──────────────────────────────────
    3 . Structured logging
    ──────────────────────────────────
    Prefix every log line with the `repo_id`/`tx_id` so CloudWatch or Loki can group messages:

    ```python
    log = logger.bind(repo=origin_url, tx=tx_id)   # if you use structlog
    log.info("cloned_to_efs", duration=secs, size_mb=repo_size)
    ```

    ──────────────────────────────────
    4 . Consider a "saga" rather than strict rollback
    ──────────────────────────────────
    Clone + upload of multi-GB repos can be long-running; rolling it back just to retry costs time and money.
    An alternative is the **saga pattern**: keep partial results but mark them as "dirty" until the saga completes, and allow a later compensating step to finish the job.
    Example: upload to S3 fails → keep EFS clone but record a *needs_upload* flag; a background reconciler reads that flag and uploads later.

    ──────────────────────────────────
    5 . Guard against stale replicas
    ──────────────────────────────────
    If a replica already exists you fast-return, but you don't check that it's *up-to-date* with the requested branch/commit.
    Hash `(origin_url, branch, commit)` into `replica_id`; that already forces a new replica when *commit* changes,
    but if `commit=""` you might reuse an old replica that's still on yesterday's HEAD.
    Possible fix:
    - pull latest changes whenever `commit==''` and `branch==default_branch`
    - or include the timestamp of the HEAD commit in the hash

    ──────────────────────────────────
    6 . Surface higher-level consistency checks
    ──────────────────────────────────
    Write a small "doctor" script (or CloudWatch alarm) that:

    - walks `EFS:/github_*`; verifies `s3://cold-storage/...` has the same tarball
    - compares Redis `git:references:*` with actual files
    - can fix simple drifts automatically.

    Apply those tweaks and the clone process will be resilient, debuggable, and fully consistent.
    """

    def __init__(self, file_system: FileSystemInterface, config: GitFileStorageConfig | None = None):
        self.config: GitFileStorageConfig | None = config
        self.file_system: FileSystemInterface | None = file_system
        self.namespace: str | None = None
        self.max_concurrent_clones: int | None = None
        self.prune_interval: int | None = None
        self.gc_interval: int | None = None
        self.root_path: Path | None = None
        self.repo_queue: deque[str] = deque()
        self.semaphore: asyncio.Semaphore | None = None
        self.cold_storage: GitColdStorage | None = None
        self.cache_manager: GitCacheManager | None = None
        self.metrics: GitFileStorageMetricsMonitor | None
        self._asyncio_tasks = []

    async def cleanup(self) -> None:
        """Cleanup background tasks and resources"""
        from polymathera.utils import cleanup_dynamic_asyncio_tasks

        try:
            await cleanup_dynamic_asyncio_tasks(self, raise_exceptions=False)
            await self.cold_storage.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up GitFileStorage tasks: {e}")

    async def get_root_path(self):
        if self.root_path is None:
            self.root_path = await self.file_system.get_root_path(self.namespace)
        return self.root_path

    async def initialize(self) -> None:
        self.config = await GitFileStorageConfig.check_or_get_component(self.config)
        self.namespace = self.config.namespace
        self.max_concurrent_clones = self.config.max_concurrent_clones
        self.prune_interval = self.config.prune_interval
        self.gc_interval = self.config.gc_interval
        self.semaphore = asyncio.Semaphore(self.max_concurrent_clones)
        self.cold_storage = GitColdStorage(self.config.cold_storage_config)
        self.cache_manager = GitCacheManager(self.config.cache_manager_config)
        self.metrics = GitFileStorageMetricsMonitor()

        await self.get_root_path()
        await asyncio.gather(*[
            self.cache_manager.initialize(),
            self.cold_storage.initialize(),
        ])
        create_dynamic_asyncio_task(self, self._monitor_code_bases())
        create_dynamic_asyncio_task(self, self._periodic_pruning())
        create_dynamic_asyncio_task(self, self._update_repositories())
        create_dynamic_asyncio_task(self, self._garbage_collection())
        # Guarantee that the metadata DD table is consistent with the actual S3 objects
        await self.cold_storage.repair_metadata_once()
        logger.info("Initialized git file storage")

    async def _monitor_code_bases(self):
        # TODO: Implement logic to monitor code bases for changes and update metadata accordingly.
        pass

    def _get_repo_path(self, origin_url: str) -> Path:
        return self.root_path / origin_url.replace(
            "https://github.com/", "github_"
        ).replace("/", "_")

    @asynccontextmanager
    async def _git_repo_lock(self, origin_url: str):
        async with self.cache_manager.repository_lock(origin_url):
            yield

    #@dispatch(str, str, str)
    @standard_retry(logger)
    async def clone_or_retrieve_repository(
        self, *, origin_url: str, branch: str, commit: str, vmr_id: str | UUID4 = "system"
    ) -> Path:
        """
        Benefits of using a hash function to generate the suffix:
        - Consistency: A hash function will always produce the same output for the same input, ensuring consistent naming across different systems and environments.
        - Security: Using a hash reduces the risk of potential security issues that might arise from maliciously crafted branch names or commit hashes.
        - Length: Hash values have a fixed length, which is beneficial for file systems that might have limitations on path lengths.
        - Uniqueness: A good hash function minimizes the chance of collisions, even with similar input strings.
        - Performance: For file systems with a large number of repositories, using a hash can lead to more efficient directory structures.
        """
        import time
        overall_start = time.time()
        vmr_id = str(vmr_id)
        # Generate a unique hash based on origin_url, branch, and commit
        replica_id = get_replica_id(origin_url, branch, commit)

        source_path = self._get_repo_path(origin_url)
        replica_path = Path(f"{source_path}_{replica_id}")

        logger.info(f"________ clone_or_retrieve_repository: source_path={source_path}, replica_path={replica_path}")

        tx = GitCloneTransaction(
            origin_url=origin_url,
            branch=branch,
            commit=commit,
            vmr_id=vmr_id,
        )

        # First, ensure that we don't have more than max_concurrent_clones clones running at the same time
        async with self.semaphore:
            logger.info(f"________ clone_or_retrieve_repository: <<<< semaphore acquired >>>>")
            # Second, ensure global mutual exclusion for the origin_url
            async with self._git_repo_lock(origin_url):
                try:
                    tx.replica_path = replica_path
                    tx.source_path = source_path
                    tx.replica_exists = await self.file_system.exists(replica_path)
                    tx.source_exists  = await self.file_system.exists(source_path)
                    tx.cold_storage_exists = await self.cold_storage.repository_exists(origin_url)
                    tx.is_cloned = await self.cache_manager.is_cloned(origin_url)
                    logger.info(f"________ clone_or_retrieve_repository: replica_exists={tx.replica_exists}, source_exists={tx.source_exists}, cold_storage_exists={tx.cold_storage_exists}")

                    if not tx.source_exists and tx.cold_storage_exists:
                        logger.info(f"________ clone_or_retrieve_repository: retrieving repository from cold storage")
                        await self.cold_storage.retrieve_repository(origin_url, source_path)
                        tx.retrieved_from_cold_storage = True
                        logger.info(f"________ clone_or_retrieve_repository: repository retrieved from cold storage")

                    if not tx.source_exists and not tx.cold_storage_exists:
                        # Clone the repository to EFS: Run git clone in a thread pool
                        # Shallow clone for speed in CI/test runs.
                        logger.info("________ clone_or_retrieve_repository: cloning repository to EFS")

                        def _clone_repo():
                            # Full clone required to ensure all commits exist locally.
                            Repo.clone_from(
                                origin_url,
                                str(source_path),
                                branch=None,  # Use remote's default branch instead of assuming 'master'
                                # depth=1, # shallow clone. TODO: When should we use this?
                                # single_branch=True,
                            )

                        await call_async(_clone_repo)
                        tx.cloned_to_efs = True
                        logger.info(f"________ clone_or_retrieve_repository: repository cloned to EFS")
                        logger.info(f"________ clone_or_retrieve_repository: adding source reference to cache")
                        await self.add_reference(origin_url, vmr_id)
                        tx.added_source_reference = True
                        logger.info(f"________ clone_or_retrieve_repository: reference added to cache")

                    if not tx.cold_storage_exists:
                        # Store in S3 cold storage
                        logger.info(f"________ clone_or_retrieve_repository: storing repository in cold storage")
                        await self.cold_storage.store_repository(origin_url, source_path)
                        tx.stored_in_cold_storage = True
                        logger.info(f"________ clone_or_retrieve_repository: repository stored in cold storage")

                    if not tx.is_cloned:
                        logger.info(f"________ clone_or_retrieve_repository: marking repository as cloned")
                        await self.cache_manager.mark_as_cloned(origin_url)
                        tx.updated_cache = True
                        logger.info(f"________ clone_or_retrieve_repository: repository marked as cloned")

                    if not tx.replica_exists:
                        logger.info(f"________ clone_or_retrieve_repository: creating replica path in EFS and copying source to it")
                        await call_async(
                            lambda: shutil.copytree(source_path, replica_path, dirs_exist_ok=True)
                        )
                        tx.created_replica_path = True
                        logger.info(f"________ clone_or_retrieve_repository: replica path created")
                        logger.info(f"________ clone_or_retrieve_repository: adding replica reference to cache")
                        await self.add_reference(
                            f"{origin_url}_{replica_id}", vmr_id
                        )  # Add a system reference for the replica
                        tx.added_replica_reference = True
                        logger.info(f"________ clone_or_retrieve_repository: replica reference added to cache")

                    # --------------------------------------------------
                    # Optional branch checkout
                    # --------------------------------------------------
                    repo = Repo(str(replica_path))
                    if branch:
                        try:
                            branch_names = {h.name for h in repo.branches}
                            if branch in branch_names:
                                logger.info(f"________ clone_or_retrieve_repository: checking out branch {branch}")
                                await call_async(
                                    lambda: repo.git.checkout(branch)
                                )
                                tx.checked_out_branch = True
                                logger.info(f"________ clone_or_retrieve_repository: branch {branch} checked out")
                            else:
                                # Get list of available branches for better error message
                                available_branches = [h.name for h in repo.branches]
                                remote_branches = [ref.name for ref in repo.remotes.origin.refs if not ref.name.endswith('/HEAD')]

                                error_msg = (
                                    f"Branch '{branch}' not found in repository {origin_url}. "
                                    f"Available local branches: {available_branches}. "
                                    f"Available remote branches: {remote_branches}. "
                                    f"This is likely a VMR configuration error - the repository may have changed its default branch. "
                                    f"Please update the VMR specification to use the correct branch name."
                                )
                                logger.error(error_msg)
                                raise ValueError(error_msg)
                        except Exception as exc:  # git.GitCommandError or others
                            logger.warning(f"Failed to checkout branch '{branch}': {exc} - skipping")

                    # --------------------------------------------------
                    # Optional commit checkout (only if commit exists)
                    # --------------------------------------------------
                    if commit and commit != "latest":
                        try:
                            # Verify commit exists – rev-parse will throw if unknown
                            repo.git.rev_parse("--verify", commit)
                            logger.info(f"________ clone_or_retrieve_repository: checking out commit {commit}")
                            await call_async(
                                lambda: repo.git.checkout(commit, force=True)
                            )
                            tx.checked_out_commit = True
                            logger.info(f"________ clone_or_retrieve_repository: commit {commit} checked out")
                        except GitCommandError:
                            logger.warning(f"Commit '{commit}' not found in repo - skipping checkout")

                    # Increment the counter only if the transaction is successful and the repository is cloned
                    if not tx.source_exists and not tx.cold_storage_exists:
                        assert tx.cloned_to_efs
                        self.metrics.clone_counter.labels(origin_url=origin_url).inc()
                        logger.info(f"________ clone_or_retrieve_repository: clone counter incremented")

                    # Whether or not we checked out a specific commit, at this
                    # point the replica is ready for use.
                    logger.info(
                        "GitFileStorage.clone_or_retrieve_repository: completed origin=%s branch=%s commit=%s path=%s elapsed=%.2fs",
                        origin_url,
                        branch,
                        commit,
                        replica_path,
                        time.time() - overall_start,
                    )
                    return replica_path
                except Exception as e:
                    logger.error(f"Error cloning or retrieving repository for {origin_url}: type {type(e)}: {e!s}")
                    await self._rollback_clone_transaction(tx)
                    raise

    async def _rollback_clone_transaction(self, tx: GitCloneTransaction):
        try:
            # Remove from EFS if it exists
            if tx.retrieved_from_cold_storage or tx.cloned_to_efs:
                await self.file_system.delete(tx.source_path)

            # Remove from S3 if it exists
            if tx.stored_in_cold_storage:
                await self.cold_storage.delete_repository(tx.origin_url)

            # Remove from cache
            if tx.updated_cache:
                await self.cache_manager.unmark_as_cloned(tx.origin_url)

            # Remove from cache
            if tx.added_source_reference:
                await self.release_reference(tx.origin_url, tx.vmr_id)

            # Remove from cache
            if tx.added_replica_reference:
                await self.release_reference(tx.origin_url, tx.branch, tx.commit, tx.vmr_id)

            # Remove from EFS if it exists
            if tx.created_replica_path:
                await self.file_system.delete(tx.replica_path)

            # Remove from S3 if it exists
            if tx.checked_out_branch or tx.checked_out_commit:
                logger.info(f"________ clone_or_retrieve_repository: Checked out branch or commit. Nothing to undo.")

        except Exception as cleanup_error:
            logger.error(f"Error during cleanup for {tx.origin_url}: {cleanup_error!s}")

    async def bulk_clone_repositories(self, repo_origin_urls: list[str]):
        self.repo_queue.extend(repo_origin_urls)
        tasks = [self._process_repo_queue() for _ in range(self.max_concurrent_clones)]
        await asyncio.gather(*tasks)

    @standard_retry(logger)
    async def git_operation(
        self, *args, origin_url: str, operation: str, branch: str, commit: str, vmr_id: str | UUID4 = "system", **kwargs
    ) -> Any:
        # First, ensure the repository is available locally
        repo_path = await self.clone_or_retrieve_repository(
            origin_url=origin_url,
            branch=branch,
            commit=commit,
            vmr_id=vmr_id,
        )

        cached_result = await self.cache_manager.get_cached_operation(
            origin_url, operation, *args, **kwargs
        )
        if cached_result is not None:
            return cached_result

        async with self._git_repo_lock(origin_url):
            with self.metrics.operation_latency.labels(operation).time():
                try:
                    repo = Repo(str(repo_path))
                    method = getattr(repo.git, operation)
                    result = await call_async(
                        lambda: method(*args, **kwargs)
                    )
                except GitError as e:
                    logger.error(
                        f"Error performing git operation {operation} on {origin_url}: {e!s}"
                    )
                    raise

            await self.cache_manager.cache_operation(
                origin_url, operation, result, *args, **kwargs
            )
            return result

    async def _process_repo_queue(self):
        while self.repo_queue:
            origin_url, branch, commit, vmr_id = self.repo_queue.popleft()
            try:
                repo_path = await self.clone_or_retrieve_repository(
                    origin_url=origin_url,
                    branch=branch,
                    commit=commit,
                    vmr_id=vmr_id,
                )
                logger.info(f"Successfully cloned repository to {repo_path}")
            except Exception as exc:
                logger.error(f"Failed to clone repository: {exc!s}")
                self.repo_queue.append(origin_url)  # Re-add to queue for retry

    async def _periodic_pruning(self):
        while True:
            try:
                repos_metadata = await self.cold_storage.get_all_repo_metadata()
                for repo_metadata in repos_metadata:
                    await self._prune_repository(repo_metadata)
            except Exception as e:
                logger.error(f"Error during periodic pruning: {e!s}")
            await asyncio.sleep(self.prune_interval)

    async def _prune_repository(self, repo_metadata: dict[str, Any]):
        try:
            repo_path = self._get_repo_path(repo_metadata["origin_url"])
            if not await self.file_system.exists(repo_path):
                return
            logger.info(f"Pruning repository at {repo_path}")
            repo = Repo(str(repo_path))
            logger.info(f"Pruning repository at {repo_path}: Created repo object")
            # Do not attempt to delete the branch that is currently checked-out.
            try:
                current_branch_name: str | None = (
                    None if repo.head.is_detached else repo.active_branch.name
                )
            except TypeError:
                current_branch_name = None  # In rare cases active_branch may fail (e.g., detached)
            logger.info(f"Pruning repository at {repo_path}: Current branch name: {current_branch_name}")

            for branch in repo.branches:
                # Skip the branch that is checked out to avoid "Cannot delete branch … checked out" errors.
                if branch.name == current_branch_name:
                    continue

                commit = branch.commit
                branch_age_days = (
                    datetime.now() - datetime.fromtimestamp(commit.committed_date)
                ).days

                if branch_age_days > 30:
                    logger.info(f"Pruning repository at {repo_path}: Deleting branch {branch.name} with age {branch_age_days} days")
                    # Deletion is an I/O heavy git operation – perform in executor
                    await call_async(
                        lambda: repo.delete_head(branch, force=True)
                    )
        except GitError as e:
            logger.error(f"Error pruning repository at {repo_path}: {e!s}")

    async def _update_repositories(self):
        while True:
            try:
                repos_metadata = await self.cold_storage.get_all_repo_metadata()
                for repo_metadata in repos_metadata:
                    await self._update_repository(repo_metadata)
            except Exception as e:
                logger.error(f"Error updating repositories: {e!s}")
            await asyncio.sleep(self.config.update_interval)

    def _should_update(self, repo_metadata: dict[str, Any]) -> bool:
        last_updated = datetime.fromisoformat(repo_metadata["last_updated"])
        return (datetime.now() - last_updated).total_seconds() > self.config.update_threshold

    async def _update_repository(self, repo_metadata: dict[str, Any]):
        if not self._should_update(repo_metadata):
            return
        origin_url = repo_metadata["origin_url"]
        repo_path = self._get_repo_path(origin_url)
        if not await self.file_system.exists(repo_path):
            return
        async with self._git_repo_lock(origin_url):
            try:
                repo = Repo(str(repo_path))
                await call_async(
                    lambda: repo.remotes.origin.pull()
                )
                await self.cold_storage.store_repository(origin_url, repo_path)
            except GitError as e:
                logger.error(f"Error updating repository {origin_url}: {e!s}")

    async def close(self) -> None:
        for task in self._asyncio_tasks:
            task.cancel()
        await asyncio.gather(*self._asyncio_tasks, return_exceptions=True)
        await self.cache_manager.close()

    #@dispatch(str, str)
    async def add_reference(self, repo_id: RepoId, vmr_id: str):
        """Add a VMR reference to a repository or its replica."""
        await self.cache_manager.add_reference(repo_id, vmr_id)
        logger.info(f"Added reference {vmr_id} to repository {repo_id}")

    @dispatch(str, str)
    async def release_reference(
        self, origin_url: str, vmr_id: str
    ):
        ref_count = await self.cache_manager.remove_reference(origin_url, vmr_id)
        logger.info(f"Removed reference {vmr_id} from repository {origin_url}")

        # Check if this was the last reference and remove the repo/replica if so
        if ref_count == 0:
            await self._remove_unreferenced_repo(origin_url)

    @dispatch(str, str, str, str)
    async def release_reference(
        self, origin_url: str, branch: str, commit: str, vmr_id: str
    ):
        replica_id = get_replica_id(origin_url, branch, commit)
        replica_cache_key = f"{origin_url}_{replica_id}"
        ref_count = await self.cache_manager.remove_reference(replica_cache_key, vmr_id)
        logger.info(f"Removed reference {vmr_id} from repository replica {origin_url}:{branch}:{commit}")

        # Check if this was the last reference and remove the repo/replica if so
        if ref_count == 0:
            await self._remove_unreferenced_repo(replica_id)

    async def get_reference_count(self, repo_id: RepoId) -> int:
        """Get the current reference count for a repository or its replica."""
        try:
            return await self.cache_manager.get_reference_count(repo_id)
        except Exception as e:
            logger.error(f"Error getting reference count for {repo_id}: {e!s}")
            return 0

    async def _remove_unreferenced_repo(self, repo_id: RepoId):
        """Remove a repository or replica that has no references."""
        try:
            if "_" in repo_id:  # It's a replica
                origin_url, suffix = repo_id.rsplit("_", 1)
                repo_path = Path(f"{self._get_repo_path(origin_url)}_{suffix}")
            else:  # It's a main repository
                repo_path = self._get_repo_path(repo_id)

            if await self.file_system.exists(repo_path):
                await self.file_system.delete(repo_path)

            if (
                "_" not in repo_id
            ):  # Only delete from cold storage if it's a main repository
                await self.cold_storage.delete_repository(repo_id)

            await self.cache_manager.unmark_as_cloned(repo_id)
            logger.info(f"Removed unreferenced repository/replica: {repo_id}")
        except Exception as e:
            logger.error(
                f"Error removing unreferenced repository/replica {repo_id}: {e!s}"
            )

    async def _garbage_collection(self):
        """Periodically check for and remove unreferenced repositories and replicas."""
        while True:
            try:
                # Get all repositories from cold storage and check their references
                # repos_metadata = await self.cold_storage.get_all_repo_metadata()
                # repo_ids = [repo_metadata["origin_url"] for repo_metadata in repos_metadata]
                repo_ids = await self.cache_manager.get_all_repo_ids()
                for repo_id in repo_ids:
                    ref_count = await self.get_reference_count(repo_id)
                    if ref_count == 0:
                        await self._remove_unreferenced_repo(repo_id)
            except Exception as e:
                logger.error(f"Error during garbage collection: {e!s}")
            await asyncio.sleep(self.gc_interval)  # Run daily by default





# class GitCloneTransactionCM:

#     async def rollback(self):
#         await asyncio.gather(
#             (step() for step in reversed(self.compensations)), return_exceptions=True
#         )

#     async def clone_or_retrieve_repository(self, origin_url: str, branch: str, commit: str, vmr_id: str = "system"):
#         async with GitCloneTransactionCM(origin_url, branch, commit, vmr_id) as tx:
#             if await fs.exists(replica_path):
#                 return replica_path # fast path
#             # 1. ensure source
#             if not await fs.exists(source_path):
#                 await clone_to_efs(origin_url, source_path)
#                 tx.compensations.append(lambda: fs.delete(source_path))
#             # 2. upload cold storage if needed
#             if not await cold_storage.exists(origin_url):
#                 await cold_storage.upload(origin_url, source_path)
#                 tx.compensations.append(lambda: cold_storage.delete(origin_url))
#             # 3. cache bookkeeping in one pipeline
#             async with redis.pipeline() as pipe:
#                 pipe.sadd(...)
#                 pipe.set(...)
#                 await pipe.execute()
#             tx.compensations.append(lambda: redis.delete(...))
#             # 4. replica
#             await call_async(
#                 lambda: shutil.copytree(source_path, replica_path, dirs_exist_ok=True)
#             )
#             tx.compensations.append(lambda: fs.delete(replica_path))
#             # 5. checkout
#             ...
#             return replica_path

