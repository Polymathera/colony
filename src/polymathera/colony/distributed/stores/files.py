import asyncio
import hashlib
import logging
import os
import zlib
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from overrides import override
from typing import AsyncGenerator

from prometheus_client import Counter, Gauge, Histogram

from ..configs import (
    FileStorageConfig,
    DistributedFileSystemConfig,
    DistributedFileSystemConfig1,
)
from ...utils.retry import standard_retry


logger = logging.getLogger(__name__)



class FileSystemInterface(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        pass

    @abstractmethod
    async def get_size_mb(self, path: Path) -> float:
        pass

    @abstractmethod
    async def exists(self, path: Path) -> bool:
        pass

    @abstractmethod
    async def glob(self, path: Path, pattern: str = "*") -> list[Path]:
        """
        Returns a list of files matching the given glob pattern.
        This method should be implemented to support globbing functionality.
        """
        pass

    @abstractmethod
    async def get_root_path(self, namespace: str) -> Path:
        pass

    @abstractmethod
    async def read_file(self, file_path: str | Path) -> str:
        pass

    @abstractmethod
    async def write_file(self, file_path: Path, data: str):
        pass

    @abstractmethod
    async def read_binary_file(self, file_path: str | Path) -> bytes:
        pass

    @abstractmethod
    async def write_binary_file(self, file_path: Path, data: bytes):
        pass

    @abstractmethod
    async def read_compressed_file(self, file_path: str | Path) -> bytes:
        pass

    @abstractmethod
    async def write_compressed_file(self, file_path: Path, data: bytes):
        pass

    @abstractmethod
    async def delete(self, file_path: Path):
        """
        Deletes a file or directory from the file system.
        """
        pass

    @abstractmethod
    async def list_files(self, directory: Path) -> list[Path]:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    @abstractmethod
    async def walk(self, path: Path) -> AsyncGenerator[tuple[Path, list[Path], list[Path]], None]:
        """
        Walk through the directory structure starting from the given path.
        This method should yield tuples of (directory_path, subdirectories, files).
        """
        pass

    @abstractmethod
    async def join(self, *paths) -> Path:
        """
        Join multiple path components together to form a complete path.
        """
        pass


class LocalFileSystem(FileSystemInterface):
    """Local filesystem backend. Uses plain pathlib operations on a root directory."""

    def __init__(self, config: DistributedFileSystemConfig | None = None):
        self.config: DistributedFileSystemConfig | None = config
        self.root_path: Path | None = None
        self.compression_level: int = 6

    @override
    async def initialize(self) -> None:
        self.config = await DistributedFileSystemConfig.check_or_get_component(self.config)
        self.root_path = Path(self.config.local_root_path)
        self.compression_level = self.config.compression_level
        self.root_path.mkdir(parents=True, exist_ok=True)
        logger.info("Local file system initialized at %s", self.root_path)

    @override
    async def cleanup(self) -> None:
        pass

    def _get_path_for_namespace(self, namespace: str) -> Path:
        namespace_hash = hashlib.md5(namespace.encode()).hexdigest()
        return self.root_path / namespace_hash[:2] / namespace_hash[2:4] / namespace_hash

    @override
    async def exists(self, path: Path) -> bool:
        return (self.root_path / path).exists()

    @override
    async def get_size_mb(self, path: Path) -> float:
        full_path = self.root_path / path
        if not full_path.exists():
            return 0.0
        return os.path.getsize(full_path) / (1024 * 1024)

    @override
    async def get_root_path(self, namespace: str) -> Path:
        root_path = self._get_path_for_namespace(namespace)
        root_path.mkdir(parents=True, exist_ok=True)
        return root_path

    @override
    async def read_file(self, file_path: str | Path) -> str:
        full_path = self.root_path / file_path
        return full_path.read_text(encoding="utf-8")

    @override
    async def write_file(self, file_path: Path, data: str) -> None:
        full_path = self.root_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(data, encoding="utf-8")

    @override
    async def read_binary_file(self, file_path: str | Path) -> bytes:
        return (self.root_path / file_path).read_bytes()

    @override
    async def write_binary_file(self, file_path: Path, data: bytes) -> None:
        full_path = self.root_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(data)

    @override
    async def read_compressed_file(self, file_path: str | Path) -> bytes:
        compressed_data = (self.root_path / file_path).read_bytes()
        return zlib.decompress(compressed_data)

    @override
    async def write_compressed_file(self, file_path: Path, data: bytes) -> None:
        full_path = self.root_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(zlib.compress(data, self.compression_level))

    @override
    async def delete(self, file_path: Path) -> None:
        import shutil
        full_path = self.root_path / file_path
        if full_path.is_file() or full_path.is_symlink():
            os.remove(full_path)
        elif full_path.is_dir():
            shutil.rmtree(full_path)

    @override
    async def list_files(self, directory: Path) -> list[Path]:
        full_path = self.root_path / directory
        if not full_path.exists():
            return []
        return [p.relative_to(self.root_path) for p in full_path.glob("*") if p.is_file()]

    @override
    async def glob(self, path: Path, pattern: str = "*") -> list[Path]:
        full_path = self.root_path / path
        if not full_path.exists():
            return []
        return [p.relative_to(self.root_path) for p in full_path.glob(pattern) if p.is_file()]

    @override
    async def close(self) -> None:
        pass

    @override
    async def walk(self, path: Path) -> AsyncGenerator[tuple[Path, list[Path], list[Path]], None]:
        full_path = self.root_path / path
        if not full_path.exists():
            return
        for dirpath, dirnames, filenames in os.walk(full_path):
            yield (
                Path(dirpath).relative_to(self.root_path),
                [Path(d) for d in dirnames],
                [Path(f) for f in filenames],
            )

    @override
    async def join(self, *paths) -> Path:
        if not paths:
            return Path()
        result = Path(paths[0])
        for path in paths[1:]:
            result = result / path
        return result


class FileStorage(FileSystemInterface):

    def __init__(self, file_system: FileSystemInterface, config: FileStorageConfig | None = None):
        self.config: FileStorageConfig | None = config
        self.file_system = file_system
        self.namespace = None
        self.root_path: Path = None

    @override
    async def initialize(self) -> None:
        self.config = await FileStorageConfig.check_or_get_component(self.config)
        self.namespace = self.config.namespace
        self.root_path = await self.file_system.get_root_path(self.namespace)
        logger.info(f"Initialized file storage at {self.root_path}")

    @override
    async def cleanup(self) -> None:
        from ...utils import cleanup_dynamic_asyncio_tasks

        try:
            await cleanup_dynamic_asyncio_tasks(self, raise_exceptions=False)
        except Exception as e:
            logger.warning(f"Error cleaning up FileStorage tasks: {e}")

    @override
    async def exists(self, path: Path) -> bool:
        return await self.file_system.exists(self.root_path / path)

    @override
    async def glob(self, path: Path, pattern: str = "*") -> list[Path]:
        """
        Returns a list of files matching the given glob pattern.
        This method should be implemented to support globbing functionality.
        """
        full_path = self.root_path / path
        if not full_path.exists():
            return []
        return [p.relative_to(self.root_path) for p in full_path.glob(pattern) if p.is_file()]

    @override
    async def get_size_mb(self, path: Path) -> float:
        return await self.file_system.get_size_mb(self.root_path / path)

    @override
    async def get_root_path(self, namespace: str) -> Path:
        return self.root_path # TODO: Implement namespace

    @override
    async def read_file(self, file_path: str | Path) -> str:
        return await self.file_system.read_file(self.root_path / file_path)

    @override
    async def write_file(self, file_path: Path, data: str) -> None:
        await self.file_system.write_file(self.root_path / file_path, data)

    @override
    async def read_binary_file(self, file_path: str | Path) -> bytes:
        return await self.file_system.read_binary_file(self.root_path / file_path)

    @override
    async def write_binary_file(self, file_path: Path, data: bytes) -> None:
        await self.file_system.write_binary_file(self.root_path / file_path, data)

    @override
    async def read_compressed_file(self, file_path: str | Path) -> bytes:
        return await self.file_system.read_compressed_file(self.root_path / file_path)

    @override
    async def write_compressed_file(self, file_path: Path, data: bytes) -> None:
        await self.file_system.write_compressed_file(self.root_path / file_path, data)

    @override
    async def delete(self, file_path: Path) -> None:
        await self.file_system.delete(self.root_path / file_path)

    @override
    async def list_files(self, directory: Path) -> list[Path]:
        return await self.file_system.list_files(self.root_path / directory)

    @override
    async def close(self) -> None:
        # Don't close the file system, because it can be shared with other objects.
        # await self.file_system.close()
        pass

    @override
    async def walk(self, path: Path) -> AsyncGenerator[tuple[Path, list[Path], list[Path]], None]:
        return await self.file_system.walk(self.root_path / path)

    @override
    async def join(self, *paths) -> Path:
        return await self.file_system.join(*paths)



class ScalableDistributedFileSystem(FileSystemInterface):
    """
    This class assumes that a shared file system (like EFS) is already mounted by the AWS CDK app
    at the path specified in the configuration (`efs_mount_path`). It will perform
    file operations within this mount point. It does not handle the creation or
    mounting of the file system itself.
    """

    def __init__(self, config: DistributedFileSystemConfig | None = None):
        self.config: DistributedFileSystemConfig | None = config
        self.efs_mount_path = None
        self.compression_level = None

        # Prometheus metrics
        self.file_ops_counter = Counter(
            "file_operations_total",
            "Total number of file operations",
            ["operation"]
        )
        self.file_size_gauge = Gauge(
            "file_size_bytes",
            "File size in bytes",
            ["file_name"]
        )
        self.operation_latency = Histogram(
            "file_operation_latency_seconds",
            "Latency of file operations",
            ["operation"],
        )

    @override
    async def initialize(self) -> None:
        """
        Ensures the base mount path exists.
        """
        self.config = await DistributedFileSystemConfig.check_or_get_component(self.config)
        self.efs_mount_path = Path(self.config.efs_mount_path)
        self.compression_level = self.config.compression_level

        import asyncio, time

        async def _path_exists(p: Path) -> bool:
            """Run blocking Path.exists in a thread so we can time-out."""
            return await asyncio.to_thread(p.exists)

        try:
            # Give the kernel at most 2 s to answer the *stat* on the mount
            # directory.  If the directory itself blocks, the mount is broken.
            logger.info("ScalableDFS: probing EFS mount %s", self.efs_mount_path)
            exists = await asyncio.wait_for(_path_exists(self.efs_mount_path), timeout=2.0)
        except (asyncio.TimeoutError, OSError):
            exists = False

        # Even when the directory exists we still need to verify that we can
        # *write* to it without the kernel hanging (happens when the host sees
        # the directory but the NFS client never completed the mount).  We do
        # that with a tiny touch-and-unlink probe wrapped in a 2 s timeout.
        if exists:
            import uuid

            probe_path = self.efs_mount_path / f".__efs_probe_{uuid.uuid4().hex}"
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(probe_path.touch, exist_ok=False), timeout=2.0
                )
                # Clean up – don’t block longer than 1 s.
                await asyncio.wait_for(
                    asyncio.to_thread(probe_path.unlink), timeout=1.0
                )
            except (asyncio.TimeoutError, OSError):
                exists = False

        if not exists:
            # In CI we can explicitly allow a fallback by setting an env var –
            # otherwise we *fail fast* so infrastructure issues surface
            # immediately instead of causing kernel stalls.
            if os.getenv("POLYMATHERA_EFS_ALLOW_FALLBACK", "false").lower() == "true":
                logger.warning(
                    "EFS mount path %s not usable; using /tmp/polymathera_efs_stub because POLYMATHERA_EFS_ALLOW_FALLBACK=true",
                    self.efs_mount_path,
                )
                self.efs_mount_path = Path("/tmp/polymathera_efs_stub")
            else:
                raise RuntimeError(
                    f"EFS mount path {self.efs_mount_path} is not available or unresponsive → aborting. "
                    "Mount the EFS volume inside the container or set POLYMATHERA_EFS_ALLOW_FALLBACK=true for local tests."
                )

        # Ensure directory exists (thread-offload avoids blocking mkdir on NFS)
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self.efs_mount_path.mkdir, parents=True, exist_ok=True),
                timeout=2.0,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Creating directory {self.efs_mount_path} blocked >2 s - likely broken NFS mount."
            )

        logger.info("Distributed file system initialized at %s", self.efs_mount_path)

    @override
    async def cleanup(self) -> None:
        from ...utils import cleanup_dynamic_asyncio_tasks

        try:
            await cleanup_dynamic_asyncio_tasks(self, raise_exceptions=False)
        except Exception as e:
            logger.warning(f"Error cleaning up ScalableDistributedFileSystem tasks: {e}")

    def _get_path_for_namespace(self, namespace: str) -> Path:
        """Determines the directory for a given namespace within the EFS mount."""
        # Use a simple hashing scheme to create a subdirectory structure
        # to avoid having too many files/directories in a single directory.
        namespace_hash = hashlib.md5(namespace.encode()).hexdigest()
        # e.g., /mnt/efs/ab/cd/abcdef12345...
        return self.efs_mount_path / namespace_hash[:2] / namespace_hash[2:4] / namespace_hash

    @override
    async def exists(self, path: Path) -> bool:
        full_path = self.efs_mount_path / path
        return full_path.exists()

    @override
    async def get_size_mb(self, path: Path) -> float:
        # This path is relative to the namespace root.
        full_path = self.efs_mount_path / path
        if not full_path.exists():
            return 0.0
        return os.path.getsize(full_path) / (1024 * 1024)

    @override
    async def get_root_path(self, namespace: str) -> Path:
        """
        Gets the root directory path for a given namespace, creating it if it doesn't exist.
        """
        root_path = self._get_path_for_namespace(namespace)
        if not root_path.exists():
            root_path.mkdir(parents=True, exist_ok=True)
        return root_path

    @standard_retry(logger)
    @override
    async def read_file(self, file_path: str | Path) -> str:
        """
        Reads a text file from the EFS mount path and returns its contents as a string.
        """
        with self.operation_latency.labels("read_file").time():
            try:
                full_path = self.efs_mount_path / file_path
                with open(full_path, "r", encoding="utf-8") as f:
                    data = f.read()
                self.file_ops_counter.labels("read_file").inc()
                self.file_size_gauge.labels(str(file_path)).set(len(data))
                return data
            except UnicodeDecodeError:
                logger.error(f"Error decoding file: {full_path}")
                raise
            except FileNotFoundError:
                logger.error(f"File not found: {full_path}")
                raise
            except Exception as e:
                logger.error(f"Error reading file: {full_path}: {e!s}")
                raise

    @standard_retry(logger)
    @override
    async def write_file(self, file_path: Path, data: str):
        """
        Writes a text file to the EFS mount path.
        """
        with self.operation_latency.labels("write_file").time():
            try:
                full_path = self.efs_mount_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                # Specify encoding explicitly for text files
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(data)
                self.file_ops_counter.labels("write_file").inc()
                self.file_size_gauge.labels(str(file_path)).set(len(data))
            except OSError as e:
                logger.error(f"Error writing file {full_path}: {e!s}")
                raise

    @standard_retry(logger)
    @override
    async def read_binary_file(self, file_path: str | Path) -> bytes:
        """
        Reads a binary file from the EFS mount path and returns its contents as bytes.
        """
        with self.operation_latency.labels("read_binary").time():
            try:
                full_path = self.efs_mount_path / file_path
                # Use open in binary mode and read all contents
                with open(full_path, "rb") as f:
                    data = f.read()
                self.file_ops_counter.labels("read_binary").inc()
                self.file_size_gauge.labels(str(file_path)).set(len(data))
                return data
            except FileNotFoundError:
                logger.error(f"File not found: {full_path}")
                raise

    @standard_retry(logger)
    @override
    async def read_compressed_file(self, file_path: str | Path) -> bytes:
        """
        Reads a compressed file from the EFS mount path and returns its contents as bytes.
        """
        with self.operation_latency.labels("read_compressed").time():
            try:
                full_path = self.efs_mount_path / file_path
                with open(full_path, "rb") as f:
                    compressed_data = f.read()
                data = zlib.decompress(compressed_data)
                self.file_ops_counter.labels("read_compressed").inc()
                self.file_size_gauge.labels(str(file_path)).set(len(data))
                return data
            except FileNotFoundError:
                logger.error(f"File not found: {full_path}")
                raise
            except zlib.error:
                logger.error(f"Decompression error for file: {full_path}")
                raise

    @standard_retry(logger)
    @override
    async def write_binary_file(self, file_path: Path, data: bytes):
        """
        Writes binary data to a file at the given path.
        """
        with self.operation_latency.labels("write_binary").time():
            try:
                full_path = self.efs_mount_path / file_path
                # Ensure the parent directory exists
                full_path.parent.mkdir(parents=True, exist_ok=True)
                # Open the file in binary write mode and write the data
                with open(full_path, "wb") as f:
                    f.write(data)
                self.file_ops_counter.labels("write_binary").inc()
                self.file_size_gauge.labels(str(file_path)).set(len(data))
            except OSError as e:
                logger.error(f"Error writing file {full_path}: {e!s}")
                raise

    @standard_retry(logger)
    @override
    async def write_compressed_file(self, file_path: Path, data: bytes):
        """
        Compresses and writes data to a file at the given path.
        """
        with self.operation_latency.labels("write_compressed").time():
            try:
                full_path = self.efs_mount_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                compressed_data = zlib.compress(data, self.compression_level)
                with open(full_path, "wb") as f:
                    f.write(compressed_data)
                self.file_ops_counter.labels("write_compressed").inc()
                self.file_size_gauge.labels(str(file_path)).set(len(data))
            except OSError as e:
                logger.error(f"Error writing file {full_path}: {e!s}")
                raise

    @standard_retry(logger)
    @override
    async def delete(self, file_path: Path):
        """
        Deletes a file or directory (recursively) at the given path.
        """
        with self.operation_latency.labels("delete").time():
            try:
                full_path = self.efs_mount_path / file_path
                if full_path.is_file() or full_path.is_symlink():
                    os.remove(full_path)
                elif full_path.is_dir():
                    import shutil
                    shutil.rmtree(full_path)
                else:
                    logger.warning(f"Path does not exist for deletion: {full_path}")
                    return
                self.file_ops_counter.labels("delete").inc()
                self.file_size_gauge.labels(str(file_path)).set(0)
            except FileNotFoundError:
                logger.warning(f"File not found for deletion: {full_path}")
            except OSError as e:
                logger.error(f"Error deleting {full_path}: {e!s}")
                raise

    @standard_retry(logger)
    @override
    async def list_files(self, directory: Path) -> list[Path]:
        with self.operation_latency.labels("list").time():
            try:
                full_path = self.efs_mount_path / directory
                files = [p for p in full_path.glob("*") if p.is_file()]
                self.file_ops_counter.labels("list").inc()
                # Return paths relative to the mount point
                return [p.relative_to(self.efs_mount_path) for p in files]
            except OSError as e:
                logger.error(f"Error listing files in directory {full_path}: {e!s}")
                raise

    @override
    async def glob(self, path: Path, pattern: str = "*") -> list[Path]:
        """
        Returns a list of files matching the given glob pattern.
        This method should be implemented to support globbing functionality.
        """
        full_path = self.efs_mount_path / path
        if not full_path.exists():
            return []
        return [p.relative_to(self.efs_mount_path) for p in full_path.glob(pattern) if p.is_file()]

    @override
    async def close(self) -> None:
        """No-op, as there are no active clients to close."""
        pass

    @override
    async def walk(self, path: Path) -> AsyncGenerator[tuple[Path, list[Path], list[Path]], None]:
        """
        Walk through the directory structure starting from the given path.
        This method yields tuples of (directory_path, subdirectories, files).
        """
        full_path = self.efs_mount_path / path
        if not full_path.exists():
            return

        for dirpath, dirnames, filenames in os.walk(full_path):
            # Convert to Path objects and yield
            yield (
                Path(dirpath).relative_to(self.efs_mount_path),
                [Path(d) for d in dirnames],
                [Path(f) for f in filenames],
            )

    @override
    async def join(self, *paths) -> Path:
        """
        Join multiple path components together to form a complete path.
        """
        if not paths:
            return Path()

        # Start with the first path and join the rest
        result = Path(paths[0])
        for path in paths[1:]:
            result = result / path
        return result



class ScalableDistributedFileSystem1(FileSystemInterface):
    """

    ScalableDistributedFileSystem provides a scalable and distributed file system service.
    It uses multiple EFS instances across different regions for sharding and implements
    caching, compression, and monitoring for improved performance and reliability.

    Key features:
    1. Sharding: Files are distributed across multiple EFS instances in multiple regions.
    2. Caching: Distributed caching is used to cache file metadata and operation results.
    3. Compression: Files are compressed using zlib to reduce storage requirements.
    4. Monitoring and auto-scaling: CloudWatch metrics are used to monitor EFS usage, and the system can automatically scale up EFS instances when needed.
    5. Error handling and retries: The tenacity library is used to implement retries for operations that may fail due to transient issues.
    6. Metrics: Prometheus metrics are used to track important system statistics.

    The `DistributedFileSystem` class may face scalability challenges when
    dealing with millions of git repositories and agents.

        1. EFS Scalability:
        Amazon EFS can handle thousands of concurrent connections and
        provides good performance for many use cases. However, for millions of
        repositories and agents, you might hit some limitations:
            - IOPS limits: EFS has a limit on the number of IOPS it can handle.
            - Latency: As the number of files grows, you might experience increasedlatency.

        2. Git Operations:
        Cloning and operating on millions of git repositories simultaneously can
        be resource-intensive and slow.

    To improve scalability, the following enhancements are used:

        1. Asynchronous operations: Using `asyncio` and `aioboto3` for non-blocking I/O operations.
        2. Elastic Throughput Mode: Using EFS with Elastic Throughput mode for better performance under varying workloads.
        3. Semaphore for concurrent operations: Limiting the number of concurrent git operations to prevent overwhelming the system.
        4. Queue-based repository processing: Implementing a queue system for bulk cloning to manage large numbers of repositories efficiently.
        5. Retrying failed operations: Re-adding failed clone operations to the queue for retry.
        6. Generic git operation method: Allowing various git operations to be performed asynchronously.

    For further scalability, implement the following:
        1. Sharding: Implement a sharding strategy to distribute repositories across multiple EFS instances or even different storage solutions.
        2. Caching: Implement a distributed caching layer for frequently accessed repository metadata or file contents.
        3. Read replicas: For read-heavy workloads, consider implementing read replicas of repositories.
        4. Compression: Implement compression for stored repositories to reduce storage requirements and improve I/O performance.
        5. Pruning: Regularly prune unnecessary branches and old commits to keep repository sizes manageable.
        6. Monitoring and auto-scaling: Implement comprehensive monitoring and auto-scaling mechanisms to adjust resources based on demand.

    While these improvements significantly enhance scalability, handling millions of
    repositories and agents simultaneously will still be challenging.
    You may need to consider distributed systems beyond a single EFS instance,
    possibly involving multiple regions and advanced caching strategies.

    This implementation includes the following enhancements:
        1. Sharding: Repositories are distributed across multiple EFS instances in multiple regions based on a hash of the repository name.
        2. Caching: Distributed caching is used to cache repository metadata and operation results.
        3. Read replicas: A method to create read replicas of repositories is implemented.
        4. Compression: Repositories are compressed using zlib to reduce storage requirements.
        5. Pruning: A periodic task runs to prune old branches and perform garbage collection on repositories.
        6. Monitoring and auto-scaling: CloudWatch metrics are used to monitor EFS usage, and the system can automatically scale up EFS instances when needed.
        7. Error handling and retries: The tenacity library is used to implement retries for operations that may fail due to transient issues.
        8. Metrics: Prometheus metrics are used to track important system statistics.

     Using S3 as a cold storage for repositories can significantly improve the scalability and efficiency of your system. This approach addresses several important concerns:
        1. Rate limiting: It reduces the number of requests to GitHub/GitLab, helping you stay within rate limits.
        2. Bandwidth: It minimizes the amount of data transferred from external Git providers.
        3. Speed: Retrieving compressed repositories from S3 can be faster than cloning from remote Git servers, especially for large repositories.
        4. Reliability: It reduces dependency on external services, making your system more robust.
        5. Cost-effective: It can be more cost-effective than using EFS, especially for read-heavy workloads.

    Using multiple S3 buckets to shard the repositories improves scalability and performance. This approach offers several benefits:
        1. Increased throughput: S3 has per-bucket rate limits. By using multiple buckets, you can increase your overall throughput.
        2. Better load distribution: Sharding across multiple buckets can help distribute the load more evenly.
        3. Improved parallel processing: Multiple buckets allow for more efficient parallel operations.
        4. Region-specific storage: You can create buckets in different regions to reduce latency for geographically distributed systems.
        5. Easier management of large-scale systems: Sharding makes it easier to manage and organize a large number of repositories.
    """

    def __init__(self, config: DistributedFileSystemConfig1 | None = None):
        self.config: DistributedFileSystemConfig1 | None = config
        try:
            import aiobotocore.session
            self.session = aiobotocore.session.get_session()
        except ImportError:
            raise ImportError(
                "aiobotocore is required for ScalableDistributedFileSystem1. "
                "Install it with: pip install aiobotocore  "
                "(or use poetry install --extras aws)"
            )
        self.efs_clients = {}
        self.ec2_client = None
        self.cloudwatch_client = None
        self.shard_count = None
        self.regions = None
        self.compression_level = None
        self.cache_ttl = None

        # Prometheus metrics
        self.file_ops_counter = Counter(
            "file_operations_total", "Total number of file operations", ["operation"]
        )
        self.file_size_gauge = Gauge(
            "file_size_bytes", "File size in bytes", ["file_name"]
        )
        self.operation_latency = Histogram(
            "file_operation_latency_seconds",
            "Latency of file operations",
            ["operation"],
        )

    @override
    async def initialize(self) -> None:
        self.config = await DistributedFileSystemConfig1.check_or_get_component(self.config)
        self.shard_count = self.config.shard_count
        self.regions = self.config.regions
        self.compression_level = self.config.compression_level
        self.cache_ttl = self.config.cache_ttl
        await self._init_clients()
        await self._init_efs_instances()
        asyncio.create_task(self._monitor_and_scale())

    async def _init_clients(self):
        self.ec2_client = self.session.create_client("ec2")
        self.cloudwatch_client = self.session.create_client("cloudwatch")
        for region in self.regions:
            self.efs_clients[region] = self.session.create_client(
                "efs", region_name=region
            )

    async def _init_efs_instances(self):
        for region in self.regions:
            efs_client = self.efs_clients[region]
            for i in range(self.shard_count):
                file_system_id = await self._create_or_get_efs(efs_client, f"shard-{i}")
                await self._create_mount_targets(efs_client, file_system_id, region)
                await self._mount_efs(region, file_system_id)

    async def _mount_efs(self, region, file_system_id):
        mount_point = self._get_mount_point(region, file_system_id)
        if not os.path.exists(mount_point):
            os.makedirs(mount_point)

        mount_command = f"sudo mount -t efs {file_system_id}:/ {mount_point}"
        process = await asyncio.create_subprocess_shell(mount_command)
        await process.wait()

    @standard_retry(logger)
    async def _create_or_get_efs(self, efs_client, shard_name):
        try:
            response = await efs_client.create_file_system(
                PerformanceMode="maxIO",
                ThroughputMode="elastic",  # 'bursting',
                Encrypted=True,
                Tags=[{"Key": "Name", "Value": f"Polymathera-{shard_name}"}],
            )
            return response["FileSystemId"]
        except efs_client.exceptions.FileSystemAlreadyExists:
            response = await efs_client.describe_file_systems()
            for fs in response["FileSystems"]:
                if any(
                    tag["Key"] == "Name" and tag["Value"] == f"Polymathera-{shard_name}"
                    for tag in fs.get("Tags", [])
                ):
                    return fs["FileSystemId"]
            raise Exception(f"EFS for shard {shard_name} not found")

    async def _create_mount_targets(self, efs_client, file_system_id, region):
        ec2 = self.session.create_client("ec2", region_name=region)
        subnets = await ec2.describe_subnets()
        for subnet in subnets["Subnets"]:
            try:
                await efs_client.create_mount_target(
                    FileSystemId=file_system_id,
                    SubnetId=subnet["SubnetId"],
                    SecurityGroups=[self.config.security_group_id],
                )
            except efs_client.exceptions.MountTargetConflict:
                pass

    def _get_shard_for_file(self, file_name: str) -> int:
        # return hash(repo_name) % self.shard_count
        # return int(hashlib.md5(repo_name.encode()).hexdigest(), 16) % self.shard_count
        return hashlib.md5(file_name.encode()).hexdigest() % self.shard_count

    def _get_region_for_file(self, file_name: str) -> str:
        return self.regions[hash(file_name) % len(self.regions)]

    def _get_mount_point(self, region: str, file_system_id: str) -> Path:
        return Path(f"/mnt/efs/{region}/{file_system_id}")

    @override
    async def exists(self, path: Path) -> bool:
        return os.path.exists(path)

    @override
    async def get_size_mb(self, path: Path) -> float:
        return os.path.getsize(path) / (1024 * 1024)

    @override
    async def get_root_path(self, namespace: str) -> Path:
        shard = self._get_shard_for_file(namespace)
        region = self._get_region_for_file(namespace)
        efs_client = self.efs_clients[region]
        file_system_id = await self._create_or_get_efs(efs_client, f"shard-{shard}")
        mount_point = self._get_mount_point(region, file_system_id)
        return mount_point

    @standard_retry(logger)
    @override
    async def read_compressed_file(self, file_path: str | Path) -> bytes:
        with self.operation_latency.labels("read_compressed").time():
            try:
                with open(file_path, "rb") as f:
                    compressed_data = f.read()
                data = zlib.decompress(compressed_data)
                self.file_ops_counter.labels("read_compressed").inc()
                self.file_size_gauge.labels(str(file_path)).set(len(data))
                return data
            except FileNotFoundError:
                logger.error(f"File not found: {file_path}")
                raise
            except zlib.error:
                logger.error(f"Decompression error for file: {file_path}")
                raise

    @standard_retry(logger)
    @override
    async def write_compressed_file(self, file_path: Path, data: bytes):
        with self.operation_latency.labels("write_compressed").time():
            try:
                compressed_data = zlib.compress(data, self.compression_level)
                with open(file_path, "wb") as f:
                    f.write(compressed_data)
                self.file_ops_counter.labels("write_compressed").inc()
                self.file_size_gauge.labels(str(file_path)).set(len(data))
            except OSError as e:
                logger.error(f"Error writing file {file_path}: {e!s}")
                raise

    @standard_retry(logger)
    @override
    async def delete(self, file_path: Path):
        """
        Deletes a file or directory (recursively) at the given path.
        """
        with self.operation_latency.labels("delete").time():
            try:
                if file_path.is_file() or file_path.is_symlink():
                    os.remove(file_path)
                elif file_path.is_dir():
                    import shutil
                    shutil.rmtree(file_path)
                else:
                    logger.warning(f"Path does not exist for deletion: {file_path}")
                    return
                self.file_ops_counter.labels("delete").inc()
                self.file_size_gauge.labels(str(file_path)).set(0)
            except FileNotFoundError:
                logger.warning(f"File not found for deletion: {file_path}")
            except OSError as e:
                logger.error(f"Error deleting {file_path}: {e!s}")
                raise

    @standard_retry(logger)
    @override
    async def list_files(self, directory: Path) -> list[Path]:
        with self.operation_latency.labels("list").time():
            try:
                files = [p for p in directory.glob("*") if p.is_file()]
                self.file_ops_counter.labels("list").inc()
                return files
            except OSError as e:
                logger.error(f"Error listing files in directory {directory}: {e!s}")
                raise

    async def _monitor_and_scale(self):
        while True:
            for region in self.regions:
                efs_client = self.efs_clients[region]
                file_systems = await efs_client.describe_file_systems()
                for fs in file_systems["FileSystems"]:
                    metrics = await self.cloudwatch_client.get_metric_statistics(
                        Namespace="AWS/EFS",
                        MetricName="PercentIOLimit",
                        Dimensions=[
                            {"Name": "FileSystemId", "Value": fs["FileSystemId"]}
                        ],
                        StartTime=datetime.now(timezone.utc) - timedelta(minutes=5),
                        EndTime=datetime.now(timezone.utc),
                        Period=300,
                        Statistics=["Average"],
                    )
                    if metrics["Datapoints"]:
                        percent_io_limit = metrics["Datapoints"][0]["Average"]
                        if percent_io_limit > 80:
                            await self._scale_up_efs(fs["FileSystemId"])
            await asyncio.sleep(300)  # Check every 5 minutes

    async def _scale_up_efs(self, file_system_id):
        efs_client = self.efs_clients[self._get_region_for_file(file_system_id)]
        await efs_client.update_file_system(
            FileSystemId=file_system_id,
            ThroughputMode="provisioned",  # 'elastic', 'bursting',
            ProvisionedThroughputInMibps=self.config.scale_up_throughput,
        )
        logger.info(f"Scaled up EFS {file_system_id}")

    @override
    async def close(self) -> None:
        for client in self.efs_clients.values():
            await client.close()
        await self.ec2_client.close()
        await self.cloudwatch_client.close()

    @override
    async def glob(self, path: Path, pattern: str = "*") -> list[Path]:
        """
        Returns a list of files matching the given glob pattern.
        """
        if not path.exists():
            return []
        return [p for p in path.glob(pattern) if p.is_file()]

    @override
    async def walk(self, path: Path) -> AsyncGenerator[tuple[Path, list[Path], list[Path]], None]:
        """
        Walk through the directory structure starting from the given path.
        """
        if not path.exists():
            return

        for dirpath, dirnames, filenames in os.walk(path):
            yield (
                Path(dirpath),
                [Path(d) for d in dirnames],
                [Path(f) for f in filenames],
            )

    @override
    async def join(self, *paths) -> Path:
        """
        Join multiple path components together to form a complete path.
        """
        if not paths:
            return Path()

        # Start with the first path and join the rest
        result = Path(paths[0])
        for path in paths[1:]:
            result = result / path
        return result

