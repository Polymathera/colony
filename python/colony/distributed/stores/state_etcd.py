from __future__ import annotations

import asyncio
from typing import AsyncIterator
from pydantic import BaseModel, Field
import logging

try:
    import etcd3
    import etcd3.exceptions
    from etcd3.client import Etcd3Client
except ImportError:
    etcd3 = None  # type: ignore[assignment]

from .state_base import StateStorageBackend, StateStorageBackendFactory

logger = logging.getLogger(__name__)



class EtcdStateStorageConfig(BaseModel):
    """Configuration for state storage"""

    namespace: str = Field(default="polymathera")
    ttl: int = 3600  # 1 hour default TTL
    max_retries: int = 3
    retry_delay: float = 0.1  # seconds

    # Etcd specific - use environment variables with fallbacks
    etcd_host: str = Field(
        default="localhost",
        description="Etcd host to connect to",
        json_schema_extra={"env": "ETCD_HOST"},
    )
    etcd_port: int = Field(
        default=2379,
        description="Etcd port to connect to",
        json_schema_extra={"env": "ETCD_PORT"},
    )
    etcd_timeout: int = 5  # seconds
    etcd_ssl: bool = False
    etcd_ca_cert: str | None = None
    etcd_cert_key: str | None = None


class AsyncEtcdEvent:
    """Wrapper for etcd3 events to provide a consistent async interface"""
    def __init__(self, event):
        self.type = "PUT" if event.event_type == "PUT" else "DELETE"
        self.value = event.value
        self.key = event.key


class AsyncEtcdWatcher:
    """Async wrapper for etcd3 watch functionality"""
    def __init__(self, etcd_client: Etcd3Client, key: str):
        self.etcd = etcd_client
        self.key = key
        self._queue = asyncio.Queue()
        self._watch_id = None
        self._running = False
        self._watch_task = None

    async def __aiter__(self) -> AsyncIterator[AsyncEtcdEvent]:
        self._running = True
        # Start the watch in a separate task
        self._watch_task = asyncio.create_task(self._watch_loop())
        try:
            while self._running:
                event = await self._queue.get()
                if event is None:  # Sentinel value for stopping
                    break
                yield AsyncEtcdEvent(event)
        finally:
            await self.stop()

    async def _watch_loop(self):
        """Background task to watch etcd and put events in the queue"""
        loop = asyncio.get_event_loop()

        def blocking_watch_iteration():
            # This runs in the executor thread
            try:
                events_iterator, cancel = self.etcd.watch(self.key)
                self._watch_id = cancel  # Store cancel function
                logger.info(f"Starting blocking iteration for etcd watch key: {self.key}")
                for event in events_iterator:
                    if not self._running:
                        logger.info(f"Watch loop for {self.key} signaled to stop during iteration.")
                        break
                    # Put event into the asyncio queue from the executor thread safely
                    loop.call_soon_threadsafe(self._queue.put_nowait, AsyncEtcdEvent(event))
                logger.info(f"Blocking iteration finished for etcd watch key: {self.key}")
            except Exception as e:
                logger.error(f"Error in blocking etcd watch iteration for key {self.key}: {e}")
                if self._running: # Signal error/completion only if not stopped
                    loop.call_soon_threadsafe(self._queue.put_nowait, None)
            finally:
                # Ensure sentinel is put if loop exits normally or via break
                 if self._running: # Signal completion only if not stopped
                     loop.call_soon_threadsafe(self._queue.put_nowait, None)

        try:
            # Run the blocking iteration in the default executor
            await loop.run_in_executor(None, blocking_watch_iteration)
        except Exception as e:
            logger.error(f"Error running etcd watch executor for key {self.key}: {e}")
            if self._running:
                # Ensure queue gets sentinel on executor failure
                await self._queue.put(None)  # Signal error/completion

    async def stop(self):
        """Stop watching"""
        if not self._running:
            return
        logger.info(f"Stopping etcd watch for key: {self.key}")
        self._running = False
        # Attempt to cancel the underlying etcd watch stream
        if self._watch_id:
            try:
                self._watch_id()  # Call the cancel function
                logger.info(f"Called cancel on underlying etcd watch for key: {self.key}")
            except Exception as e:
                logger.error(f"Error cancelling underlying etcd watch for key {self.key}: {e}")

        # Cancel the asyncio task running the executor
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                logger.info(f"Watch task for {self.key} cancelled successfully.")
            except Exception as e:
                 logger.error(f"Error awaiting cancelled watch task for {self.key}: {e}")
        # Put sentinel value AFTER ensuring the task is cancelled or finished
        # Use put_nowait as stop might be called from a different context
        self._queue.put_nowait(None)  # Ensure iterator exits
        logger.info(f"Etcd watch stop sequence complete for key: {self.key}")


class EtcdStorage(StateStorageBackend):
    """etcd-based storage backend with native compare-and-swap support"""

    def __init__(
        self,
        host: str,
        port: int,
        timeout: int = 5,
        ssl: bool = False,
        ca_cert: str | None = None,
        cert_key: str | None = None,
        ttl: int = 3600,
    ):
        if etcd3 is None:
            raise ImportError(
                "etcd3 is required for EtcdStorage. "
                "Install it with: pip install etcd3  "
                "(or use poetry install --extras distributed)"
            )

        logger.info(
            f"Initializing EtcdStorage with host={host}, port={port}, timeout={timeout}, ssl={ssl}, ca_cert={ca_cert}, cert_key={cert_key}, ttl={ttl}"
        )

        # Only include grpc_options if SSL is enabled
        kwargs = {
            "host": host,
            "port": port,
            "timeout": timeout,
            "ca_cert": ca_cert,
            "cert_key": cert_key,
        }

        if ssl:
            kwargs["grpc_options"] = {
                "grpc.ssl_target_name_override": host
            }.items()

        try:
            self.etcd = etcd3.client(**kwargs)
            # Test connection by getting cluster status
            status = self.etcd.status()
            logger.info(f"Successfully connected to etcd: {status}")
        except Exception as e:
            error_msg = f"Failed to connect to etcd at {host}:{port}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        self.ttl = ttl
        self._lease = None
        self._watchers = {}

    async def _ensure_lease(self):
        """Ensure we have a valid lease for TTL"""
        if self._lease is None or self._lease.expired():
            self._lease = self.etcd.lease(ttl=self.ttl)

    async def get_with_version(self, key: str) -> tuple[str | None, int]:
        """Get value and version (revision) atomically"""
        try:
            value, metadata = self.etcd.get(key)
            if value is None:
                return None, 0
            return value.decode(), metadata.version
        except Exception as e:
            # Handle connection errors, etc.
            raise RuntimeError(f"Failed to get value from etcd: {e}")

    async def compare_and_swap(self, key: str, value: str, version: int) -> bool:
        """
        Atomic compare-and-swap using etcd's native support.
        Returns True if successful, False if version mismatch.
        """
        try:
            await self._ensure_lease()
            success = self.etcd.put(
                key, value.encode(), prev_version=version, lease=self._lease
            )
            return success
        except etcd3.exceptions.PreconditionFailedError:
            # Version mismatch
            return False
        except Exception as e:
            # Handle connection errors, etc.
            raise RuntimeError(f"Failed to perform compare-and-swap in etcd: {e}")

    async def watch(self, key: str) -> AsyncIterator[AsyncEtcdEvent]:
        """Watch a key for changes"""
        if key not in self._watchers:
            self._watchers[key] = AsyncEtcdWatcher(self.etcd, key)
        return self._watchers[key]

    async def stop_watch(self, key: str) -> None:
        """Stop watching a key"""
        try:
            # Stop any active watchers
            if key in self._watchers:
                await self._watchers[key].stop()
                del self._watchers[key]
        except Exception as e:
            # Log but don't raise - this is cleanup
            logger.error(f"Error during etcd key {key} watch stop: {e}")

    async def cleanup(self, key: str) -> None:
        """Close etcd connection and revoke lease"""
        try:
            # Stop any active watchers
            await self.stop_watch(key)

            if self._lease:
                self._lease.revoke()
            self.etcd.delete(key)
            self.etcd.close()
        except Exception as e:
            # Log but don't raise - this is cleanup
            logger.error(f"Error during etcd cleanup: {e}")


class EtcdStateStorageBackendFactory(StateStorageBackendFactory):
    """Factory for creating EtcdStorage instances"""

    def create_backend(self, config: EtcdStateStorageConfig) -> EtcdStorage:
        """Create an EtcdStorage instance based on the provided config"""
        logger.info(
            f"EtcdStateStorageBackendFactory: {config.model_dump_json(indent=2)}"
        )
        return EtcdStorage(
            host=config.etcd_host,
            port=config.etcd_port,
            timeout=config.etcd_timeout,
            ssl=config.etcd_ssl,
            ca_cert=config.etcd_ca_cert,
            cert_key=config.etcd_cert_key,
            ttl=config.ttl,
        )

