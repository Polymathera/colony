"""Health monitoring for deployment replicas."""

import asyncio
import logging
import time
from typing import Any

import ray
from prometheus_client import Counter, Gauge

from .models import DeploymentReplicaInfo

logger = logging.getLogger(__name__)

# Metrics
health_check_total = Counter(
    "deployment_health_check_total",
    "Total number of health checks performed",
    ["deployment_name", "status"],
)
unhealthy_replicas = Gauge(
    "deployment_unhealthy_replicas",
    "Number of unhealthy replicas",
    ["deployment_name"],
)


class DeploymentHealthMonitor:
    """Monitors health of deployment replicas and restarts failed ones.

    Performs periodic health checks on replicas and marks them as unhealthy
    if they become unresponsive. Can restart failed replicas automatically.
    """

    def __init__(
        self,
        deployment_name: str,
        health_check_interval_s: float = 10.0,
        health_check_timeout_s: float = 5.0,
        max_consecutive_failures: int = 3,
    ):
        """Initialize the health monitor.

        Args:
            deployment_name: Name of the deployment being monitored.
            health_check_interval_s: Interval between health checks.
            health_check_timeout_s: Timeout for each health check.
            max_consecutive_failures: Number of failures before marking unhealthy.
        """
        self.deployment_name = deployment_name
        self.health_check_interval_s = health_check_interval_s
        self.health_check_timeout_s = health_check_timeout_s
        self.max_consecutive_failures = max_consecutive_failures

        # Track consecutive failures per replica
        self._failure_counts: dict[str, int] = {}
        self._monitor_task: asyncio.Task | None = None
        self._running = False

    async def start(self, replicas: list[DeploymentReplicaInfo]) -> None:
        """Start the health monitoring task.

        Args:
            replicas: List of replicas to monitor.
        """
        if self._running:
            logger.warning(f"Health monitor for {self.deployment_name} already running")
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(replicas))
        logger.info(f"Started health monitor for deployment {self.deployment_name}")

    async def stop(self) -> None:
        """Stop the health monitoring task."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped health monitor for deployment {self.deployment_name}")

    async def _monitor_loop(self, replicas: list[DeploymentReplicaInfo]) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_all_replicas(replicas)
                await asyncio.sleep(self.health_check_interval_s)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}", exc_info=True)
                await asyncio.sleep(self.health_check_interval_s)

    async def _check_all_replicas(self, replicas: list[DeploymentReplicaInfo]) -> None:
        """Check health of all replicas."""
        unhealthy_count = 0

        for replica in replicas:
            is_healthy = await self._check_replica_health(replica)

            if is_healthy:
                self._failure_counts[replica.replica_id] = 0
                replica.is_healthy = True
                replica.last_health_check = time.time()
                health_check_total.labels(
                    deployment_name=self.deployment_name,
                    status="success",
                ).inc()
            else:
                # Increment failure count
                self._failure_counts[replica.replica_id] = (
                    self._failure_counts.get(replica.replica_id, 0) + 1
                )

                # Mark unhealthy if exceeded max failures
                if self._failure_counts[replica.replica_id] >= self.max_consecutive_failures:
                    if replica.is_healthy:
                        logger.warning(
                            f"Replica {replica.replica_id} marked unhealthy after "
                            f"{self._failure_counts[replica.replica_id]} consecutive failures"
                        )
                    replica.is_healthy = False
                    unhealthy_count += 1

                health_check_total.labels(
                    deployment_name=self.deployment_name,
                    status="failure",
                ).inc()

        # Update metrics
        unhealthy_replicas.labels(deployment_name=self.deployment_name).set(unhealthy_count)

    async def _check_replica_health(self, replica: DeploymentReplicaInfo) -> bool:
        """Check if a single replica is healthy.

        Args:
            replica: The replica to check.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            # Try to ping the actor with a timeout
            health_check_coro = self._ping_replica(replica.actor_handle)
            await asyncio.wait_for(health_check_coro, timeout=self.health_check_timeout_s)
            return True
        except asyncio.TimeoutError:
            logger.warning(
                f"Health check timed out for replica {replica.replica_id} "
                f"after {self.health_check_timeout_s}s"
            )
            return False
        except Exception as e:
            logger.warning(f"Health check failed for replica {replica.replica_id}: {e}")
            return False

    async def _ping_replica(self, actor_handle: Any) -> None:
        """Ping a replica actor.

        Args:
            actor_handle: Ray actor handle to ping.
        """
        # Call a simple health check method on the replica
        # The wrapped deployment class will have this method
        await actor_handle.__ping__.remote()