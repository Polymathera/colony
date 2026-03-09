"""Autoscaling for deployments based on load metrics."""

import asyncio
import logging
import time
from typing import Callable

from prometheus_client import Counter, Gauge

from .models import AutoscalingConfig, DeploymentReplicaInfo

logger = logging.getLogger(__name__)

# Metrics
autoscaling_decisions = Counter(
    "deployment_autoscaling_decisions_total",
    "Number of autoscaling decisions made",
    ["deployment_name", "action"],  # action: scale_up, scale_down, no_change
)
current_replicas = Gauge(
    "deployment_current_replicas",
    "Current number of replicas",
    ["deployment_name"],
)


class DeploymentAutoscaler:
    """Autoscaler for deployment replicas based on queue length and load.

    Monitors the request queue length and number of in-flight requests
    to automatically scale the number of replicas up or down.
    """

    def __init__(
        self,
        deployment_name: str,
        config: AutoscalingConfig,
        scale_callback: Callable[[int], None],
        check_interval_s: float = 5.0,
    ):
        """Initialize the autoscaler.

        Args:
            deployment_name: Name of the deployment.
            config: Autoscaling configuration.
            scale_callback: Callback to execute scaling actions.
                Takes target replica count as argument.
            check_interval_s: Interval between autoscaling checks.
        """
        self.deployment_name = deployment_name
        self.config = config
        self.scale_callback = scale_callback
        self.check_interval_s = check_interval_s

        self._last_scale_up_time: float = 0
        self._last_scale_down_time: float = 0
        self._autoscaler_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the autoscaler task."""
        if self._running:
            logger.warning(f"Autoscaler for {self.deployment_name} already running")
            return

        self._running = True
        self._autoscaler_task = asyncio.create_task(self._autoscaler_loop())
        logger.info(f"Started autoscaler for deployment {self.deployment_name}")

    async def stop(self) -> None:
        """Stop the autoscaler task."""
        self._running = False
        if self._autoscaler_task:
            self._autoscaler_task.cancel()
            try:
                await self._autoscaler_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped autoscaler for deployment {self.deployment_name}")

    async def _autoscaler_loop(self) -> None:
        """Main autoscaling loop."""
        while self._running:
            try:
                await asyncio.sleep(self.check_interval_s)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in autoscaler loop: {e}", exc_info=True)

    async def check_and_scale(self, replicas: list[DeploymentReplicaInfo]) -> int | None:
        """Check current load and make scaling decision.

        Args:
            replicas: Current list of replicas (including unhealthy ones).

        Returns:
            Target replica count if scaling is needed, None otherwise.
        """
        current_time = time.time()
        current_replica_count = len(replicas)

        # Only consider healthy replicas for load calculation
        healthy_replicas = [r for r in replicas if r.is_healthy]
        if not healthy_replicas:
            # If we have fewer total replicas than minimum, trigger recovery.
            if current_replica_count < self.config.min_replicas:
                logger.warning(
                    f"No healthy replicas for {self.deployment_name}, "
                    f"recovering to min_replicas={self.config.min_replicas}"
                )
                return self.config.min_replicas
            logger.warning(f"No healthy replicas for {self.deployment_name}")
            return None

        # Calculate total load
        total_queue_length = sum(r.queue_length for r in healthy_replicas)
        total_in_flight = sum(r.in_flight_requests for r in healthy_replicas)
        total_load = total_queue_length + total_in_flight

        # Calculate average load per replica
        avg_load_per_replica = total_load / len(healthy_replicas)

        # Update metrics
        current_replicas.labels(deployment_name=self.deployment_name).set(current_replica_count)

        # Determine if we need to scale
        target_replicas = current_replica_count

        # Scale up if average load exceeds target
        if avg_load_per_replica > self.config.target_queue_length:
            # Check cooldown
            if current_time - self._last_scale_up_time < self.config.scale_up_cooldown_s:
                logger.debug(
                    f"Scale up for {self.deployment_name} in cooldown period "
                    f"({current_time - self._last_scale_up_time:.1f}s elapsed)"
                )
                return None

            # Calculate desired replicas based on load
            desired_replicas = int(total_load / self.config.target_queue_length) + 1
            target_replicas = min(desired_replicas, self.config.max_replicas)

            if target_replicas > current_replica_count:
                logger.info(
                    f"Scaling up {self.deployment_name} from {current_replica_count} "
                    f"to {target_replicas} replicas (avg load: {avg_load_per_replica:.1f})"
                )
                self._last_scale_up_time = current_time
                autoscaling_decisions.labels(
                    deployment_name=self.deployment_name,
                    action="scale_up",
                ).inc()
                return target_replicas

        # Scale down if load is well below target
        elif avg_load_per_replica < self.config.target_queue_length * 0.5:
            # Check cooldown
            if current_time - self._last_scale_down_time < self.config.scale_down_cooldown_s:
                logger.debug(
                    f"Scale down for {self.deployment_name} in cooldown period "
                    f"({current_time - self._last_scale_down_time:.1f}s elapsed)"
                )
                return None

            # Calculate desired replicas
            if total_load == 0:
                # Keep minimum replicas if no load
                target_replicas = self.config.min_replicas
            else:
                desired_replicas = max(
                    int(total_load / (self.config.target_queue_length * 0.7)),
                    self.config.min_replicas,
                )
                target_replicas = desired_replicas

            if target_replicas < current_replica_count:
                logger.info(
                    f"Scaling down {self.deployment_name} from {current_replica_count} "
                    f"to {target_replicas} replicas (avg load: {avg_load_per_replica:.1f})"
                )
                self._last_scale_down_time = current_time
                autoscaling_decisions.labels(
                    deployment_name=self.deployment_name,
                    action="scale_down",
                ).inc()
                return target_replicas

        # No scaling needed
        autoscaling_decisions.labels(
            deployment_name=self.deployment_name,
            action="no_change",
        ).inc()
        return None