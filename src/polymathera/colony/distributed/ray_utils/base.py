import asyncio
import logging
import time
from typing import Any

import ray

logger = logging.getLogger(__name__)


class MyRayActor:
    async def _execute_with_retry(self, cls: type[Any], method: str, *args, **kwargs: Any):
        max_retries = self.config.max_retries
        retry_delay = self.config.retry_delay_seconds

        for attempt in range(max_retries):
            try:
                return await getattr(self, method).remote(*args, **kwargs)

            except ray.exceptions.RayActorError:
                if attempt == max_retries - 1:
                    raise

                logger.warning("Actor failed, attempting recovery...")
                await self._recover_actor(cls)
                await asyncio.sleep(retry_delay)

    async def _recover_actor(self, cls: type[Any]):
        """Recreate failed actor with same name and state"""
        # Save state before recreation
        state = await self._get_actor_state()

        # Recreate actor
        self.actor = (
            ray.remote(cls)
            .options(
                name=self.actor_name,
                max_restarts=-1,  # Allow infinite restarts
            )
            .remote(self.config)
        )

        # Restore state
        await self.actor.restore_state.remote(state)


class ServiceRegistry:
    def __init__(self):
        self.services = {}
        self.health_checks = {}

    def register(self, service_name: str, actor_name: str):
        self.services[service_name] = actor_name
        self.health_checks[service_name] = time.time()

    def get_service(self, service_name: str) -> str:
        if service_name not in self.services:
            raise KeyError(f"Service {service_name} not registered")

        # Check if service is healthy
        if time.time() - self.health_checks[service_name] > HEALTH_CHECK_TIMEOUT:
            self._recover_service(service_name)

        return self.services[service_name]

    async def _recover_service(self, service_name: str):
        """Recover failed service"""
        actor_name = self.services[service_name]
        try:
            # Try to get actor
            actor = ray.get_actor(actor_name)
            # Update health check
            self.health_checks[service_name] = time.time()

        except ray.exceptions.RayActorError:
            # Actor died - recreate it
            await self._recreate_service(service_name)
