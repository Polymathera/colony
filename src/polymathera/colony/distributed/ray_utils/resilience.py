import asyncio
import logging
from typing import Any

import ray

logger = logging.getLogger(__name__)


async def call_ray_actor_with_retry(
    actor_name: str,
    method: str,
    max_retries: int = 3,
    retry_delay: float = 4.0,
    recover_actor: bool = False,
    actor_class: type | None = None,
    *args,
    **kwargs,
):
    for attempt in range(max_retries):
        try:
            actor = ray.get_actor(actor_name)
            return await getattr(actor, method).remote(*args, **kwargs)

        except ray.exceptions.RayActorError:
            if attempt == max_retries - 1:
                raise

            logger.warning(f"Actor {actor_name} failed, attempting recovery...")
            if recover_actor:
                await recover_ray_actor(actor_name, actor_class, config)
            await asyncio.sleep(retry_delay)


async def recover_ray_actor(actor_name: str, actor_class: type, config: dict[str, Any]):
    """Recreate failed actor with same name"""
    actor = ray.remote(actor_class).options(name=actor_name).remote(config)

    await actor.initialize.remote()

    # Restore state
    state = await get_actor_state(actor_name)
    await actor.restore_state.remote(state)
