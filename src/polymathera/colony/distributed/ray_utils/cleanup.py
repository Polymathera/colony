"""
Reusable utilities for Ray actor cleanup with namespace-based organization.

This module provides centralized, parameterizable cleanup utilities to avoid
code duplication across the codebase.
"""

import asyncio
from dataclasses import dataclass
from typing import List, Optional, Set, Callable, Dict, Any
from enum import Enum

import ray
from ray.util.state import list_actors

from ...utils import setup_logger

logger = setup_logger(__name__)


class CleanupStrategy(Enum):
    """Strategy for cleanup operations."""
    GRACEFUL_THEN_KILL = "graceful_then_kill"  # Try cleanup() then ray.kill()
    KILL_ONLY = "kill_only"  # Only use ray.kill()
    GRACEFUL_ONLY = "graceful_only"  # Only try cleanup() method


@dataclass
class ActorCleanupConfig:
    """Configuration for Ray actor cleanup operations."""

    # Namespace filtering
    namespace_patterns: List[str] = None  # e.g., ["polymathera_test_*", "temp_*"]
    exact_namespaces: List[str] = None    # e.g., ["polymathera_test_abc123"]
    exclude_namespaces: List[str] = None  # Never clean these namespaces

    # Actor filtering
    actor_types: List[str] = None         # e.g., ["DistributedWorker"]
    actor_states: List[str] = None        # e.g., ["ALIVE", "PENDING_CREATION"]

    # Cleanup behavior
    strategy: CleanupStrategy = CleanupStrategy.GRACEFUL_THEN_KILL
    max_concurrent_cleanups: int = 10
    cleanup_timeout_seconds: float = 30.0

    # Logging
    log_individual_actors: bool = True
    log_summary: bool = True


class RayActorCleanupUtility:
    """Centralized utility for Ray actor cleanup operations."""

    def __init__(self):
        self._cleanup_stats = {
            "total_found": 0,
            "total_cleaned": 0,
            "graceful_success": 0,
            "kill_success": 0,
            "failures": 0,
            "namespaces_processed": set(),
        }

    async def cleanup_actors(
        self,
        config: ActorCleanupConfig,
        actor_handles: Optional[List[ray.actor.ActorHandle]] = None
    ) -> Dict[str, Any]:
        """
        Clean up Ray actors based on configuration.

        Args:
            config: Cleanup configuration
            actor_handles: Optional list of specific actor handles to clean up.
                         If provided, namespace/type filtering is ignored.

        Returns:
            Dictionary with cleanup statistics
        """
        self._reset_stats()

        try:
            if actor_handles:
                # Clean up specific actor handles
                await self._cleanup_actor_handles(actor_handles, config)
            else:
                # Discover and clean up actors based on filters
                target_actors = await self._discover_target_actors(config)
                await self._cleanup_discovered_actors(target_actors, config)

            if config.log_summary:
                self._log_cleanup_summary()

            return dict(self._cleanup_stats)

        except Exception as e:
            logger.error(f"Error during Ray actor cleanup: {e}")
            self._cleanup_stats["error"] = str(e)
            return dict(self._cleanup_stats)

    async def _discover_target_actors(self, config: ActorCleanupConfig) -> List[Any]:
        """Discover actors that match the cleanup criteria."""
        logger.info("🔍 Discovering Ray actors for cleanup...")

        all_actors = list_actors()
        target_actors = []
        namespace_counts = {}

        for actor in all_actors:
            namespace = getattr(actor, 'namespace', 'anonymous') or 'anonymous'
            actor_type = actor.class_name
            state = actor.state

            # Count actors by namespace for logging
            namespace_counts[namespace] = namespace_counts.get(namespace, 0) + 1

            # Apply filters
            if not self._matches_filters(actor, namespace, actor_type, state, config):
                continue

            target_actors.append(actor)
            self._cleanup_stats["namespaces_processed"].add(namespace)

        self._cleanup_stats["total_found"] = len(target_actors)

        if config.log_summary:
            logger.info(f"Found {len(target_actors)} actors matching cleanup criteria")
            logger.info(f"Namespaces with actors: {dict(namespace_counts)}")

        return target_actors

    def _matches_filters(self, actor, namespace: str, actor_type: str, state: str, config: ActorCleanupConfig) -> bool:
        """Check if actor matches all configured filters."""

        # Exclude namespace filter (takes precedence)
        if config.exclude_namespaces:
            for exclude_pattern in config.exclude_namespaces:
                if self._matches_namespace_pattern(namespace, exclude_pattern):
                    return False

        # Namespace filters
        if config.exact_namespaces:
            if namespace not in config.exact_namespaces:
                return False
        elif config.namespace_patterns:
            matches_pattern = any(
                self._matches_namespace_pattern(namespace, pattern)
                for pattern in config.namespace_patterns
            )
            if not matches_pattern:
                return False

        # Actor type filter
        if config.actor_types and actor_type not in config.actor_types:
            return False

        # Actor state filter
        if config.actor_states and state not in config.actor_states:
            return False

        return True

    def _matches_namespace_pattern(self, namespace: str, pattern: str) -> bool:
        """Check if namespace matches pattern (supports * wildcards)."""
        if '*' not in pattern:
            return namespace == pattern

        # Simple wildcard matching - convert to regex-like behavior
        import fnmatch
        return fnmatch.fnmatch(namespace, pattern)

    async def _cleanup_discovered_actors(self, target_actors: List[Any], config: ActorCleanupConfig):
        """Clean up discovered actors using the configured strategy."""
        if not target_actors:
            logger.info("No actors found matching cleanup criteria")
            return

        logger.info(f"🧹 Cleaning up {len(target_actors)} actors using {config.strategy.value} strategy...")

        # Create cleanup tasks with concurrency control
        semaphore = asyncio.Semaphore(config.max_concurrent_cleanups)
        cleanup_tasks = [
            self._cleanup_single_discovered_actor(actor, config, semaphore)
            for actor in target_actors
        ]

        # Execute cleanup tasks
        results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                self._cleanup_stats["failures"] += 1
                if config.log_individual_actors:
                    logger.warning(f"Cleanup task failed: {result}")

    async def _cleanup_single_discovered_actor(self, actor_info, config: ActorCleanupConfig, semaphore: asyncio.Semaphore):
        """Clean up a single discovered actor."""
        async with semaphore:
            try:
                # Convert actor info to handle if possible
                actor_handle = None
                if hasattr(actor_info, 'name') and hasattr(actor_info, 'namespace'):
                    try:
                        actor_handle = ray.get_actor(actor_info.name, namespace=actor_info.namespace)
                    except Exception:
                        pass  # Will fall back to kill by ID

                success = await self._cleanup_single_actor(
                    actor_handle or actor_info,
                    config,
                    actor_name=getattr(actor_info, 'name', f"actor_{actor_info.actor_id[:8]}")
                )

                if success:
                    self._cleanup_stats["total_cleaned"] += 1
                else:
                    self._cleanup_stats["failures"] += 1

            except Exception as e:
                self._cleanup_stats["failures"] += 1
                if config.log_individual_actors:
                    logger.warning(f"Failed to cleanup actor {getattr(actor_info, 'name', 'unknown')}: {e}")

    async def _cleanup_actor_handles(self, actor_handles: List[ray.actor.ActorHandle], config: ActorCleanupConfig):
        """Clean up specific actor handles."""
        logger.info(f"🧹 Cleaning up {len(actor_handles)} provided actor handles...")
        self._cleanup_stats["total_found"] = len(actor_handles)

        # Create cleanup tasks with concurrency control
        semaphore = asyncio.Semaphore(config.max_concurrent_cleanups)
        cleanup_tasks = [
            self._cleanup_single_actor_with_semaphore(handle, config, semaphore, f"handle_{i}")
            for i, handle in enumerate(actor_handles)
        ]

        # Execute cleanup tasks
        results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                self._cleanup_stats["failures"] += 1
            elif result:
                self._cleanup_stats["total_cleaned"] += 1
            else:
                self._cleanup_stats["failures"] += 1

    async def _cleanup_single_actor_with_semaphore(self, actor_handle, config: ActorCleanupConfig, semaphore: asyncio.Semaphore, actor_name: str):
        """Wrapper to apply semaphore to single actor cleanup."""
        async with semaphore:
            return await self._cleanup_single_actor(actor_handle, config, actor_name)

    async def _cleanup_single_actor(self, actor_handle_or_info, config: ActorCleanupConfig, actor_name: str = None) -> bool:
        """
        Clean up a single actor using the configured strategy.

        Returns True if cleanup succeeded, False otherwise.
        """
        if actor_name is None:
            actor_name = "unknown_actor"

        try:
            if config.strategy == CleanupStrategy.GRACEFUL_ONLY:
                return await self._try_graceful_cleanup(actor_handle_or_info, actor_name, config)

            elif config.strategy == CleanupStrategy.KILL_ONLY:
                return await self._try_kill_cleanup(actor_handle_or_info, actor_name, config)

            elif config.strategy == CleanupStrategy.GRACEFUL_THEN_KILL:
                # Try graceful first
                if await self._try_graceful_cleanup(actor_handle_or_info, actor_name, config):
                    return True
                # Fall back to kill
                return await self._try_kill_cleanup(actor_handle_or_info, actor_name, config)

        except Exception as e:
            if config.log_individual_actors:
                logger.warning(f"Failed to cleanup {actor_name}: {e}")
            return False

        return False

    async def _try_graceful_cleanup(self, actor_handle, actor_name: str, config: ActorCleanupConfig) -> bool:
        """Try graceful cleanup via cleanup() method."""
        try:
            if hasattr(actor_handle, 'cleanup'):
                # Ray remote calls return ObjectRef, need to use ray.get() to await them
                cleanup_ref = actor_handle.cleanup.remote()
                cleanup_task = asyncio.create_task(asyncio.to_thread(ray.get, cleanup_ref))
                await asyncio.wait_for(cleanup_task, timeout=config.cleanup_timeout_seconds)

                # After graceful cleanup, still need to kill the actor
                ray.kill(actor_handle, no_restart=True)

                self._cleanup_stats["graceful_success"] += 1
                if config.log_individual_actors:
                    logger.info(f"✅ Gracefully cleaned up {actor_name}")
                return True
        except asyncio.TimeoutError:
            if config.log_individual_actors:
                logger.warning(f"⏰ Graceful cleanup timed out for {actor_name}")
        except Exception as e:
            if config.log_individual_actors:
                logger.warning(f"⚠️ Graceful cleanup failed for {actor_name}: {e}")

        return False

    async def _try_kill_cleanup(self, actor_handle_or_info, actor_name: str, config: ActorCleanupConfig) -> bool:
        """Try cleanup via ray.kill()."""
        try:
            # Handle both actor handles and actor info objects
            if hasattr(actor_handle_or_info, 'actor_id') and isinstance(actor_handle_or_info.actor_id, str):
                # This is actor info with hex actor_id
                from ray._raylet import ActorID
                actor_id = ActorID.from_hex(actor_handle_or_info.actor_id)
                ray.kill(actor_id, no_restart=True)
            else:
                # This is an actor handle
                ray.kill(actor_handle_or_info, no_restart=True)

            self._cleanup_stats["kill_success"] += 1
            if config.log_individual_actors:
                logger.info(f"🔥 Killed {actor_name}")
            return True

        except Exception as e:
            if config.log_individual_actors:
                logger.warning(f"❌ Failed to kill {actor_name}: {e}")
            return False

    def _reset_stats(self):
        """Reset cleanup statistics."""
        self._cleanup_stats = {
            "total_found": 0,
            "total_cleaned": 0,
            "graceful_success": 0,
            "kill_success": 0,
            "failures": 0,
            "namespaces_processed": set(),
        }

    def _log_cleanup_summary(self):
        """Log a summary of cleanup operations."""
        stats = self._cleanup_stats
        logger.info("🧹 Ray Actor Cleanup Summary:")
        logger.info(f"  📊 Found: {stats['total_found']}")
        logger.info(f"  ✅ Cleaned: {stats['total_cleaned']}")
        logger.info(f"  🤝 Graceful: {stats['graceful_success']}")
        logger.info(f"  🔥 Killed: {stats['kill_success']}")
        logger.info(f"  ❌ Failures: {stats['failures']}")
        logger.info(f"  📁 Namespaces: {list(stats['namespaces_processed'])}")


# Convenience functions for common use cases
async def cleanup_test_actors(namespace_pattern: str = "polymathera_test_*") -> Dict[str, Any]:
    """Clean up test actors with default configuration."""
    config = ActorCleanupConfig(
        namespace_patterns=[namespace_pattern],
        actor_types=["DistributedWorker"],
        actor_states=["ALIVE", "PENDING_CREATION"],
        strategy=CleanupStrategy.GRACEFUL_THEN_KILL,
    )

    cleanup_util = RayActorCleanupUtility()
    return await cleanup_util.cleanup_actors(config)


async def cleanup_worker_handles(worker_handles: List[ray.actor.ActorHandle]) -> Dict[str, Any]:
    """Clean up specific worker handles with graceful strategy."""
    config = ActorCleanupConfig(
        strategy=CleanupStrategy.GRACEFUL_THEN_KILL,
        max_concurrent_cleanups=len(worker_handles),  # Clean all at once
    )

    cleanup_util = RayActorCleanupUtility()
    return await cleanup_util.cleanup_actors(config, actor_handles=worker_handles)


async def emergency_cleanup_namespace(namespace: str) -> Dict[str, Any]:
    """Emergency cleanup of entire namespace - kills all actors immediately."""
    config = ActorCleanupConfig(
        exact_namespaces=[namespace],
        strategy=CleanupStrategy.KILL_ONLY,  # Fast cleanup
        log_individual_actors=True,
    )

    cleanup_util = RayActorCleanupUtility()
    return await cleanup_util.cleanup_actors(config)