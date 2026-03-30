"""Global key registry for blackboard-based page key management.

This module provides centralized management of page keys across all agents
via a global blackboard, enabling global attention mechanisms.
"""
from __future__ import annotations

import time
import logging
from typing import Any
import pickle

from .attention import PageKey
from ...base import Agent
from ....vcm.page_storage import PageStorage
from ...blackboard import EnhancedBlackboard
from ...blackboard.protocol import KeyRegistryProtocol


logger = logging.getLogger(__name__)



class GlobalPageKeyRegistry:
    """Manages global page key registry on blackboard and page storage to avoid recomputation.

    Keys are expensive to compute (require LLM inference or embedding),
    so we cache them for reuse.

    All agents can publish their page keys to a global blackboard,
    making them available for cross-cluster attention queries.

    Architecture:
    - Global blackboard scope: "page_keys"
    - Key format: "key:{page_id}" -> {key: PageKey, cluster_id: str, timestamp: float}
    - Cluster summary scope: "cluster_summaries"
    - Cluster format: "cluster:{cluster_id}" -> {summary: dict, representative_key: PageKey, ...}
    """

    def __init__(self, agent: Agent):
        """Initialize registry.

        Args:
            agent: Agent instance (for accessing blackboard)
        """
        self.agent: Agent = agent
        self.page_keys_blackboard: EnhancedBlackboard = None
        self.cluster_summaries_blackboard: EnhancedBlackboard = None
        self.page_storage: PageStorage | None = None

    async def initialize(self) -> None:
        """Initialize global blackboard for key registry."""
        self.page_keys_blackboard = await self.agent.get_colony_level_blackboard(
            namespace="page_keys"
        )
        self.cluster_summaries_blackboard = await self.agent.get_colony_level_blackboard(
            namespace="cluster_summaries"
        )
        self.page_storage: PageStorage = self.agent.get_page_storage()
        logger.info("GlobalPageKeyRegistry initialized with global blackboard")

    async def publish_page_key(
        self,
        page_id: str,
        key: PageKey,
        cluster_id: str,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Publish page key to global registry.

        Args:
            page_id: Page identifier
            key: Generated page key
            cluster_id: Cluster this page belongs to
            metadata: Optional additional metadata
        """
        try:
            await self.page_keys_blackboard.write(
                KeyRegistryProtocol.page_key(page_id),
                {
                    "key": key.model_dump(),
                    "cluster_id": cluster_id,
                    "syscontext": self.agent.syscontext.to_dict() if self.agent else None,
                    "timestamp": time.time(),
                    "agent_id": self.agent.agent_id,
                    "metadata": metadata or {}
                }
            )
            await self._set_page_key_in_page_graph(page_id, key, cluster_id)

            logger.debug(f"Published key for page {page_id} to global registry")

        except Exception as e:
            logger.error(f"Failed to publish key for {page_id}: {e}", exc_info=True)
            # Don't fail the agent if publishing fails

    async def get_page_key(self, page_id: str) -> tuple[str, PageKey, str] | None:
        """Get page key from global registry.

        Args:
            page_id: Page identifier
        Returns:
            (page_id, key, cluster_id) tuple, or None if not found
        """
        try:
            data = await self.page_keys_blackboard.read(
                KeyRegistryProtocol.page_key(page_id)
            )
            if data:
                return (page_id, PageKey(**data["key"]), data.get("cluster_id", "unknown"))

            return await self._get_page_key_from_page_graph(page_id)

        except Exception as e:
            logger.error(f"Failed to get page key for {page_id}: {e}", exc_info=True)
            return None

    async def publish_cluster_summary(
        self,
        cluster_id: str,
        summary: dict[str, Any],
        representative_key: PageKey,
        page_ids: list[str],
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Publish cluster-level summary for hierarchical attention.

        Args:
            cluster_id: Cluster identifier
            summary: Cluster synthesis summary
            representative_key: Representative key (centroid) for the cluster
            page_ids: List of page IDs in this cluster
            metadata: Optional additional metadata
        """
        try:
            await self.cluster_summaries_blackboard.write(
                KeyRegistryProtocol.cluster_key(cluster_id),
                {
                    "summary": summary,
                    "representative_key": representative_key.model_dump(),
                    "page_ids": page_ids,
                    "page_count": len(page_ids),
                    "timestamp": time.time(),
                    "agent_id": self.agent.agent_id,
                    "metadata": metadata or {}
                }
            )

            logger.info(
                f"Published cluster summary for {cluster_id} "
                f"({len(page_ids)} pages) to global registry"
            )

        except Exception as e:
            logger.error(
                f"Failed to publish cluster summary for {cluster_id}: {e}",
                exc_info=True
            )
            # Don't fail the agent if publishing fails

    async def get_all_keys(self) -> list[tuple[str, PageKey, str]]:
        """Get all page keys from global registry.

        Returns:
            List of (page_id, key, cluster_id) tuples
        """
        try:
            # Get all keys
            all_key_ids = await self.page_keys_blackboard.list_keys()  # TODO: This is too inefficient (except for RedisOM which filters by namespace).

            results = []
            for key_id in all_key_ids:
                page_id = self.page_keys_blackboard.parse_key_part(key_id, "page_id")
                if page_id is None:
                    continue

                data = await self.page_keys_blackboard.read(key_id)
                if data:
                    key = PageKey(**data["key"])
                    cluster_id = data.get("cluster_id", "unknown")
                    results.append((page_id, key, cluster_id))

            logger.debug(f"Retrieved {len(results)} keys from global registry")
            return results

        except Exception as e:
            logger.error(f"Failed to get all keys: {e}", exc_info=True)
            return []

    async def get_keys_for_cluster(self, cluster_id: str) -> list[tuple[str, PageKey]]:
        """Get page keys for a specific cluster.

        Args:
            cluster_id: Cluster identifier

        Returns:
            List of (page_id, key) tuples for this cluster
        """
        all_keys = await self.get_all_keys()
        return [
            (page_id, key)
            for page_id, key, cid in all_keys
            if cid == cluster_id
        ]

    async def get_keys_for_clusters(self, cluster_ids: list[str]) -> list[tuple[str, PageKey, str]]:
        """Get page keys for multiple clusters.

        Args:
            cluster_ids: List of cluster identifiers

        Returns:
            List of (page_id, key, cluster_id) tuples for these clusters
        """
        all_keys = await self.get_all_keys()
        cluster_set = set(cluster_ids)
        return [
            (page_id, key, cid)
            for page_id, key, cid in all_keys
            if cid in cluster_set
        ]

    async def get_all_cluster_summaries(self) -> list[tuple[str, dict, PageKey]]:
        """Get all cluster summaries from global registry.

        Returns:
            List of (cluster_id, summary, representative_key) tuples
        """
        try:
            # Get all cluster summaries
            all_cluster_ids = await self.cluster_summaries_blackboard.list_keys()

            results = []
            for cluster_key in all_cluster_ids:
                cluster_id = self.cluster_summaries_blackboard.parse_key_part(cluster_key, "cluster_id")
                if cluster_id is None:
                    continue

                data = await self.cluster_summaries_blackboard.read(cluster_key)
                if data:
                    summary = data.get("summary", {})
                    representative_key = PageKey(**data["representative_key"])
                    results.append((cluster_id, summary, representative_key))

            logger.debug(f"Retrieved {len(results)} cluster summaries from global registry")
            return results

        except Exception as e:
            logger.error(f"Failed to get cluster summaries: {e}", exc_info=True)
            return []

    async def get_cluster_summary(self, cluster_id: str) -> tuple[dict, PageKey] | None:
        """Get summary for a specific cluster.

        Args:
            cluster_id: Cluster identifier

        Returns:
            (summary, representative_key) tuple, or None if not found
        """
        try:
            data = await self.cluster_summaries_blackboard.read(
                KeyRegistryProtocol.cluster_key(cluster_id),
            )
            if data:
                summary = data.get("summary", {})
                representative_key = PageKey(**data["representative_key"])
                return (summary, representative_key)

            return None

        except Exception as e:
            logger.error(f"Failed to get cluster summary for {cluster_id}: {e}", exc_info=True)
            return None

    def _get_page_data_key(self, page_id: str) -> str:
        """Get VCM page data key for page key in page storage."""
        return f"page_key_{page_id}"

    async def _get_page_key_from_page_graph(self, page_id: str) -> PageKey | None:
        """Get cached key for page."""
        # Retrieve from VCM page storage
        try:
            return await self.page_storage.retrieve_page_graph_level_data(
                data_key=self._get_page_data_key(page_id),
            )
        except Exception as e:
            logger.warning(f"Failed to retrieve key from VCM for {page_id}: {e}")
        return None

    async def _set_page_key_in_page_graph(self, page_id: str, page_key: PageKey, cluster_id: str) -> None:
        """Cache key for page."""
        # Store in VCM page storage
        try:
            await self.page_storage.store_page_graph_level_data(  # TODO: Page keys are not really graph-level data, we should add a separate method for page-level data
                data_key=self._get_page_data_key(page_id),
                graph_data=page_key,
            )
            logger.debug(f"Cached key for {page_id} in VCM")
        except Exception as e:
            logger.error(f"Failed to cache key in VCM for {page_id}: {e}")

    async def _delete_page_key_from_page_graph(self, page_id: str) -> None:
        """Delete cached key."""
        # VCM page_storage doesn't have a delete method for page graphs
        # We could implement this by storing a tombstone or just skip deletion
        logger.debug(f"Delete not implemented for VCM backend (page_id={page_id})")

    async def _clear_page_keys_from_page_graph(self) -> None:
        """Clear all cached keys."""
        # VCM backend doesn't support bulk clear
        logger.warning("Clear not implemented for VCM backend")




