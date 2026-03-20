"""Attention policies for controlling query routing strategies.

This module provides policy-based attention strategies to balance efficiency
and effectiveness when routing queries to relevant pages.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from overrides import override
import logging

from ...base import Agent
from .attention import PageQuery, AttentionScore, AttentionScoringMechanism
from .key_registry import GlobalPageKeyRegistry
from ...models import AttentionContext

logger = logging.getLogger(__name__)


class AttentionPolicy(ABC):
    """Policy for computing attention with configurable granularity.

    Different implementations provide different trade-offs:
    - HierarchicalAttentionPolicy: Two-level (cluster → page) for efficiency
    - GlobalAttentionPolicy: Flat search across all pages (complete but expensive)
    - LocalAttentionPolicy: Constrained to specific scope (fast but limited)
    """

    def __init__(self, agent: Agent):
        """Initialize hierarchical attention policy.

        Args:
            agent: Agent instance for accessing keys/summaries and attention mechanisms
        """
        self.agent = agent
        self.key_registry = GlobalPageKeyRegistry(agent)

    async def initialize(self):
        """Initialize any async components (e.g., key registry)."""
        await self.key_registry.initialize()

    @abstractmethod
    async def find_relevant_pages(
        self,
        query: PageQuery,
        context: AttentionContext
    ) -> list[AttentionScore]:
        """Find relevant pages using this policy's strategy.

        Args:
            query: Query to match against pages
            context: Additional context (source_agent, source_cluster, etc.)

        Returns:
            List of attention scores for relevant pages
        """
        ...


class HierarchicalAttentionPolicy(AttentionPolicy):
    """Two-level hierarchical attention: cluster-level → page-level.

    First computes coarse-grained attention among clusters using representative
    keys (cluster centroids), then fine-grained attention within top-K clusters.

    This balances efficiency (fewer LLM calls) with effectiveness (finds relevant
    pages across clusters).

    Architecture:
    1. Query against cluster summaries (coarse-grained)
    2. Identify top-K relevant clusters
    3. Query against pages within those clusters (fine-grained)
    4. Return top-N pages overall
    """

    def __init__(
        self,
        agent: Agent,
        attention_mechanism: AttentionScoringMechanism,
        top_k_clusters: int = 5,
        top_n_pages_per_cluster: int = 3,
        top_n_pages_overall: int = 10
    ):
        """Initialize hierarchical attention policy.

        Args:
            agent: Agent instance for accessing keys/summaries and attention mechanisms
            attention_mechanism: Attention mechanism for computing Q·K
            top_k_clusters: Number of top clusters to search within
            top_n_pages_per_cluster: Max pages to return per cluster
            top_n_pages_overall: Max pages to return overall
        """
        super().__init__(agent)
        self.attention_mechanism = attention_mechanism
        self.top_k_clusters = top_k_clusters
        self.top_n_pages_per_cluster = top_n_pages_per_cluster
        self.top_n_pages_overall = top_n_pages_overall

    @override
    async def find_relevant_pages(
        self,
        query: PageQuery,
        context: AttentionContext
    ) -> list[AttentionScore]:
        """Find relevant pages using hierarchical attention.

        Args:
            query: Query to match
            context: Context with optional source_cluster to skip

        Returns:
            List of attention scores for relevant pages
        """
        # Level 1: Coarse-grained attention among clusters
        cluster_summaries = await self.key_registry.get_all_cluster_summaries()

        if not cluster_summaries:
            # No cluster summaries yet, fall back to global attention
            logger.warning("No cluster summaries available, falling back to global attention")
            return await self._fallback_global_attention(query, context)

        # Extract representative keys for each cluster
        cluster_keys = [representative_key for _, _, representative_key in cluster_summaries]

        # Compute attention against cluster keys
        relevant_clusters = await self.attention_mechanism.score_attention(
            query=query,
            keys=cluster_keys
        )

        logger.info(
            f"Hierarchical attention: found {len(relevant_clusters)} relevant clusters "
            f"for query '{query.query_text[:50]}...'"
        )

        # Level 2: Fine-grained attention within top-K clusters
        all_relevant_pages = []
        source_cluster = context.get("source_cluster")

        # TODO: Parallelize this loop and use a semaphore to limit concurrency at LLM level.
        for cluster_score in relevant_clusters[:self.top_k_clusters]:
            # Get cluster_id from the representative key
            cluster_id = cluster_score.page_id

            # Skip source cluster to avoid redundant queries
            if cluster_id == source_cluster:
                continue

            # Get all page keys for this cluster
            cluster_pages = await self.key_registry.get_keys_for_cluster(cluster_id)

            if not cluster_pages:
                continue

            # Extract keys
            page_keys = [key for _, key in cluster_pages]

            # Compute attention within this cluster
            relevant_pages = await self.attention_mechanism.score_attention(
                query=query,
                keys=page_keys
            )

            # Take top-N pages from this cluster
            all_relevant_pages.extend(relevant_pages[:self.top_n_pages_per_cluster])

            logger.debug(
                f"Cluster {cluster_id}: found {len(relevant_pages)} relevant pages, "
                f"taking top {min(len(relevant_pages), self.top_n_pages_per_cluster)}"
            )

        # Sort by score and return top-N overall
        all_relevant_pages.sort(key=lambda x: x.score, reverse=True)
        result = all_relevant_pages[:self.top_n_pages_overall]

        logger.info(
            f"Hierarchical attention: returning {len(result)} pages "
            f"from {self.top_k_clusters} clusters"
        )

        return result

    async def _fallback_global_attention(
        self,
        query: PageQuery,
        context: AttentionContext
    ) -> list[AttentionScore]:
        """Fallback to global attention if no cluster summaries available."""
        global_policy = GlobalAttentionPolicy(
            key_registry=self.key_registry,
            attention_mechanism=self.attention_mechanism,
            top_n_pages=self.top_n_pages_overall
        )
        return await global_policy.find_relevant_pages(query, context)


class GlobalAttentionPolicy(AttentionPolicy):
    """Flat global attention: query against ALL pages.

    Simplest strategy - query against every page key in the system.
    Complete but expensive (many LLM calls for large codebases).

    Use when:
    - Codebase is small (< 1000 pages)
    - Need exhaustive search
    - Don't have cluster summaries yet
    """

    def __init__(
        self,
        agent: Agent,
        attention_mechanism: AttentionScoringMechanism,
        top_n_pages: int = 10
    ):
        """Initialize global attention policy.

        Args:
            agent: Agent instance for accessing keys/summaries and attention mechanisms
            attention_mechanism: Attention mechanism for computing Q·K
            top_n_pages: Max pages to return
        """
        super().__init__(agent)
        self.attention_mechanism = attention_mechanism
        self.top_n_pages = top_n_pages

    @override
    async def find_relevant_pages(
        self,
        query: PageQuery,
        context: AttentionContext
    ) -> list[AttentionScore]:
        """Find relevant pages using global attention.

        Args:
            query: Query to match
            context: Context (unused for global policy)

        Returns:
            List of attention scores for relevant pages
        """
        # Get all page keys from global registry
        all_keys_data = await self.key_registry.get_all_keys()

        if not all_keys_data:
            logger.warning("No page keys available in global registry")
            return []

        # Extract just the keys
        all_keys = [key for _, key, _ in all_keys_data]

        logger.info(
            f"Global attention: querying against {len(all_keys)} pages "
            f"for '{query.query_text[:50]}...'"
        )

        # Compute attention against all keys
        relevant_pages = await self.attention_mechanism.score_attention(
            query=query,
            keys=all_keys
        )

        # Return top-N
        result = relevant_pages[:self.top_n_pages]

        logger.info(f"Global attention: returning {len(result)} pages")

        return result


class LocalAttentionPolicy(AttentionPolicy):
    """Constrained local attention: only within specified scope.

    Queries only within specified clusters or pages. Fast but limited to
    predefined scope.

    Use when:
    - Know which clusters are relevant (e.g., from prior analysis)
    - Want to limit search to specific subsystem
    - Need fast response (fewer pages to query)
    """

    def __init__(
        self,
        agent: Agent,
        attention_mechanism: AttentionScoringMechanism,
        scope_clusters: list[str] | None = None,
        top_n_pages: int = 10
    ):
        """Initialize local attention policy.

        Args:
            agent: Agent instance for accessing keys/summaries and attention mechanisms
            attention_mechanism: Attention mechanism for computing Q·K
            scope_clusters: List of cluster IDs to search within (None = use context)
            top_n_pages: Max pages to return
        """
        super().__init__(agent)
        self.attention_mechanism = attention_mechanism
        self.scope_clusters = scope_clusters
        self.top_n_pages = top_n_pages

    @override
    async def find_relevant_pages(
        self,
        query: PageQuery,
        context: AttentionContext
    ) -> list[AttentionScore]:
        """Find relevant pages using local attention.

        Args:
            query: Query to match
            context: Context with optional scope_clusters override

        Returns:
            List of attention scores for relevant pages
        """
        # Determine scope (from init or context)
        scope_clusters = context.scope_clusters or self.scope_clusters

        if not scope_clusters:
            logger.warning("No scope clusters specified for local attention")
            return []

        # Get keys for specified clusters
        keys_data = await self.key_registry.get_keys_for_clusters(scope_clusters)

        if not keys_data:
            logger.warning(f"No keys found for clusters: {scope_clusters}")
            return []

        # Extract keys
        keys = [key for _, key, _ in keys_data]

        logger.info(
            f"Local attention: querying {len(keys)} pages in clusters {scope_clusters} "
            f"for '{query.query_text[:50]}...'"
        )

        # Compute attention
        relevant_pages = await self.attention_mechanism.score_attention(
            query=query,
            keys=keys
        )

        # Return top-N
        result = relevant_pages[:self.top_n_pages]

        logger.info(f"Local attention: returning {len(result)} pages")

        return result

