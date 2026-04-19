from __future__ import annotations

from typing import Any, TYPE_CHECKING

import networkx as nx

from polymathera.colony.agents.base import Agent
from polymathera.colony.agents.patterns.capabilities.prefetching import PrefetchPolicy

if TYPE_CHECKING:
    pass


class FeedbackLoopPrefetchPolicy(PrefetchPolicy):
    """PrefetchPolicy implementation for feedback loop predictions.

    Wraps FeedbackLoopPredictor logic in the PrefetchPolicy interface,
    enabling integration with CompositePrefetchPolicy and other policies.

    Usage:
        policy = FeedbackLoopPrefetchPolicy(
            agent=agent,
            prefetch_depth=2,
            prefetch_test_pages=True,
        )

        # Call predict_needed_pages with context specifying mode
        pages = await policy.predict_needed_pages(
            current_context={
                "mode": "cross_agent",  # or "self_critique", "hypothesis_game"
                "page_id": "some_page_id",
                "role": "proposer",  # for hypothesis_game mode
            },
            page_graph=graph,
        )
    """

    def __init__(
        self,
        agent: Agent,
        prefetch_depth: int = 2,
        prefetch_test_pages: bool = True,
        max_prefetch: int = 10,
    ):
        """Initialize feedback loop prefetch policy.

        Args:
            agent: Agent instance for loading page graph
            prefetch_depth: Max hops to look for related pages
            prefetch_test_pages: Include test pages in predictions
            max_prefetch: Maximum pages to prefetch
        """
        self.agent = agent
        self.prefetch_depth = prefetch_depth
        self.prefetch_test_pages = prefetch_test_pages
        self.max_prefetch = max_prefetch
        # Internal predictor for the actual logic
        self._predictor: FeedbackLoopPredictor | None = None

    async def predict_needed_pages(
        self,
        current_context: dict[str, Any],
        page_graph: nx.DiGraph,
    ) -> list[str]:
        """Predict pages needed based on context.

        Dispatches to appropriate prediction method based on context["mode"]:
        - "self_critique": Pages needed for self-critique
        - "hypothesis_game": Pages needed for hypothesis game role
        - "cross_agent": Pages that might need refinement
        - "gap_filling": Pages that might contain missing component

        Args:
            current_context: Context dict with:
                - mode: Prediction mode
                - page_id: Page being analyzed
                - role: Game role (for hypothesis_game mode)
                - missing_component: Component name (for gap_filling mode)
                - known_pages: Already analyzed pages (for gap_filling mode)
            page_graph: Page dependency graph

        Returns:
            List of page IDs to prefetch
        """
        if self._predictor is None:
            self._predictor = FeedbackLoopPredictor(
                agent=self.agent,
                prefetch_depth=self.prefetch_depth,
                prefetch_test_pages=self.prefetch_test_pages,
            )

        mode = current_context.get("mode", "cross_agent")
        page_id = current_context.get("page_id")

        if not page_id:
            return []

        if mode == "self_critique":
            return await self._predictor.predict_self_critique_pages(page_id)
        elif mode == "hypothesis_game":
            role = current_context.get("role", "proposer")
            return await self._predictor.predict_hypothesis_game_pages(page_id, role)
        elif mode == "cross_agent":
            return await self._predictor.predict_cross_agent_pages(page_id)
        elif mode == "gap_filling":
            missing_component = current_context.get("missing_component", "")
            known_pages = current_context.get("known_pages", set())
            return await self._predictor.predict_gap_filling_pages(missing_component, known_pages)
        else:
            # Default to cross-agent prediction
            return await self._predictor.predict_cross_agent_pages(page_id)


class FeedbackLoopPredictor:
    """Predicts pages needed by feedback loops to minimize cache misses.

    Uses page graph to predict which pages will be needed during:
    - Self-critique (imports, tests for the page)
    - Hypothesis game (evidence pages for different roles)
    - Cross-agent refinement (related pages)

    This enables prefetching pages before feedback loops run.
    """

    def __init__(
        self,
        agent: Agent,
        prefetch_depth: int = 2,
        prefetch_test_pages: bool = True,
        max_prefetch: int = 5,
    ):
        """Initialize feedback loop predictor.

        Args:
            agent: Agent instance
            prefetch_depth: Max hops to look for related pages
            prefetch_test_pages: Include test pages in predictions
            max_prefetch: Maximum pages to return per prediction
        """
        self.agent = agent
        self.prefetch_depth = prefetch_depth
        self.prefetch_test_pages = prefetch_test_pages
        self.max_prefetch = max_prefetch

    async def predict_self_critique_pages(self, page_id: str) -> list[str]:
        """Predict pages needed during self-critique.

        Self-critique often needs:
        - Imported modules (for type/behavior verification)
        - Test files (for behavior verification)

        Args:
            page_id: Page being analyzed

        Returns:
            List of page IDs to prefetch
        """
        predicted = []

        page_graph = await self.agent.load_page_graph()

        if not page_graph.has_node(page_id):
            return predicted

        # Get imports (predecessors in dependency graph)
        predecessors = list(page_graph.predecessors(page_id))
        predicted.extend(predecessors[:self.prefetch_depth])

        # Get test pages if enabled
        if self.prefetch_test_pages:
            for node in page_graph.nodes():
                node_data = page_graph.nodes[node]
                # Check if this is a test page that tests our page
                if node_data.get("is_test", False):
                    # Check if test has edge to our page
                    if page_graph.has_edge(node, page_id):
                        predicted.append(node)

        return predicted[:self.max_prefetch]  # Limit to avoid over-prefetching

    async def predict_hypothesis_game_pages(self, page_id: str, role: str) -> list[str]:
        """Predict pages needed for hypothesis game roles.

        Different roles need different pages:
        - Proposer: evidence pages (tests, docs)
        - Skeptic: counter-example pages (alternative implementations)
        - Grounder: documentation/spec pages
        - Arbiter: high-level context pages

        Args:
            page_id: Page being validated
            role: Game role (proposer, skeptic, grounder, arbiter)

        Returns:
            List of page IDs to prefetch for this role
        """
        predicted = []

        page_graph = await self.agent.load_page_graph()

        if not page_graph.has_node(page_id):
            return predicted

        if role == "proposer":
            # Proposer needs evidence - tests and docs
            for node in page_graph.nodes():
                node_data = page_graph.nodes[node]
                if node_data.get("is_test", False) or node_data.get("is_doc", False):
                    if page_graph.has_edge(node, page_id):
                        predicted.append(node)

        elif role == "skeptic":
            # Skeptic needs alternative implementations - siblings in hierarchy
            for neighbor in page_graph.neighbors(page_id):
                predicted.append(neighbor)

        elif role == "grounder":
            # Grounder needs documentation pages
            for node in page_graph.nodes():
                node_data = page_graph.nodes[node]
                if node_data.get("is_doc", False):
                    predicted.append(node)

        elif role == "arbiter":
            # Arbiter needs high-level context (high-degree nodes)
            central_nodes = sorted(
                page_graph.nodes(),
                key=lambda n: page_graph.degree(n),
                reverse=True
            )[:self.max_prefetch]
            predicted.extend(central_nodes)

        return predicted[:self.max_prefetch]  # Limit to avoid over-prefetching

    async def predict_cross_agent_pages(self, source_page: str) -> list[str]:
        """Predict pages that might need refinement based on this result.

        When one page's analysis completes, related pages may need refinement.

        Args:
            source_page: Page that just completed analysis

        Returns:
            List of page IDs that might need refinement
        """
        predicted = []

        page_graph = await self.agent.load_page_graph()

        if not page_graph.has_node(source_page):
            return predicted

        # Get direct dependents (pages that depend on source)
        successors = list(page_graph.successors(source_page))
        predicted.extend(successors)

        # Get pages with shared dependencies (likely related functionality)
        my_deps = set(page_graph.predecessors(source_page))
        for node in page_graph.nodes():
            if node == source_page:
                continue
            other_deps = set(page_graph.predecessors(node))
            shared = my_deps & other_deps
            if len(shared) >= 2:  # At least 2 shared dependencies
                predicted.append(node)

        return predicted[:self.max_prefetch]  # Limit to avoid over-prefetching

    async def predict_gap_filling_pages(
        self,
        missing_component: str,
        known_pages: set[str]
    ) -> list[str]:
        """Predict pages that might contain a missing component.

        When analysis has missing context, predict where to look.

        Args:
            missing_component: Name of missing component
            known_pages: Pages already analyzed

        Returns:
            List of page IDs to search for missing component
        """
        predicted = []

        page_graph = await self.agent.load_page_graph()

        # Check page metadata for component mentions
        for node in page_graph.nodes():
            if node in known_pages:
                continue
            node_data = page_graph.nodes[node]
            # Check if page might contain the missing component
            file_path = node_data.get("file_path", "")
            if missing_component.lower() in file_path.lower():
                predicted.append(node)

        return predicted[:self.max_prefetch]

