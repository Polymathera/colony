"""Agent-facing read surface over the master §6.4 retrieval modes.

Curation lives in :class:`KnowledgeCuratorCapability`; acquisition
runs through :meth:`RepoStateProvider.ingest_repo_map_literature`
against the design monorepo's unified ``.colony/repo_map.yaml``. This
file is the third leg of the agent-driven knowledge trio —
*retrieval*, exposed as plain ``@action_executor`` methods so any
agent can ground a chat answer in the corpus.

The capability holds **zero** retrieval state. It owns a per-mode
retriever cache; the retrievers themselves (``ScopedRetrievalCapability`` &
co.) live in :mod:`polymathera.colony.knowledge.retrieval` and do all
the actual work.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, TYPE_CHECKING

from overrides import override

from ...base import AgentCapability
from ...blueprint import Blueprint
from ...models import AgentSuspensionState
from ...scopes import BlackboardScope, get_scope_prefix
from ..actions import action_executor


if TYPE_CHECKING:
    from ...base import Agent
    from ....knowledge.retrieval import RetrievalCapability, RetrievalDeps


logger = logging.getLogger(__name__)


_RetrieverName = Literal["scoped", "grounded", "graph", "budgeted", "standards"]


class KnowledgeRetrievalCapability(AgentCapability):
    """Agent capability: search the knowledge base.

    Five action methods, one per retrieval mode (master §6.4). Each is
    a thin wrapper around the corresponding ``RetrievalCapability``.

    Args:
        agent: Owning agent. Required (the capability is not useful
            detached — its primary consumer is the agent's planner).
        scope: Blackboard scope for capability-local writes.
        deps: ``RetrievalDeps`` (embedder + vector store + optional
            graph store). The caller is responsible for construction —
            typically the same bundle the colony's ``Ingestor`` uses,
            so curation and retrieval share an embedding space.
        default_retriever_name: Mode used by ``search_knowledge`` when
            the caller does not pick one. ``"scoped"`` is a safe
            default for tool-use agents.
        capability_key: Action-policy dispatch key.
        app_name: ``serving`` application name override.
    """

    def __init__(
        self,
        agent: "Agent",
        scope: BlackboardScope = BlackboardScope.SESSION,
        namespace: str = "knowledge_retrieval",
        *,
        deps: "RetrievalDeps | Blueprint",
        default_retriever_name: _RetrieverName = "scoped",
        capability_key: str = "knowledge_retrieval",
        app_name: str | None = None,
    ):
        # ``deps`` accepts either a real :class:`RetrievalDeps` (tests
        # / in-process) or a :class:`Blueprint` for it (cross-Ray
        # construction via ``default_retrieval_deps_blueprint()``);
        # same shape as ``KnowledgeCuratorCapability(ingestor=…)``.
        super().__init__(
            agent=agent,
            scope_id=get_scope_prefix(scope, agent, namespace=namespace),
            capability_key=capability_key,
            app_name=app_name,
        )
        self._deps = deps.local_instance() if isinstance(deps, Blueprint) else deps
        self._default_retriever_name: _RetrieverName = default_retriever_name
        # Lazy retriever cache — instantiated only for modes the agent
        # actually calls. Keeps the import surface small for agents
        # that only use ``search_knowledge``.
        self._retrievers: dict[str, "RetrievalCapability"] = {}

    def get_action_group_description(self) -> str:
        return (
            "Knowledge retrieval — search the colony's knowledge base "
            "(curated corpora ingested by KnowledgeCuratorCapability "
            "and the design-monorepo ingest_repo_map_literature "
            "action). Five modes are available via the ``mode`` "
            "argument: ``scoped`` (single-source), ``grounded`` (with "
            "citations), ``graph`` (knowledge graph), ``budgeted`` "
            "(token-bounded), ``standards`` (time-versioned "
            "regulatory). Default is "
            f"``{self._default_retriever_name}``."
        )

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"knowledge", "retrieval"})

    @override
    async def serialize_suspension_state(
        self, state: AgentSuspensionState,
    ) -> AgentSuspensionState:
        return state

    @override
    async def deserialize_suspension_state(
        self, state: AgentSuspensionState,
    ) -> None:
        return None

    # -------- internal --------------------------------------------------

    def _get_retriever(self, name: str) -> "RetrievalCapability":
        if name in self._retrievers:
            return self._retrievers[name]
        from ....knowledge.retrieval import (
            BudgetedRetrievalCapability,
            GraphRetrievalCapability,
            GroundedRetrievalCapability,
            ScopedRetrievalCapability,
            StandardsRetrievalCapability,
        )
        registry: dict[str, type[AgentCapability]] = {
            "scoped": ScopedRetrievalCapability,
            "grounded": GroundedRetrievalCapability,
            "graph": GraphRetrievalCapability,
            "budgeted": BudgetedRetrievalCapability,
            "standards": StandardsRetrievalCapability,
        }
        cls = registry.get(name)
        if cls is None:
            raise ValueError(
                f"unknown retrieval retriever {name!r}; choose one of "
                f"{sorted(registry)}",
            )
        # The per-mode capabilities are plumbing for this aggregator —
        # they share its scope_id so they don't need their own scope
        # resolution (which would require an active execution_context
        # at construction time even when this aggregator is being used
        # detached, e.g. in tests).
        retriever = cls(
            agent=self._agent,
            deps=self._deps,
            scope_id=f"{self.scope_id}:{name}",
            capability_key=f"{self.capability_key}:{name}",
        )
        self._retrievers[name] = retriever
        return retriever

    @staticmethod
    def _build_query(
        *,
        text: str,
        graph_query: str | None,
        source_prefix: str | None,
        data_types: list[str] | None,
        top_k: int,
        max_tokens: int | None,
        require_citations: bool,
    ):
        from ....knowledge.models import RetrievalQuery
        return RetrievalQuery(
            text=text,
            graph_query=graph_query,
            data_types=tuple(data_types or ()),
            source_prefix=source_prefix,
            max_results=top_k,
            max_tokens=max_tokens,
            require_citations=require_citations,
        )

    # -------- action surface --------------------------------------------

    @action_executor()
    async def search_knowledge(
        self,
        *,
        query: str,
        mode: _RetrieverName | None = None,
        source_prefix: str | None = None,
        data_types: list[str] | None = None,
        top_k: int = 8,
        max_tokens: int | None = None,
        require_citations: bool = False,
        graph_query: str | None = None,
    ) -> dict[str, Any]:
        """Search the knowledge base.

        ``mode`` selects the master §6.4 retrieval mode. Defaults to
        the ``default_retriever_name`` set at construction time
        (``"scoped"`` unless overridden).

        Returns the typed ``RetrievalResult`` JSON-serialised — a dict
        with ``mode``, ``hits[]`` (each hit carries a ``Chunk``,
        ``score``, ``rank``, ``explanation``) and ``total_candidates``.
        """

        retriever_name = mode or self._default_retriever_name
        retriever = self._get_retriever(retriever_name)
        retrieval_query = self._build_query(
            text=query,
            graph_query=graph_query,
            source_prefix=source_prefix,
            data_types=data_types,
            top_k=top_k,
            max_tokens=max_tokens,
            require_citations=require_citations,
        )
        result = await retriever.run(retrieval_query)
        return result.model_dump(mode="json")

    @action_executor()
    async def list_modes(self) -> dict[str, Any]:
        """List the retrieval modes this agent can call."""
        return {
            "modes": ["scoped", "grounded", "graph", "budgeted", "standards"],
            "default": self._default_retriever_name,
        }


__all__ = ("KnowledgeRetrievalCapability",)
