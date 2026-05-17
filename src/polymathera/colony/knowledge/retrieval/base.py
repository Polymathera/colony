"""Common base for the five retrieval-mode tool capabilities.

Every retrieval mode is a :class:`LocalToolCapability` subclass that
takes a :class:`RetrievalDeps` bundle (an embedder, a vector store,
and optional graph / image stores). The base supplies the single
public :func:`retrieve` action; subclasses only override
:meth:`run` to produce a :class:`RetrievalResult`.

The action signature mirrors :class:`~polymathera.colony.knowledge.models.RetrievalQuery`
so the LLM planner can pass the natural flat kwargs without having
to assemble a typed object first.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Mapping
from datetime import datetime
from typing import Any, TYPE_CHECKING

from overrides import override

from ...agents.base import AgentCapability
from ...agents.blueprint import Blueprint, blueprint
from ...agents.patterns.actions import action_executor
from ...agents.patterns.capabilities.tool import LocalToolCapability
from ...agents.scopes import BlackboardScope
from ..embedder import Embedder
from ..models import RetrievalQuery, RetrievalResult
from ..stores import GraphStore, ImageStore, VectorStore


if TYPE_CHECKING:
    from ...agents.base import Agent


logger = logging.getLogger(__name__)


@blueprint
class RetrievalDeps:
    """Dependencies shared across retrieval-mode capabilities.

    ``@blueprint`` adds a pickleable ``.bind()``. ``embedder`` /
    ``vector_store`` / ``graph_store`` / ``image_store`` accept
    either a real instance or a :class:`Blueprint` — resolved via
    ``local_instance()`` here so the same shape works in tests and
    across the Ray boundary.
    """

    def __init__(
        self,
        *,
        embedder: Embedder | Blueprint,
        vector_store: VectorStore | Blueprint,
        graph_store: GraphStore | Blueprint | None = None,
        image_store: ImageStore | Blueprint | None = None,
    ) -> None:
        self.embedder = (
            embedder.local_instance() if isinstance(embedder, Blueprint) else embedder
        )
        self.vector_store = (
            vector_store.local_instance()
            if isinstance(vector_store, Blueprint)
            else vector_store
        )
        self.graph_store = (
            graph_store.local_instance()
            if isinstance(graph_store, Blueprint)
            else graph_store
        )
        self.image_store = (
            image_store.local_instance()
            if isinstance(image_store, Blueprint)
            else image_store
        )


class RetrievalCapability(LocalToolCapability):
    """Base class for the five retrieval-mode tool capabilities.

    Subclasses set the class-level :class:`~polymathera.colony.tools.ToolSpec`
    (inherited contract from :class:`ToolCapability`) and override
    :meth:`run` to produce a :class:`RetrievalResult`. The base
    supplies the single :func:`retrieve` ``@action_executor`` method
    — the LLM-visible surface — and handles parameter parsing,
    error wrapping, and serialisation.
    """

    _TOOL_CAPABILITY_ABSTRACT = True

    mode: str = ""
    """Stable per-mode identifier surfaced in ``RetrievalResult.mode``.
    Subclasses MUST override (otherwise the result string is empty)."""

    def __init__(
        self,
        agent: "Agent | None" = None,
        # Retrieval results are naturally session-scoped (the same
        # query from two agents in the same session should hit the
        # same vector-store result set). ``SESSION`` also works when
        # ``agent is None`` (detached mode) — useful for the
        # aggregator ``KnowledgeRetrievalCapability`` that owns the
        # five mode-instances on the same agent.
        scope: BlackboardScope = BlackboardScope.SESSION,
        namespace: str | None = None,
        *,
        deps: "RetrievalDeps | Blueprint",
        scope_id: str | None = None,
        capability_key: str | None = None,
        app_name: str | None = None,
    ) -> None:
        super().__init__(
            agent=agent,
            scope=scope,
            namespace=namespace,
            scope_id=scope_id,
            capability_key=capability_key,
            app_name=app_name,
        )
        self._deps: RetrievalDeps = (
            deps.local_instance() if isinstance(deps, Blueprint) else deps
        )

    @override
    def _domain_tags(self) -> frozenset[str]:
        return frozenset({"knowledge", "retrieval"})

    @abstractmethod
    async def run(self, query: RetrievalQuery) -> RetrievalResult:
        """Per-mode retrieval body. Subclasses MUST implement."""

    @action_executor(
        planning_summary=(
            "Search the knowledge base in this retrieval mode. Returns "
            "a typed RetrievalResult (hits, total_candidates, extra)."
        ),
    )
    async def retrieve(
        self,
        *,
        text: str = "",
        source_prefix: str | None = None,
        top_k: int | None = None,
        graph_query: str | None = None,
        data_types: list[str] | None = None,
        tiers: list[str] | None = None,
        effective_at: str | None = None,
        citation_required: bool = False,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run this retrieval mode against the configured stores.

        Every field maps to :class:`RetrievalQuery`; unset / irrelevant
        fields are ignored by the mode. Returns the typed
        :class:`RetrievalResult` as a JSON dict so the LLM planner can
        consume hits directly.
        """
        try:
            query = _build_query_from_kwargs(
                text=text,
                source_prefix=source_prefix,
                top_k=top_k,
                graph_query=graph_query,
                data_types=data_types,
                tiers=tiers,
                effective_at=effective_at,
                citation_required=citation_required,
                max_tokens=max_tokens,
                extra=extra,
            )
        except Exception as exc:  # noqa: BLE001 — typed input errors
            return _error_dict(
                tool=type(self).spec.name,
                error=f"invalid retrieval parameters: {exc}",
            )
        try:
            result = await self.run(query)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "RetrievalCapability(%s): retrieval failed",
                type(self).__name__,
            )
            return _error_dict(
                tool=type(self).spec.name,
                error=f"{type(exc).__name__}: {exc}",
            )
        return result.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_query_from_kwargs(
    *,
    text: str,
    source_prefix: str | None,
    top_k: int | None,
    graph_query: str | None,
    data_types: list[str] | None,
    tiers: list[str] | None,
    effective_at: str | None,
    citation_required: bool,
    max_tokens: int | None,
    extra: dict[str, Any] | None,
) -> RetrievalQuery:
    payload: dict[str, Any] = {}
    if text:
        payload["text"] = text
    if graph_query is not None:
        payload["graph_query"] = graph_query
    if source_prefix is not None:
        payload["source_prefix"] = source_prefix
    if top_k is not None:
        payload["top_k"] = top_k
    if data_types:
        payload["data_types"] = tuple(data_types)
    if tiers:
        payload["tiers"] = tuple(tiers)
    if effective_at is not None:
        payload["effective_at"] = datetime.fromisoformat(effective_at)
    if citation_required:
        payload["citation_required"] = True
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if extra:
        payload["extra"] = dict(extra)
    return RetrievalQuery.model_validate(payload)


def _error_dict(*, tool: str, error: str) -> dict[str, Any]:
    """RetrievalResult-shaped error payload (empty hits + error in extra)."""
    return {
        "mode": tool,
        "hits": [],
        "total_candidates": 0,
        "extra": {"error": error},
    }



__all__ = ("RetrievalCapability", "RetrievalDeps")


# Quick sanity import-check: every ToolCapability is also an
# AgentCapability — guard against an MRO change in the bases.
assert issubclass(RetrievalCapability, AgentCapability)
