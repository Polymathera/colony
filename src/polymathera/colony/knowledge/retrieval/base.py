"""Common base for the five retrieval-mode adapters.

Every retrieval adapter is a ``ToolAdapter`` (so the C2 registry +
``ToolCapability`` surface work uniformly) that takes a
``RetrievalDeps`` bundle: an embedder, a vector store, and an
optional graph store. The ``invoke`` method translates the typed
``ToolCall`` into a ``RetrievalQuery``, dispatches to the mode's
``run`` method, and packages the result as a ``ToolResult`` whose
``value`` is the typed ``RetrievalResult``.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Mapping
from typing import Any

from ...agents.blueprint import Blueprint, blueprint
from ...tools import ToolAdapter, ToolCall, ToolResult
from ..embedder import Embedder
from ..models import RetrievalQuery, RetrievalResult
from ..stores import GraphStore, ImageStore, VectorStore


logger = logging.getLogger(__name__)


@blueprint
class RetrievalDeps:
    """Dependencies shared across retrieval-mode adapters.

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


class RetrievalAdapter(ToolAdapter):
    """Base class for the five retrieval-mode adapters.

    Subclasses set the class-level ``ToolSpec`` (per ``ToolAdapter``)
    and override ``run`` to produce a ``RetrievalResult``. The
    ``invoke`` method handles parameter parsing + result wrapping
    once.
    """

    mode: str = ""

    def __init__(self, *, deps: RetrievalDeps) -> None:
        self._deps = deps

    @abstractmethod
    async def run(self, query: RetrievalQuery) -> RetrievalResult:
        ...

    async def invoke(self, call: ToolCall) -> ToolResult:
        try:
            query = _build_query(call.parameters)
        except Exception as exc:  # noqa: BLE001 - typed input errors
            return ToolResult(
                call_id=call.call_id,
                adapter_name=type(self).spec.name,
                success=False,
                error=f"invalid retrieval parameters: {exc}",
            )
        try:
            result = await self.run(query)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "RetrievalAdapter(%s): retrieval failed", type(self).__name__,
            )
            return ToolResult(
                call_id=call.call_id,
                adapter_name=type(self).spec.name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        return ToolResult(
            call_id=call.call_id,
            adapter_name=type(self).spec.name,
            success=True,
            value=result.model_dump(mode="json"),
        )


def _build_query(parameters: Mapping[str, Any]) -> RetrievalQuery:
    """Translate ``ToolCall.parameters`` to a typed ``RetrievalQuery``.

    Accepts either a pre-built ``RetrievalQuery`` (under the ``"query"``
    key, useful when the caller is colony-internal) or the flat field
    set as kwargs (useful when the caller is an LLM-planned action
    that just passes JSON-shaped fields)."""

    if "query" in parameters and isinstance(parameters["query"], RetrievalQuery):
        return parameters["query"]
    if "query" in parameters and isinstance(parameters["query"], dict):
        return RetrievalQuery.model_validate(parameters["query"])
    return RetrievalQuery.model_validate(parameters)


__all__ = ("RetrievalAdapter", "RetrievalDeps")
