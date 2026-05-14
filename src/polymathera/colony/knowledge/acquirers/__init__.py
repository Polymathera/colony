"""Acquirer strategies — fetch a remote literature source to a local path.

A :class:`KnowledgeSource` row in ``.colony/repo_map.yaml`` can declare
either local globs (``paths``) or a remote acquirer (``acquirer:
{method, args}``). When the row is acquirer-shaped, the materialiser
looks up the method in :data:`AcquirerRegistry`, calls the strategy
with the source's ``destination`` directory, and feeds the resulting
path into the ingestor like any other local file.

Acquirers don't drive ingestion themselves — they're a thin layer
that resolves a logical reference (an arXiv id, a DOI, an HTTP URL,
…) to a file on disk inside the design monorepo's working tree. The
unified pipeline in :mod:`polymathera.colony.design_monorepo.materialize`
chains acquire → ingest → commit.

Today the registry ships only placeholder TODO stubs. Each real
acquirer (arXiv API + httpx + content-hash verify; Crossref DOI
resolution + Unpaywall fallback; etc.) ships as its own focused
follow-up PR. The unified schema accepts ``acquirer:`` rows for these
methods today; rows whose method has no real implementation will
surface a ``NotImplementedError`` at materialise time.
"""

from __future__ import annotations

from .base import AcquiredSource, AcquirerStrategy
from .registry import AcquirerRegistry, default_registry
from .todo_stubs import (
    _TODO_ArxivAcquirer,
    _TODO_DoiAcquirer,
    _TODO_HttpAcquirer,
    _TODO_IeeeXploreAcquirer,
    _TODO_SaeMobilusAcquirer,
    _TODO_SemanticScholarAcquirer,
)


__all__ = (
    "AcquiredSource",
    "AcquirerRegistry",
    "AcquirerStrategy",
    "default_registry",
    "_TODO_ArxivAcquirer",
    "_TODO_DoiAcquirer",
    "_TODO_HttpAcquirer",
    "_TODO_IeeeXploreAcquirer",
    "_TODO_SaeMobilusAcquirer",
    "_TODO_SemanticScholarAcquirer",
)
