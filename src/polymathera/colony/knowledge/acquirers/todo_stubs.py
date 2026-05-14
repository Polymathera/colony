"""Placeholder acquirer strategies — each raises ``NotImplementedError``
with a build-effort estimate and points at the dossier reference.

The unified ``KnowledgeSource.acquirer`` schema accepts these methods
today so a ``repo_map.yaml`` can declare them in advance of their
implementation. Until a stub is replaced with a real implementation,
materialising a row whose acquirer is one of these surfaces the
NotImplementedError to the operator (caught + logged at the row level,
the rest of the batch survives).

Each real acquirer ships as its own focused follow-up PR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import AcquiredSource, AcquirerStrategy


class _TODOAcquirer(AcquirerStrategy):
    """Base for unimplemented acquirers."""

    _METHOD: str = ""
    _DOSSIER_REF: str = ""
    _BUILD_EFFORT: str = ""

    @property
    def method(self) -> str:
        return self._METHOD

    async def acquire(
        self, *, args: dict[str, Any], destination_dir: Path,
    ) -> AcquiredSource:
        raise NotImplementedError(
            f"{type(self).__name__} is a placeholder. "
            f"{self._DOSSIER_REF}. Build effort: {self._BUILD_EFFORT}.",
        )


class _TODO_HttpAcquirer(_TODOAcquirer):
    """Method ``"http_url"`` — fetch bytes from ``args['url']`` via
    ``httpx`` with content-hash verification + on-disk cache. Master
    §6.6.1."""

    _METHOD = "http_url"
    _DOSSIER_REF = "master §6.6.1"
    _BUILD_EFFORT = "1-2 days"


class _TODO_ArxivAcquirer(_TODOAcquirer):
    """Method ``"arxiv_id"`` — resolve ``args['arxiv_id']`` to the
    arXiv canonical PDF via the export API + httpx. Master §6.6.1."""

    _METHOD = "arxiv_id"
    _DOSSIER_REF = "master §6.6.1"
    _BUILD_EFFORT = "1-2 days"


class _TODO_DoiAcquirer(_TODOAcquirer):
    """Method ``"doi"`` — resolve ``args['doi']`` via Crossref +
    Unpaywall fallback. Master §6.6.1."""

    _METHOD = "doi"
    _DOSSIER_REF = "master §6.6.1"
    _BUILD_EFFORT = "2-3 days"


class _TODO_IeeeXploreAcquirer(_TODOAcquirer):
    """Method ``"ieee_xplore"`` — IEEE Xplore API (license-gated).
    Master §6.6.1."""

    _METHOD = "ieee_xplore"
    _DOSSIER_REF = "master §6.6.1"
    _BUILD_EFFORT = "2-3 days"


class _TODO_SaeMobilusAcquirer(_TODOAcquirer):
    """Method ``"sae_mobilus"`` — SAE Mobilus search API
    (license-gated). Master §6.6.1."""

    _METHOD = "sae_mobilus"
    _DOSSIER_REF = "master §6.6.1"
    _BUILD_EFFORT = "2-3 days"


class _TODO_SemanticScholarAcquirer(_TODOAcquirer):
    """Method ``"semantic_scholar"`` — Semantic Scholar API. Master
    §6.6.1."""

    _METHOD = "semantic_scholar"
    _DOSSIER_REF = "master §6.6.1"
    _BUILD_EFFORT = "1-2 days"


__all__ = (
    "_TODO_ArxivAcquirer",
    "_TODO_DoiAcquirer",
    "_TODO_HttpAcquirer",
    "_TODO_IeeeXploreAcquirer",
    "_TODO_SaeMobilusAcquirer",
    "_TODO_SemanticScholarAcquirer",
)
