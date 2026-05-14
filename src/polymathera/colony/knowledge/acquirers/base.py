"""Acquirer ABC + result dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AcquiredSource(BaseModel):
    """Result of a single acquirer invocation.

    ``local_path`` is the absolute path on disk where the acquired
    content lives after acquisition. ``cached`` is ``True`` when the
    acquirer hit a local cache (no network round-trip) — used by the
    materialiser to decide whether to skip re-extraction even when the
    pdf_sha256 matches.
    """

    model_config = ConfigDict(frozen=True)

    local_path: Path
    """Absolute path of the acquired file on disk."""

    cached: bool = False
    """True when the acquirer served the file from cache without
    re-fetching from the remote source."""

    fetched_bytes: int = 0
    """Byte count of the acquired content. ``0`` for cache hits when
    the acquirer doesn't restat the file."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Per-method metadata the acquirer wants to propagate (e.g.,
    arXiv version id, DOI title, retrieval timestamp). Lands in the
    ingestion record's ``metadata`` for downstream consumers."""


class AcquirerStrategy(ABC):
    """One remote-source acquisition strategy.

    Subclasses implement a single ``method`` (e.g., ``"arxiv_id"``,
    ``"doi"``, ``"http_url"``) and the ``acquire(...)`` coroutine that
    fetches the source into ``destination_dir`` (typically a subtree
    of the design monorepo's working tree).
    """

    @property
    @abstractmethod
    def method(self) -> str:
        """Stable identifier used to look this strategy up by
        ``acquirer.method`` in ``repo_map.yaml``."""

    @abstractmethod
    async def acquire(
        self,
        *,
        args: dict[str, Any],
        destination_dir: Path,
    ) -> AcquiredSource:
        """Fetch the source described by ``args`` into ``destination_dir``.

        Caller guarantees ``destination_dir`` exists. The acquirer
        picks the file basename (typically from the remote source's
        canonical name — e.g., ``2407.12345v1.pdf`` for arXiv) and
        returns the absolute path it wrote to.

        Failures raise. The materialiser catches at the row level so a
        single bad acquirer doesn't poison the rest of the batch.
        """


__all__ = ("AcquiredSource", "AcquirerStrategy")
