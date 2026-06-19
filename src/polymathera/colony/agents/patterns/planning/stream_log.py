"""Durable, ordered, append-only log backing a consciousness stream.

A :class:`~polymathera.colony.agents.patterns.planning.streams.ConsciousnessStream`
is conceptually the complete, ordered history of its entries (the
"infinite linear stream"); what renders into the planning prompt is a
bounded *view* over that history. This module provides the durable
source of truth that makes the view's spillover lossless and its
compaction reversible.

Two layers:

- **Raw log** — every recorded entry, in append order, addressed by a
  monotonic per-stream ``seq``. Range-addressable
  (:meth:`StreamLogStore.read_span`) so any historical span can be
  pulled back verbatim. This is the lossless "infinite" history.
- **Index** (:class:`StreamLogIndex`) — small, bounded metadata: the
  next sequence number, the working-window floor (lowest seq still
  rendered raw), and the list of :class:`CompactionDescriptor`s
  (each: the ``[start_seq, end_seq]`` span it stands in for, the
  condensed summary, an optional spill-archive reference). Compaction
  descriptors are NOT given a position in the raw seq space — they are
  synthesized into ``compaction_summary`` view-entries at render time,
  sorted by their span's ``start_seq`` so interleaving stays correct
  even after an arbitrary :meth:`expand`.

:class:`StreamLogStore` is an ABC so the backing store is swappable
(default: a non-evicting blackboard scope). A future Redis-Streams /
SQLite / WAL backing implements the same five methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from ...blackboard.protocol import BlackboardProtocol
from ...scopes import BlackboardScope

if TYPE_CHECKING:  # pragma: no cover — type-only
    from ...blackboard import EnhancedBlackboard


# ---------------------------------------------------------------------------
# Blackboard key protocol for the per-stream log scope
# ---------------------------------------------------------------------------


class ConsciousnessLogProtocol(BlackboardProtocol):
    """Keys for a single stream's durable log.

    The log lives in its own agent-level blackboard scope (one scope
    per ``(agent, stream)``), so keys are simple and scope-relative:

    - ``entry:{seq:020d}`` — one raw recorded entry.
    - ``index`` — the singleton :class:`StreamLogIndex` record.
    """

    scope: ClassVar[BlackboardScope] = BlackboardScope.AGENT

    _ENTRY_PREFIX: ClassVar[str] = "entry:"
    _INDEX_KEY: ClassVar[str] = "index"

    @staticmethod
    def entry_key(seq: int) -> str:
        # Zero-padded so lexical order matches numeric order (cheap range scans).
        return f"{ConsciousnessLogProtocol._ENTRY_PREFIX}{seq:020d}"

    @staticmethod
    def index_key() -> str:
        return ConsciousnessLogProtocol._INDEX_KEY

    @staticmethod
    def parse_entry_key(key: str) -> int:
        if not key.startswith(ConsciousnessLogProtocol._ENTRY_PREFIX):
            raise ValueError(f"Not a consciousness-log entry key: {key!r}")
        return int(key[len(ConsciousnessLogProtocol._ENTRY_PREFIX):])


# ---------------------------------------------------------------------------
# Index types
# ---------------------------------------------------------------------------


@dataclass
class CompactionDescriptor:
    """A compacted span: the originals stay in the raw log (seq
    ``start_seq..end_seq`` inclusive); this descriptor is the durable
    stand-in rendered in the view until the span is expanded."""

    start_seq: int
    end_seq: int
    summary: str
    kinds: dict[str, int] = field(default_factory=dict)   # histogram of summarized entry kinds
    entry_count: int = 0
    archive_ref: str | None = None                        # spill-archive handle (e.g. VCM scope), if any
    produced_by: str = "auto"                             # "auto" | "agent"
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CompactionDescriptor":
        return cls(**d)


@dataclass
class StreamLogIndex:
    """Bounded, durable metadata describing the view over the raw log.

    The rendered view is reconstructed as: the synthesized summary for
    every descriptor in :attr:`compactions` (each sorted by its span's
    ``start_seq``) merged with the raw entries in ``[0, next_seq)`` that
    are NOT covered by any descriptor. Covered spans need not be
    contiguous (the agent may expand a middle span), so the view is
    derived from the descriptors directly rather than a single floor.
    """

    next_seq: int = 0
    compactions: list[CompactionDescriptor] = field(default_factory=list)
    reflector_state: dict[str, Any] = field(default_factory=dict)
    """Opaque per-stream reflector state — whatever the bound
    :class:`~polymathera.colony.agents.patterns.planning.reflection.StreamReflector`
    returned from ``serialize_state()`` at the last :meth:`flush`. Restored
    on rehydrate via the reflector's ``deserialize_state``. Default empty
    for stateless reflectors. Schema is the reflector's responsibility —
    the substrate just round-trips the JSON shape."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_seq": self.next_seq,
            "compactions": [c.to_dict() for c in self.compactions],
            "reflector_state": dict(self.reflector_state),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StreamLogIndex":
        return cls(
            next_seq=int(d.get("next_seq", 0)),
            compactions=[
                CompactionDescriptor.from_dict(c)
                for c in (d.get("compactions") or [])
            ],
            reflector_state=dict(d.get("reflector_state") or {}),
        )

    def covered_ranges(self) -> list[tuple[int, int]]:
        """Sorted ``(start_seq, end_seq)`` spans currently compacted."""
        return sorted((c.start_seq, c.end_seq) for c in self.compactions)


# ---------------------------------------------------------------------------
# Store ABC
# ---------------------------------------------------------------------------


class StreamLogStore(ABC):
    """Durable, ordered, range-addressable backing for one stream's log.

    Swap point for the storage substrate. The default
    :class:`BlackboardStreamLogStore` uses a non-evicting blackboard
    scope; alternative backings (Redis Streams, SQLite, a write-ahead
    log) implement the same contract.

    Single-writer by construction: one live agent actor owns a stream
    (``AgentManagerBase`` keeps one ``Agent`` per ``agent_id``), so
    appends + index updates need no compare-and-swap. A multi-writer
    backing would layer CAS in its own implementation.
    """

    @abstractmethod
    async def append(self, seq: int, entry: dict[str, Any]) -> None:
        """Persist a raw entry at ``seq`` (idempotent on the same seq)."""

    @abstractmethod
    async def read_span(self, start_seq: int, end_seq: int) -> list[dict[str, Any]]:
        """Return raw entries with ``start_seq <= seq <= end_seq``, in
        seq order. Missing seqs are skipped (never raises for gaps)."""

    @abstractmethod
    async def read_index(self) -> StreamLogIndex:
        """Return the stored index, or a fresh empty one if absent."""

    @abstractmethod
    async def write_index(self, index: StreamLogIndex) -> None:
        """Persist the index."""


class BlackboardStreamLogStore(StreamLogStore):
    """Default store: a non-evicting, events-off blackboard scope.

    The blackboard must be constructed with ``max_entries=None`` (no
    eviction — the log is lossless) and ``enable_events=False`` (the
    log has no subscribers; pub/sub would be pure overhead). The
    backing persistence (Redis / distributed StateManager) is whatever
    the deployment configures, so the log survives restart.
    """

    def __init__(self, blackboard: "EnhancedBlackboard", created_by: str = "consciousness_stream"):
        self._bb = blackboard
        self._created_by = created_by

    async def append(self, seq: int, entry: dict[str, Any]) -> None:
        await self._bb.write(
            key=ConsciousnessLogProtocol.entry_key(seq),
            value=entry,
            created_by=self._created_by,
        )

    async def read_span(self, start_seq: int, end_seq: int) -> list[dict[str, Any]]:
        if end_seq < start_seq:
            return []
        out: list[dict[str, Any]] = []
        for seq in range(start_seq, end_seq + 1):
            value = await self._bb.read(ConsciousnessLogProtocol.entry_key(seq))
            if isinstance(value, dict):
                out.append(value)
        return out

    async def read_index(self) -> StreamLogIndex:
        value = await self._bb.read(ConsciousnessLogProtocol.index_key())
        if isinstance(value, dict):
            return StreamLogIndex.from_dict(value)
        return StreamLogIndex()

    async def write_index(self, index: StreamLogIndex) -> None:
        await self._bb.write(
            key=ConsciousnessLogProtocol.index_key(),
            value=index.to_dict(),
            created_by=self._created_by,
        )


__all__ = (
    "ConsciousnessLogProtocol",
    "CompactionDescriptor",
    "StreamLogIndex",
    "StreamLogStore",
    "BlackboardStreamLogStore",
)
