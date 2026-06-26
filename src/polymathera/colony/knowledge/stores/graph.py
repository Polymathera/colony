"""``GraphStore`` ABC + two implementations.

The knowledge graph layer is *RDF-shaped* (subject / predicate /
object). Nodes have a typed ``label`` and an open-set ``properties``
map; edges connect two nodes with a typed predicate and their own
properties. This matches ``Claim`` from the model layer one-to-one.

Two implementations:

- ``InMemoryGraphStore`` — full implementation; supports a small
  Cypher-like query DSL (``MATCH`` ``WHERE`` ``RETURN``) sufficient
  for the §6.4 graph-retrieval mode's tests + small deployments.
- ``KuzuGraphStore`` — stub. Real Kùzu wiring lands in C1b.

The query language is intentionally *not* a full Cypher implementation
— that's a maintenance trap. We support exactly what the graph
retrieval mode uses: pattern match on ``(subject)-[predicate]->(object)``,
optional ``WHERE`` filters on node labels / properties, ``LIMIT`` /
``DEPTH``. For richer queries, the user runs Kùzu directly.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import logging
import re
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..models import Claim, CitationSpan


logger = logging.getLogger(__name__)


#: Branch context for writes. Set by capability action wrappers
#: (typically the design-monorepo capability that resolved the
#: agent's current clone branch) before any code path that may
#: call ``GraphStore.add_*`` runs. ``add_*`` mutators read this
#: when no explicit ``branch`` kwarg is passed; both absent is a
#: hard error (silently dropping claims into an untagged set would
#: hide them from every branch-filtered query).
CURRENT_BRANCH_CONTEXT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "colony_current_branch", default=None,
)


@contextlib.contextmanager
def set_current_branch(branch: str | None) -> Iterator[None]:
    """Bind :data:`CURRENT_BRANCH_CONTEXT` to ``branch`` for the
    duration of the ``with`` block. ``None`` clears it for nested
    contexts."""

    token = CURRENT_BRANCH_CONTEXT.set(branch)
    try:
        yield
    finally:
        CURRENT_BRANCH_CONTEXT.reset(token)


def _resolve_write_branch(explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    ctx = CURRENT_BRANCH_CONTEXT.get()
    if ctx is not None:
        return ctx
    raise GraphStoreError(
        "GraphStore write requires a branch: pass branch=... explicitly "
        "or bind one via set_current_branch(). Empty branch sets are "
        "rejected to avoid silently dropping claims into the union view.",
    )


class ImportResult(BaseModel):
    """Outcome of :meth:`GraphStore.import_claims`."""

    model_config = ConfigDict(frozen=True)

    added: int = 0
    """Triples whose ``(subject, predicate, object)`` was not yet in
    the store. The branch tag is attached on insert."""

    tagged: int = 0
    """Triples that already existed but did not yet carry the import
    branch in their ``branches`` set."""

    skipped: int = 0
    """Triples that already carried the import branch — nothing to do."""


#: Reserved key inside the Kùzu ``properties`` JSON blob that
#: round-trips a record's :attr:`GraphNode.branches` /
#: :attr:`GraphEdge.branches` set. Stored as a sorted list so the
#: serialised JSON is deterministic.
_BRANCH_PROP_KEY = "__branches"


def _load_props(raw: Any) -> dict[str, Any]:
    """Tolerantly load a JSON-encoded properties payload."""

    if not raw:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    try:
        loaded = json.loads(str(raw))
    except (json.JSONDecodeError, TypeError):
        return {}
    if isinstance(loaded, dict):
        return loaded
    return {}


def _split_branches(props: dict[str, Any]) -> tuple[dict[str, Any], frozenset[str]]:
    """Pop ``_BRANCH_PROP_KEY`` out of a loaded properties dict and
    return ``(remaining_props, branches)``."""

    out = dict(props)
    raw = out.pop(_BRANCH_PROP_KEY, None)
    if isinstance(raw, (list, tuple, set, frozenset)):
        return out, frozenset(str(b) for b in raw)
    return out, frozenset()


def _encode_props(props: Mapping[str, Any], branches: frozenset[str]) -> str:
    """Encode a properties dict + ``branches`` set into the JSON blob
    stored in Kùzu's ``properties`` column."""

    payload = dict(props)
    if branches:
        payload[_BRANCH_PROP_KEY] = sorted(branches)
    return json.dumps(payload, sort_keys=True)


def _iter_rows(cursor: Any) -> Iterable[tuple[Any, ...]]:
    """Iterate Kùzu's ``QueryResult`` cursor row-by-row.

    Kùzu's Python API exposes ``has_next() / get_next()``; this helper
    abstracts the cursor protocol so callers can ``for row in
    _iter_rows(cursor)``."""

    if cursor is None:
        return
    if hasattr(cursor, "has_next") and hasattr(cursor, "get_next"):
        while cursor.has_next():
            yield tuple(cursor.get_next())
        return
    # Fall back: treat the cursor as iterable.
    for row in cursor:  # pragma: no cover - kuzu always exposes has_next
        yield tuple(row)


class GraphStoreError(RuntimeError):
    """Base error for the graph store."""


class GraphNode(BaseModel):
    """One node in the knowledge graph."""

    model_config = ConfigDict(frozen=True)

    node_id: str
    label: str = "Entity"
    properties: dict[str, Any] = Field(default_factory=dict)
    branches: frozenset[str] = Field(default_factory=frozenset)
    """Branches this node is visible on. Empty = visible only to
    unfiltered (union) queries; reads with ``branch_filter`` set
    require an explicit branch tag."""


class GraphEdge(BaseModel):
    """One edge in the knowledge graph."""

    model_config = ConfigDict(frozen=True)

    edge_id: str
    source_id: str
    target_id: str
    predicate: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    citation_uri: str | None = None
    """Source URI of the chunk that produced this edge (when extracted)."""

    properties: dict[str, Any] = Field(default_factory=dict)
    branches: frozenset[str] = Field(default_factory=frozenset)
    """Branches this edge is visible on. See :attr:`GraphNode.branches`."""


class GraphQueryResult(BaseModel):
    """Result of ``GraphStore.query``."""

    model_config = ConfigDict(frozen=True)

    nodes: tuple[GraphNode, ...] = Field(default_factory=tuple)
    edges: tuple[GraphEdge, ...] = Field(default_factory=tuple)
    paths: tuple[tuple[str, ...], ...] = Field(
        default_factory=tuple,
        description=(
            "Each path is a tuple of node ids visited in order. Used "
            "by graph-retrieval mode to surface traversal explanations."
        ),
    )


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class GraphStore(ABC):
    @abstractmethod
    async def add_node(self, node: GraphNode, *, branch: str | None = None) -> None:
        """Insert or merge ``node``, tagging it with ``branch``
        (resolved from :data:`CURRENT_BRANCH_CONTEXT` if not passed).
        Existing nodes keep their first-seen label; properties merge;
        ``branches`` is the union of the existing tag set and the
        resolved branch."""

    @abstractmethod
    async def add_edge(self, edge: GraphEdge, *, branch: str | None = None) -> None:
        """Insert ``edge`` (no-op if the ``edge_id`` already exists)
        tagged with ``branch``. If the edge exists, ``branch`` is
        added to its ``branches`` set."""

    @abstractmethod
    async def add_claim(
        self, claim: Claim, *, branch: str | None = None,
    ) -> tuple[GraphNode, GraphNode, GraphEdge]:
        """Convenience: turn a ``Claim`` into a ``(subject, object, edge)``
        triple, idempotently inserting any nodes / edges that are new.
        Every record touched is tagged with ``branch``.

        Returns the (possibly newly created) records."""

    async def add_claims(
        self, claims: Sequence[Claim], *, branch: str | None = None,
    ) -> tuple[tuple[GraphNode, GraphNode, GraphEdge] | None, ...]:
        """Batch counterpart of :meth:`add_claim`. Returns one entry
        per input claim, in order; failed inserts surface as ``None``
        at the failing index so the batch does not abort on the first
        bad claim (matches the ingestion loop's prior per-claim
        try/except recovery).

        The default implementation simply loops :meth:`add_claim`,
        preserving the ABC's single-claim contract. Subclasses with a
        transactional / locked write layer (e.g.
        :class:`KuzuGraphStore`) should override to amortise the
        per-claim await + lock-acquire churn across the batch.
        """

        out: list[tuple[GraphNode, GraphNode, GraphEdge] | None] = []
        for claim in claims:
            try:
                out.append(await self.add_claim(claim, branch=branch))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "GraphStore.add_claim failed for claim %r: %s",
                    claim, exc,
                )
                out.append(None)
        return tuple(out)

    @abstractmethod
    async def get_node(
        self, node_id: str, *, branch_filter: str | None = None,
    ) -> GraphNode | None:
        """Return the node, or ``None`` if missing OR if
        ``branch_filter`` is set and the node's ``branches`` does
        not contain it. The two cases are indistinguishable from the
        consumer's perspective."""

    @abstractmethod
    async def neighbours(
        self,
        node_id: str,
        *,
        predicate: str | None = None,
        depth: int = 1,
        branch_filter: str | None = None,
    ) -> GraphQueryResult:
        ...

    @abstractmethod
    async def query(
        self, query: str, *, branch_filter: str | None = None,
    ) -> GraphQueryResult:
        """Run a small-DSL query (see module docstring). Implementations
        that don't support the DSL raise ``GraphStoreError``. When
        ``branch_filter`` is set, only nodes/edges tagged with that
        branch are returned."""

    @abstractmethod
    async def count(
        self, *, branch_filter: str | None = None,
    ) -> tuple[int, int]:
        """Return ``(node_count, edge_count)``. When ``branch_filter``
        is set, count only records tagged with that branch."""

    @abstractmethod
    def export_claims(
        self, *, branch: str | None = None,
    ) -> AsyncIterator[Claim]:
        """Yield every claim with full provenance. When ``branch`` is
        set, restrict to edges whose ``branches`` contains it. The
        emitted :class:`Claim` carries the edge's ``confidence``,
        ``citation_uri`` + properties, the subject/object surface
        strings from each node's ``properties['surface']`` (or
        ``node_id`` when absent), and any ``provenance`` stored on
        the edge's properties under the ``__provenance`` key."""

    @abstractmethod
    async def import_claims(
        self, claims: Sequence[Claim], *, branch: str,
    ) -> ImportResult:
        """Bulk-insert ``claims``, tagging every touched node/edge
        with ``branch`` (REQUIRED — rehydrate must know the source
        branch; no contextvar fallback). Returns counts of
        newly-added vs newly-tagged vs no-op (already-tagged) triples."""


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------


# Tiny query DSL: MATCH (s)-[p]->(o) [WHERE clause]* [RETURN ... ] [LIMIT n]
_MATCH_RE = re.compile(
    r"^\s*MATCH\s+\((?P<s>[a-z]\w*)(?::(?P<sl>\w+))?\)\s*"
    r"-\[(?P<r>[a-z]\w*)(?::(?P<rl>\w+))?\]->\s*"
    r"\((?P<o>[a-z]\w*)(?::(?P<ol>\w+))?\)\s*"
    r"(?:WHERE\s+(?P<where>.+?))?\s*"
    r"(?:RETURN\s+(?P<ret>.+?))?\s*"
    r"(?:LIMIT\s+(?P<limit>\d+))?\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _node_id_for(subject: str) -> str:
    """Stable lowercase-snake-case id for a free-form subject string."""

    return re.sub(r"[^a-z0-9]+", "_", subject.strip().lower()).strip("_") or "node"


def _claim_from_edge(
    edge: GraphEdge, nodes: Mapping[str, GraphNode],
) -> Claim:
    """Reconstruct a :class:`Claim` from a stored ``edge`` + its
    surrounding nodes. Pulls subject/object surface strings from
    ``node.properties['surface']`` (falls back to ``node_id``) and
    lifts the edge's ``__provenance`` private property back into
    :attr:`Claim.provenance`."""

    s_node = nodes.get(edge.source_id)
    o_node = nodes.get(edge.target_id)
    edge_props = dict(edge.properties)
    provenance = edge_props.pop("__provenance", {}) or {}
    if not isinstance(provenance, dict):
        provenance = {}
    citation = CitationSpan(
        source_uri=edge.citation_uri or "",
        section_path=str(edge_props.get("section_path") or ""),
        char_start=int(edge_props.get("char_start") or 0),
        char_end=int(edge_props.get("char_end") or 0),
    )
    subject = (
        (s_node.properties.get("surface") if s_node else None)
        or edge.source_id
    )
    obj = (
        (o_node.properties.get("surface") if o_node else None)
        or edge.target_id
    )
    return Claim(
        subject=str(subject),
        predicate=edge.predicate,
        object=str(obj),
        confidence=edge.confidence,
        citation=citation,
        provenance=provenance,
    )


class InMemoryGraphStore(GraphStore):
    """Pure-Python graph store with a tiny Cypher-like query DSL."""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges_by_id: dict[str, GraphEdge] = {}
        self._out: dict[str, list[str]] = defaultdict(list)
        """``source_id`` -> list of edge_ids."""

        self._in: dict[str, list[str]] = defaultdict(list)
        """``target_id`` -> list of edge_ids."""

    # ---- Mutation -----------------------------------------------------

    async def add_node(
        self, node: GraphNode, *, branch: str | None = None,
    ) -> None:
        tag = _resolve_write_branch(branch)
        existing = self._nodes.get(node.node_id)
        if existing is None:
            self._nodes[node.node_id] = node.model_copy(
                update={"branches": node.branches | {tag}},
            )
            return
        # Merge properties; keep first-seen label.
        merged = {**existing.properties, **node.properties}
        self._nodes[node.node_id] = existing.model_copy(
            update={
                "properties": merged,
                "branches": existing.branches | node.branches | {tag},
            },
        )

    async def add_edge(
        self, edge: GraphEdge, *, branch: str | None = None,
    ) -> None:
        tag = _resolve_write_branch(branch)
        if edge.source_id not in self._nodes or edge.target_id not in self._nodes:
            raise GraphStoreError(
                f"Edge {edge.edge_id}: missing source or target node "
                f"({edge.source_id}, {edge.target_id}).",
            )
        existing = self._edges_by_id.get(edge.edge_id)
        if existing is None:
            self._edges_by_id[edge.edge_id] = edge.model_copy(
                update={"branches": edge.branches | {tag}},
            )
            self._out[edge.source_id].append(edge.edge_id)
            self._in[edge.target_id].append(edge.edge_id)
            return
        if tag in existing.branches:
            return
        self._edges_by_id[edge.edge_id] = existing.model_copy(
            update={"branches": existing.branches | {tag}},
        )

    async def add_claim(
        self, claim: Claim, *, branch: str | None = None,
    ) -> tuple[GraphNode, GraphNode, GraphEdge]:
        tag = _resolve_write_branch(branch)
        s_id = _node_id_for(claim.subject)
        o_id = _node_id_for(claim.object_)
        s_node = self._nodes.get(s_id) or GraphNode(
            node_id=s_id, label="Entity",
            properties={"surface": claim.subject},
        )
        o_node = self._nodes.get(o_id) or GraphNode(
            node_id=o_id, label="Entity",
            properties={"surface": claim.object_},
        )
        await self.add_node(s_node, branch=tag)
        await self.add_node(o_node, branch=tag)
        edge_id = f"{s_id}--{claim.predicate}-->{o_id}"
        edge_props = {
            "section_path": claim.citation.section_path,
            "char_start": claim.citation.char_start,
            "char_end": claim.citation.char_end,
        }
        if claim.provenance:
            edge_props["__provenance"] = dict(claim.provenance)
        edge = GraphEdge(
            edge_id=edge_id,
            source_id=s_id,
            target_id=o_id,
            predicate=claim.predicate,
            confidence=claim.confidence,
            citation_uri=claim.citation.source_uri,
            properties=edge_props,
        )
        await self.add_edge(edge, branch=tag)
        return self._nodes[s_id], self._nodes[o_id], self._edges_by_id[edge_id]

    async def import_claims(
        self, claims: Sequence[Claim], *, branch: str,
    ) -> ImportResult:
        if not branch:
            raise GraphStoreError(
                "import_claims requires a non-empty branch.",
            )
        added = tagged = skipped = 0
        for claim in claims:
            s_id = _node_id_for(claim.subject)
            o_id = _node_id_for(claim.object_)
            edge_id = f"{s_id}--{claim.predicate}-->{o_id}"
            existing = self._edges_by_id.get(edge_id)
            if existing is None:
                await self.add_claim(claim, branch=branch)
                added += 1
                continue
            if branch in existing.branches:
                skipped += 1
                continue
            await self.add_node(
                self._nodes[existing.source_id], branch=branch,
            )
            await self.add_node(
                self._nodes[existing.target_id], branch=branch,
            )
            await self.add_edge(existing, branch=branch)
            tagged += 1
        return ImportResult(added=added, tagged=tagged, skipped=skipped)

    # ---- Queries ------------------------------------------------------

    async def get_node(
        self, node_id: str, *, branch_filter: str | None = None,
    ) -> GraphNode | None:
        node = self._nodes.get(node_id)
        if node is None:
            return None
        if branch_filter is not None and branch_filter not in node.branches:
            return None
        return node

    async def neighbours(
        self,
        node_id: str,
        *,
        predicate: str | None = None,
        depth: int = 1,
        branch_filter: str | None = None,
    ) -> GraphQueryResult:
        root = self._nodes.get(node_id)
        if root is None:
            return GraphQueryResult()
        if branch_filter is not None and branch_filter not in root.branches:
            return GraphQueryResult()
        seen: dict[str, GraphNode] = {node_id: root}
        edges_set: dict[str, GraphEdge] = {}
        paths: list[tuple[str, ...]] = []
        frontier = [(node_id, (node_id,))]
        for _ in range(max(0, depth)):
            next_frontier: list[tuple[str, tuple[str, ...]]] = []
            for current, path in frontier:
                for eid in self._out.get(current, ()):
                    edge = self._edges_by_id[eid]
                    if predicate is not None and edge.predicate != predicate:
                        continue
                    if branch_filter is not None and branch_filter not in edge.branches:
                        continue
                    target = self._nodes.get(edge.target_id)
                    if target is None:
                        continue
                    if branch_filter is not None and branch_filter not in target.branches:
                        continue
                    edges_set[eid] = edge
                    seen[target.node_id] = target
                    new_path = path + (target.node_id,)
                    paths.append(new_path)
                    next_frontier.append((target.node_id, new_path))
            frontier = next_frontier
        return GraphQueryResult(
            nodes=tuple(seen.values()),
            edges=tuple(edges_set.values()),
            paths=tuple(paths),
        )

    async def query(
        self, query: str, *, branch_filter: str | None = None,
    ) -> GraphQueryResult:
        m = _MATCH_RE.match(query.strip())
        if not m:
            raise GraphStoreError(
                "InMemoryGraphStore supports a narrow MATCH (s)-[p]->(o) "
                "[WHERE …] [RETURN …] [LIMIT n] DSL. Got: "
                + repr(query.strip()[:200]),
            )
        s_label = m.group("sl")
        r_label = m.group("rl")
        o_label = m.group("ol")
        where = m.group("where") or ""
        limit_str = m.group("limit")
        limit = int(limit_str) if limit_str else None
        where_filters = _parse_where(where)
        nodes: dict[str, GraphNode] = {}
        edges: dict[str, GraphEdge] = {}
        paths: list[tuple[str, ...]] = []
        for edge in self._edges_by_id.values():
            if r_label and edge.predicate != r_label:
                continue
            if branch_filter is not None and branch_filter not in edge.branches:
                continue
            s_node = self._nodes.get(edge.source_id)
            o_node = self._nodes.get(edge.target_id)
            if s_node is None or o_node is None:
                continue
            if branch_filter is not None and (
                branch_filter not in s_node.branches
                or branch_filter not in o_node.branches
            ):
                continue
            if s_label and s_node.label != s_label:
                continue
            if o_label and o_node.label != o_label:
                continue
            if not _evaluate_where(
                where_filters,
                {"s": s_node, "o": o_node, "r": edge},
            ):
                continue
            nodes[s_node.node_id] = s_node
            nodes[o_node.node_id] = o_node
            edges[edge.edge_id] = edge
            paths.append((s_node.node_id, o_node.node_id))
            if limit is not None and len(paths) >= limit:
                break
        return GraphQueryResult(
            nodes=tuple(nodes.values()),
            edges=tuple(edges.values()),
            paths=tuple(paths),
        )

    async def count(
        self, *, branch_filter: str | None = None,
    ) -> tuple[int, int]:
        if branch_filter is None:
            return len(self._nodes), len(self._edges_by_id)
        n_count = sum(1 for n in self._nodes.values() if branch_filter in n.branches)
        e_count = sum(1 for e in self._edges_by_id.values() if branch_filter in e.branches)
        return n_count, e_count

    async def export_claims(
        self, *, branch: str | None = None,
    ) -> AsyncIterator[Claim]:
        for edge in sorted(
            self._edges_by_id.values(),
            key=lambda e: (e.source_id, e.predicate, e.target_id),
        ):
            if branch is not None and branch not in edge.branches:
                continue
            yield _claim_from_edge(edge, self._nodes)


# ---------------------------------------------------------------------------
# Tiny WHERE filter helpers
# ---------------------------------------------------------------------------


_FILTER_RE = re.compile(
    r"\s*(?P<lhs>[a-z]\w*)\.(?P<key>[a-zA-Z_]\w*)\s*"
    r"(?P<op>=|!=|>=|<=|>|<)\s*"
    r"(?:\"(?P<sval>[^\"]*)\"|'(?P<svals>[^']*)'|(?P<nval>-?\d+(?:\.\d+)?))",
)


def _parse_where(text: str) -> list[tuple[str, str, str, Any]]:
    """Parse ``s.foo = "bar" AND r.confidence > 0.5`` into a list of
    ``(lhs, key, op, value)`` triples."""

    if not text.strip():
        return []
    parts = re.split(r"\s+AND\s+", text, flags=re.IGNORECASE)
    out: list[tuple[str, str, str, Any]] = []
    for part in parts:
        m = _FILTER_RE.match(part)
        if not m:
            continue
        value: Any
        if m.group("sval") is not None:
            value = m.group("sval")
        elif m.group("svals") is not None:
            value = m.group("svals")
        else:
            try:
                value = float(m.group("nval"))
                if value.is_integer():
                    value = int(value)
            except (TypeError, ValueError):
                continue
        out.append((m.group("lhs"), m.group("key"), m.group("op"), value))
    return out


def _evaluate_where(
    filters: list[tuple[str, str, str, Any]],
    bindings: dict[str, Any],
) -> bool:
    for lhs, key, op, value in filters:
        ref = bindings.get(lhs)
        if ref is None:
            return False
        if hasattr(ref, "properties") and key in getattr(ref, "properties"):
            actual: Any = ref.properties[key]
        else:
            actual = getattr(ref, key, None)
        if not _compare(actual, op, value):
            return False
    return True


def _compare(actual: Any, op: str, value: Any) -> bool:
    if actual is None:
        return False
    try:
        if op == "=":
            return actual == value
        if op == "!=":
            return actual != value
        if op == ">":
            return actual > value
        if op == ">=":
            return actual >= value
        if op == "<":
            return actual < value
        if op == "<=":
            return actual <= value
    except TypeError:
        return False
    return False


# ---------------------------------------------------------------------------
# Kùzu-backed store (Phase C1b — real implementation)
# ---------------------------------------------------------------------------


class KuzuGraphStore(GraphStore):
    """Kùzu-backed graph store (master §3.2 default).

    Kùzu is an *embedded* graph database — single-process, file-backed,
    no network service needed (it is to graph data what SQLite is to
    relational data). The store therefore opens a database directory
    locally and operates against it directly.

    Schema:

    - One node table ``Entity`` with columns ``(id STRING PRIMARY KEY,
      label STRING, properties STRING /* JSON */)``. The framework's
      open-set node labels live as a property rather than as separate
      tables to keep the schema fixed.
    - One relationship table ``Relates`` connecting ``Entity → Entity``
      with columns ``(predicate STRING, confidence DOUBLE, citation_uri
      STRING, properties STRING /* JSON */, edge_id STRING)``. Same
      open-set discipline.

    The ``add_claim`` semantics match ``InMemoryGraphStore`` exactly so
    callers (like the Phase C1a ``Ingestor``) work against either
    backend without modification.

    Imports of the ``kuzu`` library are lazy so the colony framework
    stays importable without it installed.
    """

    NODE_TABLE = "Entity"
    REL_TABLE = "Relates"

    @classmethod
    def open(cls, db_path: str | Path) -> "KuzuGraphStore":
        """Open / create a Kùzu database at ``db_path`` and return the
        ready-to-use store.

        Kùzu creates the on-disk database itself; the parent
        directory must exist but ``db_path`` itself MUST NOT already
        be a directory (Kùzu rejects that). Idempotent — opening an
        existing database reuses the schema (the ``CREATE … IF NOT
        EXISTS`` guards in ``_ensure_schema`` make bootstrap
        re-entrant).
        """

        try:
            import kuzu  # type: ignore[import-not-found]
        except ImportError as exc:
            raise GraphStoreError(
                "KuzuGraphStore requires the 'kuzu' package. Install via "
                "`pip install polymathera-colony[knowledge]`.",
            ) from exc

        path = Path(db_path)
        if path.parent != path:
            path.parent.mkdir(parents=True, exist_ok=True)
        database = kuzu.Database(str(path))
        connection = kuzu.Connection(database)
        store = cls(connection=connection, database=database)
        store._ensure_schema()
        return store

    def __init__(
        self,
        *,
        connection: Any,
        database: Any | None = None,
    ) -> None:
        self._connection = connection
        self._database = database
        self._lock = threading.RLock()

    @property
    def connection(self) -> Any:
        return self._connection

    def close(self) -> None:
        """Close the underlying Kùzu database. Subsequent calls raise."""

        try:
            close = getattr(self._database, "close", None)
            if callable(close):
                close()
        except Exception:  # noqa: BLE001
            logger.debug("KuzuGraphStore: database close raised; ignoring.")

    # ---- Schema -------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create the node + rel tables if they don't already exist."""

        with self._lock:
            self._exec(
                f"CREATE NODE TABLE IF NOT EXISTS {self.NODE_TABLE}("
                "id STRING, label STRING, properties STRING, "
                "PRIMARY KEY (id))",
            )
            self._exec(
                f"CREATE REL TABLE IF NOT EXISTS {self.REL_TABLE}("
                f"FROM {self.NODE_TABLE} TO {self.NODE_TABLE}, "
                "edge_id STRING, predicate STRING, confidence DOUBLE, "
                "citation_uri STRING, properties STRING)",
            )

    # ---- Mutation -----------------------------------------------------

    async def add_node(
        self, node: GraphNode, *, branch: str | None = None,
    ) -> None:
        tag = _resolve_write_branch(branch)
        await asyncio.to_thread(self._add_node_sync, node, tag)

    def _add_node_sync(self, node: GraphNode, tag: str) -> None:
        with self._lock:
            existing = self._fetch_node_sync(node.node_id)
            if existing is None:
                self._exec(
                    f"CREATE (n:{self.NODE_TABLE} "
                    "{id: $id, label: $label, properties: $properties})",
                    {
                        "id": node.node_id,
                        "label": node.label,
                        "properties": _encode_props(
                            node.properties, node.branches | {tag},
                        ),
                    },
                )
                return
            merged_props = {**existing.properties, **node.properties}
            merged_branches = existing.branches | node.branches | {tag}
            self._exec(
                f"MATCH (n:{self.NODE_TABLE}) WHERE n.id = $id "
                "SET n.properties = $properties",
                {
                    "id": node.node_id,
                    "properties": _encode_props(merged_props, merged_branches),
                },
            )

    async def add_edge(
        self, edge: GraphEdge, *, branch: str | None = None,
    ) -> None:
        tag = _resolve_write_branch(branch)
        await asyncio.to_thread(self._add_edge_sync, edge, tag)

    def _add_edge_sync(self, edge: GraphEdge, tag: str) -> None:
        with self._lock:
            if self._fetch_node_sync(edge.source_id) is None:
                raise GraphStoreError(
                    f"Edge {edge.edge_id}: missing source node {edge.source_id!r}.",
                )
            if self._fetch_node_sync(edge.target_id) is None:
                raise GraphStoreError(
                    f"Edge {edge.edge_id}: missing target node {edge.target_id!r}.",
                )
            existing = self._fetch_edge_sync(edge.edge_id)
            if existing is None:
                self._exec(
                    f"MATCH (s:{self.NODE_TABLE}), (t:{self.NODE_TABLE}) "
                    "WHERE s.id = $src AND t.id = $tgt "
                    f"CREATE (s)-[:{self.REL_TABLE} "
                    "{edge_id: $edge_id, predicate: $predicate, "
                    "confidence: $confidence, citation_uri: $citation_uri, "
                    "properties: $properties}]->(t)",
                    {
                        "src": edge.source_id,
                        "tgt": edge.target_id,
                        "edge_id": edge.edge_id,
                        "predicate": edge.predicate,
                        "confidence": float(edge.confidence),
                        "citation_uri": edge.citation_uri or "",
                        "properties": _encode_props(
                            edge.properties, edge.branches | {tag},
                        ),
                    },
                )
                return
            if tag in existing.branches:
                return
            self._exec(
                f"MATCH ()-[r:{self.REL_TABLE}]->() WHERE r.edge_id = $eid "
                "SET r.properties = $properties",
                {
                    "eid": edge.edge_id,
                    "properties": _encode_props(
                        existing.properties, existing.branches | {tag},
                    ),
                },
            )

    async def add_claim(
        self, claim: Claim, *, branch: str | None = None,
    ) -> tuple[GraphNode, GraphNode, GraphEdge]:
        tag = _resolve_write_branch(branch)
        return await asyncio.to_thread(self._add_claim_sync, claim, tag)

    async def add_claims(
        self, claims: Sequence[Claim], *, branch: str | None = None,
    ) -> tuple[tuple[GraphNode, GraphNode, GraphEdge] | None, ...]:
        """Batched write: one :func:`asyncio.to_thread` hop, one
        outer ``self._lock`` acquisition for the whole batch.
        ``threading.RLock`` is reentrant, so the per-claim helpers
        called inside (``_add_node_sync``, ``_add_edge_sync``,
        ``_fetch_node_sync``) re-acquire cheaply without contention.
        Per-claim failure is caught, logged, and surfaced as
        ``None`` at the failing index — same recovery semantics as
        the base-class default impl, with the lock-acquire churn
        amortised across the batch."""

        if not claims:
            return ()
        tag = _resolve_write_branch(branch)
        rows = await asyncio.to_thread(
            self._add_claims_sync, list(claims), tag,
        )
        return tuple(rows)

    async def import_claims(
        self, claims: Sequence[Claim], *, branch: str,
    ) -> ImportResult:
        if not branch:
            raise GraphStoreError(
                "import_claims requires a non-empty branch.",
            )
        if not claims:
            return ImportResult()
        return await asyncio.to_thread(
            self._import_claims_sync, list(claims), branch,
        )

    def _import_claims_sync(
        self, claims: list[Claim], branch: str,
    ) -> ImportResult:
        added = tagged = skipped = 0
        with self._lock:
            for claim in claims:
                s_id = _node_id_for(claim.subject)
                o_id = _node_id_for(claim.object_)
                edge_id = f"{s_id}--{claim.predicate}-->{o_id}"
                existing = self._fetch_edge_sync(edge_id)
                if existing is None:
                    self._add_claim_locked(claim, branch)
                    added += 1
                    continue
                if branch in existing.branches:
                    skipped += 1
                    continue
                s_node = self._fetch_node_sync(s_id)
                o_node = self._fetch_node_sync(o_id)
                if s_node is not None:
                    self._add_node_sync(s_node, branch)
                if o_node is not None:
                    self._add_node_sync(o_node, branch)
                self._add_edge_sync(existing, branch)
                tagged += 1
        return ImportResult(added=added, tagged=tagged, skipped=skipped)

    def _add_claim_sync(
        self, claim: Claim, tag: str,
    ) -> tuple[GraphNode, GraphNode, GraphEdge]:
        with self._lock:
            return self._add_claim_locked(claim, tag)

    def _add_claims_sync(
        self, claims: list[Claim], tag: str,
    ) -> list[tuple[GraphNode, GraphNode, GraphEdge] | None]:
        out: list[tuple[GraphNode, GraphNode, GraphEdge] | None] = []
        with self._lock:
            for claim in claims:
                try:
                    out.append(self._add_claim_locked(claim, tag))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "KuzuGraphStore.add_claim failed for claim %r: %s",
                        claim, exc,
                    )
                    out.append(None)
        return out

    def _add_claim_locked(
        self, claim: Claim, tag: str,
    ) -> tuple[GraphNode, GraphNode, GraphEdge]:
        """Build + insert one claim's nodes + edge inside the held
        lock. Caller MUST already hold ``self._lock``."""

        s_id = _node_id_for(claim.subject)
        o_id = _node_id_for(claim.object_)
        s_node = GraphNode(
            node_id=s_id, label="Entity",
            properties={"surface": claim.subject},
        )
        o_node = GraphNode(
            node_id=o_id, label="Entity",
            properties={"surface": claim.object_},
        )
        self._add_node_sync(s_node, tag)
        self._add_node_sync(o_node, tag)
        edge_id = f"{s_id}--{claim.predicate}-->{o_id}"
        edge_props: dict[str, Any] = {
            "section_path": claim.citation.section_path,
            "char_start": claim.citation.char_start,
            "char_end": claim.citation.char_end,
        }
        if claim.provenance:
            edge_props["__provenance"] = dict(claim.provenance)
        edge = GraphEdge(
            edge_id=edge_id,
            source_id=s_id,
            target_id=o_id,
            predicate=claim.predicate,
            confidence=claim.confidence,
            citation_uri=claim.citation.source_uri,
            properties=edge_props,
        )
        self._add_edge_sync(edge, tag)
        actual_s = self._fetch_node_sync(s_id) or s_node
        actual_o = self._fetch_node_sync(o_id) or o_node
        actual_edge = self._fetch_edge_sync(edge_id) or edge
        return actual_s, actual_o, actual_edge

    # ---- Queries ------------------------------------------------------

    async def get_node(
        self, node_id: str, *, branch_filter: str | None = None,
    ) -> GraphNode | None:
        node = await asyncio.to_thread(self._fetch_node_sync, node_id)
        if node is None:
            return None
        if branch_filter is not None and branch_filter not in node.branches:
            return None
        return node

    async def neighbours(
        self,
        node_id: str,
        *,
        predicate: str | None = None,
        depth: int = 1,
        branch_filter: str | None = None,
    ) -> GraphQueryResult:
        return await asyncio.to_thread(
            self._neighbours_sync, node_id, predicate, depth, branch_filter,
        )

    def _neighbours_sync(
        self,
        node_id: str,
        predicate: str | None,
        depth: int,
        branch_filter: str | None,
    ) -> GraphQueryResult:
        with self._lock:
            root = self._fetch_node_sync(node_id)
            if root is None:
                return GraphQueryResult()
            if branch_filter is not None and branch_filter not in root.branches:
                return GraphQueryResult()
            depth = max(0, int(depth))
            seen_nodes: dict[str, GraphNode] = {
                node_id: root
            }
            edges_set: dict[str, GraphEdge] = {}
            paths: list[tuple[str, ...]] = []
            frontier: list[tuple[str, tuple[str, ...]]] = [(node_id, (node_id,))]
            for _ in range(depth):
                next_frontier: list[tuple[str, tuple[str, ...]]] = []
                for current, path in frontier:
                    where = "WHERE s.id = $sid"
                    params = {"sid": current}
                    if predicate is not None:
                        where += " AND r.predicate = $predicate"
                        params["predicate"] = predicate
                    cursor = self._exec(
                        f"MATCH (s:{self.NODE_TABLE})-[r:{self.REL_TABLE}]->"
                        f"(t:{self.NODE_TABLE}) {where} "
                        "RETURN r.edge_id, r.predicate, r.confidence, "
                        "r.citation_uri, r.properties, "
                        "t.id, t.label, t.properties",
                        params,
                    )
                    for row in _iter_rows(cursor):
                        (
                            edge_id, pred, conf, cit, edge_props,
                            t_id, t_label, t_props,
                        ) = row
                        edge_props_clean, edge_branches = _split_branches(
                            _load_props(edge_props),
                        )
                        t_props_clean, t_branches = _split_branches(
                            _load_props(t_props),
                        )
                        if branch_filter is not None and (
                            branch_filter not in edge_branches
                            or branch_filter not in t_branches
                        ):
                            continue
                        edges_set[edge_id] = GraphEdge(
                            edge_id=edge_id,
                            source_id=current,
                            target_id=t_id,
                            predicate=pred,
                            confidence=float(conf or 0.0),
                            citation_uri=cit or None,
                            properties=edge_props_clean,
                            branches=edge_branches,
                        )
                        if t_id not in seen_nodes:
                            seen_nodes[t_id] = GraphNode(
                                node_id=t_id, label=t_label or "Entity",
                                properties=t_props_clean,
                                branches=t_branches,
                            )
                        new_path = path + (t_id,)
                        paths.append(new_path)
                        next_frontier.append((t_id, new_path))
                frontier = next_frontier
            return GraphQueryResult(
                nodes=tuple(seen_nodes.values()),
                edges=tuple(edges_set.values()),
                paths=tuple(paths),
            )

    async def query(
        self, query: str, *, branch_filter: str | None = None,
    ) -> GraphQueryResult:
        return await asyncio.to_thread(self._query_sync, query, branch_filter)

    def _query_sync(
        self, query: str, branch_filter: str | None = None,
    ) -> GraphQueryResult:
        m = _MATCH_RE.match(query.strip())
        if not m:
            raise GraphStoreError(
                "KuzuGraphStore supports the same narrow MATCH (s)-[r]->(o) "
                "[WHERE …] [RETURN …] [LIMIT n] DSL as InMemoryGraphStore. "
                f"Got: {query.strip()[:200]!r}",
            )
        s_label = m.group("sl")
        r_label = m.group("rl")
        o_label = m.group("ol")
        where = m.group("where") or ""
        limit_str = m.group("limit")
        limit = int(limit_str) if limit_str else None
        where_filters = _parse_where(where)

        # Translate to Kùzu Cypher. We match every edge of the
        # configured types, then filter post-hoc on the WHERE bindings
        # (Kùzu's parameter substitution would require generating a
        # different param set per filter; the post-hoc filter keeps
        # the implementation simple while mirroring the in-memory
        # store's semantics exactly).
        params: dict[str, Any] = {}
        cypher = (
            f"MATCH (s:{self.NODE_TABLE})-[r:{self.REL_TABLE}]->"
            f"(t:{self.NODE_TABLE}) "
        )
        conds: list[str] = []
        if r_label:
            conds.append("r.predicate = $r_label")
            params["r_label"] = r_label
        if s_label:
            conds.append("s.label = $s_label")
            params["s_label"] = s_label
        if o_label:
            conds.append("t.label = $o_label")
            params["o_label"] = o_label
        if conds:
            cypher += "WHERE " + " AND ".join(conds) + " "
        cypher += (
            "RETURN r.edge_id, r.predicate, r.confidence, "
            "r.citation_uri, r.properties, "
            "s.id, s.label, s.properties, "
            "t.id, t.label, t.properties"
        )

        cursor = self._exec(cypher, params)

        nodes_map: dict[str, GraphNode] = {}
        edges_map: dict[str, GraphEdge] = {}
        paths: list[tuple[str, str]] = []
        with self._lock:
            for row in _iter_rows(cursor):
                (
                    edge_id, pred, conf, cit, edge_props,
                    s_id, s_lbl, s_props,
                    t_id, t_lbl, t_props,
                ) = row
                s_clean, s_branches = _split_branches(_load_props(s_props))
                t_clean, t_branches = _split_branches(_load_props(t_props))
                e_clean, e_branches = _split_branches(_load_props(edge_props))
                if branch_filter is not None and (
                    branch_filter not in e_branches
                    or branch_filter not in s_branches
                    or branch_filter not in t_branches
                ):
                    continue
                s_node = GraphNode(
                    node_id=s_id, label=s_lbl or "Entity",
                    properties=s_clean, branches=s_branches,
                )
                t_node = GraphNode(
                    node_id=t_id, label=t_lbl or "Entity",
                    properties=t_clean, branches=t_branches,
                )
                edge = GraphEdge(
                    edge_id=edge_id,
                    source_id=s_id, target_id=t_id,
                    predicate=pred,
                    confidence=float(conf or 0.0),
                    citation_uri=cit or None,
                    properties=e_clean, branches=e_branches,
                )
                if not _evaluate_where(
                    where_filters, {"s": s_node, "o": t_node, "r": edge},
                ):
                    continue
                nodes_map[s_id] = s_node
                nodes_map[t_id] = t_node
                edges_map[edge_id] = edge
                paths.append((s_id, t_id))
                if limit is not None and len(paths) >= limit:
                    break
        return GraphQueryResult(
            nodes=tuple(nodes_map.values()),
            edges=tuple(edges_map.values()),
            paths=tuple(paths),
        )

    async def count(
        self, *, branch_filter: str | None = None,
    ) -> tuple[int, int]:
        return await asyncio.to_thread(self._count_sync, branch_filter)

    def _count_sync(
        self, branch_filter: str | None = None,
    ) -> tuple[int, int]:
        with self._lock:
            if branch_filter is None:
                n_cursor = self._exec(
                    f"MATCH (n:{self.NODE_TABLE}) RETURN count(n)",
                )
                n_rows = list(_iter_rows(n_cursor))
                n_count = int(n_rows[0][0]) if n_rows else 0
                r_cursor = self._exec(
                    f"MATCH ()-[r:{self.REL_TABLE}]->() RETURN count(r)",
                )
                r_rows = list(_iter_rows(r_cursor))
                r_count = int(r_rows[0][0]) if r_rows else 0
                return n_count, r_count
            n_count = sum(
                1 for _ in self._iter_nodes_locked(branch_filter)
            )
            r_count = sum(
                1 for _ in self._iter_edges_locked(branch_filter)
            )
            return n_count, r_count

    async def export_claims(
        self, *, branch: str | None = None,
    ) -> AsyncIterator[Claim]:
        rows = await asyncio.to_thread(self._collect_export_rows, branch)
        for row in rows:
            yield row

    def _collect_export_rows(self, branch: str | None) -> list[Claim]:
        out: list[Claim] = []
        with self._lock:
            node_cache: dict[str, GraphNode] = {}
            cursor = self._exec(
                f"MATCH (s:{self.NODE_TABLE})-[r:{self.REL_TABLE}]->"
                f"(t:{self.NODE_TABLE}) "
                "RETURN s.id, s.label, s.properties, "
                "t.id, t.label, t.properties, "
                "r.edge_id, r.predicate, r.confidence, "
                "r.citation_uri, r.properties",
            )
            edge_rows: list[
                tuple[str, str, str, frozenset[str], GraphEdge]
            ] = []
            for row in _iter_rows(cursor):
                (
                    s_id, s_lbl, s_props,
                    t_id, t_lbl, t_props,
                    edge_id, pred, conf, cit, edge_props,
                ) = row
                s_clean, s_branches = _split_branches(_load_props(s_props))
                t_clean, t_branches = _split_branches(_load_props(t_props))
                e_clean, e_branches = _split_branches(_load_props(edge_props))
                if branch is not None and branch not in e_branches:
                    continue
                node_cache[s_id] = GraphNode(
                    node_id=s_id, label=s_lbl or "Entity",
                    properties=s_clean, branches=s_branches,
                )
                node_cache[t_id] = GraphNode(
                    node_id=t_id, label=t_lbl or "Entity",
                    properties=t_clean, branches=t_branches,
                )
                edge = GraphEdge(
                    edge_id=edge_id,
                    source_id=s_id, target_id=t_id,
                    predicate=pred,
                    confidence=float(conf or 0.0),
                    citation_uri=cit or None,
                    properties=e_clean, branches=e_branches,
                )
                edge_rows.append((s_id, pred, t_id, e_branches, edge))
        edge_rows.sort(key=lambda r: (r[0], r[1], r[2]))
        for _, _, _, _, edge in edge_rows:
            out.append(_claim_from_edge(edge, node_cache))
        return out

    def _iter_nodes_locked(
        self, branch_filter: str,
    ) -> Iterator[GraphNode]:
        cursor = self._exec(
            f"MATCH (n:{self.NODE_TABLE}) RETURN n.id, n.label, n.properties",
        )
        for row in _iter_rows(cursor):
            nid, label, props = row
            clean, branches = _split_branches(_load_props(props))
            if branch_filter not in branches:
                continue
            yield GraphNode(
                node_id=nid, label=label or "Entity",
                properties=clean, branches=branches,
            )

    def _iter_edges_locked(
        self, branch_filter: str,
    ) -> Iterator[GraphEdge]:
        cursor = self._exec(
            f"MATCH (s:{self.NODE_TABLE})-[r:{self.REL_TABLE}]->"
            f"(t:{self.NODE_TABLE}) "
            "RETURN s.id, t.id, r.edge_id, r.predicate, r.confidence, "
            "r.citation_uri, r.properties",
        )
        for row in _iter_rows(cursor):
            s, t, eid, pred, conf, cit, props = row
            clean, branches = _split_branches(_load_props(props))
            if branch_filter not in branches:
                continue
            yield GraphEdge(
                edge_id=eid, source_id=s, target_id=t,
                predicate=pred, confidence=float(conf or 0.0),
                citation_uri=cit or None,
                properties=clean, branches=branches,
            )

    # ---- Helpers ------------------------------------------------------

    def _exec(self, query: str, params: Mapping[str, Any] | None = None) -> Any:
        if params is None:
            return self._connection.execute(query)
        # Kùzu's Python client accepts ``execute(query, parameters=dict)``.
        try:
            return self._connection.execute(query, dict(params))
        except TypeError:
            return self._connection.execute(query, parameters=dict(params))

    def _fetch_node_sync(self, node_id: str) -> GraphNode | None:
        cursor = self._exec(
            f"MATCH (n:{self.NODE_TABLE}) WHERE n.id = $id "
            "RETURN n.id, n.label, n.properties",
            {"id": node_id},
        )
        rows = list(_iter_rows(cursor))
        if not rows:
            return None
        nid, label, props = rows[0]
        clean_props, branches = _split_branches(_load_props(props))
        return GraphNode(
            node_id=nid, label=label or "Entity",
            properties=clean_props, branches=branches,
        )

    def _fetch_edge_sync(self, edge_id: str) -> GraphEdge | None:
        cursor = self._exec(
            f"MATCH (s:{self.NODE_TABLE})-[r:{self.REL_TABLE}]->"
            f"(t:{self.NODE_TABLE}) WHERE r.edge_id = $eid "
            "RETURN s.id, t.id, r.edge_id, r.predicate, r.confidence, "
            "r.citation_uri, r.properties",
            {"eid": edge_id},
        )
        rows = list(_iter_rows(cursor))
        if not rows:
            return None
        s, t, eid, pred, conf, cit, props = rows[0]
        clean_props, branches = _split_branches(_load_props(props))
        return GraphEdge(
            edge_id=eid, source_id=s, target_id=t,
            predicate=pred, confidence=float(conf or 0.0),
            citation_uri=cit or None,
            properties=clean_props, branches=branches,
        )


__all__ = (
    "GraphStore",
    "GraphStoreError",
    "GraphNode",
    "CURRENT_BRANCH_CONTEXT",
    "GraphEdge",
    "GraphQueryResult",
    "ImportResult",
    "InMemoryGraphStore",
    "KuzuGraphStore",
    "set_current_branch",
)
