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
import json
import logging
import re
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..models import Claim


logger = logging.getLogger(__name__)


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
    async def add_node(self, node: GraphNode) -> None:
        ...

    @abstractmethod
    async def add_edge(self, edge: GraphEdge) -> None:
        ...

    @abstractmethod
    async def add_claim(self, claim: Claim) -> tuple[GraphNode, GraphNode, GraphEdge]:
        """Convenience: turn a ``Claim`` into a ``(subject, object, edge)``
        triple, idempotently inserting any nodes / edges that are new.

        Returns the (possibly newly created) records."""

    @abstractmethod
    async def get_node(self, node_id: str) -> GraphNode | None:
        ...

    @abstractmethod
    async def neighbours(
        self,
        node_id: str,
        *,
        predicate: str | None = None,
        depth: int = 1,
    ) -> GraphQueryResult:
        ...

    @abstractmethod
    async def query(self, query: str) -> GraphQueryResult:
        """Run a small-DSL query (see module docstring). Implementations
        that don't support the DSL raise ``GraphStoreError``."""

    @abstractmethod
    async def count(self) -> tuple[int, int]:
        """Return ``(node_count, edge_count)``."""


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

    async def add_node(self, node: GraphNode) -> None:
        existing = self._nodes.get(node.node_id)
        if existing is None:
            self._nodes[node.node_id] = node
        else:
            # Merge properties; keep first-seen label.
            merged = {**existing.properties, **node.properties}
            self._nodes[node.node_id] = existing.model_copy(
                update={"properties": merged},
            )

    async def add_edge(self, edge: GraphEdge) -> None:
        if edge.source_id not in self._nodes or edge.target_id not in self._nodes:
            raise GraphStoreError(
                f"Edge {edge.edge_id}: missing source or target node "
                f"({edge.source_id}, {edge.target_id}).",
            )
        if edge.edge_id in self._edges_by_id:
            return
        self._edges_by_id[edge.edge_id] = edge
        self._out[edge.source_id].append(edge.edge_id)
        self._in[edge.target_id].append(edge.edge_id)

    async def add_claim(
        self, claim: Claim,
    ) -> tuple[GraphNode, GraphNode, GraphEdge]:
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
        await self.add_node(s_node)
        await self.add_node(o_node)
        edge_id = f"{s_id}--{claim.predicate}-->{o_id}"
        edge = GraphEdge(
            edge_id=edge_id,
            source_id=s_id,
            target_id=o_id,
            predicate=claim.predicate,
            confidence=claim.confidence,
            citation_uri=claim.citation.source_uri,
            properties={
                "section_path": claim.citation.section_path,
                "char_start": claim.citation.char_start,
                "char_end": claim.citation.char_end,
            },
        )
        await self.add_edge(edge)
        return self._nodes[s_id], self._nodes[o_id], self._edges_by_id[edge_id]

    # ---- Queries ------------------------------------------------------

    async def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    async def neighbours(
        self,
        node_id: str,
        *,
        predicate: str | None = None,
        depth: int = 1,
    ) -> GraphQueryResult:
        if node_id not in self._nodes:
            return GraphQueryResult()
        seen: dict[str, GraphNode] = {node_id: self._nodes[node_id]}
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
                    edges_set[eid] = edge
                    target = self._nodes.get(edge.target_id)
                    if target is None:
                        continue
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

    async def query(self, query: str) -> GraphQueryResult:
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
            s_node = self._nodes.get(edge.source_id)
            o_node = self._nodes.get(edge.target_id)
            if s_node is None or o_node is None:
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

    async def count(self) -> tuple[int, int]:
        return len(self._nodes), len(self._edges_by_id)


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

    async def add_node(self, node: GraphNode) -> None:
        await asyncio.to_thread(self._add_node_sync, node)

    def _add_node_sync(self, node: GraphNode) -> None:
        with self._lock:
            existing = self._fetch_node_sync(node.node_id)
            if existing is None:
                self._exec(
                    f"CREATE (n:{self.NODE_TABLE} "
                    "{id: $id, label: $label, properties: $properties})",
                    {
                        "id": node.node_id,
                        "label": node.label,
                        "properties": json.dumps(node.properties, sort_keys=True),
                    },
                )
                return
            merged = {**existing.properties, **node.properties}
            self._exec(
                f"MATCH (n:{self.NODE_TABLE}) WHERE n.id = $id "
                "SET n.properties = $properties",
                {"id": node.node_id, "properties": json.dumps(merged, sort_keys=True)},
            )

    async def add_edge(self, edge: GraphEdge) -> None:
        await asyncio.to_thread(self._add_edge_sync, edge)

    def _add_edge_sync(self, edge: GraphEdge) -> None:
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
            if existing is not None:
                return
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
                    "properties": json.dumps(edge.properties, sort_keys=True),
                },
            )

    async def add_claim(
        self, claim: Claim,
    ) -> tuple[GraphNode, GraphNode, GraphEdge]:
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
        await self.add_node(s_node)
        await self.add_node(o_node)
        edge_id = f"{s_id}--{claim.predicate}-->{o_id}"
        edge = GraphEdge(
            edge_id=edge_id,
            source_id=s_id,
            target_id=o_id,
            predicate=claim.predicate,
            confidence=claim.confidence,
            citation_uri=claim.citation.source_uri,
            properties={
                "section_path": claim.citation.section_path,
                "char_start": claim.citation.char_start,
                "char_end": claim.citation.char_end,
            },
        )
        await self.add_edge(edge)
        # Re-fetch so the returned models carry the merged-property view.
        actual_s = await asyncio.to_thread(self._fetch_node_sync, s_id) or s_node
        actual_o = await asyncio.to_thread(self._fetch_node_sync, o_id) or o_node
        return actual_s, actual_o, edge

    # ---- Queries ------------------------------------------------------

    async def get_node(self, node_id: str) -> GraphNode | None:
        return await asyncio.to_thread(self._fetch_node_sync, node_id)

    async def neighbours(
        self,
        node_id: str,
        *,
        predicate: str | None = None,
        depth: int = 1,
    ) -> GraphQueryResult:
        return await asyncio.to_thread(
            self._neighbours_sync, node_id, predicate, depth,
        )

    def _neighbours_sync(
        self, node_id: str, predicate: str | None, depth: int,
    ) -> GraphQueryResult:
        with self._lock:
            if self._fetch_node_sync(node_id) is None:
                return GraphQueryResult()
            depth = max(0, int(depth))
            seen_nodes: dict[str, GraphNode] = {
                node_id: self._fetch_node_sync(node_id)  # type: ignore[dict-item]
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
                        edges_set[edge_id] = GraphEdge(
                            edge_id=edge_id,
                            source_id=current,
                            target_id=t_id,
                            predicate=pred,
                            confidence=float(conf or 0.0),
                            citation_uri=cit or None,
                            properties=_load_props(edge_props),
                        )
                        if t_id not in seen_nodes:
                            seen_nodes[t_id] = GraphNode(
                                node_id=t_id, label=t_label or "Entity",
                                properties=_load_props(t_props),
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

    async def query(self, query: str) -> GraphQueryResult:
        return await asyncio.to_thread(self._query_sync, query)

    def _query_sync(self, query: str) -> GraphQueryResult:
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
                s_node = GraphNode(
                    node_id=s_id, label=s_lbl or "Entity",
                    properties=_load_props(s_props),
                )
                t_node = GraphNode(
                    node_id=t_id, label=t_lbl or "Entity",
                    properties=_load_props(t_props),
                )
                edge = GraphEdge(
                    edge_id=edge_id,
                    source_id=s_id, target_id=t_id,
                    predicate=pred,
                    confidence=float(conf or 0.0),
                    citation_uri=cit or None,
                    properties=_load_props(edge_props),
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

    async def count(self) -> tuple[int, int]:
        return await asyncio.to_thread(self._count_sync)

    def _count_sync(self) -> tuple[int, int]:
        with self._lock:
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
        return GraphNode(
            node_id=nid, label=label or "Entity",
            properties=_load_props(props),
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
        return GraphEdge(
            edge_id=eid, source_id=s, target_id=t,
            predicate=pred, confidence=float(conf or 0.0),
            citation_uri=cit or None,
            properties=_load_props(props),
        )


__all__ = (
    "GraphStore",
    "GraphStoreError",
    "GraphNode",
    "GraphEdge",
    "GraphQueryResult",
    "InMemoryGraphStore",
    "KuzuGraphStore",
)
